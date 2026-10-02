

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.diffusion import GaussianDiffusion1D, MLPWrapper
from utils.config import to_dot_dict
from utils.contact_gt_surface import (
    DEFAULT_NUM_FINGERTIPS,
    aggregate_fingertips_per_link,
    infer_num_fingertips_from_hand,
    unflatten_contacts,
)


@dataclass
class ContactDiffusionConfig:
    feature_dim: int = 512
    cond_dim: int = 256
    num_fingertips: int = 4
    hidden_dim: int = 512
    object_feat_dim: int = 128
    target_representation: str = "offset"
    contact_thresh: float = 0.015
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 100
    prediction_type: str = "epsilon"
    log_prob_type: Optional[str] = None
    ode: bool = True

    pointnext_k: int = 16
    pointnext_samples1: int = 256
    pointnext_samples2: int = 64

    fit_steps: int = 8
    fit_lr: float = 5e-3
    fit_max_trans_delta: float = 0.01
    fit_max_rot_delta: float = 0.26
    fit_max_q_delta: float = 0.10
    fit_w_contact: float = 1.0
    fit_w_pose: float = 0.05
    fit_w_joint: float = 0.02

    @property
    def contact_dim(self) -> int:
        return self.num_fingertips * 3


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid.float().unsqueeze(-1)
    denom = w.sum(dim=1).clamp(min=1.0)
    return (x * w).sum(dim=1) / denom


def _masked_max(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    masked = x.masked_fill(~valid.unsqueeze(-1), neg_inf)
    out = masked.max(dim=1).values
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b = points.shape[0]
    view_shape = list(idx.shape)
    batch_indices = torch.arange(b, dtype=torch.long, device=points.device).view(
        [b] + [1] * (idx.dim() - 1)
    )
    batch_indices = batch_indices.expand(view_shape)
    return points[batch_indices, idx]


def _sample_indices(points: torch.Tensor, valid: torch.Tensor, num_samples: int) -> torch.Tensor:
    b, n, _ = points.shape
    s = min(int(num_samples), n)
    idx_out = torch.zeros(b, s, dtype=torch.long, device=points.device)
    # Deterministic stride sampling keeps training reproducible and avoids an FPS
    # dependency. It is sufficient for cropped 1024-point object clouds.
    for i in range(b):
        valid_idx = torch.where(valid[i])[0]
        if valid_idx.numel() == 0:
            valid_idx = torch.arange(n, device=points.device)
        if valid_idx.numel() >= s:
            pos = torch.linspace(0, valid_idx.numel() - 1, steps=s, device=points.device)
            idx = valid_idx[pos.round().long()]
        else:
            repeat = (s + valid_idx.numel() - 1) // valid_idx.numel()
            idx = valid_idx.repeat(repeat)[:s]
        idx_out[i] = idx
    return idx_out


class LocalAggregation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = int(k)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        query_xyz: torch.Tensor,
        support_xyz: torch.Tensor,
        support_feat: torch.Tensor,
        support_valid: torch.Tensor,
    ) -> torch.Tensor:
        b, m, _ = query_xyz.shape
        n = support_xyz.shape[1]
        k = min(self.k, n)
        dist = torch.cdist(query_xyz, support_xyz)
        dist = dist.masked_fill(~support_valid[:, None, :], 1e6)
        nn_idx = dist.topk(k=k, dim=-1, largest=False).indices
        neigh_xyz = _index_points(support_xyz, nn_idx)
        neigh_feat = _index_points(support_feat, nn_idx)
        center_feat = neigh_feat[:, :, :1, :].expand(-1, -1, k, -1)
        rel = neigh_xyz - query_xyz[:, :, None, :]
        edge = torch.cat([center_feat, neigh_feat - center_feat, rel], dim=-1)
        edge = edge.reshape(b * m * k, -1)
        edge_feat = self.edge_mlp(edge).reshape(b, m, k, -1)
        pooled = edge_feat.max(dim=2).values
        center = self.shortcut(center_feat[:, :, 0, :])
        return pooled + center


class PointNeXtObjectEncoder(nn.Module):
    """
    Lightweight hierarchical local geometry encoder.

    Compared with the previous PointNet encoder, this explicitly aggregates KNN
    neighborhoods at two sampled resolutions before global pooling, so contact
    generation receives local surface geometry rather than only a global object
    descriptor.
    """

    def __init__(
        self,
        out_dim: int = 128,
        width: int = 64,
        k: int = 16,
        samples1: int = 256,
        samples2: int = 64,
    ):
        super().__init__()
        self.samples1 = int(samples1)
        self.samples2 = int(samples2)
        self.stem = nn.Sequential(
            nn.Linear(6, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.sa1 = LocalAggregation(width, width * 2, k=k)
        self.sa2 = LocalAggregation(width * 2, width * 4, k=k)
        self.fuse = nn.Sequential(
            nn.Linear(width * 8 + 6, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, object_pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = object_pc.abs().sum(dim=-1) > 1e-8
        b, n, _ = object_pc.shape
        if not bool(valid.any()):
            center = torch.zeros(b, 3, device=object_pc.device, dtype=object_pc.dtype)
            feat = torch.zeros(b, self.fuse[0].out_features, device=object_pc.device, dtype=object_pc.dtype)
            return feat, center

        center = _masked_mean(object_pc, valid)
        centered = object_pc - center[:, None, :]
        radius = torch.linalg.norm(centered, dim=-1).masked_fill(~valid, 0.0).amax(dim=1, keepdim=True)
        radius = radius.clamp(min=1e-4)
        norm_xyz = centered / radius[:, None, :]
        stem_in = torch.cat([norm_xyz, centered], dim=-1).reshape(b * n, 6)
        feat0 = self.stem(stem_in).reshape(b, n, -1)

        idx1 = _sample_indices(object_pc, valid, self.samples1)
        xyz1 = _index_points(object_pc, idx1)
        valid1 = torch.ones(b, xyz1.shape[1], dtype=torch.bool, device=object_pc.device)
        feat1 = self.sa1(xyz1, object_pc, feat0, valid)

        idx2 = _sample_indices(xyz1, valid1, self.samples2)
        xyz2 = _index_points(xyz1, idx2)
        valid2 = torch.ones(b, xyz2.shape[1], dtype=torch.bool, device=object_pc.device)
        feat2 = self.sa2(xyz2, xyz1, feat1, valid1)

        pooled_mean = _masked_mean(feat2, valid2)
        pooled_max = _masked_max(feat2, valid2)
        pos_inf = torch.full_like(centered, float("inf"))
        neg_inf = torch.full_like(centered, float("-inf"))
        min_xyz = torch.where(valid[:, :, None], centered, pos_inf).amin(dim=1)
        max_xyz = torch.where(valid[:, :, None], centered, neg_inf).amax(dim=1)
        min_xyz = torch.where(torch.isfinite(min_xyz), min_xyz, torch.zeros_like(min_xyz))
        max_xyz = torch.where(torch.isfinite(max_xyz), max_xyz, torch.zeros_like(max_xyz))
        extent = torch.cat([min_xyz, max_xyz], dim=-1)
        obj_feat = self.fuse(torch.cat([pooled_mean, pooled_max, extent], dim=-1))
        return obj_feat, center


class ContactDiffusionNet(nn.Module):
    def __init__(self, cfg: Optional[ContactDiffusionConfig] = None):
        super().__init__()
        self.cfg = cfg or ContactDiffusionConfig()
        self.object_encoder = PointNeXtObjectEncoder(
            out_dim=self.cfg.object_feat_dim,
            width=max(32, self.cfg.object_feat_dim // 2),
            k=self.cfg.pointnext_k,
            samples1=self.cfg.pointnext_samples1,
            samples2=self.cfg.pointnext_samples2,
        )
        in_dim = self.cfg.feature_dim + 3 + 3 + self.cfg.object_feat_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(in_dim, self.cfg.cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
            nn.ReLU(inplace=True),
        )
        self.denoiser = MLPWrapper(
            channels=self.cfg.contact_dim,
            feature_dim=self.cfg.cond_dim,
            hidden_layers_dim=[self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.hidden_dim],
            output_dim=self.cfg.contact_dim,
            act="mish",
        )
        diffusion_cfg = to_dot_dict(
            {
                "scheduler_type": "DDPMScheduler",
                "scheduler": {
                    "num_train_timesteps": self.cfg.num_train_timesteps,
                    "prediction_type": self.cfg.prediction_type,
                },
                "num_inference_timesteps": self.cfg.num_inference_timesteps,
                "log_prob_type": self.cfg.log_prob_type,
                "ode": self.cfg.ode,
            }
        )
        self.diffusion = GaussianDiffusion1D(self.denoiser, diffusion_cfg)

    def _seed_flat(self, seed_points: torch.Tensor) -> torch.Tensor:
        return seed_points[:, None, :].expand(-1, self.cfg.num_fingertips, -1).reshape(
            seed_points.shape[0], -1
        )

    def _contacts_to_targets(self, gt_contacts_flat: torch.Tensor, seed_points: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_representation == "absolute":
            return gt_contacts_flat
        return gt_contacts_flat - self._seed_flat(seed_points)

    def _targets_to_contacts(self, pred_flat: torch.Tensor, seed_points: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_representation == "absolute":
            return pred_flat
        return pred_flat + self._seed_flat(seed_points)

    def encode_condition(self, feature: torch.Tensor, seed_points: torch.Tensor, object_pc: torch.Tensor) -> torch.Tensor:
        object_feat, object_center = self.object_encoder(object_pc)
        x = torch.cat([feature, seed_points, object_center, object_feat], dim=-1)
        return self.cond_encoder(x)

    def forward(
        self,
        gt_contacts_flat: torch.Tensor,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.encode_condition(feature, seed_points, object_pc)
        targets = self._contacts_to_targets(gt_contacts_flat, seed_points)
        return self.diffusion.calculate_loss(targets, cond)

    @torch.no_grad()
    def sample_contacts(self, feature: torch.Tensor, seed_points: torch.Tensor, object_pc: torch.Tensor) -> torch.Tensor:
        cond = self.encode_condition(feature, seed_points, object_pc)
        samples, _ = self.diffusion.sample(cond)
        return self._targets_to_contacts(samples, seed_points)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / theta
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(
        axis_angle.shape[0], 3, 3
    )
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * K + (1.0 - cos_t) * torch.matmul(K, K)


def _forward_finger_representatives(
    hand_model,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    object_pc: Optional[torch.Tensor] = None,
    num_fingers: Optional[int] = None,
    cdist_chunk: int = 2048,
) -> torch.Tensor:
    if hasattr(hand_model, "forward_hand_points"):
        _, _, raw = hand_model.forward_hand_points(trans, rot, qpos)
    else:
        _, _, raw = hand_model._sdf_adam.forward_hand_points(trans, rot, qpos)
    num_fingers = num_fingers or infer_num_fingertips_from_hand(hand_model)
    return aggregate_fingertips_per_link(
        raw, num_fingers, object_pc=object_pc, cdist_chunk=cdist_chunk
    )


def fit_pose_to_contacts(
    hand_model,
    init_trans: torch.Tensor,
    init_rot: torch.Tensor,
    init_qpos: torch.Tensor,
    target_contacts: torch.Tensor,
    object_pc: Optional[torch.Tensor] = None,
    cfg: Optional[ContactDiffusionConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    cfg = cfg or ContactDiffusionConfig()
    base_trans = init_trans.detach()
    base_rot = init_rot.detach()
    base_qpos = init_qpos.detach()
    delta_trans = torch.zeros_like(base_trans, requires_grad=True)
    delta_rot_aa = torch.zeros(base_trans.shape[0], 3, device=base_trans.device, requires_grad=True)
    delta_qpos = torch.zeros_like(base_qpos, requires_grad=True)
    opt = torch.optim.Adam([delta_trans, delta_rot_aa, delta_qpos], lr=cfg.fit_lr)

    last_loss = torch.zeros((), device=base_trans.device)
    for _ in range(cfg.fit_steps):
        opt.zero_grad(set_to_none=True)
        d_trans = torch.tanh(delta_trans) * cfg.fit_max_trans_delta
        d_rot_aa = torch.tanh(delta_rot_aa) * cfg.fit_max_rot_delta
        d_q = torch.tanh(delta_qpos) * cfg.fit_max_q_delta
        rot_delta = _axis_angle_to_matrix(d_rot_aa)
        trans_out = base_trans + d_trans
        rot_out = torch.matmul(rot_delta, base_rot)
        qpos_out = base_qpos + d_q
        tips = _forward_finger_representatives(
            hand_model,
            trans_out,
            rot_out,
            qpos_out,
            object_pc=object_pc,
            num_fingers=cfg.num_fingertips,
        )
        loss_contact = F.mse_loss(tips, target_contacts)
        loss_pose = d_trans.square().mean() + d_rot_aa.square().mean() + d_q.square().mean()
        last_loss = cfg.fit_w_contact * loss_contact + cfg.fit_w_pose * loss_pose
        last_loss.backward()
        opt.step()

    with torch.no_grad():
        d_trans = torch.tanh(delta_trans) * cfg.fit_max_trans_delta
        d_rot_aa = torch.tanh(delta_rot_aa) * cfg.fit_max_rot_delta
        d_q = torch.tanh(delta_qpos) * cfg.fit_max_q_delta
        rot_delta = _axis_angle_to_matrix(d_rot_aa)
        trans_out = base_trans + d_trans
        rot_out = torch.matmul(rot_delta, base_rot)
        qpos_out = base_qpos + d_q
    return trans_out, rot_out, qpos_out, {"fit_loss": last_loss.detach()}


def apply_contact_diffusion_refinement(
    contact_net: ContactDiffusionNet,
    hand_model,
    feature: torch.Tensor,
    seed_points: torch.Tensor,
    init_trans: torch.Tensor,
    init_rot: torch.Tensor,
    init_qpos: torch.Tensor,
    object_pc: torch.Tensor,
    cfg: Optional[ContactDiffusionConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    cfg = cfg or contact_net.cfg
    flat = contact_net.sample_contacts(feature, seed_points, object_pc)
    target_contacts = unflatten_contacts(flat, cfg.num_fingertips)
    trans, rot, qpos, log = fit_pose_to_contacts(
        hand_model, init_trans, init_rot, init_qpos, target_contacts, object_pc=object_pc, cfg=cfg
    )
    return trans, rot, qpos, log


def build_contact_diffusion_cfg(config_dict: dict) -> ContactDiffusionConfig:
    c = config_dict or {}
    return ContactDiffusionConfig(
        feature_dim=int(c.get("feature_dim", 512)),
        cond_dim=int(c.get("cond_dim", 256)),
        num_fingertips=int(c.get("num_fingertips", 4)),
        hidden_dim=int(c.get("hidden_dim", 512)),
        object_feat_dim=int(c.get("object_feat_dim", 128)),
        target_representation=str(c.get("target_representation", "offset")),
        contact_thresh=float(c.get("contact_thresh", 0.015)),
        num_train_timesteps=int(c.get("num_train_timesteps", 1000)),
        num_inference_timesteps=int(c.get("num_inference_timesteps", 100)),
        prediction_type=str(c.get("prediction_type", "epsilon")),
        log_prob_type=c.get("log_prob_type", None),
        ode=bool(c.get("ode", True)),
        pointnext_k=int(c.get("pointnext_k", 16)),
        pointnext_samples1=int(c.get("pointnext_samples1", 256)),
        pointnext_samples2=int(c.get("pointnext_samples2", 64)),
        fit_steps=int(c.get("fit_steps", 8)),
        fit_lr=float(c.get("fit_lr", 5e-3)),
        fit_max_trans_delta=float(c.get("fit_max_trans_delta", 0.01)),
        fit_max_rot_delta=float(c.get("fit_max_rot_delta", 0.26)),
        fit_max_q_delta=float(c.get("fit_max_q_delta", 0.10)),
        fit_w_contact=float(c.get("fit_w_contact", 1.0)),
        fit_w_pose=float(c.get("fit_w_pose", 0.05)),
        fit_w_joint=float(c.get("fit_w_joint", 0.02)),
    )


def save_contact_diffusion_ckpt(
    path: str,
    model: ContactDiffusionNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
        "cfg": asdict(model.cfg),
        "version": "contact_diffusion_v2d_pointnext_offset",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_contact_diffusion_net(
    ckpt_path: str,
    device: torch.device | str = "cpu",
) -> ContactDiffusionNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    version = ckpt.get("version", "")
    if version != "contact_diffusion_v2d_pointnext_offset":
        raise ValueError(
            "PointNeXt offset contact loader expects "
            f"version='contact_diffusion_v2d_pointnext_offset', got {version!r}"
        )
    cfg_dict = dict(ckpt.get("cfg", {}))
    if cfg_dict.get("target_representation", "offset") != "offset":
        raise ValueError(
            "PointNeXt offset contact loader expects "
            f"cfg.target_representation='offset', got {cfg_dict.get('target_representation')!r}"
        )
    if int(cfg_dict.get("num_fingertips", DEFAULT_NUM_FINGERTIPS)) > 16:
        cfg_dict["num_fingertips"] = DEFAULT_NUM_FINGERTIPS
    cfg = ContactDiffusionConfig(**cfg_dict)
    model = ContactDiffusionNet(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model
