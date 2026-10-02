from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms as pttf

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
    contact_thresh: float = 0.015
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 100
    prediction_type: str = "epsilon"
    log_prob_type: Optional[str] = None
    ode: bool = True

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


class ContactDiffusionNet(nn.Module):
    """seed feature + object center → 1D Gaussian diffusion on flattened contact points."""

    def __init__(self, cfg: Optional[ContactDiffusionConfig] = None):
        super().__init__()
        self.cfg = cfg or ContactDiffusionConfig()
        in_dim = self.cfg.feature_dim + 3 + 3
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

    def encode_condition(
        self,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        obj_center = object_pc.mean(dim=1)
        x = torch.cat([feature, seed_points, obj_center], dim=-1)
        return self.cond_encoder(x)

    def forward(
        self,
        gt_contacts_flat: torch.Tensor,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.encode_condition(feature, seed_points, object_pc)
        return self.diffusion.calculate_loss(gt_contacts_flat, cond)

    @torch.no_grad()
    def sample_contacts(
        self,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.encode_condition(feature, seed_points, object_pc)
        samples, _ = self.diffusion.sample(cond)
        return samples


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
    """
    短步 Adam：使指尖贴近 predicted contact targets，带 pose 正则。
    """
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

        rot_delta = pttf.axis_angle_to_matrix(d_rot_aa)
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
        loss_pose = (
            d_trans.square().mean()
            + d_rot_aa.square().mean()
            + d_q.square().mean()
        )
        last_loss = cfg.fit_w_contact * loss_contact + cfg.fit_w_pose * loss_pose
        last_loss.backward()
        opt.step()

    with torch.no_grad():
        d_trans = torch.tanh(delta_trans) * cfg.fit_max_trans_delta
        d_rot_aa = torch.tanh(delta_rot_aa) * cfg.fit_max_rot_delta
        d_q = torch.tanh(delta_qpos) * cfg.fit_max_q_delta
        rot_delta = pttf.axis_angle_to_matrix(d_rot_aa)
        trans_out = base_trans + d_trans
        rot_out = torch.matmul(rot_delta, base_rot)
        qpos_out = base_qpos + d_q

    log = {"fit_loss": last_loss.detach()}
    return trans_out, rot_out, qpos_out, log


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
        hand_model,
        init_trans,
        init_rot,
        init_qpos,
        target_contacts,
        object_pc=object_pc,
        cfg=cfg,
    )
    return trans, rot, qpos, log


def build_contact_diffusion_cfg(config_dict: dict) -> ContactDiffusionConfig:
    c = config_dict or {}
    return ContactDiffusionConfig(
        feature_dim=int(c.get("feature_dim", 512)),
        cond_dim=int(c.get("cond_dim", 256)),
        num_fingertips=int(c.get("num_fingertips", 4)),
        hidden_dim=int(c.get("hidden_dim", 512)),
        contact_thresh=float(c.get("contact_thresh", 0.015)),
        num_train_timesteps=int(c.get("num_train_timesteps", 1000)),
        num_inference_timesteps=int(c.get("num_inference_timesteps", 100)),
        prediction_type=str(c.get("prediction_type", "epsilon")),
        log_prob_type=c.get("log_prob_type", None),
        ode=bool(c.get("ode", True)),
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
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_contact_diffusion_net(
    ckpt_path: str,
    device: torch.device | str = "cpu",
) -> ContactDiffusionNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(ckpt.get("cfg", {}))
    if int(cfg_dict.get("num_fingertips", DEFAULT_NUM_FINGERTIPS)) > 16:
        cfg_dict["num_fingertips"] = DEFAULT_NUM_FINGERTIPS
    cfg = ContactDiffusionConfig(**cfg_dict)
    model = ContactDiffusionNet(cfg).to(device)
    state = ckpt["model"]
    if any(k.startswith("denoiser.") for k in state.keys()):
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model
