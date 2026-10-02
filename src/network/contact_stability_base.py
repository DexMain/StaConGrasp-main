from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from network.contact_diffusion_base import (
    ContactDiffusionConfig,
    ContactDiffusionNet,
    build_contact_diffusion_cfg,
    load_contact_diffusion_net,
)
from utils.contact_gt_surface import unflatten_contacts
from utils.stability_gt_metrics import (
    STABILITY_DIM,
    StabilityGTConfig,
    compute_stability_gt_batch,
)


@dataclass
class ContactStabilityConfig:
    contact: ContactDiffusionConfig
    shape_feat_dim: int = 32
    stability_hidden_dim: int = 256
    elongated_ratio_thresh: float = 8.0
    lambda_stab: float = 1.0
    elongated_loss_boost: float = 3.0

    @property
    def stability_gt_cfg(self) -> StabilityGTConfig:
        return StabilityGTConfig(
            elongated_ratio_thresh=self.elongated_ratio_thresh,
        )


class ContactStabilityNet(nn.Module):
    """v4: ContactDiffusionNet + shape encoder + stability head (4-d)."""

    def __init__(self, cfg: ContactStabilityConfig):
        super().__init__()
        self.cfg = cfg
        self.contact_net = ContactDiffusionNet(cfg.contact)
        cd = cfg.contact

        self.shape_encoder = nn.Sequential(
            nn.Linear(7, cfg.shape_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shape_feat_dim, cfg.shape_feat_dim),
            nn.ReLU(inplace=True),
        )
        in_stab = cd.cond_dim + cfg.shape_feat_dim + cd.contact_dim
        self.stability_head = nn.Sequential(
            nn.Linear(in_stab, cfg.stability_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.stability_hidden_dim, cfg.stability_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.stability_hidden_dim, STABILITY_DIM),
        )

    @property
    def contact_cfg(self) -> ContactDiffusionConfig:
        return self.contact_net.cfg

    def encode_condition(
        self,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        return self.contact_net.encode_condition(feature, seed_points, object_pc)

    def encode_shape(self, object_pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """object_pc (B,N,3) → shape_feat (B,D), elongated_gate (B,) float."""
        valid = object_pc.abs().sum(dim=-1) > 1e-8
        b = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype
        center = torch.zeros(b, 3, device=device, dtype=dtype)
        major = torch.zeros(b, 3, device=device, dtype=dtype)
        ratio = torch.ones(b, 1, device=device, dtype=dtype)
        for i in range(b):
            pts = object_pc[i, valid[i]]
            if pts.shape[0] < 8:
                major[i, 2] = 1.0
                continue
            c = pts.mean(dim=0)
            centered = pts - c.unsqueeze(0)
            cov = centered.t().matmul(centered) / max(int(pts.shape[0]), 1)
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            axis = evecs[:, order[0]]
            axis = axis / (axis.norm() + 1e-8)
            e_max = torch.sqrt(torch.clamp(evals[order[0]], min=1e-10))
            e_min = torch.sqrt(torch.clamp(evals[order[2]], min=1e-10))
            ar = (e_max / (e_min + 1e-8)).clamp(min=1.0)
            center[i] = c
            major[i] = axis
            ratio[i, 0] = ar
        shape_vec = torch.cat([center, major, ratio], dim=-1)
        feat = self.shape_encoder(shape_vec)
        gate = (ratio[:, 0] >= self.cfg.elongated_ratio_thresh).float()
        return feat, gate

    def predict_stability(
        self,
        cond: torch.Tensor,
        shape_feat: torch.Tensor,
        contacts_flat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([cond, shape_feat, contacts_flat], dim=-1)
        pred = self.stability_head(x)
        pred = pred.clone()
        pred[:, 2:] = pred[:, 2:].sigmoid()
        pred[:, :2] = pred[:, :2].relu().clamp(0.0, 2.0)
        return pred

    def forward(
        self,
        gt_contacts_flat: torch.Tensor,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
        gt_rot: torch.Tensor,
        contact_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_contact = self.contact_net(
            gt_contacts_flat, feature, seed_points, object_pc
        )
        cond = self.encode_condition(feature, seed_points, object_pc)
        shape_feat, elong_gate = self.encode_shape(object_pc)
        contacts = unflatten_contacts(gt_contacts_flat, self.contact_cfg.num_fingertips)

        with torch.no_grad():
            stability_gt, gt_gate, _ = compute_stability_gt_batch(
                contacts,
                object_pc,
                gt_rot,
                contact_mask=contact_mask,
                cfg=self.cfg.stability_gt_cfg,
            )

        stab_pred = self.predict_stability(cond, shape_feat, gt_contacts_flat)

        w = torch.ones_like(elong_gate)
        w = w + (self.cfg.elongated_loss_boost - 1.0) * elong_gate

        loss_com = (w * (stab_pred[:, 0] - stability_gt[:, 0]).square()).mean()
        loss_moment = (w * (stab_pred[:, 1] - stability_gt[:, 1]).square()).mean()
        loss_bilateral = (w * (stab_pred[:, 2] - stability_gt[:, 2]).square()).mean()
        loss_elong = (w * (stab_pred[:, 3] - stability_gt[:, 3]).square()).mean()
        loss_stab = loss_com + loss_moment + loss_bilateral + loss_elong

        loss = loss_contact + self.cfg.lambda_stab * loss_stab
        logs = {
            "loss_contact": loss_contact.detach(),
            "loss_stab": loss_stab.detach(),
            "loss_com": loss_com.detach(),
            "loss_moment": loss_moment.detach(),
            "loss_bilateral": loss_bilateral.detach(),
            "loss_elong": loss_elong.detach(),
            "elongated_ratio": elong_gate.mean().detach(),
            "stab_gt_mean": stability_gt.mean().detach(),
            "stab_pred_mean": stab_pred.mean().detach(),
        }
        return loss, logs

    def sample_contacts(
        self,
        feature: torch.Tensor,
        seed_points: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.contact_net.sample_contacts(feature, seed_points, object_pc)


def build_contact_stability_cfg(config_dict: dict) -> ContactStabilityConfig:
    c = config_dict or {}
    contact_cfg = build_contact_diffusion_cfg(c.get("contact", c))
    return ContactStabilityConfig(
        contact=contact_cfg,
        shape_feat_dim=int(c.get("shape_feat_dim", 32)),
        stability_hidden_dim=int(c.get("stability_hidden_dim", 256)),
        elongated_ratio_thresh=float(c.get("elongated_ratio_thresh", 8.0)),
        lambda_stab=float(c.get("lambda_stab", 1.0)),
        elongated_loss_boost=float(c.get("elongated_loss_boost", 3.0)),
    )


def save_contact_stability_ckpt(
    path: str,
    model: ContactStabilityNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
        "cfg": asdict(model.cfg),
        "version": "contact_stability_v4",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_contact_stability_net(
    ckpt_path: str,
    device: torch.device | str = "cpu",
    contact_v2_ckpt: Optional[str] = None,
) -> ContactStabilityNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "version" in ckpt and ckpt.get("cfg", {}).get("contact"):
        cfg = ContactStabilityConfig(
            contact=ContactDiffusionConfig(**ckpt["cfg"]["contact"]),
            shape_feat_dim=int(ckpt["cfg"].get("shape_feat_dim", 32)),
            stability_hidden_dim=int(ckpt["cfg"].get("stability_hidden_dim", 256)),
            elongated_ratio_thresh=float(ckpt["cfg"].get("elongated_ratio_thresh", 8.0)),
            lambda_stab=float(ckpt["cfg"].get("lambda_stab", 1.0)),
            elongated_loss_boost=float(ckpt["cfg"].get("elongated_loss_boost", 3.0)),
        )
    else:
        cfg = build_contact_stability_cfg({})

    model = ContactStabilityNet(cfg).to(device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    elif contact_v2_ckpt:
        contact_only = load_contact_diffusion_net(contact_v2_ckpt, device=device)
        model.contact_net.load_state_dict(contact_only.state_dict(), strict=True)

    model.eval()
    return model


def init_contact_stability_from_v2(
    model: ContactStabilityNet,
    v2_ckpt_path: str,
    device: torch.device | str,
) -> None:
    contact_net = load_contact_diffusion_net(v2_ckpt_path, device=device)
    model.contact_net.load_state_dict(contact_net.state_dict(), strict=True)
