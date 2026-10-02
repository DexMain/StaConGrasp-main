"""
Contact–Stability v4 GT：4 维稳定性标签 + elongated gate。

stability_gt [B, 4]:
  0 com_offset        : 接触质心沿长轴偏移（归一化到 [0,1]）
  1 moment_arm        : 重力力臂 proxy（归一化）
  2 bilateral_score   : 短轴双侧夹持质量 [0,1]
  3 elongated_stability : 细长体综合稳定性 [0,1]

elongated_gate: object_pc PCA λ1/λ2 > ratio_thresh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

STABILITY_DIM = 4
STABILITY_NAMES = ("com_offset", "moment_arm", "bilateral_score", "elongated_stability")


@dataclass
class StabilityGTConfig:
    elongated_ratio_thresh: float = 8.0
    min_object_points: int = 32
    gravity_dir: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    bilateral_tau: float = 0.01
    long_decay: float = 2.0
    moment_decay: float = 2.0


def masked_mean(
    pts: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    if mask is None:
        return pts.mean(dim=dim)
    w = mask.clamp(min=0.0)
    denom = w.sum(dim=dim).clamp(min=1.0)
    return (pts * w.unsqueeze(-1)).sum(dim=dim) / denom.unsqueeze(-1)


def pca_axes_batch(
    object_pc: torch.Tensor,
    min_points: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        com (B,3), axis_long (B,3), axis_s1 (B,3), axis_s2 (B,3),
        half_length (B,), aspect_ratio (B,)
    """
    b, _, _ = object_pc.shape
    device = object_pc.device
    dtype = object_pc.dtype

    valid = object_pc.abs().sum(dim=-1) > 1e-8
    com = torch.zeros(b, 3, device=device, dtype=dtype)
    axis_long = torch.zeros(b, 3, device=device, dtype=dtype)
    axis_s1 = torch.zeros(b, 3, device=device, dtype=dtype)
    axis_s2 = torch.zeros(b, 3, device=device, dtype=dtype)
    half_length = torch.ones(b, device=device, dtype=dtype)
    aspect_ratio = torch.ones(b, device=device, dtype=dtype)

    for i in range(b):
        pts = object_pc[i, valid[i]]
        if pts.shape[0] < min_points:
            axis_long[i, 2] = 1.0
            axis_s1[i, 0] = 1.0
            axis_s2[i, 1] = 1.0
            continue
        
        # PCA 计算物体点云的质心、主轴、长短轴比
        c = pts.mean(dim=0)
        centered = pts - c.unsqueeze(0)
        cov = centered.t().matmul(centered) / max(int(pts.shape[0]), 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)

        al = evecs[:, order[0]]
        as1 = evecs[:, order[1]]
        as2 = evecs[:, order[2]]
        al = al / (al.norm() + 1e-8)
        as1 = as1 / (as1.norm() + 1e-8)
        as2 = as2 / (as2.norm() + 1e-8)

        e_max = torch.sqrt(torch.clamp(evals[order[0]], min=1e-10))
        e_min = torch.sqrt(torch.clamp(evals[order[2]], min=1e-10))
        ar = (e_max / (e_min + 1e-8)).clamp(min=1.0)
        half_ext = centered.matmul(al).abs().max().clamp(min=1e-3)

        com[i] = c
        axis_long[i] = al
        axis_s1[i] = as1
        axis_s2[i] = as2
        half_length[i] = half_ext
        aspect_ratio[i] = ar

    return com, axis_long, axis_s1, axis_s2, half_length, aspect_ratio


def compute_bilateral_score_on_axis(
    contact_pts: torch.Tensor,
    com: torch.Tensor,
    axis: torch.Tensor,
    contact_mask: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    rel = contact_pts - com.unsqueeze(1)
    u = (rel * axis.unsqueeze(1)).sum(dim=-1)

    if contact_mask is not None:
        invalid = contact_mask < 0.5
        u_max = u.masked_fill(invalid, float("-inf")).max(dim=-1).values
        u_min = u.masked_fill(invalid, float("inf")).min(dim=-1).values
        has_contact = contact_mask.sum(dim=-1) >= 1
        u_max = torch.where(has_contact, u_max, torch.zeros_like(u_max))
        u_min = torch.where(has_contact, u_min, torch.zeros_like(u_min))
    else:
        u_max = u.max(dim=-1).values
        u_min = u.min(dim=-1).values

    tau = max(float(tau), 1e-6)
    return (
        torch.sigmoid(u_max / tau) * torch.sigmoid(-u_min / tau)
    ).clamp(0.0, 1.0)


def compute_bilateral_score(
    contact_pts: torch.Tensor,
    com: torch.Tensor,
    axis_s1: torch.Tensor,
    contact_mask: Optional[torch.Tensor],
    tau: float,
    axis_s2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """短轴 bilateral；若提供 axis_s2，取两轴上的最大值。"""
    score_s1 = compute_bilateral_score_on_axis(
        contact_pts, com, axis_s1, contact_mask, tau
    )
    if axis_s2 is None:
        return score_s1
    score_s2 = compute_bilateral_score_on_axis(
        contact_pts, com, axis_s2, contact_mask, tau
    )
    return torch.maximum(score_s1, score_s2)


def compute_stability_metrics(
    contact_pts: torch.Tensor,
    object_pc: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    cfg: Optional[StabilityGTConfig] = None,
) -> Dict[str, torch.Tensor]:
    """从 contact 点与 object_pc 计算 stability 中间量（训练 GT 与推理 energy 共用）。"""
    cfg = cfg or StabilityGTConfig()
    device = contact_pts.device
    dtype = contact_pts.dtype

    com, axis_long, axis_s1, axis_s2, half_length, aspect_ratio = pca_axes_batch(
        object_pc, min_points=cfg.min_object_points
    )
    contact_center = masked_mean(contact_pts, contact_mask, dim=1)

    offset_vec = contact_center - com
    long_offset = (offset_vec * axis_long).sum(dim=-1).abs()
    long_offset_norm = (long_offset / half_length.clamp(min=1e-3)).clamp(0.0, 2.0)

    g = torch.tensor(cfg.gravity_dir, device=device, dtype=dtype)
    g = g / (g.norm() + 1e-8)
    moment_arm = torch.cross(offset_vec, g.expand_as(offset_vec), dim=-1).norm(dim=-1)
    moment_arm_norm = (moment_arm / half_length.clamp(min=1e-3)).clamp(0.0, 2.0)

    bilateral_score = compute_bilateral_score(
        contact_pts,
        com,
        axis_s1,
        contact_mask,
        cfg.bilateral_tau,
        axis_s2=axis_s2,
    )

    elongated_gate = (aspect_ratio >= cfg.elongated_ratio_thresh).float()
    elongated_stability = (
        torch.exp(-cfg.long_decay * long_offset_norm)
        * bilateral_score
        * torch.exp(-cfg.moment_decay * moment_arm_norm)
    ).clamp(0.0, 1.0)

    return {
        "com": com,
        "axis_long": axis_long,
        "axis_s1": axis_s1,
        "axis_s2": axis_s2,
        "half_length": half_length,
        "aspect_ratio": aspect_ratio,
        "elongated_gate": elongated_gate,
        "long_offset": long_offset,
        "long_offset_norm": long_offset_norm,
        "moment_arm": moment_arm,
        "moment_arm_norm": moment_arm_norm,
        "bilateral_score": bilateral_score,
        "elongated_stability": elongated_stability,
        "contact_center": contact_center,
    }

'''
    稳定性 GT 需要的是 (B, F, 3) 的结构化接触点，因为内部要按指尖做几何运算，例如：
        质心偏移（com_offset）：接触点集合与物体质心的关系
        力矩臂（moment_arm）：接触点相对某轴/中心的力矩
        双侧对称（bilateral_score）：不同指尖/contact 之间的空间对称性
        细长物体稳定性（elongated_stability）：结合物体 PCA 主轴与接触分布
        这些都要 逐个指尖访问 (x,y,z)，(B, 12) 的 flat 形式不方便直接做；(B, 4, 3) 才符合 (B, F, 3) 的语义。

        contact_mask 也是 (B, F)，与 unflatten 后的 contacts 在指尖维度上一一对应

'''
def compute_stability_gt_batch(
    contact_pts: torch.Tensor,
    object_pc: torch.Tensor,
    gt_rot: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    cfg: Optional[StabilityGTConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        contact_pts: (B, F, 3)
        object_pc: (B, N, 3)
        gt_rot: (B, 3, 3) 保留接口，v4 GT 不直接使用
        contact_mask: (B, F) optional

    Returns:
        stability_gt (B, 4), elongated_gate (B,), parts
    """
    del gt_rot
    parts = compute_stability_metrics(
        contact_pts, object_pc, contact_mask=contact_mask, cfg=cfg
    )
    stability_gt = torch.stack(
        [
            parts["long_offset_norm"],
            parts["moment_arm_norm"],
            parts["bilateral_score"],
            parts["elongated_stability"],
        ],
        dim=-1,
    )
    return stability_gt, parts["elongated_gate"], parts
