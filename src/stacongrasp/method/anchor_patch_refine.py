from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.contact_gt_surface import aggregate_fingertips_per_link
from utils.stability_gt_metrics import (
    StabilityGTConfig,
    compute_stability_metrics,
)
from optimizer.stability_energy_refine_pen_gated import (
    PhysicsGuidanceV4StabilityPenGatedConfig,
    PhysicsGuidedPoseRefinerV4StabilityPenGated,
)

LAST_PATCH_LOG = {}


@dataclass
class AnchorPatchConfig:
    patch_radius: float = 0.02
    patch_topk: int = 64
    patch_sigma: float = 0.01
    normal_radius: float = 0.025
    normal_topk: int = 32
    normal_weight: float = 0.0
    reach_sigma: float = 0.03
    reach_weight: float = 0.0
    min_patch_points: int = 4
    stability_coverage_weight: float = 0.05
    stability_spread_weight: float = 0.05
    accept_require_patch_improve: bool = True
    accept_patch_min_improve: float = 1e-8
    accept_stability_eps: float = 0.0


def _valid_object_mask(object_pc: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(object_pc).all(dim=-1) & (object_pc.abs().sum(dim=-1) > 1e-8)


def _batched_knn_to_object(
    query: torch.Tensor,
    object_pc: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, q, _ = query.shape
    n = object_pc.shape[1]
    k = max(1, min(int(k), int(n)))
    all_idx = torch.zeros(b, q, k, device=query.device, dtype=torch.long)
    all_dist = torch.full((b, q, k), float("inf"), device=query.device, dtype=query.dtype)
    valid = _valid_object_mask(object_pc)
    for bi in range(b):
        obj = object_pc[bi]
        valid_idx = torch.nonzero(valid[bi], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        pts = obj[valid_idx]
        d = torch.cdist(query[bi : bi + 1], pts.unsqueeze(0))[0]
        kk = min(k, int(pts.shape[0]))
        dist, local = torch.topk(d, kk, dim=-1, largest=False)
        idx = valid_idx[local]
        all_idx[bi, :, :kk] = idx
        all_dist[bi, :, :kk] = dist
        if kk < k:
            all_idx[bi, :, kk:] = idx[:, -1:].expand(-1, k - kk)
            all_dist[bi, :, kk:] = dist[:, -1:].expand(-1, k - kk)
    return all_idx, all_dist


def project_anchors_to_object(
    anchors: torch.Tensor,
    object_pc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx, dist = _batched_knn_to_object(anchors, object_pc, k=1)
    gather = idx[..., 0].unsqueeze(-1).expand(-1, -1, 3)
    projected = torch.gather(object_pc, 1, gather)
    return projected, dist[..., 0]


def estimate_patch_normals(
    centers: torch.Tensor,
    object_pc: torch.Tensor,
    topk: int = 32,
) -> torch.Tensor:
    idx, _ = _batched_knn_to_object(centers, object_pc, k=topk)
    b, f, k = idx.shape
    gather = idx.reshape(b, f * k).unsqueeze(-1).expand(-1, -1, 3)
    neigh = torch.gather(object_pc, 1, gather).reshape(b, f, k, 3)
    rel = neigh - neigh.mean(dim=2, keepdim=True)
    cov = rel.transpose(-1, -2).matmul(rel) / max(int(k), 1)
    evals, evecs = torch.linalg.eigh(cov)
    normals = evecs[..., 0]
    view_dir = centers - object_pc.mean(dim=1, keepdim=True)
    flip = (normals * view_dir).sum(dim=-1, keepdim=True) < 0
    normals = torch.where(flip, -normals, normals)
    return F.normalize(normals, dim=-1, eps=1e-8)

# 把 4 个点变成 4 个 patch
def build_anchor_patches(
    anchors: torch.Tensor,
    object_pc: torch.Tensor,
    tips: Optional[torch.Tensor] = None,
    cfg: Optional[AnchorPatchConfig] = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or AnchorPatchConfig()
    centers, projected_dist = project_anchors_to_object(anchors, object_pc)
    idx, dist = _batched_knn_to_object(centers, object_pc, k=int(cfg.patch_topk))
    b, f, k = idx.shape
    gather = idx.reshape(b, f * k).unsqueeze(-1).expand(-1, -1, 3)
    points = torch.gather(object_pc, 1, gather).reshape(b, f, k, 3)

    radius = max(float(cfg.patch_radius), 1e-6)
    sigma = max(float(cfg.patch_sigma), 1e-6)
    mask = dist <= radius
    enough = mask.sum(dim=-1, keepdim=True) >= max(int(cfg.min_patch_points), 1)
    fallback_k = min(max(int(cfg.min_patch_points), 1), k)
    rank = torch.arange(k, device=dist.device).view(1, 1, k)
    fallback = rank < fallback_k
    mask = torch.where(enough, mask, fallback)

    weights = torch.exp(-0.5 * (dist / sigma).square()) * mask.float()
    normals = estimate_patch_normals(centers, object_pc, topk=int(cfg.normal_topk))
    if tips is not None and float(cfg.reach_weight) > 0.0:
        reach_sigma = max(float(cfg.reach_sigma), 1e-6)
        reach = torch.linalg.norm(points - tips.unsqueeze(2), dim=-1)
        reach_w = torch.exp(-0.5 * (reach / reach_sigma).square())
        weights = weights * (1.0 + float(cfg.reach_weight) * reach_w)

    denom = weights.sum(dim=-1).clamp_min(1e-8)
    representatives = (points * weights.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
    spread = (torch.linalg.norm(points - representatives.unsqueeze(2), dim=-1) * weights).sum(
        dim=-1
    ) / denom
    coverage = mask.float().mean(dim=-1)
    return {
        "anchor_contacts": anchors,
        "patch_centers": centers,
        "patch_points": points,
        "patch_mask": mask,
        "patch_weights": weights,
        "patch_normals": normals,
        "patch_representatives": representatives,
        "patch_coverage": coverage,
        "patch_spread": spread,
        "anchor_projected_dist": projected_dist,
    }

# 指尖到patch区域的加权距离---目的是让指尖更贴近，E_patch 没降，说明优化 没真正改善接触匹配，不应采纳
def tip_to_patch_energy(
    tips: torch.Tensor,
    patch_points: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_weights: torch.Tensor,
    patch_normals: Optional[torch.Tensor] = None,
    normal_weight: float = 0.0,
) -> torch.Tensor:
    dist2 = (tips.unsqueeze(2) - patch_points).square().sum(dim=-1)
    weights = patch_weights * patch_mask.float()
    masked = dist2 * weights
    e_dist = masked.sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-8)
    e = e_dist.mean(dim=-1)
    if patch_normals is not None and float(normal_weight) > 0.0:
        nearest = torch.argmin(
            dist2.masked_fill(~patch_mask, float("inf")),
            dim=-1,
        )
        nearest_pts = torch.gather(
            patch_points,
            2,
            nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
        ).squeeze(2)
        approach = F.normalize(tips - nearest_pts, dim=-1, eps=1e-8)
        normal_pen = (approach * patch_normals).sum(dim=-1).abs()
        e = e + float(normal_weight) * normal_pen.mean(dim=-1)
    return e


class AnchorPatchPoseRefiner(PhysicsGuidedPoseRefinerV4StabilityPenGated):
    """v16e-compatible refiner that replaces tip-to-point with tip-to-patch."""

    def __init__(
        self,
        hand_model=None,
        cfg: Optional[PhysicsGuidanceV4StabilityPenGatedConfig] = None,
        patch_cfg: Optional[AnchorPatchConfig] = None,
    ):
        super().__init__(hand_model=hand_model, cfg=cfg)
        self.patch_cfg = patch_cfg or AnchorPatchConfig()
        self.last_patch_log = {}
    
    # tip-to-patch + SDF + stab
    def compute_energy(
        self,
        pose: torch.Tensor,
        init_pose: torch.Tensor,
        object_pc: torch.Tensor,
        sdf_grid: torch.Tensor,
        sdf_origin: torch.Tensor,
        sdf_voxel_size: torch.Tensor,
        T_cam_to_obj: torch.Tensor,
        target_contacts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        trans, rot, qpos = self.unpack_pose(pose)
        init_trans, _, init_qpos = self.unpack_pose(init_pose)

        if target_contacts is not None and cfg.detach_target_contacts:
            target_contacts = target_contacts.detach()

        num_fingers = int(target_contacts.shape[1]) if target_contacts is not None else 4
        tips = self._forward_fingertips(trans, rot, qpos, object_pc, num_fingers=num_fingers)
        # 用 patch 约束优化 pose
        if target_contacts is not None:
            patch = build_anchor_patches(target_contacts, object_pc, tips=tips, cfg=self.patch_cfg)
            tip_anchor = tip_to_patch_energy(
                tips,
                patch["patch_points"],
                patch["patch_mask"],
                patch["patch_weights"],
                patch_normals=patch["patch_normals"],
                normal_weight=float(self.patch_cfg.normal_weight),
            )
        else:
            patch = {}
            tip_anchor = torch.zeros(trans.shape[0], device=trans.device, dtype=trans.dtype)

        pen_pts = tips if cfg.pen_on_tips_only else self.compute_hand_points(trans, rot, qpos)
        pen_pts_obj = self.transform_points_cam_to_obj(pen_pts, T_cam_to_obj)
        sdf = self.query_sdf_trilinear(pen_pts_obj, sdf_grid, sdf_origin, sdf_voxel_size)
        pen = F.relu(cfg.pen_margin - sdf).mean(dim=-1)

        e_stab, stab_parts = self._stability_energy(tips, object_pc)
        if patch:
            e_stab = (
                e_stab
                + float(self.patch_cfg.stability_spread_weight)
                * patch["patch_spread"].mean(dim=-1)
                - float(self.patch_cfg.stability_coverage_weight)
                * patch["patch_coverage"].mean(dim=-1)
            )
            patch_metrics = compute_stability_metrics(
                patch["patch_representatives"],
                object_pc,
                contact_mask=None,
                cfg=cfg.stability_cfg or StabilityGTConfig(),
            )
            patch_score = (
                -patch_metrics["long_offset_norm"]
                - patch_metrics["moment_arm_norm"]
                + patch_metrics["bilateral_score"]
                - float(self.patch_cfg.stability_spread_weight)
                * patch["patch_spread"].mean(dim=-1)
                + float(self.patch_cfg.stability_coverage_weight)
                * patch["patch_coverage"].mean(dim=-1)
            )

        pose_reg = torch.norm(trans - init_trans, dim=-1).pow(2)
        joint_reg = torch.norm(qpos - init_qpos, dim=-1).pow(2)
        energy = (
            cfg.w_tip * tip_anchor
            + cfg.w_pen * pen
            + e_stab
            + cfg.w_pose * pose_reg
            + cfg.w_joint * joint_reg
        )
        parts = {
            "E_total": energy.detach(),
            "E_contact": tip_anchor.detach(),
            "E_tip_anchor": tip_anchor.detach(),
            "E_patch": tip_anchor.detach(),
            "E_pen": pen.detach(),
            "E_pose": pose_reg.detach(),
            "E_joint": joint_reg.detach(),
        }
        parts.update(stab_parts)
        if patch:
            parts.update(
                {
                    "patch_coverage": patch["patch_coverage"].mean(dim=-1).detach(),
                    "patch_spread": patch["patch_spread"].mean(dim=-1).detach(),
                    "anchor_projected_dist": patch["anchor_projected_dist"].mean(dim=-1).detach(),
                    "patch_stability_score": patch_score.detach(),
                }
            )
            self.last_patch_log = {k: v.detach() for k, v in patch.items()}
            self.last_patch_log["patch_stability_score"] = patch_score.detach()
            global LAST_PATCH_LOG
            LAST_PATCH_LOG = self.last_patch_log
        return energy, parts

    def forward(
        self,
        init_trans: torch.Tensor,
        init_rot: torch.Tensor,
        init_qpos: torch.Tensor,
        object_pc: torch.Tensor,
        sdf_grid: torch.Tensor,
        sdf_origin: torch.Tensor,
        sdf_voxel_size: torch.Tensor,
        T_cam_to_obj: torch.Tensor,
        target_contacts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(
            init_trans=init_trans,
            init_rot=init_rot,
            init_qpos=init_qpos,
            object_pc=object_pc,
            sdf_grid=sdf_grid,
            sdf_origin=sdf_origin,
            sdf_voxel_size=sdf_voxel_size,
            T_cam_to_obj=T_cam_to_obj,
            target_contacts=target_contacts,
        )
        
        # 决定 selected_pose
        base_accept = out["use_refined"].bool()
        patch_ok = torch.ones_like(base_accept)
        stab_ok = torch.ones_like(base_accept)
        if bool(self.patch_cfg.accept_require_patch_improve) and "init_E_patch" in out:
            patch_improve = out["init_E_patch"] - out["final_E_patch"]
            patch_ok = patch_improve > float(self.patch_cfg.accept_patch_min_improve)
            out["patch_improve"] = patch_improve.detach()
        # E_stab：基于 PCA 的稳定性 energy（长轴偏移、力矩臂、双侧对称等，细长物体加权）
        if "init_E_stab" in out and "final_E_stab" in out:
            stab_ok = out["final_E_stab"] <= (
                out["init_E_stab"] + float(self.patch_cfg.accept_stability_eps)
            )

        accept = base_accept & patch_ok & stab_ok
        init_pose = out["init_pose"]
        refined_pose = out["refined_pose"]
        selected_pose = torch.where(accept[:, None], refined_pose, init_pose)
        selected_trans, selected_rot, selected_qpos = self.unpack_pose(selected_pose)
        out["selected_pose"] = selected_pose
        out["selected_trans"] = selected_trans
        out["selected_rot"] = selected_rot
        out["selected_qpos"] = selected_qpos
        out["use_refined_base"] = base_accept.float()
        out["use_refined"] = accept.float()
        out["patch_accept_ok"] = patch_ok.float()
        out["patch_stability_ok"] = stab_ok.float()
        global LAST_PATCH_LOG
        LAST_PATCH_LOG = dict(LAST_PATCH_LOG)
        LAST_PATCH_LOG["patch_accept_ok"] = patch_ok.detach().float()
        LAST_PATCH_LOG["patch_stability_ok"] = stab_ok.detach().float()
        if "patch_improve" in out:
            LAST_PATCH_LOG["patch_improve"] = out["patch_improve"].detach()

        n = int(accept.numel())
        patch_score = LAST_PATCH_LOG.get("patch_stability_score")
        coverage = LAST_PATCH_LOG.get("patch_coverage")
        spread = LAST_PATCH_LOG.get("patch_spread")
        proj = LAST_PATCH_LOG.get("anchor_projected_dist")
        e_patch_init = out.get("init_E_patch")
        e_patch_final = out.get("final_E_patch")
        print(
            "[AnchorPatch] refine "
            f"n={n} "
            f"base_accept={float(base_accept.float().mean()):.3f} "
            f"patch_ok={float(patch_ok.float().mean()):.3f} "
            f"stab_ok={float(stab_ok.float().mean()):.3f} "
            f"final_accept={float(accept.float().mean()):.3f} "
            f"({int(accept.sum().item())}/{n}) | "
            f"E_patch {float(e_patch_init.mean()) if e_patch_init is not None else float('nan'):.6f}"
            f" -> {float(e_patch_final.mean()) if e_patch_final is not None else float('nan'):.6f} | "
            f"patch_score mean={float(patch_score.mean()) if patch_score is not None else float('nan'):.4f} | "
            f"coverage mean={float(coverage.mean()) if coverage is not None else float('nan'):.4f} | "
            f"spread mean={float(spread.mean()) if spread is not None else float('nan'):.4f} | "
            f"anchor_proj_dist mean={float(proj.mean()) if proj is not None else float('nan'):.4f}",
            flush=True,
        )
        return out


__all__ = [
    "AnchorPatchConfig",
    "AnchorPatchPoseRefiner",
    "build_anchor_patches",
    "project_anchors_to_object",
    "tip_to_patch_energy",
]
