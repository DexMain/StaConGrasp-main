"""
Anchor patch refiner with per-step dynamic patch rebuild (Scheme 3).

Each Adam step blends predicted target_contacts with current fingertip positions
to form anchors, then rebuilds local patches before computing tip-to-patch energy.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from utils.stability_gt_metrics import StabilityGTConfig, compute_stability_metrics
from optimizer.stability_energy_refine_pen_gated import (
    PhysicsGuidanceV4StabilityPenGatedConfig,
)
from stacongrasp.method.anchor_patch_refine import (
    LAST_PATCH_LOG,
    AnchorPatchConfig,
    AnchorPatchPoseRefiner,
    build_anchor_patches,
    project_anchors_to_object,
    tip_to_patch_energy,
)
from stacongrasp.method.anchor_patch_post_rerank import capture_refine_metrics


@dataclass
class AnchorPatchDynConfig(AnchorPatchConfig):
    anchor_mode: str = "tip_blend"  # fixed | tip_blend | tip_only
    blend_alpha: float = 0.35
    blend_alpha_max: float = 0.7
    adaptive_alpha: bool = True
    adaptive_alpha_tau: float = 0.02


def compute_blend_alpha(
    target_contacts: torch.Tensor,
    tips: torch.Tensor,
    cfg: AnchorPatchDynConfig,
) -> torch.Tensor:
    b = target_contacts.shape[0]
    if not bool(cfg.adaptive_alpha):
        alpha = torch.full(
            (b, 1, 1),
            float(cfg.blend_alpha),
            device=target_contacts.device,
            dtype=target_contacts.dtype,
        )
        return alpha.clamp(0.0, float(cfg.blend_alpha_max))

    dist = torch.linalg.norm(target_contacts - tips, dim=-1).mean(dim=-1, keepdim=True)
    tau = max(float(cfg.adaptive_alpha_tau), 1e-6)
    alpha = (dist / tau).clamp(0.0, float(cfg.blend_alpha_max))
    base = float(cfg.blend_alpha)
    alpha = torch.maximum(alpha, torch.full_like(alpha, base * 0.5))
    return alpha.unsqueeze(-1)


def build_dynamic_anchors(
    target_contacts: torch.Tensor,
    tips: torch.Tensor,
    object_pc: torch.Tensor,
    cfg: AnchorPatchDynConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(cfg.anchor_mode)
    if mode == "fixed":
        anchors = target_contacts
        alpha = torch.zeros(target_contacts.shape[0], 1, 1, device=target_contacts.device)
    elif mode == "tip_only":
        anchors = tips
        alpha = torch.ones(target_contacts.shape[0], 1, 1, device=target_contacts.device)
    else:
        alpha = compute_blend_alpha(target_contacts, tips, cfg)
        anchors = (1.0 - alpha) * target_contacts + alpha * tips

    projected, _ = project_anchors_to_object(anchors, object_pc)
    return projected, alpha.squeeze(-1).squeeze(-1)


class AnchorPatchPoseRefinerDyn(AnchorPatchPoseRefiner):
    """Scheme 3: rebuild patch anchors from current tips every energy evaluation."""

    def __init__(
        self,
        hand_model=None,
        cfg: Optional[PhysicsGuidanceV4StabilityPenGatedConfig] = None,
        patch_cfg: Optional[AnchorPatchDynConfig] = None,
    ):
        super().__init__(hand_model=hand_model, cfg=cfg, patch_cfg=patch_cfg)
        self.patch_cfg = patch_cfg or AnchorPatchDynConfig()

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
    ):
        import torch.nn.functional as F

        cfg = self.cfg
        trans, rot, qpos = self.unpack_pose(pose)
        init_trans, _, init_qpos = self.unpack_pose(init_pose)

        if target_contacts is not None and cfg.detach_target_contacts:
            target_contacts = target_contacts.detach()

        num_fingers = int(target_contacts.shape[1]) if target_contacts is not None else 4
        tips = self._forward_fingertips(trans, rot, qpos, object_pc, num_fingers=num_fingers)

        patch = {}
        if target_contacts is not None:
            dyn_cfg: AnchorPatchDynConfig = self.patch_cfg
            anchors, blend_alpha = build_dynamic_anchors(
                target_contacts, tips, object_pc, dyn_cfg
            )
            patch = build_anchor_patches(anchors, object_pc, tips=tips, cfg=dyn_cfg)
            patch["dynamic_blend_alpha"] = blend_alpha.detach()
            tip_anchor = tip_to_patch_energy(
                tips,
                patch["patch_points"],
                patch["patch_mask"],
                patch["patch_weights"],
                patch_normals=patch["patch_normals"],
                normal_weight=float(dyn_cfg.normal_weight),
            )
        else:
            tip_anchor = torch.zeros(trans.shape[0], device=trans.device, dtype=trans.dtype)

        pen_pts = tips if cfg.pen_on_tips_only else self.compute_hand_points(trans, rot, qpos)
        pen_pts_obj = self.transform_points_cam_to_obj(pen_pts, T_cam_to_obj)
        sdf = self.query_sdf_trilinear(pen_pts_obj, sdf_grid, sdf_origin, sdf_voxel_size)
        pen = F.relu(cfg.pen_margin - sdf).mean(dim=-1)

        e_stab, stab_parts = self._stability_energy(tips, object_pc)
        patch_score = None
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
                    "dynamic_blend_alpha": patch["dynamic_blend_alpha"].detach(),
                }
            )
            self.last_patch_log = {k: v.detach() for k, v in patch.items() if isinstance(v, torch.Tensor)}
            self.last_patch_log["patch_stability_score"] = patch_score.detach()
            # Write into anchor_patch_refine.LAST_PATCH_LOG (savez reads that module).
            # A plain ``global LAST_PATCH_LOG`` here would only set this file's binding.
            import stacongrasp.method.anchor_patch_refine as patch_mod

            patch_mod.LAST_PATCH_LOG = self.last_patch_log
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
        capture_refine_metrics(out)

        dyn_cfg: AnchorPatchDynConfig = self.patch_cfg
        blend_alpha = LAST_PATCH_LOG.get("dynamic_blend_alpha")
        n = int(out["use_refined"].numel())
        base_accept = out["use_refined_base"].bool() if "use_refined_base" in out else out["use_refined"].bool()
        patch_ok = out.get("patch_accept_ok")
        stab_ok = out.get("patch_stability_ok")
        accept = out["use_refined"].bool()
        print(
            "[AnchorPatchDyn] "
            f"mode={dyn_cfg.anchor_mode} blend_alpha={float(dyn_cfg.blend_alpha):.3f} "
            f"adaptive={int(dyn_cfg.adaptive_alpha)} | "
            f"n={n} final_accept={float(accept.float().mean()):.3f} "
            f"({int(accept.sum().item())}/{n}) | "
            f"dyn_alpha_mean={float(blend_alpha.mean()) if blend_alpha is not None else float('nan'):.4f}",
            flush=True,
        )
        return out


__all__ = [
    "AnchorPatchDynConfig",
    "AnchorPatchPoseRefinerDyn",
    "build_dynamic_anchors",
    "compute_blend_alpha",
]
