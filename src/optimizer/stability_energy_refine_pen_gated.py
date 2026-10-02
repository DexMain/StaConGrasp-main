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
from optimizer.physics_guided_diffusion_patch import PhysicsGuidedPoseRefiner


@dataclass
class PhysicsGuidanceV4StabilityPenGatedConfig:
    steps: int = 3
    lr: float = 5e-4
    w_tip: float = 0.5
    w_pen: float = 5.0
    w_mid: float = 1.0
    w_moment: float = 1.0
    w_bilateral: float = 0.5
    w_pose: float = 10.0
    w_joint: float = 0.05
    pen_margin: float = 0.0
    pen_threshold: float = 0.002
    min_pen_improve: float = 1e-6
    max_trans_delta: float = 0.003
    max_q_delta: float = 0.08
    pen_on_tips_only: bool = True
    detach_init: bool = True
    detach_target_contacts: bool = True
    stability_cfg: Optional[StabilityGTConfig] = None


class PhysicsGuidedPoseRefinerV4StabilityPenGated(PhysicsGuidedPoseRefiner):
    """V4b: stability energy refiner + v14-style pen-gated accept."""

    def __init__(
        self,
        hand_model=None,
        cfg: Optional[PhysicsGuidanceV4StabilityPenGatedConfig] = None,
    ):
        super().__init__(hand_model=hand_model, cfg=None)
        self.cfg = cfg or PhysicsGuidanceV4StabilityPenGatedConfig()

    def _forward_fingertips(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        qpos: torch.Tensor,
        object_pc: torch.Tensor,
        num_fingers: int,
    ) -> torch.Tensor:
        if hasattr(self.hand_model, "forward_hand_points"):
            _, _, raw = self.hand_model.forward_hand_points(trans, rot, qpos)
        elif hasattr(self.hand_model, "_sdf_adam"):
            _, _, raw = self.hand_model._sdf_adam.forward_hand_points(trans, rot, qpos)
        else:
            raise AttributeError("hand_model must expose forward_hand_points")
        return aggregate_fingertips_per_link(
            raw, num_fingers, object_pc=object_pc
        )

    def _stability_energy(
        self,
        tips: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        stab_cfg = cfg.stability_cfg or StabilityGTConfig()
        metrics = compute_stability_metrics(tips, object_pc, contact_mask=None, cfg=stab_cfg)
        gate = metrics["elongated_gate"]
        e_mid = metrics["long_offset_norm"]
        e_moment = metrics["moment_arm_norm"]
        e_bilateral_pen = 1.0 - metrics["bilateral_score"]
        e_stab = gate * (
            cfg.w_mid * e_mid
            + cfg.w_moment * e_moment
            + cfg.w_bilateral * e_bilateral_pen
        )
        parts = {
            "E_stab": e_stab.detach(),
            "E_mid": (gate * e_mid).detach(),
            "E_moment": (gate * e_moment).detach(),
            "E_bilateral": (gate * e_bilateral_pen).detach(),
            "elongated_gate": gate.detach(),
        }
        return e_stab, parts

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
        cfg: PhysicsGuidanceV4StabilityPenGatedConfig = self.cfg
        trans, rot, qpos = self.unpack_pose(pose)
        init_trans, _, init_qpos = self.unpack_pose(init_pose)

        if target_contacts is not None and cfg.detach_target_contacts:
            target_contacts = target_contacts.detach()

        num_fingers = (
            int(target_contacts.shape[1]) if target_contacts is not None else 4
        )
        tips = self._forward_fingertips(
            trans, rot, qpos, object_pc, num_fingers=num_fingers
        )

        if target_contacts is not None:
            tip_anchor = (tips - target_contacts).square().sum(dim=-1).mean(dim=-1)
        else:
            tip_anchor = torch.zeros(trans.shape[0], device=trans.device, dtype=trans.dtype)

        if cfg.pen_on_tips_only:
            pen_pts = tips
        else:
            pen_pts = self.compute_hand_points(trans, rot, qpos)

        pen_pts_obj = self.transform_points_cam_to_obj(pen_pts, T_cam_to_obj)
        sdf = self.query_sdf_trilinear(
            pen_pts_obj, sdf_grid, sdf_origin, sdf_voxel_size
        )
        pen = F.relu(cfg.pen_margin - sdf).mean(dim=-1)

        e_stab, stab_parts = self._stability_energy(tips, object_pc)

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
            "E_pen": pen.detach(),
            "E_pose": pose_reg.detach(),
            "E_joint": joint_reg.detach(),
        }
        parts.update(stab_parts)
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
        cfg: PhysicsGuidanceV4StabilityPenGatedConfig = self.cfg
        init_pose = self.pack_pose(init_trans, init_rot, init_qpos)
        if cfg.detach_init:
            init_pose = init_pose.detach()

        pose = init_pose.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([pose], lr=cfg.lr)

        with torch.enable_grad():
            init_energy, init_parts = self.compute_energy(
                init_pose,
                init_pose,
                object_pc,
                sdf_grid,
                sdf_origin,
                sdf_voxel_size,
                T_cam_to_obj,
                target_contacts=target_contacts,
            )

            for _ in range(cfg.steps):
                opt.zero_grad(set_to_none=True)
                energy, _ = self.compute_energy(
                    pose,
                    init_pose,
                    object_pc,
                    sdf_grid,
                    sdf_origin,
                    sdf_voxel_size,
                    T_cam_to_obj,
                    target_contacts=target_contacts,
                )
                loss = energy.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([pose], max_norm=1.0)
                opt.step()

        refined_pose = pose.detach()
        final_energy, final_parts = self.compute_energy(
            refined_pose,
            init_pose,
            object_pc,
            sdf_grid,
            sdf_origin,
            sdf_voxel_size,
            T_cam_to_obj,
            target_contacts=target_contacts,
        )

        trans_delta = torch.norm(refined_pose[:, :3] - init_pose[:, :3], dim=-1)
        _, _, refined_q = self.unpack_pose(refined_pose)
        _, _, init_q = self.unpack_pose(init_pose)
        q_delta = torch.norm(refined_q - init_q, dim=-1)
        energy_improve = init_energy.detach() - final_energy.detach()
        init_E_pen = init_parts["E_pen"]
        final_E_pen = final_parts["E_pen"]
        pen_improve = init_E_pen.detach() - final_E_pen.detach()

        use_refined = (
            (init_E_pen > cfg.pen_threshold)
            & (final_E_pen < init_E_pen)
            & (pen_improve > cfg.min_pen_improve)
            & (trans_delta < cfg.max_trans_delta)
            & (q_delta < cfg.max_q_delta)
        )

        selected_pose = torch.where(use_refined[:, None], refined_pose, init_pose)
        selected_trans, selected_rot, selected_qpos = self.unpack_pose(selected_pose)
        refined_trans, refined_rot, refined_qpos = self.unpack_pose(refined_pose)

        out = {
            "selected_pose": selected_pose,
            "selected_trans": selected_trans,
            "selected_rot": selected_rot,
            "selected_qpos": selected_qpos,
            "refined_pose": refined_pose,
            "refined_trans": refined_trans,
            "refined_rot": refined_rot,
            "refined_qpos": refined_qpos,
            "init_pose": init_pose,
            "use_refined": use_refined.float(),
            "init_energy": init_energy.detach(),
            "final_energy": final_energy.detach(),
            "energy_improve": energy_improve.detach(),
            "pen_improve": pen_improve.detach(),
            "init_E_pen": init_E_pen.detach(),
            "final_E_pen": final_E_pen.detach(),
            "init_E_tip_anchor": init_parts["E_tip_anchor"].detach(),
            "final_E_tip_anchor": final_parts["E_tip_anchor"].detach(),
            "trans_delta": trans_delta.detach(),
            "q_delta": q_delta.detach(),
        }
        out.update({f"init_{k}": v for k, v in init_parts.items()})
        out.update({f"final_{k}": v for k, v in final_parts.items()})
        return out


__all__ = [
    "PhysicsGuidanceV4StabilityPenGatedConfig",
    "PhysicsGuidedPoseRefinerV4StabilityPenGated",
]
