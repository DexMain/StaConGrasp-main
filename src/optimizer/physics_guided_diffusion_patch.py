from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from optimizer.rot6d import robust_compute_rotation_matrix_from_ortho6d
from optimizer.sdf_adam_robot_offline import SDFAdam


@dataclass
class PhysicsGuidanceConfig:
    steps: int = 5
    lr: float = 1e-3
    w_contact: float = 1.0
    w_pen: float = 5.0
    w_pose: float = 0.05
    w_joint: float = 0.05
    pen_margin: float = 0.0
    contact_topk: int = 32
    max_trans_delta: float = 0.005
    max_q_delta: float = 0.10
    min_energy_improve: float = 1e-3
    pen_threshold: float = 0.002
    use_conservative_accept: bool = True
    detach_init: bool = True


class SDFAdamHandPointsProvider:
    """
    将 SDFAdam.forward_hand_points 暴露为 PhysicsGuidedPoseRefiner 所需的
    get_hand_points / get_surface_points 接口。

    返回相机坐标系下的手部表面点 [B, H, 3]（palm + finger shaft + fingertip 拼接）。
    """

    def __init__(
        self,
        urdf_path: str,
        meta_path: str,
        hand_name: str = "leap_hand",
        device: Union[str, torch.device] = "cuda",
        n_points_each_link: int = 32,
    ):
        self._sdf_adam = SDFAdam(
            hand_name=hand_name,
            urdf_path=urdf_path,
            meta_path=meta_path,
            parallel_num=1,
            device=str(device),
            n_points_each_link=n_points_each_link,
        )

    @property
    def robot_model(self):
        return self._sdf_adam.robot_model

    def forward_hand_points(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        qpos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sdf_adam.forward_hand_points(trans, rot, qpos)

    def get_hand_points(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        palm_pts, finger_pts, fingertip_pts = self.forward_hand_points(trans, rot, qpos)
        parts = [palm_pts, finger_pts, fingertip_pts]
        return torch.cat(parts, dim=1)

    def get_surface_points(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_hand_points(trans, rot, qpos)


class PhysicsGuidedPoseRefiner(torch.nn.Module):
    """
    对 coarse dexterous grasp 做少步可微 energy guidance。

    pose 表示：[trans(3), rot6d(6), qpos(J)]

    hand_model 推荐传入 SDFAdamHandPointsProvider，或任何实现
    get_hand_points / get_surface_points 的对象。
    """

    def __init__(self, hand_model=None, cfg: Optional[PhysicsGuidanceConfig] = None):
        super().__init__()
        self.hand_model = hand_model
        self.cfg = cfg or PhysicsGuidanceConfig()

    @staticmethod
    def matrix_to_rot6d(rot: torch.Tensor) -> torch.Tensor:
        return rot[:, :, :2].permute(0, 2, 1).reshape(rot.shape[0], 6)

    @staticmethod
    def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        return robust_compute_rotation_matrix_from_ortho6d(rot6d)

    def pack_pose(self, trans: torch.Tensor, rot: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        rot6d = self.matrix_to_rot6d(rot)
        return torch.cat([trans, rot6d, qpos], dim=-1)

    def unpack_pose(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans = pose[:, :3]
        rot6d = pose[:, 3:9]
        qpos = pose[:, 9:]
        rot = self.rot6d_to_matrix(rot6d)
        return trans, rot, qpos

    def compute_hand_points(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回 hand surface points，形状 [B, H, 3]，相机坐标系。

        优先调用 hand_model 的 get_hand_points / get_surface_points；
        若 hand_model 为 SDFAdam 实例，则直接调用其 forward_hand_points。
        """
        if self.hand_model is None:
            raise RuntimeError(
                "hand_model is None。请传入 SDFAdamHandPointsProvider 或实现 "
                "get_hand_points(trans, rot, qpos) 的适配器。"
            )

        if hasattr(self.hand_model, "get_hand_points"):
            return self.hand_model.get_hand_points(trans, rot, qpos)

        if hasattr(self.hand_model, "get_surface_points"):
            return self.hand_model.get_surface_points(trans, rot, qpos)

        if hasattr(self.hand_model, "forward_hand_points"):
            palm_pts, finger_pts, fingertip_pts = self.hand_model.forward_hand_points(
                trans, rot, qpos
            )
            return torch.cat([palm_pts, finger_pts, fingertip_pts], dim=1)

        raise RuntimeError(
            "hand_model 需实现 get_hand_points / get_surface_points / forward_hand_points 之一。"
        )

    @staticmethod
    def transform_points_cam_to_obj(points_cam: torch.Tensor, T_cam_to_obj: torch.Tensor) -> torch.Tensor:
        B, H, _ = points_cam.shape
        ones = torch.ones(B, H, 1, device=points_cam.device, dtype=points_cam.dtype)
        homo = torch.cat([points_cam, ones], dim=-1)
        obj = torch.bmm(homo, T_cam_to_obj.transpose(1, 2))[..., :3]
        return obj

    @staticmethod
    def query_sdf_trilinear(
        points_obj: torch.Tensor,
        sdf_grid: torch.Tensor,
        sdf_origin: torch.Tensor,
        sdf_voxel_size: torch.Tensor,
    ) -> torch.Tensor:
        B, H, _ = points_obj.shape
        D = sdf_grid.shape[-1]

        if sdf_voxel_size.dim() == 1:
            sdf_voxel_size = sdf_voxel_size[:, None]

        grid_coord = (points_obj - sdf_origin[:, None, :]) / sdf_voxel_size[:, None, :]
        norm = grid_coord / max(D - 1, 1) * 2.0 - 1.0

        sample_grid = norm[:, None, None, :, :]
        sdf_vol = sdf_grid[:, None]
        val = F.grid_sample(
            sdf_vol,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return val[:, 0, 0, 0, :]

    def compute_energy(
        self,
        pose: torch.Tensor,
        init_pose: torch.Tensor,
        object_pc: torch.Tensor,
        sdf_grid: torch.Tensor,
        sdf_origin: torch.Tensor,
        sdf_voxel_size: torch.Tensor,
        T_cam_to_obj: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        trans, rot, qpos = self.unpack_pose(pose)
        init_trans, _, init_qpos = self.unpack_pose(init_pose)

        hand_pts = self.compute_hand_points(trans, rot, qpos)  # palm + 手指 + 指尖上采样的全部 3D 点（不是只有 4 个指尖）

        dists = torch.cdist(hand_pts, object_pc)  # 该 grasp 对应 view 里 pred_main 物体 的点云（相机坐标系，最多 1024 点）。
        min_d = dists.min(dim=-1).values   # 对每个手面点，算到 object_pc 的最近距离 → 得到很多个 min_d。
        k = min(cfg.contact_topk, min_d.shape[-1])
        contact = torch.topk(min_d, k=k, dim=-1, largest=False).values.mean(dim=-1)

        hand_obj = self.transform_points_cam_to_obj(hand_pts, T_cam_to_obj)
        sdf = self.query_sdf_trilinear(hand_obj, sdf_grid, sdf_origin, sdf_voxel_size)
        pen = F.relu(cfg.pen_margin - sdf).mean(dim=-1)

        pose_reg = torch.norm(trans - init_trans, dim=-1).pow(2)
        joint_reg = torch.norm(qpos - init_qpos, dim=-1).pow(2)

        energy = (
            cfg.w_contact * contact
            + cfg.w_pen * pen
            + cfg.w_pose * pose_reg
            + cfg.w_joint * joint_reg
        )

        parts = {
            "E_total": energy.detach(),
            "E_contact": contact.detach(),
            "E_pen": pen.detach(),
            "E_pose": pose_reg.detach(),
            "E_joint": joint_reg.detach(),
        }
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
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
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
            )

            last_parts = init_parts
            for _ in range(cfg.steps):
                opt.zero_grad(set_to_none=True)
                energy, parts = self.compute_energy(
                    pose,
                    init_pose,
                    object_pc,
                    sdf_grid,
                    sdf_origin,
                    sdf_voxel_size,
                    T_cam_to_obj,
                )
                loss = energy.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([pose], max_norm=1.0)
                opt.step()
                last_parts = parts

        refined_pose = pose.detach()
        final_energy, final_parts = self.compute_energy(
            refined_pose,
            init_pose,
            object_pc,
            sdf_grid,
            sdf_origin,
            sdf_voxel_size,
            T_cam_to_obj,
        )

        trans_delta = torch.norm(refined_pose[:, :3] - init_pose[:, :3], dim=-1)
        q_delta = torch.norm(refined_pose[:, 9:] - init_pose[:, 9:], dim=-1)
        energy_improve = init_energy.detach() - final_energy.detach()
        init_E_pen = init_parts["E_pen"]
        final_E_pen = final_parts["E_pen"]

        if cfg.use_conservative_accept:
            use_refined = (
                (init_E_pen > cfg.pen_threshold)
                & (final_E_pen < init_E_pen)
                & (energy_improve > cfg.min_energy_improve)
                & (trans_delta < cfg.max_trans_delta)
                & (q_delta < cfg.max_q_delta)
            )
        else:
            use_refined = torch.ones_like(energy_improve, dtype=torch.bool)

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
            "init_E_pen": init_E_pen.detach(),
            "final_E_pen": final_E_pen.detach(),
            "trans_delta": trans_delta.detach(),
            "q_delta": q_delta.detach(),
        }
        out.update({f"init_{k}": v for k, v in init_parts.items()})
        out.update({f"final_{k}": v for k, v in final_parts.items()})
        return out


class PhysicsLossForDiffusionTraining(torch.nn.Module):
    """可选：在 diffusion 训练时对 predicted pose 加 physics loss。"""

    def __init__(self, refiner: PhysicsGuidedPoseRefiner, weight: float = 0.1):
        super().__init__()
        self.refiner = refiner
        self.weight = weight

    def forward(
        self,
        pred_pose: torch.Tensor,
        init_pose: torch.Tensor,
        object_pc: torch.Tensor,
        sdf_grid: torch.Tensor,
        sdf_origin: torch.Tensor,
        sdf_voxel_size: torch.Tensor,
        T_cam_to_obj: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        energy, parts = self.refiner.compute_energy(
            pred_pose,
            init_pose,
            object_pc,
            sdf_grid,
            sdf_origin,
            sdf_voxel_size,
            T_cam_to_obj,
        )
        loss = self.weight * energy.mean()
        return loss, parts
