from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizer.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.robot_model import RobotModel, RobotPoints
from optimizer.mesh_sdf_offline import sample_sdf_grid


class SDFAdam(nn.Module):
    def __init__(
        self,
        hand_name,
        urdf_path,
        meta_path,
        parallel_num=10,
        device="cuda",
        contact_margin=0.006,
        penetration_margin=0.002,
        n_points_each_link=4,
    ):
        super().__init__()
        self.hand_name = hand_name
        self.device = torch.device(device)
        self.parallel_num = parallel_num
        self.contact_margin = contact_margin
        self.penetration_margin = penetration_margin
        # self.n_points_each_link = n_points_each_link
        self.n_points_each_link = max(n_points_each_link, 32)

        self.robot_model = RobotModel(urdf_path, meta_path)
        self.joint_names = self.robot_model.movable_joint_names
        self.num_joints = len(self.joint_names)

        digit_links = list(
            dict.fromkeys(
                list(self.robot_model.thumb_link_names)
                + list(self.robot_model.other_finger_link_names)
            )
        )
        self._fingertip_link_names = list(
            getattr(self.robot_model, "fingertip_link_names", [])
        )
        if not self._fingertip_link_names:
            self._fingertip_link_names = [ln for ln in digit_links if "fingertip" in ln]
        self._finger_shaft_link_names = [ln for ln in digit_links if ln not in self._fingertip_link_names]
        digit_set = set(digit_links)
        self._palm_link_names = [ln for ln in self.robot_model.link_names if ln not in digit_set]

        self.opt_q = None
        self.q_init = None
        self.object_pc = None
        self.energy = None
        self.optimizer = None
        self.scheduler = None

    def reset(
        self,
        init_trans,
        init_rot,
        init_qpos,
        object_pc,
        learning_rate=5e-3,
        lr_decay=0.5,
        decay_every=100,
        sdf_grid=None,
        sdf_origin=None,
        sdf_voxel_size=None,
        T_cam_to_obj=None,
    ):
        """
        init_trans: (G, 3)
        init_rot:   (G, 3, 3)
        init_qpos:  (G, J)
        object_pc:  (G, N, 3)
        """

        G = init_trans.shape[0]

        init_trans = init_trans.to(self.device)
        init_rot = init_rot.to(self.device)
        init_qpos = init_qpos.to(self.device)
        object_pc = object_pc.to(self.device)
        
        # 新加采样
        max_obj_points = 1024
        if object_pc.shape[1] > max_obj_points:
            idx = torch.randperm(object_pc.shape[1], device=object_pc.device)[:max_obj_points]
            object_pc = object_pc[:, idx]
        

        init_trans = init_trans[:, None].repeat(1, self.parallel_num, 1).reshape(-1, 3)
        init_rot = init_rot[:, None].repeat(1, self.parallel_num, 1, 1).reshape(-1, 3, 3)
        init_qpos = init_qpos[:, None].repeat(1, self.parallel_num, 1).reshape(-1, self.num_joints)
        object_pc = object_pc[:, None].repeat(1, self.parallel_num, 1, 1).reshape(-1, object_pc.shape[1], 3)

        rot6d = init_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        noise_trans = 0.005 * torch.randn_like(init_trans)
        noise_qpos = 0.03 * torch.randn_like(init_qpos)

        opt_init = torch.cat(
            [
                init_trans + noise_trans,
                rot6d,
                init_qpos + noise_qpos,
            ],
            dim=-1,
        )

        self.opt_q = nn.Parameter(opt_init)
        self.q_init = opt_init.detach().clone()
        self.object_pc = object_pc

        self.optimizer = torch.optim.Adam([self.opt_q], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_every,
            gamma=lr_decay,
        )
        
        if sdf_grid is not None:
            sdf_grid = sdf_grid.to(self.device)
            sdf_origin = sdf_origin.to(self.device)
            sdf_voxel_size = sdf_voxel_size.to(self.device)

            sdf_grid = sdf_grid[:, None].repeat(
                1, self.parallel_num, 1, 1, 1
            ).reshape(-1, sdf_grid.shape[1], sdf_grid.shape[2], sdf_grid.shape[3])

            sdf_origin = sdf_origin[:, None].repeat(
                1, self.parallel_num, 1
            ).reshape(-1, 3)

            sdf_voxel_size = sdf_voxel_size[:, None].repeat(
                1, self.parallel_num
            ).reshape(-1)

        self.sdf_grid = sdf_grid
        self.sdf_origin = sdf_origin
        self.sdf_voxel_size = sdf_voxel_size
        
        if T_cam_to_obj is not None:
            T_cam_to_obj = T_cam_to_obj.to(self.device)
            T_cam_to_obj = T_cam_to_obj[:, None].repeat(
                1, self.parallel_num, 1, 1
            ).reshape(-1, 4, 4)

        self.T_cam_to_obj = T_cam_to_obj
        
    def transform_points_cam_to_obj(self, pts):
        """
        pts: (B, M, 3), camera frame
        return: (B, M, 3), object canonical frame
        """
        if self.T_cam_to_obj is None:
            return pts

        R = self.T_cam_to_obj[:, :3, :3]
        t = self.T_cam_to_obj[:, :3, 3]

        return torch.einsum("bij,bmj->bmi", R, pts) + t[:, None, :]
        
    def signed_penetration_loss(self, pts):
        if self.sdf_grid is None:
            return self.penetration_proxy_loss(pts, self.object_pc)

        pts_obj = self.transform_points_cam_to_obj(pts)

        sdf = sample_sdf_grid(
            sdf_grid=self.sdf_grid,
            origin=self.sdf_origin,
            voxel_size=self.sdf_voxel_size,
            query_pts=pts_obj,
        )

        return F.relu(-sdf).mean(dim=1)

    def get_opt_q(self):
        return self.opt_q

    def parse_q(self):
        trans = self.opt_q[:, :3]
        rot6d = self.opt_q[:, 3:9]
        qpos = self.opt_q[:, 9:]

        rot = robust_compute_rotation_matrix_from_ortho6d(rot6d)
        return trans, rot, qpos

    def nearest_distance(self, query_pts, object_pc):
        """
        query_pts: (B, M, 3)
        object_pc: (B, N, 3)
        """
        dist = torch.cdist(query_pts, object_pc)
        min_dist, nn_idx = torch.min(dist, dim=-1)
        return min_dist, nn_idx

    def _qpos_tensor_to_dict(self, qpos: torch.Tensor) -> dict:
        qpos_dict = {
            name: qpos[:, i].float()
            for i, name in enumerate(self.robot_model.movable_joint_names)
        }
        batch_size = qpos.shape[0]
        for name in self.robot_model.joint_names:
            if name not in qpos_dict:
                qpos_dict[name] = torch.zeros(batch_size, device=qpos.device, dtype=qpos.dtype)
        return qpos_dict

    @staticmethod
    def _subsampling_fallback(whole: torch.Tensor, max_pts: int = 48) -> torch.Tensor:
        """当某一分组无采样点时，从 whole 中均匀取点，避免 cdist 空维度。"""
        b, n, _ = whole.shape
        if n == 0:
            raise RuntimeError("手部表面点为空，请检查 URDF / meta 与 sample_surface_points。")
        if n <= max_pts:
            return whole
        idx = torch.linspace(0, n - 1, max_pts, device=whole.device, dtype=torch.long)
        return whole[:, idx]

    def forward_hand_points(self, trans, rot, qpos):
        """
        使用 RobotModel：缓存各 link 局部表面点，用可微 FK + 全局位姿变到世界系。

        Returns:
            palm_pts:      (B, Mp, 3)
            finger_pts:    (B, Mf, 3)
            fingertip_pts: (B, Mt, 3)
        """
        device = qpos.device
        trans_f = trans.to(device=device, dtype=torch.float32)
        rot_f = rot.to(device=device, dtype=torch.float32)

        qpos_dict = self._qpos_tensor_to_dict(qpos)

        if self.robot_model.surface_points is None:
            self.robot_model.surface_points = self.robot_model.sample_surface_points(
                n_points_each_link=self.n_points_each_link
            )

        robot_points = RobotPoints(OrderedDict(self.robot_model.surface_points))

        link_translations, link_rotations = self.robot_model.forward_kinematics(qpos_dict)
        link_translations = {
            k: torch.einsum("nab,nb->na", rot_f, v) + trans_f for k, v in link_translations.items()
        }
        link_rotations = {k: torch.einsum("nab,nbc->nac", rot_f, v) for k, v in link_rotations.items()}

        def gather(link_list):
            if not link_list:
                return torch.zeros(trans_f.shape[0], 0, 3, device=device, dtype=torch.float32)
            return robot_points.get_points(
                local_translations=link_translations,
                local_rotations=link_rotations,
                robot_frame=True,
                link_names=link_list,
            )

        palm_pts = gather(self._palm_link_names)
        finger_pts = gather(self._finger_shaft_link_names)
        fingertip_pts = gather(self._fingertip_link_names)
        whole_pts = gather(list(self.robot_model.link_names))

        if palm_pts.shape[1] == 0:
            palm_pts = self._subsampling_fallback(whole_pts)
        if finger_pts.shape[1] == 0:
            finger_pts = self._subsampling_fallback(whole_pts)
        if fingertip_pts.shape[1] == 0:
            fingertip_pts = self._subsampling_fallback(whole_pts)

        return palm_pts, finger_pts, fingertip_pts

    def contact_loss(self, pts, object_pc, topk_ratio=0.2):
        dist, _ = self.nearest_distance(pts, object_pc)  # (B, M)
        k = max(1, int(dist.shape[1] * topk_ratio))
        near_dist, _ = torch.topk(dist, k=k, dim=1, largest=False)
        return F.relu(near_dist - self.contact_margin).mean(dim=1)  # (B,)

    def penetration_proxy_loss(self, pts, object_pc):
        dist, _ = self.nearest_distance(pts, object_pc)
        return F.relu(self.penetration_margin - dist).mean(dim=1)  # (B,)

    def joint_regularization_loss(self, qpos):
        qpos_init = self.q_init[:, 9:]
        return ((qpos - qpos_init) ** 2).mean(dim=1)  # (B,)


    def pose_regularization_loss(self):
        return ((self.opt_q[:, :9] - self.q_init[:, :9]) ** 2).mean(dim=1)  # (B,)
    
    # 新加的函数
    def compute_energy_only(self):
        trans, rot, qpos = self.parse_q()

        palm_pts, finger_pts, fingertip_pts = self.forward_hand_points(
            trans, rot, qpos
        )

        # =========================
        # Contact losses
        # =========================
        E_palm_contact = self.contact_loss(palm_pts, self.object_pc)
        E_finger_contact = self.contact_loss(finger_pts, self.object_pc)
        E_tip_contact = self.contact_loss(fingertip_pts, self.object_pc)

        # =========================
        # Penetration losses
        # =========================
        E_palm_pen = self.signed_penetration_loss(palm_pts)
        E_finger_pen = self.signed_penetration_loss(finger_pts)
        E_tip_pen = self.signed_penetration_loss(fingertip_pts)

        # =========================
        # Regularization
        # =========================
        E_joint = self.joint_regularization_loss(qpos)
        E_pose = self.pose_regularization_loss()

        E_contact = (
            0.2 * E_palm_contact
            + 0.5 * E_finger_contact
            + 1.0 * E_tip_contact
        )

        E_pen = (
            0.2 * E_palm_pen
            + 0.5 * E_finger_pen
            + 1.0 * E_tip_pen
        )

        energy = (
            1.0 * E_contact
            + 5.0 * E_pen
            + 0.05 * E_joint
            + 0.05 * E_pose
        )

        return {
            "energy": energy,
            "E_pen": E_pen,
            "E_contact": E_contact,
        }

    def step(self, energy_dict=None):
        self.optimizer.zero_grad()

        trans, rot, qpos = self.parse_q()
        palm_pts, finger_pts, fingertip_pts = self.forward_hand_points(trans, rot, qpos)

        E_palm_contact = self.contact_loss(palm_pts, self.object_pc)
        E_finger_contact = self.contact_loss(finger_pts, self.object_pc)
        E_tip_contact = self.contact_loss(fingertip_pts, self.object_pc)

        E_palm_pen = self.signed_penetration_loss(palm_pts)
        E_finger_pen = self.signed_penetration_loss(finger_pts)
        E_tip_pen = self.signed_penetration_loss(fingertip_pts)

        E_joint = self.joint_regularization_loss(qpos)
        E_pose = self.pose_regularization_loss()

        E_contact = (
            0.2 * E_palm_contact
            + 0.5 * E_finger_contact
            + 1.0 * E_tip_contact
        )

        E_pen = (
            0.2 * E_palm_pen
            + 0.5 * E_finger_pen
            + 1.0 * E_tip_pen
        )

        energy = (
            1.0 * E_contact
            + 5.0 * E_pen
            + 0.05 * E_joint
            + 0.05 * E_pose
        )  # (B,)

        loss = energy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.opt_q], max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.energy = energy.detach()

        if energy_dict is not None:
            energy_dict["E_contact"].append(E_contact.detach())
            energy_dict["E_pen"].append(E_pen.detach())
            energy_dict["E_joint"].append(E_joint.detach())
            energy_dict["E_pose"].append(E_pose.detach())

        return energy
