"""
Contact GT v2 (IBS-Lite)：从 IBS contact 体素区域为每指分配代表接触点（table 系）。

与 v1 最近 surface 点相比：
  - contact 定义对齐 CADGrasp IBS（手–物双近 + bisector 区域）
  - 每指在 IBS contact 点云中取最近点，而非 object_pc 最近点
  - 无 IBS contact 或距离过大时 fallback 到 v1 nearest surface

预计算 cache 由 precompute_contact_gt_ibs_cache.py 生成；
训练时 table→camera 变换与 GraspNetDataset 一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from utils.contact_gt_surface import (
    aggregate_fingertips_per_link,
    compute_gt_contact_points,
    infer_num_fingertips_from_hand,
    meta_id_to_object_code,
)
from ibs.utils.ibs_repr import IBS
from ibs.utils.transforms import transform_points


CONTACT_SOURCE_IBS = 0
CONTACT_SOURCE_NEAREST = 1


@dataclass
class IBSLiteContactConfig:
    max_assign_dist: float = 0.025
    contact_thresh: float = 0.015
    cdist_chunk: int = 2048
    n_points_each_link: int = 64


def table_to_camera_points(
    pts_table: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """与 GraspNetDatasetContactDiffusion 一致：table → camera。"""
    r = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    return (pts_table - t) @ r


def camera_to_table_points(
    pts_camera: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    r = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    return pts_camera @ r.T + t


def ibs_contact_points_hand(voxel: np.ndarray, w2h: np.ndarray) -> np.ndarray:
    """IBS contact 点云，hand 坐标系，shape (M, 3)。"""
    return ibs_contact_points_hand_fast(voxel)


def ibs_contact_points_hand_fast(voxel: np.ndarray) -> np.ndarray:
    """直接从 contact 通道取点，避免 IBS 类开销。w2h 未使用（点在 hand 系）。"""
    contact_mask = voxel[..., 1].astype(bool)
    if not contact_mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    bound = 0.1
    resolution = 0.005
    indices = np.argwhere(contact_mask)
    origin = np.array(
        [-bound + resolution / 2, -bound + resolution / 2, -bound + resolution / 2],
        dtype=np.float32,
    )
    return indices.astype(np.float32) * resolution + origin


def assign_ibs_contacts_per_finger_batch(
    finger_tips_hand: np.ndarray,
    contact_hand: np.ndarray,
    max_assign_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """finger_tips_hand (B,F,3), contact_hand 列表长度 B，每元素 (M,3)。"""
    b, num_fingers, _ = finger_tips_hand.shape
    out = np.zeros((b, num_fingers, 3), dtype=np.float32)
    mask = np.zeros((b, num_fingers), dtype=np.float32)
    for i in range(b):
        sel, mk = assign_ibs_contacts_per_finger_numpy(
            finger_tips_hand[i], contact_hand[i], max_assign_dist
        )
        out[i] = sel
        mask[i] = mk
    return out, mask


def _finalize_ibs_lite_chunk(
    finger_tips_table: np.ndarray,
    w2h_batch: np.ndarray,
    ibs_voxels: np.ndarray,
    cfg: IBSLiteContactConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    finger_tips_table: (B,F,3)
    Returns contact_table (B,F,3), mask (B,F), source (B,)
    """
    b, num_fingers, _ = finger_tips_table.shape
    contact_table = np.zeros((b, num_fingers, 3), dtype=np.float32)
    mask = np.zeros((b, num_fingers), dtype=np.float32)
    source = np.full(b, CONTACT_SOURCE_NEAREST, dtype=np.uint8)

    contact_hand_list = [ibs_contact_points_hand_fast(ibs_voxels[i]) for i in range(b)]
    finger_hand = np.zeros_like(finger_tips_table)
    for i in range(b):
        finger_hand[i] = transform_points(finger_tips_table[i], w2h_batch[i].astype(np.float64))

    sel_hand, mk = assign_ibs_contacts_per_finger_batch(
        finger_hand, contact_hand_list, cfg.max_assign_dist
    )
    for i in range(b):
        h2t = np.linalg.inv(w2h_batch[i].astype(np.float64)).astype(np.float32)
        contact_table[i] = transform_points(sel_hand[i], h2t)
        mask[i] = mk[i]
        if mk[i].sum() >= 1:
            source[i] = CONTACT_SOURCE_IBS
    return contact_table, mask, source


def compute_ibs_lite_contact_chunk(
    hand_model,
    trans_table: np.ndarray,
    rot_table: np.ndarray,
    qpos: np.ndarray,
    object_pc_table: np.ndarray,
    ibs_voxels: np.ndarray,
    w2h_batch: np.ndarray,
    cfg: Optional[IBSLiteContactConfig] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    批量 IBS-Lite（B 条 grasp）。FK batched；IBS 分配仍逐条 numpy。

    Returns:
        contact_table (B,F,3), mask (B,F), source (B,) uint8
    """
    cfg = cfg or IBSLiteContactConfig()
    dev = torch.device(device)
    b = trans_table.shape[0]
    num_fingers = infer_num_fingertips_from_hand(hand_model)

    trans_t = torch.tensor(trans_table, dtype=torch.float32, device=dev)
    rot_t = torch.tensor(rot_table, dtype=torch.float32, device=dev)
    qpos_t = torch.tensor(qpos, dtype=torch.float32, device=dev)
    obj_t = torch.tensor(object_pc_table, dtype=torch.float32, device=dev)
    obj_batch = obj_t.unsqueeze(0).expand(b, -1, -1)

    with torch.no_grad():
        raw_tips = _forward_fingertips(hand_model, trans_t, rot_t, qpos_t)
        finger_tips = aggregate_fingertips_per_link(
            raw_tips, num_fingers, object_pc=obj_batch, cdist_chunk=cfg.cdist_chunk
        )
    finger_table = finger_tips.detach().cpu().numpy()

    contact_table, mask, source = _finalize_ibs_lite_chunk(
        finger_table, w2h_batch, ibs_voxels, cfg
    )

    miss = source == CONTACT_SOURCE_NEAREST
    if miss.any():
        miss_idx = np.where(miss)[0]
        with torch.no_grad():
            fb_pts, _fb_mask, fb_dist = compute_gt_contact_points(
                hand_model,
                trans_t[miss_idx],
                rot_t[miss_idx],
                qpos_t[miss_idx],
                obj_batch[miss_idx],
                cdist_chunk=cfg.cdist_chunk,
                contact_thresh=cfg.contact_thresh,
                num_fingers=num_fingers,
            )
        fb_table = fb_pts.detach().cpu().numpy()
        fb_mask = (
            fb_dist.detach().cpu().numpy() < cfg.contact_thresh
        ).astype(np.float32)
        contact_table[miss_idx] = fb_table
        mask[miss_idx] = fb_mask
    return contact_table, mask, source


def _forward_fingertips(hand_model, trans, rot, qpos):
    if hasattr(hand_model, "forward_hand_points"):
        _, _, tips = hand_model.forward_hand_points(trans, rot, qpos)
    elif hasattr(hand_model, "_sdf_adam"):
        _, _, tips = hand_model._sdf_adam.forward_hand_points(trans, rot, qpos)
    else:
        raise AttributeError("hand_model must expose forward_hand_points")
    return tips


def assign_ibs_contacts_per_finger_numpy(
    finger_tips_hand: np.ndarray,
    contact_hand: np.ndarray,
    max_assign_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    finger_tips_hand: (F, 3), contact_hand: (M, 3)
    Returns contact_hand (F, 3), mask (F,)
    """
    num_fingers = finger_tips_hand.shape[0]
    out = np.zeros((num_fingers, 3), dtype=np.float32)
    mask = np.zeros(num_fingers, dtype=np.float32)
    if contact_hand.shape[0] == 0:
        return out, mask
    for f in range(num_fingers):
        d = np.linalg.norm(contact_hand - finger_tips_hand[f : f + 1], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= max_assign_dist:
            out[f] = contact_hand[j]
            mask[f] = 1.0
    return out, mask


def compute_ibs_lite_contact_single_numpy(
    hand_model,
    trans_table: np.ndarray,
    rot_table: np.ndarray,
    qpos: np.ndarray,
    object_pc_table: np.ndarray,
    ibs_voxel: np.ndarray,
    w2h: np.ndarray,
    cfg: Optional[IBSLiteContactConfig] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    单条 grasp 的 IBS-Lite contact（table 系）。

    Returns:
        contact_pts: (F, 3)
        contact_mask: (F,)
        contact_source: CONTACT_SOURCE_IBS or CONTACT_SOURCE_NEAREST
    """
    cfg = cfg or IBSLiteContactConfig()
    num_fingers = infer_num_fingertips_from_hand(hand_model)
    dev = torch.device(device)

    trans_t = torch.tensor(trans_table, dtype=torch.float32, device=dev).unsqueeze(0)
    rot_t = torch.tensor(rot_table, dtype=torch.float32, device=dev).unsqueeze(0)
    qpos_t = torch.tensor(qpos, dtype=torch.float32, device=dev).unsqueeze(0)
    obj_t = torch.tensor(object_pc_table, dtype=torch.float32, device=dev).unsqueeze(0)

    with torch.no_grad():
        raw_tips = _forward_fingertips(hand_model, trans_t, rot_t, qpos_t)
        finger_tips = aggregate_fingertips_per_link(
            raw_tips,
            num_fingers,
            object_pc=obj_t,
            cdist_chunk=cfg.cdist_chunk,
        )
    finger_table = finger_tips.squeeze(0).detach().cpu().numpy()
    finger_hand = transform_points(finger_table, w2h.astype(np.float64))

    contact_hand = ibs_contact_points_hand_fast(ibs_voxel)
    contact_hand_sel, mask = assign_ibs_contacts_per_finger_numpy(
        finger_hand,
        contact_hand,
        cfg.max_assign_dist,
    )

    h2t = np.linalg.inv(w2h.astype(np.float64)).astype(np.float32)
    contact_table = transform_points(contact_hand_sel, h2t)

    if float(mask.sum()) >= 1.0:
        return contact_table.astype(np.float32), mask.astype(np.float32), CONTACT_SOURCE_IBS

    with torch.no_grad():
        fb_pts, fb_mask, fb_dist = compute_gt_contact_points(
            hand_model,
            trans_t,
            rot_t,
            qpos_t,
            obj_t,
            cdist_chunk=cfg.cdist_chunk,
            contact_thresh=cfg.contact_thresh,
            num_fingers=num_fingers,
        )
    fb_table = fb_pts.squeeze(0).detach().cpu().numpy()
    fb_mask_np = (fb_dist.squeeze(0).detach().cpu().numpy() < cfg.contact_thresh).astype(
        np.float32
    )
    return fb_table.astype(np.float32), fb_mask_np, CONTACT_SOURCE_NEAREST


def compute_ibs_lite_contact_batch(
    hand_model,
    trans_table: torch.Tensor,
    rot_table: torch.Tensor,
    qpos: torch.Tensor,
    object_pc_table: torch.Tensor,
    ibs_voxels: np.ndarray,
    w2h_batch: np.ndarray,
    cfg: Optional[IBSLiteContactConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Batch IBS-Lite（逐条 numpy IBS，适合 precompute 小 batch）。

    Returns:
        contact_pts: (B, F, 3) table
        contact_mask: (B, F)
        contact_source: (B,) uint8
    """
    cfg = cfg or IBSLiteContactConfig()
    b = trans_table.shape[0]
    num_fingers = infer_num_fingertips_from_hand(hand_model)
    device = trans_table.device

    contacts = []
    masks = []
    sources = []
    for i in range(b):
        obj_pc = object_pc_table[i].detach().cpu().numpy()
        ct, mk, src = compute_ibs_lite_contact_single_numpy(
            hand_model,
            trans_table[i].detach().cpu().numpy(),
            rot_table[i].detach().cpu().numpy(),
            qpos[i].detach().cpu().numpy(),
            obj_pc,
            ibs_voxels[i],
            w2h_batch[i],
            cfg=cfg,
            device=device,
        )
        contacts.append(ct)
        masks.append(mk)
        sources.append(src)

    return (
        torch.tensor(np.stack(contacts), device=device, dtype=torch.float32),
        torch.tensor(np.stack(masks), device=device, dtype=torch.float32),
        np.array(sources, dtype=np.uint8),
    )


def flatten_contacts_v2(contact_pts: torch.Tensor) -> torch.Tensor:
    return contact_pts.reshape(contact_pts.shape[0], -1)


def table_to_camera_points_batch(
    pts_table: np.ndarray,
    camera_poses: np.ndarray,
) -> np.ndarray:
    """pts_table (N,F,3), camera_poses (N,4,4) → (N,F,3)。"""
    r = camera_poses[:, :3, :3]
    t = camera_poses[:, :3, 3]
    return (pts_table - t[:, None, :]) @ r


class ContactGTv2Cache:
    """加载 precompute 的 scene/object npz，O(1) 按 grasp index 查询。"""

    def __init__(self, cache_root: str, robot: str = "leap_hand"):
        import os

        self.cache_root = cache_root
        self.robot = robot
        self._loaded: dict[tuple[int, str], dict] = {}
        self._row_map: dict[tuple[int, str], dict[int, int]] = {}
        self.cached_scenes: set[int] = set()
        if os.path.isdir(cache_root):
            for name in os.listdir(cache_root):
                if name.startswith("scene_") and os.path.isdir(
                    os.path.join(cache_root, name, robot)
                ):
                    try:
                        self.cached_scenes.add(int(name.split("_")[-1]))
                    except ValueError:
                        pass

    def _path(self, scene_id: int, object_code: str) -> str:
        import os

        return os.path.join(
            self.cache_root,
            f"scene_{int(scene_id):04d}",
            self.robot,
            f"{object_code}.npz",
        )

    def _ensure_loaded(self, scene_id: int, object_code: str) -> bool:
        import os

        key = (int(scene_id), str(object_code).zfill(3))
        if key in self._loaded:
            return self._loaded[key] is not None
        path = self._path(key[0], key[1])
        if not os.path.isfile(path):
            self._loaded[key] = None  # type: ignore
            return False
        data = dict(np.load(path))
        self._loaded[key] = data
        gidx = data["grasp_indices"].astype(np.int64)
        self._row_map[key] = {int(g): int(i) for i, g in enumerate(gidx)}
        return True

    def lookup(
        self,
        scene_id: int,
        object_code: str,
        grasp_index: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        if int(scene_id) not in self.cached_scenes:
            return None
        code = str(object_code).zfill(3)
        key = (int(scene_id), code)
        if not self._ensure_loaded(key[0], code):
            return None
        row = self._row_map[key].get(int(grasp_index))
        if row is None:
            return None
        data = self._loaded[key]
        return (
            data["contact_pts"][row].astype(np.float32),
            data["contact_mask"][row].astype(np.float32),
            int(data["contact_source"][row]),
        )

    def lookup_batch_numpy(
        self,
        scene_ids: np.ndarray,
        meta_ids: np.ndarray,
        grasp_indices: np.ndarray,
        camera_poses: np.ndarray,
        num_fingers: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量 cache 查询 + table→camera。

        Returns:
            contact_cam (N,F,3), mask (N,F), hit (N,) bool
        """
        n = len(scene_ids)
        contact_table = np.zeros((n, num_fingers, 3), dtype=np.float32)
        mask = np.zeros((n, num_fingers), dtype=np.float32)
        hit = np.zeros(n, dtype=bool)

        for i in range(n):
            sc = int(scene_ids[i])
            if sc not in self.cached_scenes:
                continue
            code = meta_id_to_object_code(int(meta_ids[i]))
            key = (sc, code)
            if not self._ensure_loaded(sc, code):
                continue
            row = self._row_map[key].get(int(grasp_indices[i]))
            if row is None:
                continue
            data = self._loaded[key]
            f = min(num_fingers, data["contact_pts"].shape[1])
            contact_table[i, :f] = data["contact_pts"][row, :f]
            mask[i, :f] = data["contact_mask"][row, :f]
            hit[i] = True

        contact_cam = table_to_camera_points_batch(contact_table, camera_poses)
        return contact_cam, mask, hit

    def preload_all(self) -> int:
        """启动时预加载全部 cache 到内存（约 18MB），避免训练时反复 np.load。"""
        import glob
        import os

        n_files = 0
        pattern = os.path.join(
            self.cache_root, "scene_*", self.robot, "*.npz"
        )
        for path in glob.glob(pattern):
            parts = path.split(os.sep)
            scene_name = parts[-3]
            code = os.path.splitext(parts[-1])[0]
            try:
                sc = int(scene_name.split("_")[-1])
            except ValueError:
                continue
            if self._ensure_loaded(sc, code):
                n_files += 1
        return n_files

    def lookup_batch(
        self,
        scene_ids: torch.Tensor,
        meta_ids: torch.Tensor,
        grasp_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量查询；miss 的条目 contact_mask=0，由 caller fallback。

        Returns:
            contact_table (N, F, 3), mask (N, F), hit (N,), source (N,)
        """
        n = scene_ids.shape[0]
        num_fingers = 4
        contact = torch.zeros(n, num_fingers, 3, dtype=torch.float32)
        mask = torch.zeros(n, num_fingers, dtype=torch.float32)
        hit = torch.zeros(n, dtype=torch.bool)
        source = torch.full((n,), CONTACT_SOURCE_NEAREST, dtype=torch.long)

        for i in range(n):
            sc = int(scene_ids[i].item())
            code = meta_id_to_object_code(int(meta_ids[i].item()))
            gi = int(grasp_indices[i].item())
            row = self.lookup(sc, code, gi)
            if row is None:
                continue
            pts, mk, src = row
            f = pts.shape[0]
            contact[i, :f] = torch.from_numpy(pts)
            mask[i, :f] = torch.from_numpy(mk)
            hit[i] = True
            source[i] = int(src)
        return contact, mask, hit, source
