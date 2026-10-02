"""
Contact GT v2b (IBS-Lite + thumb split)：在 v2 基础上对齐原 IBS 分配策略。

与 contact_gt_ibs 的差异：
  - 拇指用 channel-2 thumb_contact 点池 + thumb_contact_delta (8.5mm)
  - 其余三指用 (contact & ~thumb) 点池 + contact_delta (7.5mm)
  - IBS 分配只决定每指候选区域，最终 GT contact 投影到 object surface
  - ContactGTv2bCache.lookup_batch_numpy 支持 align_mat @ camera_pose（network camera 系）

预计算：precompute_contact_gt_ibs_thumb_split_cache.py
训练：train_contact_diffusion_base.py（contact 初始化）、train_contact_stability_base.py（v16c 直接加载）

适配train_contact_diffusion_base.py
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
from utils.contact_gt_ibs import (
    CONTACT_SOURCE_IBS,
    CONTACT_SOURCE_NEAREST,
    ContactGTv2Cache,
    _forward_fingertips,
    table_to_camera_points_batch,
)
from ibs.utils.transforms import transform_points


@dataclass
class IBSLiteContactConfigV2b:
    """分配阈值对齐 IBSConfig.contact_delta / thumb_contact_delta。"""
    contact_delta: float = 0.0075
    thumb_contact_delta: float = 0.0085
    thumb_finger_index: int = 0
    project_to_object_surface: bool = True
    contact_thresh: float = 0.015
    cdist_chunk: int = 2048
    n_points_each_link: int = 64
    region_max_points: int = 64
    region_surface_radius: float = 0.015
    region_score_sigma: float = 0.006


def table_to_network_camera_points(
    pts_table: np.ndarray,
    align_mat: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """table 系 → network_input 点云同款 camera 系：extrinsics = align_mat @ camera_pose。"""
    extr = align_mat @ camera_pose
    return (pts_table.astype(np.float64) - extr[:3, 3]) @ extr[:3, :3]


def table_to_network_camera_points_batch(
    pts_table: np.ndarray,
    align_mats: np.ndarray,
    camera_poses: np.ndarray,
) -> np.ndarray:
    """pts_table (N,F,3) → network camera 系。"""
    extr = np.matmul(align_mats, camera_poses)
    r = extr[:, :3, :3]
    t = extr[:, :3, 3]
    return (pts_table - t[:, None, :]) @ r


def _voxel_mask_to_points_hand(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    bound = 0.1
    resolution = 0.005
    indices = np.argwhere(mask)
    origin = np.array(
        [-bound + resolution / 2, -bound + resolution / 2, -bound + resolution / 2],
        dtype=np.float32,
    )
    return indices.astype(np.float32) * resolution + origin


def project_contacts_to_object_surface_numpy(
    contact_table: np.ndarray,
    contact_mask: np.ndarray,
    object_pc_table: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """
    将 IBS 分配出的 table 系候选 contact 投影到目标物体 surface 最近点。

    IBS contact voxel 在 bisector/接触区域附近，不保证落在物体表面；contact
    diffusion 的监督目标应是物体表面点，后续 fingertip anchor 才有明确几何意义。
    """
    out = contact_table.astype(np.float32, copy=True)
    valid_obj = object_pc_table.astype(np.float32)
    valid_obj = valid_obj[np.isfinite(valid_obj).all(axis=1)]
    if valid_obj.shape[0] == 0:
        return out

    valid_rows = np.argwhere(contact_mask > 0.5)
    if valid_rows.shape[0] == 0:
        return out

    pts = out[valid_rows[:, 0], valid_rows[:, 1]]
    nearest = np.zeros_like(pts, dtype=np.float32)
    min_d2 = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, valid_obj.shape[0], chunk_size):
        obj = valid_obj[start : start + chunk_size]
        d2 = ((pts[:, None, :] - obj[None, :, :]) ** 2).sum(axis=-1)
        idx = d2.argmin(axis=1)
        val = d2[np.arange(d2.shape[0]), idx]
        upd = val < min_d2
        if np.any(upd):
            min_d2[upd] = val[upd]
            nearest[upd] = obj[idx[upd]]

    out[valid_rows[:, 0], valid_rows[:, 1]] = nearest
    return out


def ibs_contact_points_hand_pools(
    voxel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    仿原 IBS 分拇指 / 非拇指点池（hand 系）。
    Returns:
        thumb_contact_hand, other_contact_hand  (other = contact & ~thumb)
    """
    occupancy = voxel[..., 0].astype(bool)
    contact = voxel[..., 1].astype(bool) & occupancy
    thumb = voxel[..., 2].astype(bool) & occupancy
    other = contact & ~thumb
    return _voxel_mask_to_points_hand(thumb), _voxel_mask_to_points_hand(other)


def assign_ibs_contacts_thumb_split_numpy(
    finger_tips_hand: np.ndarray,
    thumb_contact_hand: np.ndarray,
    other_contact_hand: np.ndarray,
    cfg: IBSLiteContactConfigV2b,
) -> Tuple[np.ndarray, np.ndarray]:
    num_fingers = finger_tips_hand.shape[0]
    out = np.zeros((num_fingers, 3), dtype=np.float32)
    mask = np.zeros(num_fingers, dtype=np.float32)
    thumb_idx = int(cfg.thumb_finger_index)

    if thumb_contact_hand.shape[0] > 0 and 0 <= thumb_idx < num_fingers:
        d = np.linalg.norm(
            thumb_contact_hand - finger_tips_hand[thumb_idx : thumb_idx + 1], axis=1
        )
        j = int(np.argmin(d))
        if float(d[j]) <= cfg.thumb_contact_delta:
            out[thumb_idx] = thumb_contact_hand[j]
            mask[thumb_idx] = 1.0

    for f in range(num_fingers):
        if f == thumb_idx or other_contact_hand.shape[0] == 0:
            continue
        d = np.linalg.norm(other_contact_hand - finger_tips_hand[f : f + 1], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= cfg.contact_delta:
            out[f] = other_contact_hand[j]
            mask[f] = 1.0
    return out, mask


def assign_ibs_contact_regions_thumb_split_numpy(
    finger_tips_hand: np.ndarray,
    thumb_contact_hand: np.ndarray,
    other_contact_hand: np.ndarray,
    cfg: IBSLiteContactConfigV2b,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign IBS contact *regions* to each finger.

    Instead of selecting a single nearest IBS contact point per finger, keep the
    IBS contact-pool points within that finger's assignment threshold. The
    caller projects these points to object surface and stores the resulting
    surface region.
    """
    num_fingers = finger_tips_hand.shape[0]
    max_k = max(int(cfg.region_max_points), 1)
    region = np.zeros((num_fingers, max_k, 3), dtype=np.float32)
    mask = np.zeros((num_fingers, max_k), dtype=np.float32)
    score = np.zeros((num_fingers, max_k), dtype=np.float32)
    thumb_idx = int(cfg.thumb_finger_index)

    for f in range(num_fingers):
        is_thumb = f == thumb_idx
        pool = thumb_contact_hand if is_thumb else other_contact_hand
        thresh = cfg.thumb_contact_delta if is_thumb else cfg.contact_delta
        if pool.shape[0] == 0:
            continue
        d = np.linalg.norm(pool - finger_tips_hand[f : f + 1], axis=1)
        cand = np.where(d <= float(thresh))[0]
        if cand.shape[0] == 0:
            continue
        order = cand[np.argsort(d[cand])[:max_k]]
        k = int(order.shape[0])
        sigma = max(float(cfg.region_score_sigma), 1e-6)
        region[f, :k] = pool[order].astype(np.float32)
        mask[f, :k] = 1.0
        score[f, :k] = np.exp(-(d[order].astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    return region, mask, score


def assign_ibs_contacts_thumb_split_batch(
    finger_tips_hand: np.ndarray,
    thumb_pools: List[np.ndarray],
    other_pools: List[np.ndarray],
    cfg: IBSLiteContactConfigV2b,
) -> Tuple[np.ndarray, np.ndarray]:
    b = finger_tips_hand.shape[0]
    out = np.zeros((b, finger_tips_hand.shape[1], 3), dtype=np.float32)
    mask = np.zeros((b, finger_tips_hand.shape[1]), dtype=np.float32)
    for i in range(b):
        sel, mk = assign_ibs_contacts_thumb_split_numpy(
            finger_tips_hand[i], thumb_pools[i], other_pools[i], cfg
        )
        out[i] = sel
        mask[i] = mk
    return out, mask


def _finalize_ibs_lite_chunk_v2b(
    finger_tips_table: np.ndarray,
    w2h_batch: np.ndarray,
    ibs_voxels: np.ndarray,
    cfg: IBSLiteContactConfigV2b,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, num_fingers, _ = finger_tips_table.shape
    contact_table = np.zeros((b, num_fingers, 3), dtype=np.float32)
    mask = np.zeros((b, num_fingers), dtype=np.float32)
    source = np.full(b, CONTACT_SOURCE_NEAREST, dtype=np.uint8)

    thumb_pools = []
    other_pools = []
    for i in range(b):
        thumb_pts, other_pts = ibs_contact_points_hand_pools(ibs_voxels[i])
        thumb_pools.append(thumb_pts)
        other_pools.append(other_pts)

    finger_hand = np.zeros_like(finger_tips_table)
    for i in range(b):
        finger_hand[i] = transform_points(finger_tips_table[i], w2h_batch[i].astype(np.float64))

    sel_hand, mk = assign_ibs_contacts_thumb_split_batch(
        finger_hand, thumb_pools, other_pools, cfg
    )
    for i in range(b):
        h2t = np.linalg.inv(w2h_batch[i].astype(np.float64)).astype(np.float32)
        contact_table[i] = transform_points(sel_hand[i], h2t)
        mask[i] = mk[i]
        if mk[i].sum() >= 1:
            source[i] = CONTACT_SOURCE_IBS
    return contact_table, mask, source


def project_regions_to_object_surface_numpy(
    region_table: np.ndarray,
    region_mask: np.ndarray,
    object_pc_table: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Project region points to nearest object-surface points."""
    out = region_table.astype(np.float32, copy=True)
    valid_obj = object_pc_table.astype(np.float32)
    valid_obj = valid_obj[np.isfinite(valid_obj).all(axis=1)]
    if valid_obj.shape[0] == 0:
        return out
    valid_rows = np.argwhere(region_mask > 0.5)
    if valid_rows.shape[0] == 0:
        return out

    pts = out[valid_rows[:, 0], valid_rows[:, 1], valid_rows[:, 2]]
    nearest = np.zeros_like(pts, dtype=np.float32)
    min_d2 = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, valid_obj.shape[0], chunk_size):
        obj = valid_obj[start : start + chunk_size]
        d2 = ((pts[:, None, :] - obj[None, :, :]) ** 2).sum(axis=-1)
        idx = d2.argmin(axis=1)
        val = d2[np.arange(d2.shape[0]), idx]
        upd = val < min_d2
        if np.any(upd):
            min_d2[upd] = val[upd]
            nearest[upd] = obj[idx[upd]]

    out[valid_rows[:, 0], valid_rows[:, 1], valid_rows[:, 2]] = nearest
    return out


def expand_contact_centers_to_object_regions_numpy(
    contact_table: np.ndarray,
    contact_mask: np.ndarray,
    object_pc_table: np.ndarray,
    max_points: int = 64,
    radius: float = 0.015,
    score_sigma: float = 0.006,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand per-finger contact centers into object-surface regions.

    The centers are expected to be in table coordinates and already projected
    to the object surface. For each valid finger center, keep up to max_points
    object surface points within radius, sorted by distance to the center.
    """
    b, num_fingers, _ = contact_table.shape
    max_k = max(int(max_points), 1)
    region = np.zeros((b, num_fingers, max_k, 3), dtype=np.float32)
    mask = np.zeros((b, num_fingers, max_k), dtype=np.float32)
    score = np.zeros((b, num_fingers, max_k), dtype=np.float32)

    obj = object_pc_table.astype(np.float32)
    obj = obj[np.isfinite(obj).all(axis=1)]
    if obj.shape[0] == 0:
        return region, mask, score

    radius = max(float(radius), 1e-6)
    sigma = max(float(score_sigma), 1e-6)
    for i in range(b):
        for f in range(num_fingers):
            if contact_mask[i, f] <= 0.5:
                continue
            center = contact_table[i, f].astype(np.float32)
            if not np.isfinite(center).all():
                continue
            d = np.linalg.norm(obj - center[None, :], axis=1)
            cand = np.where(d <= radius)[0]
            if cand.shape[0] == 0:
                cand = np.array([int(np.argmin(d))], dtype=np.int64)
            order = cand[np.argsort(d[cand])[:max_k]]
            k = int(order.shape[0])
            region[i, f, :k] = obj[order]
            mask[i, f, :k] = 1.0
            score[i, f, :k] = np.exp(-(d[order].astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    return region, mask, score


def compute_ibs_lite_contact_region_chunk_v2b(
    hand_model,
    trans_table: np.ndarray,
    rot_table: np.ndarray,
    qpos: np.ndarray,
    object_pc_table: np.ndarray,
    ibs_voxels: np.ndarray,
    w2h_batch: np.ndarray,
    cfg: Optional[IBSLiteContactConfigV2b] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or IBSLiteContactConfigV2b()
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

    max_k = max(int(cfg.region_max_points), 1)
    region_table = np.zeros((b, num_fingers, max_k, 3), dtype=np.float32)
    region_mask = np.zeros((b, num_fingers, max_k), dtype=np.float32)
    region_score = np.zeros((b, num_fingers, max_k), dtype=np.float32)

    contact_table, contact_mask, _source = _finalize_ibs_lite_chunk_v2b(
        finger_table, w2h_batch, ibs_voxels, cfg
    )
    if cfg.project_to_object_surface:
        contact_table = project_contacts_to_object_surface_numpy(
            contact_table,
            contact_mask,
            object_pc_table,
            chunk_size=cfg.cdist_chunk,
        )
    region_table, region_mask, region_score = expand_contact_centers_to_object_regions_numpy(
        contact_table,
        contact_mask,
        object_pc_table,
        max_points=max_k,
        radius=cfg.region_surface_radius,
        score_sigma=cfg.region_score_sigma,
    )
    return region_table, region_mask, region_score


def compute_ibs_lite_contact_chunk_v2b(
    hand_model,
    trans_table: np.ndarray,
    rot_table: np.ndarray,
    qpos: np.ndarray,
    object_pc_table: np.ndarray,
    ibs_voxels: np.ndarray,
    w2h_batch: np.ndarray,
    cfg: Optional[IBSLiteContactConfigV2b] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or IBSLiteContactConfigV2b()
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

    contact_table, mask, source = _finalize_ibs_lite_chunk_v2b(
        finger_table, w2h_batch, ibs_voxels, cfg
    )
    if cfg.project_to_object_surface:
        contact_table = project_contacts_to_object_surface_numpy(
            contact_table,
            mask,
            object_pc_table,
            chunk_size=cfg.cdist_chunk,
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


def compute_ibs_lite_contact_single_numpy_v2b(
    hand_model,
    trans_table: np.ndarray,
    rot_table: np.ndarray,
    qpos: np.ndarray,
    object_pc_table: np.ndarray,
    ibs_voxel: np.ndarray,
    w2h: np.ndarray,
    cfg: Optional[IBSLiteContactConfigV2b] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray, int]:
    cfg = cfg or IBSLiteContactConfigV2b()
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

    thumb_pool, other_pool = ibs_contact_points_hand_pools(ibs_voxel)
    contact_hand_sel, mask = assign_ibs_contacts_thumb_split_numpy(
        finger_hand, thumb_pool, other_pool, cfg
    )

    h2t = np.linalg.inv(w2h.astype(np.float64)).astype(np.float32)
    contact_table = transform_points(contact_hand_sel, h2t)

    if float(mask.sum()) >= 1.0:
        if cfg.project_to_object_surface:
            contact_table = project_contacts_to_object_surface_numpy(
                contact_table[None],
                mask[None],
                object_pc_table,
                chunk_size=cfg.cdist_chunk,
            )[0]
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


class ContactGTv2bCache(ContactGTv2Cache):
    """v2b cache 查询：默认使用 align_mat @ camera_pose 变换到 network camera 系。"""

    def lookup_batch_numpy(
        self,
        scene_ids: np.ndarray,
        meta_ids: np.ndarray,
        grasp_indices: np.ndarray,
        camera_poses: np.ndarray,
        num_fingers: int = 4,
        align_mats: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        if align_mats is not None:
            contact_cam = table_to_network_camera_points_batch(
                contact_table, align_mats, camera_poses
            )
        else:
            contact_cam = table_to_camera_points_batch(contact_table, camera_poses)
        return contact_cam, mask, hit

    def lookup_region_batch_numpy(
        self,
        scene_ids: np.ndarray,
        meta_ids: np.ndarray,
        grasp_indices: np.ndarray,
        camera_poses: np.ndarray,
        num_fingers: int = 4,
        align_mats: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(scene_ids)
        max_k = 1
        loaded_rows = []
        hit = np.zeros(n, dtype=bool)

        for i in range(n):
            sc = int(scene_ids[i])
            if sc not in self.cached_scenes:
                loaded_rows.append(None)
                continue
            code = meta_id_to_object_code(int(meta_ids[i]))
            key = (sc, code)
            if not self._ensure_loaded(sc, code):
                loaded_rows.append(None)
                continue
            data = self._loaded[key]
            if "region_pts" not in data or "region_mask" not in data:
                loaded_rows.append(None)
                continue
            row = self._row_map[key].get(int(grasp_indices[i]))
            if row is None:
                loaded_rows.append(None)
                continue
            max_k = max(max_k, int(data["region_pts"].shape[2]))
            loaded_rows.append((data, row))
            hit[i] = True

        region_table = np.zeros((n, num_fingers, max_k, 3), dtype=np.float32)
        region_mask = np.zeros((n, num_fingers, max_k), dtype=np.float32)
        region_score = np.zeros((n, num_fingers, max_k), dtype=np.float32)

        for i, item in enumerate(loaded_rows):
            if item is None:
                continue
            data, row = item
            f = min(num_fingers, data["region_pts"].shape[1])
            k = min(max_k, data["region_pts"].shape[2])
            region_table[i, :f, :k] = data["region_pts"][row, :f, :k]
            region_mask[i, :f, :k] = data["region_mask"][row, :f, :k]
            if "region_score" in data:
                region_score[i, :f, :k] = data["region_score"][row, :f, :k]
            else:
                region_score[i, :f, :k] = region_mask[i, :f, :k]

        flat_table = region_table.reshape(n, num_fingers * max_k, 3)
        if align_mats is not None:
            flat_cam = table_to_network_camera_points_batch(flat_table, align_mats, camera_poses)
        else:
            flat_cam = table_to_camera_points_batch(flat_table, camera_poses)
        region_cam = flat_cam.reshape(n, num_fingers, max_k, 3)
        return region_cam, region_mask, region_score, hit


def apply_aug_rotmat_to_ib_cache_contacts(
    contact_pts: torch.Tensor,
    cache_hit: torch.Tensor,
    aug_rotmat: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    cache GT 在 augment 前变换到 camera 系；需与 point_clouds/trans 同步施加 Z 旋转。
    v1 fallback 已用增强后的 pose 计算，无需再旋转。
    """
    if aug_rotmat is None or not bool(cache_hit.any()):
        return contact_pts
    out = contact_pts.clone()
    hit = cache_hit
    out[hit] = torch.einsum("bij,bfj->bfi", aug_rotmat[hit], contact_pts[hit])
    return out


def apply_aug_rotmat_to_ib_cache_regions(
    region_pts: torch.Tensor,
    cache_hit: torch.Tensor,
    aug_rotmat: Optional[torch.Tensor],
) -> torch.Tensor:
    if aug_rotmat is None or not bool(cache_hit.any()):
        return region_pts
    out = region_pts.clone()
    hit = cache_hit
    out[hit] = torch.einsum("bij,bfkj->bfki", aug_rotmat[hit], region_pts[hit])
    return out
