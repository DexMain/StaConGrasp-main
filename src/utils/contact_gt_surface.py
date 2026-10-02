"""
Contact Diffusion GT：dex_grasps FK → 目标物体 surface 最近点（camera 系）。

默认与 GraspNet 编号约定一致：object code C → meta/seg id = int(C) + 1。
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

GRASPNET_META_ID_OFFSET = 1
DEFAULT_NUM_FINGERTIPS = 4


def infer_num_fingertips_from_hand(hand_model) -> int:
    """LEAP 手：4 个 fingertip link，而非 surface 采样点总数。"""
    sdf = getattr(hand_model, "_sdf_adam", hand_model)
    names = getattr(sdf, "_fingertip_link_names", None)
    if names:
        return len(names)
    return DEFAULT_NUM_FINGERTIPS


def _min_dist_to_object_pc(
    hand_pts: torch.Tensor,
    object_pc: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """hand_pts (B, H, 3) → (B, H) min distance to object_pc."""
    b, h, _ = hand_pts.shape
    min_d = torch.full(
        (b, h), float("inf"), device=hand_pts.device, dtype=hand_pts.dtype
    )
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, object_pc.shape[1], chunk_size):
        end = min(start + chunk_size, object_pc.shape[1])
        dists = torch.cdist(hand_pts, object_pc[:, start:end])
        min_d = torch.minimum(min_d, dists.min(dim=-1).values)
    return min_d


def aggregate_fingertips_per_link(
    fingertip_pts: torch.Tensor,
    num_fingers: int,
    object_pc: Optional[torch.Tensor] = None,
    cdist_chunk: int = 2048,
) -> torch.Tensor:
    """
    将 (B, N, 3) 指尖 surface 点聚合为 (B, num_fingers, 3)。
    每 link 取距 object 最近点；无 object 时用 link 质心。
    """
    b, n_pts, _ = fingertip_pts.shape
    if n_pts == num_fingers:
        return fingertip_pts
    if n_pts < num_fingers:
        pad = fingertip_pts[:, -1:].expand(b, num_fingers - n_pts, 3)
        return torch.cat([fingertip_pts, pad], dim=1)

    pts_per_link = n_pts // num_fingers
    if pts_per_link * num_fingers != n_pts:
        idx = torch.linspace(
            0,
            n_pts - 1,
            num_fingers,
            device=fingertip_pts.device,
            dtype=torch.long,
        )
        return fingertip_pts[:, idx]

    grouped = fingertip_pts.view(b, num_fingers, pts_per_link, 3)
    if object_pc is None:
        return grouped.mean(dim=2)

    reps = []
    for i in range(num_fingers):
        link_pts = grouped[:, i]
        dists = _min_dist_to_object_pc(link_pts, object_pc, chunk_size=cdist_chunk)
        pick = dists.argmin(dim=-1)
        reps.append(link_pts[torch.arange(b, device=link_pts.device), pick])
    return torch.stack(reps, dim=1)


def object_code_to_meta_id(object_code: Union[str, int]) -> int:
    return int(object_code) + GRASPNET_META_ID_OFFSET


def meta_id_to_object_code(meta_id: int) -> str:
    return str(int(meta_id) - GRASPNET_META_ID_OFFSET).zfill(3)


def _forward_fingertips(hand_model, trans, rot, qpos):
    if hasattr(hand_model, "forward_hand_points"):
        _, _, fingertip_pts = hand_model.forward_hand_points(trans, rot, qpos)
    elif hasattr(hand_model, "_sdf_adam"):
        _, _, fingertip_pts = hand_model._sdf_adam.forward_hand_points(trans, rot, qpos)
    else:
        raise AttributeError("hand_model must expose forward_hand_points")
    return fingertip_pts


def nearest_surface_points(
    query_pts: torch.Tensor,
    object_pc: torch.Tensor,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    query_pts: (B, Q, 3), object_pc: (B, N, 3)
    返回 surface 上最近点 (B, Q, 3) 与距离 (B, Q)。
    """
    b, q, _ = query_pts.shape
    n_pc = object_pc.shape[1]
    nearest = torch.zeros(b, q, 3, device=query_pts.device, dtype=query_pts.dtype)
    min_dist = torch.full(
        (b, q), float("inf"), device=query_pts.device, dtype=query_pts.dtype
    )
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, n_pc, chunk_size):
        end = min(start + chunk_size, n_pc)
        dists = torch.cdist(query_pts, object_pc[:, start:end])
        vals, idxs = dists.min(dim=-1)
        update = vals < min_dist
        min_dist = torch.where(update, vals, min_dist)
        gather_idx = idxs.unsqueeze(-1).expand(-1, -1, 3)
        chunk_pts = object_pc[:, start:end]
        nearest_candidates = torch.gather(chunk_pts, 1, gather_idx)
        nearest = torch.where(update.unsqueeze(-1), nearest_candidates, nearest)
    return nearest, min_dist


def crop_object_pc_by_meta_id(
    point_cloud: torch.Tensor,
    seg: torch.Tensor,
    meta_id: int,
    max_points: int,
) -> torch.Tensor:
    """point_cloud (N,3), seg (N,) → (1, P, 3)。"""
    mask = seg == int(meta_id)
    if int(mask.sum()) >= 8:
        obj_pc = point_cloud[mask]
    else:
        obj_pc = point_cloud
    if obj_pc.shape[0] > max_points:
        idx = torch.randperm(obj_pc.shape[0], device=obj_pc.device)[:max_points]
        obj_pc = obj_pc[idx]
    elif obj_pc.shape[0] < max_points:
        pad_n = max_points - obj_pc.shape[0]
        pad = obj_pc[-1:].expand(pad_n, -1)
        obj_pc = torch.cat([obj_pc, pad], dim=0)
    return obj_pc.unsqueeze(0)


def build_object_pc_batch(
    point_clouds: torch.Tensor,
    seg: torch.Tensor,
    batch_indices: torch.Tensor,
    meta_ids: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    """为 batch 内每条 grasp 裁剪 object 点云，返回 (B, P, 3)。"""
    rows: List[torch.Tensor] = []
    for i in range(len(batch_indices)):
        b = int(batch_indices[i].item())
        meta_id = int(meta_ids[i].item())
        rows.append(
            crop_object_pc_by_meta_id(
                point_clouds[b], seg[b], meta_id, max_points
            ).squeeze(0)
        )
    return torch.stack(rows, dim=0)


def compute_gt_contact_points(
    hand_model,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    object_pc: torch.Tensor,
    cdist_chunk: int = 2048,
    contact_thresh: float = 0.015,
    num_fingers: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FK 每指代表点 → 物体 surface 最近点（GT contact targets）。

    Returns:
        contact_pts: (B, F, 3)  F=4 for LEAP
        contact_mask: (B, F) 1=有效接触
        tip_min_dist: (B, F)
    """
    num_fingers = num_fingers or infer_num_fingertips_from_hand(hand_model)
    raw_tips = _forward_fingertips(hand_model, trans, rot, qpos)
    finger_tips = aggregate_fingertips_per_link(
        raw_tips, num_fingers, object_pc=object_pc, cdist_chunk=cdist_chunk
    )
    contact_pts, tip_min_dist = nearest_surface_points(
        finger_tips, object_pc, chunk_size=cdist_chunk
    )
    contact_mask = (tip_min_dist < float(contact_thresh)).float()
    return contact_pts, contact_mask, tip_min_dist


def flatten_contacts(contact_pts: torch.Tensor) -> torch.Tensor:
    return contact_pts.reshape(contact_pts.shape[0], -1)


def unflatten_contacts(flat: torch.Tensor, num_fingertips: int) -> torch.Tensor:
    return flat.reshape(flat.shape[0], num_fingertips, 3)


def estimate_grasped_meta_id_numpy(
    tips_cam: np.ndarray,
    pc_cam: np.ndarray,
    seg_cam: np.ndarray,
) -> int | None:
    """指尖在点云上的最近邻 seg 多数票（与 vis_contact_stability 一致）。"""
    if tips_cam.size == 0:
        return None
    votes = []
    for tip in tips_cam:
        dist = np.linalg.norm(pc_cam - tip[None, :], axis=1)
        oid = int(seg_cam[int(np.argmin(dist))])
        if oid > 0:
            votes.append(oid)
    if not votes:
        return None
    vals, counts = np.unique(np.array(votes), return_counts=True)
    return int(vals[int(np.argmax(counts))])


@torch.no_grad()
def estimate_grasped_meta_ids_batch(
    hand_provider,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    view_ids: np.ndarray,
    num_fingers: int = DEFAULT_NUM_FINGERTIPS,
    fallback_meta_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    由 coarse hand pose FK 指尖估计每个 grasp 的抓取物体 meta/seg id。
    与训练时 dataset 的 object_meta_ids（dex 物体）对齐，供 contact 网络 conditioning。
    """
    from network.contact_diffusion_base import (
        _forward_finger_representatives,
    )

    tips = _forward_finger_representatives(
        hand_provider,
        trans,
        rot,
        qpos,
        object_pc=None,
        num_fingers=num_fingers,
    )
    tips_np = tips.detach().cpu().numpy()
    b = tips_np.shape[0]
    out = np.zeros(b, dtype=np.int64)
    for i in range(b):
        v = int(view_ids[i])
        pc = pc_all[v].detach().cpu().numpy()
        seg = seg_all[v].detach().cpu().numpy()
        meta = estimate_grasped_meta_id_numpy(tips_np[i], pc, seg)
        if meta is not None:
            out[i] = int(meta)
        elif fallback_meta_ids is not None:
            out[i] = int(fallback_meta_ids[i])
    return out


def contact_reconstruction_loss(
    hand_model,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    target_contacts: torch.Tensor,
    object_pc: Optional[torch.Tensor] = None,
    contact_mask: Optional[torch.Tensor] = None,
    num_fingers: Optional[int] = None,
    cdist_chunk: int = 2048,
) -> torch.Tensor:
    """拟合 pose 后每指代表点与 target contact 的 MSE（带 mask）。"""
    num_fingers = num_fingers or infer_num_fingertips_from_hand(hand_model)
    raw_tips = _forward_fingertips(hand_model, trans, rot, qpos)
    tips = aggregate_fingertips_per_link(
        raw_tips, num_fingers, object_pc=object_pc, cdist_chunk=cdist_chunk
    )
    if contact_mask is not None:
        diff = (tips - target_contacts).square() * contact_mask.unsqueeze(-1)
        denom = contact_mask.sum() * 3 + 1e-6
        return diff.sum() / denom
    return F.mse_loss(tips, target_contacts)
