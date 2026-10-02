import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import scipy.io as scio

from tqdm import trange
from termcolor import cprint
from torch.utils.tensorboard import SummaryWriter

from utils.robot_model import RobotModel
from utils.qpos_adapter import (
    apply_allegro_finger_curl_prior,
    apply_allegro_thumb_shape_prior,
    build_allegro_canonical_qpos,
    align_allegro_pose_to_teacher_fingertips,
    align_allegro_wrist_to_teacher_fingertips,
    fit_allegro_thumb_finger_opposition,
    fit_allegro_thumb_to_teacher_tip,
    fit_allegro_qpos_to_contacts,
    fit_allegro_qpos_to_contacts_two_phase,
    retarget_leap_pose_to_allegro,
    retarget_leap_to_allegro,
    warm_start_allegro_wrist_toward_contacts,
)
from utils.util import set_seed
from utils.config import ckpt_to_config
from utils.dataset import get_sparse_tensor
from network.graspness_sample_with_feature import GraspnessSampleWithFeature

from optimizer.mesh_sdf_offline import load_cached_sdf, load_or_build_cached_sdf
from utils.acronym_scene import AcronymScene
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider
from optimizer.stability_energy_refine_pen_gated import (
    PhysicsGuidanceV4StabilityPenGatedConfig,
    PhysicsGuidedPoseRefinerV4StabilityPenGated,
)
from eval.contact_stability_infer_pointnext import (
    ContactStabilityInferConfig,
    load_contact_stability_for_infer,
    refine_poses_with_contact_stability,
)
from network.contact_diffusion_pointnext_offset import (
    _forward_finger_representatives,
)
from eval.topk_select import select_top_n_per_view_with_feature

def resolve_result_exp_root(args) -> str:
    """grasps.npz 根目录：优先 --result_exp_root，否则 contact-stability/contact 实验目录。"""
    if args.result_exp_root:
        return args.result_exp_root
    if getattr(args, "contact_stability_ckpt", "") and args.contact_stability_ckpt:
        return os.path.dirname(os.path.dirname(args.contact_stability_ckpt))
    if getattr(args, "contact_dynamics_ckpt", "") and args.contact_dynamics_ckpt:
        return os.path.dirname(os.path.dirname(args.contact_dynamics_ckpt))
    if getattr(args, "contact_diffusion_ckpt", "") and args.contact_diffusion_ckpt:
        return os.path.dirname(os.path.dirname(args.contact_diffusion_ckpt))
    return os.path.dirname(os.path.dirname(args.ckpt_path))


def transform_pose_to_world(rotations, translations, extrinsics):
    world_rotations = extrinsics[:, :3, :3] @ rotations
    world_translations = (
        extrinsics[:, :3, :3] @ translations[:, :, None]
        + extrinsics[:, :3, 3:]
    )[:, :, 0]
    return world_rotations, world_translations


def load_view_object_poses(scene_id, camera, num_views, data_root="data"):
    meta_dir = os.path.join(data_root, "scenes", scene_id, camera, "meta")
    view_obj_poses = []

    for v in range(num_views):
        meta_path = os.path.join(meta_dir, f"{v:04d}.mat")
        meta = scio.loadmat(meta_path)

        cls_indexes = meta["cls_indexes"].reshape(-1).astype(np.int64)

        if "posed" in meta:
            poses = meta["posed"]
        elif "poses" in meta:
            poses = meta["poses"]
        else:
            raise KeyError(f"No 'posed' or 'poses' found in {meta_path}")

        pose_dict = {}

        for i, obj_id in enumerate(cls_indexes):
            T = np.eye(4, dtype=np.float32)

            if poses.shape[0] == 3 and poses.shape[1] == 4:
                T[:3, :4] = poses[:, :, i]
            elif poses.shape[0] == 4 and poses.shape[1] == 4:
                T = poses[:, :, i].astype(np.float32)
            else:
                raise ValueError(f"Unknown pose shape {poses.shape} in {meta_path}")

            pose_dict[int(obj_id)] = T

        view_obj_poses.append(pose_dict)

    return view_obj_poses


def get_main_object_ids_largest_seg(seg_all: torch.Tensor) -> np.ndarray:
    """原逻辑：每 view 取 seg 前景像素最多的物体 id。"""
    view_obj_ids = []

    for v in range(seg_all.shape[0]):
        ids = torch.unique(seg_all[v])
        ids = ids[ids > 0]

        if len(ids) == 0:
            view_obj_ids.append(1)
            continue

        counts = torch.stack([(seg_all[v] == obj_id).sum() for obj_id in ids])
        main_obj_id = ids[counts.argmax()]
        view_obj_ids.append(int(main_obj_id.item()))

    return np.array(view_obj_ids, dtype=np.int64)


# GraspNet：simulation annotations/0000.xml 的 obj_id 比 meta/seg 的 cls_indexes 小 1。
# 同一物体在两个编号空间相差 1（见 GraspNet meta id offset）。
GRASPNET_META_ID_OFFSET = 1


def ann_id_to_meta_id(ann_id: int) -> int:
    return int(ann_id) + GRASPNET_META_ID_OFFSET


def meta_id_to_ann_id(meta_id: int) -> int:
    return int(meta_id) - GRASPNET_META_ID_OFFSET


def load_graspnet_scene_object_ids(scene_id: str, data_root: str) -> List[int]:
    """仿真用 annotations/0000.xml 中的 obj_id 列表。"""
    ann_path = os.path.join(
        data_root, "scenes", scene_id, "realsense", "annotations", "0000.xml"
    )
    if not os.path.isfile(ann_path):
        return []
    root = ET.parse(ann_path).getroot()
    return [int(obj.find("obj_id").text) for obj in root.findall("obj")]


def load_acronym_scene_object_ids(scene_id: str, data_root: str) -> List[int]:
    split = scene_id.split("_")[1]
    ann_path = os.path.join(
        data_root, f"acronym_test_scenes/test_acronym_{split}", f"{scene_id}.npz"
    )
    if not os.path.isfile(ann_path):
        return []
    ann = np.load(ann_path, allow_pickle=True)["arr_0"][None][0]
    return list(ann.keys())


def _pick_main_from_seg(
    seg_v: torch.Tensor, candidate_ids: Optional[Sequence[int]]
) -> Optional[int]:
    ids = torch.unique(seg_v)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None

    if candidate_ids is not None and len(candidate_ids) > 0:
        cand_set = {int(x) for x in candidate_ids}
        valid = [int(i) for i in ids.tolist() if int(i) in cand_set]
        if valid:
            counts = torch.stack([(seg_v == oid).sum() for oid in valid])
            return int(valid[int(counts.argmax().item())])

    return None


def get_main_object_ids_ann_meta_intersection(
    seg_all: torch.Tensor,
    ann_object_ids: Sequence[int],
    meta_cls_per_view: List[List[int]],
    fallback: str = "largest_seg",
) -> Tuple[np.ndarray, int]:
    """
    每 view：在 (ann ∩ meta_cls) 中，选 seg 像素最多的 id 作为 SDF 目标。
    若交集为空，按 fallback 回退。
    """
    ann_set: Set[int] = {int(x) for x in ann_object_ids}
    view_obj_ids: List[int] = []
    n_fallback = 0

    for v in range(seg_all.shape[0]):
        meta_ids = meta_cls_per_view[v] if v < len(meta_cls_per_view) else []
        intersection = ann_set & {int(x) for x in meta_ids}

        chosen = _pick_main_from_seg(seg_all[v], list(intersection))

        if chosen is None:
            n_fallback += 1
            if fallback == "ann_only":
                chosen = _pick_main_from_seg(seg_all[v], list(ann_set))
            elif fallback == "meta_only":
                chosen = _pick_main_from_seg(seg_all[v], meta_ids)
            if chosen is None:
                chosen = int(get_main_object_ids_largest_seg(seg_all[v : v + 1])[0])

        view_obj_ids.append(int(chosen))

    return np.array(view_obj_ids, dtype=np.int64), n_fallback


def get_main_object_ids_ann_meta_aligned(
    seg_all: torch.Tensor,
    ann_object_ids: Sequence[int],
    meta_cls_per_view: List[List[int]],
) -> Tuple[np.ndarray, int]:
    """
    v9：在 ann（仿真）空间选主物体，返回 meta/seg id（cls_indexes = ann_id + 1）。

    每 view：在 ann 列表中，取 meta_id=ann_id+1 且在当前 view seg 可见的物体，
    选 seg 像素最多者；不再用「与 ann 数值相交」或 blind fallback。
    """
    view_meta_ids: List[int] = []
    n_fallback = 0

    for v in range(seg_all.shape[0]):
        meta_ids = meta_cls_per_view[v] if v < len(meta_cls_per_view) else []
        meta_set = {int(x) for x in meta_ids}

        ann_visible: List[int] = []
        for ann_id in ann_object_ids:
            meta_id = ann_id_to_meta_id(ann_id)
            if meta_id in meta_set and bool((seg_all[v] == meta_id).any()):
                ann_visible.append(int(ann_id))

        if ann_visible:
            best_ann = max(
                ann_visible,
                key=lambda a: int((seg_all[v] == ann_id_to_meta_id(a)).sum()),
            )
            chosen_meta = ann_id_to_meta_id(best_ann)
        else:
            n_fallback += 1
            meta_from_ann = [
                ann_id_to_meta_id(a)
                for a in ann_object_ids
                if ann_id_to_meta_id(a) in meta_set
            ]
            chosen_meta = _pick_main_from_seg(seg_all[v], meta_from_ann)
            if chosen_meta is None and meta_set:
                n_fallback += 1
                chosen_meta = _pick_main_from_seg(seg_all[v], list(meta_set))
            if chosen_meta is None:
                n_fallback += 1
                chosen_meta = int(
                    get_main_object_ids_largest_seg(seg_all[v : v + 1])[0]
                )
            chosen_meta = int(chosen_meta)

        view_meta_ids.append(chosen_meta)

    return np.array(view_meta_ids, dtype=np.int64), n_fallback


def mask_seg_to_main_objects(
    seg_all: torch.Tensor,
    view_obj_ids: np.ndarray,
    min_object_points: int = 32,
) -> Tuple[torch.Tensor, int]:
    """
        非 pred_main 像素 seg→0，使 diffusion seed 只能落在 pred_main 上。
        pred_main 预测的主抓取物体
    
    """
    seg_masked = seg_all.clone()
    n_degraded = 0
    for v in range(seg_all.shape[0]):
        oid = int(view_obj_ids[v])
        n_main = int((seg_all[v] == oid).sum().item())
        if n_main < min_object_points:
            n_degraded += 1
            continue
        seg_masked[v][seg_all[v] != oid] = 0
    return seg_masked, n_degraded


def build_object_pc_for_grasps(
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    cand_view_ids: np.ndarray,
    cand_obj_ids: np.ndarray,
    max_points: int,
    device: torch.device,
) -> torch.Tensor:
    """每个 grasp 仅使用对应 view 的 pred_main 物体点云。"""
    n = len(cand_view_ids)
    out = torch.zeros(n, max_points, 3, device=device, dtype=torch.float32)
    for i in range(n):
        v = int(cand_view_ids[i])
        oid = int(cand_obj_ids[i])
        mask = seg_all[v] == oid
        pts = pc_all[v][mask]
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > max_points:
            idx = torch.randperm(pts.shape[0])[:max_points]
            pts = pts[idx]
        out[i, : pts.shape[0]] = pts.to(device)
    return out


def _mean_nn_to_seg_points(
    query: np.ndarray,
    pc: torch.Tensor,
    seg: torch.Tensor,
    meta_id: int,
) -> Tuple[float, int]:
    mask = seg == int(meta_id)
    pts = pc[mask].detach().cpu().numpy().astype(np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return float("inf"), 0
    dists = []
    for q in query.astype(np.float32):
        d = np.linalg.norm(pts - q[None], axis=1)
        dists.append(float(d.min()))
    return float(np.mean(dists)) if dists else float("inf"), int(pts.shape[0])


def estimate_grasped_object_meta_ids(
    hand_provider,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    cand_view_ids: np.ndarray,
    pred_meta_ids: np.ndarray,
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate which segmented object the coarse hand is actually contacting.

    This is only used for the v16e A/B branch where simulation success is
    object-agnostic.  If the estimate is not clearly better than pred_main,
    keep pred_main.
    """
    with torch.no_grad():
        tips = _forward_finger_representatives(
            hand_provider,
            trans,
            rot,
            qpos,
            object_pc=None,
            num_fingers=4,
        ).detach().cpu().numpy()

    n = len(cand_view_ids)
    grasped_meta = np.array(pred_meta_ids, dtype=np.int64, copy=True)
    contact_meta = np.array(pred_meta_ids, dtype=np.int64, copy=True)
    tip_to_pred = np.full(n, np.inf, dtype=np.float32)
    tip_to_grasped = np.full(n, np.inf, dtype=np.float32)
    switched = np.zeros(n, dtype=np.int64)
    grasped_points = np.zeros(n, dtype=np.int64)

    for i in range(n):
        view_id = int(cand_view_ids[i])
        pred_meta = int(pred_meta_ids[i])
        pc = pc_all[view_id]
        seg = seg_all[view_id]
        tip_i = tips[i]

        valid = seg > 0
        pc_valid = pc[valid].detach().cpu().numpy().astype(np.float32)
        seg_valid = seg[valid].detach().cpu().numpy().astype(np.int64)
        if pc_valid.shape[0] == 0:
            continue

        votes = []
        for q in tip_i.astype(np.float32):
            d = np.linalg.norm(pc_valid - q[None], axis=1)
            votes.append(int(seg_valid[int(np.argmin(d))]))
        if votes:
            ids, counts = np.unique(np.asarray(votes, dtype=np.int64), return_counts=True)
            grasped = int(ids[int(np.argmax(counts))])
        else:
            grasped = pred_meta
        grasped_meta[i] = grasped

        d_pred, _n_pred = _mean_nn_to_seg_points(tip_i, pc, seg, pred_meta)
        d_grasp, n_grasp = _mean_nn_to_seg_points(tip_i, pc, seg, grasped)
        tip_to_pred[i] = d_pred
        tip_to_grasped[i] = d_grasp
        grasped_points[i] = n_grasp

        if (
            grasped != pred_meta
            and n_grasp >= int(cfg.contact_grasped_min_points)
            and d_grasp < float(cfg.contact_grasped_max_tip_dist)
            and (d_pred - d_grasp) > float(cfg.contact_grasped_margin)
        ):
            contact_meta[i] = grasped
            switched[i] = 1

    return contact_meta, grasped_meta, tip_to_pred, tip_to_grasped, switched, grasped_points


def resolve_main_object_ids(
    seg_all: torch.Tensor,
    mode: str,
    ann_object_ids: Sequence[int],
    meta_cls_per_view: List[List[int]],
    ann_meta_fallback: str,
) -> Tuple[np.ndarray, int]:
    if mode == "largest_seg":
        return get_main_object_ids_largest_seg(seg_all), 0
    if mode == "ann_meta_intersection":
        return get_main_object_ids_ann_meta_intersection(
            seg_all,
            ann_object_ids,
            meta_cls_per_view,
            fallback=ann_meta_fallback,
        )
    if mode == "ann_meta_aligned":
        return get_main_object_ids_ann_meta_aligned(
            seg_all,
            ann_object_ids,
            meta_cls_per_view,
        )
    raise ValueError(f"Unknown main_object_mode={mode!r}")


# 兼容旧名
get_main_object_ids = get_main_object_ids_largest_seg


def apply_pointnext_strict_physics_gate(refine_out, args):
    """Outer accept gate for PointNeXt contact-guided physics refinement."""
    base_ok = refine_out["use_refined"].bool()
    device = base_ok.device
    dtype = refine_out["init_E_pen"].dtype

    if int(args.pointnext_strict_require_base_gate):
        accept = base_ok.clone()
    else:
        accept = torch.ones_like(base_ok, dtype=torch.bool, device=device)

    init_contact = refine_out.get("init_E_tip_anchor")
    final_contact = refine_out.get("final_E_tip_anchor")
    if init_contact is not None and final_contact is not None:
        min_contact_improve = torch.as_tensor(
            float(args.pointnext_strict_contact_min_improve),
            device=device,
            dtype=dtype,
        )
        contact_ok = final_contact <= (init_contact - min_contact_improve)
    else:
        contact_ok = torch.ones_like(base_ok, dtype=torch.bool, device=device)

    pen_eps = torch.as_tensor(
        float(args.pointnext_strict_pen_eps), device=device, dtype=dtype
    )
    pen_ok = refine_out["final_E_pen"] <= (refine_out["init_E_pen"] + pen_eps)

    init_stab = refine_out.get("init_E_stab")
    final_stab = refine_out.get("final_E_stab")
    if init_stab is not None and final_stab is not None:
        stab_eps = torch.as_tensor(
            float(args.pointnext_strict_stability_eps),
            device=device,
            dtype=dtype,
        )
        stability_ok = final_stab <= (init_stab + stab_eps)
    else:
        stability_ok = torch.ones_like(base_ok, dtype=torch.bool, device=device)

    max_trans_delta = (
        float(args.pointnext_strict_max_trans_delta)
        if float(args.pointnext_strict_max_trans_delta) > 0.0
        else float(args.max_trans_delta)
    )
    max_q_delta = (
        float(args.pointnext_strict_max_q_delta)
        if float(args.pointnext_strict_max_q_delta) > 0.0
        else float(args.max_q_delta)
    )
    pose_ok = (
        (refine_out["trans_delta"] <= max_trans_delta)
        & (refine_out["q_delta"] <= max_q_delta)
    )

    accept = accept & contact_ok & pen_ok & stability_ok & pose_ok

    selected_trans = torch.where(
        accept[:, None], refine_out["refined_trans"], refine_out["init_pose"][:, :3]
    )
    selected_rot = torch.where(
        accept[:, None, None],
        refine_out["refined_rot"],
        PhysicsGuidedPoseRefinerV4StabilityPenGated.rot6d_to_matrix(
            refine_out["init_pose"][:, 3:9]
        ),
    )
    selected_qpos = torch.where(
        accept[:, None], refine_out["refined_qpos"], refine_out["init_pose"][:, 9:]
    )

    strict_out = dict(refine_out)
    strict_out["selected_trans"] = selected_trans
    strict_out["selected_rot"] = selected_rot
    strict_out["selected_qpos"] = selected_qpos
    strict_out["use_refined"] = accept.float()
    strict_out["use_refined_base"] = base_ok.float()
    strict_out["strict_contact_ok"] = contact_ok.float()
    strict_out["strict_pen_ok"] = pen_ok.float()
    strict_out["strict_stability_ok"] = stability_ok.float()
    strict_out["strict_pose_ok"] = pose_ok.float()
    return strict_out


def apply_allegro_transfer_profile(args) -> None:
    """Recommended Allegro cross-embodiment settings (Leap path unchanged)."""
    mode = str(getattr(args, "allegro_embodiment_mode", "optimize")).lower()
    explicit_max_trans_delta = bool(
        getattr(args, "_allegro_contact_ik_max_trans_delta_explicit", False)
    )
    explicit_post_physics_ik = bool(
        getattr(args, "_allegro_post_physics_ik_explicit", False)
    )
    explicit_strict_max_trans_delta = bool(
        getattr(args, "_pointnext_strict_max_trans_delta_explicit", False)
    )
    explicit_strict_max_q_delta = bool(
        getattr(args, "_pointnext_strict_max_q_delta_explicit", False)
    )
    args.allegro_contact_ik_tip_mode = "biotac"
    # Method-style stage2: equal tip weights (no hard index bias). Leap path
    # never used Allegro finger weights; keep equal unless CLI overrides.
    if not bool(
        getattr(args, "_allegro_contact_ik_finger_weights_explicit", False)
    ):
        args.allegro_contact_ik_finger_weights = "1,1,1,1"

    if mode in ("optimize", "hybrid"):
        args.allegro_embodiment_mode = mode
        args.allegro_contact_ik_pose_reg = 0.01
        args.allegro_contact_ik_lock_abduction = 0
        args.allegro_contact_ik_abduction_reg_weight = 2.0
        # Keep the preset's permissive default, but honor an explicit CLI
        # value so Allegro wrist-translation ablations are reproducible.
        if not explicit_max_trans_delta:
            args.allegro_contact_ik_max_trans_delta = max(
                float(args.allegro_contact_ik_max_trans_delta), 0.12
            )
        args.allegro_contact_ik_max_rot_delta = max(
            float(args.allegro_contact_ik_max_rot_delta), 1.0
        )
        args.allegro_contact_ik_steps = max(int(args.allegro_contact_ik_steps), 200)
        args.allegro_contact_ik_lr = max(float(args.allegro_contact_ik_lr), 0.04)
        args.max_iters = max(int(args.max_iters), 30)
        args.max_trans_delta = max(float(args.max_trans_delta), 0.008)
        args.max_q_delta = max(float(args.max_q_delta), 0.10)
        args.physics_w_tip = max(float(args.physics_w_tip), 1.0)
        args.physics_pen_threshold = max(float(args.physics_pen_threshold), 0.002)
        args.physics_min_pen_improve = max(float(args.physics_min_pen_improve), 1e-5)
        args.pointnext_strict_require_base_gate = 1
        if not explicit_strict_max_trans_delta:
            args.pointnext_strict_max_trans_delta = 0.015
        if not explicit_strict_max_q_delta:
            args.pointnext_strict_max_q_delta = 0.10
        args.pointnext_strict_pen_eps = 1e-5
        args.pointnext_strict_stability_eps = 5e-4
        args.pointnext_strict_contact_min_improve = 0.0
        args.allegro_contact_rerank = 0
        if not explicit_post_physics_ik:
            args.allegro_post_physics_ik = 1
        args.allegro_post_physics_ik_steps = max(
            int(getattr(args, "allegro_post_physics_ik_steps", 100)), 100
        )
    else:
        args.allegro_embodiment_mode = "retarget"
        args.allegro_contact_ik_pose_reg = 0.03
        args.allegro_contact_ik_lock_abduction = 1
        args.allegro_contact_ik_abduction_reg_weight = 3.0
        args.allegro_contact_ik_max_rot_delta = min(
            float(args.allegro_contact_ik_max_rot_delta), 0.60
        )
        args.pointnext_strict_require_base_gate = 0
        args.pointnext_strict_max_trans_delta = 0.02
        args.pointnext_strict_max_q_delta = 0.12
        args.pointnext_strict_pen_eps = 1e-5
        args.pointnext_strict_stability_eps = 5e-4
        args.pointnext_strict_contact_min_improve = 0.0
        args.physics_pen_threshold = min(float(args.physics_pen_threshold), 0.001)
        args.physics_min_pen_improve = 0.0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid v9: pred_main 对齐 diffusion+physics；"
            "ann_meta_aligned=ann∩meta 选主物体 + seg 掩码 + 物体点云"
        )
    )

    parser.add_argument(
        "--config_yaml",
        type=str,
        default="",
        help="模型 config；默认从 ckpt 旁 experiments/*/config.yaml 推断，缺失时用 train_dex_ours.yaml",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--urdf_path",
        type=str,
        default="robot_models/urdf/leap_hand_simplified.urdf",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="robot_models/meta/leap_hand/meta.yaml",
    )
    parser.add_argument("--hand_name", type=str, default="leap_hand")
    parser.add_argument(
        "--qpos_adapter", choices=["auto", "none", "leap_to_allegro"], default="auto",
        help="Allegro cross-hand initialization; default auto maps Leap qpos to Allegro.",
    )
    parser.add_argument(
        "--allegro_thumb_retarget_mode",
        choices=["semantic", "shape_prior"],
        default="shape_prior",
        help=(
            "Allegro-only thumb retarget correction. semantic copies the "
            "Leap thumb joint semantics; shape_prior additionally closes "
            "thumb joints 1/2/3 according to the three primary fingers."
        ),
    )
    parser.add_argument(
        "--allegro_thumb_opposition",
        type=int,
        default=1,
        help=(
            "After shape_prior (+ optional teacher tip fit), optimize only "
            "thumb joints so the pad faces index/middle (1=on)."
        ),
    )
    parser.add_argument(
        "--allegro_thumb_opposition_steps",
        type=int,
        default=80,
        help="Steps for Allegro thumb finger-opposition fit.",
    )
    parser.add_argument(
        "--allegro_thumb_teacher_fit",
        type=int,
        default=1,
        help=(
            "Allegro-only: after wrist geometry alignment, fit only thumb "
            "joints to the Leap teacher thumb tip (0=off)."
        ),
    )
    parser.add_argument(
        "--allegro_thumb_teacher_fit_steps",
        type=int,
        default=120,
        help="Optimization steps for the Allegro-only static thumb fit.",
    )
    parser.add_argument(
        "--allegro_thumb_teacher_fit_lr",
        type=float,
        default=0.04,
        help="Learning rate for the Allegro-only static thumb fit.",
    )
    parser.add_argument(
        "--allegro_thumb_teacher_fit_reg",
        type=float,
        default=0.0001,
        help="Thumb joint-change regularization for the static teacher fit.",
    )
    parser.add_argument(
        "--pose_adapter", choices=["auto", "none", "leap_to_allegro"], default="auto",
        help="Allegro mount/palm frame conversion; default auto maps Leap pose to Allegro.",
    )
    parser.add_argument(
        "--allegro_pose_translation_mode",
        choices=["root", "palm"],
        default="palm",
        help=(
            "Allegro-only pose translation convention. "
            "palm preserves the legacy mount-to-palm offset; "
            "root preserves the Leap root origin for Allegro's free root."
        ),
    )
    parser.add_argument(
        "--allegro_pose_rotation_mode",
        choices=["canonical", "palm"],
        default="canonical",
        help=(
            "Allegro-only pose rotation convention. canonical aligns evaluator "
            "grasp axes; palm aligns Allegro palm_link orientation with the "
            "Leap teacher root orientation."
        ),
    )
    parser.add_argument(
        "--allegro_coarse_wrist_align",
        type=int,
        default=0,
        help=(
            "Allegro-only: translate the retargeted coarse wrist so the "
            "Allegro fingertip centroid matches the Leap teacher (0=off)."
        ),
    )
    parser.add_argument(
        "--allegro_coarse_wrist_align_alpha",
        type=float,
        default=1.0,
        help="Interpolation strength for Allegro coarse fingertip-centroid alignment.",
    )
    parser.add_argument(
        "--allegro_coarse_geometry_align",
        type=int,
        default=0,
        help=(
            "Allegro-only: align the retargeted wrist rotation and translation "
            "to the Leap teacher using corresponding fingertip geometry (0=off)."
        ),
    )
    parser.add_argument(
        "--source_urdf_path", default="robot_models/urdf/leap_hand_simplified.urdf",
    )
    parser.add_argument(
        "--source_meta_path", default="robot_models/meta/leap_hand/meta.yaml",
    )
    parser.add_argument("--allegro_contact_ik_steps", type=int, default=60)
    parser.add_argument("--allegro_contact_ik_lr", type=float, default=0.03)
    parser.add_argument("--allegro_contact_ik_pose_reg", type=float, default=0.01)
    parser.add_argument("--allegro_contact_ik_max_trans_delta", type=float, default=0.05)
    parser.add_argument("--allegro_contact_ik_max_rot_delta", type=float, default=0.60)
    parser.add_argument(
        "--allegro_contact_ik_tip_mode",
        choices=["biotac", "representative"],
        default="biotac",
        help=(
            "Allegro-only contact IK fingertip definition. "
            "'biotac' aligns URDF biotac tips with Isaac sim; "
            "'representative' uses hand_provider surface points (legacy)."
        ),
    )
    parser.add_argument(
        "--allegro_contact_ik_lock_abduction",
        type=int,
        default=0,
        help="1=Allegro IK keeps joint_0 abduction at retarget values",
    )
    parser.add_argument(
        "--allegro_contact_ik_finger_weights",
        type=str,
        default="",
        help="Comma-separated thumb,index,middle,ring contact weights (Allegro only)",
    )
    parser.add_argument(
        "--allegro_contact_ik_abduction_reg_weight",
        type=float,
        default=2.0,
        help="Extra regularization on Allegro joint_0 during contact IK",
    )
    parser.add_argument(
        "--allegro_contact_ik_coordination_weight",
        type=float,
        default=0.0,
        help=(
            "Allegro-only soft finger synergy penalty. "
            "0 keeps the existing IK objective unchanged."
        ),
    )
    parser.add_argument(
        "--allegro_contact_ik_canonical_abduction_weight",
        type=float,
        default=0.0,
        help=(
            "Allegro-only penalty for excessive joint_0 abduction relative "
            "to the canonical pose. 0 disables it."
        ),
    )
    parser.add_argument(
        "--allegro_contact_ik_coordination_q2_ratio",
        type=float,
        default=0.75,
        help="Soft target ratio for Allegro joint_2 relative to joint_1.",
    )
    parser.add_argument(
        "--allegro_contact_ik_coordination_q3_ratio",
        type=float,
        default=0.50,
        help="Soft target ratio for Allegro joint_3 relative to joint_2.",
    )
    parser.add_argument(
        "--allegro_contact_ik_coordination_abduction_soft_limit",
        type=float,
        default=0.25,
        help="Allegro joint_0 abduction envelope in radians before penalty.",
    )
    parser.add_argument(
        "--allegro_contact_rerank",
        type=int,
        default=1,
        help=(
            "Allegro only: rerank K contact candidates with quick biotac IK "
            "before selecting target_contacts (0=off, 1=on)."
        ),
    )
    parser.add_argument(
        "--allegro_contact_rerank_ik_steps",
        type=int,
        default=40,
        help="Quick IK steps per contact candidate during Allegro rerank",
    )
    parser.add_argument(
        "--allegro_contact_rerank_w_ik",
        type=float,
        default=800.0,
        help="Penalty weight on quick-IK MSE when reranking Allegro contacts",
    )
    parser.add_argument(
        "--allegro_contact_rerank_w_stab",
        type=float,
        default=1.0,
        help="Multiplier on stability score during Allegro contact rerank",
    )
    parser.add_argument(
        "--allegro_contact_rerank_min_ik_rel_improve",
        type=float,
        default=0.25,
        help="Only switch contact if quick-IK MSE improves by this relative margin",
    )
    parser.add_argument(
        "--allegro_contact_rerank_max_stab_drop",
        type=float,
        default=0.35,
        help="Max allowed stability-score drop when switching Allegro contact",
    )
    parser.add_argument(
        "--allegro_transfer_profile",
        choices=["auto", "legacy", "off"],
        default="auto",
        help=(
            "Allegro cross-embodiment preset (Leap path ignores this). "
            "auto=mode-specific biotac IK + physics gates; "
            "legacy=keep CLI values; off=disable preset."
        ),
    )
    parser.add_argument(
        "--allegro_embodiment_mode",
        choices=["optimize", "hybrid", "retarget"],
        default="hybrid",
        help=(
            "Allegro cross-embodiment initialization (Leap path ignores this). "
            "hybrid=Leap qpos retarget + wrist + contact IK (recommended); "
            "optimize=canonical qpos + teacher wrist + contact IK; "
            "retarget=copy Leap joints only."
        ),
    )
    parser.add_argument(
        "--allegro_post_physics_ik",
        type=int,
        default=0,
        help="Allegro only: re-run biotac IK after physics/anchor patch (0=off, 1=on).",
    )
    parser.add_argument(
        "--allegro_post_physics_ik_steps",
        type=int,
        default=100,
        help="IK steps for post-physics Allegro biotac polish.",
    )
    parser.add_argument(
        "--allegro_coarse_only",
        type=int,
        default=0,
        help=(
            "Allegro-only diagnostic: save teacher coarse grasp after "
            "Leap->Allegro qpos/pose retarget and skip contact/physics stages."
        ),
    )
    parser.add_argument(
        "--allegro_init_grasps_npz",
        type=str,
        default="",
        help=(
            "Allegro stage-2 method path: load frozen coarse grasps (table/world) "
            "and run contact-select + pen-gated physics from them. Skips "
            "destructive open-hand priors. Empty=disabled."
        ),
    )
    parser.add_argument(
        "--allegro_preserve_coarse_pregrasp",
        type=int,
        default=1,
        help=(
            "When allegro_init_grasps_npz is set, also save coarse_* fields so "
            "the evaluator keeps open approach from stage-1 (1=on)."
        ),
    )

    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--scene_id", type=str, default="scene_0100")
    parser.add_argument("--dataset", type=str, default="graspnet", choices=["graspnet", "acronym"])
    parser.add_argument("--all_scene_ids_acronym", type=str, nargs="*", default=None)

    parser.add_argument("--grasp_num", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32)

    # physics post-refine（对齐 v4_physics）
    parser.add_argument("--max_iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pen_threshold", type=float, default=0.0)
    parser.add_argument("--max_trans_delta", type=float, default=0.003)
    parser.add_argument("--max_q_delta", type=float, default=0.08)
    parser.add_argument("--physics_min_energy_improve", type=float, default=0.0001)
    parser.add_argument(
        "--physics_w_tip",
        type=float,
        default=0.5,
        help="Physics 指尖 anchor 到 V4 target_contacts 的权重",
    )
    parser.add_argument(
        "--physics_w_mid",
        type=float,
        default=1.0,
        help="细长体长轴中段约束权重（仅 elongated_gate=1）",
    )
    parser.add_argument(
        "--physics_w_moment",
        type=float,
        default=1.0,
        help="重力力臂约束权重（仅 elongated_gate=1）",
    )
    parser.add_argument(
        "--physics_w_bilateral",
        type=float,
        default=0.5,
        help="双侧夹持奖励权重（仅 elongated_gate=1，energy 中为负项）",
    )
    parser.add_argument(
        "--physics_w_pose",
        type=float,
        default=10.0,
        help="pose 相对 teacher/contact-fit 初始 pose 的正则权重",
    )
    parser.add_argument(
        "--physics_pen_on_tips_only",
        type=int,
        default=1,
        help="1=SDF 穿透仅惩罚指尖代表点",
    )
    parser.add_argument(
        "--physics_pen_threshold",
        type=float,
        default=0.002,
        help="v16b pen-gated accept：初始 E_pen 须大于此值才考虑采纳 physics",
    )
    parser.add_argument(
        "--physics_min_pen_improve",
        type=float,
        default=1e-6,
        help="v16b pen-gated accept：final_E_pen 须严格小于 init_E_pen 且改善量大于此值",
    )
    parser.add_argument(
        "--pointnext_strict_physics_gate",
        type=int,
        default=1,
        help="1=PointNeXt 链路外层严格门控 physics replace：contact/pen/stability/pose 均通过才替换",
    )
    parser.add_argument(
        "--pointnext_strict_require_base_gate",
        type=int,
        default=1,
        help="1=严格门控必须同时满足原 pen-gated use_refined；0=只使用外层严格条件",
    )
    parser.add_argument(
        "--pointnext_strict_contact_min_improve",
        type=float,
        default=1e-7,
        help="refined 的 tip-target contact energy 至少降低该值才允许 replace",
    )
    parser.add_argument(
        "--pointnext_strict_pen_eps",
        type=float,
        default=0.0,
        help="允许 final_E_pen 比 init_E_pen 增加的容忍量；默认不允许变差",
    )
    parser.add_argument(
        "--pointnext_strict_stability_eps",
        type=float,
        default=0.0,
        help="允许 final_E_stab 比 init_E_stab 增加的容忍量；默认不允许稳定性能量变差",
    )
    parser.add_argument(
        "--pointnext_strict_max_trans_delta",
        type=float,
        default=-1.0,
        help="严格门控最大平移改变量；<=0 时沿用 --max_trans_delta",
    )
    parser.add_argument(
        "--pointnext_strict_max_q_delta",
        type=float,
        default=-1.0,
        help="严格门控最大关节改变量；<=0 时沿用 --max_q_delta",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--scene_num", type=int, default=10)
    parser.add_argument("--logs_path", type=str, default="logs/sdf_hybrid_predict")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--mesh_root", type=str, default="/data/meshdata")
    parser.add_argument(
        "--acronym_sdf_root",
        type=str,
        default="/data/acronym_sdf",
        help="Acronym object-code keyed SDF cache root",
    )
    parser.add_argument(
        "--acronym_mesh_root",
        type=str,
        default="/data/acronym/meshes/models",
        help="Acronym hash object-code mesh root",
    )
    parser.add_argument(
        "--acronym_mapping_root",
        type=str,
        default="",
        help="Optional scene mapping cache root; empty uses data_root/acronym_test_scenes/scene_object_mapping",
    )
    parser.add_argument("--sdf_grid_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default="/data")

    parser.add_argument(
        "--main_object_mode",
        type=str,
        default="ann_meta_aligned",
        choices=["largest_seg", "ann_meta_intersection", "ann_meta_aligned"],
        help=(
            "ann_meta_aligned(v9 默认)=ann∩meta + pred_main∈ann + diffusion seg 掩码；"
            "ann_meta_intersection=v8 行为"
        ),
    )
    parser.add_argument(
        "--ann_meta_fallback",
        type=str,
        default="ann_only",
        choices=["largest_seg", "ann_only", "meta_only"],
        help="ann_meta_intersection 模式下 ann∩meta 为空时的回退；aligned 模式忽略",
    )
    parser.add_argument(
        "--min_main_object_points",
        type=int,
        default=32,
        help="view 内 pred_main seg 像素少于此值时不做 seg 掩码（保留全场景）",
    )
    parser.add_argument(
        "--mask_diffusion_to_main",
        type=int,
        default=1,
        help="1=diffusion 输入 seg 仅保留 pred_main（v9 核心）",
    )
    parser.add_argument(
        "--crop_object_pc_to_main",
        type=int,
        default=1,
        help="1=physics 的 object_pc 仅用 pred_main 点云",
    )
    parser.add_argument(
        "--contact_object_mode",
        type=str,
        default="grasped_gate",
        choices=["pred_main", "grasped_gate"],
        help="v16e: pred_main=沿用主物体；grasped_gate=粗手指尖明显更靠近其他物体时切换 contact/SDF object",
    )
    parser.add_argument("--contact_grasped_max_tip_dist", type=float, default=0.06)
    parser.add_argument("--contact_grasped_margin", type=float, default=0.05)
    parser.add_argument("--contact_grasped_min_points", type=int, default=32)
    parser.add_argument(
        "--max_obj_points",
        type=int,
        default=2048,
        help="PointNeXt contact/stability conditioning 使用的最大 segmented object 点数",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="",
        help="结果子目录名；默认 ann_meta→results_sdf_hybrid_v8_ann_meta，largest_seg→results_sdf_hybrid_v7",
    )
    parser.add_argument(
        "--result_exp_root",
        type=str,
        default="",
        help="grasps.npz 保存根目录；默认用 contact-stability/teacher ckpt 旁目录",
    )
    parser.add_argument(
        "--contact_stability_ckpt",
        type=str,
        default="",
        help="Contact–Stability v4b-offset 权重（含 contact + stability head）",
    )
    parser.add_argument(
        "--contact_dynamics_ckpt",
        type=str,
        default="",
        help="已废弃，请用 --contact_stability_ckpt",
    )
    parser.add_argument(
        "--contact_v2_ckpt",
        type=str,
        default="",
        help="可选：v4e-pointnext ckpt 缺 contact 权重时 fallback 到 v2d-pointnext contact ckpt",
    )
    parser.add_argument(
        "--contact_diffusion_ckpt",
        type=str,
        default="",
        help="已废弃，请用 --contact_dynamics_ckpt",
    )
    parser.add_argument(
        "--contact_fit_steps",
        type=int,
        default=0,
        help="覆盖 contact fit 步数；0=使用 ckpt 内 cfg",
    )
    parser.add_argument(
        "--contact_num_samples",
        type=int,
        default=8,
        help="每个 grasp 采样的 contact candidate 数，stability_head + object-distance rerank 选最优",
    )
    parser.add_argument(
        "--contact_obj_dist_weight",
        type=float,
        default=20.0,
        help="contact candidate 到 conditioning object_pc 平均距离的 rerank 扣分权重；0=只按 stability score",
    )
    parser.add_argument(
        "--contact_obj_dist_clip",
        type=float,
        default=0.20,
        help="contact-object 距离扣分裁剪上限，单位 m",
    )
    parser.add_argument(
        "--contact_project_to_object",
        type=int,
        default=1,
        help="1=将采样 contact 投影到 conditioning object_pc 最近点后再打分/输出；0=只 rerank 不投影",
    )
    parser.add_argument(
        "--contact_fit_max_trans_delta",
        type=float,
        default=0.003,
        help="contact fit conservative accept：相对 teacher 最大平移",
    )
    parser.add_argument(
        "--contact_fit_max_q_delta",
        type=float,
        default=0.08,
        help="contact fit conservative accept：相对 teacher 最大关节变化",
    )

    args = parser.parse_args()
    args._allegro_contact_ik_max_trans_delta_explicit = (
        "--allegro_contact_ik_max_trans_delta" in sys.argv
    )
    args._allegro_contact_ik_finger_weights_explicit = (
        "--allegro_contact_ik_finger_weights" in sys.argv
    )
    args._allegro_post_physics_ik_explicit = (
        "--allegro_post_physics_ik" in sys.argv
    )
    args._pointnext_strict_max_trans_delta_explicit = (
        "--pointnext_strict_max_trans_delta" in sys.argv
    )
    args._pointnext_strict_max_q_delta_explicit = (
        "--pointnext_strict_max_q_delta" in sys.argv
    )

    if not args.contact_stability_ckpt and args.contact_dynamics_ckpt:
        args.contact_stability_ckpt = args.contact_dynamics_ckpt
    if not args.contact_stability_ckpt and args.contact_diffusion_ckpt:
        args.contact_stability_ckpt = args.contact_diffusion_ckpt
    if not args.contact_stability_ckpt:
        default_cs = (
            "experiments/contact_stability_v4e_pointnext_offset/ckpt/ckpt_1000.pth"
        )
        if os.path.isfile(default_cs):
            args.contact_stability_ckpt = default_cs
    if not args.contact_v2_ckpt:
        default_v2 = (
            "experiments/contact_diffusion_v2d_pointnext_offset/ckpt/ckpt_1000.pth"
        )
        if os.path.isfile(default_v2):
            args.contact_v2_ckpt = default_v2
    if args.hand_name == "allegro_hand":
        if "leap_hand" in os.path.basename(args.urdf_path):
            args.urdf_path = "robot_models/urdf/allegro_hand_simplified.urdf"
        if "leap_hand" in args.meta_path.replace("\\", "/"):
            args.meta_path = "robot_models/meta/allegro_hand/meta.yaml"
    if int(args.allegro_coarse_only):
        if args.hand_name != "allegro_hand":
            raise ValueError("--allegro_coarse_only requires --hand_name allegro_hand")
        # The Scheme-C wrapper injects default contact checkpoints. Clear all
        # second-stage components after parsing so this mode is unambiguous.
        args.contact_stability_ckpt = ""
        args.contact_v2_ckpt = ""
        args.contact_dynamics_ckpt = ""
        args.contact_diffusion_ckpt = ""
        args.allegro_post_physics_ik = 0
        args.pointnext_strict_physics_gate = 0

    if not args.result_subdir:
        if args.main_object_mode == "ann_meta_aligned":
            args.result_subdir = (
                "results_sdf_hybrid_v16e_pointnext_grasped_object_offset"
                if args.dataset == "graspnet"
                else "results_sdf_hybrid_acronym_v16e_grasped_object_offset"
            )
        elif args.main_object_mode == "ann_meta_intersection":
            args.result_subdir = (
                "results_sdf_hybrid_v8_ann_meta"
                if args.dataset == "graspnet"
                else "results_sdf_hybrid_acronym_ann_meta"
            )
        else:
            args.result_subdir = (
                "results_sdf_hybrid_v8"
                if args.dataset == "graspnet"
                else "results_sdf_hybrid_acronym"
            )

    set_seed(args.seed)
    device = torch.device(args.device)
    result_exp_root = resolve_result_exp_root(args)

    cprint(
        f"[Hybrid-AnnMeta-V16c-Offset-SelectOnly] Loading model | main_object_mode={args.main_object_mode} "
        f"| mask_diffusion={args.mask_diffusion_to_main} "
        f"| crop_object_pc={args.crop_object_pc_to_main} "
        f"| contact_stability_ckpt={args.contact_stability_ckpt or 'none'} "
        f"| result_subdir={args.result_subdir} | result_exp_root={result_exp_root}",
        "cyan",
    )

    robot_model = RobotModel(args.urdf_path, args.meta_path)
    source_robot_model = None
    use_qpos_adapter = args.qpos_adapter == "leap_to_allegro" or (
        args.qpos_adapter == "auto" and args.hand_name == "allegro_hand"
    )
    if use_qpos_adapter:
        if args.hand_name != "allegro_hand":
            raise ValueError("--qpos_adapter leap_to_allegro requires --hand_name allegro_hand")
        source_robot_model = RobotModel(args.source_urdf_path, args.source_meta_path)
        if args.allegro_transfer_profile == "auto":
            apply_allegro_transfer_profile(args)
            if str(args.allegro_embodiment_mode).lower() == "hybrid":
                cprint(
                    "[Allegro transfer] profile=auto, embodiment=hybrid: "
                    + (
                        "Leap qpos/pose retarget only (coarse-only)"
                        if int(args.allegro_coarse_only)
                        else "Leap qpos retarget + two-phase IK + gated physics + post-IK polish"
                    ),
                    "cyan",
                )
            elif str(args.allegro_embodiment_mode).lower() == "optimize":
                cprint(
                    "[Allegro transfer] profile=auto, embodiment=optimize: "
                    "canonical qpos + two-phase IK + gated physics + post-IK polish",
                    "cyan",
                )
            else:
                cprint(
                    "[Allegro transfer] profile=auto, embodiment=retarget: "
                    "Leap qpos retarget + biotac IK + relaxed physics/patch gates",
                    "cyan",
                )

    config = ckpt_to_config(
        args.ckpt_path,
        fallback_yaml=args.config_yaml or None,
    )
    
    # coarse抓取模型加载
    dex_model = GraspnessSampleWithFeature(config.model)
    dex_model.config.voxel_size = config.data.voxel_size

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    missing, unexpected = dex_model.load_state_dict(ckpt["model"], strict=False)
    cprint(
        f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}",
        "yellow" if missing or unexpected else "green",
    )
    if missing:
        for key in missing:
            cprint(f"  missing: {key}", "red")
    if unexpected:
        for key in unexpected:
            cprint(f"  unexpected: {key}", "yellow")
    cprint(
        f"[inference config] diffusion.log_prob_type="
        f"{getattr(config.model.diffusion, 'log_prob_type', None)!r}",
        "cyan",
    )
    dex_model.to(device)
    dex_model.eval()
    
    # 手模型加载（提供 SDF 查询和物理优化接口）
    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=args.urdf_path,
        meta_path=args.meta_path,
        hand_name=args.hand_name,
        device=str(device),
    )
    # Cross-embodiment: grasped_gate must run on Leap teacher tips before
    # Leap→Allegro retarget. Allegro hand_provider cannot FK Leap qpos.
    source_hand_provider = None
    if source_robot_model is not None:
        source_hand_provider = SDFAdamHandPointsProvider(
            urdf_path=args.source_urdf_path,
            meta_path=args.source_meta_path,
            hand_name="leap_hand",
            device=str(device),
        )
    # physics refiner 配置
    physics_cfg = PhysicsGuidanceV4StabilityPenGatedConfig(
        steps=args.max_iters,
        lr=args.lr,
        w_tip=float(args.physics_w_tip),
        w_mid=float(args.physics_w_mid),
        w_moment=float(args.physics_w_moment),
        w_bilateral=float(args.physics_w_bilateral),
        w_pose=float(args.physics_w_pose),
        pen_threshold=float(args.physics_pen_threshold),
        min_pen_improve=float(args.physics_min_pen_improve),
        max_trans_delta=args.max_trans_delta,
        max_q_delta=args.max_q_delta,
        pen_on_tips_only=bool(int(args.physics_pen_on_tips_only)),
    )
    # v16c 物理 refiner：仅提供稳定性优化，且采纳 physics 优化结果需满足穿模惩罚门控（初始穿模严重且最终穿模改善明显才采纳）
    physics_refiner = PhysicsGuidedPoseRefinerV4StabilityPenGated(
        hand_model=hand_provider,
        cfg=physics_cfg,
    )
    cprint("[Hybrid] physics-only refine (CSNet + DAP-Opt)", "cyan")

    contact_stability_net = None
    stability_pred = None
    stability_elong_gate = None
    
    # v16c 新增 contact-stability 评估（仅 select 最优 candidate，不替换 pose）
    if args.contact_stability_ckpt:
        contact_stability_net = load_contact_stability_for_infer(
            args.contact_stability_ckpt,
            device=device,
            contact_v2_ckpt=args.contact_v2_ckpt or None,
        )
        if args.contact_fit_steps > 0:
            contact_stability_net.contact_cfg.fit_steps = int(args.contact_fit_steps)
        stab_ckpt = torch.load(args.contact_stability_ckpt, map_location="cpu")
        stab_version = stab_ckpt.get("version", "unknown")
        contact_repr = getattr(
            contact_stability_net.contact_cfg, "target_representation", "unknown"
        )
        cprint(
            f"[Hybrid-V16c-Offset] contact-stability ON (select only, no pose replace): "
            f"{args.contact_stability_ckpt} "
            f"(fit_steps={contact_stability_net.contact_cfg.fit_steps}, "
            f"version={stab_version}, target_representation={contact_repr})",
            "cyan",
        )
        if args.contact_v2_ckpt:
            contact_ckpt = torch.load(args.contact_v2_ckpt, map_location="cpu")
            cprint(
                f"[Hybrid-V16c-Offset] contact_v2_ckpt: {args.contact_v2_ckpt} "
                f"(version={contact_ckpt.get('version', 'unknown')}, "
                f"target_representation={contact_ckpt.get('cfg', {}).get('target_representation', 'unknown')})",
                "cyan",
            )

    os.makedirs(args.logs_path, exist_ok=True)
    writer = SummaryWriter(args.logs_path)

    if args.dataset == "graspnet":
        start_idx = int(args.scene_id.split("_")[-1])
    else:
        start_idx = int(args.scene_id)
    
    # 逐场景处理，每场景逐 view 预测主物体 → 生成 grasp → physics refiner 优
    for scene_idx in trange(args.scene_num, desc="Processing scenes"):
        if args.dataset == "graspnet":
            if start_idx < 190:
                if start_idx + scene_idx >= 190:
                    break
                scene_id = f"scene_{start_idx + scene_idx:04d}"
            elif start_idx < 380:
                if start_idx + scene_idx >= 380:
                    break
                scene_id = f"scene_{start_idx + scene_idx:04d}"
            elif start_idx > 8500:
                if start_idx + scene_idx * 5 >= 9900:
                    break
                scene_id = f"scene_{start_idx + scene_idx * 5:04d}"
            else:
                scene_id = f"scene_{start_idx + scene_idx:04d}"

            load_path = os.path.join(
                args.data_root,
                "scenes",
                scene_id,
                args.camera,
                "network_input.npz",
            )

            save_path = os.path.join(
                result_exp_root,
                args.result_subdir,
                scene_id,
                "grasps.npz",
            )
        # acronym 数据集没有连续编号，需从提供的列表中取 scene_id
        else:
            all_scene_ids = args.all_scene_ids_acronym
            if all_scene_ids is None:
                raise ValueError("--all_scene_ids_acronym must be provided for acronym dataset")
            if start_idx + scene_idx >= len(all_scene_ids):
                break

            scene_id = all_scene_ids[start_idx + scene_idx].strip(",").strip("[").strip("]")
            split = scene_id.split("_")[1]

            load_path = os.path.join(
                args.data_root,
                f"acronym_test_scenes/network_input_{split}",
                scene_id,
                args.camera,
                "network_input.npz",
            )

            save_path = os.path.join(
                result_exp_root,
                args.result_subdir,
                scene_id,
                "grasps.npz",
            )

        if os.path.exists(save_path) and not args.overwrite:
            continue

        cprint(f"[Hybrid] Processing {scene_id}", "cyan")
        # 加载网络输入数据
        try:
            network_input = dict(np.load(load_path))

            pc_all = torch.tensor(network_input["pc"], dtype=torch.float)
            seg_all = torch.tensor(network_input["seg"], dtype=torch.long)
            extrinsics_all = network_input["extrinsics"]

            num_views = pc_all.shape[0]

            acronym_scene = None
            if args.dataset == "acronym":
                acronym_scene = AcronymScene(
                    scene_id=scene_id,
                    data_root=args.data_root,
                    mesh_root=args.acronym_mesh_root,
                    camera=args.camera,
                    mapping_root=args.acronym_mapping_root or None,
                )
                view_obj_poses = [
                    {
                        int(seg_label): acronym_scene.pose_obj_to_cam(
                            int(seg_label), extrinsics_all[v]
                        )
                        for seg_label in acronym_scene.seg_to_code
                    }
                    for v in range(num_views)
                ]
                meta_cls_per_view = [list(view_obj_poses[v].keys()) for v in range(num_views)]
                ann_object_ids = sorted(acronym_scene.seg_to_code)
            else:
                view_obj_poses = load_view_object_poses(
                    scene_id=scene_id,
                    camera=args.camera,
                    num_views=num_views,
                    data_root=args.data_root,
                )
                meta_cls_per_view = [
                    list(view_obj_poses[v].keys()) for v in range(num_views)
                ]
                ann_object_ids = load_graspnet_scene_object_ids(
                    scene_id, args.data_root
                )

            if args.main_object_mode == "ann_meta_intersection" and not ann_object_ids:
                cprint(
                    f"[Hybrid-AnnMeta-V9] Warning: empty ann_object_ids for {scene_id}, "
                    f"fallback={args.ann_meta_fallback}",
                    "yellow",
                )

            main_object_mode = args.main_object_mode
            if args.dataset == "acronym" and main_object_mode == "ann_meta_aligned":
                main_object_mode = "ann_meta_intersection"
            view_meta_ids, n_main_fallback = resolve_main_object_ids(
                seg_all,
                mode=main_object_mode,
                ann_object_ids=ann_object_ids,
                meta_cls_per_view=meta_cls_per_view,
                ann_meta_fallback=args.ann_meta_fallback,
            )
            if args.dataset == "acronym":
                view_ann_ids = [int(x) for x in view_meta_ids]
            else:
                view_ann_ids = [meta_id_to_ann_id(int(x)) for x in view_meta_ids]
            in_ann = sum(int(a in set(ann_object_ids)) for a in view_ann_ids)
            in_meta = sum(
                int(mid in set(meta_cls_per_view[v]))
                for v, mid in enumerate(view_meta_ids)
            )
            cprint(
                f"[Hybrid-AnnMeta-V9] {scene_id} ann={ann_object_ids} | "
                f"pred_main(ann) in ann: {in_ann}/{num_views} | "
                f"pred_main(meta) in meta: {in_meta}/{num_views} | "
                f"fallback_views={n_main_fallback}",
                "cyan",
            ) 
    
            seg_for_diffusion = seg_all
            if args.mask_diffusion_to_main and args.main_object_mode != "largest_seg":
                seg_for_diffusion, n_seg_degraded = mask_seg_to_main_objects(
                    seg_all,
                    view_meta_ids,
                    min_object_points=args.min_main_object_points,
                )
                if n_seg_degraded:
                    cprint(
                        f"[Hybrid-AnnMeta-V9] {scene_id} seg mask skipped on "
                        f"{n_seg_degraded}/{num_views} views (<{args.min_main_object_points} px)",
                        "yellow",
                    )
            # v9 核心：pred_main 掩码 + 仅 pred_main 选点，提升 diffusion 抓取质量（尤其是 contact 相关的特征和后续评估）
            with torch.no_grad():
                rotations, translations, qposs, scores = [], [], [], []
                grasp_points_list = []
                features_list = []

                for i in range(0, num_views, args.stride):
                    pc_part = pc_all[i : i + args.stride]
                    seg_part = seg_for_diffusion[i : i + args.stride]

                    data_part = get_sparse_tensor(pc_part, config.data.voxel_size)
                    data_part["seg"] = seg_part
                    data_part = {k: v.to(device) for k, v in data_part.items()}
                    # 逐 view 采样 + Top-N 筛选（得到 coarse pose）
                    result = dex_model.sample(
                        data_part,
                        args.grasp_num,
                        graspness_scale=5,
                        allow_fail=True,
                        cate=False,
                        with_score_parts=True,
                        with_point=True,
                        with_feature=True,
                    )

                    rotation = result[0].cpu()
                    translation = result[1].cpu()
                    qpos = result[2].cpu()
                    score = result[3].cpu()
                    grasp_point = result[7].cpu()
                    features = result[8].cpu()

                    batch_size = rotation.shape[0]
                    feat_dim = features.shape[-1]

                    rotations.append(rotation)
                    translations.append(translation)
                    qposs.append(qpos)
                    scores.append(score)
                    grasp_points_list.append(grasp_point.reshape(batch_size, -1, 3))
                    features_list.append(features.reshape(batch_size, -1, feat_dim))

                rotations = torch.cat(rotations, dim=0)
                translations = torch.cat(translations, dim=0)
                qposs = torch.cat(qposs, dim=0)
                scores = torch.cat(scores, dim=0)
                grasp_points = torch.cat(grasp_points_list, dim=0)
                features_all = torch.cat(features_list, dim=0)

            (
                sel_rot,
                sel_trans,
                sel_qpos,
                sel_scores,
                sel_grasp_points,
                sel_features,
            ) = select_top_n_per_view_with_feature(
                scores=scores,
                rotations=rotations,
                translations=translations,
                qposs=qposs,
                grasp_points=grasp_points,
                features=features_all,
                top_n=args.top_n,
            )

            top_n = sel_trans.shape[1]
            feat_dim = sel_features.shape[-1]
            # coarse抓取（DexGraspNet teacher = Leap 坐标系）
            coarse_rot = sel_rot.reshape(-1, 3, 3).to(device)
            coarse_trans = sel_trans.reshape(-1, 3).to(device)
            coarse_qpos = sel_qpos.reshape(-1, sel_qpos.shape[-1]).to(device)
            teacher_rot = coarse_rot.clone()
            teacher_trans = coarse_trans.clone()
            teacher_qpos = coarse_qpos.clone()
            # 供contact dynamics 和 guided 分支使用的特征和种子点
            top_features = sel_features.reshape(-1, feat_dim).to(device)
            top_seed_points = sel_grasp_points.reshape(-1, 3).to(device)

            cand_meta_ids = np.repeat(view_meta_ids, top_n)
            if args.dataset == "acronym":
                cand_ann_ids = cand_meta_ids.astype(np.int64, copy=True)
            else:
                cand_ann_ids = np.array(
                    [meta_id_to_ann_id(int(x)) for x in cand_meta_ids], dtype=np.int64
                )
            cand_view_ids = np.repeat(np.arange(num_views), top_n)
            contact_meta_ids = cand_meta_ids.astype(np.int64, copy=True)
            grasped_meta_ids = cand_meta_ids.astype(np.int64, copy=True)
            tip_to_pred_main = np.full_like(cand_meta_ids, np.inf, dtype=np.float32)
            tip_to_grasped = np.full_like(cand_meta_ids, np.inf, dtype=np.float32)
            contact_object_switched = np.zeros_like(cand_meta_ids, dtype=np.int64)
            grasped_object_points = np.zeros_like(cand_meta_ids, dtype=np.int64)

            # IMPORTANT (cross-embodiment): run grasped_gate on the Leap teacher
            # pose BEFORE Leap→Allegro retarget. After retarget, Allegro tips are
            # often > contact_grasped_max_tip_dist (default 6cm) from the true
            # object, so the gate never switches and contact/IK lock onto the
            # wrong pred_main object (observed: hand near obj 048, contacts on 058).
            if args.contact_object_mode == "grasped_gate":
                gate_hand = (
                    source_hand_provider
                    if source_hand_provider is not None
                    else hand_provider
                )
                (
                    contact_meta_ids,
                    grasped_meta_ids,
                    tip_to_pred_main,
                    tip_to_grasped,
                    contact_object_switched,
                    grasped_object_points,
                ) = estimate_grasped_object_meta_ids(
                    gate_hand,
                    coarse_trans,
                    coarse_rot,
                    coarse_qpos,
                    pc_all,
                    seg_all,
                    cand_view_ids,
                    cand_meta_ids,
                    args,
                )
                cprint(
                    "[Hybrid-V16e] contact object switched "
                    f"{int(contact_object_switched.sum())}/{len(contact_object_switched)} "
                    f"(mode={args.contact_object_mode}, "
                    f"tip_pred_mean={float(np.nanmean(tip_to_pred_main)):.4f}, "
                    f"tip_contact_mean={float(np.nanmean(tip_to_grasped)):.4f}"
                    f"{', pre_retarget=1' if source_hand_provider is not None else ''})",
                    "cyan",
                )

            use_frozen_allegro_coarse = False
            frozen_coarse_world = None
            embodiment_mode = str(args.allegro_embodiment_mode).lower()
            if source_robot_model is not None:
                if embodiment_mode == "optimize":
                    if args.pose_adapter in ("auto", "leap_to_allegro"):
                        coarse_trans, coarse_rot = retarget_leap_pose_to_allegro(
                            coarse_trans,
                            coarse_rot,
                            translation_mode=args.allegro_pose_translation_mode,
                            rotation_mode=args.allegro_pose_rotation_mode,
                            target_robot=robot_model,
                        )
                    width_mapper_meta = os.path.join(
                        os.path.dirname(args.meta_path), "width_mapper_meta.yaml"
                    )
                    coarse_qpos = build_allegro_canonical_qpos(
                        robot_model,
                        coarse_qpos.shape[0],
                        device=coarse_qpos.device,
                        dtype=coarse_qpos.dtype,
                        width_mapper_meta_path=width_mapper_meta,
                    )
                    cprint(
                        "[Allegro] embodiment=optimize: canonical qpos + wrist pose "
                        "from teacher (no Leap joint retarget)",
                        "cyan",
                    )
                elif embodiment_mode == "hybrid":
                    coarse_qpos = retarget_leap_to_allegro(
                        coarse_qpos, source_robot_model, robot_model)
                    if args.pose_adapter in ("auto", "leap_to_allegro"):
                        coarse_trans, coarse_rot = retarget_leap_pose_to_allegro(
                            coarse_trans,
                            coarse_rot,
                            translation_mode=args.allegro_pose_translation_mode,
                            rotation_mode=args.allegro_pose_rotation_mode,
                            target_robot=robot_model,
                        )
                    cprint(
                        "[Allegro] embodiment=hybrid: "
                        + (
                            "Leap qpos/pose retarget only (coarse-only)"
                            if int(args.allegro_coarse_only)
                            else "Leap qpos/pose retarget + contact IK"
                        ),
                        "cyan",
                    )
                else:
                    coarse_qpos = retarget_leap_to_allegro(
                        coarse_qpos, source_robot_model, robot_model)
                    if args.pose_adapter in ("auto", "leap_to_allegro"):
                        coarse_trans, coarse_rot = retarget_leap_pose_to_allegro(
                            coarse_trans,
                            coarse_rot,
                            translation_mode=args.allegro_pose_translation_mode,
                            rotation_mode=args.allegro_pose_rotation_mode,
                            target_robot=robot_model,
                        )
                    cprint(
                        "[Allegro] embodiment=retarget: Leap qpos/pose mapped to Allegro",
                        "cyan",
                    )

            # Stage-2 method: replace coarse with frozen baseline71 (world→cam),
            # then run contact select + pen-gated physics without rewriting the
            # open-hand approach pose via curl/opposition priors.
            allegro_init_npz = str(
                getattr(args, "allegro_init_grasps_npz", "") or ""
            ).strip()
            if source_robot_model is not None and bool(allegro_init_npz):
                if not os.path.isfile(allegro_init_npz):
                    raise FileNotFoundError(
                        f"--allegro_init_grasps_npz not found: {allegro_init_npz}"
                    )
                init_g = np.load(allegro_init_npz, allow_pickle=True)
                batch = int(coarse_trans.shape[0])
                if len(init_g["translation"]) < batch:
                    raise ValueError(
                        f"Init grasps N={len(init_g['translation'])} < batch={batch}"
                    )
                # Align rows to current cand_view_ids when possible.
                init_idx = np.arange(batch, dtype=np.int64)
                if "valid_view_indices" in init_g.files:
                    init_views = np.asarray(init_g["valid_view_indices"]).astype(
                        np.int64
                    )
                    view_to_row = {int(v): i for i, v in enumerate(init_views)}
                    missing_views = [
                        int(v) for v in cand_view_ids if int(v) not in view_to_row
                    ]
                    if missing_views:
                        raise KeyError(
                            "Init grasps missing views for frozen inject: "
                            f"{missing_views[:8]}"
                        )
                    init_idx = np.asarray(
                        [view_to_row[int(v)] for v in cand_view_ids], dtype=np.int64
                    )
                tw = torch.tensor(
                    init_g["translation"][init_idx],
                    device=device,
                    dtype=torch.float32,
                )
                rw = torch.tensor(
                    init_g["rotation"][init_idx],
                    device=device,
                    dtype=torch.float32,
                )
                joint_names = list(robot_model.movable_joint_names)
                missing = [n for n in joint_names if n not in init_g.files]
                if missing:
                    raise KeyError(
                        f"Init grasps missing joints: {missing[:8]}"
                    )
                qw = torch.stack(
                    [
                        torch.tensor(
                            init_g[n][init_idx],
                            device=device,
                            dtype=torch.float32,
                        )
                        for n in joint_names
                    ],
                    dim=1,
                )
                frozen_coarse_world = {
                    "translation": tw.detach().cpu().numpy().astype(np.float32),
                    "rotation": rw.detach().cpu().numpy().astype(np.float32),
                    "qpos": {
                        n: np.asarray(init_g[n][init_idx], dtype=np.float32)
                        for n in joint_names
                    },
                }
                E = torch.tensor(
                    extrinsics_all[cand_view_ids],
                    device=device,
                    dtype=torch.float32,
                )
                if E.ndim != 3 or E.shape[-2:] != (4, 4):
                    raise ValueError(
                        f"Unexpected extrinsics shape for init inject: {tuple(E.shape)}"
                    )
                R_e = E[:, :3, :3]
                t_e = E[:, :3, 3]
                coarse_trans = torch.einsum(
                    "bij,bj->bi", R_e.transpose(1, 2), tw - t_e
                )
                coarse_rot = torch.einsum(
                    "bij,bjk->bik", R_e.transpose(1, 2), rw
                )
                coarse_qpos = qw
                use_frozen_allegro_coarse = True
                # Keep contact IK near the frozen coarse grasp.
                args.allegro_contact_ik_pose_reg = max(
                    float(args.allegro_contact_ik_pose_reg), 0.05
                )
                args.allegro_contact_ik_max_trans_delta = min(
                    float(args.allegro_contact_ik_max_trans_delta), 0.04
                )
                args.allegro_contact_ik_max_rot_delta = min(
                    float(args.allegro_contact_ik_max_rot_delta), 0.35
                )
                cprint(
                    "[Allegro stage2-method] loaded frozen coarse from "
                    f"{allegro_init_npz} (N={batch}); open-hand priors skipped; "
                    "contact+physics refine grasp only",
                    "cyan",
                )

            if (
                source_robot_model is not None
                and not use_frozen_allegro_coarse
                and embodiment_mode in ("hybrid", "retarget")
                and str(args.allegro_thumb_retarget_mode).lower() == "shape_prior"
            ):
                thumb_names = [
                    "thumb_joint_0",
                    "thumb_joint_1",
                    "thumb_joint_2",
                    "thumb_joint_3",
                ]
                thumb_indices = [
                    robot_model.movable_joint_names.index(name)
                    for name in thumb_names
                ]
                thumb_before = coarse_qpos[:, thumb_indices].mean(dim=0)
                coarse_qpos = apply_allegro_thumb_shape_prior(
                    coarse_qpos,
                    robot_model,
                )
                coarse_qpos = apply_allegro_finger_curl_prior(
                    coarse_qpos,
                    robot_model,
                )
                thumb_after = coarse_qpos[:, thumb_indices].mean(dim=0)
                cprint(
                    "[Allegro thumb shape prior + finger curl] "
                    f"mode=shape_prior mean_thumb_qpos "
                    f"{thumb_before.detach().cpu().tolist()} -> "
                    f"{thumb_after.detach().cpu().tolist()}",
                    "cyan",
                )

            if (
                source_robot_model is not None
                and not use_frozen_allegro_coarse
                and int(args.allegro_coarse_geometry_align)
                and embodiment_mode in ("optimize", "hybrid", "retarget")
            ):
                coarse_trans, coarse_rot = align_allegro_pose_to_teacher_fingertips(
                    source_translation=teacher_trans,
                    source_rotation=teacher_rot,
                    source_qpos=teacher_qpos,
                    target_translation=coarse_trans,
                    target_rotation=coarse_rot,
                    target_qpos=coarse_qpos,
                    source_robot=source_robot_model,
                    target_robot=robot_model,
                )
                cprint(
                    "[Allegro coarse geometry align] "
                    "fingertip rigid alignment enabled",
                    "cyan",
                )

            if (
                source_robot_model is not None
                and not use_frozen_allegro_coarse
                and int(args.allegro_coarse_wrist_align)
                and embodiment_mode in ("optimize", "hybrid", "retarget")
            ):
                coarse_trans, wrist_correction = (
                    align_allegro_wrist_to_teacher_fingertips(
                        source_translation=teacher_trans,
                        source_rotation=teacher_rot,
                        source_qpos=teacher_qpos,
                        target_translation=coarse_trans,
                        target_rotation=coarse_rot,
                        target_qpos=coarse_qpos,
                        source_robot=source_robot_model,
                        target_robot=robot_model,
                        alpha=float(args.allegro_coarse_wrist_align_alpha),
                    )
                )
                cprint(
                    "[Allegro coarse wrist align] "
                    f"alpha={float(args.allegro_coarse_wrist_align_alpha):.3f} "
                    f"correction_mean={wrist_correction.mean(dim=0).tolist()} "
                    f"correction_norm_mean={wrist_correction.norm(dim=-1).mean().item():.4f}m",
                    "cyan",
                )

            if (
                source_robot_model is not None
                and not use_frozen_allegro_coarse
                and int(args.allegro_thumb_teacher_fit)
                and str(args.allegro_thumb_retarget_mode).lower() == "shape_prior"
                and embodiment_mode in ("hybrid", "retarget")
            ):
                coarse_qpos, thumb_fit_mse = fit_allegro_thumb_to_teacher_tip(
                    qpos=coarse_qpos,
                    translation=coarse_trans,
                    rotation=coarse_rot,
                    source_qpos=teacher_qpos,
                    source_translation=teacher_trans,
                    source_rotation=teacher_rot,
                    source_robot=source_robot_model,
                    target_robot=robot_model,
                    steps=int(args.allegro_thumb_teacher_fit_steps),
                    lr=float(args.allegro_thumb_teacher_fit_lr),
                    pose_reg=float(args.allegro_thumb_teacher_fit_reg),
                )
                cprint(
                    "[Allegro thumb teacher fit] "
                    f"steps={int(args.allegro_thumb_teacher_fit_steps)} "
                    f"tip_mse_mean={thumb_fit_mse.mean().item():.8f} "
                    f"tip_rmse={thumb_fit_mse.mean().sqrt().item():.4f}m",
                    "cyan",
                )

            if (
                source_robot_model is not None
                and not use_frozen_allegro_coarse
                and int(getattr(args, "allegro_thumb_opposition", 0))
                and str(args.allegro_thumb_retarget_mode).lower() == "shape_prior"
                and embodiment_mode in ("hybrid", "retarget")
            ):
                coarse_qpos, thumb_align = fit_allegro_thumb_finger_opposition(
                    coarse_qpos,
                    robot_model,
                    steps=int(getattr(args, "allegro_thumb_opposition_steps", 80)),
                )
                cprint(
                    "[Allegro thumb opposition] "
                    f"steps={int(getattr(args, 'allegro_thumb_opposition_steps', 80))} "
                    f"thumb_n·to_index mean={thumb_align.mean().item():+.3f}",
                    "cyan",
                )

            if int(args.allegro_coarse_only):
                object_pc = None
                sdf_grid = None
                sdf_origin = None
                sdf_voxel_size = None
                T_cam_to_obj = None
            else:
                max_obj_points = int(args.max_obj_points)
                if args.crop_object_pc_to_main and args.main_object_mode != "largest_seg":
                    object_pc = build_object_pc_for_grasps(
                        pc_all,
                        seg_all,
                        cand_view_ids,
                        contact_meta_ids,
                        max_obj_points,
                        device,
                    )
                else:
                    pc_expanded = pc_all.unsqueeze(1).expand(-1, top_n, -1, -1)
                    object_pc = pc_expanded.reshape(-1, pc_all.shape[1], 3).to(device)
                    if object_pc.shape[1] > max_obj_points:
                        idx = torch.randperm(object_pc.shape[1], device=object_pc.device)[
                            :max_obj_points
                        ]
                        object_pc = object_pc[:, idx]

                sdf_cache = {}
                sdf_grids = []
                sdf_origins = []
                sdf_voxel_sizes = []
                T_cam_to_objs = []

                for idx_obj, (view_id, meta_id) in enumerate(zip(cand_view_ids, contact_meta_ids)):
                    meta_id = int(meta_id)
                    view_id = int(view_id)
                    if meta_id not in view_obj_poses[view_id]:
                        meta_id = int(cand_meta_ids[idx_obj])
                        contact_meta_ids[idx_obj] = meta_id
                        contact_object_switched[idx_obj] = 0
                    if args.dataset == "acronym":
                        mesh_obj_id = acronym_scene.object_code(meta_id)
                        if mesh_obj_id not in sdf_cache:
                            sdf_cache[mesh_obj_id] = load_or_build_cached_sdf(
                                mesh_path=acronym_scene.mesh_path(meta_id),
                                sdf_path=acronym_scene.sdf_path(
                                    meta_id, args.acronym_sdf_root, args.sdf_grid_size
                                ),
                                grid_size=args.sdf_grid_size,
                                device=str(device),
                            )
                    else:
                        mesh_obj_id = meta_id_to_ann_id(meta_id)
                        if mesh_obj_id not in sdf_cache:
                            sdf_cache[mesh_obj_id] = load_cached_sdf(
                                mesh_root=args.mesh_root,
                                obj_id=mesh_obj_id,
                                grid_size=args.sdf_grid_size,
                                device=str(device),
                            )

                    if meta_id not in view_obj_poses[view_id]:
                        visible = sorted(view_obj_poses[view_id].keys())
                        raise KeyError(
                            f"seg_label={meta_id} (object_id={mesh_obj_id}) not found in scene "
                            f"for scene={scene_id}, view={view_id} "
                            f"(visible meta ids={visible})"
                        )

                    T_obj_to_cam = view_obj_poses[view_id][meta_id]
                    T_cam_to_obj = np.linalg.inv(T_obj_to_cam).astype(np.float32)

                    sdf_grid, sdf_origin, sdf_voxel_size = sdf_cache[mesh_obj_id]

                    sdf_grids.append(sdf_grid)
                    sdf_origins.append(sdf_origin)
                    sdf_voxel_sizes.append(sdf_voxel_size)
                    T_cam_to_objs.append(torch.from_numpy(T_cam_to_obj).to(device))

                sdf_grid = torch.stack(sdf_grids, dim=0).to(device)
                sdf_origin = torch.stack(sdf_origins, dim=0).to(device)
                sdf_voxel_size = torch.stack(sdf_voxel_sizes, dim=0).to(device)
                T_cam_to_obj = torch.stack(T_cam_to_objs, dim=0).to(device)
            
            use_contact_diffusion = torch.zeros(
                coarse_trans.shape[0], device=device, dtype=torch.float32
            )
            v4_target_contacts = None
            
            '''
                Contact fit accept:
                对每个 grasp，先用 contact 网络预测接触点并做 pose 优化
                优化结果只有满足条件才替换 teacher pose，否则回退到 DexGraspNet teacher 原 pose
                和后面的physics accept（use_physics_refined） 是两套独立逻辑
                contact fit 在前，pen-gated physics 在后。
            
            '''
            if contact_stability_net is not None:
                allegro_rerank = None
                if (
                    source_robot_model is not None
                    and int(args.allegro_contact_rerank)
                ):
                    allegro_rerank = {
                        "enabled": True,
                        "robot_model": robot_model,
                        "qpos": coarse_qpos,
                        "trans": coarse_trans,
                        "rot": coarse_rot,
                        "ik_steps": int(args.allegro_contact_rerank_ik_steps),
                        "w_ik": float(args.allegro_contact_rerank_w_ik),
                        "w_stab": float(args.allegro_contact_rerank_w_stab),
                        "lr": float(args.allegro_contact_ik_lr),
                        "max_trans_delta": float(
                            args.allegro_contact_ik_max_trans_delta
                        ),
                        "max_rot_delta": float(args.allegro_contact_ik_max_rot_delta),
                        "min_ik_rel_improve": float(
                            args.allegro_contact_rerank_min_ik_rel_improve
                        ),
                        "max_stab_drop": float(args.allegro_contact_rerank_max_stab_drop),
                        "coordination_reg_weight": float(
                            args.allegro_contact_ik_coordination_weight
                        ),
                        "canonical_abduction_reg_weight": float(
                            args.allegro_contact_ik_canonical_abduction_weight
                        ),
                    }
                cprint(
                    f"[Hybrid-V16c-Offset] Contact-stability select {coarse_trans.shape[0]} grasps "
                    f"(K={args.contact_num_samples}, skip_pose_replace"
                    f"{', allegro_rerank=1' if allegro_rerank else ''})...",
                    "cyan",
                )
                coarse_trans, coarse_rot, coarse_qpos, cs_log = (
                    refine_poses_with_contact_stability(
                        contact_stability_net,
                        hand_provider,
                        top_features,
                        top_seed_points,
                        coarse_trans,
                        coarse_rot,
                        coarse_qpos,
                        object_pc,
                        infer_cfg=ContactStabilityInferConfig(
                            num_contact_samples=int(args.contact_num_samples),
                            fit_max_trans_delta=float(args.contact_fit_max_trans_delta),
                            fit_max_q_delta=float(args.contact_fit_max_q_delta),
                            contact_obj_dist_weight=float(args.contact_obj_dist_weight),
                            contact_obj_dist_clip=float(args.contact_obj_dist_clip),
                            project_contacts_to_object=bool(int(args.contact_project_to_object)),
                            skip_pose_replace=True,
                            allegro_contact_rerank=bool(allegro_rerank),
                            allegro_contact_rerank_ik_steps=int(
                                args.allegro_contact_rerank_ik_steps
                            ),
                            allegro_contact_rerank_w_ik=float(
                                args.allegro_contact_rerank_w_ik
                            ),
                            allegro_contact_rerank_w_stab=float(
                                args.allegro_contact_rerank_w_stab
                            ),
                        ),
                        sdf_grid=sdf_grid,
                        sdf_origin=sdf_origin,
                        sdf_voxel_size=sdf_voxel_size,
                        T_cam_to_obj=T_cam_to_obj,
                        allegro_rerank=allegro_rerank,
                    )
                )
                if allegro_rerank and "allegro_contact_rerank_switched" in cs_log:
                    proposed = cs_log.get("allegro_contact_rerank_proposed")
                    proposed_mean = (
                        proposed.float().mean().item() if proposed is not None else 0.0
                    )
                    cprint(
                        "[Allegro contact rerank] proposed "
                        f"{proposed_mean:.3f} switched "
                        f"{cs_log['allegro_contact_rerank_switched'].float().mean().item():.3f} "
                        f"quick_ik_mse mean="
                        f"{cs_log['allegro_contact_rerank_ik_mse'].mean().item():.6f}",
                        "cyan",
                    )
                use_contact_diffusion[:] = 1.0
                stability_pred = cs_log.get("stability_pred")
                stability_elong_gate = cs_log.get("elongated_gate")
                v4_target_contacts = cs_log.get("target_contacts")
                allegro_contact_ik_mse = None
                if (
                    source_robot_model is not None
                    and v4_target_contacts is not None
                    and int(args.allegro_contact_ik_steps) > 0
                ):
                    allegro_ik_hand_model = (
                        hand_provider
                        if args.allegro_contact_ik_tip_mode == "representative"
                        else None
                    )
                    allegro_finger_weights = None
                    if args.allegro_contact_ik_finger_weights:
                        allegro_finger_weights = [
                            float(x)
                            for x in str(args.allegro_contact_ik_finger_weights).split(",")
                        ]
                    allegro_ik_fn = fit_allegro_qpos_to_contacts
                    allegro_ik_kwargs = {}
                    if embodiment_mode in ("optimize", "hybrid"):
                        # Frozen stage-2: keep wrist near baseline71; do not
                        # warm-start toward contacts (that rewrites approach).
                        if not use_frozen_allegro_coarse:
                            coarse_trans, coarse_rot = (
                                warm_start_allegro_wrist_toward_contacts(
                                    coarse_trans,
                                    coarse_rot,
                                    v4_target_contacts,
                                )
                            )
                        allegro_ik_fn = fit_allegro_qpos_to_contacts_two_phase
                    (
                        coarse_qpos,
                        coarse_trans,
                        coarse_rot,
                        allegro_contact_ik_mse,
                    ) = allegro_ik_fn(
                        coarse_qpos,
                        coarse_trans,
                        coarse_rot,
                        v4_target_contacts,
                        robot_model,
                        hand_model=allegro_ik_hand_model,
                        object_pc=object_pc,
                        steps=int(args.allegro_contact_ik_steps),
                        lr=float(args.allegro_contact_ik_lr),
                        pose_reg=float(args.allegro_contact_ik_pose_reg),
                        max_trans_delta=float(args.allegro_contact_ik_max_trans_delta),
                        max_rot_delta=float(args.allegro_contact_ik_max_rot_delta),
                        lock_abduction=bool(int(args.allegro_contact_ik_lock_abduction)),
                        finger_weights=allegro_finger_weights,
                        abduction_reg_weight=float(
                            args.allegro_contact_ik_abduction_reg_weight
                        ),
                        coordination_reg_weight=float(
                            args.allegro_contact_ik_coordination_weight
                        ),
                        canonical_abduction_reg_weight=float(
                            args.allegro_contact_ik_canonical_abduction_weight
                        ),
                        coordination_q2_ratio=float(
                            args.allegro_contact_ik_coordination_q2_ratio
                        ),
                        coordination_q3_ratio=float(
                            args.allegro_contact_ik_coordination_q3_ratio
                        ),
                        coordination_abduction_soft_limit=float(
                            args.allegro_contact_ik_coordination_abduction_soft_limit
                        ),
                        **allegro_ik_kwargs,
                    )
                    cprint(
                        f"[Allegro IK] mode={args.allegro_embodiment_mode} "
                        f"tip_mode={args.allegro_contact_ik_tip_mode} "
                        f"lock_abd={int(args.allegro_contact_ik_lock_abduction)} "
                        f"pose_reg={float(args.allegro_contact_ik_pose_reg):.4f} "
                        f"coord_w={float(args.allegro_contact_ik_coordination_weight):.4f} "
                        f"abd_can_w={float(args.allegro_contact_ik_canonical_abduction_weight):.4f} "
                        f"steps={int(args.allegro_contact_ik_steps)} "
                        f"contact_mse mean={allegro_contact_ik_mse.mean().item():.6f} "
                        f"max={allegro_contact_ik_mse.max().item():.6f} "
                        f"rmse_per_tip={(allegro_contact_ik_mse.mean().sqrt() * (3.0 ** 0.5)).item():.4f}m",
                        "cyan",
                    )
                cprint(
                    f"[Hybrid-V16c-Offset] contact teacher_mse mean={cs_log['fit_loss_init'].mean().item():.6f} "
                    f"fit_accept mean={float(cs_log['use_contact_fit'].mean()):.3f} "
                    f"stab_score mean={float(cs_log['stability_score'].mean()) if 'stability_score' in cs_log else 0.0:.4f} "
                    f"raw_stab mean={float(cs_log['contact_raw_stability_score'].mean()) if 'contact_raw_stability_score' in cs_log else 0.0:.4f} "
                    f"contact_obj_dist mean={float(cs_log['contact_obj_dist'].mean()) if 'contact_obj_dist' in cs_log else 0.0:.4f} "
                    f"projected_obj_dist mean={float(cs_log['contact_projected_obj_dist'].mean()) if 'contact_projected_obj_dist' in cs_log else 0.0:.4f} "
                    f"elong_gate mean={float(stability_elong_gate.mean()) if stability_elong_gate is not None else 0.0:.4f}",
                    "cyan",
                )

            if int(args.allegro_coarse_only):
                final_trans = coarse_trans
                final_rot = coarse_rot
                final_qpos = coarse_qpos
                use_physics = torch.zeros(
                    len(final_trans), device=device, dtype=torch.float32
                )
                use_physics_base = use_physics.clone()
                raw_refine_out = None
                refine_out = None
                cprint(
                    "[Allegro] coarse-only: saved Leap->Allegro retargeted "
                    "teacher grasp; contact/physics stages skipped",
                    "cyan",
                )
            else:
                cprint(
                    f"[Hybrid-V16c-Offset] Stability Physics (pen-gated) refine {coarse_trans.shape[0]} grasps "
                    f"(max_iters={args.max_iters}, pen_thresh={physics_cfg.pen_threshold}, "
                    f"w_tip={physics_cfg.w_tip})...",
                    "cyan",
                )
                # v16c 物理 refiner：仅稳定性优化，且采纳 physics 优化结果需满足穿模惩罚门控（初始穿模严重且最终穿模明显才采纳）
                refine_out = physics_refiner(
                    init_trans=coarse_trans,
                    init_rot=coarse_rot,
                    init_qpos=coarse_qpos,
                    object_pc=object_pc,
                    sdf_grid=sdf_grid,
                    sdf_origin=sdf_origin,
                    sdf_voxel_size=sdf_voxel_size,
                    T_cam_to_obj=T_cam_to_obj,
                    target_contacts=v4_target_contacts,
                )

                raw_refine_out = refine_out
                if int(args.pointnext_strict_physics_gate):
                    refine_out = apply_pointnext_strict_physics_gate(refine_out, args)
                    use_physics_base = refine_out["use_refined_base"].bool()
                    use_physics = refine_out["use_refined"].bool()
                    cprint(
                        f"[Hybrid-V16e-PointNeXt] strict physics gate: "
                        f"base={use_physics_base.float().mean().item():.3f} "
                        f"strict={use_physics.float().mean().item():.3f} "
                        f"contact_ok={refine_out['strict_contact_ok'].float().mean().item():.3f} "
                        f"pen_ok={refine_out['strict_pen_ok'].float().mean().item():.3f} "
                        f"stab_ok={refine_out['strict_stability_ok'].float().mean().item():.3f} "
                        f"pose_ok={refine_out['strict_pose_ok'].float().mean().item():.3f}",
                        "cyan",
                    )
                else:
                    use_physics = refine_out["use_refined"].bool()
                    use_physics_base = use_physics
                cprint(
                    f"[Hybrid-V16c-Offset] physics use_refined: {use_physics.float().mean().item():.3f} "
                    f"({use_physics.sum().item()}/{len(use_physics)})",
                    "cyan",
                )
                print("physics init_E_pen mean:", refine_out["init_E_pen"].mean().item())
                print("physics final_E_pen mean:", refine_out["final_E_pen"].mean().item())
                print("physics init_E_total mean:", refine_out["init_E_total"].mean().item())
                print("physics final_E_total mean:", refine_out["final_E_total"].mean().item())
                if "init_E_stab" in refine_out:
                    print(
                        "physics init_E_stab mean:",
                        refine_out["init_E_stab"].mean().item(),
                    )
                    print(
                        "physics final_E_stab mean:",
                        refine_out["final_E_stab"].mean().item(),
                    )

                final_trans = refine_out["selected_trans"]
                final_rot = refine_out["selected_rot"]
                final_qpos = refine_out["selected_qpos"]
            if (
                source_robot_model is not None
                and v4_target_contacts is not None
                and int(args.allegro_post_physics_ik) > 0
            ):
                post_finger_weights = None
                if args.allegro_contact_ik_finger_weights:
                    post_finger_weights = [
                        float(x)
                        for x in str(args.allegro_contact_ik_finger_weights).split(",")
                    ]
                final_qpos, final_trans, final_rot, post_ik_mse = (
                    fit_allegro_qpos_to_contacts(
                        final_qpos,
                        final_trans,
                        final_rot,
                        v4_target_contacts,
                        robot_model,
                        hand_model=None,
                        object_pc=object_pc,
                        steps=int(args.allegro_post_physics_ik_steps),
                        lr=float(args.allegro_contact_ik_lr),
                        pose_reg=max(float(args.allegro_contact_ik_pose_reg) * 0.5, 0.005),
                        max_trans_delta=float(args.allegro_contact_ik_max_trans_delta),
                        max_rot_delta=float(args.allegro_contact_ik_max_rot_delta),
                        lock_abduction=bool(int(args.allegro_contact_ik_lock_abduction)),
                        finger_weights=post_finger_weights,
                        abduction_reg_weight=float(
                            args.allegro_contact_ik_abduction_reg_weight
                        ),
                        coordination_reg_weight=float(
                            args.allegro_contact_ik_coordination_weight
                        ),
                        canonical_abduction_reg_weight=float(
                            args.allegro_contact_ik_canonical_abduction_weight
                        ),
                        coordination_q2_ratio=float(
                            args.allegro_contact_ik_coordination_q2_ratio
                        ),
                        coordination_q3_ratio=float(
                            args.allegro_contact_ik_coordination_q3_ratio
                        ),
                        coordination_abduction_soft_limit=float(
                            args.allegro_contact_ik_coordination_abduction_soft_limit
                        ),
                    )
                )
                cprint(
                    f"[Allegro post-physics IK] steps={int(args.allegro_post_physics_ik_steps)} "
                    f"contact_mse mean={post_ik_mse.mean().item():.6f} "
                    f"rmse_per_tip={(post_ik_mse.mean().sqrt() * (3.0 ** 0.5)).item():.4f}m",
                    "cyan",
                )

            opt_translations = final_trans.detach().cpu().numpy()
            opt_rotations = final_rot.detach().cpu().numpy()
            opt_qpos = final_qpos.detach().cpu().numpy()

            view_indices = np.arange(num_views).repeat(top_n)
            valid_extrinsics = extrinsics_all[view_indices]

            world_rotations, world_translations = transform_pose_to_world(
                opt_rotations,
                opt_translations,
                valid_extrinsics,
            )

            grasps = {
                "rotation": world_rotations,
                "translation": world_translations,
                "valid_view_indices": view_indices,
                "pred_main_object_id": cand_ann_ids,
                "pred_main_meta_id": cand_meta_ids.astype(np.int64),
                "contact_object_id": (
                    np.array(
                        [int(x) for x in contact_meta_ids], dtype=np.int64
                    )
                    if args.dataset == "acronym"
                    else np.array(
                        [meta_id_to_ann_id(int(x)) for x in contact_meta_ids], dtype=np.int64
                    )
                ),
                "contact_object_meta_id": contact_meta_ids.astype(np.int64),
                "grasped_object_meta_id": grasped_meta_ids.astype(np.int64),
                "contact_object_switched": contact_object_switched.astype(np.int64),
                "tip_to_pred_main": tip_to_pred_main.astype(np.float32),
                "tip_to_grasped": tip_to_grasped.astype(np.float32),
                "grasped_object_points": grasped_object_points.astype(np.int64),
                "score": sel_scores.reshape(-1).cpu().numpy(),
                "use_physics_refined": use_physics.detach().cpu().numpy(),
                "use_physics_refined_base": use_physics_base.detach().cpu().numpy(),
            }
            if args.dataset == "acronym":
                grasps["pred_main_object_code"] = np.asarray(
                    [acronym_scene.object_code(int(x)) for x in cand_meta_ids], dtype=object
                )
                grasps["contact_object_code"] = np.asarray(
                    [acronym_scene.object_code(int(x)) for x in contact_meta_ids], dtype=object
                )
                grasps["grasped_object_code"] = np.asarray(
                    [acronym_scene.object_code(int(x)) for x in grasped_meta_ids], dtype=object
                )
                grasps["pred_main_seg_label"] = cand_meta_ids.astype(np.int64)
                grasps["contact_object_seg_label"] = contact_meta_ids.astype(np.int64)
                grasps["grasped_object_seg_label"] = grasped_meta_ids.astype(np.int64)
            if int(args.pointnext_strict_physics_gate):
                grasps["pointnext_strict_physics_gate"] = np.ones(
                    len(use_physics), dtype=np.int64
                )
                grasps["strict_contact_ok"] = (
                    refine_out["strict_contact_ok"].detach().cpu().numpy()
                )
                grasps["strict_pen_ok"] = (
                    refine_out["strict_pen_ok"].detach().cpu().numpy()
                )
                grasps["strict_stability_ok"] = (
                    refine_out["strict_stability_ok"].detach().cpu().numpy()
                )
                grasps["strict_pose_ok"] = (
                    refine_out["strict_pose_ok"].detach().cpu().numpy()
                )
                grasps["physics_raw_use_refined"] = (
                    raw_refine_out["use_refined"].detach().cpu().numpy()
                )
            if source_robot_model is not None:
                # Persist the actual target kinematic asset.  Allegro qpos
                # names are shared by the simplified and ROS-V5 models, but
                # their joint axes/origins are not interchangeable.
                grasps["allegro_target_urdf_path"] = np.asarray(
                    [str(args.urdf_path)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_target_meta_path"] = np.asarray(
                    [str(args.meta_path)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_embodiment_mode"] = np.asarray(
                    [str(args.allegro_embodiment_mode)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_pose_translation_mode"] = np.asarray(
                    [str(args.allegro_pose_translation_mode)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_pose_rotation_mode"] = np.asarray(
                    [str(args.allegro_pose_rotation_mode)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_thumb_retarget_mode"] = np.asarray(
                    [str(args.allegro_thumb_retarget_mode)] * len(world_translations),
                    dtype=object,
                )
                grasps["allegro_thumb_teacher_fit"] = np.full(
                    len(world_translations),
                    int(args.allegro_thumb_teacher_fit),
                    dtype=np.int64,
                )
                grasps["allegro_thumb_teacher_fit_steps"] = np.full(
                    len(world_translations),
                    int(args.allegro_thumb_teacher_fit_steps),
                    dtype=np.int64,
                )
                grasps["allegro_coarse_wrist_align"] = np.full(
                    len(world_translations),
                    int(args.allegro_coarse_wrist_align),
                    dtype=np.int64,
                )
                grasps["allegro_coarse_wrist_align_alpha"] = np.full(
                    len(world_translations),
                    float(args.allegro_coarse_wrist_align_alpha),
                    dtype=np.float32,
                )
                grasps["allegro_coarse_geometry_align"] = np.full(
                    len(world_translations),
                    int(args.allegro_coarse_geometry_align),
                    dtype=np.int64,
                )
            if contact_stability_net is not None:
                grasps["use_contact_diffusion"] = use_contact_diffusion.detach().cpu().numpy()
                grasps["use_contact_fit"] = cs_log["use_contact_fit"].detach().cpu().numpy()
                if stability_pred is not None:
                    grasps["stability_pred"] = stability_pred.detach().cpu().numpy()
                if "stability_score" in cs_log:
                    grasps["stability_score"] = cs_log["stability_score"].detach().cpu().numpy()
                if "contact_raw_stability_score" in cs_log:
                    grasps["contact_raw_stability_score"] = (
                        cs_log["contact_raw_stability_score"].detach().cpu().numpy()
                    )
                if "contact_rerank_score" in cs_log:
                    grasps["contact_rerank_score"] = (
                        cs_log["contact_rerank_score"].detach().cpu().numpy()
                    )
                if "contact_obj_dist" in cs_log:
                    grasps["contact_obj_dist"] = cs_log["contact_obj_dist"].detach().cpu().numpy()
                if "contact_projected_obj_dist" in cs_log:
                    grasps["contact_projected_obj_dist"] = (
                        cs_log["contact_projected_obj_dist"].detach().cpu().numpy()
                    )
                if stability_elong_gate is not None:
                    grasps["stability_elongated_gate"] = stability_elong_gate.detach().cpu().numpy()
                if v4_target_contacts is not None:
                    grasps["target_contacts_cam"] = (
                        v4_target_contacts.detach().cpu().numpy()
                    )
                if "allegro_contact_rerank_ik_mse" in cs_log:
                    grasps["allegro_contact_rerank_ik_mse"] = (
                        cs_log["allegro_contact_rerank_ik_mse"].detach().cpu().numpy()
                    )
                if "allegro_contact_rerank_switched" in cs_log:
                    grasps["allegro_contact_rerank_switched"] = (
                        cs_log["allegro_contact_rerank_switched"].detach().cpu().numpy()
                    )
                if "allegro_contact_ik_mse" in locals() and allegro_contact_ik_mse is not None:
                    grasps["allegro_contact_ik_mse"] = allegro_contact_ik_mse.cpu().numpy()

            for i, joint_name in enumerate(robot_model.movable_joint_names):
                grasps[joint_name] = opt_qpos[:, i]

            # Stage-2 method: freeze open approach as coarse_* (table/world).
            # Evaluator uses coarse for pregrasp/cover; main fields = refined grasp.
            if (
                use_frozen_allegro_coarse
                and frozen_coarse_world is not None
                and int(getattr(args, "allegro_preserve_coarse_pregrasp", 1))
            ):
                n_save = len(world_translations)
                grasps["coarse_translation"] = np.asarray(
                    frozen_coarse_world["translation"][:n_save], dtype=np.float32
                )
                grasps["coarse_rotation"] = np.asarray(
                    frozen_coarse_world["rotation"][:n_save], dtype=np.float32
                )
                for joint_name, q_arr in frozen_coarse_world["qpos"].items():
                    grasps[f"coarse_{joint_name}"] = np.asarray(
                        q_arr[:n_save], dtype=np.float32
                    )
                grasps["allegro_stage2_mode"] = np.asarray(
                    ["coarse_pregrasp_method_refine"] * n_save, dtype=object
                )
                grasps["allegro_init_grasps_npz"] = np.asarray(
                    [str(getattr(args, "allegro_init_grasps_npz", ""))]
                    * n_save,
                    dtype=object,
                )
                dt = np.linalg.norm(
                    grasps["translation"] - grasps["coarse_translation"], axis=1
                )
                cprint(
                    "[Allegro stage2-method] saved coarse_* from frozen init; "
                    f"grasp vs coarse |Δt| mm mean={float(dt.mean() * 1000):.2f} "
                    f"max={float(dt.max() * 1000):.2f}",
                    "cyan",
                )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **grasps)

            cprint(f"[Hybrid] Saved {len(world_translations)} grasps to {save_path}", "green")

        except Exception as e:
            cprint(f"[Hybrid] Error processing {scene_id}: {e}", "red")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    writer.close()
    cprint("[Hybrid-AnnMeta] Done!", "green")


if __name__ == "__main__":
    main()
