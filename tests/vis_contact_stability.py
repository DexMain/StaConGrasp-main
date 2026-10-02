from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import plotly.graph_objects as go
import torch

from eval.contact_stability_infer_pointnext import (
    load_contact_stability_for_infer,
    sample_and_rank_contacts,
    stability_score_from_pred,
)
from network.contact_diffusion_base import _forward_finger_representatives
from network.graspness_sample_with_feature import GraspnessSampleWithFeature
from utils.config import ckpt_to_config
from utils.contact_gt_surface import (
    compute_gt_contact_points,
    crop_object_pc_by_meta_id,
    meta_id_to_object_code,
    object_code_to_meta_id,
    unflatten_contacts,
)
from utils.contact_gt_ibs import (
    CONTACT_SOURCE_IBS,
    CONTACT_SOURCE_NEAREST,
    ContactGTv2Cache,
    table_to_camera_points,
)
from utils.dataset import get_sparse_tensor
from utils.util import set_seed
from utils.vis_plotly import Vis
from eval.topk_select import select_top_n_per_view_with_feature
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider
from paths import project_path

# 图例：◇=FK 指尖，●=网络预测 contact，线段=两者偏差

FINGER_COLORS = ["#FF0033", "#FFCC00", "#00FFFF", "#FF33FF"]
FINGER_LABELS = ["finger0", "finger1", "finger2", "finger3"]
MAIN_OBJECT_COLOR = "#ff6600"
GT_OBJECT_COLOR = "#00C853"
NON_MAIN_OBJECT_COLOR = "#b0b0b0"
GRASPED_OBJECT_COLOR = "#5C9FD6"
GT_V2_IBS_COLOR = "#FF00FF"
GT_V2_FB_COLOR = "#FFFFFF"
HAND_MESH_COLOR = "#78909C"
MARKER_OUTLINE_DARK = "#111111"
MARKER_OUTLINE_LIGHT = "#FFFFFF"


def parse_scene_id(raw: str) -> str:
    """统一为 scene_XXXX（4 位）；scene_00100 / scene_0100 / 100 均 → scene_0100。"""
    raw = raw.strip()
    if raw.startswith("scene_"):
        sid = int(raw.split("_", 1)[1])
        return f"scene_{sid:04d}"
    return f"scene_{int(raw):04d}"


def world_to_camera(rot_w: np.ndarray, trans_w: np.ndarray, extrinsic: np.ndarray):
    """extrinsic: 4x4 camera-to-world (same as predict transform_pose_to_world)."""
    r_c2w = extrinsic[:3, :3]
    t_c2w = extrinsic[:3, 3]
    rot_c = r_c2w.T @ rot_w
    trans_c = r_c2w.T @ (trans_w - t_c2w)
    return rot_c, trans_c


def camera_to_world(points_c: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    if points_c.size == 0:
        return points_c.reshape(0, 3)
    r_c2w = extrinsic[:3, :3]
    t_c2w = extrinsic[:3, 3]
    return (points_c @ r_c2w.T) + t_c2w


def load_align_mat(data_root: str, scene_id: str, camera: str) -> np.ndarray:
    path = os.path.join(
        data_root, "scenes", parse_scene_id(scene_id), camera, "cam0_wrt_table.npy"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cam0_wrt_table not found: {path}")
    return np.load(path).astype(np.float64)


def load_world_extrinsic(
    data_root: str, scene_id: str, camera: str, view_idx: int
) -> np.ndarray:
    """network_input.extrinsics = align_mat @ camera_poses[view]（table/world 系）。"""
    align = load_align_mat(data_root, scene_id, camera)
    cam_pose = load_camera_pose(data_root, scene_id, camera, view_idx)
    return align @ cam_pose


def dex_world_to_camera(
    rot_w: np.ndarray, trans_w: np.ndarray, cam_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """与 GraspNetDatasetContactDiffusionV2 一致：dex world → camera。"""
    r_c2w = cam_pose[:3, :3]
    t_c2w = cam_pose[:3, 3]
    rot_c = r_c2w.T @ rot_w
    trans_c = r_c2w.T @ (trans_w.astype(np.float64) - t_c2w)
    return rot_c, trans_c


def dex_world_to_table(
    rot_w: np.ndarray, trans_w: np.ndarray, align_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """dex world → table（与 precompute_contact_gt_v2_cache 一致）。"""
    r, t = align_mat[:3, :3], align_mat[:3, 3]
    trans_t = (r @ trans_w.astype(np.float64).T).T + t
    rot_t = np.einsum("ij,jk->ik", r, rot_w)
    return rot_t, trans_t


def table_to_network_camera_points(
    pts_table: np.ndarray,
    align_mat: np.ndarray,
    cam_pose: np.ndarray,
) -> np.ndarray:
    """
    table 系 GT → network_input.pc 同款 camera 系。
    extrinsics = align_mat @ cam_pose；勿仅用 table_to_camera_points(., cam_pose)。
    """
    extr = align_mat @ cam_pose
    return (pts_table.astype(np.float64) - extr[:3, 3]) @ extr[:3, :3]


def load_grasps_npz(result_root: str, scene_id: str) -> dict:
    path = os.path.join(result_root, scene_id, "grasps.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"grasps.npz not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def load_sim_success(result_root: str, scene_id: str) -> np.ndarray | None:
    path = os.path.join(result_root, scene_id, "sim_success.npy")
    if not os.path.isfile(path):
        return None
    return np.asarray(np.load(path), dtype=bool)


def ensure_teacher_feature_cache(
    view_idx: int,
    feature_cache: dict,
    dex_model,
    config,
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    pred_main_meta_ids: np.ndarray,
    device,
    grasp_num: int,
    top_n: int,
    stride: int,
    main_meta_id: int,
    trans_c: np.ndarray,
    rot_c: np.ndarray,
):
    if view_idx in feature_cache:
        return feature_cache[view_idx]
    feat, seed, trans_top, rot_top, batch_start = infer_teacher_feature_like_predict(
        dex_model,
        config,
        pc_all,
        seg_all,
        view_idx,
        pred_main_meta_ids,
        device,
        grasp_num,
        top_n,
        stride,
    )
    object_pc = load_object_pc_main_only(
        pc_all, seg_all, view_idx, main_meta_id, max_points=2048
    )
    trans_err = float(np.linalg.norm(trans_top - trans_c))
    rot_err = float(np.linalg.norm(rot_top - rot_c))
    feature_cache[view_idx] = (
        feat.to(device),
        seed.to(device),
        object_pc.to(device),
        trans_err,
        rot_err,
        batch_start,
    )
    return feature_cache[view_idx]


def load_dex_qpos_row(data: np.lib.npyio.NpzFile, dex_idx: int, joint_names: list) -> np.ndarray:
    if all(n in data.files for n in joint_names):
        return np.stack([data[n][dex_idx] for n in joint_names], axis=-1).astype(np.float64)
    j_keys = sorted(k for k in data.files if k.startswith("j") and k[1:].isdigit())
    if j_keys:
        return np.stack([data[k][dex_idx] for k in j_keys], axis=-1).astype(np.float64)
    extra = [k for k in data.files if k not in ("point", "translation", "rotation")]
    return np.stack([data[k][dex_idx] for k in extra], axis=-1).astype(np.float64)


def list_dataset_samples(args) -> None:
    """列出训练 scene 内各 object 的 FPS 样本数与 GT cache 是否可用。"""
    scene_id = parse_scene_id(args.scene_id)
    sid = scene_name_to_int(scene_id)
    if sid >= 100:
        print(f"WARN: {scene_id} 为测试集 (>=100)，通常无训练 GT cache。")
    dex_dir = os.path.join(
        args.data_root, "dex_grasps_new", scene_id, args.hand_name
    )
    fps_dir = os.path.join(
        args.data_root, "fps_sampled_indices", scene_id, args.hand_name
    )
    cache_dir = os.path.join(
        args.contact_gt_cache, scene_id, args.hand_name
    )
    if not os.path.isdir(dex_dir):
        print(f"dex_grasps_new not found: {dex_dir}")
        return
    print(f"Training samples in {scene_id} (robot={args.hand_name}):")
    print(f"  {'object':>6}  {'meta':>4}  {'fps':>5}  {'cache':>5}  dex_path")
    cache = ContactGTv2Cache(args.contact_gt_cache, robot=args.hand_name)
    for grasp_file in sorted(os.listdir(dex_dir)):
        if not grasp_file.endswith(".npz"):
            continue
        code = os.path.splitext(grasp_file)[0]
        meta_id = object_code_to_meta_id(code)
        n_dex = len(np.load(os.path.join(dex_dir, grasp_file))["translation"])
        fps_path = os.path.join(fps_dir, grasp_file)
        n_fps = len(np.load(fps_path)["fps_indices"]) if os.path.isfile(fps_path) else 0
        has_cache = os.path.isfile(os.path.join(cache_dir, grasp_file))
        in_cache = sid in cache.cached_scenes and has_cache
        print(
            f"  {code:>6}  {meta_id:>4}  {n_fps:>5}  "
            f"{'yes' if in_cache else 'no':>5}  n_dex={n_dex}"
        )
    print(
        "\nVis 示例:\n"
        f"  PYTHONPATH=src python tests/vis_contact_stability.py \\\n"
        f"    --from_dataset --scene_id {sid} --object_code 058 \\\n"
        f"    --fps_local_index 0 --view_idx 0 --with_object_pc --with_hand --save_plot"
    )


def build_grasps_from_dataset(
    args,
    scene_id: str,
    joint_names: list,
) -> tuple[dict, list[int]]:
    """
    从 dex + FPS 构造与 dataset_contact_diffusion_fps 对齐的单条样本（world pose）。
    grasp_indices 恒为 [0]（批内唯一样本）。
    """
    scene_id = parse_scene_id(scene_id)
    sid = scene_name_to_int(scene_id)
    if sid >= 100:
        raise ValueError(
            f"{scene_id} 为测试集 (scene>=100)，无训练 GT。"
            "请使用 scene_0000–0099。"
        )

    object_code = args.object_code
    if object_code is None and args.gt_object_meta_id is not None:
        object_code = meta_id_to_object_code(args.gt_object_meta_id)
    if object_code is None:
        raise ValueError("--from_dataset 需要 --object_code 或 --gt_object_meta_id")
    object_code = str(object_code).zfill(3)
    meta_id = object_code_to_meta_id(object_code)

    dex_path = os.path.join(
        args.data_root, "dex_grasps_new", scene_id, args.hand_name, f"{object_code}.npz"
    )
    fps_path = os.path.join(
        args.data_root, "fps_sampled_indices", scene_id, args.hand_name, f"{object_code}.npz"
    )
    if not os.path.isfile(dex_path):
        raise FileNotFoundError(f"dex grasp not found: {dex_path}")

    dex_data = np.load(dex_path)
    fps_indices = None
    if os.path.isfile(fps_path):
        fps_indices = np.load(fps_path)["fps_indices"]

    if args.dex_grasp_index is not None:
        dex_idx = int(args.dex_grasp_index)
    elif args.fps_local_index is not None:
        if fps_indices is None or len(fps_indices) == 0:
            raise FileNotFoundError(f"FPS indices required: {fps_path}")
        local_i = int(args.fps_local_index)
        if local_i < 0 or local_i >= len(fps_indices):
            raise IndexError(
                f"fps_local_index={local_i} out of range [0, {len(fps_indices) - 1}]"
            )
        dex_idx = int(fps_indices[local_i])
    elif fps_indices is not None and len(fps_indices) > 0:
        dex_idx = int(fps_indices[0])
    else:
        dex_idx = 0

    n_dex = len(dex_data["translation"])
    if dex_idx < 0 or dex_idx >= n_dex:
        raise IndexError(f"dex_grasp_index={dex_idx} out of range [0, {n_dex - 1}]")

    view_idx = int(args.view_idx)
    qpos_row = load_dex_qpos_row(dex_data, dex_idx, joint_names)

    grasps: dict = {
        "translation": dex_data["translation"][dex_idx : dex_idx + 1].astype(np.float64),
        "rotation": dex_data["rotation"][dex_idx : dex_idx + 1].astype(np.float64),
        "valid_view_indices": np.array([view_idx], dtype=np.int64),
        "pred_main_meta_id": np.array([meta_id], dtype=np.int64),
        "pred_main_object_id": np.array([meta_id - 1], dtype=np.int64),
        "_dataset_object_code": np.array([object_code]),
        "_dataset_dex_grasp_index": np.array([dex_idx], dtype=np.int64),
        "_dataset_meta_id": np.array([meta_id], dtype=np.int64),
        "_dataset_fps_local_index": np.array(
            [int(args.fps_local_index) if args.fps_local_index is not None else -1],
            dtype=np.int64,
        ),
    }
    for j, name in enumerate(joint_names):
        grasps[name] = np.array([float(qpos_row[j])], dtype=np.float64)

    print(
        f"Dataset sample: {scene_id} object={object_code} meta={meta_id} "
        f"dex_idx={dex_idx} view={view_idx}"
        + (
            f" fps_local={args.fps_local_index}"
            if args.fps_local_index is not None
            else ""
        )
    )
    return grasps, [0]


def build_qpos_tensor(grasps: dict, grasp_idx: int, joint_names, device) -> torch.Tensor:
    q = []
    for name in joint_names:
        if name not in grasps:
            raise KeyError(f"joint {name} missing in grasps.npz")
        q.append(float(grasps[name][grasp_idx]))
    return torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)


def load_object_pc_by_meta_id(
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    view_idx: int,
    meta_id: int,
    max_points: int = 4096,
    device: torch.device | None = None,
) -> torch.Tensor:
    """与 train prepare_batch_v2 的 build_object_pc_batch 一致（单 view）。"""
    obj = crop_object_pc_by_meta_id(
        pc_all[view_idx], seg_all[view_idx], int(meta_id), max_points
    )
    if device is not None:
        obj = obj.to(device)
    return obj


def load_object_pc_main_only(
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    view_idx: int,
    main_object_id: int,
    max_points: int = 2048,
) -> torch.Tensor:
    """与 predict build_object_pc_for_grasps 一致：仅 pred_main 物体点云。"""
    mask = seg_all[view_idx] == int(main_object_id)
    pts = pc_all[view_idx][mask]
    out = torch.zeros(1, max_points, 3, dtype=pc_all.dtype)
    if pts.shape[0] == 0:
        return out
    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[idx]
    out[0, : pts.shape[0]] = pts
    return out


def mask_seg_to_main(seg_view: torch.Tensor, main_object_id: int) -> torch.Tensor:
    seg_masked = seg_view.clone()
    seg_masked[seg_view != int(main_object_id)] = 0
    return seg_masked


def extract_segmented_objects(
    pc_cam: np.ndarray,
    seg_cam: np.ndarray,
    extrinsic: np.ndarray,
    frame: str,
    main_object_id: int | None = None,
    gt_object_id: int | None = None,
    grasped_object_id: int | None = None,
    max_points_per_obj: int = 1024,
    seed: int = 0,
) -> list:
    """按 seg id 拆分物体点云；gt_object_id=dex GT 物体，main_object_id=pred_main。"""
    rng = np.random.default_rng(seed)
    objects = []
    obj_ids = np.unique(seg_cam.astype(np.int64))
    obj_ids = obj_ids[obj_ids > 0]
    for oid in sorted(int(x) for x in obj_ids):
        mask = seg_cam == oid
        pts = pc_cam[mask]
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > max_points_per_obj:
            idx = rng.choice(pts.shape[0], max_points_per_obj, replace=False)
            pts = pts[idx]
        if frame == "world":
            pts = camera_to_world(pts, extrinsic)
        is_gt = gt_object_id is not None and oid == int(gt_object_id)
        is_main = (
            not is_gt
            and main_object_id is not None
            and oid == int(main_object_id)
        )
        is_grasped = (
            not is_gt
            and not is_main
            and grasped_object_id is not None
            and oid == int(grasped_object_id)
        )
        if is_gt:
            color = GT_OBJECT_COLOR
        elif is_main:
            color = MAIN_OBJECT_COLOR
        elif is_grasped:
            color = GRASPED_OBJECT_COLOR
        else:
            color = NON_MAIN_OBJECT_COLOR
        objects.append(
            {
                "obj_id": oid,
                "points": pts,
                "color": color,
                "is_main": is_main,
                "is_gt": is_gt,
                "is_grasped": is_grasped,
            }
        )
    return objects


def add_segmented_object_traces(
    fig: go.Figure,
    objects: list,
    view_idx: int,
    grasp_id: int | None = None,
) -> None:
    for obj in objects:
        if obj.get("is_gt"):
            label = f"GT obj{obj['obj_id']}"
        elif obj["is_main"]:
            label = f"MAIN obj{obj['obj_id']}"
        elif obj.get("is_grasped"):
            label = f"GRASPED obj{obj['obj_id']}"
        else:
            label = f"obj{obj['obj_id']}"
        if grasp_id is not None:
            label = f"{label} (view {view_idx})"
        hi = obj.get("is_gt") or obj["is_main"] or obj.get("is_grasped")
        add_points_trace(
            fig,
            obj["points"],
            label,
            obj["color"],
            size=5 if hi else 2,
            opacity=0.9 if obj.get("is_gt") else (0.85 if obj["is_main"] else (0.35 if obj.get("is_grasped") else 0.35)),
        )


def estimate_grasped_meta_id(
    tips_c: np.ndarray,
    pc_cam: np.ndarray,
    seg_cam: np.ndarray,
) -> int | None:
    """指尖最近邻点所属的 seg id（多数票）。"""
    if tips_c.size == 0:
        return None
    votes = []
    for tip in tips_c:
        dist = np.linalg.norm(pc_cam - tip[None, :], axis=1)
        oid = int(seg_cam[int(np.argmin(dist))])
        if oid > 0:
            votes.append(oid)
    if not votes:
        return None
    vals, counts = np.unique(np.array(votes), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def get_main_meta_id(grasps: dict, grasp_id: int) -> int:
    """predict 用 meta/seg id（pred_main_meta_id），不是 ann id。"""
    if "pred_main_meta_id" in grasps:
        return int(grasps["pred_main_meta_id"][grasp_id])
    return int(grasps["pred_main_object_id"][grasp_id])


def mean_nn_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return float("nan")
    diff = points_a[:, None, :] - points_b[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=-1)).min(axis=1).mean())


def nearest_on_point_cloud(
    query: np.ndarray, cloud: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """query (F,3), cloud (N,3) → nearest pts (F,3), dists (F,)."""
    if cloud.size == 0:
        nan = np.full(query.shape[0], np.nan, dtype=np.float64)
        return np.zeros_like(query), nan
    diff = query[:, None, :] - cloud[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    idx = dist.argmin(axis=1)
    return cloud[idx], dist[np.arange(query.shape[0]), idx]


def valid_object_pc_rows(object_pc: torch.Tensor) -> np.ndarray:
    pts = object_pc[0].detach().cpu().numpy()
    return pts[np.abs(pts).sum(axis=-1) > 1e-6]


def scene_name_to_int(scene_id: str) -> int:
    return int(parse_scene_id(scene_id).split("_")[-1])


def load_camera_pose(data_root: str, scene_id: str, camera: str, view_idx: int) -> np.ndarray:
    path = os.path.join(
        data_root, "scenes", parse_scene_id(scene_id), camera, "camera_poses.npy"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"camera_poses not found: {path}")
    return np.load(path)[int(view_idx)].astype(np.float64)


def find_nearest_dex_grasp_match(
    grasps: dict,
    grasp_id: int,
    scene_id: str,
    data_root: str,
    robot: str = "leap_hand",
) -> tuple[int | None, int | None, float]:
    """在 scene 全部 dex 文件中匹配最近 grasp，返回 (object_meta_id, dex_index, dist)。"""
    dex_dir = os.path.join(
        data_root, "dex_grasps_new", parse_scene_id(scene_id), robot
    )
    if not os.path.isdir(dex_dir):
        return None, None, float("inf")
    trans_w = grasps["translation"][grasp_id].astype(np.float64)
    rot_w = grasps["rotation"][grasp_id].astype(np.float64)
    best_meta: int | None = None
    best_idx: int | None = None
    best_d = float("inf")
    for grasp_file in sorted(os.listdir(dex_dir)):
        if not grasp_file.endswith(".npz"):
            continue
        meta_id = object_code_to_meta_id(os.path.splitext(grasp_file)[0])
        data = np.load(os.path.join(dex_dir, grasp_file))
        for i in range(len(data["translation"])):
            dt = float(np.linalg.norm(data["translation"][i].astype(np.float64) - trans_w))
            dr = float(np.linalg.norm(data["rotation"][i].astype(np.float64) - rot_w))
            d = dt + 0.05 * dr
            if d < best_d:
                best_d = d
                best_idx = int(i)
                best_meta = int(meta_id)
    return best_meta, best_idx, best_d


def resolve_contact_gt_v2_cam(
    args,
    grasps: dict,
    grasp_id: int,
    scene_id: str,
    view_idx: int,
    hand_provider,
    trans_c: torch.Tensor,
    rot_c: torch.Tensor,
    qpos: torch.Tensor,
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    num_fingers: int,
    device: torch.device,
) -> dict:
    """
    与 train_contact_diffusion.prepare_batch_v2 相同的 GT 逻辑（camera 系输出）。

    1) object_meta_id / grasp_index 来自 dex（非 pred_main）
    2) cache hit 且 pose 与 dex 足够近 → IBS-Lite cache
    3) 否则 → compute_gt_contact_points fallback（当前 npz pose + dex 物体点云）
    """
    empty = {
        "gt_cam": None,
        "gt_table": None,
        "gt_mask": None,
        "gt_source": None,
        "cache_hit": False,
        "object_meta_id": None,
        "dex_grasp_index": None,
        "dex_match_dist": float("nan"),
        "object_pc": None,
    }
    if not args.with_contact_gt:
        return empty

    meta_id = args.gt_object_meta_id
    dex_idx = args.dex_grasp_index
    dex_dist = 0.0 if dex_idx is not None else float("nan")

    if "_dataset_meta_id" in grasps:
        meta_id = int(grasps["_dataset_meta_id"][grasp_id])
        dex_idx = int(grasps["_dataset_dex_grasp_index"][grasp_id])
        dex_dist = 0.0
    elif meta_id is None or dex_idx is None:
        if args.match_dex_grasp or dex_idx is not None:
            m_meta, m_idx, m_dist = find_nearest_dex_grasp_match(
                grasps, grasp_id, scene_id, args.data_root, args.hand_name
            )
            if meta_id is None:
                meta_id = m_meta
            if dex_idx is None:
                dex_idx = m_idx
                dex_dist = m_dist
        elif meta_id is None:
            return empty

    if meta_id is None:
        if args.gt_fallback_grasped and args._fallback_gt_meta_id is not None:
            meta_id = int(args._fallback_gt_meta_id)
            print(
                f"  WARN grasp {grasp_id}: dex 未匹配，GT 回退到 meta_id={meta_id} "
                f"（仅 v1 nearest fallback，无 IBS cache）"
            )
        else:
            print(
                f"  WARN grasp {grasp_id}: 无法确定 GT object_meta_id"
                f"（需 dex_grasps_new 或 --gt_object_meta_id）"
            )
            return empty

    object_pc = load_object_pc_by_meta_id(
        pc_all,
        seg_all,
        view_idx,
        meta_id,
        max_points=args.gt_max_object_points,
        device=device,
    )

    try:
        cam_pose = load_camera_pose(args.data_root, scene_id, args.camera, view_idx)
    except FileNotFoundError:
        cam_pose = None

    gt_cam: np.ndarray | None = None
    gt_table: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    gt_source: int | None = None
    cache_hit = False

    pose_matches_dex = (
        dex_idx is not None
        and (
            "_dataset_meta_id" in grasps
            or args.dex_grasp_index is not None
            or dex_dist <= args.dex_match_trans_thresh
        )
    )
    if pose_matches_dex and cam_pose is not None and args.contact_gt_cache:
        cache = ContactGTv2Cache(args.contact_gt_cache, robot=args.hand_name)
        row = cache.lookup(
            scene_name_to_int(scene_id),
            meta_id_to_object_code(meta_id),
            int(dex_idx),
        )
        if row is not None:
            pts_table, mask, src = row
            gt_table = pts_table.astype(np.float64)
            align = load_align_mat(args.data_root, scene_id, args.camera)
            gt_cam = table_to_network_camera_points(gt_table, align, cam_pose)
            gt_mask = mask.astype(np.float64)
            gt_source = int(src)
            cache_hit = True

    if gt_cam is None:
        fb_pts, fb_mask, _ = compute_gt_contact_points(
            hand_provider,
            trans_c,
            rot_c,
            qpos,
            object_pc,
            contact_thresh=args.gt_contact_thresh,
            num_fingers=num_fingers,
        )
        gt_cam = fb_pts[0].detach().cpu().numpy()
        gt_mask = fb_mask[0].detach().cpu().numpy()
        gt_source = CONTACT_SOURCE_NEAREST

    return {
        "gt_cam": gt_cam,
        "gt_table": gt_table,
        "gt_mask": gt_mask,
        "gt_source": gt_source,
        "cache_hit": cache_hit,
        "object_meta_id": int(meta_id),
        "dex_grasp_index": int(dex_idx) if dex_idx is not None else None,
        "dex_match_dist": float(dex_dist) if dex_idx is not None else float("nan"),
        "object_pc": object_pc,
    }


def mask_seg_batch_to_main(
    seg_all: torch.Tensor,
    view_start: int,
    view_end: int,
    pred_main_meta_ids: np.ndarray,
) -> torch.Tensor:
    seg_part = seg_all[view_start:view_end].clone()
    for li, v in enumerate(range(view_start, view_end)):
        oid = int(pred_main_meta_ids[v])
        seg_part[li][seg_all[v] != oid] = 0
    return seg_part


@torch.no_grad()
def infer_teacher_feature_like_predict(
    dex_model,
    config,
    pc_all: torch.Tensor,
    seg_all: torch.Tensor,
    view_idx: int,
    pred_main_meta_ids: np.ndarray,
    device,
    grasp_num: int,
    top_n: int,
    stride: int,
):
    """与 predict v16c 一致：stride batch + main seg mask + score top-N。"""
    batch_start = (view_idx // stride) * stride
    batch_end = min(batch_start + stride, pc_all.shape[0])
    local_idx = view_idx - batch_start

    seg_part = mask_seg_batch_to_main(
        seg_all, batch_start, batch_end, pred_main_meta_ids
    )
    pc_part = pc_all[batch_start:batch_end]
    data = get_sparse_tensor(pc_part, config.data.voxel_size)
    data["seg"] = seg_part
    data = {k: v.to(device) for k, v in data.items()}
    result = dex_model.sample(
        data,
        grasp_num,
        graspness_scale=5,
        allow_fail=True,
        cate=False,
        with_score_parts=True,
        with_point=True,
        with_feature=True,
    )
    rot, trans, qpos, score = result[0], result[1], result[2], result[3]
    grasp_point = result[7]
    features = result[8]
    feat_dim = features.shape[-1]

    sel = select_top_n_per_view_with_feature(
        score,
        rot,
        trans,
        qpos,
        grasp_point.reshape(rot.shape[0], -1, 3),
        features.reshape(rot.shape[0], -1, feat_dim),
        top_n=top_n,
    )
    feat = sel[5][local_idx : local_idx + 1].reshape(1, feat_dim)
    seed = sel[4][local_idx : local_idx + 1].reshape(1, 3)
    trans_top = sel[1][local_idx, 0].detach().cpu().numpy()
    rot_top = sel[0][local_idx, 0].detach().cpu().numpy()
    return feat, seed, trans_top, rot_top, batch_start


def add_points_trace(
    fig,
    pts,
    name,
    color,
    size=6,
    opacity=0.9,
    symbol="circle",
    outline_color: str | None = None,
    outline_width: int = 0,
):
    if pts is None or len(pts) == 0:
        return
    marker = dict(size=size, color=color, opacity=opacity, symbol=symbol)
    if outline_color is not None and outline_width > 0:
        marker["line"] = dict(color=outline_color, width=outline_width)
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=marker,
            name=name,
        )
    )


def add_contact_marker_trace(
    fig,
    pts,
    name,
    color,
    *,
    size=12,
    symbol="circle",
    outline_color: str = MARKER_OUTLINE_DARK,
    outline_width: int = 3,
):
    """高对比 contact 标记：大 marker + 描边，避免与 GRASPED 点云同色淹没。"""
    add_points_trace(
        fig,
        pts,
        name,
        color,
        size=size,
        opacity=1.0,
        symbol=symbol,
        outline_color=outline_color,
        outline_width=outline_width,
    )


def visualize(args):
    if getattr(args, "pred_only", False):
        args.gt_only = False
        args.with_contact_gt = False

    scene_id = parse_scene_id(args.scene_id)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    set_seed(args.seed)
    args._fallback_gt_meta_id = None

    if args.list_dataset_samples:
        list_dataset_samples(args)
        return

    exp_root = os.path.dirname(os.path.dirname(os.path.abspath(args.contact_stability_ckpt)))
    result_root = os.path.join(exp_root, args.result_subdir)

    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=args.urdf_path,
        meta_path=args.meta_path,
        hand_name=args.hand_name,
        device=str(device),
    )
    vis = None
    joint_names = None
    if args.with_hand or args.from_dataset:
        vis = Vis(
            robot_name=args.hand_name,
            urdf_path=args.urdf_path,
            meta_path=args.meta_path,
        )
        joint_names = vis.robot_joints

    if args.from_dataset:
        args.gt_only = True
        args.with_contact_gt = True
        if joint_names is None:
            raise RuntimeError("failed to load hand joint names")
        grasps, grasp_indices = build_grasps_from_dataset(args, scene_id, joint_names)
        n_grasps = 1
        has_saved_contacts = False
        show_pred = False
        sim_success = None
        result_root = args.output_dir or os.path.join(
            exp_root, "dataset_gt_vis", args.result_subdir
        )
        print("from_dataset 模式：训练样本 Contact GT v2（dex + FPS + cache）")
        if args.frame != "camera":
            print(
                "WARN: from_dataset 使用 frame=camera（与 train v2 / dataset 点云、GT 同系）；"
                "已忽略 --frame world。"
            )
            args.frame = "camera"
    else:
        grasps = load_grasps_npz(result_root, scene_id)
        n_grasps = grasps["translation"].shape[0]
        has_saved_contacts = "target_contacts_cam" in grasps
        show_pred = not args.gt_only
        sim_success = (
            load_sim_success(result_root, scene_id) if args.show_sim_status else None
        )

        if show_pred and not has_saved_contacts and not args.allow_reinfer:
            print(
                "ERROR: grasps.npz 缺少 target_contacts_cam，vis 重推理会与 predict 不一致。\n"
                "请对 scene 重跑 v16c predict（--overwrite 1），或加 --allow_reinfer 强制旧路径。\n"
                "若只看训练 GT，请用 --from_dataset（不需 grasps.npz）。"
            )
            print(
                "  PYTHONPATH=src python -m "
                "eval.predict_stacongrasp_pointnext "
                f"--scene_id {scene_id} --scene_num 1 --overwrite 1"
            )
            sys.exit(1)
        if not has_saved_contacts:
            print(
                "WARN: 无 target_contacts_cam，使用 --allow_reinfer 重跑 teacher+contact（可能与 predict 不一致）"
            )

        if args.gt_only:
            args.with_contact_gt = True
            sid = scene_name_to_int(scene_id)
            if sid >= 100:
                print(
                    "WARN: scene_0100+ 为测试集，无 IBS/dex/cache；"
                    "GT 仅为 grasped 物体上的 v1 nearest 回退，非 train v2 IBS-Lite GT。\n"
                    "要看训练 GT 请用: --from_dataset --scene_id 29 --object_code XXX"
                )
            print("GT-only 模式：FK 指尖 + Contact GT v2（与 train_contact_diffusion 一致）")

        if args.grasp_indices:
            grasp_indices = [i for i in args.grasp_indices if 0 <= i < n_grasps]
        else:
            start = max(0, n_grasps - args.max_grasps)
            grasp_indices = list(range(start, n_grasps))

        if (
            args.save_per_grasp
            and len(grasp_indices) > 1
            and not getattr(args, "_per_grasp_recursion", False)
        ):
            for gi in grasp_indices:
                sub = copy.copy(args)
                sub.grasp_indices = [gi]
                sub.max_grasps = 1
                sub._per_grasp_recursion = True
                visualize(sub)
            return

    ni_path = os.path.join(
        args.data_root, "scenes", scene_id, args.camera, "network_input.npz"
    )
    if not os.path.isfile(ni_path):
        raise FileNotFoundError(ni_path)
    network_input = dict(np.load(ni_path))
    pc_all = torch.tensor(network_input["pc"], dtype=torch.float32)
    seg_all = torch.tensor(network_input["seg"], dtype=torch.long)
    extrinsics_all = network_input["extrinsics"]

    mode_str = (
        "from_dataset"
        if args.from_dataset
        else ("saved" if has_saved_contacts and show_pred else ("gt_only" if args.gt_only else "re-infer"))
    )
    print(
        f"{scene_id}: visualizing {len(grasp_indices)} sample(s), mode={mode_str}"
    )

    need_cs = show_pred and (
        (not has_saved_contacts) or args.show_all_k or args.allow_reinfer
    )
    dex_model = cs_net = config = None
    if need_cs:
        config = ckpt_to_config(args.ckpt_path)
        dex_model = GraspnessSampleWithFeature(config.model)
        dex_model.config.voxel_size = config.data.voxel_size
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        dex_model.load_state_dict(ckpt["model"], strict=False)
        dex_model.to(device).eval()
        cs_net = load_contact_stability_for_infer(
            args.contact_stability_ckpt, device=device, contact_v2_ckpt=None
        )

    if vis is None and args.with_hand:
        vis = Vis(
            robot_name=args.hand_name,
            urdf_path=args.urdf_path,
            meta_path=args.meta_path,
        )
        joint_names = vis.robot_joints

    pred_main_meta_ids = (
        grasps["pred_main_meta_id"]
        if "pred_main_meta_id" in grasps
        else grasps["pred_main_object_id"]
    )

    feature_cache = {}
    object_pc_drawn_views: set = set()
    fig = go.Figure()

    if args.with_scene_pc:
        view_for_pc = grasp_indices[0] if grasp_indices else 0
        view_for_pc = int(grasps["valid_view_indices"][view_for_pc])
        extr_pc = extrinsics_all[view_for_pc]
        pc_cam = pc_all[view_for_pc].numpy()
        seg_cam = seg_all[view_for_pc].numpy()
        main_id = get_main_meta_id(grasps, grasp_indices[0]) if grasp_indices else None
        scene_objects = extract_segmented_objects(
            pc_cam,
            seg_cam,
            extr_pc,
            args.frame,
            main_object_id=main_id,
            max_points_per_obj=args.max_points_per_obj,
            seed=args.seed,
        )
        for obj in scene_objects:
            label = f"Scene MAIN obj{obj['obj_id']}" if obj["is_main"] else f"Scene obj{obj['obj_id']}"
            add_points_trace(
                fig,
                obj["points"],
                label,
                obj["color"],
                size=2 if obj["is_main"] else 1,
                opacity=0.55 if obj["is_main"] else 0.2,
            )

    for gi, grasp_id in enumerate(grasp_indices):
        view_idx = int(grasps["valid_view_indices"][grasp_id])
        pc_cam = pc_all[view_idx].numpy()
        seg_cam = seg_all[view_idx].numpy()

        rot_w = grasps["rotation"][grasp_id]
        trans_w = grasps["translation"][grasp_id]

        if args.from_dataset:
            cam_pose = load_camera_pose(args.data_root, scene_id, args.camera, view_idx)
            rot_c, trans_c = dex_world_to_camera(rot_w, trans_w, cam_pose)
            plot_extr = None
        else:
            extr = extrinsics_all[view_idx]
            rot_c, trans_c = world_to_camera(rot_w, trans_w, extr)
            plot_extr = extr if args.frame == "world" else None
            cam_pose = None

        if joint_names is None:
            if vis is None:
                vis = Vis(
                    robot_name=args.hand_name,
                    urdf_path=args.urdf_path,
                    meta_path=args.meta_path,
                )
            joint_names = vis.robot_joints
        qpos = build_qpos_tensor(grasps, grasp_id, joint_names, device)
        rot_t = torch.tensor(rot_c, dtype=torch.float32, device=device).unsqueeze(0)
        trans_t = torch.tensor(trans_c, dtype=torch.float32, device=device).unsqueeze(0)

        main_meta_id = get_main_meta_id(grasps, grasp_id)
        contact_meta_id = main_meta_id
        if "contact_object_meta_id" in grasps:
            contact_meta_id = int(grasps["contact_object_meta_id"][grasp_id])
        elif not args.from_dataset:
            contact_meta_id = None  # resolved after FK tips below

        main_obj_pc = load_object_pc_main_only(
            pc_all, seg_all, view_idx, main_meta_id, max_points=2048
        )
        contact_obj_pc = main_obj_pc
        num_fingers = 4
        trans_err = rot_err = float("nan")
        score_best = best_idx = None
        stab_pred = None

        if show_pred and has_saved_contacts:
            targets_c = np.asarray(grasps["target_contacts_cam"][grasp_id], dtype=np.float64)
            num_fingers = int(targets_c.shape[0])
            if "stability_score" in grasps:
                score_best = float(grasps["stability_score"][grasp_id])
            if "stability_pred" in grasps:
                stab_pred = np.asarray(grasps["stability_pred"][grasp_id])
            if args.show_all_k and cs_net is not None:
                ensure_teacher_feature_cache(
                    view_idx,
                    feature_cache,
                    dex_model,
                    config,
                    pc_all,
                    seg_all,
                    pred_main_meta_ids,
                    device,
                    args.grasp_num,
                    args.top_n,
                    args.stride,
                    main_meta_id,
                    trans_c,
                    rot_c,
                )
        elif show_pred:
            feature, seed_point, object_pc, trans_err, rot_err, batch_start = (
                ensure_teacher_feature_cache(
                    view_idx,
                    feature_cache,
                    dex_model,
                    config,
                    pc_all,
                    seg_all,
                    pred_main_meta_ids,
                    device,
                    args.grasp_num,
                    args.top_n,
                    args.stride,
                    main_meta_id,
                    trans_c,
                    rot_c,
                )
            )
            main_obj_pc = object_pc

            flat_best, stab_pred_t, score_best_t, best_idx = sample_and_rank_contacts(
                cs_net,
                feature,
                seed_point,
                object_pc,
                num_samples=args.contact_num_samples,
            )
            target_contacts = unflatten_contacts(flat_best, cs_net.contact_cfg.num_fingertips)
            targets_c = target_contacts[0].detach().cpu().numpy()
            num_fingers = cs_net.contact_cfg.num_fingertips
            score_best = float(score_best_t.item())
            stab_pred = stab_pred_t[0].detach().cpu().numpy()
            best_idx = int(best_idx.item())
        else:
            targets_c = np.zeros((num_fingers, 3), dtype=np.float64)

        if not args.from_dataset:
            tips_init_c = _forward_finger_representatives(
                hand_provider,
                trans_t,
                rot_t,
                qpos,
                object_pc=None,
                num_fingers=num_fingers,
            )[0].detach().cpu().numpy()
            args._fallback_gt_meta_id = estimate_grasped_meta_id(
                tips_init_c, pc_cam, seg_cam
            )
        else:
            tips_init_c = None

        fk_snap_pc = None
        if args.from_dataset or args.gt_only or args.fk_tip_snap:
            fk_snap_pc = main_obj_pc.to(device)
        tips_c = _forward_finger_representatives(
            hand_provider,
            trans_t,
            rot_t,
            qpos,
            object_pc=fk_snap_pc,
            num_fingers=num_fingers,
        )[0].detach().cpu().numpy()
        tips_snap_c = None
        if show_pred and not args.fk_tip_snap:
            tips_snap_c = _forward_finger_representatives(
                hand_provider,
                trans_t,
                rot_t,
                qpos,
                object_pc=main_obj_pc.to(device),
                num_fingers=num_fingers,
            )[0].detach().cpu().numpy()

        gt_info = resolve_contact_gt_v2_cam(
            args,
            grasps,
            grasp_id,
            scene_id,
            view_idx,
            hand_provider,
            trans_t,
            rot_t,
            qpos,
            pc_all,
            seg_all,
            num_fingers,
            device,
        )
        gt_meta_id = gt_info["object_meta_id"]
        gt_obj_pc = gt_info["object_pc"]
        if args.with_contact_gt and gt_obj_pc is not None:
            tips_c = _forward_finger_representatives(
                hand_provider,
                trans_t,
                rot_t,
                qpos,
                object_pc=gt_obj_pc,
                num_fingers=num_fingers,
            )[0].detach().cpu().numpy()

        grasped_meta_id = estimate_grasped_meta_id(
            tips_init_c if tips_init_c is not None else tips_c, pc_cam, seg_cam
        )
        if contact_meta_id is None and grasped_meta_id is not None:
            contact_meta_id = int(grasped_meta_id)
        elif contact_meta_id is None:
            contact_meta_id = main_meta_id

        contact_obj_pc = load_object_pc_by_meta_id(
            pc_all,
            seg_all,
            view_idx,
            contact_meta_id,
            max_points=2048,
        )

        gt_c = gt_info["gt_cam"]
        gt_table = gt_info.get("gt_table")
        gt_mask = gt_info["gt_mask"]

        cond_pts_c = valid_object_pc_rows(contact_obj_pc)
        pred_proj_c, pred_proj_dist = nearest_on_point_cloud(targets_c, cond_pts_c)

        if args.frame == "world":
            extr_w = plot_extr if plot_extr is not None else load_world_extrinsic(
                args.data_root, scene_id, args.camera, view_idx
            )
            tips = camera_to_world(tips_c, extr_w)
            targets = camera_to_world(targets_c, extr_w) if show_pred else targets_c
            pred_proj = camera_to_world(pred_proj_c, extr_w) if show_pred else pred_proj_c
            if tips_snap_c is not None:
                tips_snap = camera_to_world(tips_snap_c, extr_w)
            else:
                tips_snap = None
            if gt_table is not None:
                gt_pts = gt_table
            elif gt_c is not None:
                gt_pts = camera_to_world(gt_c, extr_w)
            else:
                gt_pts = None
            cond_pts = camera_to_world(cond_pts_c, extr_w) if cond_pts_c.size else cond_pts_c
            pc_plot_extr = extr_w
        else:
            tips = tips_c
            targets = targets_c if show_pred else targets_c
            pred_proj = pred_proj_c if show_pred else pred_proj_c
            tips_snap = tips_snap_c
            gt_pts = gt_c
            cond_pts = cond_pts_c
            pc_plot_extr = None

        if args.with_cond_object_pc and cond_pts.size > 0:
            cond_label = (
                f"Contact cond PC meta={contact_meta_id}"
                + ("=pred_main" if contact_meta_id == main_meta_id else "≠pred_main")
            )
            add_points_trace(
                fig,
                cond_pts,
                cond_label,
                "#AA00FF",
                size=3,
                opacity=0.55,
            )

        if args.with_object_pc and view_idx not in object_pc_drawn_views:
            view_objects = extract_segmented_objects(
                pc_cam,
                seg_cam,
                pc_plot_extr if pc_plot_extr is not None else extrinsics_all[view_idx],
                args.frame,
                main_object_id=main_meta_id,
                gt_object_id=gt_meta_id,
                grasped_object_id=grasped_meta_id,
                max_points_per_obj=args.max_points_per_obj,
                seed=args.seed + view_idx,
            )
            add_segmented_object_traces(fig, view_objects, view_idx, grasp_id=grasp_id)
            object_pc_drawn_views.add(view_idx)

        for fi in range(tips.shape[0]):
            add_contact_marker_trace(
                fig,
                tips[fi : fi + 1],
                f"FK tip(raw)#{grasp_id}-{FINGER_LABELS[fi]}"
                if show_pred and not args.fk_tip_snap
                else f"FK tip#{grasp_id}-{FINGER_LABELS[fi]}",
                FINGER_COLORS[fi % len(FINGER_COLORS)],
                size=11,
                symbol="diamond",
            )
            if show_pred and tips_snap is not None:
                add_contact_marker_trace(
                    fig,
                    tips_snap[fi : fi + 1],
                    f"FK tip(snap)#{grasp_id}-{FINGER_LABELS[fi]}",
                    FINGER_COLORS[fi % len(FINGER_COLORS)],
                    size=7,
                    symbol="diamond-open",
                    outline_color=MARKER_OUTLINE_LIGHT,
                )
            if args.with_contact_gt and gt_pts is not None and fi < gt_pts.shape[0]:
                if gt_mask is None or gt_mask[fi] > 0.5 or not args.gt_mask_only:
                    is_ibs = gt_info["gt_source"] == CONTACT_SOURCE_IBS
                    src_tag = "IBS-Lite" if is_ibs else "nearest-fb"
                    add_contact_marker_trace(
                        fig,
                        gt_pts[fi : fi + 1],
                        f"GT v2({src_tag})#{grasp_id}-{FINGER_LABELS[fi]}",
                        GT_V2_IBS_COLOR if is_ibs else GT_V2_FB_COLOR,
                        size=14,
                        symbol="x" if is_ibs else "diamond-open",
                        outline_color=MARKER_OUTLINE_LIGHT if is_ibs else MARKER_OUTLINE_DARK,
                    )
            if show_pred:
                add_contact_marker_trace(
                    fig,
                    targets[fi : fi + 1],
                    f"Pred contact#{grasp_id}-{FINGER_LABELS[fi]}",
                    FINGER_COLORS[fi % len(FINGER_COLORS)],
                    size=10,
                    symbol="circle",
                )
                if args.project_pred_to_object and np.isfinite(pred_proj_dist[fi]):
                    add_contact_marker_trace(
                        fig,
                        pred_proj[fi : fi + 1],
                        f"Pred→obj proj#{grasp_id}-{FINGER_LABELS[fi]}",
                        FINGER_COLORS[fi % len(FINGER_COLORS)],
                        size=8,
                        symbol="circle-open",
                        outline_color=MARKER_OUTLINE_LIGHT,
                    )
                    seg_proj = np.stack([targets[fi], pred_proj[fi]], axis=0)
                    fig.add_trace(
                        go.Scatter3d(
                            x=seg_proj[:, 0],
                            y=seg_proj[:, 1],
                            z=seg_proj[:, 2],
                            mode="lines",
                            line=dict(color=FINGER_COLORS[fi], width=2, dash="dot"),
                            name=f"Pred→obj#{grasp_id}-{FINGER_LABELS[fi]}",
                            showlegend=False,
                        )
                    )
            if show_pred and tips.shape[0] == targets.shape[0]:
                seg_pts = np.stack([tips[fi], targets[fi]], axis=0)
                fig.add_trace(
                    go.Scatter3d(
                        x=seg_pts[:, 0],
                        y=seg_pts[:, 1],
                        z=seg_pts[:, 2],
                        mode="lines",
                        line=dict(color=FINGER_COLORS[fi], width=4),
                        name=f"Err#{grasp_id}-{FINGER_LABELS[fi]}",
                        showlegend=False,
                    )
                )

        if show_pred and args.show_all_k and cs_net is not None and view_idx in feature_cache:
            with torch.no_grad():
                feature, seed_point, object_pc, _, _, _ = feature_cache[view_idx]
                k = max(int(args.contact_num_samples), 1)
                flat_k = torch.stack(
                    [
                        cs_net.sample_contacts(feature, seed_point, object_pc)
                        for _ in range(k)
                    ],
                    dim=1,
                )
                b = feature.shape[0]
                cond = cs_net.encode_condition(feature, seed_point, object_pc)
                shape_feat, _ = cs_net.encode_shape(object_pc)
                cond_k = cond.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
                shape_k = shape_feat.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
                flat_flat = flat_k.reshape(b * k, -1)
                stab_k = cs_net.predict_stability(cond_k, shape_k, flat_flat).reshape(b, k, -1)
                score_k = stability_score_from_pred(stab_k)[0].detach().cpu().numpy()
                if best_idx is None and has_saved_contacts:
                    best_idx = int(np.argmax(score_k))
                for ki in range(k):
                    if best_idx is not None and ki == best_idx:
                        continue
                    cand = unflatten_contacts(
                        flat_k[:, ki], cs_net.contact_cfg.num_fingertips
                    )
                    cand_c = cand[0].detach().cpu().numpy()
                    cand_plot = (
                        camera_to_world(cand_c, extr)
                        if args.frame == "world"
                        else cand_c
                    )
                    add_points_trace(
                        fig,
                        cand_plot,
                        f"K#{grasp_id}-{ki} score={score_k[ki]:.3f}",
                        "lightgray",
                        size=4,
                        opacity=0.35,
                    )

        if args.with_hand and vis is not None:
            if args.frame == "world" and args.from_dataset:
                align = load_align_mat(args.data_root, scene_id, args.camera)
                rot_plot, trans_plot = dex_world_to_table(rot_w, trans_w, align)
                h_rot = torch.tensor(rot_plot, dtype=torch.float32).unsqueeze(0)
                h_trans = torch.tensor(trans_plot, dtype=torch.float32).unsqueeze(0)
            elif args.frame == "world":
                h_rot = torch.tensor(rot_w, dtype=torch.float32).unsqueeze(0)
                h_trans = torch.tensor(trans_w, dtype=torch.float32).unsqueeze(0)
            else:
                h_rot = rot_t.cpu()
                h_trans = trans_t.cpu()
            qpos_cpu = qpos.cpu()
            for trace in vis.robot_plotly(
                h_trans,
                h_rot,
                qpos_cpu,
                opacity=0.25,
                color=HAND_MESH_COLOR,
            ):
                trace.name = f"Hand#{grasp_id}"
                fig.add_trace(trace)

        main_obj_pts = main_obj_pc[0].detach().cpu().numpy()
        main_obj_pts = main_obj_pts[np.abs(main_obj_pts).sum(axis=-1) > 1e-6]
        gt_obj_pts = np.zeros((0, 3), dtype=np.float64)
        if gt_obj_pc is not None:
            gt_obj_pts = gt_obj_pc[0].detach().cpu().numpy()
            gt_obj_pts = gt_obj_pts[np.abs(gt_obj_pts).sum(axis=-1) > 1e-6]
        target_obj_dist = mean_nn_distance(targets_c, cond_pts_c) if show_pred else float("nan")
        pred_proj_mean = (
            float(np.nanmean(pred_proj_dist)) if show_pred else float("nan")
        )
        tip_obj_dist = mean_nn_distance(tips_c, cond_pts_c)
        gt_obj_dist = (
            mean_nn_distance(gt_c, gt_obj_pts) if gt_c is not None and gt_obj_pts.size else float("nan")
        )
        tip_gt_mse = (
            float(((tips_c - gt_c) ** 2).mean()) if gt_c is not None else float("nan")
        )
        pred_gt_mse = (
            float(((targets_c - gt_c) ** 2).mean())
            if show_pred and gt_c is not None
            else float("nan")
        )
        grasped_note = ""
        gt_note_str = ""
        tip_grasped_dist = pred_grasped_dist = float("nan")
        if grasped_meta_id is not None:
            grasped_pts = pc_cam[seg_cam == int(grasped_meta_id)]
            if grasped_pts.size > 0:
                tip_grasped_dist = mean_nn_distance(tips_c, grasped_pts)
                pred_grasped_dist = mean_nn_distance(targets_c, grasped_pts)
            if grasped_meta_id != main_meta_id:
                grasped_note = f" grasped_meta={grasped_meta_id}≠main"
                if show_pred and contact_meta_id == grasped_meta_id:
                    if "contact_object_meta_id" in grasps:
                        print(
                            f"  NOTE grasp {grasp_id}: npz contact_object_meta_id={contact_meta_id} "
                            f"(pred_main={main_meta_id})；target_contacts_cam 来自 predict 保存，vis 不重跑网络。"
                        )
                    else:
                        print(
                            f"  NOTE grasp {grasp_id}: vis 估计 contact cond=grasped={contact_meta_id} "
                            f"(pred_main={main_meta_id})；npz 无 contact_object_meta_id，"
                            f"Pred ● 仍是旧 v16c predict 结果。"
                        )
                elif show_pred and contact_meta_id == main_meta_id:
                    print(
                        f"  WARN grasp {grasp_id}: pred_main={main_meta_id} "
                        f"≠ grasped={grasped_meta_id}；请用 predict_stacongrasp_pointnext 重跑。"
                    )
        if gt_meta_id is not None and gt_meta_id != main_meta_id:
            gt_note_str = f" gt_meta={gt_meta_id}≠main"

        score_str = (
            f"stab_score={score_best:.4f} "
            if score_best is not None
            else ""
        )
        stab_str = (
            f"stab_pred={stab_pred}"
            if stab_pred is not None
            else "stab_pred=n/a"
        )
        reinfer_str = ""
        if not args.from_dataset and not has_saved_contacts and show_pred and view_idx in feature_cache:
            reinfer_str = (
                f"batch={feature_cache[view_idx][5]} "
                f"teacher_top1_trans_err={trans_err:.5f}m rot_err={rot_err:.4f} "
                f"best_k={best_idx} "
            )

        gt_note = ""
        if args.with_contact_gt and gt_c is not None:
            src = (
                "IBS-cache"
                if gt_info["cache_hit"] and gt_info["gt_source"] == CONTACT_SOURCE_IBS
                else ("nearest-fb" if gt_info["gt_source"] == CONTACT_SOURCE_NEAREST else "?")
            )
            gt_note = (
                f" gt_v2->obj={gt_obj_dist:.4f}m tip_gt_mse={tip_gt_mse:.6f}"
                f" src={src} cache_hit={int(gt_info['cache_hit'])}"
            )
            if gt_info["dex_grasp_index"] is not None:
                gt_note += (
                    f" dex_idx={gt_info['dex_grasp_index']}"
                    f" match={gt_info['dex_match_dist']:.4f}m"
                )
            if show_pred:
                gt_note += f" pred_gt_mse={pred_gt_mse:.6f}"

        print(
            f"  grasp {grasp_id} view={view_idx} "
            f"{'sim=OK ' if sim_success is not None and grasp_id < len(sim_success) and sim_success[grasp_id] else ''}"
            f"{'sim=FAIL ' if sim_success is not None and grasp_id < len(sim_success) and not sim_success[grasp_id] else ''}"
            f"main_meta={main_meta_id} contact_meta={contact_meta_id}{grasped_note}{gt_note_str} "
            f"{score_str}{reinfer_str}"
            f"{'tip_mse=' + f'{float(((tips_c - targets_c) ** 2).mean()):.6f} ' if show_pred else ''}"
            f"{'tip->cond=' + f'{tip_obj_dist:.4f}m ' if show_pred or args.with_cond_object_pc else ''}"
            f"{'pred->cond=' + f'{target_obj_dist:.4f}m ' if show_pred else ''}"
            f"{'pred_proj->cond=' + f'{pred_proj_mean:.4f}m ' if show_pred and args.project_pred_to_object else ''}"
            f"tip->grasped={tip_grasped_dist:.4f}m "
            f"{'pred->grasped=' + f'{pred_grasped_dist:.4f}m ' if show_pred else ''}"
            f"{gt_note} "
            f"{stab_str}"
        )

    mode_tag = (
        "DatasetGT"
        if args.from_dataset
        else ("GT" if args.gt_only else ("Pred+GT" if args.with_contact_gt else "Pred"))
    )
    title_extra = ""
    if args.from_dataset and "_dataset_object_code" in grasps:
        title_extra = (
            f" obj={grasps['_dataset_object_code'][0]}"
            f" dex={grasps['_dataset_dex_grasp_index'][0]}"
        )
    fig.update_layout(
        title=(
            f"Contact vis ({mode_tag}) | {scene_id}{title_extra} | "
            f"{len(grasp_indices)} sample(s) | frame={args.frame}"
        ),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        showlegend=True,
    )

    if args.save_plot:
        out_dir = os.path.join(result_root, scene_id)
        os.makedirs(out_dir, exist_ok=True)
        if args.from_dataset and "_dataset_object_code" in grasps:
            obj_code = grasps["_dataset_object_code"][0]
            dex_idx = int(grasps["_dataset_dex_grasp_index"][0])
            out_path = os.path.join(
                out_dir, f"vis_gt_dataset_{scene_id}_obj{obj_code}_dex{dex_idx:04d}.html"
            )
        else:
            suffix = "_gt" if args.gt_only else ("_pred_gt" if args.with_contact_gt else "_pred")
            if args.save_per_grasp and len(grasp_indices) == 1:
                gi = grasp_indices[0]
                sim_tag = ""
                if sim_success is not None and 0 <= gi < len(sim_success):
                    sim_tag = "_ok" if sim_success[gi] else "_fail"
                out_path = os.path.join(
                    out_dir, f"vis_contacts_pred_{scene_id}_grasp{gi}{sim_tag}.html"
                )
            else:
                out_path = os.path.join(out_dir, f"vis_contacts{suffix}_{scene_id}.html")
        fig.write_html(out_path)
        print(f"Saved to {out_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Contact GT v2 (dataset) or predict contacts (grasps.npz)"
    )
    parser.add_argument("--scene_id", type=str, required=True, help="e.g. 29 or scene_0029")
    parser.add_argument(
        "--from_dataset",
        action="store_true",
        help="从训练集 dex+FPS+cache 读样本（不需 grasps.npz，推荐看 GT）",
    )
    parser.add_argument(
        "--list_dataset_samples",
        action="store_true",
        help="列出 scene 内各 object 的 FPS 样本数与 cache 可用性",
    )
    parser.add_argument(
        "--object_code",
        type=str,
        default=None,
        help="from_dataset：物体 code，如 058（与 dex npz 文件名一致）",
    )
    parser.add_argument(
        "--fps_local_index",
        type=int,
        default=None,
        help="from_dataset：FPS 子集内序号（映射到 dex 原始 grasp_index）",
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="from_dataset：camera view 索引（默认 0）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="from_dataset：HTML 输出根目录（默认 experiments/.../dataset_gt_vis/...）",
    )
    parser.add_argument(
        "--contact_stability_ckpt",
        type=str,
        default="experiments/contact_stability_v4b/ckpt/ckpt_600.pth",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
        help="Teacher ckpt (for re-extracting diffusion feature per view)",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="results_sdf_hybrid_v16c_contact_stability_pen_gated",
    )
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--grasp_indices", type=int, nargs="+", default=None)
    parser.add_argument("--max_grasps", type=int, default=3)
    parser.add_argument("--contact_num_samples", type=int, default=8)
    parser.add_argument("--grasp_num", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32, help="与 predict 一致")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frame", type=str, default="world", choices=["world", "camera"])
    parser.add_argument("--with_scene_pc", action="store_true")
    parser.add_argument("--with_object_pc", action="store_true", help="按 seg 分物体着色显示点云")
    parser.add_argument("--max_points_per_obj", type=int, default=1024)
    parser.add_argument("--with_hand", action="store_true", help="Show hand mesh")
    parser.add_argument("--show_all_k", action="store_true", help="Show non-selected K candidates")
    parser.add_argument(
        "--allow_reinfer",
        action="store_true",
        help="无 target_contacts_cam 时仍重跑 teacher+contact（易与 predict 不一致，不推荐）",
    )
    parser.add_argument(
        "--with_contact_gt",
        action="store_true",
        help="叠加 Contact GT v2（IBS-Lite cache + v1 fallback，与 train v2 一致）",
    )
    parser.add_argument(
        "--gt_only",
        action="store_true",
        help="仅可视化 GT v2 + FK 指尖（不绘制 Pred ●，不需 target_contacts_cam）",
    )
    parser.add_argument(
        "--contact_gt_cache",
        type=str,
        default="/data/contact_gt_v2_cache",
        help="IBS-Lite GT 预计算 cache（train_contact_diffusion 同源）",
    )
    parser.add_argument(
        "--gt_object_meta_id",
        type=int,
        default=None,
        help="手动指定 GT 物体 meta/seg id（默认 dex 全 scene 最近邻匹配）",
    )
    parser.add_argument(
        "--gt_contact_thresh",
        type=float,
        default=0.015,
        help="v1 fallback contact_mask 阈值（与 train v2 默认一致）",
    )
    parser.add_argument(
        "--gt_max_object_points",
        type=int,
        default=4096,
        help="GT object_pc 最大点数（与 train phy_num_points 默认一致）",
    )
    parser.add_argument(
        "--dex_grasp_index",
        type=int,
        default=None,
        help="指定 dex_grasps 原始 index 读取 cache GT；默认自动最近邻匹配",
    )
    parser.add_argument(
        "--match_dex_grasp",
        action="store_true",
        default=True,
        help="自动在 dex_grasps_new 全物体中匹配与 npz pose 最近的 index（默认开）",
    )
    parser.add_argument(
        "--no_match_dex_grasp",
        action="store_false",
        dest="match_dex_grasp",
        help="关闭 dex 自动匹配；需 --gt_object_meta_id 与 --dex_grasp_index",
    )
    parser.add_argument(
        "--dex_match_trans_thresh",
        type=float,
        default=0.08,
        help="自动匹配 dex grasp 的最大距离阈值（translation+0.05*rot）",
    )
    parser.add_argument(
        "--gt_fallback_grasped",
        action="store_true",
        default=True,
        help="dex 不可匹配时，用 FK 质心多数票物体作 GT object（仅 v1 fallback）",
    )
    parser.add_argument(
        "--no_gt_fallback_grasped",
        action="store_false",
        dest="gt_fallback_grasped",
        help="关闭 grasped 回退；dex 失败则跳过 GT",
    )
    parser.add_argument(
        "--gt_mask_only",
        action="store_true",
        help="仅绘制 contact_mask=1 的有效 GT 指",
    )
    parser.add_argument(
        "--fk_tip_snap",
        action="store_true",
        help="FK 指尖吸附到 pred_main object_pc（旧行为，易显得 Pred ● 离物体很远）",
    )
    parser.add_argument(
        "--with_cond_object_pc",
        action="store_true",
        help="绘制 contact 网络 conditioning 的 object_pc（pred_main 裁剪点云）",
    )
    parser.add_argument(
        "--project_pred_to_object",
        action="store_true",
        help="绘制 Pred 在 cond object_pc 上的最近点投影（○ + 虚线）",
    )
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument(
        "--save_per_grasp",
        action="store_true",
        help="每个 grasp 单独输出 HTML（vis_contacts_pred_{scene}_grasp{id}.html）",
    )
    parser.add_argument(
        "--pred_only",
        action="store_true",
        help="仅 Pred contact + FK 指尖（不叠加 GT，predict 模式推荐）",
    )
    parser.add_argument(
        "--show_sim_status",
        action="store_true",
        help="读取 sim_success.npy，在日志/HTML 文件名中标注 OK/FAIL",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=str(project_path("robot_models/urdf/leap_hand_simplified.urdf")),
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=str(project_path("robot_models/meta/leap_hand/meta.yaml")),
    )
    parser.add_argument("--hand_name", type=str, default="leap_hand")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
