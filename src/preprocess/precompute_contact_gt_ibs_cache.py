from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from termcolor import cprint
from tqdm import tqdm

from eval.diagnose_grasp_contact import load_graspnet_object_surface_points
from utils.contact_gt_ibs import (
    CONTACT_SOURCE_IBS,
    CONTACT_SOURCE_NEAREST,
    IBSLiteContactConfig,
    compute_ibs_lite_contact_chunk,
)
from utils.robot_model import RobotModel
from ibs.scripts.load_fps_grasps import load_fps_grasps_for_scene
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider


def _load_qpos(data: np.lib.npyio.NpzFile, joint_names: List[str]) -> np.ndarray:
    if all(n in data.files for n in joint_names):
        return np.stack([data[n] for n in joint_names], axis=-1).astype(np.float32)
    j_keys = sorted(k for k in data.files if k.startswith("j") and k[1:].isdigit())
    if j_keys:
        return np.stack([data[k] for k in j_keys], axis=-1).astype(np.float32)
    extra = [k for k in data.files if k not in ("point", "translation", "rotation")]
    return np.stack([data[k] for k in extra], axis=-1).astype(np.float32)


def precompute_scene_object(
    scene_name: str,
    object_code: str,
    batch_indices: List[int],
    fps_orig_indices: np.ndarray,
    ibs_voxels: np.ndarray,
    w2h_all: np.ndarray,
    grasp_npz_path: str,
    object_pc_table: np.ndarray,
    extrinsics: np.ndarray,
    hand_provider,
    joint_names: List[str],
    cfg: IBSLiteContactConfig,
    device: str,
    chunk_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(grasp_npz_path)
    qpos_all = _load_qpos(data, joint_names)
    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    n_out = len(batch_indices)
    grasp_indices = np.zeros(n_out, dtype=np.int32)
    ibs_batch_indices = np.zeros(n_out, dtype=np.int32)
    contact_pts = np.zeros((n_out, 4, 3), dtype=np.float32)
    contact_mask = np.zeros((n_out, 4), dtype=np.float32)
    contact_source = np.zeros(n_out, dtype=np.uint8)

    n_ibs = 0
    n_fallback = 0

    # 预收集所有 grasp 的 table 系 pose
    orig_list = [int(fps_orig_indices[i]) for i in range(len(batch_indices))]
    trans_all = []
    rot_all = []
    qpos_list = []
    ibs_idx_list = []
    for local_i, batch_i in enumerate(batch_indices):
        orig_idx = orig_list[local_i]
        trans_w = data["translation"][orig_idx].astype(np.float32)
        rot_w = data["rotation"][orig_idx].astype(np.float32)
        trans_all.append(((r @ trans_w.T).T + t).astype(np.float32))
        rot_all.append(np.einsum("ij,jk->ik", r, rot_w).astype(np.float32))
        qpos_list.append(qpos_all[orig_idx])
        ibs_idx_list.append(int(batch_i))

    trans_all = np.stack(trans_all, axis=0)
    rot_all = np.stack(rot_all, axis=0)
    qpos_arr = np.stack(qpos_list, axis=0)

    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        chunk_ibs = np.stack([ibs_voxels[ibs_idx_list[i]] for i in range(start, end)], axis=0)
        chunk_w2h = np.stack([w2h_all[ibs_idx_list[i]] for i in range(start, end)], axis=0)
        pts, mask, src = compute_ibs_lite_contact_chunk(
            hand_provider,
            trans_all[start:end],
            rot_all[start:end],
            qpos_arr[start:end],
            object_pc_table,
            chunk_ibs,
            chunk_w2h,
            cfg=cfg,
            device=device,
        )
        for j in range(end - start):
            out_i = start + j
            grasp_indices[out_i] = orig_list[out_i]
            ibs_batch_indices[out_i] = ibs_idx_list[out_i]
            f = pts.shape[1]
            contact_pts[out_i, :f] = pts[j, :f]
            contact_mask[out_i, :f] = mask[j, :f]
            contact_source[out_i] = src[j]
            if src[j] == CONTACT_SOURCE_IBS:
                n_ibs += 1
            else:
                n_fallback += 1

    stats = {
        "n": n_out,
        "n_ibs": n_ibs,
        "n_fallback": n_fallback,
        "ibs_ratio": n_ibs / max(n_out, 1),
    }
    return grasp_indices, ibs_batch_indices, contact_pts, contact_mask, contact_source, stats


def precompute_scene(
    scene_id: int,
    ibs_root: str,
    cache_root: str,
    dex_grasps_root: str,
    fps_root: str,
    scenes_root: str,
    mesh_root: str,
    robot: str,
    camera: str,
    urdf_path: str,
    meta_path: str,
    hand_name: str,
    cfg: IBSLiteContactConfig,
    device: str,
    skip_existing: bool,
    chunk_size: int,
) -> dict:
    scene_name = f"scene_{scene_id:04d}"
    ibs_path = os.path.join(ibs_root, "ibs", f"{scene_name}.npy")
    w2h_path = os.path.join(ibs_root, "w2h_trans", f"{scene_name}.npy")

    if not os.path.isfile(ibs_path) or not os.path.isfile(w2h_path):
        return {"status": "no_ibs", "scene": scene_name}

    out_dir = os.path.join(cache_root, scene_name, robot)
    os.makedirs(out_dir, exist_ok=True)

    ibs_all = np.load(ibs_path)
    w2h_all = np.load(w2h_path)

    _, object_indices = load_fps_grasps_for_scene(
        scene_name,
        robot,
        grasp_base_path=dex_grasps_root,
        fps_base_path=fps_root,
        return_object_ids=True,
    )
    if not object_indices:
        return {"status": "no_fps", "scene": scene_name}

    extrinsics = np.load(
        os.path.join(scenes_root, scene_name, camera, "cam0_wrt_table.npy")
    )
    surf_dict = load_graspnet_object_surface_points(
        scene_name, mesh_root, scenes_root=scenes_root, camera=camera
    )

    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=urdf_path,
        meta_path=meta_path,
        hand_name=hand_name,
        device=device,
    )
    robot_model = RobotModel(urdf_path, meta_path)

    scene_stats = {"status": "ok", "scene": scene_name, "objects": {}}

    for obj_file, batch_indices in sorted(object_indices.items()):
        object_code = os.path.splitext(obj_file)[0]
        if object_code not in surf_dict:
            continue

        grasp_path = os.path.join(dex_grasps_root, scene_name, robot, obj_file)
        fps_path = os.path.join(fps_root, scene_name, robot, obj_file)
        if not os.path.isfile(grasp_path) or not os.path.isfile(fps_path):
            continue

        fps_orig = np.load(fps_path)["fps_indices"]
        out_path = os.path.join(out_dir, f"{object_code}.npz")
        if skip_existing and os.path.isfile(out_path):
            continue

        grasp_indices, ibs_batch_idx, contact_pts, contact_mask, contact_source, stats = (
            precompute_scene_object(
                scene_name,
                object_code,
                batch_indices,
                fps_orig,
                ibs_all,
                w2h_all,
                grasp_path,
                surf_dict[object_code],
                extrinsics,
                hand_provider,
                robot_model.joint_names,
                cfg,
                device,
                chunk_size=chunk_size,
            )
        )

        np.savez_compressed(
            out_path,
            grasp_indices=grasp_indices,
            ibs_batch_indices=ibs_batch_idx,
            contact_pts=contact_pts,
            contact_mask=contact_mask,
            contact_source=contact_source,
        )
        scene_stats["objects"][object_code] = stats
        cprint(
            f"  {scene_name}/{object_code}: n={stats['n']} "
            f"ibs={stats['n_ibs']} fallback={stats['n_fallback']} "
            f"({stats['ibs_ratio']*100:.1f}% IBS)",
            "green",
        )

    return scene_stats


def parse_scene_range(spec: str) -> List[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi)))
    return [int(spec)]


def main():
    parser = argparse.ArgumentParser(description="Precompute IBS-Lite contact GT v2 cache")
    parser.add_argument("--scene_id", type=int, default=None, help="单个 scene 编号")
    parser.add_argument(
        "--scene_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("LO", "HI"),
        help="scene 编号区间 [LO, HI)",
    )
    parser.add_argument("--scene_list", type=str, default=None, help="逗号分隔 scene 编号")
    parser.add_argument("--ibs_root", type=str, default="/data/ibsdata")
    parser.add_argument("--cache_root", type=str, default="/data/contact_gt_v2_cache")
    parser.add_argument("--dex_grasps_root", type=str, default="/data/dex_grasps_new")
    parser.add_argument("--fps_root", type=str, default="/data/fps_sampled_indices")
    parser.add_argument("--scenes_root", type=str, default="/data/scenes")
    parser.add_argument("--mesh_root", type=str, default="/data/meshdata")
    parser.add_argument("--robot", type=str, default="leap_hand")
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--urdf_path", type=str, default="robot_models/urdf/leap_hand_simplified.urdf")
    parser.add_argument("--meta_path", type=str, default="robot_models/meta/leap_hand/meta.yaml")
    parser.add_argument("--hand_name", type=str, default="leap_hand")
    parser.add_argument("--max_assign_dist", type=float, default=0.025)
    parser.add_argument("--chunk_size", type=int, default=128, help="FK batch size per object")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_existing", type=int, default=1)
    args = parser.parse_args()

    if args.scene_id is not None:
        scene_ids = [args.scene_id]
    elif args.scene_range is not None:
        scene_ids = list(range(args.scene_range[0], args.scene_range[1]))
    elif args.scene_list:
        scene_ids = [int(x.strip()) for x in args.scene_list.split(",") if x.strip()]
    else:
        parser.error("请指定 --scene_id / --scene_range / --scene_list")

    cfg = IBSLiteContactConfig(max_assign_dist=args.max_assign_dist)
    skip = bool(args.skip_existing)

    summary = []
    for sid in tqdm(scene_ids, desc="scenes"):
        st = precompute_scene(
            sid,
            args.ibs_root,
            args.cache_root,
            args.dex_grasps_root,
            args.fps_root,
            args.scenes_root,
            args.mesh_root,
            args.robot,
            args.camera,
            args.urdf_path,
            args.meta_path,
            args.hand_name,
            cfg,
            args.device,
            skip,
            args.chunk_size,
        )
        summary.append(st)

    ok = [s for s in summary if s.get("status") == "ok"]
    if ok:
        total_ibs = sum(
            o.get("n_ibs", 0)
            for s in ok
            for o in s.get("objects", {}).values()
        )
        total_n = sum(
            o.get("n", 0) for s in ok for o in s.get("objects", {}).values()
        )
        cprint(
            f"\n[done] {len(ok)} scenes, {total_n} grasps, "
            f"IBS assign {total_ibs}/{total_n} ({100*total_ibs/max(total_n,1):.1f}%)",
            "cyan",
        )
    cprint(f"cache_root={args.cache_root}", "cyan")


if __name__ == "__main__":
    main()
