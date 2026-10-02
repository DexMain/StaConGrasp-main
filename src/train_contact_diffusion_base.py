

from __future__ import annotations

import argparse
import os
from pprint import pprint
from typing import Dict, Optional

import numpy as np
import torch
from einops import repeat
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from network.contact_diffusion_base import (
    ContactDiffusionNet,
    build_contact_diffusion_cfg,
    save_contact_diffusion_ckpt,
)
from network.graspness_sample_with_feature import GraspnessSampleWithFeature
from utils.config import add_argparse, load_config
from utils.contact_gt_surface import (
    build_object_pc_batch,
    compute_gt_contact_points,
    flatten_contacts,
    infer_num_fingertips_from_hand,
)
from utils.contact_gt_ibs_thumb_split import (
    ContactGTv2bCache,
    apply_aug_rotmat_to_ib_cache_contacts,
)
from utils.dataset import Loader, minkowski_collate_fn
from utils.dataset_contact_diffusion_fps import (
    GraspNetDatasetContactDiffusionV2,
)
from utils.logger import Logger
from utils.util import set_seed
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider


def cfg_val(config, key, default):
    val = getattr(config, key, None)
    return default if val is None else val


arg_mapping = [
    ("exp_name", ("exp_name", str, None)),
    (
        "yaml",
        (
            "yaml",
            str,
            os.path.join("configs", "network", "train_contact_diffusion.yaml"),
        ),
    ),
    ("ckpt", ("ckpt", str, None)),
    ("iter", ("max_iter", int, None)),
    ("lr", ("lr", float, None)),
    ("batch_size", ("batch_size", int, None)),
    ("max_grasps", ("max_grasps", int, None)),
    ("contact_gt_cache", ("contact_gt_cache", str, "/data/contact_gt_v2b_cache")),
    ("urdf_path", ("urdf_path", str, "robot_models/urdf/leap_hand_simplified.urdf")),
    ("meta_path", ("meta_path", str, "robot_models/meta/leap_hand/meta.yaml")),
    ("hand_name", ("hand_name", str, "leap_hand")),
    ("use_success_filter", ("use_success_filter", int, 1)),
    ("use_fps_grasps_only", ("use_fps_grasps_only", int, 1)),
    ("gt_sanity_check", ("gt_sanity_check", int, 1)),
    ("gt_sanity_max_dist", ("gt_sanity_max_dist", float, 0.08)),
]


def resolve_data_root(config) -> str:
    """与 GraspNetDatasetRobust 一致：config.data_root，缺省 /data。"""
    return getattr(config, "data_root", None) or "/data"


def freeze_teacher(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


class AlignMatCache:
    """按 scene_id 缓存 cam0_wrt_table.npy。"""

    def __init__(self, scenes_root: str, camera: str):
        self.scenes_root = scenes_root
        self.camera = camera
        self._cache: Dict[int, np.ndarray] = {}

    def get_batch(self, scene_ids: np.ndarray) -> np.ndarray:
        out = np.zeros((len(scene_ids), 4, 4), dtype=np.float64)
        for i, sid in enumerate(scene_ids):
            sid_int = int(sid)
            if sid_int not in self._cache:
                scene_name = f"scene_{sid_int:04d}"
                path = os.path.join(
                    self.scenes_root, scene_name, self.camera, "cam0_wrt_table.npy"
                )
                self._cache[sid_int] = np.load(path).astype(np.float64)
            out[i] = self._cache[sid_int]
        return out


def _gt_to_object_mean_dist(
    contact_pts: torch.Tensor,
    contact_mask: torch.Tensor,
    object_pc: torch.Tensor,
    meta_ids: torch.Tensor,
    chunk: int = 2048,
) -> float:
    """GT contact 到同 meta 物体点云的平均最近邻距离（camera 系）。"""
    dists = []
    n = contact_pts.shape[0]
    for i in range(n):
        valid = contact_mask[i] > 0.5
        if not bool(valid.any()):
            continue
        meta = int(meta_ids[i].item())
        obj_mask = (object_pc[i].abs().sum(dim=-1) > 1e-6)
        if not bool(obj_mask.any()):
            continue
        pts = contact_pts[i, valid]
        obj = object_pc[i, obj_mask]
        min_d = torch.full((pts.shape[0],), float("inf"), device=pts.device)
        for start in range(0, obj.shape[0], chunk):
            end = min(start + chunk, obj.shape[0])
            d = torch.cdist(pts, obj[start:end]).min(dim=-1).values
            min_d = torch.minimum(min_d, d)
        dists.append(min_d.mean())
    if not dists:
        return float("nan")
    return float(torch.stack(dists).mean().item())


def prepare_batch_v2b(
    data: dict,
    teacher: GraspnessSampleWithFeature,
    hand_provider,
    gt_cache: ContactGTv2bCache,
    align_cache: AlignMatCache,
    max_grasps: int,
    phy_num_points: int,
    cd_cfg,
    cdist_chunk: int,
    gt_sanity_check: bool,
    gt_sanity_max_dist: float,
) -> Optional[dict]:
    batch_size, k_grasps = data["trans"].shape[0], data["trans"].shape[1]
    with torch.no_grad():
        feature = teacher.get_feature(data)

    arange = repeat(
        torch.arange(batch_size, device=feature.device),
        "n -> (n k)",
        k=k_grasps,
    )
    indices = data["centers"].reshape(-1).long()
    sel_feature = feature[arange, indices]
    seed_points = data["point_clouds"][arange, indices]
    gt_trans = data["trans"].reshape(-1, 3)
    gt_rot = data["rot"].reshape(-1, 3, 3)
    gt_qpos = data["qpos"].reshape(-1, data["qpos"].shape[-1])
    batch_ids = arange // k_grasps
    meta_ids = data["object_meta_ids"].reshape(-1).long()
    scene_ids = data["scene_id"].reshape(-1).long()
    grasp_indices = data["grasp_indices"].reshape(-1).long()
    camera_poses = data["camera_pose"].reshape(-1, 4, 4)

    object_pc = build_object_pc_batch(
        data["point_clouds"],
        data["seg"],
        batch_ids,
        meta_ids,
        max_points=phy_num_points,
    )

    n = sel_feature.shape[0]
    if n > max_grasps:
        perm = torch.randperm(n, device=sel_feature.device)[:max_grasps]
        sel_feature = sel_feature[perm]
        seed_points = seed_points[perm]
        gt_trans = gt_trans[perm]
        gt_rot = gt_rot[perm]
        gt_qpos = gt_qpos[perm]
        object_pc = object_pc[perm]
        meta_ids = meta_ids[perm]
        scene_ids = scene_ids[perm]
        grasp_indices = grasp_indices[perm]
        camera_poses = camera_poses[perm]
        batch_ids = batch_ids[perm]

    n = sel_feature.shape[0]
    num_fingers = cd_cfg.num_fingertips

    aug_rotmat = None
    if "aug_rotmat" in data:
        aug_rotmat = data["aug_rotmat"][batch_ids].to(
            device=sel_feature.device, dtype=torch.float32
        )

    scene_ids_np = scene_ids.detach().cpu().numpy()
    align_mats = align_cache.get_batch(scene_ids_np)

    with torch.no_grad():
        contact_cam, mask_np, hit_np = gt_cache.lookup_batch_numpy(
            scene_ids_np,
            meta_ids.detach().cpu().numpy(),
            grasp_indices.detach().cpu().numpy(),
            camera_poses.detach().cpu().numpy(),
            num_fingers=num_fingers,
            align_mats=align_mats,
        )
        contact_pts = torch.from_numpy(contact_cam).to(
            device=sel_feature.device, dtype=torch.float32
        )
        contact_mask = torch.from_numpy(mask_np).to(
            device=sel_feature.device, dtype=torch.float32
        )
        cache_hit = torch.from_numpy(hit_np).to(device=sel_feature.device)

        contact_pts = apply_aug_rotmat_to_ib_cache_contacts(
            contact_pts, cache_hit, aug_rotmat
        )

        invalid_cached = cache_hit & (contact_mask < 0.5).any(dim=1)
        miss = ~cache_hit
        need_fb = miss | invalid_cached
        tip_dist = torch.full(
            (n, num_fingers), cd_cfg.contact_thresh * 0.5, device=sel_feature.device
        )
        if need_fb.any():
            fb_pts, fb_mask, fb_dist = compute_gt_contact_points(
                hand_provider,
                gt_trans[need_fb],
                gt_rot[need_fb],
                gt_qpos[need_fb],
                object_pc[need_fb],
                cdist_chunk=cdist_chunk,
                contact_thresh=cd_cfg.contact_thresh,
                num_fingers=num_fingers,
            )
            need_idx = torch.where(need_fb)[0]
            miss_local = miss[need_fb]
            if bool(miss_local.any()):
                rows = need_idx[miss_local]
                contact_pts[rows] = fb_pts[miss_local]
                contact_mask[rows] = fb_mask[miss_local]
                tip_dist[rows] = fb_dist[miss_local]

            hit_invalid_local = invalid_cached[need_fb]
            if bool(hit_invalid_local.any()):
                rows = need_idx[hit_invalid_local]
                invalid_f = contact_mask[rows] < 0.5
                contact_pts[rows] = torch.where(
                    invalid_f.unsqueeze(-1),
                    fb_pts[hit_invalid_local],
                    contact_pts[rows],
                )
                tip_dist[rows] = torch.where(
                    invalid_f,
                    fb_dist[hit_invalid_local],
                    tip_dist[rows],
                )

        contacts_flat = flatten_contacts(contact_pts)

    valid = contact_mask.sum(dim=-1) >= 1
    if int(valid.sum()) < 2:
        return None

    cache_hit_f = cache_hit[valid].float().mean().item()
    gt_obj_dist = _gt_to_object_mean_dist(
        contact_pts[valid],
        contact_mask[valid],
        object_pc[valid],
        meta_ids[valid],
    )
    if gt_sanity_check and np.isfinite(gt_obj_dist) and gt_obj_dist > gt_sanity_max_dist:
        print(
            f"[warn] gt->object mean dist {gt_obj_dist:.4f}m > "
            f"{gt_sanity_max_dist:.4f}m "
            f"(IBS 点在 bisector 上，距 depth 物体面 2–4cm 可正常；"
            f"若持续 >5cm 检查 cache/aug)"
        )

    return {
        "feature": sel_feature[valid],
        "seed_points": seed_points[valid],
        "object_pc": object_pc[valid],
        "contacts_flat": contacts_flat[valid],
        "contact_mask": contact_mask[valid],
        "tip_dist": tip_dist[valid],
        "gt_trans": gt_trans[valid],
        "gt_rot": gt_rot[valid],
        "gt_qpos": gt_qpos[valid],
        "scene_ids": scene_ids[valid],
        "meta_ids": meta_ids[valid],
        "grasp_indices": grasp_indices[valid],
        "camera_poses": camera_poses[valid],
        "align_mats": torch.from_numpy(align_mats).to(
            device=sel_feature.device, dtype=torch.float32
        )[valid],
        "cache_hit": cache_hit[valid],
        "aug_rotmat": aug_rotmat[valid] if aug_rotmat is not None else None,
        "cache_hit_ratio": cache_hit_f,
        "gt_obj_dist": gt_obj_dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Contact Diffusion v2b (IBS thumb-split GT + fixed camera transform)"
    )
    add_argparse(parser, arg_mapping)
    args = parser.parse_args()

    config = load_config(args.yaml, arg_mapping, args)
    if config.ckpt is None:
        raise ValueError("必须提供 --ckpt 作为 frozen teacher 权重")

    pprint(config)
    set_seed(config.seed)
    logger = Logger(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_success = bool(int(cfg_val(config, "use_success_filter", 1)))
    use_fps = bool(int(cfg_val(config, "use_fps_grasps_only", 1)))
    success_root = cfg_val(config, "success_indices_root", None)
    fps_root = cfg_val(config, "fps_root", None)
    cache_root = cfg_val(config, "contact_gt_cache", "/data/contact_gt_v2b_cache")
    gt_sanity_check = bool(int(cfg_val(config, "gt_sanity_check", 1)))
    gt_sanity_max_dist = float(cfg_val(config, "gt_sanity_max_dist", 0.08))

    gt_cache = ContactGTv2bCache(cache_root, robot=config.data.robot)
    n_preload = gt_cache.preload_all()
    data_root = resolve_data_root(config)
    scenes_root = os.path.join(data_root, "scenes")
    align_cache = AlignMatCache(scenes_root, config.data.camera)
    print(f"IBS-Lite contact cache: {cache_root}")
    print(f"data_root: {data_root}")
    print(
        f"cached seminal scenes: {len(gt_cache.cached_scenes)} "
        f"(preloaded {n_preload} object npz)"
    )
    print("GT camera transform: align_mat @ camera_pose (v2b)")

    train_dataset = GraspNetDatasetContactDiffusionV2(
        config,
        config.train_split,
        is_train=True,
        use_success_filter=use_success,
        success_indices_root=success_root,
        use_fps_grasps_only=use_fps,
        fps_root=fps_root,
    )
    val_datasets = [
        GraspNetDatasetContactDiffusionV2(
            config,
            split,
            is_train=False,
            use_success_filter=use_success,
            success_indices_root=success_root,
            use_fps_grasps_only=use_fps,
            fps_root=fps_root,
        )
        for split in config.val_split
    ]
    train_loader = Loader(
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            drop_last=True,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=minkowski_collate_fn,
        )
    )
    val_loader = [
        Loader(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                drop_last=True,
                num_workers=config.num_workers,
                shuffle=True,
                collate_fn=minkowski_collate_fn,
            )
        )
        for dataset in val_datasets
    ]

    config.model["voxel_size"] = config.data.voxel_size

    teacher = GraspnessSampleWithFeature(config.model)
    ckpt = torch.load(config.ckpt, map_location="cpu")
    teacher.load_state_dict(ckpt["model"], strict=False)
    freeze_teacher(teacher)
    teacher.to(device)
    print(f"loaded frozen teacher from {config.ckpt}")

    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=config.urdf_path,
        meta_path=config.meta_path,
        hand_name=config.hand_name,
        device=str(device),
    )

    cd_cfg = build_contact_diffusion_cfg(config.contact_diffusion)
    cd_cfg.num_fingertips = infer_num_fingertips_from_hand(hand_provider)
    print(f"num_fingertips={cd_cfg.num_fingertips}, contact_dim={cd_cfg.contact_dim}")

    contact_net = ContactDiffusionNet(cd_cfg).to(device)

    optimizer = torch.optim.Adam(contact_net.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, config.max_iter, eta_min=config.lr_min)
    trainable = list(contact_net.parameters())

    max_grasps = int(cfg_val(config, "max_grasps", 16))
    phy_num_points = int(cfg_val(config, "phy_num_points", 4096))
    phy_cdist_chunk = int(cfg_val(config, "phy_cdist_chunk", 2048))

    save_contact_diffusion_ckpt(
        os.path.join(logger.ckpt_path, "ckpt_0.pth"),
        contact_net,
        optimizer,
        0,
        extra={
            "teacher_ckpt": config.ckpt,
            "contact_gt": "ibs_lite_v2b_thumb_split",
        },
    )

    contact_net.train()
    for it in trange(config.max_iter):
        optimizer.zero_grad(set_to_none=True)
        data = train_loader.get()
        data = {k: v.to(device) for k, v in data.items()}

        batch = prepare_batch_v2b(
            data,
            teacher,
            hand_provider,
            gt_cache,
            align_cache,
            max_grasps,
            phy_num_points,
            cd_cfg,
            phy_cdist_chunk,
            gt_sanity_check,
            gt_sanity_max_dist,
        )
        if batch is None:
            continue

        loss = contact_net(
            batch["contacts_flat"],
            batch["feature"],
            batch["seed_points"],
            batch["object_pc"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        scheduler.step()

        if it % config.log_every == 0:
            log_dict = {
                "loss": loss.item(),
                "tip_dist_mean": batch["tip_dist"].mean().item(),
                "cache_hit_ratio": batch["cache_hit_ratio"],
                "batch_size": float(batch["contacts_flat"].shape[0]),
            }
            if np.isfinite(batch["gt_obj_dist"]):
                log_dict["gt_obj_dist"] = batch["gt_obj_dist"]
            logger.log(log_dict, "train", it)

        if (it + 1) % config.save_every == 0 or it + 1 == config.max_iter:
            save_contact_diffusion_ckpt(
                os.path.join(logger.ckpt_path, f"ckpt_{it + 1}.pth"),
                contact_net,
                optimizer,
                it + 1,
                extra={
                    "teacher_ckpt": config.ckpt,
                    "contact_gt": "ibs_lite_v2b_thumb_split",
                },
            )

        if (it + 1) % config.val_every == 0:
            contact_net.eval()
            val_logs = []
            with torch.no_grad():
                for _ in range(config.val_num):
                    vdata = val_loader[0].get()
                    vdata = {k: v.to(device) for k, v in vdata.items()}
                    vbatch = prepare_batch_v2b(
                        vdata,
                        teacher,
                        hand_provider,
                        gt_cache,
                        align_cache,
                        max_grasps,
                        phy_num_points,
                        cd_cfg,
                        phy_cdist_chunk,
                        gt_sanity_check,
                        gt_sanity_max_dist,
                    )
                    if vbatch is None:
                        continue
                    vloss = contact_net(
                        vbatch["contacts_flat"],
                        vbatch["feature"],
                        vbatch["seed_points"],
                        vbatch["object_pc"],
                    )
                    val_logs.append(vloss.item())
            if val_logs:
                logger.log({"loss": sum(val_logs) / len(val_logs)}, "val", it)
            contact_net.train()


if __name__ == "__main__":
    main()
