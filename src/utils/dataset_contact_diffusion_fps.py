"""
GraspNetDataset v2：返回 grasp_indices + scene_id，供 IBS-Lite cache 查询。

可选 use_fps_grasps_only：仅从 FPS 子集采样（与 IBS/cache 对齐）。
"""

from __future__ import annotations

import os
import random

import numpy as np

from utils.contact_gt_surface import object_code_to_meta_id
from utils.dataset_contact_diffusion_base import (
    GraspNetDatasetContactDiffusion,
)


class GraspNetDatasetContactDiffusionV2(GraspNetDatasetContactDiffusion):
    def __init__(
        self,
        *args,
        use_fps_grasps_only: bool = True,
        fps_root: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_fps_grasps_only = bool(use_fps_grasps_only)
        self._fps_root = fps_root or os.path.join(
            self._data_root, "fps_sampled_indices"
        )
        # scene 级缓存：[(grasp_file, prefiltered_grasps_dict), ...]
        self._index_pool_cache: dict[
            tuple[str, str, str], list[tuple[str, dict[str, np.ndarray]]]
        ] = {}

    def _load_fps_indices(self, scene: str, robot: str, grasp_file: str) -> np.ndarray | None:
        path = os.path.join(self._fps_root, scene, robot, grasp_file)
        if not os.path.isfile(path):
            return None
        return np.load(path)["fps_indices"]

    def _build_index_pools(
        self,
        scene: str,
        robot: str,
        frac_suffix: str,
    ) -> list[tuple[str, dict[str, np.ndarray]]] | None:
        """返回 [(grasp_file, prefiltered_grasps_dict), ...]，scene 级只构建一次。"""
        grasp_dir = os.path.join(
            self._data_root, "dex_grasps_new" + frac_suffix, scene, robot
        )
        if not os.path.isdir(grasp_dir):
            return None

        valid_pools: list[tuple[str, dict[str, np.ndarray]]] = []
        for grasp_file in sorted(os.listdir(grasp_dir)):
            if not grasp_file.endswith(".npz"):
                continue
            orig_path = os.path.join(grasp_dir, grasp_file)
            orig = np.load(orig_path)
            n_orig = len(orig["point"])

            success_set = set(range(n_orig))
            if self.use_success_filter:
                success_path = os.path.join(
                    self._success_root, scene, robot, grasp_file
                )
                if os.path.isfile(success_path):
                    success_set = set(
                        np.load(success_path)["success_indices"].tolist()
                    )
                if not success_set:
                    continue

            if self.use_fps_grasps_only:
                fps_idx = self._load_fps_indices(scene, robot, grasp_file)
                if fps_idx is None or len(fps_idx) == 0:
                    continue
                pick = np.array(
                    [int(i) for i in fps_idx if int(i) in success_set],
                    dtype=np.int64,
                )
                if len(pick) == 0:
                    continue
            else:
                pick = np.array(sorted(success_set), dtype=np.int64)

            grasps = {k: orig[k][pick] for k in orig.files}
            grasps["_orig_grasp_indices"] = pick
            valid_pools.append((grasp_file, grasps))

        return valid_pools or None

    def _get_index_pools(
        self,
        scene: str,
        robot: str,
        frac_suffix: str,
    ) -> list[tuple[str, dict[str, np.ndarray]]] | None:
        key = (scene, robot, frac_suffix)
        if key not in self._index_pool_cache:
            pools = self._build_index_pools(scene, robot, frac_suffix)
            if pools is None:
                return None
            self._index_pool_cache[key] = pools
        return self._index_pool_cache[key]

    def _sample_dex_grasps_with_meta(
        self,
        scene: str,
        robot: str,
        frac_suffix: str,
        sample_total: int,
    ) -> tuple[dict, np.ndarray, np.ndarray] | None:
        valid_pools = self._get_index_pools(scene, robot, frac_suffix)
        if not valid_pools:
            return None

        merged_samples: dict[str, list] = {}
        object_meta_ids: list[int] = []
        grasp_indices: list[int] = []
        for _ in range(sample_total):
            grasp_file, grasps = valid_pools[
                np.random.randint(0, len(valid_pools))
            ]
            n_grasp = len(grasps["point"])
            sel = int(np.random.randint(0, n_grasp))
            for key, val in grasps.items():
                if key == "_orig_grasp_indices":
                    continue
                merged_samples.setdefault(key, []).append(val[sel])
            object_code = os.path.splitext(grasp_file)[0]
            object_meta_ids.append(object_code_to_meta_id(object_code))
            grasp_indices.append(int(grasps["_orig_grasp_indices"][sel]))

        merged = {k: np.stack(v, axis=0) for k, v in merged_samples.items()}
        permute = np.random.permutation(sample_total)
        merged = {k: v[permute] for k, v in merged.items()}
        meta_arr = np.array(object_meta_ids, dtype=np.int64)[permute]
        grasp_idx_arr = np.array(grasp_indices, dtype=np.int64)[permute]
        return merged, meta_arr, grasp_idx_arr

    def __getitem__(self, dataset_idx: int, _retry: int = 0):
        if _retry >= self.max_load_retries:
            raise RuntimeError(
                f"GraspNetDatasetContactDiffusionV2: exceeded {self.max_load_retries} retries"
            )

        if self.config.robot == "gripper":
            ret = super(GraspNetDatasetContactDiffusion, self).__getitem__(
                dataset_idx, _retry
            )
            if not self.is_eval and isinstance(ret, dict):
                ret["object_meta_ids"] = np.zeros(ret["trans"].shape[0], dtype=np.int64)
                ret["grasp_indices"] = np.zeros(ret["trans"].shape[0], dtype=np.int64)
                ret["scene_id"] = np.zeros(1, dtype=np.int64)
            return ret

        cate = random.choice(self.cates)
        if cate == "orig":
            if not self.is_eval:
                scene, view = random.choice(self.views)
            else:
                scene, view = self.views[dataset_idx]
        elif cate == "part":
            orig_scenes = list(range(100))[:: self.config.scene_fraction]
            scene = f"scene_{1000 + random.choice(orig_scenes) * 75 + random.randint(0, 74)}"
            view = random.randint(0, 255)

        try:
            from PIL import Image
            import scipy.io as scio

            from utils.dataset_graspnet_robust import load_graspness_aligned
            from utils.pc import depth_image_to_point_cloud, get_workspace_mask

            str_view = str(view).zfill(4)
            suffix = "_gt" if self.config.render else ""

            path = os.path.join(self._data_root, "scenes", scene, self.config.camera)
            depth = np.array(Image.open(os.path.join(path, "depth" + suffix, str_view + ".png")))
            depth_shape = depth.shape
            seg = np.array(Image.open(os.path.join(path, "label" + suffix, str_view + ".png")))
            meta = scio.loadmat(os.path.join(path, "meta", str_view + ".mat"))
            instrincs = meta["intrinsic_matrix"]
            factor_depth = meta["factor_depth"]
            camera_poses = np.load(os.path.join(path, "camera_poses.npy"))
            align_mat = np.load(os.path.join(path, "cam0_wrt_table.npy"))

            cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)
            depth_mask = depth > 0
            trans_mat = np.dot(align_mat, camera_poses[view])
            if not seg.any():
                return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
            workspace_mask = get_workspace_mask(cloud, seg, trans_mat)
            mask = depth_mask & workspace_mask
            cloud = cloud[mask]
            seg = seg[mask]

            idxs = np.random.choice(len(cloud), self.config.num_points, replace=True)
            cloud = cloud[idxs]
            seg = seg[idxs]

            if self.is_eval:
                return {
                    "scene": np.array([int(scene.split("_")[-1])]),
                    "view": np.array([view]),
                    "point_clouds": cloud.astype(np.float32),
                    "coors": cloud.astype(np.float32) / self.config.voxel_size,
                    "feats": np.ones_like(cloud).astype(np.float32),
                    "seg": seg.astype(np.int64),
                }

            frac_suffix = "" if self.config.fraction == 1 else f"_{self.config.fraction}"
            graspness_path = os.path.join(
                self._data_root,
                self.config.graspness_data + frac_suffix,
                scene,
                self.config.camera,
                str_view + ".npy",
            )
            fallback_path = None
            if self._graspness_fallback:
                fallback_path = os.path.join(
                    self._data_root,
                    self._graspness_fallback + frac_suffix,
                    scene,
                    self.config.camera,
                    str_view + ".npy",
                )
            graspness, has_graspness = load_graspness_aligned(
                graspness_path, depth_shape, mask, idxs, fallback_path
            )

            assert self.config.resample
            sampled = self._sample_dex_grasps_with_meta(
                scene,
                self.config.robot,
                frac_suffix,
                self.config.sample_total,
            )
            if sampled is None:
                return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
            samples, object_meta_ids_all, grasp_indices_all = sampled
            rot = samples["rotation"]
            trans = samples["translation"]

            new_rot = np.einsum("ji,njk->nik", camera_poses[view, :3, :3], rot)
            new_trans = np.einsum(
                "ji,nj->ni", camera_poses[view, :3, :3], trans - camera_poses[view, :3, 3]
            )

            grasp_points = samples["point"]
            grasp_points = np.einsum(
                "ba,nb->na", camera_poses[view, :3, :3], grasp_points - camera_poses[view, :3, 3]
            )

            centers = np.zeros((self.config.sample_total,))
            available = []
            for i in range(len(centers)):
                if len(available) >= self.config.k:
                    break
                try:
                    nearest_idx = np.linalg.norm(cloud - grasp_points[i], axis=1).argmin()
                    if np.linalg.norm(cloud[nearest_idx] - grasp_points[i]) > self.config.max_point_dis:
                        raise RuntimeError("seed too far")
                    centers[i] = nearest_idx
                    available.append(i)
                except RuntimeError:
                    pass

            if len(available) == 0:
                return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)

            indices = np.random.choice(np.array(available), self.config.k, replace=True)
            qpos = np.stack([samples[j] for j in self.joint_names], axis=-1)
            qpos = qpos[indices]
            new_rot = new_rot[indices]
            new_trans = new_trans[indices]
            centers = centers[indices]
            object_meta_ids = object_meta_ids_all[indices]
            grasp_indices = grasp_indices_all[indices]
            scene_id_int = int(scene.split("_")[-1])
            cam_pose = camera_poses[view].astype(np.float32)

            ret_dict = {
                "point_clouds": cloud.astype(np.float32),
                "coors": cloud.astype(np.float32) / self.config.voxel_size,
                "feats": np.ones_like(cloud).astype(np.float32),
                "seg": seg.astype(np.int64),
                "objectness": (seg > 0).astype(np.int64),
                "graspness": graspness.astype(np.float32),
                "rot": new_rot.astype(np.float32),
                "trans": new_trans.astype(np.float32),
                "centers": centers.astype(np.float32),
                "qpos": qpos.astype(np.float32),
                "object_meta_ids": object_meta_ids.astype(np.int64),
                "grasp_indices": grasp_indices.astype(np.int64),
                "scene_id": np.full((self.config.k,), scene_id_int, dtype=np.int64),
                "camera_pose": np.tile(cam_pose, (self.config.k, 1, 1)).astype(np.float32),
                "has_graspness": np.array([has_graspness]),
            }

            if self.is_train:
                ret_dict = self.augment_data(ret_dict)

            return ret_dict
        except Exception as e:
            if _retry == 0:
                print("Unknown error in GraspNetDatasetContactDiffusionV2")
                print(f"{cate} {scene} {view}")
                print(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)

    def augment_data(self, ret_dict: dict) -> dict:
        """Z 轴旋转增强；记录 aug_rotmat 供 cache GT 同步旋转。"""
        cloud = ret_dict["point_clouds"]
        rot = ret_dict["rot"]
        trans = ret_dict["trans"]

        theta = np.random.rand() * 2 * np.pi
        rotmat = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        ret_dict["point_clouds"] = np.einsum("ij,nj->ni", rotmat, cloud)
        ret_dict["trans"] = np.einsum("ij,nj->ni", rotmat, trans)
        ret_dict["rot"] = np.einsum("ij,njk->nik", rotmat, rot)
        ret_dict["coors"] = ret_dict["point_clouds"] / self.config.voxel_size
        ret_dict["aug_rotmat"] = rotmat
        return ret_dict
