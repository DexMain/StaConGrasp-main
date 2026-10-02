"""
GraspNetDataset 扩展：返回 object_meta_ids，并可选 dex_grasps_success_indices 筛选。
"""

from __future__ import annotations

import os
import random

import numpy as np

from utils.contact_gt_surface import object_code_to_meta_id
from utils.dataset_graspnet_robust import GraspNetDatasetRobust


class GraspNetDatasetContactDiffusion(GraspNetDatasetRobust):
    def __init__(
        self,
        *args,
        use_success_filter: bool = True,
        success_indices_root: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_success_filter = bool(use_success_filter)
        self._success_root = success_indices_root or os.path.join(
            self._data_root, "dex_grasps_success_indices"
        )

    def _filter_grasps_by_success(
        self,
        scene: str,
        robot: str,
        grasp_file: str,
        grasps: dict,
    ) -> dict | None:
        if not self.use_success_filter:
            return grasps
        success_path = os.path.join(
            self._success_root, scene, robot, grasp_file
        )
        if not os.path.isfile(success_path):
            return grasps
        success_idx = np.load(success_path)["success_indices"]
        if len(success_idx) == 0:
            return None
        return {k: v[success_idx] for k, v in grasps.items()}

    def _sample_dex_grasps_with_meta(
        self,
        scene: str,
        robot: str,
        frac_suffix: str,
        sample_total: int,
    ) -> tuple[dict, np.ndarray] | None:
        data_root = self._data_root
        grasp_dir = os.path.join(
            data_root, "dex_grasps_new" + frac_suffix, scene, robot
        )
        if not os.path.isdir(grasp_dir):
            return None

        valid_pools: list[tuple[str, dict]] = []
        for grasp_file in sorted(os.listdir(grasp_dir)):
            if not grasp_file.endswith(".npz"):
                continue
            grasps = np.load(os.path.join(grasp_dir, grasp_file))
            grasps = self._filter_grasps_by_success(
                scene, robot, grasp_file, dict(grasps)
            )
            if grasps is None or len(grasps.get("point", [])) == 0:
                continue
            valid_pools.append((grasp_file, grasps))

        if not valid_pools:
            return None

        merged_samples: dict[str, list] = {}
        object_meta_ids: list[int] = []
        for _ in range(sample_total):
            grasp_file, grasps = valid_pools[np.random.randint(0, len(valid_pools))]
            n_grasp = len(grasps["point"])
            sel = int(np.random.randint(0, n_grasp))
            for key, val in grasps.items():
                merged_samples.setdefault(key, []).append(val[sel])
            object_code = os.path.splitext(grasp_file)[0]
            object_meta_ids.append(object_code_to_meta_id(object_code))

        merged = {k: np.stack(v, axis=0) for k, v in merged_samples.items()}
        permute = np.random.permutation(sample_total)
        merged = {k: v[permute] for k, v in merged.items()}
        meta_arr = np.array(object_meta_ids, dtype=np.int64)[permute]
        return merged, meta_arr

    def __getitem__(self, dataset_idx: int, _retry: int = 0):
        if _retry >= self.max_load_retries:
            raise RuntimeError(
                f"GraspNetDatasetContactDiffusion: exceeded {self.max_load_retries} retries"
            )

        if self.config.robot == "gripper":
            ret = super().__getitem__(dataset_idx, _retry)
            if not self.is_eval and isinstance(ret, dict):
                ret["object_meta_ids"] = np.zeros(ret["trans"].shape[0], dtype=np.int64)
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
            samples, object_meta_ids_all = sampled
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
                "has_graspness": np.array([has_graspness]),
            }

            if self.is_train:
                ret_dict = self.augment_data(ret_dict)

            return ret_dict
        except Exception as e:
            if _retry == 0:
                print("Unknown error in GraspNetDatasetContactDiffusion")
                print(f"{cate} {scene} {view}")
                print(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
