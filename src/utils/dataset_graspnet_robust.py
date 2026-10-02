"""
GraspNetDataset 的健壮版本：正确对齐 graspness 与 mask 后点云，避免索引越界。

dex_graspness_new 为 FPS 子采样，长度与 workspace mask 后点云不一致；
dex_graspness_rebuild 与 mask 一一对应。本模块在长度不匹配时自动 fallback。
"""

from __future__ import annotations

import os
import random

import numpy as np
import scipy.io as scio
from PIL import Image

from utils.dataset import GraspNetDataset
from utils.pc import depth_image_to_point_cloud, get_workspace_mask


def load_graspness_aligned(
    graspness_path: str,
    depth_shape: tuple[int, int],
    mask: np.ndarray,
    sample_idxs: np.ndarray,
    fallback_path: str | None = None,
) -> tuple[np.ndarray, int]:
    """
    将 graspness 与 mask 后点云对齐，再按 sample_idxs 子采样。

    支持三种存储格式：
    - (H, W) 或 H*W：先 mask 再 subsample
    - len == mask.sum()：与 mask 后点云逐点对应
    """
    n_mask = int(mask.sum())
    num_points = len(sample_idxs)

    def _try_path(path: str | None) -> np.ndarray | None:
        if path is None or not os.path.exists(path):
            return None
        raw = np.load(path).reshape(-1)
        h, w = depth_shape
        if raw.size == h * w:
            aligned = raw.reshape(h, w)[mask]
        elif raw.size == n_mask:
            aligned = raw
        else:
            return None
        if len(aligned) != n_mask:
            return None
        if sample_idxs.max(initial=0) >= len(aligned):
            return None
        return np.log(aligned[sample_idxs].astype(np.float64) + 1e-3).astype(np.float32)

    graspness = _try_path(graspness_path)
    if graspness is not None:
        return graspness, 1

    graspness = _try_path(fallback_path)
    if graspness is not None:
        return graspness, 1

    return np.zeros((num_points,), dtype=np.float32), 0


class GraspNetDatasetRobust(GraspNetDataset):
    """对齐 graspness 索引；限制失败重试次数，减少日志刷屏。"""

    def __init__(self, *args, max_load_retries: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_load_retries = max_load_retries
        self._data_root = getattr(self.config, "data_root", None) or "/data"
        self._graspness_fallback = getattr(
            self.config, "graspness_fallback", "dex_graspness_rebuild"
        ) or None

    def __getitem__(self, dataset_idx: int, _retry: int = 0):
        if _retry >= self.max_load_retries:
            raise RuntimeError(
                f"GraspNetDatasetRobust: exceeded {self.max_load_retries} load retries"
            )

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
            trans = np.dot(align_mat, camera_poses[view])
            if not seg.any():
                return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
            workspace_mask = get_workspace_mask(cloud, seg, trans)
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

            if self.config.robot == "gripper":
                poses_6d = np.load(
                    os.path.join(
                        self._data_root, "gripper_grasps", scene, self.config.camera, "poses.npy"
                    )
                )[:: self.config.fraction]
                grasp_points = np.load(
                    os.path.join(
                        self._data_root, "gripper_grasps", scene, self.config.camera, "points.npy"
                    )
                )[:: self.config.fraction]

                assert self.config.sample_total >= self.config.k
                if self.config.resample:
                    can_grasp_ids = np.unique(poses_6d[:, -1])
                    rand_idxs = np.random.randint(0, len(can_grasp_ids), self.config.sample_total)
                    samples = []
                    point_samples = []
                    for i, idx in enumerate(can_grasp_ids):
                        num = (rand_idxs == i).sum()
                        obj_poses_6d = poses_6d[poses_6d[:, -1] == idx]
                        obj_rand_idxs = np.random.choice(len(obj_poses_6d), num, replace=True)
                        samples.append(obj_poses_6d[obj_rand_idxs])
                        point_samples.append(grasp_points[poses_6d[:, -1] == idx][obj_rand_idxs])
                    poses_6d = np.concatenate(samples)
                    point_samples = np.concatenate(point_samples)
                    permute = np.random.permutation(len(poses_6d))
                    poses_6d = poses_6d[permute]
                    point_samples = point_samples[permute]
                else:
                    pose_idxs = np.random.choice(len(poses_6d), self.config.sample_total, replace=True)
                    poses_6d = poses_6d[pose_idxs]
                    point_samples = grasp_points[pose_idxs]
                rot = poses_6d[:, -13:-4].reshape(-1, 3, 3)
                trans = poses_6d[:, -4:-1]

            else:
                data_root = self._data_root
                assert self.config.resample
                grasp_files = os.listdir(
                    os.path.join(data_root, "dex_grasps_new" + frac_suffix, scene, self.config.robot)
                )
                if len(grasp_files) == 0:
                    return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
                rand_idxs = np.random.randint(0, len(grasp_files), self.config.sample_total)
                samples = []
                for i, f in enumerate(grasp_files):
                    num = (rand_idxs == i).sum()
                    grasps = np.load(
                        os.path.join(
                            data_root, "dex_grasps_new" + frac_suffix, scene, self.config.robot, f
                        )
                    )
                    sel_idxs = np.random.choice(len(grasps["point"]), num, replace=True)
                    samples.append({k: grasps[k][sel_idxs] for k in grasps.keys()})
                permute = np.random.permutation(self.config.sample_total)
                samples = {
                    k: np.concatenate([sample[k] for sample in samples])[permute]
                    for k in samples[0].keys()
                }
                rot = samples["rotation"]
                trans = samples["translation"]

            new_rot = np.einsum("ji,njk->nik", camera_poses[view, :3, :3], rot)
            new_trans = np.einsum(
                "ji,nj->ni", camera_poses[view, :3, :3], trans - camera_poses[view, :3, 3]
            )

            grasp_points = point_samples if self.config.robot == "gripper" else samples["point"]
            grasp_points = np.einsum(
                "ba,nb->na", camera_poses[view, :3, :3], grasp_points - camera_poses[view, :3, 3]
            )
            centers = np.zeros((self.config.sample_total,))
            available = []
            dis = []
            for i in range(len(centers)):
                if len(available) >= self.config.k:
                    break
                try:
                    nearest_idx = np.linalg.norm(cloud - grasp_points[i], axis=1).argmin()
                    if np.linalg.norm(cloud[nearest_idx] - grasp_points[i]) > self.config.max_point_dis:
                        raise Exception
                    centers[i] = nearest_idx
                    dis.append(np.linalg.norm(cloud[nearest_idx] - grasp_points[i]))
                    available.append(i)
                except Exception:
                    pass
            if len(available) == 0:
                return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
            dis = np.array(dis)
            indices = np.random.choice(np.array(available), self.config.k, replace=True)
            if self.config.robot == "gripper":
                poses_6d = poses_6d[indices]
            else:
                qpos = np.stack([samples[j] for j in self.joint_names], axis=-1)
                qpos = qpos[indices]

            new_rot = new_rot[indices]
            new_trans = new_trans[indices]
            centers = centers[indices]

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
                "has_graspness": np.array([has_graspness]),
            }

            if self.config.robot == "gripper":
                ret_dict.update({"qpos": poses_6d[:, [1]].astype(np.float32)})
            else:
                ret_dict.update({"qpos": qpos.astype(np.float32)})

            if self.is_train:
                ret_dict = self.augment_data(ret_dict)

            return ret_dict
        except Exception as e:
            if _retry == 0:
                print("Unknow error in loading dataset")
                print(f"{cate} {scene} {view}")
                print(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1), _retry + 1)
