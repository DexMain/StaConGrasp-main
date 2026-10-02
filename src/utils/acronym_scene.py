from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AcronymObject:
    object_code: str
    seg_label: int
    pose_obj_to_table: np.ndarray
    mesh_path: str


class AcronymScene:
    """Scene adapter for Acronym hash object codes and rendered seg labels."""

    def __init__(
        self,
        scene_id: str,
        data_root: str = "/data",
        mesh_root: Optional[str] = None,
        camera: str = "realsense",
        mapping_root: Optional[str] = None,
    ) -> None:
        self.scene_id = str(scene_id)
        self.split = self.scene_id.split("_")[1]
        self.data_root = data_root
        self.camera = camera
        self.mesh_root = mesh_root or os.path.join(data_root, "acronym", "meshes", "models")
        self.annotation_path = os.path.join(
            data_root,
            f"acronym_test_scenes/test_acronym_{self.split}",
            f"{self.scene_id}.npz",
        )
        self.network_input_path = os.path.join(
            data_root,
            f"acronym_test_scenes/network_input_{self.split}",
            self.scene_id,
            camera,
            "network_input.npz",
        )
        self.label_root = os.path.join(
            data_root,
            f"acronym_test_scenes/test_acronym_{self.split}_label_gt",
            camera,
            self.scene_id,
        )
        self.mapping_root = mapping_root or os.path.join(
            data_root, "acronym_test_scenes", "scene_object_mapping"
        )
        self.annotation = self._load_annotation()
        self.objects = self._load_objects()
        self.seg_to_code = self._load_or_infer_mapping()
        self.code_to_seg = {code: label for label, code in self.seg_to_code.items()}
        self._pose_by_seg = {
            self.code_to_seg[code]: self._pose_to_table(info)
            for code, info in self.annotation.items()
            if code in self.code_to_seg
        }

    def _load_annotation(self) -> Mapping[str, Mapping[str, np.ndarray]]:
        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(f"Acronym annotation not found: {self.annotation_path}")
        raw = np.load(self.annotation_path, allow_pickle=True)["arr_0"].item()
        return {str(code): info for code, info in raw.items()}

    def _load_objects(self) -> Dict[str, AcronymObject]:
        result = {}
        for code, info in self.annotation.items():
            mesh_path = os.path.join(self.mesh_root, code, "scaled.obj")
            if not os.path.isfile(mesh_path):
                mesh_path = os.path.join(self.mesh_root, code, "simplified.obj")
            if not os.path.isfile(mesh_path):
                raise FileNotFoundError(f"Acronym mesh not found for {code}: {mesh_path}")
            result[code] = AcronymObject(
                object_code=code,
                seg_label=-1,
                pose_obj_to_table=self._pose_to_table(info),
                mesh_path=mesh_path,
            )
        return result

    @staticmethod
    def _pose_to_table(info: Mapping[str, np.ndarray]) -> np.ndarray:
        # Acronym stores quaternions as xyzw; project evaluation uses the same order.
        quat_xyzw = np.asarray(info["rest_pose_quat"], dtype=np.float64).reshape(4)
        trans = np.asarray(info["rest_pose_trans"], dtype=np.float64).reshape(3).copy()
        trans[2] -= 0.05
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        pose[:3, 3] = trans.astype(np.float32)
        return pose

    def _mapping_path(self) -> str:
        return os.path.join(self.mapping_root, f"{self.scene_id}.json")

    def _load_or_infer_mapping(self) -> Dict[int, str]:
        path = self._mapping_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            mapping = {int(k): str(v) for k, v in payload["seg_label_to_object_code"].items()}
            if set(mapping.values()).issubset(self.annotation):
                return mapping

        data = np.load(self.network_input_path)
        pc = np.asarray(data["pc"], dtype=np.float32)[..., :3]
        seg = np.asarray(data["seg"], dtype=np.int64)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float32)
        data.close()

        labels = sorted(int(x) for x in np.unique(seg) if int(x) > 0)
        if len(labels) > len(self.annotation):
            raise ValueError(
                f"Acronym scene {self.scene_id}: {len(labels)} nonzero seg labels "
                f"exceed {len(self.annotation)} annotated objects"
            )

        # Convert network-input points to table coordinates and compare visible
        # label centroids with annotated object translations. The same mapping is
        # stable over views, so aggregate all views before assignment.
        points_h = np.concatenate(
            [pc, np.ones((*pc.shape[:2], 1), dtype=np.float32)], axis=-1
        )
        points_table = np.einsum("vij,vnj->vni", extrinsics, points_h)[..., :3]
        label_centroids = []
        for label in labels:
            masks = seg == label
            counts = masks.sum(axis=1)
            visible = counts > 0
            if not np.any(visible):
                label_centroids.append(np.full(3, np.nan, dtype=np.float32))
                continue
            centroids = np.stack(
                [points_table[v, masks[v]].mean(axis=0) for v in np.flatnonzero(visible)]
            )
            weights = counts[visible].astype(np.float64)
            label_centroids.append(np.average(centroids, axis=0, weights=weights))
        label_centroids = np.asarray(label_centroids, dtype=np.float64)

        codes = sorted(self.annotation)
        object_centroids = np.asarray(
            [self.annotation[code]["rest_pose_trans"] for code in codes], dtype=np.float64
        )
        object_centroids[:, 2] -= 0.05
        cost = np.linalg.norm(label_centroids[:, None, :] - object_centroids[None, :, :], axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {labels[int(row)]: codes[int(col)] for row, col in zip(row_ind, col_ind)}
        if len(mapping) != len(labels):
            raise ValueError(f"Could not establish a complete visible Acronym mapping for {self.scene_id}")

        try:
            os.makedirs(self.mapping_root, exist_ok=True)
        except OSError:
            self.mapping_root = os.path.abspath(
                os.path.join("outputs", "acronym_scene_mapping")
            )
            os.makedirs(self.mapping_root, exist_ok=True)
            path = self._mapping_path()
        payload = {
            "scene_id": self.scene_id,
            "method": "table_centroid_hungarian_visible_objects",
            "seg_label_to_object_code": {str(k): v for k, v in mapping.items()},
            "object_code_to_seg_label": {v: k for k, v in mapping.items()},
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        return mapping

    def pose_by_seg_label(self, seg_label: int) -> np.ndarray:
        return self._pose_by_seg[int(seg_label)].copy()

    def object_code(self, seg_label: int) -> str:
        return self.seg_to_code[int(seg_label)]

    def mesh_path(self, seg_label: int) -> str:
        return self.objects[self.object_code(seg_label)].mesh_path

    def sdf_path(self, seg_label: int, sdf_root: str, grid_size: int = 64) -> str:
        return os.path.join(sdf_root, self.object_code(seg_label), f"sdf_{int(grid_size)}.npz")

    def pose_obj_to_cam(self, seg_label: int, extrinsic_cam_to_table: np.ndarray) -> np.ndarray:
        return np.linalg.inv(np.asarray(extrinsic_cam_to_table, dtype=np.float32)) @ self.pose_by_seg_label(seg_label)
