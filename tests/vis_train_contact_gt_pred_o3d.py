"""
Open3D visualization for training contact GT and predicted contacts.

"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader

import train_contact_diffusion_base as train_v2b
from network.contact_diffusion_base import (
    _forward_finger_representatives as _forward_finger_representatives_abs,
    fit_pose_to_contacts as fit_pose_to_contacts_abs,
)
from network.contact_diffusion_pointnet_offset import (
    _forward_finger_representatives as _forward_finger_representatives_offset,
    fit_pose_to_contacts as fit_pose_to_contacts_offset,
)
from eval.contact_stability_infer_pointnext import (
    sample_and_rank_contacts as sample_and_rank_contacts_abs,
)
from eval.contact_stability_infer_pointnet import (
    sample_and_rank_contacts as sample_and_rank_contacts_offset,
)
from network.contact_diffusion_base import (
    load_contact_diffusion_net as load_contact_diffusion_abs,
)
from network.contact_diffusion_pointnet_offset import (
    load_contact_diffusion_net as load_contact_diffusion_offset,
)
from network.contact_stability_base import (
    load_contact_stability_net as load_contact_stability_abs,
)
from network.contact_stability_pointnet_offset import (
    load_contact_stability_net as load_contact_stability_offset,
)
from network.graspness_sample_with_feature import (
    GraspnessSampleWithFeature,
)
from utils.config import load_config
from utils.contact_gt_surface import (
    crop_object_pc_by_meta_id,
    unflatten_contacts,
)
from utils.dataset import get_sparse_tensor
from utils.contact_gt_ibs_thumb_split import ContactGTv2bCache
from utils.dataset import minkowski_collate_fn
from utils.dataset_contact_diffusion_fps import (
    GraspNetDatasetContactDiffusionV2,
)
from utils.robot_model import RobotModel
from utils.util import set_seed
from eval.topk_select import select_top_n_per_view_with_feature
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


MAIN_OBJECT_RGB = _hex_to_rgb("#ff6600")
GT_OBJECT_RGB = _hex_to_rgb("#00C853")
NON_MAIN_OBJECT_RGB = _hex_to_rgb("#b0b0b0")
GRASPED_OBJECT_RGB = _hex_to_rgb("#5C9FD6")
HAND_MESH_RGB = _hex_to_rgb("#78909C")
PRED_HAND_MESH_RGB = _hex_to_rgb("#FF7043")
SCENE_GREY_RGB = (0.78, 0.78, 0.78)
TARGET_HIGHLIGHT_RGB = (1.0, 0.42, 0.04)

FINGER_COLORS_HEX = ["#FF0033", "#FFCC00", "#00FFFF", "#FF33FF"]
FINGER_COLORS = np.array([_hex_to_rgb(c) for c in FINGER_COLORS_HEX], dtype=np.float64)
FINGER_LABELS = ["finger0", "finger1", "finger2", "finger3"]
GT_V2_IBS_RGB = _hex_to_rgb("#FF00FF")
GT_V2_FB_RGB = (1.0, 1.0, 1.0)
MARKER_OUTLINE_RGB = (0.07, 0.07, 0.07)


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_teacher(config, device: torch.device) -> GraspnessSampleWithFeature:
    config.model["voxel_size"] = config.data.voxel_size
    teacher = GraspnessSampleWithFeature(config.model)
    ckpt = torch.load(config.ckpt, map_location="cpu")
    teacher.load_state_dict(ckpt["model"], strict=False)
    freeze(teacher)
    teacher.to(device)
    return teacher


def load_contact_model(
    contact_ckpt: str,
    contact_stability_ckpt: str,
    device: torch.device,
):
    if bool(contact_ckpt) == bool(contact_stability_ckpt):
        raise ValueError("Please provide exactly one of --contact_ckpt or --contact_stability_ckpt")

    ckpt_path = contact_stability_ckpt or contact_ckpt
    meta = torch.load(ckpt_path, map_location="cpu")
    version = str(meta.get("version", ""))

    if contact_stability_ckpt:
        if version == "contact_stability_v4b_offset":
            model = load_contact_stability_offset(contact_stability_ckpt, device=device)
            return model, "stability_offset", version
        model = load_contact_stability_abs(contact_stability_ckpt, device=device)
        return model, "stability_absolute", version or "contact_stability_legacy"

    if version == "contact_diffusion_v2b_offset":
        model = load_contact_diffusion_offset(contact_ckpt, device=device)
        return model, "contact_offset", version
    model = load_contact_diffusion_abs(contact_ckpt, device=device)
    return model, "contact_absolute", version or "contact_diffusion_legacy"


def make_pcd(points: np.ndarray, color: Tuple[float, float, float], stride: int = 1):
    pts = np.asarray(points, dtype=np.float64)
    if stride > 1:
        pts = pts[::stride]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)
    return pcd


def pick_scene_cloud_and_seg_for_seed(
    data: dict, seed_np: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the source scene cloud whose sampled points are closest to this seed."""
    scene_clouds = data["point_clouds"].detach().cpu().numpy()
    scene_segs = data["seg"].detach().cpu().numpy()
    seed_np = np.asarray(seed_np, dtype=np.float32)
    best_i = 0
    best_d = float("inf")
    for i in range(scene_clouds.shape[0]):
        pts = scene_clouds[i]
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.shape[0] == 0:
            continue
        d = float(np.linalg.norm(pts - seed_np[None], axis=1).min())
        if d < best_d:
            best_d = d
            best_i = i
    return scene_clouds[best_i], scene_segs[best_i]


def estimate_contact_meta_id(
    contact_pts: np.ndarray,
    scene_pc: np.ndarray,
    scene_seg: np.ndarray,
) -> Optional[int]:
    """Estimate grasped object id by nearest segmented point majority vote."""
    pts = np.asarray(contact_pts, dtype=np.float32)
    if pts.size == 0 or scene_pc.size == 0:
        return None
    valid_scene = np.isfinite(scene_pc).all(axis=1) & (scene_seg.astype(np.int64) > 0)
    pc = scene_pc[valid_scene]
    seg = scene_seg[valid_scene].astype(np.int64)
    if pc.shape[0] == 0:
        return None
    votes = []
    for p in pts.reshape(-1, 3):
        if not np.isfinite(p).all():
            continue
        d = np.linalg.norm(pc - p[None], axis=1)
        votes.append(int(seg[int(np.argmin(d))]))
    if not votes:
        return None
    ids, counts = np.unique(np.asarray(votes, dtype=np.int64), return_counts=True)
    return int(ids[int(np.argmax(counts))])


def _subsample_points(
    pts: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """max_points <= 0 表示不 subsample，保留该物体全部点。"""
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts
    sel = rng.choice(pts.shape[0], int(max_points), replace=False)
    return pts[sel]


def extract_segmented_objects_cam(
    pc_cam: np.ndarray,
    seg_cam: np.ndarray,
    main_object_id: Optional[int] = None,
    gt_object_id: Optional[int] = None,
    grasped_object_id: Optional[int] = None,
    max_points_per_obj: int = 20000,
    max_points_per_bg_obj: int = 6000,
    seed: int = 0,
) -> List[dict]:
    """按 seg id 拆分物体点云（network camera 系），配色对齐 vis_contact_stability。"""
    rng = np.random.default_rng(seed)
    objects: List[dict] = []
    obj_ids = np.unique(seg_cam.astype(np.int64))
    obj_ids = obj_ids[obj_ids > 0]
    for oid in sorted(int(x) for x in obj_ids):
        mask = seg_cam.astype(np.int64) == oid
        pts = pc_cam[mask]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            continue
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
        is_highlight = is_gt or is_main or is_grasped
        cap = int(max_points_per_obj if is_highlight else max_points_per_bg_obj)
        pts = _subsample_points(pts, cap, rng)
        if is_gt:
            color = GT_OBJECT_RGB
        elif is_main:
            color = MAIN_OBJECT_RGB
        elif is_grasped:
            color = GRASPED_OBJECT_RGB
        else:
            color = NON_MAIN_OBJECT_RGB
        objects.append(
            {
                "obj_id": oid,
                "points": pts.astype(np.float32, copy=False),
                "color": color,
                "is_gt": is_gt,
                "is_main": is_main,
                "is_grasped": is_grasped,
            }
        )
    return objects


def get_hand_mesh_cam(
    robot_model: RobotModel,
    trans: np.ndarray,
    rot: np.ndarray,
    qpos: np.ndarray,
    joint_names: List[str],
    color: Tuple[float, float, float],
    mesh_type: str = "collision",
) -> o3d.geometry.TriangleMesh:
    """训练 pose（camera 系）下的手 mesh，逻辑对齐 Vis.robot_plotly / vis_ibs_o3d。"""
    trans_t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    rot_t = torch.from_numpy(rot.astype(np.float32)).unsqueeze(0)
    qpos_dict = {
        name: torch.tensor([[float(qpos[i])]], dtype=torch.float32)
        for i, name in enumerate(joint_names)
    }
    link_trans, link_rots = robot_model.forward_kinematics(qpos_dict)
    link_trans = {
        k: torch.einsum("nab,nb->na", rot_t, v) + trans_t for k, v in link_trans.items()
    }
    link_rots = {
        k: torch.einsum("nab,nbc->nac", rot_t, v) for k, v in link_rots.items()
    }

    combined = o3d.geometry.TriangleMesh()
    for link_name in link_trans:
        if link_name not in robot_model._geometry:
            continue
        if f"{mesh_type}_vertices" not in robot_model._geometry[link_name]:
            continue
        vertices, faces = robot_model.get_link_mesh(link_name, mesh_type)
        if vertices.numel() == 0:
            continue
        link_t = link_trans[link_name][0].detach().cpu().numpy()
        link_r = link_rots[link_name][0].detach().cpu().numpy()
        verts = (link_r @ vertices.numpy().T).T + link_t
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.numpy())
        combined += mesh

    combined.compute_vertex_normals()
    combined.paint_uniform_color(color)
    return combined


def paint_mesh_ghost(
    mesh: o3d.geometry.TriangleMesh,
    color: Tuple[float, float, float],
    opacity: float,
    background: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    """在固定背景上混合颜色，模拟半透明；保持 legacy Visualizer 背景不变。"""
    out = o3d.geometry.TriangleMesh(mesh)
    a = float(np.clip(opacity, 0.0, 1.0))
    bg = np.asarray(background, dtype=np.float64)
    c = np.asarray(color, dtype=np.float64)
    ghost = tuple((a * c + (1.0 - a) * bg).tolist())
    out.paint_uniform_color(ghost)
    out.compute_vertex_normals()
    return out


def mesh_wireframe_lines(
    mesh: o3d.geometry.TriangleMesh,
    color: Tuple[float, float, float],
) -> o3d.geometry.LineSet:
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color(color)
    return lines


def object_points_from_seg(
    scene_pc: np.ndarray,
    scene_seg: np.ndarray,
    meta_id: Optional[int],
    max_points: int,
    seed: int,
) -> np.ndarray:
    if meta_id is None:
        return np.zeros((0, 3), dtype=np.float32)
    mask = scene_seg.astype(np.int64) == int(meta_id)
    pts = scene_pc[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], int(max_points), replace=False)
        pts = pts[idx]
    return pts.astype(np.float32, copy=False)


def nearest_points_on_cloud(
    query: np.ndarray,
    cloud: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    cloud = np.asarray(cloud, dtype=np.float32)
    nearest = np.zeros_like(query, dtype=np.float32)
    dist = np.full((query.shape[0],), np.inf, dtype=np.float32)
    if query.shape[0] == 0 or cloud.shape[0] == 0:
        return nearest, dist
    for i, p in enumerate(query):
        if not np.isfinite(p).all():
            continue
        d = np.linalg.norm(cloud - p[None], axis=1)
        j = int(np.argmin(d))
        nearest[i] = cloud[j]
        dist[i] = float(d[j])
    return nearest, dist


def nearest_seg_ids(
    query: np.ndarray,
    scene_pc: np.ndarray,
    scene_seg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    valid_scene = np.isfinite(scene_pc).all(axis=1) & (scene_seg.astype(np.int64) > 0)
    pc = scene_pc[valid_scene]
    seg = scene_seg[valid_scene].astype(np.int64)
    ids = np.full((query.shape[0],), -1, dtype=np.int64)
    dist = np.full((query.shape[0],), np.inf, dtype=np.float32)
    if pc.shape[0] == 0:
        return ids, dist
    for i, p in enumerate(query):
        if not np.isfinite(p).all():
            continue
        d = np.linalg.norm(pc - p[None], axis=1)
        j = int(np.argmin(d))
        ids[i] = int(seg[j])
        dist[i] = float(d[j])
    return ids, dist


def estimate_surface_normal(
    center: np.ndarray,
    cloud: np.ndarray,
    k: int = 24,
) -> np.ndarray:
    """物体点云局部 PCA 法向（用于贴面 contact disc）。"""
    center = np.asarray(center, dtype=np.float64)
    cloud = np.asarray(cloud, dtype=np.float64)
    valid = cloud[np.isfinite(cloud).all(axis=1)]
    if valid.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.linalg.norm(valid - center[None], axis=1)
    k = min(int(k), valid.shape[0])
    nn_idx = np.argpartition(d, k - 1)[:k]
    pts = valid[nn_idx] - center
    cov = (pts.T @ pts) / max(len(nn_idx), 1)
    _, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / nrm


def _tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(normal[2])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(normal, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(normal, u)
    return u, v


def _circle_points_on_plane(
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius: float,
    segments: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    return center + radius * (
        np.cos(angles)[:, None] * u[None] + np.sin(angles)[:, None] * v[None]
    )


def make_contact_ring_lines(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    color: Tuple[float, float, float],
    segments: int = 32,
) -> o3d.geometry.LineSet:
    """○ 空心圆环（对齐 vis_contact_stability circle-open）。"""
    u, v = _tangent_basis(normal)
    circle = _circle_points_on_plane(center, u, v, radius, segments)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(circle.astype(np.float64))
    idx = np.arange(segments, dtype=np.int32)
    lines = np.stack([idx, np.roll(idx, -1)], axis=1)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector(np.tile(color, (segments, 1)))
    return line


def _lift_from_surface(
    center: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """沿法向微抬，减少与物体点云 z-fighting。"""
    if float(offset) <= 0.0:
        return np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    return np.asarray(center, dtype=np.float64) + float(offset) * n


def make_contact_disc_mesh(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    color: Tuple[float, float, float],
    with_outline: bool = True,
    segments: int = 32,
    with_outer_ring: bool = True,
) -> List:
    """● 贴面实心圆盘 + 描边 + 外圈（对齐 vis_contact_stability，加强可见性）。"""
    center = np.asarray(center, dtype=np.float64)
    u, v = _tangent_basis(normal)
    circle = _circle_points_on_plane(center, u, v, radius, segments)
    verts = np.vstack([center[None], circle])
    tris = []
    for i in range(segments):
        tris.append([0, 1 + i, 1 + (i + 1) % segments])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    out: List = [mesh]
    if with_outline:
        out.append(
            make_contact_ring_lines(center, normal, radius, MARKER_OUTLINE_RGB, segments)
        )
    if with_outer_ring:
        out.append(
            make_contact_ring_lines(center, normal, radius * 1.18, color, segments)
        )
    return out


def make_contact_symbol_lines(
    center: np.ndarray,
    normal: np.ndarray,
    size: float,
    color: Tuple[float, float, float],
    symbol: str,
) -> Optional[o3d.geometry.LineSet]:
    """× / ◇ 符号（GT / FK tip）。"""
    center = np.asarray(center, dtype=np.float64)
    u, v = _tangent_basis(normal)
    s = float(size)
    if symbol == "x":
        pts = np.stack(
            [
                center + s * (u + v),
                center - s * (u + v),
                center + s * (u - v),
                center - s * (u - v),
            ],
            axis=0,
        )
        lines = np.array([[0, 1], [2, 3]], dtype=np.int32)
    elif symbol == "diamond":
        pts = np.stack(
            [
                center + s * u,
                center + s * v,
                center - s * u,
                center - s * v,
            ],
            axis=0,
        )
        lines = np.stack(
            [np.arange(4), np.roll(np.arange(4), -1)], axis=1
        ).astype(np.int32)
    elif symbol == "diamond-open":
        pts = np.stack(
            [
                center + 0.85 * s * u,
                center + 0.85 * s * v,
                center - 0.85 * s * u,
                center - 0.85 * s * v,
            ],
            axis=0,
        )
        lines = np.stack(
            [np.arange(4), np.roll(np.arange(4), -1)], axis=1
        ).astype(np.int32)
    else:
        return None
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (lines.shape[0], 1))
    )
    return line


def make_contact_surface_patch(
    center: np.ndarray,
    cloud: np.ndarray,
    radius: float,
    color: Tuple[float, float, float],
    max_points: int = 180,
) -> Optional[o3d.geometry.PointCloud]:
    """接触点附近物体表面点高亮，形成“接触面”观感。"""
    center = np.asarray(center, dtype=np.float32)
    cloud = np.asarray(cloud, dtype=np.float32)
    valid = cloud[np.isfinite(cloud).all(axis=1)]
    if valid.shape[0] == 0:
        return None
    d = np.linalg.norm(valid - center[None], axis=1)
    mask = d <= float(radius)
    pts = valid[mask]
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], int(max_points), replace=False)
        pts = pts[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def append_contact_marker(
    geometries: List,
    center: np.ndarray,
    surface_cloud: np.ndarray,
    color: Tuple[float, float, float],
    marker_style: str,
    kind: str,
    disc_radius: float,
    ring_radius: float,
    patch_radius: float,
    with_surface_patch: bool,
    normal_offset: float = 0.0008,
    segments: int = 32,
) -> None:
    """
    kind: pred | pred_proj | gt_ibs | gt_fb | fk_tip
    marker_style: disc（默认，对齐 vis_contact_stability）| sphere（旧行为）
    """
    center = np.asarray(center, dtype=np.float64)
    if not np.isfinite(center).all():
        return
    if marker_style == "sphere":
        r = disc_radius if kind in ("pred", "gt_ibs", "gt_fb") else ring_radius
        geometries.append(make_sphere(center, r, color))
        return

    normal = estimate_surface_normal(center, surface_cloud)
    draw_center = _lift_from_surface(center, normal, normal_offset)
    if kind == "pred":
        geometries.extend(
            make_contact_disc_mesh(
                draw_center,
                normal,
                disc_radius,
                color,
                with_outline=True,
                segments=segments,
                with_outer_ring=True,
            )
        )
        if with_surface_patch:
            patch = make_contact_surface_patch(
                center, surface_cloud, patch_radius, color, max_points=180
            )
            if patch is not None:
                geometries.append(patch)
    elif kind == "pred_proj":
        proj_center = _lift_from_surface(center, normal, normal_offset * 0.5)
        geometries.append(
            make_contact_ring_lines(proj_center, normal, ring_radius, color, segments)
        )
        geometries.append(
            make_contact_ring_lines(
                proj_center, normal, ring_radius * 0.72, MARKER_OUTLINE_RGB, segments
            )
        )
    elif kind == "gt_ibs":
        gt_center = _lift_from_surface(center, normal, normal_offset)
        sym = make_contact_symbol_lines(
            gt_center, normal, disc_radius * 1.25, GT_V2_IBS_RGB, "x"
        )
        if sym is not None:
            geometries.append(sym)
        if with_surface_patch:
            patch = make_contact_surface_patch(
                center, surface_cloud, patch_radius, GT_V2_IBS_RGB, max_points=180
            )
            if patch is not None:
                geometries.append(patch)
    elif kind == "gt_fb":
        sym = make_contact_symbol_lines(
            _lift_from_surface(center, normal, normal_offset),
            normal,
            disc_radius * 1.1,
            GT_V2_FB_RGB,
            "diamond-open",
        )
        if sym is not None:
            geometries.append(sym)
    elif kind == "fk_tip":
        sym = make_contact_symbol_lines(
            _lift_from_surface(center, normal, normal_offset),
            normal,
            disc_radius * 1.05,
            color,
            "diamond",
        )
        if sym is not None:
            geometries.append(sym)


def make_sphere(
    center: np.ndarray,
    radius: float,
    color: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    mesh.translate(np.asarray(center, dtype=np.float64))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def make_lines(
    starts: np.ndarray,
    ends: np.ndarray,
    color: Tuple[float, float, float],
) -> Optional[o3d.geometry.LineSet]:
    if starts.size == 0 or ends.size == 0:
        return None
    pts = np.vstack([starts, ends]).astype(np.float64)
    n = starts.shape[0]
    lines = np.stack([np.arange(n), np.arange(n, 2 * n)], axis=1)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    return line_set


def nearest_dist(src: np.ndarray, dst: np.ndarray, chunk: int = 8192) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.full((src.shape[0],), np.inf, dtype=np.float32)
    for start in range(0, dst.shape[0], chunk):
        obj = dst[start : start + chunk]
        d2 = ((src[:, None, :] - obj[None, :, :]) ** 2).sum(axis=-1)
        out = np.minimum(out, np.sqrt(d2.min(axis=1)))
    return out


def describe_dist(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{name}: empty")
        return
    print(
        f"{name}: mean={values.mean():.5f}m "
        f"median={np.median(values):.5f}m max={values.max():.5f}m"
    )


def contact_fit_backend(model_kind: str) -> Tuple[Callable, Callable]:
    if model_kind in ("stability_offset", "contact_offset"):
        return fit_pose_to_contacts_offset, _forward_finger_representatives_offset
    return fit_pose_to_contacts_abs, _forward_finger_representatives_abs


def fit_hand_pose_from_pred_contacts(
    hand_provider,
    init_trans: torch.Tensor,
    init_rot: torch.Tensor,
    init_qpos: torch.Tensor,
    target_contacts: torch.Tensor,
    object_pc: torch.Tensor,
    cd_cfg,
    model_kind: str,
    fit_steps: int = 32,
    fit_max_trans_delta: float = 0.10,
    fit_max_q_delta: float = 0.30,
    fit_max_rot_delta: float = 0.40,
    fit_w_pose: float = 0.01,
    use_model_fit_cfg: bool = False,
):
    fit_fn, tip_fn = contact_fit_backend(model_kind)
    fit_cfg = cd_cfg
    if not use_model_fit_cfg:
        fit_cfg = replace(
            cd_cfg,
            fit_steps=int(fit_steps),
            fit_max_trans_delta=float(fit_max_trans_delta),
            fit_max_q_delta=float(fit_max_q_delta),
            fit_max_rot_delta=float(fit_max_rot_delta),
            fit_w_pose=float(fit_w_pose),
        )
    elif int(fit_steps) > 0:
        fit_cfg = replace(cd_cfg, fit_steps=int(fit_steps))
    trans_fit, rot_fit, qpos_fit, fit_log = fit_fn(
        hand_provider,
        init_trans,
        init_rot,
        init_qpos,
        target_contacts,
        object_pc=object_pc,
        cfg=fit_cfg,
    )
    with torch.no_grad():
        tips_fit = tip_fn(
            hand_provider,
            trans_fit,
            rot_fit,
            qpos_fit,
            object_pc=object_pc,
            num_fingers=int(cd_cfg.num_fingertips),
        )
        tip_err = torch.linalg.norm(tips_fit - target_contacts, dim=-1)
        tip_mse = torch.mean((tips_fit - target_contacts) ** 2).item()
        trans_delta = torch.norm(trans_fit - init_trans, dim=-1).item()
        q_delta = torch.norm(qpos_fit - init_qpos, dim=-1).item()
    stats = {
        "fit_loss": float(fit_log["fit_loss"].item()),
        "tip_mse": tip_mse,
        "tip_err_mean": float(tip_err.mean().item()),
        "tip_err_max": float(tip_err.max().item()),
        "trans_delta": trans_delta,
        "q_delta": q_delta,
        "fit_cfg": fit_cfg,
    }
    return trans_fit, rot_fit, qpos_fit, tips_fit, stats


def parse_background_color(name: str) -> np.ndarray:
    presets = {
        "white": (0.95, 0.95, 0.95),
        "grey": (0.82, 0.82, 0.82),
        "dark": (0.05, 0.05, 0.08),
    }
    key = str(name).lower()
    if key not in presets:
        raise ValueError(f"unknown --background {name!r}, choose from {list(presets)}")
    return np.asarray(presets[key], dtype=np.float64)


def draw_geometries_interactive(
    geometries: List,
    window_name: str,
    width: int,
    height: int,
    zoom: float,
    front: List[float],
    lookat: List[float],
    up: List[float],
    point_size: float = 3.0,
    background_color: Optional[np.ndarray] = None,
) -> None:
    """Open3D 标准弹窗（draw_geometries_with_animation_callback）。"""
    bg = (
        background_color
        if background_color is not None
        else np.asarray([0.95, 0.95, 0.95], dtype=np.float64)
    )
    front_v = np.asarray(front, dtype=np.float64)
    lookat_v = np.asarray(lookat, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    def _setup_view(vis: o3d.visualization.Visualizer) -> bool:
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = bg
        ctr = vis.get_view_control()
        ctr.set_front(front_v)
        ctr.set_lookat(lookat_v)
        ctr.set_up(up_v)
        ctr.set_zoom(float(zoom))
        return False

    o3d.visualization.draw_geometries_with_animation_callback(
        geometries,
        _setup_view,
        window_name=window_name,
        width=width,
        height=height,
    )


def view_params(points: np.ndarray, mode: str):
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        lookat = [0.0, 0.0, 0.0]
    else:
        lookat = pts.mean(axis=0).tolist()

    mode = str(mode).lower()
    if mode == "front":
        return dict(front=[0.0, -1.0, 0.0], up=[0.0, 0.0, 1.0], lookat=lookat)
    if mode == "iso":
        return dict(front=[-0.55, -0.45, -0.70], up=[-0.15, -0.95, 0.25], lookat=lookat)
    return dict(front=[0.0, 0.0, -1.0], up=[0.0, -1.0, 0.0], lookat=lookat)


@torch.no_grad()
def predict_contacts(
    model,
    model_kind: str,
    feature: torch.Tensor,
    seed_points: torch.Tensor,
    object_pc: torch.Tensor,
    num_contact_samples: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if model_kind == "stability_offset":
        flat, stab_pred, _score, _best = sample_and_rank_contacts_offset(
            model, feature, seed_points, object_pc, num_samples=num_contact_samples
        )
        return flat, stab_pred
    if model_kind == "stability_absolute":
        flat, stab_pred, _score, _best = sample_and_rank_contacts_abs(
            model, feature, seed_points, object_pc, num_samples=num_contact_samples
        )
        return flat, stab_pred
    return model.sample_contacts(feature, seed_points, object_pc), None


def get_main_object_ids_largest_seg(seg_all: torch.Tensor) -> np.ndarray:
    """与 predict v16：每 view 取 seg 前景像素最多的 meta_id。"""
    view_obj_ids = []
    for v in range(seg_all.shape[0]):
        ids = torch.unique(seg_all[v])
        ids = ids[ids > 0]
        if len(ids) == 0:
            view_obj_ids.append(1)
            continue
        counts = torch.stack([(seg_all[v] == obj_id).sum() for obj_id in ids])
        view_obj_ids.append(int(ids[counts.argmax()].item()))
    return np.array(view_obj_ids, dtype=np.int64)


def build_batch(args, config, device: torch.device):
    use_success = bool(int(getattr(config, "use_success_filter", 1)))
    use_fps = bool(int(getattr(config, "use_fps_grasps_only", 1)))
    success_root = getattr(config, "success_indices_root", None)
    fps_root = getattr(config, "fps_root", None)

    dataset = GraspNetDatasetContactDiffusionV2(
        config,
        args.split,
        is_train=bool(args.augment),
        use_success_filter=use_success,
        success_indices_root=success_root,
        use_fps_grasps_only=use_fps,
        fps_root=fps_root,
    )
    if int(args.scene_id) >= 0:
        scene = f"scene_{int(args.scene_id):04d}"
        view = int(args.view_idx)
        if view < 0:
            raise ValueError("--view_idx must be >= 0 when --scene_id is set")
        dataset.views = [(scene, view)]
        dataset.cates = ["orig"]
        print(f"visualizing fixed dataset view: {scene}, view={view:04d}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config.batch_size or 1),
        drop_last=False,
        num_workers=0,
        shuffle=False,
        collate_fn=minkowski_collate_fn,
    )
    data_iter = iter(loader)
    data = None
    for _ in range(max(int(args.skip_batches), 0) + 1):
        data = next(data_iter)
    assert data is not None
    if "scene_id" in data:
        sid = data["scene_id"].reshape(-1)[0].item()
        print(f"loaded batch scene_id={int(sid):04d}")
    return {k: v.to(device) for k, v in data.items()}


@torch.no_grad()
def prepare_test_infer_sample(
    args,
    config,
    device: torch.device,
    teacher: GraspnessSampleWithFeature,
) -> dict:
    """
    测试集 scene（通常 100–189）：无 dex_grasps / contact GT cache。
    走 predict 同款 teacher.sample → top-N → contact 网络。
    """
    sid = int(args.scene_id)
    if sid < 0:
        raise ValueError("--test_infer 必须指定 --scene_id（测试集常用 100–189）")
    if sid < 100 or sid > 189:
        print(
            f"WARN: scene_{sid:04d} 不在 GraspNet test 100–189；"
            "仍尝试仅加载场景点云并推理。"
        )
    scene = f"scene_{sid:04d}"
    view = int(args.view_idx)
    if view < 0:
        raise ValueError("--view_idx must be >= 0 for --test_infer")

    split = str(args.split).split("-")[0]
    if split not in ("test", "test_seen", "test_similar", "test_novel"):
        split = "test"

    dataset = GraspNetDatasetContactDiffusionV2(
        config,
        split,
        is_train=False,
        is_eval=True,
        use_success_filter=False,
        use_fps_grasps_only=False,
    )
    dataset.views = [(scene, view)]
    dataset.cates = ["orig"]
    print(f"test_infer: loading {scene} view={view:04d} split={split} (no GT grasps)")

    sample = dataset[0]
    pc_np = np.asarray(sample["point_clouds"], dtype=np.float32)
    seg_np = np.asarray(sample["seg"], dtype=np.int64)
    pc_cpu = torch.from_numpy(pc_np).float().unsqueeze(0)
    pc = pc_cpu.to(device)
    seg = torch.from_numpy(seg_np).long().unsqueeze(0).to(device)

    main_meta_ids = get_main_object_ids_largest_seg(seg)
    meta_id = int(main_meta_ids[0])
    seg_masked = seg.clone()
    seg_masked[seg != meta_id] = 0

    grasp_num = int(getattr(args, "grasp_num", 0) or getattr(config, "grasp_num", 0) or 64)
    top_n = max(int(args.top_n), 1)
    rank = max(int(args.grasp_rank), 0)

    data_sparse = get_sparse_tensor(pc_cpu, config.data.voxel_size)
    data_sparse["seg"] = seg_masked.cpu()
    data_sparse = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_sparse.items()}
    result = teacher.sample(
        data_sparse,
        grasp_num,
        graspness_scale=float(args.graspness_scale),
        allow_fail=True,
        cate=False,
        with_score_parts=True,
        with_point=True,
        with_feature=True,
    )
    rot, trans, qpos = result[0], result[1], result[2]
    score = result[3]
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
    rank = min(rank, top_n - 1)
    feature = sel[5][0, rank].reshape(1, -1).to(device)
    seed = sel[4][0, rank].reshape(1, 3).to(device)
    init_trans = sel[1][0, rank].reshape(1, 3).to(device)
    init_rot = sel[0][0, rank].reshape(1, 3, 3).to(device)
    init_qpos = sel[2][0, rank].reshape(1, -1).to(device)
    init_score = float(sel[3][0, rank].item())

    object_pc = crop_object_pc_by_meta_id(
        pc[0], seg[0], meta_id, int(args.phy_num_points)
    )

    data = {
        "point_clouds": pc,
        "seg": seg,
        "scene_id": torch.tensor([sid], device=device),
    }
    return {
        "has_gt": False,
        "feature": feature,
        "seed": seed,
        "object_pc": object_pc,
        "init_trans": init_trans,
        "init_rot": init_rot,
        "init_qpos": init_qpos,
        "init_score": init_score,
        "scene_pc": pc_np,
        "scene_seg": seg_np,
        "target_meta": meta_id,
        "scene_loaded": sid,
        "data": data,
        "idx": 0,
        "grasp_rank": rank,
    }


def visualize(args) -> None:
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    config = load_config(args.yaml, train_v2b.arg_mapping, args)
    if not config.ckpt:
        raise ValueError("--ckpt is required for teacher feature extraction")

    data_root = train_v2b.resolve_data_root(config)
    teacher = load_teacher(config, device)
    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=config.urdf_path,
        meta_path=config.meta_path,
        hand_name=config.hand_name,
        device=str(device),
    )

    model, model_kind, version = load_contact_model(
        args.contact_ckpt, args.contact_stability_ckpt, device
    )
    model.eval()
    cd_cfg = getattr(model, "contact_cfg", getattr(model, "cfg", None))
    if cd_cfg is None:
        raise RuntimeError("Loaded contact model does not expose .cfg or .contact_cfg")

    has_gt = not bool(args.test_infer)
    if args.test_infer:
        infer = prepare_test_infer_sample(args, config, device, teacher)
        idx = int(infer["idx"])
        feature = infer["feature"]
        seed = infer["seed"]
        object_pc = infer["object_pc"]
        init_trans = infer["init_trans"]
        init_rot = infer["init_rot"]
        init_qpos = infer["init_qpos"]
        data = infer["data"]
        scene_pc = infer["scene_pc"]
        scene_seg = infer["scene_seg"]
        target_meta = infer["target_meta"]
        scene_loaded = infer["scene_loaded"]
        print(
            f"pred_main meta_id={target_meta}  "
            f"teacher_score={infer['init_score']:.4f}  "
            f"grasp_rank={infer['grasp_rank']}/{max(int(args.top_n), 1) - 1}"
        )
    else:
        gt_cache = ContactGTv2bCache(args.contact_gt_cache, robot=config.data.robot)
        gt_cache.preload_all()
        align_cache = train_v2b.AlignMatCache(
            os.path.join(data_root, "scenes"), config.data.camera
        )
        data = build_batch(args, config, device)
        batch = train_v2b.prepare_batch_v2b(
            data,
            teacher,
            hand_provider,
            gt_cache,
            align_cache,
            max_grasps=int(args.max_grasps or getattr(config, "max_grasps", 16) or 16),
            phy_num_points=int(args.phy_num_points),
            cd_cfg=cd_cfg,
            cdist_chunk=int(args.phy_cdist_chunk),
            gt_sanity_check=False,
            gt_sanity_max_dist=float(args.gt_sanity_max_dist),
        )
        if batch is None:
            raise RuntimeError("prepare_batch_v2b returned no valid contact samples")

        idx = min(max(int(args.sample_in_batch), 0), batch["feature"].shape[0] - 1)
        feature = batch["feature"][idx : idx + 1]
        seed = batch["seed_points"][idx : idx + 1]
        object_pc = batch["object_pc"][idx : idx + 1]
        gt_flat = batch["contacts_flat"][idx : idx + 1]
        mask = batch["contact_mask"][idx].detach().cpu().numpy() > 0.5
        init_trans = batch["gt_trans"][idx : idx + 1]
        init_rot = batch["gt_rot"][idx : idx + 1]
        init_qpos = batch["gt_qpos"][idx : idx + 1]
        seed_np_early = seed[0].detach().cpu().numpy()
        scene_pc, scene_seg = pick_scene_cloud_and_seg_for_seed(data, seed_np_early)
        scene_loaded = None
        if "scene_id" in data:
            scene_loaded = int(data["scene_id"].reshape(-1)[0].detach().cpu().item())

    pred_flat, stab_pred = predict_contacts(
        model,
        model_kind,
        feature,
        seed,
        object_pc,
        int(args.num_contact_samples),
    )

    num_fingers = int(cd_cfg.num_fingertips)
    pred = unflatten_contacts(pred_flat, num_fingers)[0].detach().cpu().numpy()
    seed_np = seed[0].detach().cpu().numpy()
    obj = object_pc[0].detach().cpu().numpy()
    obj = obj[np.isfinite(obj).all(axis=1)]
    obj = obj[np.linalg.norm(obj, axis=1) > 1e-8]

    if args.test_infer:
        scene_valid = np.isfinite(scene_pc).all(axis=1)
        scene_pc = scene_pc[scene_valid]
        scene_seg = scene_seg[scene_valid]
        mask = np.zeros(num_fingers, dtype=bool)
        gt = np.zeros((num_fingers, 3), dtype=np.float64)
    else:
        gt = unflatten_contacts(gt_flat, num_fingers)[0].detach().cpu().numpy()
        scene_valid = np.isfinite(scene_pc).all(axis=1)
        scene_pc = scene_pc[scene_valid]
        scene_seg = scene_seg[scene_valid]

    valid_gt = gt[mask] if has_gt else np.zeros((0, 3), dtype=np.float32)
    valid_pred_all = pred[np.isfinite(pred).all(axis=1)]
    meta_from_gt = estimate_contact_meta_id(valid_gt, scene_pc, scene_seg) if has_gt else None
    meta_from_pred = estimate_contact_meta_id(valid_pred_all, scene_pc, scene_seg)
    if has_gt:
        target_meta = meta_from_gt if meta_from_gt is not None else meta_from_pred
    else:
        target_meta = int(target_meta) if args.test_infer else meta_from_pred
    target_obj = object_points_from_seg(
        scene_pc,
        scene_seg,
        target_meta,
        max_points=int(args.max_target_points),
        seed=int(args.seed),
    )
    if target_obj.shape[0] < 8:
        target_obj = obj
        print(
            "WARN: cannot isolate target object by seg; "
            "falling back to training condition object_pc for highlight."
        )

    n_valid_gt = int(mask.sum())
    if has_gt:
        print(
            f"contact_mask={mask.astype(int).tolist()}  "
            f"valid_gt={n_valid_gt}/{num_fingers}  pred_spheres={num_fingers} (always 4)"
        )
        if n_valid_gt < num_fingers:
            print(
                f"  NOTE: 仅 {n_valid_gt} 根手指有 GT contact（IBS/cache mask=0 的指不画 GT 球）；"
                f"Pred 仍显示全部 {num_fingers} 指。"
            )
        describe_dist("GT -> object_pc", nearest_dist(valid_gt, obj))
        describe_dist("GT -> highlighted target", nearest_dist(valid_gt, target_obj))
        if mask.any():
            describe_dist(
                "Pred -> same-finger GT (masked only)",
                np.linalg.norm(pred[mask] - gt[mask], axis=1),
            )
        for f in range(num_fingers):
            if not mask[f]:
                continue
            d_obj = float(nearest_dist(pred[f : f + 1], obj)[0])
            d_gt = float(np.linalg.norm(pred[f] - gt[f]))
            print(
                f"  finger {f}: pred->obj={d_obj:.4f}m  pred->gt={d_gt:.4f}m  "
                f"gt->obj={float(nearest_dist(gt[f : f + 1], obj)[0]):.4f}m"
            )
        gt_seg_ids, gt_seg_dist = nearest_seg_ids(valid_gt, scene_pc, scene_seg)
        print(
            "GT nearest seg ids/dist: "
            + ", ".join(
                f"{int(s)}:{float(d):.4f}m" for s, d in zip(gt_seg_ids, gt_seg_dist)
            )
        )
    else:
        print(f"test_infer: no GT contacts / grasps (pred only, {num_fingers} fingers)")

    describe_dist("Pred -> object_pc (all fingers)", nearest_dist(valid_pred_all, obj))
    describe_dist(
        "Pred -> highlighted target (all fingers)",
        nearest_dist(valid_pred_all, target_obj),
    )
    if stab_pred is not None:
        print(f"stability_pred[{idx}] = {stab_pred[0].detach().cpu().numpy()}")
    if not args.test_infer and scene_loaded is None and "scene_id" in data:
        scene_loaded = int(data["scene_id"].reshape(-1)[0].detach().cpu().item())
    print(
        f"highlight target meta_id={target_meta} "
        f"(from_gt={meta_from_gt}, from_pred={meta_from_pred}, "
        f"target_points={target_obj.shape[0]}, cond_points={obj.shape[0]})"
    )
    pred_seg_ids, pred_seg_dist = nearest_seg_ids(valid_pred_all, scene_pc, scene_seg)
    print(
        "Pred nearest seg ids/dist: "
        + ", ".join(
            f"{int(s)}:{float(d):.4f}m" for s, d in zip(pred_seg_ids, pred_seg_dist)
        )
    )
    pred_proj, pred_proj_dist = nearest_points_on_cloud(pred, target_obj)

    _, tip_fn = contact_fit_backend(model_kind)
    pred_contacts_t = unflatten_contacts(pred_flat, num_fingers)

    pred_fit_stats: Optional[dict] = None
    pred_trans_fit = pred_rot_fit = pred_qpos_fit = None
    pred_tips_fit: Optional[torch.Tensor] = None
    if args.with_pred_hand:
        pred_trans_fit, pred_rot_fit, pred_qpos_fit, pred_tips_fit, pred_fit_stats = (
            fit_hand_pose_from_pred_contacts(
                hand_provider,
                init_trans,
                init_rot,
                init_qpos,
                pred_contacts_t,
                object_pc,
                cd_cfg,
                model_kind,
                fit_steps=int(args.fit_steps),
                fit_max_trans_delta=float(args.fit_max_trans_delta),
                fit_max_q_delta=float(args.fit_max_q_delta),
                fit_max_rot_delta=float(args.fit_max_rot_delta),
                fit_w_pose=float(args.fit_w_pose),
                use_model_fit_cfg=bool(args.fit_use_model_cfg),
            )
        )
        fc = pred_fit_stats["fit_cfg"]
        print(
            "Pred-hand fit (init="
            + ("GT grasp" if has_gt else "teacher coarse grasp")
            + "): "
            f"loss={pred_fit_stats['fit_loss']:.6f}  "
            f"tip_err mean={pred_fit_stats['tip_err_mean']:.4f}m  "
            f"max={pred_fit_stats['tip_err_max']:.4f}m  "
            f"trans_delta={pred_fit_stats['trans_delta']:.4f}m  "
            f"q_delta={pred_fit_stats['q_delta']:.4f}"
        )
        print(
            f"  fit_cfg: steps={fc.fit_steps}  "
            f"max_trans={fc.fit_max_trans_delta:.3f}  "
            f"max_q={fc.fit_max_q_delta:.3f}  "
            f"max_rot={fc.fit_max_rot_delta:.3f}"
        )
        if pred_fit_stats["tip_err_max"] > 0.02:
            print(
                "  WARN: 拟合后指尖仍离 Pred 接触点较远；"
                "可增大 --fit_max_trans_delta / --fit_max_q_delta 或 --fit_steps"
            )

    tips_cam: Optional[np.ndarray] = None
    grasped_meta_id: Optional[int] = None
    if args.with_scene_pc or args.with_hand or args.with_fk_tips:
        with torch.no_grad():
            tips_cam = tip_fn(
                hand_provider,
                init_trans,
                init_rot,
                init_qpos,
                object_pc=None,
                num_fingers=num_fingers,
            )[0].detach().cpu().numpy()
        grasped_meta_id = estimate_contact_meta_id(tips_cam, scene_pc, scene_seg)

    geometries: List = []
    scene_pc_for_view = scene_pc
    if args.with_scene_pc:
        scene_objects = extract_segmented_objects_cam(
            scene_pc,
            scene_seg,
            main_object_id=target_meta,
            gt_object_id=target_meta if has_gt else None,
            grasped_object_id=grasped_meta_id,
            max_points_per_obj=int(args.max_points_per_obj),
            max_points_per_bg_obj=int(args.max_points_per_bg_obj),
            seed=int(args.seed),
        )
        for obj in scene_objects:
            stride = 1
            if not (obj["is_gt"] or obj["is_main"] or obj["is_grasped"]):
                stride = max(int(args.scene_stride), 1)
            geometries.append(
                make_pcd(
                    obj["points"],
                    obj["color"],
                    stride=stride,
                )
            )
        if scene_objects:
            scene_pc_for_view = np.concatenate(
                [o["points"] for o in scene_objects], axis=0
            )
        counts = ", ".join(
            f"obj{o['obj_id']}:{o['points'].shape[0]}" for o in scene_objects
        )
        print(
            f"with_scene_pc: {len(scene_objects)} objects "
            f"(main meta={target_meta}, grasped meta={grasped_meta_id}) "
            f"points=[{counts}]"
        )
    else:
        geometries.append(
            make_pcd(
                scene_pc,
                SCENE_GREY_RGB,
                stride=max(int(args.scene_stride), 1),
            )
        )
        geometries.append(
            make_pcd(
                target_obj,
                TARGET_HIGHLIGHT_RGB,
                stride=max(int(args.object_stride), 1),
            )
        )
    if args.show_cond_object_pc:
        geometries.append(
            make_pcd(
                obj,
                (0.55, 0.15, 0.95),
                stride=max(int(args.object_stride), 1),
            )
        )
    geometries.append(make_sphere(seed_np, float(args.seed_radius), (0.0, 0.0, 0.0)))

    marker_style = str(args.contact_marker_style)
    surface_cloud = target_obj if target_obj.shape[0] >= 8 else obj
    disc_r = float(args.contact_disc_radius)
    ring_r = float(args.contact_ring_radius)
    patch_r = float(args.contact_patch_radius)
    with_patch = bool(args.contact_surface_patch)
    normal_off = float(args.contact_normal_offset)
    hand_geometries: List = []
    bg_rgb = tuple(parse_background_color(args.background).tolist())

    gt_valid_for_lines = []
    pred_valid_for_lines = []
    for f in range(num_fingers):
        color = FINGER_COLORS[f % len(FINGER_COLORS)]
        finger_color = tuple(color.tolist())
        append_contact_marker(
            geometries,
            pred[f],
            surface_cloud,
            finger_color,
            marker_style,
            "pred",
            disc_r,
            ring_r,
            patch_r,
            with_patch,
            normal_offset=normal_off,
        )
        seed_pred_line = make_lines(
            seed_np[None],
            pred[f : f + 1],
            tuple((0.55 * color + 0.45 * np.ones(3)).tolist()),
        )
        if seed_pred_line is not None:
            geometries.append(seed_pred_line)
        if args.project_pred_to_target and np.isfinite(pred_proj_dist[f]):
            append_contact_marker(
                geometries,
                pred_proj[f],
                surface_cloud,
                finger_color,
                marker_style,
                "pred_proj",
                disc_r,
                ring_r,
                patch_r,
                False,
                normal_offset=normal_off,
            )
            if float(pred_proj_dist[f]) > 0.002:
                proj_line = make_lines(
                    pred[f : f + 1],
                    pred_proj[f : f + 1],
                    finger_color,
                )
                if proj_line is not None:
                    geometries.append(proj_line)
        if mask[f]:
            append_contact_marker(
                geometries,
                gt[f],
                surface_cloud,
                finger_color,
                marker_style,
                "gt_ibs",
                float(args.gt_radius),
                ring_r,
                patch_r,
                with_patch,
                normal_offset=normal_off,
            )
            gt_valid_for_lines.append(gt[f])
            pred_valid_for_lines.append(pred[f])
            seed_gt_line = make_lines(seed_np[None], gt[f : f + 1], (0.2, 0.2, 0.2))
            if seed_gt_line is not None:
                geometries.append(seed_gt_line)

    if tips_cam is not None and args.with_fk_tips:
        for f in range(min(num_fingers, tips_cam.shape[0])):
            color = FINGER_COLORS[f % len(FINGER_COLORS)]
            finger_color = tuple(color.tolist())
            append_contact_marker(
                geometries,
                tips_cam[f],
                surface_cloud,
                finger_color,
                marker_style,
                "fk_tip",
                float(args.fk_tip_radius),
                ring_r,
                patch_r,
                False,
                normal_offset=normal_off,
            )
            fk_line = make_lines(tips_cam[f : f + 1], pred[f : f + 1], (0.1, 0.1, 0.1))
            if fk_line is not None:
                geometries.append(fk_line)

    if gt_valid_for_lines:
        gt_arr = np.asarray(gt_valid_for_lines)
        pred_arr = np.asarray(pred_valid_for_lines)
        line = make_lines(gt_arr, pred_arr, (1.0, 0.0, 1.0))
        if line is not None:
            geometries.append(line)

    if args.with_hand or args.with_pred_hand:
        hand_robot = RobotModel(config.urdf_path, config.meta_path)
    else:
        hand_robot = None

    hand_opacity = float(np.clip(args.hand_opacity, 0.05, 1.0))
    pred_hand_opacity = float(np.clip(args.pred_hand_opacity, 0.05, 1.0))
    hand_wireframe = bool(args.hand_wireframe)

    if args.with_hand:
        assert hand_robot is not None
        trans_cam = init_trans[0].detach().cpu().numpy()
        rot_cam = init_rot[0].detach().cpu().numpy()
        qpos_np = init_qpos[0].detach().cpu().numpy()
        hand_mesh = get_hand_mesh_cam(
            hand_robot,
            trans_cam,
            rot_cam,
            qpos_np,
            hand_robot.joint_names,
            HAND_MESH_RGB,
            mesh_type=str(args.hand_mesh_type),
        )
        hand_geometries.append(
            paint_mesh_ghost(hand_mesh, HAND_MESH_RGB, hand_opacity, bg_rgb)
        )
        if hand_wireframe:
            hand_geometries.append(
                mesh_wireframe_lines(hand_mesh, tuple(np.asarray(HAND_MESH_RGB) * 0.55))
            )

    if args.with_pred_hand:
        assert hand_robot is not None
        assert pred_trans_fit is not None and pred_rot_fit is not None and pred_qpos_fit is not None
        pred_hand_mesh = get_hand_mesh_cam(
            hand_robot,
            pred_trans_fit[0].detach().cpu().numpy(),
            pred_rot_fit[0].detach().cpu().numpy(),
            pred_qpos_fit[0].detach().cpu().numpy(),
            hand_robot.joint_names,
            PRED_HAND_MESH_RGB,
            mesh_type=str(args.hand_mesh_type),
        )
        hand_geometries.append(
            paint_mesh_ghost(pred_hand_mesh, PRED_HAND_MESH_RGB, pred_hand_opacity, bg_rgb)
        )
        if hand_wireframe:
            hand_geometries.append(
                mesh_wireframe_lines(
                    pred_hand_mesh, tuple(np.asarray(PRED_HAND_MESH_RGB) * 0.55)
                )
            )
        if pred_tips_fit is not None:
            tips_fit_np = pred_tips_fit[0].detach().cpu().numpy()
            for f in range(min(num_fingers, tips_fit_np.shape[0])):
                append_contact_marker(
                    geometries,
                    tips_fit_np[f],
                    surface_cloud,
                    PRED_HAND_MESH_RGB,
                    marker_style,
                    "pred",
                    float(args.pred_fk_tip_radius),
                    ring_r,
                    patch_r,
                    False,
                    normal_offset=normal_off,
                )
                tip_line = make_lines(
                    tips_fit_np[f : f + 1],
                    pred[f : f + 1],
                    PRED_HAND_MESH_RGB,
                )
                if tip_line is not None:
                    geometries.append(tip_line)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.frame_size))
    geometries.append(frame)

    print("\n--- Contact Visualization Legend ---")
    if args.with_scene_pc:
        if has_gt:
            print("Seg-colored scene: green=GT/main object, orange=main, blue=grasped, grey=others")
        else:
            print("Seg-colored scene: orange=pred_main, blue=grasped, grey=others (no GT)")
    else:
        print("Light grey points: full scene cloud")
        print("Orange points: highlighted target object by seg/contact nearest vote")
    if args.with_hand:
        if has_gt:
            print(
                f"Ghost-tinted blue-grey hand: GT grasp (opacity={hand_opacity:.2f}, "
                "legacy renderer, background unchanged)"
            )
        else:
            print(
                f"Ghost-tinted blue-grey hand: teacher coarse (opacity={hand_opacity:.2f})"
            )
    if args.with_pred_hand:
        init_label = "GT grasp" if has_gt else "teacher coarse grasp"
        print(
            f"Ghost-tinted orange hand: fitted Pred (init={init_label}, "
            f"opacity={pred_hand_opacity:.2f})"
        )
    if args.show_cond_object_pc:
        print("Purple points: contact network conditioning object_pc (pred_main crop)")
    print("Black sphere: seed point")
    if marker_style == "disc":
        print("● Filled disc on surface: Pred contact (vis_contact_stability style)")
        if with_patch:
            print("  + colored surface patch around each Pred/GT contact")
        print("○ Ring on surface: Pred projected to target object")
        if has_gt:
            print("× Magenta cross on surface: GT contact (IBS/cache)")
        print("◇ Diamond: FK fingertip")
    else:
        print("Spheres: legacy contact markers (--contact_marker_style sphere)")
    print("Light lines: seed -> each Pred")
    if has_gt:
        print("Dark lines: seed -> GT")
    if args.with_fk_tips:
        print("Grey lines: FK fingertip -> Pred")
    if args.project_pred_to_target:
        print("Lines: Pred -> surface projection ring")
    if has_gt:
        print("Magenta lines: same-finger GT -> Pred error")
    if scene_loaded is not None:
        print(f"Loaded scene: scene_{scene_loaded:04d}")
    mode_label = "test_infer" if args.test_infer else "train_gt"
    print(f"mode={mode_label}, model_kind={model_kind}, ckpt_version={version}, sample_index={idx}")
    print("------------------------------------\n")

    if args.save_ply:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ply)), exist_ok=True)
        merged = o3d.geometry.TriangleMesh()
        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                merged += geom
        o3d.io.write_triangle_mesh(args.save_ply, merged)
        print(f"Saved sphere/frame mesh to {args.save_ply}")

    if args.with_scene_pc:
        view_parts = [scene_pc_for_view]
    else:
        view_parts = [scene_pc_for_view[:: max(int(args.scene_stride), 1)]]
        view_parts.append(target_obj)
    view_parts.extend([pred, pred_proj])
    if has_gt:
        view_parts.append(gt)
    all_for_view = np.concatenate(view_parts, axis=0)
    vp = view_params(all_for_view, args.view)
    title_mode = "Test Contact Pred" if args.test_infer else "Train Contact GT vs Pred"
    draw_geometries = hand_geometries + geometries
    draw_point_size = max(float(args.point_size), float(args.contact_point_size))
    draw_geometries_interactive(
        draw_geometries,
        window_name=(
            f"{title_mode} ({model_kind})"
            + (f" scene_{scene_loaded:04d}" if scene_loaded is not None else "")
        ),
        width=1280,
        height=720,
        zoom=float(args.zoom),
        front=vp["front"],
        lookat=vp["lookat"],
        up=vp["up"],
        point_size=draw_point_size,
        background_color=parse_background_color(args.background),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize training contact GT and predicted contacts with Open3D"
    )
    train_v2b.add_argparse(parser, train_v2b.arg_mapping)
    parser.add_argument("--contact_ckpt", type=str, default="")
    parser.add_argument("--contact_stability_ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--test_infer",
        action="store_true",
        help="测试集推理模式（scene 100–189 等无 dex GT / contact cache）；"
        "teacher coarse grasp + contact 预测，仅显示 Pred",
    )
    parser.add_argument(
        "--grasp_num",
        type=int,
        default=64,
        help="test_infer：teacher 每 view 采样抓取数",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=8,
        help="test_infer：teacher score Top-N 后再选 --grasp_rank",
    )
    parser.add_argument(
        "--grasp_rank",
        type=int,
        default=0,
        help="test_infer：在 Top-N 中选第几个 coarse grasp（0=最高分）",
    )
    parser.add_argument(
        "--graspness_scale",
        type=float,
        default=5.0,
        help="test_infer：teacher graspness_scale",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=-1,
        help="fixed scene id to visualize; -1 keeps DataLoader/default sampling",
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="fixed camera view index used with --scene_id",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_batches", type=int, default=0)
    parser.add_argument("--sample_in_batch", type=int, default=0)
    parser.add_argument("--augment", type=int, default=0, help="1 matches train-time Z augmentation")
    parser.add_argument("--num_contact_samples", type=int, default=8)
    parser.add_argument("--phy_num_points", type=int, default=4096)
    parser.add_argument("--phy_cdist_chunk", type=int, default=2048)
    parser.add_argument(
        "--scene_stride",
        type=int,
        default=1,
        help="非高亮物体的点云 stride；GT/main/grasped 物体始终 stride=1",
    )
    parser.add_argument("--object_stride", type=int, default=1)
    parser.add_argument("--max_target_points", type=int, default=40000)
    parser.add_argument(
        "--with_scene_pc",
        action="store_true",
        help="按 seg 分物体着色显示场景点云（配色对齐 vis_contact_stability）",
    )
    parser.add_argument(
        "--max_points_per_obj",
        type=int,
        default=0,
        help="with_scene_pc 时 GT/main/grasped 物体最多点数；0=不截断（推荐）",
    )
    parser.add_argument(
        "--max_points_per_bg_obj",
        type=int,
        default=8000,
        help="with_scene_pc 时背景物体最多点数；0=不截断",
    )
    parser.add_argument(
        "--with_hand",
        action="store_true",
        help="显示训练样本 gt_trans/rot/qpos 对应的手 mesh（camera 系）",
    )
    parser.add_argument(
        "--with_pred_hand",
        action="store_true",
        help="对 Pred 接触点做 FK 拟合，实心橙色 mesh 显示第二只手",
    )
    parser.add_argument(
        "--fit_steps",
        type=int,
        default=32,
        help="Pred 手 pose 拟合 Adam 步数",
    )
    parser.add_argument(
        "--fit_max_trans_delta",
        type=float,
        default=0.10,
        help="拟合允许的最大平移（米）；Pred 离 GT 远时需加大",
    )
    parser.add_argument(
        "--fit_max_q_delta",
        type=float,
        default=0.30,
        help="拟合允许的最大关节增量（弧度范数）",
    )
    parser.add_argument(
        "--fit_max_rot_delta",
        type=float,
        default=0.40,
        help="拟合允许的最大旋转增量（弧度）",
    )
    parser.add_argument(
        "--fit_w_pose",
        type=float,
        default=0.01,
        help="拟合 pose 正则权重（越小越愿意偏离 GT init）",
    )
    parser.add_argument(
        "--fit_use_model_cfg",
        action="store_true",
        help="使用 ckpt 内 contact cfg 的 fit 参数（默认用上面 vis 专用更松约束）",
    )
    parser.add_argument(
        "--pred_fk_tip_radius",
        type=float,
        default=0.010,
        help="Pred 拟合手 FK 指尖球半径",
    )
    parser.add_argument(
        "--with_fk_tips",
        action="store_true",
        default=True,
        help="显示 FK 指尖球 + 指尖到 Pred 连线（默认开；与 --with_hand 独立）",
    )
    parser.add_argument(
        "--no_fk_tips",
        action="store_false",
        dest="with_fk_tips",
        help="关闭 FK 指尖标记",
    )
    parser.add_argument(
        "--hand_opacity",
        type=float,
        default=0.25,
        help="手 mesh 模拟透明度（与背景色混合；legacy 渲染，不改变窗口背景）",
    )
    parser.add_argument(
        "--pred_hand_opacity",
        type=float,
        default=0.35,
        help="Pred 拟合手 mesh 模拟透明度",
    )
    parser.add_argument(
        "--hand_wireframe",
        action="store_true",
        default=True,
        help="ghost 手叠加 wireframe 轮廓（默认开）",
    )
    parser.add_argument(
        "--no_hand_wireframe",
        action="store_false",
        dest="hand_wireframe",
        help="关闭手 mesh wireframe",
    )
    parser.add_argument(
        "--hand_mesh_type",
        type=str,
        default="collision",
        choices=["collision", "visual"],
        help="手 mesh 类型",
    )
    parser.add_argument(
        "--show_cond_object_pc",
        action="store_true",
        help="also draw the exact training condition object_pc in purple",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        choices=["white", "grey", "dark"],
        help="Open3D 背景色（默认 white；之前误设为 dark 导致黑底）",
    )
    parser.add_argument("--gt_radius", type=float, default=0.014)
    parser.add_argument("--pred_radius", type=float, default=0.012)
    parser.add_argument(
        "--contact_marker_style",
        type=str,
        default="disc",
        choices=["disc", "sphere"],
        help="接触点样式：disc=贴面圆盘+圆环（对齐 vis_contact_stability ●○）；sphere=旧小球",
    )
    parser.add_argument(
        "--contact_disc_radius",
        type=float,
        default=0.018,
        help="● Pred/GT 贴面实心圆盘半径（米）",
    )
    parser.add_argument(
        "--contact_ring_radius",
        type=float,
        default=0.014,
        help="○ Pred 投影圆环半径（米）",
    )
    parser.add_argument(
        "--contact_patch_radius",
        type=float,
        default=0.024,
        help="接触点周围物体表面高亮半径（米）",
    )
    parser.add_argument(
        "--contact_normal_offset",
        type=float,
        default=0.0008,
        help="接触 marker 沿表面法向微抬（米），减轻 z-fighting",
    )
    parser.add_argument(
        "--contact_point_size",
        type=float,
        default=6.0,
        help="接触面高亮点云最小 point_size（与 --point_size 取较大值）",
    )
    parser.add_argument(
        "--contact_surface_patch",
        action="store_true",
        default=True,
        help="在接触点处高亮物体表面点，形成接触面观感（默认开）",
    )
    parser.add_argument(
        "--no_contact_surface_patch",
        action="store_false",
        dest="contact_surface_patch",
        help="关闭接触面表面点高亮",
    )
    parser.add_argument("--fk_tip_radius", type=float, default=0.010)
    parser.add_argument("--proj_radius", type=float, default=0.008)
    parser.add_argument("--seed_radius", type=float, default=0.010)
    parser.add_argument("--frame_size", type=float, default=0.05)
    parser.add_argument(
        "--view",
        type=str,
        default="top",
        choices=["top", "front", "iso"],
        help="Open3D initial camera view; top is a top-down view in network camera coordinates",
    )
    parser.add_argument("--zoom", type=float, default=0.65)
    parser.add_argument(
        "--point_size",
        type=float,
        default=3.5,
        help="Open3D 点云渲染点大小（默认 3.5，过小会看不清物体形状）",
    )
    parser.add_argument(
        "--project_pred_to_target",
        type=int,
        default=1,
        help="1 draws nearest projection from each predicted contact to highlighted target object",
    )
    parser.add_argument("--save_ply", type=str, default="")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
