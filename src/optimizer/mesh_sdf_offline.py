import os
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def load_mesh(mesh_root, obj_id, filename="simplified.obj"):
    obj_dir = os.path.join(mesh_root, f"{int(obj_id):03d}")
    mesh_path = os.path.join(obj_dir, filename)

    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(obj_dir, "textured.obj")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"No mesh found for object {obj_id}: {mesh_path}")

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    return mesh


def load_mesh_path(mesh_path):
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    try:
        mesh.process(validate=True)
    except Exception:
        pass
    return mesh

# load cached sdf grid
def load_cached_sdf(mesh_root, obj_id, grid_size=64, device="cuda"):
    path = os.path.join(mesh_root, f"{int(obj_id):03d}", f"sdf_{grid_size}.npz")
    return load_cached_sdf_path(path, device=device)


def load_cached_sdf_path(path, device="cuda"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached SDF not found: {path}")

    data = np.load(path)

    sdf_grid = torch.from_numpy(data["sdf_grid"]).float().to(device)
    origin = torch.from_numpy(data["origin"]).float().to(device)
    voxel_size = torch.tensor(float(data["voxel_size"]), dtype=torch.float32, device=device)

    return sdf_grid, origin, voxel_size


def build_cached_sdf(mesh_path, output_path, grid_size=64, padding=0.01):
    """Build a compact SDF cache for a mesh in its canonical object frame."""
    mesh = load_mesh_path(mesh_path)
    bounds = mesh.bounds.astype(np.float32)
    extent = bounds[1] - bounds[0]
    pad = max(float(padding), float(extent.max()) * 0.05)
    center = 0.5 * (bounds[0] + bounds[1])
    half_extent = 0.5 * float(extent.max()) + pad
    bounds_min = center - half_extent
    bounds_max = center + half_extent

    lin = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float32)
    ix, iy, iz = np.meshgrid(lin, lin, lin, indexing="ij")
    points = bounds_min[None, None, None, :] + np.stack(
        [ix, iy, iz], axis=-1
    ) * (bounds_max - bounds_min)[None, None, None, :]
    points = points.reshape(-1, 3)
    sdf = trimesh.proximity.signed_distance(mesh, points).astype(np.float32)
    voxel_size = float(np.max(bounds_max - bounds_min) / max(int(grid_size) - 1, 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    np.savez_compressed(
        tmp_path,
        sdf_grid=sdf.reshape(int(grid_size), int(grid_size), int(grid_size)),
        origin=bounds_min.astype(np.float32),
        voxel_size=np.float32(voxel_size),
    )
    generated_path = f"{tmp_path}.npz"
    os.replace(generated_path, output_path)
    return output_path


def load_or_build_cached_sdf(
    mesh_path,
    sdf_path,
    grid_size=64,
    device="cuda",
):
    if not os.path.isfile(sdf_path):
        build_cached_sdf(mesh_path, sdf_path, grid_size=grid_size)
    return load_cached_sdf_path(sdf_path, device=device)

# 当有离线缓存的 SDF grid 时，可以不用重新计算，直接加载
# def load_mesh_in_camera_frame(mesh_root, obj_id, T_obj_to_cam, filename="simplified.obj"):
#     mesh = load_mesh(mesh_root, obj_id, filename)
#     mesh.apply_transform(T_obj_to_cam)
#     return mesh


def make_grid(bounds_min, bounds_max, grid_size, device):
    lin = torch.linspace(0, grid_size - 1, grid_size, device=device)
    ix, iy, iz = torch.meshgrid(lin, lin, lin, indexing="ij")
    idx = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3)

    bounds_min = torch.as_tensor(bounds_min, dtype=torch.float32, device=device)
    bounds_max = torch.as_tensor(bounds_max, dtype=torch.float32, device=device)

    extent = bounds_max - bounds_min
    max_extent = extent.max()
    center = 0.5 * (bounds_min + bounds_max)
    origin = center - 0.5 * max_extent
    voxel_size = max_extent / float(grid_size - 1)

    pts = origin[None, :] + idx * voxel_size
    return pts, origin, voxel_size

# 当有离线缓存的 SDF grid 时，可以不用重新计算，直接加载
# class MeshSDFGridBuilder:
#     def __init__(
#         self,
#         mesh_root="/data/meshdata",
#         mesh_filename="simplified.obj",
#         grid_size=64,
#         padding=0.04,
#         truncation=0.04,
#         chunk_size=200000,
#     ):
#         self.mesh_root = mesh_root
#         self.mesh_filename = mesh_filename
#         self.grid_size = grid_size
#         self.padding = padding
#         self.truncation = truncation
#         self.chunk_size = chunk_size

#     def build_from_mesh(self, mesh, device="cuda"):
#         device = torch.device(device)

#         bounds = mesh.bounds.astype(np.float32)
#         bounds_min = bounds[0] - self.padding
#         bounds_max = bounds[1] + self.padding

#         pts, origin, voxel_size = make_grid(
#             bounds_min=bounds_min,
#             bounds_max=bounds_max,
#             grid_size=self.grid_size,
#             device=device,
#         )

#         pts_np = pts.detach().cpu().numpy()
#         sdf_chunks = []

#         for s in range(0, len(pts_np), self.chunk_size):
#             q = pts_np[s : s + self.chunk_size]

#             sd = trimesh.proximity.signed_distance(mesh, q)

#             # trimesh 通常 inside 为正，outside 为负；
#             # 这里转为机器人常用约定：outside > 0, inside < 0
#             sd = -sd

#             sd = np.clip(sd, -self.truncation, self.truncation)
#             sdf_chunks.append(sd.astype(np.float32))

#         sdf = np.concatenate(sdf_chunks, axis=0)

#         sdf_grid = torch.from_numpy(sdf).to(device).reshape(
#             self.grid_size,
#             self.grid_size,
#             self.grid_size,
#         )

#         return sdf_grid, origin, torch.as_tensor(voxel_size, device=device)


def sample_sdf_grid(sdf_grid, origin, voxel_size, query_pts):
    """
    sdf_grid: (B, G, G, G)
    origin: (B, 3)
    voxel_size: (B,)
    query_pts: (B, M, 3)

    return:
        sdf: (B, M)

    Convention:
        sdf > 0: outside
        sdf = 0: surface
        sdf < 0: inside / penetration
    """
    B, G, _, _ = sdf_grid.shape
    M = query_pts.shape[1]

    idx = (query_pts - origin[:, None, :]) / voxel_size[:, None, None]

    x = idx[..., 0] / float(G - 1) * 2.0 - 1.0
    y = idx[..., 1] / float(G - 1) * 2.0 - 1.0
    z = idx[..., 2] / float(G - 1) * 2.0 - 1.0

    grid = torch.stack([x, y, z], dim=-1).reshape(B, M, 1, 1, 3)

    # grid_sample input: (B, C, D, H, W)
    volume = sdf_grid.permute(0, 3, 2, 1).unsqueeze(1)

    sdf = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return sdf[:, 0, :, 0, 0]
