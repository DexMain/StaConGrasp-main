import os
import argparse
import numpy as np
import torch
import trimesh
from tqdm import tqdm


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    return mesh


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


def compute_sdf_grid(
    mesh,
    grid_size=64,
    padding=0.04,
    truncation=0.04,
    chunk_size=200000,
    device="cpu",
):
    bounds = mesh.bounds.astype(np.float32)
    bounds_min = bounds[0] - padding
    bounds_max = bounds[1] + padding

    pts, origin, voxel_size = make_grid(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        grid_size=grid_size,
        device=torch.device(device),
    )

    pts_np = pts.cpu().numpy()

    sdf_chunks = []
    for s in range(0, len(pts_np), chunk_size):
        q = pts_np[s : s + chunk_size]

        sd = trimesh.proximity.signed_distance(mesh, q)

        # trimesh: inside 通常为正，outside 通常为负
        # 我们转成机器人常用约定：
        # outside > 0, inside < 0
        sd = -sd

        sd = np.clip(sd, -truncation, truncation)
        sdf_chunks.append(sd.astype(np.float32))

    sdf = np.concatenate(sdf_chunks, axis=0).reshape(
        grid_size,
        grid_size,
        grid_size,
    )

    return sdf, origin.cpu().numpy().astype(np.float32), np.float32(voxel_size.cpu().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_root", type=str, default="/data/meshdata")
    parser.add_argument("--mesh_filename", type=str, default="simplified.obj")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--padding", type=float, default=0.04)
    parser.add_argument("--truncation", type=float, default=0.04)
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    obj_dirs = sorted([
        d for d in os.listdir(args.mesh_root)
        if os.path.isdir(os.path.join(args.mesh_root, d)) and d.isdigit()
    ])

    print(f"Found {len(obj_dirs)} object folders in {args.mesh_root}")

    for obj_id in tqdm(obj_dirs):
        obj_dir = os.path.join(args.mesh_root, obj_id)

        mesh_path = os.path.join(obj_dir, args.mesh_filename)
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(obj_dir, "textured.obj")

        if not os.path.exists(mesh_path):
            print(f"[Skip] no mesh found for {obj_id}")
            continue

        save_path = os.path.join(obj_dir, f"sdf_{args.grid_size}.npz")

        if os.path.exists(save_path) and not args.overwrite:
            continue

        try:
            mesh = load_mesh(mesh_path)

            sdf_grid, origin, voxel_size = compute_sdf_grid(
                mesh=mesh,
                grid_size=args.grid_size,
                padding=args.padding,
                truncation=args.truncation,
                chunk_size=args.chunk_size,
                device=args.device,
            )

            np.savez_compressed(
                save_path,
                sdf_grid=sdf_grid.astype(np.float32),
                origin=origin.astype(np.float32),
                voxel_size=np.float32(voxel_size),
                grid_size=np.int32(args.grid_size),
                padding=np.float32(args.padding),
                truncation=np.float32(args.truncation),
                mesh_filename=os.path.basename(mesh_path),
            )

        except Exception as e:
            print(f"[Error] obj {obj_id}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()