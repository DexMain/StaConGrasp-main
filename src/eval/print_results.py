"""Summarize Isaac Gym success rates from sim_success.npy.

"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)


def _scene_rate(result_dir: str, scene_id: str) -> Optional[float]:
    path = os.path.join(result_dir, scene_id, "sim_success.npy")
    try:
        sim_success = np.load(path)
    except OSError:
        return None
    if getattr(sim_success, "size", 0) == 0:
        return None
    return float(np.asarray(sim_success, dtype=np.float64).mean())


def _graspnet_split_scenes(split: str) -> List[str]:
    if split == "dense":
        return [f"scene_{i:04d}" for i in range(100, 190)]
    if split == "loose":
        return [f"scene_{i:04d}" for i in range(200, 380)]
    if split == "random":
        return [f"scene_{i:04d}" for i in range(9000, 9900, 5)]
    raise ValueError(f"Unknown graspnet split: {split}")


def _acronym_split_scenes(result_dir: str, split: str) -> List[str]:
    if not os.path.isdir(result_dir):
        return []
    names = sorted(os.listdir(result_dir))
    if split == "dense":
        return [n for n in names if "dense" in n]
    if split == "random":
        return [n for n in names if "random" in n]
    if split == "loose":
        return [n for n in names if "loose" in n]
    raise ValueError(f"Unknown acronym split: {split}")


def _collect_rates(
    result_dir: str, scene_ids: Sequence[str]
) -> Tuple[List[str], List[float], List[str]]:
    found: List[str] = []
    rates: List[float] = []
    missing: List[str] = []
    for scene_id in scene_ids:
        rate = _scene_rate(result_dir, scene_id)
        if rate is None:
            missing.append(scene_id)
            continue
        found.append(scene_id)
        rates.append(rate)
    return found, rates, missing


def _resolve_result_dir(args: argparse.Namespace, dataset: str) -> str:
    if args.result_root:
        return os.path.abspath(args.result_root)
    ckpt_path = os.path.abspath(args.ckpt_path)
    exp_root = os.path.dirname(os.path.dirname(ckpt_path))
    if dataset == "acronym" and not args.result_subdir:
        return os.path.join(exp_root, "results_acronym")
    subdir = args.result_subdir or "results_sdf_hybrid_v16e_pointnext_grasped_object_offset"
    return os.path.join(exp_root, subdir)


def _splits_for(split: str) -> List[str]:
    if split == "all":
        return ["dense", "random", "loose"]
    return [split]


def summarize_dataset(args: argparse.Namespace, dataset: str) -> None:
    result_dir = _resolve_result_dir(args, dataset)
    if not os.path.isdir(result_dir):
        print(f"[{dataset}] missing result dir: {result_dir}, skip.")
        return

    split_names = _splits_for(args.split)
    all_found_rates: List[float] = []

    for split_name in split_names:
        if dataset == "graspnet":
            scene_ids = _graspnet_split_scenes(split_name)
        else:
            scene_ids = _acronym_split_scenes(result_dir, split_name)

        found, rates, missing = _collect_rates(result_dir, scene_ids)
        if args.per_scene:
            for scene_id, rate in zip(found, rates):
                print(f"{scene_id} success rate: {rate:.3f}")
            for scene_id in missing:
                print(f"{scene_id}: missing sim_success.npy, skip.")

        if not rates:
            print(
                f"[{dataset} {split_name}] no sim_success.npy under {result_dir}, skip."
            )
            continue

        mean_rate = float(np.mean(rates))
        print(
            f"{dataset} {split_name} (n={len(rates)}/{len(scene_ids)}) "
            f"mean success rate: {mean_rate:.3f}"
        )
        all_found_rates.extend(rates)

    if args.split == "all" and all_found_rates:
        print(
            f"Average {dataset} success rate "
            f"(n={len(all_found_rates)}): {float(np.mean(all_found_rates)):.3f}"
        )


def main() -> None:
    os.chdir(_REPO_ROOT)
    parser = argparse.ArgumentParser(
        description="Print mean Isaac Gym success rates from sim_success.npy"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="experiments/dex_ours/ckpt/ckpt_50000.pth",
        help="Used to locate exp_root (parent of ckpt/)",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="results_sdf_hybrid_v16e_pointnext_grasped_object_offset",
        help="Result folder under exp_root (or under --result_root if set)",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="Optional absolute result root; overrides ckpt_path/result_subdir layout",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="graspnet",
        choices=["graspnet", "acronym", "all"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dense",
        choices=["dense", "random", "loose", "all"],
        help="Scene split to summarize (all = dense+random+loose)",
    )
    parser.add_argument(
        "--per_scene",
        type=int,
        default=0,
        help="If 1, print per-scene success rates",
    )
    args = parser.parse_args()

    datasets = ["graspnet", "acronym"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        summarize_dataset(args, dataset)


if __name__ == "__main__":
    main()
