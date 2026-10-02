"""
Summarize GraspNet IsaacGym success rates by Dense / Random / Loose.

"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from region_contact.batch_scene_utils import graspnet_scene_list

DEFAULT_EXP_ROOT = "/data/Final_exp_result/graspnet"

DEFAULT_SPLIT_RELPATHS = {
    "dense": "dense/results_anchor_patch_scheme_c_dyn_v16e",
    "loose": "loose/results_anchor_patch_scheme_c_dyn_graspnet_loose_v16e",
    "random": "random/results_anchor_patch_scheme_c_dyn_graspnet_random_v16e",
}


def _load_scene_rate(result_dir: str, scene_id: str) -> Optional[float]:
    path = os.path.join(result_dir, scene_id, "sim_success.npy")
    try:
        sim_success = np.load(path)
    except OSError:
        return None
    if sim_success.size == 0:
        return None
    return float(np.asarray(sim_success, dtype=np.float64).mean())


def collect_split_rates(
    result_dir: str, scene_ids: Sequence[str]
) -> Tuple[List[str], List[float], List[str]]:
    found_ids: List[str] = []
    rates: List[float] = []
    missing: List[str] = []
    for scene_id in scene_ids:
        rate = _load_scene_rate(result_dir, scene_id)
        if rate is None:
            missing.append(scene_id)
            continue
        found_ids.append(scene_id)
        rates.append(rate)
    return found_ids, rates, missing


def resolve_split_dirs(args: argparse.Namespace) -> Dict[str, str]:
    if args.result_root:
        root = os.path.abspath(args.result_root)
        return {split: root for split in DEFAULT_SPLIT_RELPATHS}

    exp_root = os.path.abspath(args.result_exp_root)
    mapping = {
        "dense": args.dense_relpath,
        "loose": args.loose_relpath,
        "random": args.random_relpath,
    }
    return {
        split: os.path.join(exp_root, relpath)
        for split, relpath in mapping.items()
    }


def summarize(
    split_dirs: Dict[str, str],
    show_scenes: bool = False,
    show_missing: bool = True,
) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}
    all_rates: List[float] = []
    total_expected = 0

    for split in ("dense", "loose", "random"):
        scene_ids = graspnet_scene_list(split)
        total_expected += len(scene_ids)
        result_dir = split_dirs[split]
        print(f"{split:6s}  result_dir: {result_dir}")
        found_ids, rates, missing = collect_split_rates(result_dir, scene_ids)
        all_rates.extend(rates)
        mean = float(np.mean(rates)) if rates else None
        summary[split] = mean
        mean_str = f"{mean:.3f}  ({mean * 100.0:.1f}%)" if mean is not None else "n/a"
        print(
            f"{split:6s}  evaluated={len(rates):3d}/{len(rates):3d}  "
            f"coverage={len(rates):3d}/{len(scene_ids):3d}  success={mean_str}"
        )
        if show_scenes:
            for scene_id, rate in zip(found_ids, rates):
                print(f"  {scene_id}: {rate:.3f}")
        if show_missing and missing:
            print(f"{split:6s}  missing: {', '.join(missing)}")

    if all_rates:
        all_mean = float(np.mean(all_rates))
        summary["all"] = all_mean
        print(
            f"{'all':6s}  evaluated={len(all_rates):3d}/{len(all_rates):3d}  "
            f"coverage={len(all_rates):3d}/{total_expected:3d}  "
            f"success={all_mean:.3f}  ({all_mean * 100.0:.1f}%)"
        )
    else:
        summary["all"] = None
        print(
            f"{'all':6s}  evaluated=  0/  0  coverage=  0/{total_expected:3d}  "
            "success=n/a"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print GraspNet Dense/Random/Loose IsaacGym success rates"
    )
    parser.add_argument(
        "--result_exp_root",
        type=str,
        default=DEFAULT_EXP_ROOT,
        help="Parent of dense/loose/random result folders. "
        "Default: /data/Final_exp_result/graspnet",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="If all three splits live in one folder, pass that folder here. "
        "Overrides --result_exp_root and the per-split relpaths.",
    )
    parser.add_argument("--dense_relpath", type=str, default=DEFAULT_SPLIT_RELPATHS["dense"])
    parser.add_argument("--loose_relpath", type=str, default=DEFAULT_SPLIT_RELPATHS["loose"])
    parser.add_argument("--random_relpath", type=str, default=DEFAULT_SPLIT_RELPATHS["random"])
    parser.add_argument(
        "--show_scenes",
        action="store_true",
        help="Also print per-scene success rates.",
    )
    parser.add_argument(
        "--hide_missing",
        action="store_true",
        help="Do not print missing scene ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dirs = resolve_split_dirs(args)
    summarize(
        split_dirs,
        show_scenes=bool(args.show_scenes),
        show_missing=not bool(args.hide_missing),
    )


if __name__ == "__main__":
    main()
