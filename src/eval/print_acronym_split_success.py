"""
Summarize ACRONYM IsaacGym success rates by Dense / Random / Loose.


"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

EXPECTED_SCENES = {
    "dense": 100,
    "random": 90,
    "loose": 30,
}

EXPECTED_SCENE_IDS = {
    "dense": [f"scene_dense_{i}" for i in range(100)],
    "random": [f"scene_random_{i}" for i in range(90)],
    "loose": [f"scene_loose_{i}" for i in range(30)],
}


def resolve_result_root(args: argparse.Namespace) -> str:
    if args.result_root:
        return os.path.abspath(args.result_root)
    if not args.ckpt_path:
        raise SystemExit("Provide --result_root, or both --ckpt_path and --result_subdir.")
    exp_root = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt_path)))
    return os.path.join(exp_root, args.result_subdir)


def collect_split_rates(result_root: str, split: str) -> Tuple[List[str], List[float]]:
    pattern = os.path.join(result_root, f"scene_{split}_*", "sim_success.npy")
    paths = sorted(glob.glob(pattern))
    scene_ids: List[str] = []
    rates: List[float] = []
    for path in paths:
        scene_id = os.path.basename(os.path.dirname(path))
        try:
            sim_success = np.load(path)
        except OSError:
            continue
        if sim_success.size == 0:
            continue
        scene_ids.append(scene_id)
        rates.append(float(np.asarray(sim_success, dtype=np.float64).mean()))
    return scene_ids, rates


def summarize(
    result_root: str,
    show_scenes: bool = False,
    show_missing: bool = True,
) -> Dict[str, Optional[float]]:
    print(f"result_root: {result_root}")
    if not os.path.isdir(result_root):
        raise SystemExit(f"result root does not exist: {result_root}")

    summary: Dict[str, Optional[float]] = {}
    all_rates: List[float] = []
    total_expected = sum(EXPECTED_SCENES.values())

    for split, n_expected in EXPECTED_SCENES.items():
        scene_ids, rates = collect_split_rates(result_root, split)
        all_rates.extend(rates)
        mean = float(np.mean(rates)) if rates else None
        summary[split] = mean
        mean_str = f"{mean:.3f}  ({mean * 100.0:.1f}%)" if mean is not None else "n/a"
        print(
            f"{split:6s}  evaluated={len(rates):3d}/{len(rates):3d}  "
            f"coverage={len(rates):3d}/{n_expected:3d}  success={mean_str}"
        )
        if show_scenes:
            for scene_id, rate in zip(scene_ids, rates):
                print(f"  {scene_id}: {rate:.3f}")
        if show_missing:
            found = set(scene_ids)
            missing = [sid for sid in EXPECTED_SCENE_IDS[split] if sid not in found]
            if missing:
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
        description="Print ACRONYM Dense/Random/Loose IsaacGym success rates"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Used with --result_subdir to resolve "
        "<ckpt_parent_parent>/<result_subdir>.",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="results_anchor_patch_scheme_c_dyn_acronym_v16e",
        help="Result subdirectory under the experiment root of --ckpt_path.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="Absolute/relative path to the result folder that contains "
        "scene_dense_*/sim_success.npy. Overrides --ckpt_path/--result_subdir.",
    )
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
    result_root = resolve_result_root(args)
    summarize(
        result_root,
        show_scenes=bool(args.show_scenes),
        show_missing=not bool(args.hide_missing),
    )


if __name__ == "__main__":
    main()
