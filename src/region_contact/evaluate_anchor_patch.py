from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anchor-patch prediction with IsaacGym")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--result_subdir", default="results_anchor_patch_v16e")
    parser.add_argument(
        "--result_exp_root",
        default="",
        help="Optional root for grasps.npz / sim_success.npy; forwarded to evaluate_dexterous_sdf.",
    )
    parser.add_argument(
        "--grasp_result_subdir",
        default="",
        help="Input grasp directory; defaults to result_subdir.",
    )
    parser.add_argument("--dataset", default="graspnet", choices=["graspnet", "acronym"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mesh_root", default="/data/meshdata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument(
        "--allegro_waypoint_steps",
        type=int,
        nargs=4,
        default=None,
        metavar=("PRE", "COVER", "GRASP", "LIFT"),
    )
    parser.add_argument("--allegro_pregrasp_offset", type=float, default=None)
    parser.add_argument(
        "--allegro_pregrasp_qpos_mode",
        choices=["canonical", "relative"],
        default=None,
    )
    parser.add_argument("--allegro_debug_state", type=int, default=0)
    parser.add_argument("--allegro_squeeze_delta", type=float, default=None)
    parser.add_argument("--allegro_approach_lift", type=float, default=None)
    parser.add_argument("--allegro_approach_clearance", type=float, default=None)
    parser.add_argument("--allegro_pregrasp_z_offset", type=float, default=None)
    parser.add_argument("--allegro_allow_model_mismatch", type=int, default=0)
    parser.add_argument("--allegro_drive_stiffness", type=float, default=None)
    parser.add_argument("--allegro_drive_damping", type=float, default=None)
    parser.add_argument("--allegro_drive_effort", type=float, default=None)
    parser.add_argument("--allegro_hand_friction", type=float, default=None)
    args, passthrough = parser.parse_known_args()
    cmd = [
        sys.executable,
        "-m",
        "eval.evaluate_dexterous_sdf",
        "--ckpt_path_list",
        args.ckpt_path,
        "--scene_id",
        args.scene_id,
        "--result_subdir",
        args.result_subdir,
        "--dataset",
        args.dataset,
        "--device",
        args.device,
        "--mesh_root",
        args.mesh_root,
        "--batch_size",
        str(args.batch_size),
        "--strategy",
        "ours",
        "--headless",
        str(args.headless),
        "--overwrite",
        str(args.overwrite),
        "--seed",
        str(args.seed),
    ]
    if args.result_exp_root:
        cmd += ["--result_exp_root", args.result_exp_root]
    if args.grasp_result_subdir:
        cmd += ["--grasp_result_subdir", args.grasp_result_subdir]
    if args.allegro_waypoint_steps is not None:
        cmd += ["--allegro_waypoint_steps"] + [
            str(value) for value in args.allegro_waypoint_steps
        ]
    if args.allegro_pregrasp_offset is not None:
        cmd += ["--allegro_pregrasp_offset", str(args.allegro_pregrasp_offset)]
    if args.allegro_pregrasp_qpos_mode is not None:
        cmd += ["--allegro_pregrasp_qpos_mode", args.allegro_pregrasp_qpos_mode]
    if args.allegro_debug_state:
        cmd += ["--allegro_debug_state", str(args.allegro_debug_state)]
    if args.allegro_squeeze_delta is not None:
        cmd += ["--allegro_squeeze_delta", str(args.allegro_squeeze_delta)]
    if args.allegro_approach_lift is not None:
        cmd += ["--allegro_approach_lift", str(args.allegro_approach_lift)]
    if args.allegro_approach_clearance is not None:
        cmd += ["--allegro_approach_clearance", str(args.allegro_approach_clearance)]
    if args.allegro_pregrasp_z_offset is not None:
        cmd += ["--allegro_pregrasp_z_offset", str(args.allegro_pregrasp_z_offset)]
    if args.allegro_allow_model_mismatch:
        cmd += ["--allegro_allow_model_mismatch", str(args.allegro_allow_model_mismatch)]
    for flag, value in (
        ("--allegro_drive_stiffness", args.allegro_drive_stiffness),
        ("--allegro_drive_damping", args.allegro_drive_damping),
        ("--allegro_drive_effort", args.allegro_drive_effort),
        ("--allegro_hand_friction", args.allegro_hand_friction),
    ):
        if value is not None:
            cmd += [flag, str(value)]
    cmd += passthrough
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
