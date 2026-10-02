from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tests.vis_contact_stability import visualize
from paths import project_path
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize predicted contacts from v16c grasps.npz"
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument(
        "--contact_stability_ckpt",
        type=str,
        default="experiments/contact_stability_v4b/ckpt/ckpt_600.pth",
        help="用于定位 experiments/<exp>/results_... 目录",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="results_sdf_hybrid_v16c_contact_stability_pen_gated",
    )
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--grasp_indices", type=int, nargs="+", default=None)
    parser.add_argument("--max_grasps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--frame",
        type=str,
        default="camera",
        choices=["world", "camera"],
        help="默认 camera（与 target_contacts_cam 同系）；world 与 sim/GIF 同系",
    )
    parser.add_argument("--with_object_pc", action="store_true")
    parser.add_argument(
        "--no_cond_object_pc",
        action="store_true",
        help="不绘制 contact 网络 conditioning object_pc",
    )
    parser.add_argument(
        "--no_project_pred",
        action="store_true",
        help="不绘制 Pred 在物体点云上的最近点投影",
    )
    parser.add_argument(
        "--fk_tip_snap",
        action="store_true",
        help="FK 指尖吸附到 pred_main 点云（旧行为，易显得 Pred 很远）",
    )
    parser.add_argument("--with_hand", action="store_true")
    parser.add_argument("--with_scene_pc", action="store_true")
    parser.add_argument("--max_points_per_obj", type=int, default=1024)
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--save_per_grasp", action="store_true")
    parser.add_argument(
        "--show_sim_status",
        action="store_true",
        help="标注 sim_success.npy 中的 OK/FAIL",
    )
    parser.add_argument(
        "--show_all_k",
        action="store_true",
        help="叠加 K 路 contact 候选（需加载 teacher+contact，较慢）",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
        help="Teacher ckpt（仅 --show_all_k 时需要）",
    )
    parser.add_argument("--contact_num_samples", type=int, default=8)
    parser.add_argument("--grasp_num", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=str(project_path("robot_models/urdf/leap_hand_simplified.urdf")),
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=str(project_path("robot_models/meta/leap_hand/meta.yaml")),
    )
    parser.add_argument("--hand_name", type=str, default="leap_hand")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.pred_only = True
    args.gt_only = False
    args.with_contact_gt = False
    args.from_dataset = False
    args.list_dataset_samples = False
    args.allow_reinfer = bool(args.show_all_k)
    args.with_cond_object_pc = not args.no_cond_object_pc
    args.project_pred_to_object = not args.no_project_pred
    visualize(args)
