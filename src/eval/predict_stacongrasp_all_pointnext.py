import os
import argparse
import multiprocessing
from rich.progress import track


def predict_grasps(scene_id):
    worker = multiprocessing.current_process()._identity[0]
    gpu = args.gpu_list[worker - 1]

    command = " ".join([
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={args.pythonpath}",
        "python -m eval.predict_stacongrasp_pointnext",
        f"--ckpt_path {args.ckpt_path}",
        "--device cuda:0",
        f"--urdf_path {args.urdf_path}",
        f"--meta_path {args.meta_path}",
        f"--hand_name {args.hand_name}",
        f"--camera {args.camera}",
        f"--scene_id {scene_id}",
        f"--grasp_num {args.grasp_num}",
        f"--top_n {args.top_n}",
        f"--stride {args.stride}",
        f"--max_iters {args.max_iters}",
        f"--lr {args.lr}",
        f"--pen_threshold {args.pen_threshold}",
        f"--max_trans_delta {args.max_trans_delta}",
        f"--max_q_delta {args.max_q_delta}",
        f"--physics_min_energy_improve {args.physics_min_energy_improve}",
        f"--physics_w_tip {args.physics_w_tip}",
        f"--physics_w_mid {args.physics_w_mid}",
        f"--physics_w_moment {args.physics_w_moment}",
        f"--physics_w_bilateral {args.physics_w_bilateral}",
        f"--physics_w_pose {args.physics_w_pose}",
        f"--physics_pen_on_tips_only {args.physics_pen_on_tips_only}",
        f"--physics_pen_threshold {args.physics_pen_threshold}",
        f"--physics_min_pen_improve {args.physics_min_pen_improve}",
        f"--pointnext_strict_physics_gate {args.pointnext_strict_physics_gate}",
        f"--pointnext_strict_require_base_gate {args.pointnext_strict_require_base_gate}",
        f"--pointnext_strict_contact_min_improve {args.pointnext_strict_contact_min_improve}",
        f"--pointnext_strict_pen_eps {args.pointnext_strict_pen_eps}",
        f"--pointnext_strict_stability_eps {args.pointnext_strict_stability_eps}",
        f"--pointnext_strict_max_trans_delta {args.pointnext_strict_max_trans_delta}",
        f"--pointnext_strict_max_q_delta {args.pointnext_strict_max_q_delta}",
        f"--seed {args.seed}",
        f"--overwrite {args.overwrite}",
        f"--scene_num {args.scene_num}",
        f"--dataset {args.dataset}",
        f"--mesh_root {args.mesh_root}",
        f"--sdf_grid_size {args.sdf_grid_size}",
        f"--data_root {args.data_root}",
        f"--logs_path {args.logs_path}",
        f"--main_object_mode {args.main_object_mode}",
        f"--ann_meta_fallback {args.ann_meta_fallback}",
        f"--min_main_object_points {args.min_main_object_points}",
        f"--mask_diffusion_to_main {args.mask_diffusion_to_main}",
        f"--crop_object_pc_to_main {args.crop_object_pc_to_main}",
    ])

    if args.result_subdir:
        command += f" --result_subdir {args.result_subdir}"
    if args.result_exp_root:
        command += f" --result_exp_root {args.result_exp_root}"
    if args.config_yaml:
        command += f" --config_yaml {args.config_yaml}"

    if args.contact_stability_ckpt:
        command += f" --contact_stability_ckpt {args.contact_stability_ckpt}"
    if args.contact_v2_ckpt:
        command += f" --contact_v2_ckpt {args.contact_v2_ckpt}"
    if args.contact_fit_steps > 0:
        command += f" --contact_fit_steps {args.contact_fit_steps}"
    if args.contact_num_samples > 0:
        command += f" --contact_num_samples {args.contact_num_samples}"
    command += f" --contact_obj_dist_weight {args.contact_obj_dist_weight}"
    command += f" --contact_obj_dist_clip {args.contact_obj_dist_clip}"
    command += f" --contact_project_to_object {args.contact_project_to_object}"
    command += f" --contact_object_mode {args.contact_object_mode}"
    command += f" --contact_grasped_max_tip_dist {args.contact_grasped_max_tip_dist}"
    command += f" --contact_grasped_margin {args.contact_grasped_margin}"
    command += f" --contact_grasped_min_points {args.contact_grasped_min_points}"
    command += f" --contact_fit_max_trans_delta {args.contact_fit_max_trans_delta}"
    command += f" --contact_fit_max_q_delta {args.contact_fit_max_q_delta}"
    command += f" --max_obj_points {args.max_obj_points}"

    if args.verbose:
        command += " --verbose"

    if args.dataset == "acronym" and args.all_scene_ids_acronym is not None:
        command += " --all_scene_ids_acronym " + " ".join(args.all_scene_ids_acronym)

    print(f"[Worker {worker} | GPU {gpu}] {command}")
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Batch launcher for "
            "predict_stacongrasp_pointnext"
        )
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth",
    )
    parser.add_argument(
        "--pythonpath",
        type=str,
        default="/home/lxy/CADGrasp-main_513/src",
    )

    parser.add_argument("--gpu_list", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="robot_models/urdf/leap_hand_simplified.urdf",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="robot_models/meta/leap_hand/meta.yaml",
    )
    parser.add_argument("--hand_name", type=str, default="leap_hand")

    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--scene_id_start", type=int, default=100)
    parser.add_argument("--scene_id_end", type=int, default=190)

    parser.add_argument("--grasp_num", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pen_threshold", type=float, default=0.0)
    parser.add_argument("--max_trans_delta", type=float, default=0.003)
    parser.add_argument("--max_q_delta", type=float, default=0.08)
    parser.add_argument("--physics_min_energy_improve", type=float, default=0.0001)
    parser.add_argument("--physics_w_tip", type=float, default=0.5)
    parser.add_argument("--physics_w_mid", type=float, default=1.0)
    parser.add_argument("--physics_w_moment", type=float, default=1.0)
    parser.add_argument("--physics_w_bilateral", type=float, default=0.5)
    parser.add_argument("--physics_w_pose", type=float, default=10.0)
    parser.add_argument("--physics_pen_on_tips_only", type=int, default=1)
    parser.add_argument("--physics_pen_threshold", type=float, default=0.002)
    parser.add_argument("--physics_min_pen_improve", type=float, default=1e-6)
    parser.add_argument("--pointnext_strict_physics_gate", type=int, default=1)
    parser.add_argument("--pointnext_strict_require_base_gate", type=int, default=1)
    parser.add_argument("--pointnext_strict_contact_min_improve", type=float, default=1e-7)
    parser.add_argument("--pointnext_strict_pen_eps", type=float, default=0.0)
    parser.add_argument("--pointnext_strict_stability_eps", type=float, default=0.0)
    parser.add_argument("--pointnext_strict_max_trans_delta", type=float, default=-1.0)
    parser.add_argument("--pointnext_strict_max_q_delta", type=float, default=-1.0)

    parser.add_argument(
        "--main_object_mode",
        type=str,
        default="ann_meta_aligned",
        choices=["largest_seg", "ann_meta_intersection", "ann_meta_aligned"],
    )
    parser.add_argument(
        "--ann_meta_fallback",
        type=str,
        default="ann_only",
        choices=["largest_seg", "ann_only", "meta_only"],
    )
    parser.add_argument("--min_main_object_points", type=int, default=32)
    parser.add_argument("--mask_diffusion_to_main", type=int, default=1)
    parser.add_argument("--crop_object_pc_to_main", type=int, default=1)
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="results_sdf_hybrid_v16e_pointnext_grasped_object_offset",
    )
    parser.add_argument("--contact_stability_ckpt", type=str, default="")
    parser.add_argument("--contact_v2_ckpt", type=str, default="")
    parser.add_argument("--contact_fit_steps", type=int, default=0)
    parser.add_argument("--contact_num_samples", type=int, default=8)
    parser.add_argument("--contact_obj_dist_weight", type=float, default=20.0)
    parser.add_argument("--contact_obj_dist_clip", type=float, default=0.20)
    parser.add_argument("--contact_project_to_object", type=int, default=1)
    parser.add_argument(
        "--contact_object_mode",
        type=str,
        default="grasped_gate",
        choices=["pred_main", "grasped_gate"],
    )
    parser.add_argument("--contact_grasped_max_tip_dist", type=float, default=0.06)
    parser.add_argument("--contact_grasped_margin", type=float, default=0.05)
    parser.add_argument("--contact_grasped_min_points", type=int, default=32)
    parser.add_argument("--contact_fit_max_trans_delta", type=float, default=0.003)
    parser.add_argument("--contact_fit_max_q_delta", type=float, default=0.08)
    parser.add_argument("--max_obj_points", type=int, default=2048)
    parser.add_argument("--result_exp_root", type=str, default="")
    parser.add_argument("--config_yaml", type=str, default="")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--scene_num", type=int, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        default="graspnet",
        choices=["graspnet", "acronym"],
    )
    parser.add_argument("--all_scene_ids_acronym", type=str, nargs="*", default=None)

    parser.add_argument("--mesh_root", type=str, default="/data/meshdata")
    parser.add_argument("--sdf_grid_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--logs_path", type=str, default="logs/sdf_hybrid_predict")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.dataset == "graspnet":
        if args.scene_id_start < 8500:
            if args.scene_num is None:
                args.scene_num = int(
                    (args.scene_id_end - args.scene_id_start) / len(args.gpu_list)
                ) + 1

            scene_id_list = [
                f"scene_{str(i).zfill(4)}"
                for i in range(args.scene_id_start, args.scene_id_end, args.scene_num)
            ]
        else:
            if args.scene_num is None:
                args.scene_num = int(
                    (args.scene_id_end - args.scene_id_start) / 5 / len(args.gpu_list)
                ) + 1

            scene_id_list = [
                f"scene_{str(i).zfill(4)}"
                for i in range(args.scene_id_start, args.scene_id_end, args.scene_num * 5)
            ]

    elif args.dataset == "acronym":
        scene_name_list = []
        scene_name_list += [f"scene_dense_{i}" for i in range(100)]
        scene_name_list += [f"scene_random_{i}" for i in range(90)]
        scene_name_list += [f"scene_loose_{i}" for i in range(30)]

        args.all_scene_ids_acronym = scene_name_list
        args.scene_num = int(len(args.all_scene_ids_acronym) / len(args.gpu_list)) + 1
        scene_id_list = range(0, len(scene_name_list), args.scene_num)

    with multiprocessing.Pool(len(args.gpu_list)) as pool:
        it = track(
            pool.imap_unordered(predict_grasps, scene_id_list),
            total=len(scene_id_list),
            description="predicting sdf-hybrid-v16e-grasped-object-offset",
        )
        list(it)
