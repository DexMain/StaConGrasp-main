from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_contact_diffusion_base as train_v2b
import vis_contact_stability as vcs
import vis_train_contact_gt_pred_o3d as vto
from utils.config import load_config
from utils.contact_gt_surface import unflatten_contacts
from utils.contact_gt_ibs_thumb_split import ContactGTv2bCache
from utils.util import set_seed
from utils.vis_plotly import Vis
from optimizer.physics_guided_diffusion_patch import SDFAdamHandPointsProvider

HAND_MESH_COLOR = vcs.HAND_MESH_COLOR
PRED_HAND_MESH_COLOR = "#FF7043"
FINGER_COLORS = vcs.FINGER_COLORS
FINGER_LABELS = vcs.FINGER_LABELS
GT_V2_IBS_COLOR = vcs.GT_V2_IBS_COLOR


def _scene_tag(scene_id: int) -> str:
    return f"scene_{int(scene_id):04d}"


def _resolve_output_dir(args, model_kind: str, version: str) -> str:
    if args.output_dir:
        return args.output_dir
    ckpt = args.contact_stability_ckpt or args.contact_ckpt
    exp_root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    mode = "test_infer" if args.test_infer else "train_gt"
    sub = os.path.join(exp_root, "plotly_vis", f"{mode}_{model_kind}_{version}")
    return sub


@torch.no_grad()
def run_inference(args, config, device):
    """加载 teacher + contact 模型，跑 train_gt 或 test_infer 推理。"""
    teacher = vto.load_teacher(config, device)
    model, model_kind, version = vto.load_contact_model(
        args.contact_ckpt, args.contact_stability_ckpt, device
    )
    model.eval()
    cd_cfg = getattr(model, "contact_cfg", getattr(model, "cfg", None))
    if cd_cfg is None:
        raise RuntimeError("Loaded contact model does not expose .cfg or .contact_cfg")

    has_gt = not bool(args.test_infer)
    if args.test_infer:
        infer = vto.prepare_test_infer_sample(args, config, device, teacher)
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
        view_idx = int(args.view_idx)
        print(
            f"pred_main meta_id={target_meta}  "
            f"teacher_score={infer['init_score']:.4f}  "
            f"grasp_rank={infer['grasp_rank']}/{max(int(args.top_n), 1) - 1}"
        )
        mask = np.zeros(int(cd_cfg.num_fingertips), dtype=bool)
        gt_flat = None
    else:
        data_root = train_v2b.resolve_data_root(config)
        gt_cache = ContactGTv2bCache(args.contact_gt_cache, robot=config.data.robot)
        gt_cache.preload_all()
        align_cache = train_v2b.AlignMatCache(
            os.path.join(data_root, "scenes"), config.data.camera
        )
        data = vto.build_batch(args, config, device)
        batch = train_v2b.prepare_batch_v2b(
            data,
            teacher,
            SDFAdamHandPointsProvider(
                urdf_path=config.urdf_path,
                meta_path=config.meta_path,
                hand_name=config.hand_name,
                device=str(device),
            ),
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
        scene_pc, scene_seg = vto.pick_scene_cloud_and_seg_for_seed(data, seed_np_early)
        scene_loaded = None
        if "scene_id" in data:
            scene_loaded = int(data["scene_id"].reshape(-1)[0].detach().cpu().item())
        view_idx = int(args.view_idx)
        target_meta = None

    pred_flat, stab_pred = vto.predict_contacts(
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
        gt = np.zeros((num_fingers, 3), dtype=np.float64)
    else:
        gt = unflatten_contacts(gt_flat, num_fingers)[0].detach().cpu().numpy()
        scene_valid = np.isfinite(scene_pc).all(axis=1)
        scene_pc = scene_pc[scene_valid]
        scene_seg = scene_seg[scene_valid]

    valid_gt = gt[mask] if has_gt else np.zeros((0, 3), dtype=np.float32)
    valid_pred = pred[np.isfinite(pred).all(axis=1)]
    meta_from_gt = vto.estimate_contact_meta_id(valid_gt, scene_pc, scene_seg) if has_gt else None
    meta_from_pred = vto.estimate_contact_meta_id(valid_pred, scene_pc, scene_seg)
    if has_gt:
        target_meta = meta_from_gt if meta_from_gt is not None else meta_from_pred
    elif target_meta is None:
        target_meta = meta_from_pred

    if has_gt:
        print(
            f"contact_mask={mask.astype(int).tolist()}  "
            f"valid_gt={int(mask.sum())}/{num_fingers}"
        )
        vto.describe_dist("GT -> object_pc", vto.nearest_dist(valid_gt, obj))
        vto.describe_dist("Pred -> object_pc (all fingers)", vto.nearest_dist(valid_pred, obj))
        if mask.any():
            vto.describe_dist(
                "Pred -> same-finger GT (masked only)",
                np.linalg.norm(pred[mask] - gt[mask], axis=1),
            )
    else:
        print(f"test_infer: no GT (pred only, {num_fingers} fingers)")
        vto.describe_dist("Pred -> object_pc (all fingers)", vto.nearest_dist(valid_pred, obj))

    if stab_pred is not None:
        print(f"stability_pred[{idx}] = {stab_pred[0].detach().cpu().numpy()}")
    print(
        f"highlight target meta_id={target_meta} "
        f"(from_gt={meta_from_gt}, from_pred={meta_from_pred}, cond_points={obj.shape[0]})"
    )

    hand_provider = SDFAdamHandPointsProvider(
        urdf_path=config.urdf_path,
        meta_path=config.meta_path,
        hand_name=config.hand_name,
        device=str(device),
    )
    _, tip_fn = vto.contact_fit_backend(model_kind)
    tips_cam = tip_fn(
        hand_provider,
        init_trans,
        init_rot,
        init_qpos,
        object_pc=None,
        num_fingers=num_fingers,
    )[0].detach().cpu().numpy()
    grasped_meta_id = vto.estimate_contact_meta_id(tips_cam, scene_pc, scene_seg)

    pred_trans_fit = pred_rot_fit = pred_qpos_fit = None
    pred_fit_stats = None
    if args.with_pred_hand:
        pred_contacts_t = unflatten_contacts(pred_flat, num_fingers)
        # fit_pose_to_contacts 需要 Adam backward，不能处在 @torch.no_grad() 内
        with torch.enable_grad():
            pred_trans_fit, pred_rot_fit, pred_qpos_fit, _, pred_fit_stats = (
                vto.fit_hand_pose_from_pred_contacts(
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
        print(
            "Pred-hand fit: "
            f"tip_err mean={pred_fit_stats['tip_err_mean']:.4f}m  "
            f"max={pred_fit_stats['tip_err_max']:.4f}m"
        )

    pred_proj, pred_proj_dist = vto.nearest_points_on_cloud(pred, obj)

    return {
        "has_gt": has_gt,
        "model_kind": model_kind,
        "version": version,
        "idx": idx,
        "scene_loaded": scene_loaded,
        "view_idx": view_idx,
        "scene_pc": scene_pc,
        "scene_seg": scene_seg,
        "target_meta": target_meta,
        "grasped_meta_id": grasped_meta_id,
        "gt": gt,
        "pred": pred,
        "mask": mask,
        "seed_np": seed_np,
        "obj": obj,
        "tips_cam": tips_cam,
        "pred_proj": pred_proj,
        "pred_proj_dist": pred_proj_dist,
        "init_trans": init_trans,
        "init_rot": init_rot,
        "init_qpos": init_qpos,
        "pred_trans_fit": pred_trans_fit,
        "pred_rot_fit": pred_rot_fit,
        "pred_qpos_fit": pred_qpos_fit,
        "stab_pred": stab_pred[0].detach().cpu().numpy() if stab_pred is not None else None,
    }


def build_plotly_figure(args, result: dict) -> go.Figure:
    """用 vis_contact_stability 同款 Plotly trace 绘制。"""
    fig = go.Figure()
    scene_pc = result["scene_pc"]
    scene_seg = result["scene_seg"]
    target_meta = result["target_meta"]
    grasped_meta_id = result["grasped_meta_id"]
    gt = result["gt"]
    pred = result["pred"]
    mask = result["mask"]
    obj = result["obj"]
    tips_cam = result["tips_cam"]
    pred_proj = result["pred_proj"]
    pred_proj_dist = result["pred_proj_dist"]
    has_gt = result["has_gt"]
    num_fingers = pred.shape[0]

    if args.with_scene_pc or args.with_object_pc:
        max_pts = int(args.max_points_per_obj)
        if max_pts <= 0:
            max_pts = 1_000_000
        scene_objects = vcs.extract_segmented_objects(
            scene_pc,
            scene_seg,
            extrinsic=np.eye(4),
            frame="camera",
            main_object_id=target_meta,
            gt_object_id=target_meta if has_gt else None,
            grasped_object_id=grasped_meta_id,
            max_points_per_obj=max_pts,
            seed=int(args.seed),
        )
        if args.with_scene_pc:
            for obj_info in scene_objects:
                label = (
                    f"Scene MAIN obj{obj_info['obj_id']}"
                    if obj_info["is_main"]
                    else (
                        f"Scene GRASPED obj{obj_info['obj_id']}"
                        if obj_info.get("is_grasped")
                        else f"Scene obj{obj_info['obj_id']}"
                    )
                )
                vcs.add_points_trace(
                    fig,
                    obj_info["points"],
                    label,
                    obj_info["color"],
                    size=2 if (obj_info["is_main"] or obj_info.get("is_grasped")) else 1,
                    opacity=0.55 if (obj_info["is_main"] or obj_info.get("is_grasped")) else 0.2,
                )
        else:
            vcs.add_segmented_object_traces(fig, scene_objects, result["view_idx"])

    if args.with_cond_object_pc and obj.size > 0:
        vcs.add_points_trace(
            fig,
            obj,
            f"Cond object_pc (meta={target_meta})",
            "#AA00FF",
            size=3,
            opacity=0.55,
        )

    for fi in range(num_fingers):
        color = FINGER_COLORS[fi % len(FINGER_COLORS)]
        vcs.add_contact_marker_trace(
            fig,
            tips_cam[fi : fi + 1],
            f"FK tip-{FINGER_LABELS[fi]}",
            color,
            size=11,
            symbol="diamond",
        )
        if has_gt and mask[fi]:
            vcs.add_contact_marker_trace(
                fig,
                gt[fi : fi + 1],
                f"GT v2-{FINGER_LABELS[fi]}",
                GT_V2_IBS_COLOR,
                size=14,
                symbol="x",
                outline_color=vcs.MARKER_OUTLINE_LIGHT,
            )
        vcs.add_contact_marker_trace(
            fig,
            pred[fi : fi + 1],
            f"Pred contact-{FINGER_LABELS[fi]}",
            color,
            size=10,
            symbol="circle",
        )
        if args.project_pred_to_object and np.isfinite(pred_proj_dist[fi]):
            vcs.add_contact_marker_trace(
                fig,
                pred_proj[fi : fi + 1],
                f"Pred→obj proj-{FINGER_LABELS[fi]}",
                color,
                size=8,
                symbol="circle-open",
                outline_color=vcs.MARKER_OUTLINE_LIGHT,
            )
            seg_proj = np.stack([pred[fi], pred_proj[fi]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=seg_proj[:, 0],
                    y=seg_proj[:, 1],
                    z=seg_proj[:, 2],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=f"Pred→obj-{FINGER_LABELS[fi]}",
                    showlegend=False,
                )
            )
        if has_gt and mask[fi]:
            seg_err = np.stack([gt[fi], pred[fi]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=seg_err[:, 0],
                    y=seg_err[:, 1],
                    z=seg_err[:, 2],
                    mode="lines",
                    line=dict(color="#FF00FF", width=4),
                    name=f"GT→Pred err-{FINGER_LABELS[fi]}",
                    showlegend=False,
                )
            )
        seg_tip = np.stack([tips_cam[fi], pred[fi]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=seg_tip[:, 0],
                y=seg_tip[:, 1],
                z=seg_tip[:, 2],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"FK→Pred-{FINGER_LABELS[fi]}",
                showlegend=False,
            )
        )

    if args.with_hand or args.with_pred_hand:
        vis = Vis(
            robot_name=args.hand_name,
            urdf_path=args.urdf_path,
            meta_path=args.meta_path,
        )
        if args.with_hand:
            for trace in vis.robot_plotly(
                result["init_trans"].cpu(),
                result["init_rot"].cpu(),
                result["init_qpos"].cpu(),
                opacity=float(args.hand_opacity),
                color=HAND_MESH_COLOR,
            ):
                trace.name = "Hand (GT/teacher)"
                fig.add_trace(trace)
        if args.with_pred_hand and result["pred_trans_fit"] is not None:
            for trace in vis.robot_plotly(
                result["pred_trans_fit"].cpu(),
                result["pred_rot_fit"].cpu(),
                result["pred_qpos_fit"].cpu(),
                opacity=float(args.pred_hand_opacity),
                color=PRED_HAND_MESH_COLOR,
            ):
                trace.name = "Hand (Pred fit)"
                fig.add_trace(trace)

    mode_tag = "Test Pred" if not has_gt else "Train GT vs Pred"
    scene_str = (
        _scene_tag(result["scene_loaded"])
        if result["scene_loaded"] is not None
        else f"view={result['view_idx']}"
    )
    fig.update_layout(
        title=(
            f"Contact vis ({mode_tag}) | {scene_str} | "
            f"{result['model_kind']} | sample={result['idx']} | frame=camera"
        ),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        showlegend=True,
    )
    return fig


def _apply_o3d_vis_defaults(args) -> None:
    """补齐 o3d visualize 所需字段，便于 plotly 脚本直接调 vto.visualize。"""
    proj = getattr(args, "project_pred_to_object", True)
    defaults = {
        "scene_stride": 1,
        "object_stride": 1,
        "max_target_points": 40000,
        "max_points_per_bg_obj": 8000,
        "with_fk_tips": True,
        "hand_wireframe": True,
        "hand_mesh_type": "collision",
        "show_cond_object_pc": getattr(args, "with_cond_object_pc", False),
        "background": "white",
        "gt_radius": 0.014,
        "pred_radius": 0.012,
        "contact_marker_style": "disc",
        "contact_disc_radius": 0.018,
        "contact_ring_radius": 0.014,
        "contact_patch_radius": 0.024,
        "contact_normal_offset": 0.0008,
        "contact_point_size": 6.0,
        "contact_surface_patch": True,
        "fk_tip_radius": 0.010,
        "proj_radius": 0.008,
        "seed_radius": 0.010,
        "pred_fk_tip_radius": 0.010,
        "frame_size": 0.05,
        "view": "top",
        "zoom": 0.65,
        "point_size": 3.5,
        "project_pred_to_target": int(proj),
        "save_ply": "",
    }
    for key, val in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, val)


def visualize(args) -> None:
    if not args.save_plot:
        _apply_o3d_vis_defaults(args)
        print("Open3D 弹窗模式（vis_train_contact_gt_pred_o3d.visualize）")
        vto.visualize(args)
        return

    set_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu"
    )
    config = load_config(args.yaml, train_v2b.arg_mapping, args)
    if not config.ckpt:
        raise ValueError("--ckpt is required for teacher feature extraction")

    result = run_inference(args, config, device)
    fig = build_plotly_figure(args, result)

    if args.save_plot:
        out_root = _resolve_output_dir(args, result["model_kind"], result["version"])
        sid = result["scene_loaded"]
        if sid is None:
            sid = int(args.scene_id)
        out_dir = os.path.join(out_root, _scene_tag(sid))
        os.makedirs(out_dir, exist_ok=True)
        mode = "test" if args.test_infer else "train"
        out_path = os.path.join(
            out_dir,
            f"vis_contact_{mode}_{_scene_tag(sid)}_view{int(args.view_idx):04d}.html",
        )
        fig.write_html(out_path)
        print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contact vis: Open3D 弹窗（默认）或 Plotly HTML（--save_plot）"
    )
    train_v2b.add_argparse(parser, train_v2b.arg_mapping)
    parser.add_argument("--contact_ckpt", type=str, default="")
    parser.add_argument("--contact_stability_ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--test_infer",
        action="store_true",
        help="测试集推理（无 dex GT / contact cache）",
    )
    parser.add_argument("--grasp_num", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--grasp_rank", type=int, default=0)
    parser.add_argument("--graspness_scale", type=float, default=5.0)
    parser.add_argument(
        "--scene_id",
        type=int,
        default=-1,
        help="fixed scene id; -1 keeps DataLoader sampling",
    )
    parser.add_argument("--view_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_batches", type=int, default=0)
    parser.add_argument("--sample_in_batch", type=int, default=0)
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--num_contact_samples", type=int, default=8)
    parser.add_argument("--phy_num_points", type=int, default=4096)
    parser.add_argument("--phy_cdist_chunk", type=int, default=2048)
    parser.add_argument("--with_scene_pc", action="store_true")
    parser.add_argument(
        "--with_object_pc",
        action="store_true",
        help="按 seg 分物体着色（与 with_scene_pc 二选一或同时开）",
    )
    parser.add_argument(
        "--max_points_per_obj",
        type=int,
        default=0,
        help="每物体最多点数；0=不截断",
    )
    parser.add_argument("--with_hand", action="store_true")
    parser.add_argument("--with_pred_hand", action="store_true")
    parser.add_argument("--hand_opacity", type=float, default=0.25)
    parser.add_argument("--pred_hand_opacity", type=float, default=0.35)
    parser.add_argument("--fit_steps", type=int, default=32)
    parser.add_argument("--fit_max_trans_delta", type=float, default=0.10)
    parser.add_argument("--fit_max_q_delta", type=float, default=0.30)
    parser.add_argument("--fit_max_rot_delta", type=float, default=0.40)
    parser.add_argument("--fit_w_pose", type=float, default=0.01)
    parser.add_argument("--fit_use_model_cfg", action="store_true")
    parser.add_argument("--with_cond_object_pc", action="store_true")
    parser.add_argument(
        "--project_pred_to_object",
        action="store_true",
        default=True,
        help="绘制 Pred 在 cond object_pc 上的投影（○）",
    )
    parser.add_argument(
        "--no_project_pred_to_object",
        action="store_false",
        dest="project_pred_to_object",
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="存 Plotly HTML；不加则 Open3D 交互弹窗",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="HTML 输出根目录（默认 experiments/.../plotly_vis/...）",
    )
    args = parser.parse_args()

    if int(args.scene_id) < 0 and not args.test_infer:
        raise ValueError("请指定 --scene_id（训练 vis 固定 scene/view）")
    if args.test_infer and int(args.scene_id) < 0:
        raise ValueError("--test_infer 必须指定 --scene_id")
    if bool(args.contact_ckpt) == bool(args.contact_stability_ckpt):
        raise ValueError("请提供 --contact_ckpt 或 --contact_stability_ckpt 之一")

    visualize(args)


if __name__ == "__main__":
    main()
