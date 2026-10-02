'''
StaConGrasp adapter for the bundled v16e evaluation backend:
  - all top_n coarse candidates are refined
  - dynamic per-step patch rebuild (tip/contact blend anchors)
  - physics-aware post-refine rerank before saving

'''

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

from stacongrasp.method.anchor_patch_post_rerank import (
    PostRefineRerankConfig,
    merge_refine_metrics_into_kwargs,
    reset_refine_metrics,
    select_top1_per_view_post_rerank,
)
from stacongrasp.method.anchor_patch_refine_dyn import (
    AnchorPatchDynConfig,
    AnchorPatchPoseRefinerDyn,
)
V16E_POINTNEXT_MODULE = "eval.predict_stacongrasp_pointnext"
V16E_V4B_MODULE = "eval.predict_stacongrasp_pointnet"


def _get_arg_value(argv, name, default=None):
    if name not in argv:
        return default
    idx = argv.index(name)
    if idx + 1 >= len(argv):
        return default
    return argv[idx + 1]


def _contact_ckpt_version(path):
    if not path:
        return ""
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        return str(ckpt.get("version", ""))
    return ""


def _parse_stacongrasp_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor_patch_radius", type=float, default=0.02)
    parser.add_argument("--anchor_patch_topk", type=int, default=64)
    parser.add_argument("--anchor_patch_sigma", type=float, default=0.01)
    parser.add_argument("--anchor_patch_normal_topk", type=int, default=32)
    parser.add_argument("--anchor_patch_normal_weight", type=float, default=0.0)
    parser.add_argument("--anchor_patch_reach_sigma", type=float, default=0.03)
    parser.add_argument("--anchor_patch_reach_weight", type=float, default=0.0)
    parser.add_argument("--anchor_patch_min_points", type=int, default=4)
    parser.add_argument("--anchor_patch_stability_coverage_weight", type=float, default=0.05)
    parser.add_argument("--anchor_patch_stability_spread_weight", type=float, default=0.05)
    parser.add_argument("--anchor_patch_accept_require_patch_improve", type=int, default=1)
    parser.add_argument("--anchor_patch_accept_patch_min_improve", type=float, default=1e-8)
    parser.add_argument("--anchor_patch_accept_stability_eps", type=float, default=0.0)
    parser.add_argument("--anchor_patch_select_top1", type=int, default=1)

    parser.add_argument(
        "--anchor_patch_anchor_mode",
        type=str,
        default="tip_blend",
        choices=["fixed", "tip_blend", "tip_only"],
    )
    parser.add_argument("--anchor_patch_blend_alpha", type=float, default=0.35)
    parser.add_argument("--anchor_patch_blend_alpha_max", type=float, default=0.7)
    parser.add_argument("--anchor_patch_adaptive_alpha", type=int, default=1)
    parser.add_argument("--anchor_patch_adaptive_alpha_tau", type=float, default=0.02)

    parser.add_argument("--post_rerank_w_dex", type=float, default=0.2)
    parser.add_argument("--post_rerank_w_patch", type=float, default=0.3)
    parser.add_argument("--post_rerank_w_contact", type=float, default=0.3)
    parser.add_argument("--post_rerank_w_pen", type=float, default=0.5)
    parser.add_argument("--post_rerank_w_energy", type=float, default=0.5)
    parser.add_argument("--post_rerank_w_refined", type=float, default=0.1)
    parser.add_argument("--post_rerank_w_patch_improve", type=float, default=0.2)
    parser.add_argument("--post_rerank_w_pen_improve", type=float, default=0.2)
    parser.add_argument("--post_rerank_penalty_contact_dist", type=float, default=0.1)
    return parser.parse_known_args(argv)


def _savez_with_stacongrasp_fields(orig_savez, rerank_cfg: PostRefineRerankConfig, top_n: int):
    def wrapped(file, *args, **kwargs):
        try:
            from stacongrasp.method import anchor_patch_refine as patch_mod
            patch_log = getattr(patch_mod, "LAST_PATCH_LOG", {}) or {}
            mapping = {
                "patch_centers": "anchor_patch_centers_cam",
                "patch_points": "anchor_patch_points_cam",
                "patch_mask": "anchor_patch_mask",
                "patch_weights": "anchor_patch_weights",
                "patch_normals": "anchor_patch_normals_cam",
                "patch_representatives": "anchor_patch_representatives_cam",
                "patch_coverage": "anchor_patch_coverage",
                "patch_spread": "anchor_patch_spread",
                "anchor_projected_dist": "anchor_patch_projected_dist",
                "patch_stability_score": "anchor_patch_stability_score",
                "patch_accept_ok": "anchor_patch_accept_ok",
                "patch_stability_ok": "anchor_patch_stability_ok",
                "patch_improve": "anchor_patch_improve",
                "dynamic_blend_alpha": "anchor_patch_dynamic_blend_alpha",
            }
            for src, dst in mapping.items():
                if src in patch_log and dst not in kwargs:
                    kwargs[dst] = patch_log[src].detach().cpu().numpy()
            kwargs.setdefault("use_anchor_patch", np.ones(len(kwargs["translation"]), dtype=np.int64))
            kwargs.setdefault("use_anchor_patch_dyn", np.ones(len(kwargs["translation"]), dtype=np.int64))
        except Exception:
            pass

        kwargs = merge_refine_metrics_into_kwargs(kwargs)
        n_before = len(kwargs.get("translation", []))
        if int(rerank_cfg.enabled):
            kwargs = select_top1_per_view_post_rerank(kwargs, rerank_cfg, top_n=top_n)
            n_after = len(kwargs.get("translation", []))
            print(
                f"[StaConGrasp] post-refine rerank top1: {n_before} -> {n_after} "
                f"(w_pen={rerank_cfg.w_pen}, w_energy={rerank_cfg.w_energy})",
                flush=True,
            )
        return orig_savez(file, *args, **kwargs)

    return wrapped


def _ensure_cli_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv:
        return argv
    return argv + [flag, value]


DEFAULT_TEACHER_CKPT = "/data/DexGraspNet2.0-ckpts/OURS/ckpt/ckpt_50000.pth"
DEFAULT_CONTACT_STABILITY_CKPT = (
    "experiments/contact_stability_v4e_pointnext_offset/ckpt/ckpt_1000.pth"
)
DEFAULT_CONTACT_V2_CKPT = (
    "experiments/contact_diffusion_v2d_pointnext_offset/ckpt/ckpt_1000.pth"
)
DEFAULT_ALLEGRO_URDF = "robot_models/urdf/allegro_hand_simplified.urdf"
DEFAULT_ALLEGRO_META = "robot_models/meta/allegro_hand/meta.yaml"


def main() -> None:
    reset_refine_metrics()
    scheme_args, remaining = _parse_stacongrasp_args(sys.argv[1:])
    hand_name = _get_arg_value(remaining, "--hand_name", "leap_hand") or "leap_hand"
    transfer_profile = _get_arg_value(remaining, "--allegro_transfer_profile", "auto") or "auto"
    embodiment_mode = (
        _get_arg_value(remaining, "--allegro_embodiment_mode", "optimize") or "optimize"
    )
    if (
        hand_name == "allegro_hand"
        and transfer_profile == "auto"
        and embodiment_mode == "retarget"
    ):
        scheme_args.anchor_patch_accept_require_patch_improve = 0
        scheme_args.anchor_patch_accept_patch_min_improve = 0.0
        scheme_args.anchor_patch_accept_stability_eps = max(
            float(scheme_args.anchor_patch_accept_stability_eps), 1e-3
        )

    remaining = _ensure_cli_arg(remaining, "--ckpt_path", DEFAULT_TEACHER_CKPT)
    remaining = _ensure_cli_arg(
        remaining, "--contact_stability_ckpt", DEFAULT_CONTACT_STABILITY_CKPT
    )
    remaining = _ensure_cli_arg(remaining, "--contact_v2_ckpt", DEFAULT_CONTACT_V2_CKPT)
    if hand_name == "allegro_hand":
        remaining = _ensure_cli_arg(remaining, "--urdf_path", DEFAULT_ALLEGRO_URDF)
        remaining = _ensure_cli_arg(remaining, "--meta_path", DEFAULT_ALLEGRO_META)

    top_n = int(_get_arg_value(remaining, "--top_n", "5") or 5)
    rerank_cfg = PostRefineRerankConfig(
        enabled=bool(int(scheme_args.anchor_patch_select_top1)),
        w_dex=float(scheme_args.post_rerank_w_dex),
        w_patch=float(scheme_args.post_rerank_w_patch),
        w_contact=float(scheme_args.post_rerank_w_contact),
        w_pen=float(scheme_args.post_rerank_w_pen),
        w_energy=float(scheme_args.post_rerank_w_energy),
        w_refined=float(scheme_args.post_rerank_w_refined),
        w_patch_improve=float(scheme_args.post_rerank_w_patch_improve),
        w_pen_improve=float(scheme_args.post_rerank_w_pen_improve),
        penalty_contact_dist=float(scheme_args.post_rerank_penalty_contact_dist),
    )
    patch_cfg = AnchorPatchDynConfig(
        patch_radius=float(scheme_args.anchor_patch_radius),
        patch_topk=int(scheme_args.anchor_patch_topk),
        patch_sigma=float(scheme_args.anchor_patch_sigma),
        normal_topk=int(scheme_args.anchor_patch_normal_topk),
        normal_weight=float(scheme_args.anchor_patch_normal_weight),
        reach_sigma=float(scheme_args.anchor_patch_reach_sigma),
        reach_weight=float(scheme_args.anchor_patch_reach_weight),
        min_patch_points=int(scheme_args.anchor_patch_min_points),
        stability_coverage_weight=float(scheme_args.anchor_patch_stability_coverage_weight),
        stability_spread_weight=float(scheme_args.anchor_patch_stability_spread_weight),
        accept_require_patch_improve=bool(int(scheme_args.anchor_patch_accept_require_patch_improve)),
        accept_patch_min_improve=float(scheme_args.anchor_patch_accept_patch_min_improve),
        accept_stability_eps=float(scheme_args.anchor_patch_accept_stability_eps),
        anchor_mode=str(scheme_args.anchor_patch_anchor_mode),
        blend_alpha=float(scheme_args.anchor_patch_blend_alpha),
        blend_alpha_max=float(scheme_args.anchor_patch_blend_alpha_max),
        adaptive_alpha=bool(int(scheme_args.anchor_patch_adaptive_alpha)),
        adaptive_alpha_tau=float(scheme_args.anchor_patch_adaptive_alpha_tau),
    )

    contact_ckpt = _get_arg_value(remaining, "--contact_stability_ckpt", "")
    version = _contact_ckpt_version(contact_ckpt)
    if version == "contact_stability_v4b_offset":
        module_name = V16E_V4B_MODULE
    else:
        module_name = V16E_POINTNEXT_MODULE

    allegro_patch_note = ""
    if hand_name == "allegro_hand" and transfer_profile == "auto":
        allegro_patch_note = (
            " | allegro_patch_accept=relaxed"
            if embodiment_mode == "retarget"
            else " | allegro_patch_accept=strict"
        )

    print(
        f"[StaConGrasp] contact_stability version={version or 'unknown'} -> backend={module_name} | "
        f"anchor_mode={patch_cfg.anchor_mode} blend_alpha={patch_cfg.blend_alpha} "
        f"adaptive={int(patch_cfg.adaptive_alpha)}"
        + allegro_patch_note,
        flush=True,
    )
    v16e = importlib.import_module(module_name)

    class _StaConGraspRefiner(AnchorPatchPoseRefinerDyn):
        def __init__(self, hand_model=None, cfg=None):
            super().__init__(hand_model=hand_model, cfg=cfg, patch_cfg=patch_cfg)

    v16e.PhysicsGuidedPoseRefinerV4StabilityPenGated = _StaConGraspRefiner
    np.savez = _savez_with_stacongrasp_fields(np.savez, rerank_cfg, top_n=top_n)

    sys.argv = [sys.argv[0]] + remaining
    if "--result_subdir" not in remaining:
        sys.argv += ["--result_subdir", "results_stacongrasp_v16e"]
    if "--result_exp_root" not in remaining and "--ckpt_path" in remaining:
        ckpt = remaining[remaining.index("--ckpt_path") + 1]
        sys.argv += ["--result_exp_root", os.path.dirname(os.path.dirname(ckpt))]
    v16e.main()


if __name__ == "__main__":
    main()
