"""
Post-refine physics-aware per-view rerank (Scheme C).

After all top_n candidates are refined, score each candidate with refine-after
signals (penetration, total energy, patch improve, use_refined, etc.) and keep
the best one per view before saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

LAST_REFINE_METRICS: Dict[str, np.ndarray] = {}


@dataclass
class PostRefineRerankConfig:
    enabled: bool = True
    w_dex: float = 0.2
    w_patch: float = 0.3
    w_contact: float = 0.3
    w_pen: float = 0.5
    w_energy: float = 0.5
    w_refined: float = 0.1
    w_patch_improve: float = 0.2
    w_pen_improve: float = 0.2
    penalty_contact_dist: float = 0.1


def reset_refine_metrics() -> None:
    global LAST_REFINE_METRICS
    LAST_REFINE_METRICS = {}


def capture_refine_metrics(refine_out: Dict[str, Any]) -> None:
    """Store per-candidate refine metrics from the latest refiner forward pass."""
    global LAST_REFINE_METRICS
    out: Dict[str, np.ndarray] = {}
    mapping = {
        "init_E_pen": "physics_init_E_pen",
        "final_E_pen": "physics_final_E_pen",
        "init_energy": "physics_init_energy",
        "final_energy": "physics_final_energy",
        "init_E_total": "physics_init_E_total",
        "final_E_total": "physics_final_E_total",
        "energy_improve": "physics_energy_improve",
        "pen_improve": "physics_pen_improve",
        "init_E_patch": "physics_init_E_patch",
        "final_E_patch": "physics_final_E_patch",
        "patch_improve": "anchor_patch_improve",
    }
    for src, dst in mapping.items():
        val = refine_out.get(src)
        if val is None:
            continue
        if hasattr(val, "detach"):
            out[dst] = val.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            out[dst] = np.asarray(val, dtype=np.float32).reshape(-1)
    LAST_REFINE_METRICS = out


def _as_f32(kwargs: Dict[str, Any], key: str, n: int) -> np.ndarray:
    if key not in kwargs:
        return np.zeros(n, dtype=np.float32)
    return np.asarray(kwargs[key], dtype=np.float32).reshape(-1)


def _zscore_global(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - np.nanmean(x)) / std).astype(np.float32)


def _zscore_per_view(
    values: np.ndarray,
    view_ids: np.ndarray,
    top_n: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    view_ids = np.asarray(view_ids, dtype=np.int64).reshape(-1)
    out = np.zeros_like(values, dtype=np.float32)
    if top_n <= 1:
        return out
    for view in np.unique(view_ids):
        idx = np.where(view_ids == view)[0]
        if idx.size <= 1:
            continue
        grp = values[idx]
        std = np.nanstd(grp)
        if not np.isfinite(std) or std < 1e-8:
            out[idx] = 0.0
        else:
            out[idx] = ((grp - np.nanmean(grp)) / std).astype(np.float32)
    return out


def build_post_refine_score(
    kwargs: Dict[str, Any],
    cfg: PostRefineRerankConfig,
    top_n: int = 5,
    viewwise: bool = True,
) -> np.ndarray:
    n = len(kwargs.get("translation", []))
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    view = _as_f32(kwargs, "valid_view_indices", n).astype(np.int64)
    dex = _as_f32(kwargs, "score", n)
    patch = _as_f32(kwargs, "anchor_patch_stability_score", n)
    contact = _as_f32(kwargs, "contact_rerank_score", n)
    if not np.any(contact):
        contact = _as_f32(kwargs, "stability_score", n)
    refined = _as_f32(kwargs, "use_physics_refined", n)
    contact_dist = _as_f32(kwargs, "contact_obj_dist", n)

    init_pen = _as_f32(kwargs, "physics_init_E_pen", n)
    final_pen = _as_f32(kwargs, "physics_final_E_pen", n)
    init_energy = _as_f32(kwargs, "physics_init_energy", n)
    if not np.any(init_energy):
        init_energy = _as_f32(kwargs, "physics_init_E_total", n)
    final_energy = _as_f32(kwargs, "physics_final_energy", n)
    if not np.any(final_energy):
        final_energy = _as_f32(kwargs, "physics_final_E_total", n)
    patch_improve = _as_f32(kwargs, "anchor_patch_improve", n)
    pen_improve = _as_f32(kwargs, "physics_pen_improve", n)
    if not np.any(pen_improve):
        pen_improve = init_pen - final_pen

    zfn = (
        (lambda x: _zscore_per_view(x, view, top_n))
        if viewwise
        else _zscore_global
    )

    score = (
        float(cfg.w_dex) * zfn(dex)
        + float(cfg.w_patch) * zfn(patch)
        + float(cfg.w_contact) * zfn(contact)
        + float(cfg.w_pen) * zfn(-final_pen)
        + float(cfg.w_energy) * zfn(-final_energy)
        + float(cfg.w_refined) * refined
        + float(cfg.w_patch_improve) * zfn(patch_improve)
        + float(cfg.w_pen_improve) * zfn(pen_improve)
        - float(cfg.penalty_contact_dist) * zfn(contact_dist)
    )
    return score.astype(np.float32)


def merge_refine_metrics_into_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    for key, value in LAST_REFINE_METRICS.items():
        if key not in out:
            out[key] = value
    return out


def select_top1_per_view_post_rerank(
    kwargs: Dict[str, Any],
    cfg: PostRefineRerankConfig,
    top_n: int = 5,
) -> Dict[str, Any]:
    if not cfg.enabled:
        return kwargs
    if "valid_view_indices" not in kwargs or "translation" not in kwargs:
        return kwargs

    kwargs = merge_refine_metrics_into_kwargs(kwargs)
    view = np.asarray(kwargs["valid_view_indices"]).reshape(-1)
    n = len(view)
    if n == 0:
        return kwargs

    select_score = build_post_refine_score(kwargs, cfg, top_n=top_n, viewwise=True)
    keep = []
    for v in np.unique(view):
        idx = np.where(view == v)[0]
        if idx.size == 0:
            continue
        best = idx[int(np.nanargmax(select_score[idx]))]
        keep.append(int(best))
    keep = np.asarray(keep, dtype=np.int64)

    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[keep]
        else:
            out[key] = value
    out["anchor_patch_top1_selected_from"] = np.full(len(keep), n, dtype=np.int64)
    out["anchor_patch_top1_select_score"] = select_score[keep]
    out["anchor_patch_top1_select_mode"] = np.array(
        ["post_refine_physics"] * len(keep)
    )
    out["post_refine_rerank_score"] = select_score[keep]
    return out


__all__ = [
    "LAST_REFINE_METRICS",
    "PostRefineRerankConfig",
    "build_post_refine_score",
    "capture_refine_metrics",
    "merge_refine_metrics_into_kwargs",
    "reset_refine_metrics",
    "select_top1_per_view_post_rerank",
]
