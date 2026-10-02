"""Per-view top-n selection helpers for coarse grasp candidates."""

from __future__ import annotations


def select_top_n_per_view_with_feature(
    scores,
    rotations,
    translations,
    qposs,
    grasp_points,
    features,
    top_n,
):
    import torch

    _B, k = scores.shape
    top_n = min(int(top_n), int(k))

    _, top_indices = torch.topk(scores, top_n, dim=1)
    batch_idx = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand(
        -1, top_n
    )

    return (
        rotations[batch_idx, top_indices],
        translations[batch_idx, top_indices],
        qposs[batch_idx, top_indices],
        scores[batch_idx, top_indices],
        grasp_points[batch_idx, top_indices],
        features[batch_idx, top_indices],
    )
