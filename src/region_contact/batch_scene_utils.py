"""
Shared scene-list helpers for anchor-patch batch predict / evaluate.

Aligns with eval.evaluate_dexterous_all splits and
predict_dexterous_all / v16e hybrid scene scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class BatchPlan:
    tasks: List[BatchSceneTask]
    acronym_scene_ids: Optional[List[str]] = None


@dataclass(frozen=True)
class BatchSceneTask:
    """One scene job for batch launchers."""

    label: str
    predict_scene_id: str
    eval_scene_id: str


def acronym_scene_list(split: str = "all") -> List[str]:
    dense = [f"scene_dense_{i}" for i in range(100)]
    random_scenes = [f"scene_random_{i}" for i in range(90)]
    loose = [f"scene_loose_{i}" for i in range(30)]
    mapping = {
        "dense": dense,
        "random": random_scenes,
        "loose": loose,
        "all": dense + random_scenes + loose,
    }
    if split not in mapping:
        raise ValueError(
            f"Unsupported acronym split {split!r}; choose from {sorted(mapping)}"
        )
    return mapping[split]


def graspnet_scene_list(
    split: str,
    scene_id_start: Optional[int] = None,
    scene_id_end: Optional[int] = None,
) -> List[str]:
    predefined = {
        "debug": [f"scene_{i:04d}" for i in range(100, 101)],
        "seen": [f"scene_{i:04d}" for i in range(100, 130)],
        "similar": [f"scene_{i:04d}" for i in range(130, 160)],
        "novel": [f"scene_{i:04d}" for i in range(160, 190)],
        "dense": [f"scene_{i:04d}" for i in range(100, 190)],
        "loose": [f"scene_{i:04d}" for i in range(200, 380)],
        "random": [f"scene_{i:04d}" for i in range(9000, 9900, 5)],
    }
    if split in predefined:
        return predefined[split]
    if scene_id_start is None or scene_id_end is None:
        raise ValueError(
            "graspnet split not set; provide --split or both --scene_id_start and --scene_id_end"
        )
    start = int(scene_id_start)
    end = int(scene_id_end)
    if start >= 8500:
        return [f"scene_{i:04d}" for i in range(start, end, 5)]
    return [f"scene_{i:04d}" for i in range(start, end)]


def build_batch_tasks(
    dataset: str,
    split: str = "",
    scene_id_start: Optional[int] = None,
    scene_id_end: Optional[int] = None,
    scene_ids: Optional[Sequence[str]] = None,
) -> BatchPlan:
    dataset = str(dataset)
    split = str(split or "").strip()
    explicit_ids = [str(s).strip() for s in (scene_ids or []) if str(s).strip()]

    if dataset == "graspnet":
        if explicit_ids:
            scenes = explicit_ids
        elif split in ("", "custom"):
            scenes = graspnet_scene_list("custom", scene_id_start, scene_id_end)
        else:
            scenes = graspnet_scene_list(split, scene_id_start, scene_id_end)
        tasks = [
            BatchSceneTask(label=s, predict_scene_id=s, eval_scene_id=s)
            for s in scenes
        ]
        return BatchPlan(tasks=tasks, acronym_scene_ids=None)

    if dataset == "acronym":
        all_scenes = acronym_scene_list("all")
        if explicit_ids:
            unknown = [s for s in explicit_ids if s not in all_scenes]
            if unknown:
                raise ValueError(f"Unknown ACRONYM scene ids: {unknown}")
            scenes = explicit_ids
            index_of = {name: i for i, name in enumerate(all_scenes)}
            tasks = [
                BatchSceneTask(
                    label=scene_name,
                    predict_scene_id=str(index_of[scene_name]),
                    eval_scene_id=scene_name,
                )
                for scene_name in scenes
            ]
            return BatchPlan(tasks=tasks, acronym_scene_ids=list(all_scenes))

        acr_split = split if split in ("dense", "random", "loose", "all") else "all"
        scenes = acronym_scene_list(acr_split)
        tasks = [
            BatchSceneTask(
                label=scene_name,
                predict_scene_id=str(idx),
                eval_scene_id=scene_name,
            )
            for idx, scene_name in enumerate(scenes)
        ]
        return BatchPlan(tasks=tasks, acronym_scene_ids=list(scenes))

    raise ValueError(f"Unsupported dataset {dataset!r}")


__all__ = [
    "BatchPlan",
    "BatchSceneTask",
    "acronym_scene_list",
    "build_batch_tasks",
    "graspnet_scene_list",
]
