from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading

from region_contact.batch_scene_utils import BatchSceneTask, build_batch_tasks


def _worker(worker_id, gpu, tasks, args, passthrough):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.path.abspath("src")
    while True:
        try:
            task: BatchSceneTask = tasks.get_nowait()
        except queue.Empty:
            return
        cmd = [
            sys.executable,
            "-m",
            "region_contact.evaluate_anchor_patch",
            "--ckpt_path",
            args.ckpt_path,
            "--device",
            "cuda:0",
            "--scene_id",
            task.eval_scene_id,
            "--seed",
            str(args.seed),
            "--headless",
            str(args.headless),
            "--batch_size",
            str(args.batch_size),
            "--overwrite",
            str(args.overwrite),
            "--dataset",
            args.dataset,
            "--mesh_root",
            args.mesh_root,
            "--result_subdir",
            args.result_subdir,
        ]
        if args.result_exp_root:
            cmd += ["--result_exp_root", args.result_exp_root]
        cmd += passthrough
        print(f"[Worker {worker_id} | GPU {gpu}] {' '.join(cmd)}", flush=True)
        ret = subprocess.call(cmd, env=env)
        if ret != 0:
            print(f"[AnchorPatch] evaluate {task.label} failed with code {ret}", flush=True)
        tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate anchor-patch results")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--result_subdir", default="results_anchor_patch_v16e")
    parser.add_argument(
        "--result_exp_root",
        type=str,
        default="",
        help="Optional root for grasps.npz / sim_success.npy "
        "(e.g. /data/Final_exp_result/graspnet). Must match predict.",
    )
    parser.add_argument("--gpu_list", type=str, default="0")
    parser.add_argument("--scene_id_start", type=int, default=None)
    parser.add_argument("--scene_id_end", type=int, default=None)
    parser.add_argument(
        "--scene_ids",
        type=str,
        nargs="*",
        default=None,
        help="Explicit scene ids, e.g. scene_0235 scene_9540. "
        "Overrides --split / --scene_id_start/--scene_id_end.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="GraspNet: debug/seen/similar/novel/dense/loose/random; "
        "ACRONYM: dense/random/loose/all. If empty, use scene_id_start/end "
        "or --scene_ids.",
    )
    parser.add_argument("--dataset", default="graspnet", choices=["graspnet", "acronym"])
    parser.add_argument("--mesh_root", default="/data/meshdata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)
    args, passthrough = parser.parse_known_args()

    if (
        not args.scene_ids
        and not args.split
        and (args.scene_id_start is None or args.scene_id_end is None)
    ):
        raise SystemExit(
            "Provide --scene_ids, --split (e.g. dense / all), "
            "or both --scene_id_start and --scene_id_end."
        )

    plan = build_batch_tasks(
        dataset=args.dataset,
        split=args.split,
        scene_id_start=args.scene_id_start,
        scene_id_end=args.scene_id_end,
        scene_ids=args.scene_ids,
    )
    print(
        f"[AnchorPatchEval] dataset={args.dataset} split={args.split or 'custom'} "
        f"scenes={len(plan.tasks)} result_exp_root={args.result_exp_root or '(ckpt parent)'} "
        f"result_subdir={args.result_subdir}",
        flush=True,
    )

    tasks = queue.Queue()
    for task in plan.tasks:
        tasks.put(task)
    gpus = [x.strip() for x in str(args.gpu_list).split(",") if x.strip()]
    threads = []
    for i, gpu in enumerate(gpus):
        th = threading.Thread(target=_worker, args=(i + 1, gpu, tasks, args, passthrough))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()


if __name__ == "__main__":
    main()
