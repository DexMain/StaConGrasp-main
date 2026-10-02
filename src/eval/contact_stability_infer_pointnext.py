"""
Contact-Stability pointnext-offset inference：

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.qpos_adapter import score_allegro_contact_candidates_batched

from network.contact_diffusion_pointnext_offset import (
    ContactDiffusionConfig,
    _forward_finger_representatives,
    fit_pose_to_contacts,
)
from network.contact_stability_pointnext_offset import (
    ContactStabilityNet,
    load_contact_stability_net,
)
from utils.contact_gt_surface import unflatten_contacts
from optimizer.physics_guided_diffusion_patch import PhysicsGuidedPoseRefiner


@dataclass
class ContactStabilityInferConfig:
    num_contact_samples: int = 8
    fit_max_trans_delta: float = 0.003
    fit_max_q_delta: float = 0.08
    fit_pen_eps: float = 1e-6
    use_fit_conservative_accept: bool = True
    skip_pose_replace: bool = False
    contact_obj_dist_weight: float = 20.0
    contact_obj_dist_clip: float = 0.20
    project_contacts_to_object: bool = True
    allegro_contact_rerank: bool = False
    allegro_contact_rerank_ik_steps: int = 40
    allegro_contact_rerank_w_ik: float = 800.0
    allegro_contact_rerank_w_stab: float = 1.0
    allegro_contact_rerank_min_ik_rel_improve: float = 0.25
    allegro_contact_rerank_max_stab_drop: float = 0.35


def stability_score_from_pred(stab_pred: torch.Tensor) -> torch.Tensor:
    """
    stab_pred (..., 4): [com_offset, moment_arm, bilateral, elongated_stability]
    越高越稳定。
    """
    return (
        -stab_pred[..., 0]
        - stab_pred[..., 1]
        + stab_pred[..., 2]
        + stab_pred[..., 3]
    )


def _contact_mse_loss(
    hand_provider,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    target_contacts: torch.Tensor,
    object_pc: torch.Tensor,
    num_fingers: int,
) -> torch.Tensor:
    tips = _forward_finger_representatives(
        hand_provider,
        trans,
        rot,
        qpos,
        object_pc=object_pc,
        num_fingers=num_fingers,
    )
    return (tips - target_contacts).square().mean(dim=(-1, -2))


def _contact_to_object_mean_dist(
    contacts: torch.Tensor,
    object_pc: torch.Tensor,
    chunk: int = 2048,
) -> torch.Tensor:
    """
    contacts: (B,K,F,3) or (B,F,3)
    object_pc: (B,N,3), padded rows allowed.
    Returns mean nearest distance over fingers: (B,K) or (B,)
    """
    squeeze_k = False
    if contacts.dim() == 3:
        contacts = contacts.unsqueeze(1)
        squeeze_k = True

    b, k, f, _ = contacts.shape
    out = torch.zeros(b, k, device=contacts.device, dtype=contacts.dtype)
    for bi in range(b):
        obj = object_pc[bi]
        valid = torch.isfinite(obj).all(dim=-1) & (obj.abs().sum(dim=-1) > 1e-8)
        obj = obj[valid]
        if obj.shape[0] == 0:
            out[bi].fill_(float("inf"))
            continue
        pts = contacts[bi].reshape(k * f, 3)
        min_d = torch.full((pts.shape[0],), float("inf"), device=pts.device, dtype=pts.dtype)
        for start in range(0, obj.shape[0], chunk):
            d = torch.cdist(pts, obj[start : start + chunk]).min(dim=-1).values
            min_d = torch.minimum(min_d, d)
        out[bi] = min_d.reshape(k, f).mean(dim=-1)
    return out.squeeze(1) if squeeze_k else out


def _project_contacts_to_object(
    contacts: torch.Tensor,
    object_pc: torch.Tensor,
    chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project every contact candidate to its nearest point on conditioning object_pc.

    contacts: (B,K,F,3)
    Returns:
        projected contacts with same shape
        raw nearest distances before projection: (B,K,F)
    """
    b, k, f, _ = contacts.shape
    projected = contacts.clone()
    dist_out = torch.full((b, k, f), float("inf"), device=contacts.device, dtype=contacts.dtype)
    for bi in range(b):
        obj = object_pc[bi]
        valid = torch.isfinite(obj).all(dim=-1) & (obj.abs().sum(dim=-1) > 1e-8)
        obj = obj[valid]
        if obj.shape[0] == 0:
            continue
        pts = contacts[bi].reshape(k * f, 3)
        nearest = pts.clone()
        min_d = torch.full((pts.shape[0],), float("inf"), device=pts.device, dtype=pts.dtype)
        for start in range(0, obj.shape[0], chunk):
            chunk_obj = obj[start : start + chunk]
            dmat = torch.cdist(pts, chunk_obj)
            vals, idx = dmat.min(dim=-1)
            update = vals < min_d
            if bool(update.any()):
                min_d = torch.where(update, vals, min_d)
                nearest = torch.where(update[:, None], chunk_obj[idx], nearest)
        projected[bi] = nearest.reshape(k, f, 3)
        dist_out[bi] = min_d.reshape(k, f)
    return projected, dist_out


def _fingertip_penetration(
    hand_provider,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    object_pc: torch.Tensor,
    num_fingers: int,
    sdf_grid: torch.Tensor,
    sdf_origin: torch.Tensor,
    sdf_voxel_size: torch.Tensor,
    T_cam_to_obj: torch.Tensor,
) -> torch.Tensor:
    tips = _forward_finger_representatives(
        hand_provider,
        trans,
        rot,
        qpos,
        object_pc=object_pc,
        num_fingers=num_fingers,
    )
    refiner = PhysicsGuidedPoseRefiner(hand_model=hand_provider)
    tips_obj = refiner.transform_points_cam_to_obj(tips, T_cam_to_obj)
    sdf = refiner.query_sdf_trilinear(
        tips_obj, sdf_grid, sdf_origin, sdf_voxel_size
    )
    return F.relu(-sdf).mean(dim=-1)


def sample_and_rank_contacts(
    model: ContactStabilityNet,
    feature: torch.Tensor,
    seed_points: torch.Tensor,
    object_pc: torch.Tensor,
    num_samples: int = 8,
    contact_obj_dist_weight: float = 20.0,
    contact_obj_dist_clip: float = 0.20,
    project_contacts_to_object: bool = True,
    return_aux: bool = False,
    allegro_rerank: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        flat_best (B, contact_dim) 绝对坐标
        stab_pred_best (B, 4)
        score_best (B,)
        best_idx (B,) int64
    """
    b = feature.shape[0]
    k = max(int(num_samples), 1)
    flat_k = torch.stack(
        [
            model.sample_contacts(feature, seed_points, object_pc)
            for _ in range(k)
        ],
        dim=1,
    )

    with torch.no_grad():
        cond = model.encode_condition(feature, seed_points, object_pc)
        shape_feat, _ = model.encode_shape(object_pc)
        raw_contacts_k = unflatten_contacts(
            flat_k.reshape(b * k, -1), model.contact_cfg.num_fingertips
        ).reshape(b, k, model.contact_cfg.num_fingertips, 3)
        projected_contacts_k, raw_dist_per_finger_k = _project_contacts_to_object(
            raw_contacts_k, object_pc
        )
        obj_dist_k = raw_dist_per_finger_k.mean(dim=-1)
        contacts_for_score_k = projected_contacts_k if project_contacts_to_object else raw_contacts_k
        flat_for_score_k = contacts_for_score_k.reshape(b, k, -1)

        cond_k = cond.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        shape_k = shape_feat.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        flat_flat = flat_for_score_k.reshape(b * k, -1)
        stab_k = model.predict_stability(cond_k, shape_k, flat_flat).reshape(b, k, -1)
        raw_score_k = stability_score_from_pred(stab_k)
        obj_penalty_k = torch.clamp(obj_dist_k, max=float(contact_obj_dist_clip))
        score_k = raw_score_k - float(contact_obj_dist_weight) * obj_penalty_k
        stab_best_idx = score_k.argmax(dim=1)
        allegro_ik_mse_k = None
        if allegro_rerank is not None and bool(allegro_rerank.get("enabled", False)):
            with torch.enable_grad():
                ik_mse_k = score_allegro_contact_candidates_batched(
                    contacts_for_score_k,
                    allegro_rerank["qpos"],
                    allegro_rerank["trans"],
                    allegro_rerank["rot"],
                    allegro_rerank["robot_model"],
                    ik_steps=int(allegro_rerank.get("ik_steps", 40)),
                    lr=float(allegro_rerank.get("lr", 0.04)),
                    pose_reg=float(allegro_rerank.get("pose_reg", 0.05)),
                    max_trans_delta=float(allegro_rerank.get("max_trans_delta", 0.10)),
                    max_rot_delta=float(allegro_rerank.get("max_rot_delta", 0.80)),
                    wrist_warm_start=bool(allegro_rerank.get("wrist_warm_start", True)),
                    coordination_reg_weight=float(
                        allegro_rerank.get("coordination_reg_weight", 0.0)
                    ),
                    canonical_abduction_reg_weight=float(
                        allegro_rerank.get("canonical_abduction_reg_weight", 0.0)
                    ),
                )
            allegro_ik_mse_k = ik_mse_k.detach()
            w_ik = float(allegro_rerank.get("w_ik", 800.0))
            w_stab = float(allegro_rerank.get("w_stab", 1.0))
            score_k = w_stab * score_k - w_ik * allegro_ik_mse_k
            allegro_best_idx = score_k.argmax(dim=1)
            min_rel_improve = float(allegro_rerank.get("min_ik_rel_improve", 0.25))
            max_stab_drop = float(allegro_rerank.get("max_stab_drop", 0.35))
            batch_ids = torch.arange(b, device=score_k.device)
            ik_at_stab = allegro_ik_mse_k[batch_ids, stab_best_idx]
            ik_at_allegro = allegro_ik_mse_k[batch_ids, allegro_best_idx]
            stab_at_stab = raw_score_k[batch_ids, stab_best_idx]
            stab_at_allegro = raw_score_k[batch_ids, allegro_best_idx]
            improve_ok = ik_at_allegro <= ik_at_stab * (1.0 - min_rel_improve)
            stab_ok = stab_at_allegro >= (stab_at_stab - max_stab_drop)
            best_idx = torch.where(improve_ok & stab_ok, allegro_best_idx, stab_best_idx)
        else:
            best_idx = stab_best_idx
        flat_best = flat_for_score_k[torch.arange(b, device=flat_for_score_k.device), best_idx]
        stab_best = stab_k[torch.arange(b, device=stab_k.device), best_idx]
        score_best = score_k[torch.arange(b, device=score_k.device), best_idx]
        raw_score_best = raw_score_k[torch.arange(b, device=raw_score_k.device), best_idx]
        obj_dist_best = obj_dist_k[torch.arange(b, device=obj_dist_k.device), best_idx]
        projected_obj_dist_best = _contact_to_object_mean_dist(
            contacts_for_score_k[torch.arange(b, device=contacts_for_score_k.device), best_idx],
            object_pc,
        )

    if return_aux:
        aux = {
            "raw_stability_score": raw_score_best,
            "rerank_score": score_best,
            "contact_obj_dist": obj_dist_best,
            "contact_projected_obj_dist": projected_obj_dist_best,
            "raw_stability_score_k": raw_score_k,
            "rerank_score_k": score_k,
            "contact_obj_dist_k": obj_dist_k,
            "contact_stab_best_idx": stab_best_idx,
        }
        if allegro_ik_mse_k is not None:
            aux["allegro_contact_rerank_ik_mse_k"] = allegro_ik_mse_k
            aux["allegro_contact_rerank_ik_mse_best"] = allegro_ik_mse_k[
                torch.arange(b, device=allegro_ik_mse_k.device), best_idx
            ]
            aux["allegro_contact_rerank_switched"] = (
                best_idx != stab_best_idx
            ).float()
            aux["allegro_contact_rerank_proposed"] = (
                allegro_best_idx != stab_best_idx
            ).float()
        return flat_best, stab_best, score_best, best_idx, aux
    return flat_best, stab_best, score_best, best_idx


def load_contact_stability_for_infer(
    ckpt_path: str,
    device: torch.device | str,
    contact_v2_ckpt: Optional[str] = None,
) -> ContactStabilityNet:
    return load_contact_stability_net(
        ckpt_path, device=device, contact_v2_ckpt=contact_v2_ckpt
    )


def refine_poses_with_contact_stability(
    model: ContactStabilityNet,
    hand_provider,
    feature: torch.Tensor,
    seed_points: torch.Tensor,
    trans: torch.Tensor,
    rot: torch.Tensor,
    qpos: torch.Tensor,
    object_pc: torch.Tensor,
    fit_cfg: Optional[ContactDiffusionConfig] = None,
    infer_cfg: Optional[ContactStabilityInferConfig] = None,
    sdf_grid: Optional[torch.Tensor] = None,
    sdf_origin: Optional[torch.Tensor] = None,
    sdf_voxel_size: Optional[torch.Tensor] = None,
    T_cam_to_obj: Optional[torch.Tensor] = None,
    allegro_rerank: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    fit_cfg = fit_cfg or model.contact_cfg
    infer_cfg = infer_cfg or ContactStabilityInferConfig()

    teacher_trans = trans.detach().clone()
    teacher_rot = rot.detach().clone()
    teacher_qpos = qpos.detach().clone()

    flat, stab_pred, score_best, best_idx, rank_aux = sample_and_rank_contacts(
        model,
        feature,
        seed_points,
        object_pc,
        num_samples=infer_cfg.num_contact_samples,
        contact_obj_dist_weight=infer_cfg.contact_obj_dist_weight,
        contact_obj_dist_clip=infer_cfg.contact_obj_dist_clip,
        project_contacts_to_object=infer_cfg.project_contacts_to_object,
        return_aux=True,
        allegro_rerank=allegro_rerank,
    )
    target_contacts = unflatten_contacts(flat, fit_cfg.num_fingertips)
    num_fingers = fit_cfg.num_fingertips

    with torch.no_grad():
        fit_loss_init = _contact_mse_loss(
            hand_provider,
            teacher_trans,
            teacher_rot,
            teacher_qpos,
            target_contacts,
            object_pc,
            num_fingers,
        )
        _, elong_gate = model.encode_shape(object_pc)

    if infer_cfg.skip_pose_replace:
        b = teacher_trans.shape[0]
        device = teacher_trans.device
        log = {
            "fit_loss": fit_loss_init.detach(),
            "target_contacts": target_contacts.detach(),
            "stability_pred": stab_pred.detach(),
            "stability_score": score_best.detach(),
            "contact_raw_stability_score": rank_aux["raw_stability_score"].detach(),
            "contact_rerank_score": rank_aux["rerank_score"].detach(),
            "contact_obj_dist": rank_aux["contact_obj_dist"].detach(),
            "contact_projected_obj_dist": rank_aux["contact_projected_obj_dist"].detach(),
            "contact_best_idx": best_idx.detach(),
            "elongated_gate": elong_gate.detach(),
            "fit_loss_init": fit_loss_init.detach(),
            "fit_loss_final": fit_loss_init.detach(),
            "use_contact_fit": torch.zeros(b, device=device),
            "fit_trans_delta": torch.zeros(b, device=device),
            "fit_q_delta": torch.zeros(b, device=device),
        }
        if "contact_stab_best_idx" in rank_aux:
            log["contact_stab_best_idx"] = rank_aux["contact_stab_best_idx"].detach()
        if "allegro_contact_rerank_ik_mse_best" in rank_aux:
            log["allegro_contact_rerank_ik_mse"] = rank_aux[
                "allegro_contact_rerank_ik_mse_best"
            ].detach()
            log["allegro_contact_rerank_switched"] = rank_aux[
                "allegro_contact_rerank_switched"
            ].detach()
            if "allegro_contact_rerank_proposed" in rank_aux:
                log["allegro_contact_rerank_proposed"] = rank_aux[
                    "allegro_contact_rerank_proposed"
                ].detach()
        return teacher_trans, teacher_rot, teacher_qpos, log

    pen_teacher = None
    if (
        sdf_grid is not None
        and sdf_origin is not None
        and sdf_voxel_size is not None
        and T_cam_to_obj is not None
    ):
        with torch.no_grad():
            pen_teacher = _fingertip_penetration(
                hand_provider,
                teacher_trans,
                teacher_rot,
                teacher_qpos,
                object_pc,
                num_fingers,
                sdf_grid,
                sdf_origin,
                sdf_voxel_size,
                T_cam_to_obj,
            )

    trans_fit, rot_fit, qpos_fit, fit_log = fit_pose_to_contacts(
        hand_provider,
        teacher_trans,
        teacher_rot,
        teacher_qpos,
        target_contacts,
        object_pc=object_pc,
        cfg=fit_cfg,
    )

    with torch.no_grad():
        fit_loss_final = _contact_mse_loss(
            hand_provider,
            trans_fit,
            rot_fit,
            qpos_fit,
            target_contacts,
            object_pc,
            num_fingers,
        )
        trans_delta = torch.norm(trans_fit - teacher_trans, dim=-1)
        q_delta = torch.norm(qpos_fit - teacher_qpos, dim=-1)
        if infer_cfg.use_fit_conservative_accept:
            accept = (
                (fit_loss_final < fit_loss_init)
                & (trans_delta < infer_cfg.fit_max_trans_delta)
                & (q_delta < infer_cfg.fit_max_q_delta)
            )
            if pen_teacher is not None:
                pen_fit = _fingertip_penetration(
                    hand_provider,
                    trans_fit,
                    rot_fit,
                    qpos_fit,
                    object_pc,
                    num_fingers,
                    sdf_grid,
                    sdf_origin,
                    sdf_voxel_size,
                    T_cam_to_obj,
                )
                accept = accept & (pen_fit <= pen_teacher + infer_cfg.fit_pen_eps)
            else:
                pen_fit = None
        else:
            accept = torch.ones_like(fit_loss_init, dtype=torch.bool)
            pen_fit = None

        trans_out = torch.where(accept[:, None], trans_fit, teacher_trans)
        rot_out = torch.where(accept[:, None, None], rot_fit, teacher_rot)
        qpos_out = torch.where(accept[:, None], qpos_fit, teacher_qpos)

    log = dict(fit_log)
    log["target_contacts"] = target_contacts.detach()
    log["stability_pred"] = stab_pred.detach()
    log["stability_score"] = score_best.detach()
    log["contact_raw_stability_score"] = rank_aux["raw_stability_score"].detach()
    log["contact_rerank_score"] = rank_aux["rerank_score"].detach()
    log["contact_obj_dist"] = rank_aux["contact_obj_dist"].detach()
    log["contact_projected_obj_dist"] = rank_aux["contact_projected_obj_dist"].detach()
    log["contact_best_idx"] = best_idx.detach()
    log["elongated_gate"] = elong_gate.detach()
    if "contact_stab_best_idx" in rank_aux:
        log["contact_stab_best_idx"] = rank_aux["contact_stab_best_idx"].detach()
    if "allegro_contact_rerank_ik_mse_best" in rank_aux:
        log["allegro_contact_rerank_ik_mse"] = rank_aux[
            "allegro_contact_rerank_ik_mse_best"
        ].detach()
        log["allegro_contact_rerank_switched"] = rank_aux[
            "allegro_contact_rerank_switched"
        ].detach()
        if "allegro_contact_rerank_proposed" in rank_aux:
            log["allegro_contact_rerank_proposed"] = rank_aux[
                "allegro_contact_rerank_proposed"
            ].detach()
    log["fit_loss_init"] = fit_loss_init.detach()
    log["fit_loss_final"] = fit_loss_final.detach()
    log["use_contact_fit"] = accept.float().detach()
    log["fit_trans_delta"] = trans_delta.detach()
    log["fit_q_delta"] = q_delta.detach()
    if pen_teacher is not None and pen_fit is not None:
        log["pen_teacher"] = pen_teacher.detach()
        log["pen_fit"] = pen_fit.detach()
    return trans_out, rot_out, qpos_out, log
