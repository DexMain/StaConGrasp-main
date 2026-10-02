from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Union

import torch
import yaml


ALLEGRO_MOUNT_TO_PALM_ROT = torch.tensor(
    [
        [-2.5973486e-06, -7.0710665e-01, -7.0710689e-01],
        [-2.5973477e-06, 7.0710689e-01, -7.0710665e-01],
        [1.0, 0.0, -3.6732051e-06],
    ], dtype=torch.float32)
ALLEGRO_MOUNT_TO_PALM_TRANSLATION = torch.tensor(
    [-0.008219, -0.02063, 0.08086], dtype=torch.float32)

# These matrices are the same hand-canonical frames used by the simulation
# evaluator.  A cross-hand pose conversion must preserve the canonical grasp
# axes, rather than equating the Allegro palm-link axes with Leap's root axes.
LEAP_CANONICAL_FRAME_ROT = torch.tensor(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)
ALLEGRO_CANONICAL_FRAME_ROT = torch.tensor(
    [
        [-0.70710678, -0.70710678, 0.0],
        [-0.70710678, 0.70710678, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=torch.float32,
)

ALLEGRO_FINGER_GROUPS: Dict[str, Sequence[str]] = {
    "index": ("index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"),
    "middle": ("middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"),
    "ring": ("ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"),
    "thumb": ("thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"),
}


def _canonical_qpos(robot) -> Dict[str, float]:
    """Load the robot's neutral pose used by the corresponding dataset."""
    urdf_path = os.path.abspath(str(robot._urdf_path))
    robot_name = os.path.basename(urdf_path)
    robot_name = robot_name.replace("_simplified_free.urdf", "")
    robot_name = robot_name.replace("_simplified.urdf", "")
    robot_name = robot_name.replace("_free.urdf", "")
    robot_name = robot_name.replace(".urdf", "")
    width_meta_path = None
    # Assets may live directly below robot_models/urdf or in a dedicated
    # subdirectory (the ROS-V5 import does the latter).  Walk upward until
    # the sibling robot_models/meta tree is found instead of assuming a
    # fixed directory depth.
    current = os.path.dirname(urdf_path)
    while current and current != os.path.dirname(current):
        candidate = os.path.join(
            current, "meta", robot_name, "width_mapper_meta.yaml"
        )
        if os.path.isfile(candidate):
            width_meta_path = candidate
            break
        current = os.path.dirname(current)
    if width_meta_path is None:
        raise FileNotFoundError(
            f"Cannot locate width_mapper_meta.yaml for {robot_name} "
            f"relative to {urdf_path}"
        )
    with open(width_meta_path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle) or {}
    values = meta.get("canonical_pose", {}).get("qpos", {})
    if not values:
        raise KeyError(f"canonical_pose.qpos missing in {width_meta_path}")
    return {str(name): float(value) for name, value in values.items()}


def _map_joint_from_canonical(
    values: torch.Tensor,
    source_canonical: float,
    source_lower: float,
    source_upper: float,
    target_canonical: float,
    target_lower: float,
    target_upper: float,
    direction: float = 1.0,
) -> torch.Tensor:
    """Map signed motion around canonical poses while preserving both poses.

    ``direction=-1`` is used for Leap MCP flexion (Leap's open-hand
    convention decreases from 1.57 rad, whereas Allegro flexion increases
    from 0 rad).  A single absolute-limit normalization cannot preserve this
    convention and shifts Allegro's neutral pose, especially for the thumb.
    """
    source_delta = values - float(source_canonical)
    positive_source = source_delta >= 0.0
    source_span = torch.where(
        positive_source,
        torch.as_tensor(
            float(source_upper - source_canonical),
            device=values.device,
            dtype=values.dtype,
        ),
        torch.as_tensor(
            float(source_canonical - source_lower),
            device=values.device,
            dtype=values.dtype,
        ),
    ).clamp_min(1e-6)
    desired_sign = torch.where(
        positive_source,
        torch.as_tensor(float(direction), device=values.device, dtype=values.dtype),
        torch.as_tensor(-float(direction), device=values.device, dtype=values.dtype),
    )
    target_span = torch.where(
        desired_sign >= 0.0,
        torch.as_tensor(
            float(target_upper - target_canonical),
            device=values.device,
            dtype=values.dtype,
        ),
        torch.as_tensor(
            float(target_canonical - target_lower),
            device=values.device,
            dtype=values.dtype,
        ),
    )
    mapped = float(target_canonical) + desired_sign * source_delta.abs() * (
        target_span / source_span
    )
    return mapped.clamp(float(target_lower), float(target_upper))


def retarget_leap_to_allegro(
    qpos: torch.Tensor,
    source_robot,
    target_robot,
) -> torch.Tensor:
    """Retarget a Leap qpos tensor to Allegro by finger and joint semantics.

    This is an initialization adapter, not a learned cross-hand model. Each
    corresponding joint is transferred as motion around the source and target
    canonical poses. Subsequent contact/physics refinement is responsible for
    fitting the Allegro geometry.
    """
    source_names = list(source_robot.movable_joint_names)
    target_names = list(target_robot.movable_joint_names)
    if qpos.shape[-1] != len(source_names):
        raise ValueError(
            f'Leap qpos has {qpos.shape[-1]} values, but the source URDF exposes '
            f'{len(source_names)} movable joints: {source_names}')

    source_index = {name: index for index, name in enumerate(source_names)}
    target_index = {name: index for index, name in enumerate(target_names)}
    # Map by joint ROLE, not Leap URDF kinematic order.
    # Leap index chain is j1(base→mcp flex) → j0(mcp→pip, abd limits) → j2 → j3,
    # while Allegro is joint_0(abd from palm) → joint_1/2/3(flex). Using
    # ('j1','j2','j3','j0') previously put MCP flexion into Allegro abduction
    # and abduction into the fingertip DOF, which yields non-anthropomorphic
    # finger poses and deep penetrations (sim "runaway").
    finger_map: Dict[str, Sequence[str]] = {
        'index': ('j0', 'j1', 'j2', 'j3'),    # abd, MCP flex, PIP, DIP
        'middle': ('j4', 'j5', 'j6', 'j7'),
        'ring': ('j8', 'j9', 'j10', 'j11'),
        'thumb': ('j12', 'j13', 'j14', 'j15'),
    }
    target_groups = ALLEGRO_FINGER_GROUPS
    source_canonical = _canonical_qpos(source_robot)
    target_canonical = _canonical_qpos(target_robot)

    output = qpos.new_zeros((*qpos.shape[:-1], len(target_names)))
    for finger, source_group in finger_map.items():
        for joint_idx, (source_name, target_name) in enumerate(
            zip(source_group, target_groups[finger])
        ):
            if source_name not in source_index or target_name not in target_index:
                raise KeyError(f'Cannot map {source_name} -> {target_name}')
            source_joint_index = source_index[source_name]
            target_joint_index = target_index[target_name]
            source_urdf_index = source_robot.joint_names.index(source_name)
            target_urdf_index = target_robot.joint_names.index(target_name)
            # Leap non-thumb MCP flexion uses the opposite open-hand zero
            # convention; all other corresponding roles keep their sign.
            direction = -1.0 if finger != "thumb" and joint_idx == 1 else 1.0
            output[..., target_joint_index] = _map_joint_from_canonical(
                qpos[..., source_joint_index],
                source_canonical.get(source_name, 0.0),
                source_robot.joints_lower[source_urdf_index],
                source_robot.joints_upper[source_urdf_index],
                target_canonical.get(target_name, 0.0),
                target_robot.joints_lower[target_urdf_index],
                target_robot.joints_upper[target_urdf_index],
                direction=direction,
            )
    return output


def apply_allegro_thumb_shape_prior(
    qpos: torch.Tensor,
    robot_model,
    thumb_q0_min: float = 1.20,
    thumb_q1_ratio: float = 0.0,
    thumb_q2_ratio: float = 0.0,
    thumb_q2_bias: float = 0.0,
    thumb_q3_ratio: float = 0.0,
    thumb_q3_bias: float = 0.0,
    flex_from_fingers: bool = False,
) -> torch.Tensor:
    """Floor Allegro thumb opposition (``thumb_joint_0``) after Leap retarget.

    Leap successful grasps keep ``j13/j14/j15`` nearly straight while ``j12``
    rotates the thumb to face the three fingers.  Copying finger-curl ratios
    into Allegro ``thumb_joint_1/2/3`` therefore over-bends the thumb into a
    claw.  By default this prior only raises CMC opposition (``joint_0``) and
    leaves thumb flexion at the Leap-mapped values.

    Set ``flex_from_fingers=True`` only for legacy experiments that intentionally
    couple thumb flex to index/middle/ring.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )

    name_to_idx = {
        name: idx for idx, name in enumerate(robot_model.movable_joint_names)
    }
    if "thumb_joint_0" not in name_to_idx:
        raise KeyError("Allegro thumb shape prior requires thumb_joint_0")

    corrected = qpos.clone()
    thumb_q0_idx = name_to_idx["thumb_joint_0"]
    urdf_i0 = robot_model.joint_names.index("thumb_joint_0")
    q0_hi = float(robot_model.joints_upper[urdf_i0])
    corrected[:, thumb_q0_idx] = torch.maximum(
        corrected[:, thumb_q0_idx],
        torch.full_like(
            corrected[:, thumb_q0_idx],
            min(float(thumb_q0_min), q0_hi),
        ),
    )

    if flex_from_fingers:
        required = [
            "thumb_joint_1",
            "thumb_joint_2",
            "thumb_joint_3",
            "index_joint_1",
            "middle_joint_1",
            "ring_joint_1",
            "index_joint_2",
            "middle_joint_2",
            "ring_joint_2",
            "index_joint_3",
            "middle_joint_3",
            "ring_joint_3",
        ]
        missing = [name for name in required if name not in name_to_idx]
        if missing:
            raise KeyError(f"Allegro thumb flex prior requires joints: {missing}")

        def columns(names: Sequence[str]) -> torch.Tensor:
            return torch.stack([qpos[:, name_to_idx[name]] for name in names], dim=1)

        nonthumb_q1 = columns(
            ["index_joint_1", "middle_joint_1", "ring_joint_1"]
        ).mean(dim=1)
        nonthumb_q2 = columns(
            ["index_joint_2", "middle_joint_2", "ring_joint_2"]
        ).mean(dim=1)
        nonthumb_q3 = columns(
            ["index_joint_3", "middle_joint_3", "ring_joint_3"]
        ).mean(dim=1)
        corrected[:, name_to_idx["thumb_joint_1"]] = torch.maximum(
            corrected[:, name_to_idx["thumb_joint_1"]],
            torch.clamp(float(thumb_q1_ratio) * nonthumb_q1, min=0.0),
        )
        corrected[:, name_to_idx["thumb_joint_2"]] = torch.maximum(
            corrected[:, name_to_idx["thumb_joint_2"]],
            torch.clamp(
                float(thumb_q2_ratio) * nonthumb_q2 + float(thumb_q2_bias),
                min=0.0,
            ),
        )
        corrected[:, name_to_idx["thumb_joint_3"]] = torch.maximum(
            corrected[:, name_to_idx["thumb_joint_3"]],
            torch.clamp(
                float(thumb_q3_ratio) * nonthumb_q3 + float(thumb_q3_bias),
                min=0.0,
            ),
        )

    lower = torch.tensor(
        [
            robot_model.joints_lower[robot_model.joint_names.index(name)]
            for name in robot_model.movable_joint_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [
            robot_model.joints_upper[robot_model.joint_names.index(name)]
            for name in robot_model.movable_joint_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(
        torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi)
    )
    finite_upper = torch.where(
        torch.isfinite(upper), upper, torch.full_like(upper, torch.pi)
    )
    return corrected.clamp(finite_lower, finite_upper)


def apply_allegro_thumb_opposition_pose_prior(
    qpos: torch.Tensor,
    robot_model,
    thumb_q0_min: float = 1.15,
    # Leap success: j13≈0, j14≈0.4, j15≈0. Keep Allegro near that, not claw.
    thumb_q1_min: float = 0.0,
    thumb_q2_min: float = 0.10,
    thumb_q3_min: float = 0.0,
    thumb_q1_max: float = 0.28,
    thumb_q2_max: float = 0.40,
    thumb_q3_max: float = 0.12,
) -> torch.Tensor:
    """Allegro thumb: oppose (j0), Leap-like open flex (avoid tip poke).

    Keep distal thumb nearly straight so an open hand does not stab the object.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )

    name_to_idx = {
        name: idx for idx, name in enumerate(robot_model.movable_joint_names)
    }
    thumb_mins = {
        "thumb_joint_0": float(thumb_q0_min),
        "thumb_joint_1": float(thumb_q1_min),
        "thumb_joint_2": float(thumb_q2_min),
        "thumb_joint_3": float(thumb_q3_min),
    }
    thumb_maxs = {
        "thumb_joint_1": float(thumb_q1_max),
        "thumb_joint_2": float(thumb_q2_max),
        "thumb_joint_3": float(thumb_q3_max),
    }
    missing = [name for name in thumb_mins if name not in name_to_idx]
    if missing:
        raise KeyError(f"Allegro thumb pose prior requires joints: {missing}")

    corrected = qpos.clone()
    for name, floor in thumb_mins.items():
        col = name_to_idx[name]
        urdf_i = robot_model.joint_names.index(name)
        lo = float(robot_model.joints_lower[urdf_i])
        hi = float(robot_model.joints_upper[urdf_i])
        target = min(max(floor, lo), hi)
        corrected[:, col] = torch.maximum(
            corrected[:, col],
            torch.full_like(corrected[:, col], target),
        )
    for name, ceiling in thumb_maxs.items():
        col = name_to_idx[name]
        urdf_i = robot_model.joint_names.index(name)
        lo = float(robot_model.joints_lower[urdf_i])
        hi = float(robot_model.joints_upper[urdf_i])
        target = min(max(ceiling, lo), hi)
        corrected[:, col] = torch.minimum(
            corrected[:, col],
            torch.full_like(corrected[:, col], target),
        )

    lower = torch.tensor(
        [
            robot_model.joints_lower[robot_model.joint_names.index(name)]
            for name in robot_model.movable_joint_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [
            robot_model.joints_upper[robot_model.joint_names.index(name)]
            for name in robot_model.movable_joint_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(
        torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi)
    )
    finite_upper = torch.where(
        torch.isfinite(upper), upper, torch.full_like(upper, torch.pi)
    )
    return corrected.clamp(finite_lower, finite_upper)


def center_allegro_grasp_between_thumb_fingers(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    qpos: torch.Tensor,
    robot_model,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Shift the wrist so the object sits between thumb and the three fingers.

    Pure Leap→Allegro retarget often leaves the (longer) thumb tip on the
    object while index/middle/ring approach from a similar direction — a tip
    poke instead of a two-sided envelope.  Moving the root by
    ``alpha * (thumb_tip - midpoint)`` places that midpoint where the thumb
    tip was, so the thumb backs off one side and the fingers advance on the
    other without changing qpos or palm orientation.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )
    alpha = float(alpha)
    if alpha == 0.0:
        return translation

    names = list(robot_model.movable_joint_names)
    qpos_dict = {name: qpos[:, index] for index, name in enumerate(names)}
    link_t, _ = robot_model.forward_kinematics(qpos_dict)
    tip_names = list(robot_model.fingertip_link_names)
    if len(tip_names) < 2:
        raise KeyError("Need thumb + finger tips to center the grasp")
    thumb_local = link_t[tip_names[0]]
    finger_local = torch.stack(
        [link_t[name] for name in tip_names[1:]], dim=1
    ).mean(dim=1)
    thumb_world = translation + torch.einsum("bij,bj->bi", rotation, thumb_local)
    finger_world = translation + torch.einsum("bij,bj->bi", rotation, finger_local)
    midpoint = 0.5 * (thumb_world + finger_world)
    delta = thumb_world - midpoint
    return translation + alpha * delta


def close_allegro_side_envelope(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    qpos: torch.Tensor,
    robot_model,
    angle_deg: Union[float, torch.Tensor] = 35.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pitch the hand so thumb and fingers share height (side envelope).

    After centering, Allegro often still has the thumb tip lower than the three
    fingers (top-down poke).  A rotation about ``(finger-thumb) × world_up``,
    fixing the tip midpoint, lowers the fingers and raises the thumb without
    the large Kabsch palm twist.  Does not change qpos.

    ``angle_deg`` may be a scalar or a ``(batch,)`` tensor of degrees.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )

    batch = int(qpos.shape[0])
    if isinstance(angle_deg, torch.Tensor):
        ang = angle_deg.to(device=translation.device, dtype=translation.dtype)
        if ang.ndim == 0:
            ang = ang.expand(batch)
        elif int(ang.shape[0]) != batch:
            raise ValueError(
                f"angle_deg batch {tuple(ang.shape)} != qpos batch {batch}"
            )
        ang = ang * (torch.pi / 180.0)
        if bool((ang.abs() < 1e-8).all().item()):
            return translation, rotation
    else:
        angle = float(angle_deg)
        if abs(angle) < 1e-6:
            return translation, rotation
        ang = translation.new_full((batch,), angle * torch.pi / 180.0)

    names = list(robot_model.movable_joint_names)
    qpos_dict = {name: qpos[:, index] for index, name in enumerate(names)}
    link_t, _ = robot_model.forward_kinematics(qpos_dict)
    tip_names = list(robot_model.fingertip_link_names)
    thumb_local = link_t[tip_names[0]]
    finger_local = torch.stack(
        [link_t[name] for name in tip_names[1:]], dim=1
    ).mean(dim=1)
    thumb_world = translation + torch.einsum("bij,bj->bi", rotation, thumb_local)
    finger_world = translation + torch.einsum("bij,bj->bi", rotation, finger_local)
    midpoint = 0.5 * (thumb_world + finger_world)
    sep = finger_world - thumb_world
    up = torch.zeros(
        (batch, 3), device=translation.device, dtype=translation.dtype
    )
    up[:, 2] = 1.0
    axis = torch.linalg.cross(sep, up)
    axis_norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    axis = axis / axis_norm

    finger_off = finger_world - midpoint
    cross = torch.linalg.cross(axis, finger_off)
    # Positive angle should move fingers downward (-Z).
    sign = torch.where(
        cross[:, 2] < 0,
        torch.ones(batch, device=translation.device, dtype=translation.dtype),
        -torch.ones(batch, device=translation.device, dtype=translation.dtype),
    )
    ang = sign * ang
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1.0 - c
    rot_delta = torch.stack(
        [
            c + x * x * C,
            x * y * C - z * s,
            x * z * C + y * s,
            y * x * C + z * s,
            c + y * y * C,
            y * z * C - x * s,
            z * x * C - y * s,
            z * y * C + x * s,
            c + z * z * C,
        ],
        dim=1,
    ).view(batch, 3, 3)

    mid_local = torch.einsum("bji,bj->bi", rotation, midpoint - translation)
    new_rotation = torch.einsum("bij,bjk->bik", rot_delta, rotation)
    new_translation = midpoint - torch.einsum(
        "bij,bj->bi", new_rotation, mid_local
    )
    return new_translation, new_rotation


def align_allegro_side_opposition_to_teacher(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    qpos: torch.Tensor,
    robot_model,
    teacher_translation: torch.Tensor,
    teacher_rotation: torch.Tensor,
    teacher_qpos: torch.Tensor,
    source_robot,
    max_yaw_deg: float = 75.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put Allegro thumb/fingers on opposite sides of the object (Leap layout).

    Constrained 1-DOF yaw about the canonical approach axis so the Allegro
    thumb→finger tip axis aligns with the Leap teacher, then translate so the
    tip midpoint matches Leap (object proxy).  No unconstrained Kabsch.
    """
    if qpos.ndim != 2 or teacher_qpos.ndim != 2:
        raise ValueError("qpos tensors must have shape (batch, joints)")
    batch = int(qpos.shape[0])
    if int(teacher_qpos.shape[0]) != batch:
        raise ValueError("teacher/target batch sizes must match")

    def _world_tips(robot, q, trans, rot):
        names = list(robot.movable_joint_names)
        qdict = {name: q[:, index] for index, name in enumerate(names)}
        for name in robot.joint_names:
            if name not in qdict:
                qdict[name] = torch.zeros(
                    batch, device=q.device, dtype=q.dtype
                )
        link_t, _ = robot.forward_kinematics(qdict)
        tip_names = list(robot.fingertip_link_names)
        return torch.stack(
            [
                torch.einsum("bij,bj->bi", rot, link_t[name]) + trans
                for name in tip_names
            ],
            dim=1,
        )

    tips_t = _world_tips(
        source_robot, teacher_qpos, teacher_translation, teacher_rotation
    )
    tips_a = _world_tips(robot_model, qpos, translation, rotation)
    finger_t = tips_t[:, 1:].mean(dim=1)
    finger_a = tips_a[:, 1:].mean(dim=1)
    mid_t = 0.5 * (tips_t[:, 0] + finger_t)
    mid_a = 0.5 * (tips_a[:, 0] + finger_a)
    sep_t = finger_t - tips_t[:, 0]
    sep_a = finger_a - tips_a[:, 0]

    # Canonical approach (shared under Leap→Allegro canonical retarget).
    leap_canonical = LEAP_CANONICAL_FRAME_ROT.to(
        device=translation.device, dtype=translation.dtype
    )
    approach = torch.einsum("bij,j->bi", teacher_rotation, leap_canonical[0])
    approach = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _perp(v: torch.Tensor) -> torch.Tensor:
        return v - (v * approach).sum(dim=-1, keepdim=True) * approach

    sa = _perp(sep_a)
    sl = _perp(sep_t)
    sa = sa / sa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sl = sl / sl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # Signed yaw about approach: R such that R @ sa ≈ sl.
    cross = torch.linalg.cross(sa, sl)
    sin_a = (approach * cross).sum(dim=-1)
    cos_a = (sa * sl).sum(dim=-1)
    delta = torch.atan2(sin_a, cos_a)
    max_rad = float(max_yaw_deg) * torch.pi / 180.0
    delta = delta.clamp(-max_rad, max_rad)

    x, y, z = approach[:, 0], approach[:, 1], approach[:, 2]
    c, s = torch.cos(delta), torch.sin(delta)
    C = 1.0 - c
    rot_delta = torch.stack(
        [
            c + x * x * C,
            x * y * C - z * s,
            x * z * C + y * s,
            y * x * C + z * s,
            c + y * y * C,
            y * z * C - x * s,
            z * x * C - y * s,
            z * y * C + x * s,
            c + z * z * C,
        ],
        dim=1,
    ).view(batch, 3, 3)

    mid_local = torch.einsum("bji,bj->bi", rotation, mid_a - translation)
    new_rotation = torch.einsum("bij,bjk->bik", rot_delta, rotation)
    new_translation = mid_a - torch.einsum(
        "bij,bj->bi", new_rotation, mid_local
    )
    # Place tip midpoint on the Leap midpoint (object between thumb/fingers).
    tips_new = _world_tips(robot_model, qpos, new_translation, new_rotation)
    mid_new = 0.5 * (tips_new[:, 0] + tips_new[:, 1:].mean(dim=1))
    new_translation = new_translation + (mid_t - mid_new)
    return new_translation, new_rotation


def align_allegro_palm_canonical_side_envelope(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    qpos: torch.Tensor,
    robot_model,
    teacher_translation: torch.Tensor,
    teacher_rotation: torch.Tensor,
    teacher_qpos: torch.Tensor,
    source_robot,
    max_angle_deg: float = 35.0,
    span_clamp: tuple[float, float] = (0.04, 0.12),
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-DOF side envelope about ``(sep × up)`` to match Leap tip height.

    Keeps the canonical grasp frame (no unconstrained tip-cloud Kabsch).
    Only pitches so Allegro thumb–finger world-z gap approaches the Leap
    teacher.  Tip-cloud midpoint is fixed; qpos is unchanged.
    """
    if qpos.ndim != 2 or teacher_qpos.ndim != 2:
        raise ValueError("qpos tensors must have shape (batch, joints)")
    batch = int(qpos.shape[0])
    if int(teacher_qpos.shape[0]) != batch:
        raise ValueError("teacher/target batch sizes must match")

    def _world_tips(robot, q, trans, rot):
        names = list(robot.movable_joint_names)
        qdict = {name: q[:, index] for index, name in enumerate(names)}
        for name in robot.joint_names:
            if name not in qdict:
                qdict[name] = torch.zeros(
                    batch, device=q.device, dtype=q.dtype
                )
        link_t, _ = robot.forward_kinematics(qdict)
        tip_names = list(robot.fingertip_link_names)
        return torch.stack(
            [
                torch.einsum("bij,bj->bi", rot, link_t[name]) + trans
                for name in tip_names
            ],
            dim=1,
        )

    tips_t = _world_tips(
        source_robot, teacher_qpos, teacher_translation, teacher_rotation
    )
    tips_a = _world_tips(robot_model, qpos, translation, rotation)
    finger_t = tips_t[:, 1:].mean(dim=1)
    finger_a = tips_a[:, 1:].mean(dim=1)
    dz_t = tips_t[:, 0, 2] - finger_t[:, 2]
    dz_a = tips_a[:, 0, 2] - finger_a[:, 2]
    sep_xy = (finger_a - tips_a[:, 0])[:, :2].norm(dim=-1)
    lo, hi = float(span_clamp[0]), float(span_clamp[1])
    span = sep_xy.clamp(min=lo, max=hi)
    # Thumb lower than Leap (dz_a < dz_t) → positive pitch.
    angle_rad = torch.atan2(dz_t - dz_a, span)
    max_rad = float(max_angle_deg) * torch.pi / 180.0
    angle_deg = (angle_rad.clamp(0.0, max_rad) * (180.0 / torch.pi))
    return close_allegro_side_envelope(
        translation, rotation, qpos, robot_model, angle_deg=angle_deg
    )


def apply_allegro_finger_curl_prior(
    qpos: torch.Tensor,
    robot_model,
    flex_scale: float = 0.72,
    flex_bias: float = 0.0,
    joints: Sequence[int] = (1, 2, 3),
    j3_scale: float = 0.70,
    j3_bias: float = 0.0,
    j3_min: float = 0.0,
    open_hand: bool = True,
) -> torch.Tensor:
    """Adjust Allegro index/middle/ring flexion after Leap→Allegro retarget.

    With ``open_hand=True`` (default), scales flexion down so the three fingers
    stay more open and the thumb is less likely to stab the object tip-first.
    Set ``open_hand=False`` to recover the older close-further boost behavior.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )
    name_to_idx = {
        name: idx for idx, name in enumerate(robot_model.movable_joint_names)
    }
    corrected = qpos.clone()
    for finger in ("index", "middle", "ring"):
        for joint_idx in joints:
            name = f"{finger}_joint_{joint_idx}"
            if name not in name_to_idx:
                raise KeyError(f"Missing Allegro joint {name!r}")
            col = name_to_idx[name]
            urdf_i = robot_model.joint_names.index(name)
            lo = float(robot_model.joints_lower[urdf_i])
            hi = float(robot_model.joints_upper[urdf_i])
            if int(joint_idx) == 3:
                target = corrected[:, col] * float(j3_scale) + float(j3_bias)
                if float(j3_min) > 0.0 and not open_hand:
                    target = torch.maximum(
                        target,
                        torch.full_like(target, float(j3_min)),
                    )
            else:
                target = corrected[:, col] * float(flex_scale) + float(flex_bias)
            if open_hand:
                # Open: never close further than the scaled target.
                corrected[:, col] = torch.minimum(corrected[:, col], target).clamp(
                    lo, hi
                )
            else:
                corrected[:, col] = torch.maximum(corrected[:, col], target).clamp(
                    lo, hi
                )
    return corrected


def fit_allegro_thumb_to_teacher_tip(
    qpos: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    source_qpos: torch.Tensor,
    source_translation: torch.Tensor,
    source_rotation: torch.Tensor,
    source_robot,
    target_robot,
    steps: int = 120,
    lr: float = 0.04,
    pose_reg: float = 0.0001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit only Allegro thumb joints to the Leap teacher thumb fingertip.

    Semantic mapping and the joint-synergy prior do not account for different
    thumb link origins in the ROS-V5 URDF. This static correction keeps the
    Allegro wrist pose and all three primary fingers fixed, and moves only
    ``thumb_joint_0..3`` toward the world-space Leap thumb tip. It does not
    use object geometry, contact IK, or physics.
    """
    if qpos.ndim != 2 or source_qpos.ndim != 2:
        raise ValueError("Source and target qpos must have shape (batch, joints)")
    if qpos.shape[0] != source_qpos.shape[0]:
        raise ValueError("Source and target qpos batch sizes must match")
    if translation.shape != source_translation.shape:
        raise ValueError("Source and target translations must have the same shape")
    if rotation.shape != source_rotation.shape:
        raise ValueError("Source and target rotations must have the same shape")
    if int(steps) <= 0:
        raise ValueError(f"Thumb fit steps must be positive, got {steps}")

    source_names = list(source_robot.movable_joint_names)
    target_names = list(target_robot.movable_joint_names)
    if source_qpos.shape[1] != len(source_names):
        raise ValueError("Source qpos width does not match source robot")
    if qpos.shape[1] != len(target_names):
        raise ValueError("Target qpos width does not match target robot")

    source_tip_name = "thumb_fingertip"
    target_tip_name = "thumb_biotac_tip"
    if source_tip_name not in source_robot.link_names:
        raise KeyError(f"Source robot lacks {source_tip_name!r}")
    if target_tip_name not in target_robot.link_names:
        raise KeyError(f"Target robot lacks {target_tip_name!r}")

    source_qpos_dict = {
        name: source_qpos[:, index]
        for index, name in enumerate(source_names)
    }
    with torch.no_grad():
        source_translations, _ = source_robot.forward_kinematics(
            source_qpos_dict
        )
        teacher_tip = (
            torch.einsum(
                "bij,bj->bi",
                source_rotation,
                source_translations[source_tip_name],
            )
            + source_translation
        )

    thumb_names = tuple(f"thumb_joint_{index}" for index in range(4))
    thumb_indices = [target_names.index(name) for name in thumb_names]
    nonthumb_mask = torch.ones(
        qpos.shape[1], dtype=torch.bool, device=qpos.device
    )
    nonthumb_mask[thumb_indices] = False
    base_qpos = qpos.detach()
    fitted_qpos = base_qpos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([fitted_qpos], lr=float(lr))

    lower = torch.tensor(
        [
            target_robot.joints_lower[target_robot.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [
            target_robot.joints_upper[target_robot.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(
        torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi)
    )
    finite_upper = torch.where(
        torch.isfinite(upper), upper, torch.full_like(upper, torch.pi)
    )

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        target_qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        target_translations, _ = target_robot.forward_kinematics(
            target_qpos_dict
        )
        target_tip = (
            torch.einsum(
                "bij,bj->bi",
                rotation,
                target_translations[target_tip_name],
            )
            + translation
        )
        tip_error = target_tip - teacher_tip
        thumb_delta = fitted_qpos[:, thumb_indices] - base_qpos[:, thumb_indices]
        loss = (
            tip_error.square().sum(dim=-1).mean()
            + float(pose_reg) * thumb_delta.square().mean()
        )
        loss.backward()
        if fitted_qpos.grad is not None:
            fitted_qpos.grad[:, nonthumb_mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            fitted_qpos.clamp_(finite_lower, finite_upper)
            fitted_qpos[:, nonthumb_mask] = base_qpos[:, nonthumb_mask]

    with torch.no_grad():
        target_qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        target_translations, _ = target_robot.forward_kinematics(
            target_qpos_dict
        )
        target_tip = (
            torch.einsum(
                "bij,bj->bi",
                rotation,
                target_translations[target_tip_name],
            )
            + translation
        )
        tip_mse = (target_tip - teacher_tip).square().mean(dim=-1)
    return fitted_qpos.detach(), tip_mse.detach()


def fit_allegro_thumb_finger_opposition(
    qpos: torch.Tensor,
    robot_model,
    steps: int = 120,
    lr: float = 0.05,
    approach_frac: float = 0.70,
    normal_weight: float = 0.30,
    pose_reg: float = 0.0005,
    tip_normal_local: Sequence[float] = (0.0, -1.0, 0.0),
    normal_target: str = "index",
    open_hand_soft_caps: bool = False,
    thumb_q0_min: float = 1.15,
    thumb_q1_max: float = 0.28,
    thumb_q2_max: float = 0.40,
    thumb_q3_max: float = 0.12,
    flex_reg: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate/close only the thumb so its pad faces the three primary fingers.

    Leap→Allegro semantic mapping often leaves Allegro ``thumb_joint_1/3`` near
    open-hand values, so the thumb pad faces away from index/middle/ring.  This
    static correction (no object contacts) keeps wrist and non-thumb joints
    fixed, pulls the thumb tip part-way toward a finger-side goal, and aligns
    the biotac local normal with a chosen thumb→finger direction.

    ``normal_target`` (also used for tip goal):
      - ``index``  — thumb→index (legacy; default, matches older exports)
      - ``middle`` — thumb→middle
      - ``im``     — thumb→index–middle midpoint
      - ``three``  — thumb→mean(index, middle, ring)  (Leap-like)

    ``open_hand_soft_caps``: keep thumb flexion in a Leap-like open envelope
    while optimizing pad direction (avoids tip-poke claw from unconstrained
    opposition). Soft caps are applied each step; ``flex_reg`` gently
    penalizes joints above the open-hand ceilings.

    Prefer running this *without* ``fit_allegro_thumb_to_teacher_tip`` first:
    Leap tip IK often lowers ``thumb_joint_0`` and yields a table-poking stick
    instead of an opposed cradle.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )
    if int(steps) <= 0:
        raise ValueError(f"Opposition steps must be positive, got {steps}")
    target_mode = str(normal_target).lower().strip()
    if target_mode not in ("index", "middle", "im", "three"):
        raise ValueError(
            f"normal_target must be index|middle|im|three, got {normal_target!r}"
        )

    target_names = list(robot_model.movable_joint_names)
    thumb_names = tuple(f"thumb_joint_{index}" for index in range(4))
    thumb_indices = [target_names.index(name) for name in thumb_names]
    nonthumb_mask = torch.ones(
        qpos.shape[1], dtype=torch.bool, device=qpos.device
    )
    nonthumb_mask[thumb_indices] = False
    tip_names = (
        "thumb_biotac_tip",
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
    )
    for name in tip_names:
        if name not in robot_model.link_names:
            raise KeyError(f"Robot lacks fingertip link {name!r}")

    local_normal = torch.tensor(
        list(tip_normal_local),
        device=qpos.device,
        dtype=qpos.dtype,
    )

    def _finger_anchor(link_t) -> torch.Tensor:
        index_tip = link_t["index_biotac_tip"]
        middle_tip = link_t["middle_biotac_tip"]
        ring_tip = link_t["ring_biotac_tip"]
        if target_mode == "index":
            return index_tip
        if target_mode == "middle":
            return middle_tip
        if target_mode == "im":
            return 0.5 * (index_tip + middle_tip)
        return (index_tip + middle_tip + ring_tip) / 3.0

    def _apply_open_hand_thumb_caps(q: torch.Tensor) -> None:
        """In-place soft open-hand envelope on thumb joints only."""
        # j0 floor (opposition), j1/j2/j3 ceilings (avoid claw / tip poke)
        caps = {
            "thumb_joint_0": (float(thumb_q0_min), None),
            "thumb_joint_1": (None, float(thumb_q1_max)),
            "thumb_joint_2": (None, float(thumb_q2_max)),
            "thumb_joint_3": (None, float(thumb_q3_max)),
        }
        for name, (lo_soft, hi_soft) in caps.items():
            col = target_names.index(name)
            urdf_i = robot_model.joint_names.index(name)
            lo = float(robot_model.joints_lower[urdf_i])
            hi = float(robot_model.joints_upper[urdf_i])
            if lo_soft is not None:
                q[:, col].clamp_(min=min(max(lo_soft, lo), hi))
            if hi_soft is not None:
                q[:, col].clamp_(max=min(max(hi_soft, lo), hi))

    base_qpos = qpos.detach()
    with torch.no_grad():
        base_dict = {
            name: base_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        base_t, _ = robot_model.forward_kinematics(base_dict)
        thumb0 = base_t["thumb_biotac_tip"]
        anchor0 = _finger_anchor(base_t)
        goal = thumb0 + float(approach_frac) * (anchor0 - thumb0)

    fitted_qpos = base_qpos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([fitted_qpos], lr=float(lr))
    lower = torch.tensor(
        [
            robot_model.joints_lower[robot_model.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [
            robot_model.joints_upper[robot_model.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(
        torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi)
    )
    finite_upper = torch.where(
        torch.isfinite(upper), upper, torch.full_like(upper, torch.pi)
    )

    # Soft ceilings used by flex_reg (j1/j2/j3 only).
    flex_ceiling = torch.tensor(
        [
            float(thumb_q1_max),
            float(thumb_q2_max),
            float(thumb_q3_max),
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    flex_cols = thumb_indices[1:4]

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        link_t, link_r = robot_model.forward_kinematics(qpos_dict)
        tip = link_t["thumb_biotac_tip"]
        anchor = _finger_anchor(link_t)
        normal = torch.einsum("bij,j->bi", link_r["thumb_biotac_tip"], local_normal)
        to_anchor = anchor - tip
        to_anchor = to_anchor / to_anchor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        align = (normal * to_anchor).sum(dim=-1)
        thumb_delta = fitted_qpos[:, thumb_indices] - base_qpos[:, thumb_indices]
        loss = (
            (tip - goal).square().sum(dim=-1).mean()
            + float(normal_weight) * (1.0 - align).square().mean()
            + float(pose_reg) * thumb_delta.square().mean()
        )
        if bool(open_hand_soft_caps) and float(flex_reg) > 0.0:
            flex = fitted_qpos[:, flex_cols]
            over = torch.relu(flex - flex_ceiling.view(1, -1))
            loss = loss + float(flex_reg) * over.square().mean()
        loss.backward()
        if fitted_qpos.grad is not None:
            fitted_qpos.grad[:, nonthumb_mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            fitted_qpos.clamp_(finite_lower, finite_upper)
            fitted_qpos[:, nonthumb_mask] = base_qpos[:, nonthumb_mask]
            if bool(open_hand_soft_caps):
                _apply_open_hand_thumb_caps(fitted_qpos)

    with torch.no_grad():
        if bool(open_hand_soft_caps):
            _apply_open_hand_thumb_caps(fitted_qpos)
        qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        link_t, link_r = robot_model.forward_kinematics(qpos_dict)
        tip = link_t["thumb_biotac_tip"]
        anchor = _finger_anchor(link_t)
        normal = torch.einsum("bij,j->bi", link_r["thumb_biotac_tip"], local_normal)
        to_anchor = anchor - tip
        to_anchor = to_anchor / to_anchor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        align = (normal * to_anchor).sum(dim=-1)
    return fitted_qpos.detach(), align.detach()


def equalize_allegro_thumb_approach_lead(
    qpos: torch.Tensor,
    robot_model,
    target_lead: float = 0.01,
    steps: int = 80,
    lr: float = 0.05,
    pose_reg: float = 0.001,
    open_hand_soft_caps: bool = True,
    thumb_q0_min: float = 1.15,
    thumb_q1_max: float = 0.28,
    thumb_q2_max: float = 0.40,
    thumb_q3_max: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull Allegro thumb tip back along canonical approach to match fingers.

    Leap-like open coarse often leaves the thumb protruding along the evaluator
    approach axis (``R @ Cᵀ @ e_x``), so descent becomes thumb-first.  This
    keeps non-thumb joints fixed and only adjusts thumb DOFs so

        (thumb_tip - finger_mid) · approach_local  ≈  ``target_lead``

    in the robot local frame (``approach_local = Cᵀ e_x``). Soft open-hand
    caps are optional so the aperture is preserved while lead is reduced.
    Returns ``(qpos, lead)`` with lead in metres.
    """
    if qpos.ndim != 2:
        raise ValueError(
            f"Allegro qpos must have shape (batch, joints), got {tuple(qpos.shape)}"
        )
    target_names = list(robot_model.movable_joint_names)
    thumb_names = tuple(f"thumb_joint_{index}" for index in range(4))
    thumb_indices = [target_names.index(name) for name in thumb_names]
    nonthumb_mask = torch.ones(
        qpos.shape[1], dtype=torch.bool, device=qpos.device
    )
    nonthumb_mask[thumb_indices] = False
    for name in (
        "thumb_biotac_tip",
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
    ):
        if name not in robot_model.link_names:
            raise KeyError(f"Robot lacks fingertip link {name!r}")

    approach_local = ALLEGRO_CANONICAL_FRAME_ROT.to(
        device=qpos.device, dtype=qpos.dtype
    ).transpose(0, 1)[:, 0]
    approach_local = approach_local / approach_local.norm().clamp_min(1e-6)

    def _lead(link_t) -> torch.Tensor:
        thumb = link_t["thumb_biotac_tip"]
        finger_mid = (
            link_t["index_biotac_tip"]
            + link_t["middle_biotac_tip"]
            + link_t["ring_biotac_tip"]
        ) / 3.0
        return ((thumb - finger_mid) * approach_local.view(1, 3)).sum(dim=-1)

    def _apply_open_hand_thumb_caps(q: torch.Tensor) -> None:
        caps = {
            "thumb_joint_0": (float(thumb_q0_min), None),
            "thumb_joint_1": (None, float(thumb_q1_max)),
            "thumb_joint_2": (None, float(thumb_q2_max)),
            "thumb_joint_3": (None, float(thumb_q3_max)),
        }
        for name, (lo_soft, hi_soft) in caps.items():
            col = target_names.index(name)
            urdf_i = robot_model.joint_names.index(name)
            lo = float(robot_model.joints_lower[urdf_i])
            hi = float(robot_model.joints_upper[urdf_i])
            if lo_soft is not None:
                q[:, col].clamp_(min=min(max(lo_soft, lo), hi))
            if hi_soft is not None:
                q[:, col].clamp_(max=min(max(hi_soft, lo), hi))

    base_qpos = qpos.detach()
    fitted_qpos = base_qpos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([fitted_qpos], lr=float(lr))
    lower = torch.tensor(
        [
            robot_model.joints_lower[robot_model.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [
            robot_model.joints_upper[robot_model.joint_names.index(name)]
            for name in target_names
        ],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(
        torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi)
    )
    finite_upper = torch.where(
        torch.isfinite(upper), upper, torch.full_like(upper, torch.pi)
    )
    target = float(target_lead)

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        link_t, _ = robot_model.forward_kinematics(qpos_dict)
        lead = _lead(link_t)
        # Penalize only overshoot beyond target (keep fingers-first / flat).
        over = torch.relu(lead - target)
        thumb_delta = fitted_qpos[:, thumb_indices] - base_qpos[:, thumb_indices]
        loss = over.square().mean() + float(pose_reg) * thumb_delta.square().mean()
        loss.backward()
        if fitted_qpos.grad is not None:
            fitted_qpos.grad[:, nonthumb_mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            fitted_qpos.clamp_(finite_lower, finite_upper)
            fitted_qpos[:, nonthumb_mask] = base_qpos[:, nonthumb_mask]
            if bool(open_hand_soft_caps):
                _apply_open_hand_thumb_caps(fitted_qpos)

    with torch.no_grad():
        if bool(open_hand_soft_caps):
            _apply_open_hand_thumb_caps(fitted_qpos)
        qpos_dict = {
            name: fitted_qpos[:, index]
            for index, name in enumerate(target_names)
        }
        link_t, _ = robot_model.forward_kinematics(qpos_dict)
        lead = _lead(link_t)
    return fitted_qpos.detach(), lead.detach()


def build_allegro_canonical_qpos(
    robot_model,
    batch_size: int,
    device=None,
    dtype=torch.float32,
    width_mapper_meta_path: str | None = None,
) -> torch.Tensor:
    """Build Allegro qpos from width_mapper canonical_pose (open-hand preset).

    Used by cross-embodiment ``optimize`` mode: start from a neutral Allegro
    posture instead of retargeting Leap joint angles, then fit to contacts.
    """
    if width_mapper_meta_path is None:
        width_mapper_meta_path = "robot_models/meta/allegro_hand/width_mapper_meta.yaml"
    if not os.path.isfile(width_mapper_meta_path):
        raise FileNotFoundError(
            f"Allegro width_mapper meta not found: {width_mapper_meta_path}"
        )
    with open(width_mapper_meta_path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle) or {}
    canonical_qpos = meta.get("canonical_pose", {}).get("qpos", {})
    if not canonical_qpos:
        raise KeyError(
            f"canonical_pose.qpos missing in width_mapper meta: {width_mapper_meta_path}"
        )

    joint_names = list(robot_model.movable_joint_names)
    values = []
    missing = []
    for name in joint_names:
        if name not in canonical_qpos:
            missing.append(name)
            values.append(0.0)
        else:
            values.append(float(canonical_qpos[name]))
    if missing:
        raise KeyError(
            "canonical_pose.qpos missing Allegro joints: "
            f"{missing} (meta={width_mapper_meta_path})"
        )

    if device is None:
        device = torch.device("cpu")
    row = torch.tensor(values, device=device, dtype=dtype)
    return row.unsqueeze(0).expand(int(batch_size), -1).contiguous()


def retarget_leap_pose_to_allegro(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    translation_mode: str = "palm",
    rotation_mode: str = "canonical",
    target_robot=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a Leap root pose to an Allegro free-root pose.

    The predictor pose is expressed at the Leap ``hand_base_link`` root.  The
    Allegro free URDF is evaluated at its ``allegro_mount`` root.  ``palm``
    subtracts the target Allegro ``allegro_mount -> palm_link`` fixed-joint
    translation when ``target_robot`` is supplied.  Without a target model,
    the legacy simplified-Allegro constants are used as a fallback.  ``root``
    preserves the Leap root origin.

    ``canonical`` preserves the evaluator grasp frame.  ``palm`` instead
    aligns Allegro ``palm_link`` orientation directly with the Leap root
    orientation, which is useful as a visualization/diagnostic check for palm
    direction.

    The canonical orientation right-multiplication is derived from

        R_alg C_alg.T = R_leap C_leap.T,

        hence ``R_alg = R_leap C_leap.T C_alg``.
    """
    mode = str(translation_mode).lower()
    if mode not in ("root", "palm"):
        raise ValueError(
            f"Unsupported Allegro pose translation mode {translation_mode!r}; "
            "expected 'root' or 'palm'."
        )
    rot_mode = str(rotation_mode).lower()
    if rot_mode not in ("canonical", "palm"):
        raise ValueError(
            f"Unsupported Allegro pose rotation mode {rotation_mode!r}; "
            "expected 'canonical' or 'palm'."
        )
    device = translation.device
    dtype = translation.dtype
    mount_to_palm_rotation = ALLEGRO_MOUNT_TO_PALM_ROT.to(
        device=device, dtype=dtype
    )
    mount_to_palm_translation = ALLEGRO_MOUNT_TO_PALM_TRANSLATION.to(
        device=device, dtype=dtype
    )
    if target_robot is not None:
        # The simplified and ROS-V5 assets share joint names but not the
        # fixed mount->palm frame. Read the target URDF through RobotModel.
        batch_size = translation.shape[0]
        zero_qpos = {
            name: torch.zeros(batch_size, device=device, dtype=dtype)
            for name in target_robot.movable_joint_names
        }
        fixed_t, fixed_r = target_robot.forward_kinematics(zero_qpos)
        root_name = target_robot._root_link_name
        if root_name not in fixed_t or "palm_link" not in fixed_t:
            raise KeyError(
                "target_robot must expose its root and palm_link for "
                "Allegro pose retargeting"
            )
        mount_to_palm_translation = (
            fixed_t["palm_link"][0] - fixed_t[root_name][0]
        )
        mount_to_palm_rotation = fixed_r["palm_link"][0]
    if rot_mode == "canonical":
        leap_canonical = LEAP_CANONICAL_FRAME_ROT.to(device=device, dtype=dtype)
        allegro_canonical = ALLEGRO_CANONICAL_FRAME_ROT.to(device=device, dtype=dtype)
        root_map = leap_canonical.transpose(0, 1) @ allegro_canonical
    else:
        root_map = mount_to_palm_rotation.transpose(0, 1)
    mount_rotation = rotation @ root_map
    mount_translation = translation
    if mode == "palm":
        mount_translation = mount_translation - torch.matmul(
            mount_rotation, mount_to_palm_translation.reshape(3, 1)
        ).squeeze(-1)
    return mount_translation, mount_rotation


def align_allegro_wrist_to_teacher_fingertips(
    source_translation: torch.Tensor,
    source_rotation: torch.Tensor,
    source_qpos: torch.Tensor,
    target_translation: torch.Tensor,
    target_rotation: torch.Tensor,
    target_qpos: torch.Tensor,
    source_robot,
    target_robot,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate Allegro so its fingertip centroid matches the Leap teacher.

    The two robots use different root origins and have different finger
    lengths.  Mapping only the root pose therefore preserves orientation but
    can leave the Allegro fingertips far from the teacher/object.  This
    correction changes translation only; rotations and both qpos tensors are
    untouched.  It is intended for the Allegro coarse diagnostic path.
    """
    if source_qpos.ndim != 2 or target_qpos.ndim != 2:
        raise ValueError("Source and target qpos must have shape (batch, joints)")
    if source_qpos.shape[0] != target_qpos.shape[0]:
        raise ValueError("Source and target qpos batch sizes must match")
    if source_translation.shape != target_translation.shape:
        raise ValueError("Source and target translations must have the same shape")

    source_qpos_dict = {
        name: source_qpos[:, index]
        for index, name in enumerate(source_robot.movable_joint_names)
    }
    target_qpos_dict = {
        name: target_qpos[:, index]
        for index, name in enumerate(target_robot.movable_joint_names)
    }
    source_link_translations, _ = source_robot.forward_kinematics(source_qpos_dict)
    target_link_translations, _ = target_robot.forward_kinematics(target_qpos_dict)

    source_tip_names = list(source_robot.fingertip_link_names)
    target_tip_names = list(target_robot.fingertip_link_names)
    if not source_tip_names or not target_tip_names:
        raise ValueError("Both robot models must define fingertip_link_names")

    source_centroid = torch.stack(
        [source_link_translations[name] for name in source_tip_names], dim=1
    ).mean(dim=1)
    target_centroid = torch.stack(
        [target_link_translations[name] for name in target_tip_names], dim=1
    ).mean(dim=1)
    source_centroid_world = torch.einsum(
        "bij,bj->bi", source_rotation, source_centroid
    )
    target_centroid_world = torch.einsum(
        "bij,bj->bi", target_rotation, target_centroid
    )
    correction = source_centroid_world - target_centroid_world
    return target_translation + float(alpha) * correction, correction


def align_allegro_pose_to_teacher_fingertips(
    source_translation: torch.Tensor,
    source_rotation: torch.Tensor,
    source_qpos: torch.Tensor,
    target_translation: torch.Tensor,
    target_rotation: torch.Tensor,
    target_qpos: torch.Tensor,
    source_robot,
    target_robot,
    translation_mode: str = "preserve_target",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigidly align the Allegro fingertip layout to the Leap teacher.

    Estimates a per-grasp rotation from corresponding fingertip clouds so the
    Allegro thumb/finger relative layout matches Leap (fixes the long-thumb
    reach gap).  Translation modes:

    - ``preserve_target`` (default): keep Allegro's pre-align tip-cloud
      centroid in world.  Balances thumb vs fingers without lifting the hand
      off the object (matching Leap tip height often floats Allegro ~3–5cm).
    - ``source``: place Allegro tip centroid at the Leap tip centroid (legacy).

    Allegro-only; does not change either robot's qpos.
    """
    mode = str(translation_mode).lower()
    if mode not in ("preserve_target", "source"):
        raise ValueError(
            f"Unsupported translation_mode {translation_mode!r}; "
            "expected 'preserve_target' or 'source'."
        )
    source_qpos_dict = {
        name: source_qpos[:, index]
        for index, name in enumerate(source_robot.movable_joint_names)
    }
    target_qpos_dict = {
        name: target_qpos[:, index]
        for index, name in enumerate(target_robot.movable_joint_names)
    }
    source_link_translations, _ = source_robot.forward_kinematics(source_qpos_dict)
    target_link_translations, _ = target_robot.forward_kinematics(target_qpos_dict)
    source_tip_names = list(source_robot.fingertip_link_names)
    target_tip_names = list(target_robot.fingertip_link_names)
    if len(source_tip_names) != len(target_tip_names) or len(source_tip_names) < 3:
        raise ValueError("Source and target fingertip definitions are incompatible")

    source_points = torch.stack(
        [source_link_translations[name] for name in source_tip_names], dim=1
    )
    target_points = torch.stack(
        [target_link_translations[name] for name in target_tip_names], dim=1
    )
    source_center = source_points.mean(dim=1, keepdim=True)
    target_center = target_points.mean(dim=1, keepdim=True)
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # Solve Q * target_local ~= source_local, with row-vector SVD then
    # transpose back to the column-vector convention used by RobotModel.
    covariance = target_centered.transpose(1, 2) @ source_centered
    u, _, vh = torch.linalg.svd(covariance)
    q_target_to_source = vh.transpose(1, 2) @ u.transpose(1, 2)
    det = torch.linalg.det(q_target_to_source)
    if bool((det < 0.0).any()):
        vh_fixed = vh.clone()
        vh_fixed[det < 0.0, -1, :] *= -1.0
        q_target_to_source = vh_fixed.transpose(1, 2) @ u.transpose(1, 2)

    aligned_rotation = torch.einsum(
        "bij,bjk->bik", source_rotation, q_target_to_source
    )
    target_center_local = target_center[:, 0]
    if mode == "source":
        source_center_world = torch.einsum(
            "bij,bj->bi", source_rotation, source_center[:, 0]
        )
        target_center_world = torch.einsum(
            "bij,bj->bi", aligned_rotation, target_center_local
        )
        aligned_translation = (
            source_translation + source_center_world - target_center_world
        )
    else:
        # Keep pre-align Allegro tip centroid depth/position.
        old_target_center_world = target_translation + torch.einsum(
            "bij,bj->bi", target_rotation, target_center_local
        )
        aligned_translation = old_target_center_world - torch.einsum(
            "bij,bj->bi", aligned_rotation, target_center_local
        )
    return aligned_translation, aligned_rotation


def warm_start_allegro_wrist_toward_contacts(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    target_contacts: torch.Tensor,
    alpha: float = 0.35,
    max_shift: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift Allegro wrist toward contact centroid before IK (optimize mode)."""
    centroid = target_contacts.mean(dim=1)
    delta = torch.clamp(centroid - translation, -float(max_shift), float(max_shift))
    return translation + float(alpha) * delta, rotation


def _parse_allegro_finger_weights(
    finger_weights: Optional[Union[Sequence[float], torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    default = (1.0, 1.0, 1.0, 1.0)
    values = tuple(float(x) for x in (finger_weights or default))
    if len(values) != 4:
        raise ValueError(f"Expected 4 finger weights, got {len(values)}")
    weights = torch.tensor(values, device=device, dtype=dtype)
    return weights * (4.0 / weights.sum().clamp_min(1e-6))


def _allegro_joint_shape_penalties(
    qpos: torch.Tensor,
    robot_model,
    canonical: Optional[Dict[str, float]] = None,
    coordination_q2_ratio: float = 0.75,
    coordination_q3_ratio: float = 0.50,
    abduction_soft_limit: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft Allegro-only shape penalties.

    The contact objective alone can use abduction and the proximal joint to
    compensate for a Leap-to-Allegro geometry mismatch.  These penalties are
    deliberately soft:

    - flexion joints stay non-negative around the Allegro canonical pose;
    - PIP/DIP flexion follows a mild finger synergy;
    - MCP abduction is only penalized after it exceeds a small canonical-frame
      envelope, so legitimate finger spread is still possible.

    Both outputs are per-batch scalars and are dimensionless after the
    normalization constants below.  The caller controls their weights.
    """
    if canonical is None:
        canonical = _canonical_qpos(robot_model)
    device = qpos.device
    dtype = qpos.dtype
    coordination = torch.zeros(qpos.shape[0], device=device, dtype=dtype)
    canonical_abduction = torch.zeros_like(coordination)
    name_to_idx = {
        name: idx for idx, name in enumerate(robot_model.movable_joint_names)
    }

    for finger, names in ALLEGRO_FINGER_GROUPS.items():
        if not all(name in name_to_idx for name in names):
            continue
        q0, q1, q2, q3 = (qpos[:, name_to_idx[name]] for name in names)
        c0 = float(canonical.get(names[0], 0.0))
        c1 = float(canonical.get(names[1], 0.0))
        c2 = float(canonical.get(names[2], 0.0))
        c3 = float(canonical.get(names[3], 0.0))

        # Work in motion relative to the open/canonical pose.  This keeps the
        # thumb's canonical 0.8-rad abduction from being treated as flexion.
        flex = torch.stack(
            [
                q1 - c1,
                q2 - c2,
                q3 - c3,
            ],
            dim=-1,
        )
        flex_nonnegative = torch.relu(-flex / 0.25).square().mean(dim=-1)
        pip_synergy = (
            (flex[:, 1] - float(coordination_q2_ratio) * flex[:, 0]) / 0.80
        ).square()
        dip_synergy = (
            (flex[:, 2] - float(coordination_q3_ratio) * flex[:, 1]) / 0.80
        ).square()
        # A distal joint should not bend more than its proximal joint by a
        # large amount; use a hinge so ordinary partial grasps are unaffected.
        non_monotonic = torch.relu((flex[:, 2] - flex[:, 1]) / 0.25).square()
        coordination = coordination + (
            flex_nonnegative + pip_synergy + dip_synergy + non_monotonic
        )

        abd_excess = torch.relu(
            (torch.abs(q0 - c0) - float(abduction_soft_limit)) / 0.25
        )
        canonical_abduction = canonical_abduction + abd_excess.square()

    num_fingers = float(len(ALLEGRO_FINGER_GROUPS))
    return coordination / num_fingers, canonical_abduction / num_fingers


def fit_allegro_qpos_to_contacts(
    qpos: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    target_contacts: torch.Tensor,
    robot_model,
    hand_model=None,
    object_pc: torch.Tensor | None = None,
    fingertip_local_points: torch.Tensor | None = None,
    steps: int = 60,
    lr: float = 0.03,
    pose_reg: float = 0.01,
    max_trans_delta: float = 0.05,
    max_rot_delta: float = 0.60,
    lock_abduction: bool = False,
    finger_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    abduction_reg_weight: float = 2.0,
    coordination_reg_weight: float = 0.0,
    canonical_abduction_reg_weight: float = 0.0,
    coordination_q2_ratio: float = 0.75,
    coordination_q3_ratio: float = 0.50,
    coordination_abduction_soft_limit: float = 0.25,
):
    """Fit Allegro pose and joint angles to four camera-frame contacts.

    Contact order is thumb, index, middle, ring, matching both the Leap
    contact network convention and the Allegro metadata. The optimization
    starts from the current Allegro initialization (canonical qpos in
    ``optimize`` mode, or Leap retarget in ``retarget`` mode) and jointly
    adjusts the Allegro translation, rotation, and 16 revolute joints.

    When ``hand_model`` is None (default for Allegro sim transfer), fingertip
    targets are the URDF ``*_biotac_tip`` links used by Isaac collision and
    width mapping. ``fingertip_local_points`` optionally replaces each link
    origin with one fixed point in that link's collision-mesh frame, ordered
    thumb/index/middle/ring. This is useful for a diagnostic fit because the
    link origin is not necessarily the point on the finger mesh facing the
    object. Passing a hand surface provider optimizes mesh representative
    points instead, which can look good in ``contact_mse`` but leave biotac
    tips several centimeters from the predicted contacts.
    """
    if target_contacts.ndim != 3 or target_contacts.shape[-1] != 3:
        raise ValueError(
            "target_contacts must have shape (batch, 4, 3), got "
            f"{tuple(target_contacts.shape)}"
        )
    if target_contacts.shape[1] != 4:
        raise ValueError(
            f"Expected 4 target contacts, got {target_contacts.shape[1]}"
        )
    if qpos.ndim != 2 or qpos.shape[1] != len(robot_model.movable_joint_names):
        raise ValueError(
            "Allegro qpos must have shape (batch, num_movable_joints), got "
            f"{tuple(qpos.shape)}; expected second dimension "
            f"{len(robot_model.movable_joint_names)}"
        )

    if hand_model is None:
        fingertip_links = [
            "thumb_biotac_tip",
            "index_biotac_tip",
            "middle_biotac_tip",
            "ring_biotac_tip",
        ]
        missing = [name for name in fingertip_links if name not in robot_model.link_names]
        if missing:
            raise KeyError(f"Allegro fingertip links missing from URDF: {missing}")
        if fingertip_local_points is not None:
            fingertip_local_points = torch.as_tensor(
                fingertip_local_points,
                device=qpos.device,
                dtype=qpos.dtype,
            )
            if fingertip_local_points.shape != (4, 3):
                raise ValueError(
                    "fingertip_local_points must have shape (4, 3), got "
                    f"{tuple(fingertip_local_points.shape)}"
                )

    base_translation = translation.detach()
    base_rotation = rotation.detach()
    contact_weights = _parse_allegro_finger_weights(
        finger_weights, qpos.device, qpos.dtype
    )
    abd_indices = [
        idx
        for idx, name in enumerate(robot_model.movable_joint_names)
        if name.endswith("_joint_0")
    ]

    def fingertip_representatives(
        current_translation: torch.Tensor,
        current_rotation: torch.Tensor,
        current_qpos: torch.Tensor,
    ) -> torch.Tensor:
        if hand_model is not None:
            from network.contact_diffusion_pointnext_offset import (
                _forward_finger_representatives,
            )

            return _forward_finger_representatives(
                hand_model,
                current_translation,
                current_rotation,
                current_qpos,
                object_pc=object_pc,
                num_fingers=4,
            )

        qpos_dict = {
            name: current_qpos[:, i]
            for i, name in enumerate(robot_model.movable_joint_names)
        }
        link_translations, link_rotations = robot_model.forward_kinematics(qpos_dict)
        local_tips = torch.stack(
            [link_translations[name] for name in fingertip_links], dim=1
        )
        if fingertip_local_points is not None:
            offsets = fingertip_local_points.unsqueeze(0).expand(
                current_qpos.shape[0], -1, -1
            )
            local_tip_rotations = torch.stack(
                [link_rotations[name] for name in fingertip_links], dim=1
            )
            local_tips = local_tips + torch.einsum(
                "bfij,bfj->bfi",
                local_tip_rotations,
                offsets,
            )
        return torch.einsum("bij,bfj->bfi", current_rotation, local_tips) + current_translation[:, None]

    lower = torch.tensor(
        [robot_model.joints_lower[robot_model.joint_names.index(name)]
         for name in robot_model.movable_joint_names],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    upper = torch.tensor(
        [robot_model.joints_upper[robot_model.joint_names.index(name)]
         for name in robot_model.movable_joint_names],
        device=qpos.device,
        dtype=qpos.dtype,
    )
    finite_lower = torch.where(torch.isfinite(lower), lower, torch.full_like(lower, -torch.pi))
    finite_upper = torch.where(torch.isfinite(upper), upper, torch.full_like(upper, torch.pi))

    q = qpos.detach().clone().requires_grad_(True)
    delta_translation = torch.zeros_like(base_translation, requires_grad=True)
    delta_rotation = torch.zeros(
        base_translation.shape[0], 3,
        device=base_translation.device,
        dtype=base_translation.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [q, delta_translation, delta_rotation], lr=float(lr)
    )
    base_q = qpos.detach()
    use_shape_prior = (
        abs(float(coordination_reg_weight)) > 0.0
        or abs(float(canonical_abduction_reg_weight)) > 0.0
    )
    canonical = _canonical_qpos(robot_model) if use_shape_prior else None
    joint_reg_weights = torch.ones(
        qpos.shape[1], device=qpos.device, dtype=qpos.dtype
    )
    for idx in abd_indices:
        joint_reg_weights[idx] = float(abduction_reg_weight)
    last_loss = None
    for _ in range(max(int(steps), 0)):
        optimizer.zero_grad(set_to_none=True)
        current_translation = base_translation + torch.tanh(delta_translation) * float(max_trans_delta)
        rotation_vector = torch.tanh(delta_rotation) * float(max_rot_delta)
        theta = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
        axis = rotation_vector / theta.clamp_min(1e-8)
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(-1, 3, 3)
        eye = torch.eye(
            3, device=base_rotation.device, dtype=base_rotation.dtype
        ).expand_as(skew)
        rot_delta = (
            eye
            + torch.sin(theta)[..., None] * skew
            + (1.0 - torch.cos(theta))[..., None] * torch.matmul(skew, skew)
        )
        current_rotation = torch.matmul(rot_delta, base_rotation)
        tips = fingertip_representatives(current_translation, current_rotation, q)
        per_finger_sq = (tips - target_contacts).square().sum(dim=-1)
        contact_loss = (per_finger_sq * contact_weights.view(1, 4)).mean()
        q_delta = q - base_q
        regularization = (
            (q_delta.square() * joint_reg_weights.view(1, -1)).mean()
            + delta_translation.square().mean()
            + delta_rotation.square().mean()
        )
        if use_shape_prior:
            coordination_penalty, canonical_abduction_penalty = (
                _allegro_joint_shape_penalties(
                    q,
                    robot_model,
                    canonical=canonical,
                    coordination_q2_ratio=float(coordination_q2_ratio),
                    coordination_q3_ratio=float(coordination_q3_ratio),
                    abduction_soft_limit=float(coordination_abduction_soft_limit),
                )
            )
        else:
            coordination_penalty = torch.zeros_like(q[:, 0])
            canonical_abduction_penalty = torch.zeros_like(q[:, 0])
        last_loss = (
            contact_loss
            + float(pose_reg) * regularization
            + float(coordination_reg_weight) * coordination_penalty.mean()
            + float(canonical_abduction_reg_weight)
            * canonical_abduction_penalty.mean()
        )
        last_loss.backward()
        optimizer.step()
        with torch.no_grad():
            q.clamp_(finite_lower, finite_upper)
            if lock_abduction and abd_indices:
                q[:, abd_indices] = base_q[:, abd_indices]

    with torch.no_grad():
        current_translation = base_translation + torch.tanh(delta_translation) * float(max_trans_delta)
        rotation_vector = torch.tanh(delta_rotation) * float(max_rot_delta)
        theta = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
        axis = rotation_vector / theta.clamp_min(1e-8)
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(-1, 3, 3)
        eye = torch.eye(
            3, device=base_rotation.device, dtype=base_rotation.dtype
        ).expand_as(skew)
        rot_delta = (
            eye
            + torch.sin(theta)[..., None] * skew
            + (1.0 - torch.cos(theta))[..., None] * torch.matmul(skew, skew)
        )
        current_rotation = torch.matmul(rot_delta, base_rotation)
        fitted_tips = fingertip_representatives(current_translation, current_rotation, q)
        contact_mse = (fitted_tips - target_contacts).square().mean(dim=(1, 2))
    return (
        q.detach(),
        current_translation.detach(),
        current_rotation.detach(),
        contact_mse.detach(),
    )


def fit_allegro_qpos_to_contacts_two_phase(
    qpos: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    target_contacts: torch.Tensor,
    robot_model,
    hand_model=None,
    object_pc: torch.Tensor | None = None,
    steps: int = 200,
    lr: float = 0.03,
    pose_reg: float = 0.01,
    max_trans_delta: float = 0.12,
    max_rot_delta: float = 1.0,
    lock_abduction: bool = False,
    finger_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    abduction_reg_weight: float = 2.0,
    coordination_reg_weight: float = 0.0,
    canonical_abduction_reg_weight: float = 0.0,
    coordination_q2_ratio: float = 0.75,
    coordination_q3_ratio: float = 0.50,
    coordination_abduction_soft_limit: float = 0.25,
):
    """Two-phase Allegro IK: explore from canonical, then refine contacts."""
    phase1_steps = max(int(steps) // 2, 80)
    phase2_steps = max(int(steps) - phase1_steps, 80)
    qpos, translation, rotation, _ = fit_allegro_qpos_to_contacts(
        qpos,
        translation,
        rotation,
        target_contacts,
        robot_model,
        hand_model=hand_model,
        object_pc=object_pc,
        steps=phase1_steps,
        lr=float(lr) * 1.2,
        pose_reg=float(pose_reg) * 2.5,
        max_trans_delta=float(max_trans_delta),
        max_rot_delta=float(max_rot_delta),
        lock_abduction=lock_abduction,
        finger_weights=finger_weights,
        abduction_reg_weight=abduction_reg_weight,
        coordination_reg_weight=coordination_reg_weight,
        canonical_abduction_reg_weight=canonical_abduction_reg_weight,
        coordination_q2_ratio=coordination_q2_ratio,
        coordination_q3_ratio=coordination_q3_ratio,
        coordination_abduction_soft_limit=coordination_abduction_soft_limit,
    )
    return fit_allegro_qpos_to_contacts(
        qpos,
        translation,
        rotation,
        target_contacts,
        robot_model,
        hand_model=hand_model,
        object_pc=object_pc,
        steps=phase2_steps,
        lr=float(lr),
        pose_reg=max(float(pose_reg) * 0.35, 1e-4),
        max_trans_delta=float(max_trans_delta),
        max_rot_delta=float(max_rot_delta),
        lock_abduction=lock_abduction,
        finger_weights=finger_weights,
        abduction_reg_weight=abduction_reg_weight,
        coordination_reg_weight=coordination_reg_weight,
        canonical_abduction_reg_weight=canonical_abduction_reg_weight,
        coordination_q2_ratio=coordination_q2_ratio,
        coordination_q3_ratio=coordination_q3_ratio,
        coordination_abduction_soft_limit=coordination_abduction_soft_limit,
    )


def score_allegro_contact_candidates_batched(
    contacts_k: torch.Tensor,
    qpos_init: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    robot_model,
    ik_steps: int = 40,
    lr: float = 0.04,
    pose_reg: float = 0.05,
    max_trans_delta: float = 0.10,
    max_rot_delta: float = 0.80,
    wrist_warm_start: bool = True,
    wrist_warm_alpha: float = 0.35,
    wrist_warm_max_shift: float = 0.08,
    coordination_reg_weight: float = 0.0,
    canonical_abduction_reg_weight: float = 0.0,
) -> torch.Tensor:
    """Quick biotac IK reachability score for K contact candidates per grasp.

    Returns ``ik_mse`` with shape (batch, K). Lower is more reachable for Allegro.
    """
    if contacts_k.ndim != 4 or contacts_k.shape[2] != 4:
        raise ValueError(
            "contacts_k must have shape (batch, K, 4, 3), got "
            f"{tuple(contacts_k.shape)}"
        )
    batch_size, num_candidates = contacts_k.shape[:2]
    qpos_flat = (
        qpos_init.unsqueeze(1)
        .expand(-1, num_candidates, -1)
        .reshape(batch_size * num_candidates, -1)
    )
    rot_flat = (
        rotation.unsqueeze(1)
        .expand(-1, num_candidates, -1, -1)
        .reshape(batch_size * num_candidates, 3, 3)
    )
    if wrist_warm_start:
        centroids = contacts_k.mean(dim=2)
        delta = torch.clamp(
            centroids - translation.unsqueeze(1),
            -float(wrist_warm_max_shift),
            float(wrist_warm_max_shift),
        )
        trans_k = translation.unsqueeze(1) + float(wrist_warm_alpha) * delta
        trans_flat = trans_k.reshape(batch_size * num_candidates, 3)
    else:
        trans_flat = (
            translation.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, 3)
        )
    targets_flat = contacts_k.reshape(batch_size * num_candidates, 4, 3)
    _, _, _, ik_mse = fit_allegro_qpos_to_contacts(
        qpos_flat,
        trans_flat,
        rot_flat,
        targets_flat,
        robot_model,
        hand_model=None,
        object_pc=None,
        steps=int(ik_steps),
        lr=float(lr),
        pose_reg=float(pose_reg),
        max_trans_delta=float(max_trans_delta),
        max_rot_delta=float(max_rot_delta),
        lock_abduction=False,
        coordination_reg_weight=float(coordination_reg_weight),
        canonical_abduction_reg_weight=float(canonical_abduction_reg_weight),
    )
    return ik_mse.reshape(batch_size, num_candidates)
