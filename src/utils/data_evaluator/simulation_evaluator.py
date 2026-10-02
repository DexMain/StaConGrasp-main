import os
import yaml
import torch
import numpy as np
from typing import Optional, Union
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

from utils.data_evaluator.data_evaluator import DataEvaluator
from utils.simulator.simulator import get_simulator
from utils.robot_model import RobotModel
from utils.width_mapper import WidthMapper
from utils.collision_checker import CollisionChecker
from utils.qpos_adapter import (
    apply_allegro_thumb_shape_prior,
    equalize_allegro_thumb_approach_lead,
)
import time

class SimulationEvaluator(DataEvaluator):
    """
    class for simulation evaluator
    """
    
    def __init__(
        self, 
        config: dict, 
        device: torch.device,
    ):
        """
        initialize the class
        
        Args:
        - config: dict, config of the simulation evaluator
        - device: torch.device, device
        """
        super().__init__(config, device)
        
        # create robot model
        self._robot_model = RobotModel(config['urdf_path'], config['meta_path'])
        
        # create width mapper
        self._width_mapper = WidthMapper(self._robot_model, config['width_mapper_meta_path'])
        self._debug_state = bool(config.get('debug_state', False))
        self._debug_contacts = bool(config.get('debug_contacts', False))
        self._debug_printed_asset = False
        
        # load collision checker
        collision_checker_config_path = os.path.join(
            'configs/collision_checker', config['robot_name'], 'CollisionChecker.yaml')
        collision_checker_config = yaml.safe_load(open(collision_checker_config_path, 'r'))
        self._collision_checker = CollisionChecker(collision_checker_config, device)
    
    def set_environments(
        self, 
        object_pose_dict: dict, 
        object_surface_points_dict: dict, 
        num_envs: int, 
        dataset = 'graspnet'
    ):
        """
        set environments
        
        Args:
        - object_pose_dict: dict[str, np.ndarray[4, 4]], object code -> pose
        - object_surface_points_dict: dict[str, torch.tensor[num_points, 3]]
        - num_envs: int, number of environments
        """
        
        self._object_pose_dict = object_pose_dict
        self._object_surface_points_dict = object_surface_points_dict
        self._num_envs = num_envs
        
        # create simulator
        simulator_config_path = os.path.join(
            'configs/simulator', f'{self._config["simulator_type"]}.yaml')
        simulator_config = yaml.safe_load(open(simulator_config_path, 'r'))
        simulator_config['table_height'] = 0
        simulator_config['defer_viewer_init'] = bool(
            self._config.get('defer_viewer_init', False)
        )
        # Keep simulator defaults untouched; optional evaluator overrides are
        # used for Allegro-only A/B experiments.
        dof_overrides = self._config.get('simulator_dof_props_overrides', {})
        friction_override = self._config.get('simulator_friction_override')
        if self._config['robot_name'].startswith('allegro_hand') and (
            dof_overrides or friction_override is not None
        ):
            actor_config = simulator_config['actor'][self._config['robot_name']]['actor_config']
            if dof_overrides:
                dof_props = dict(actor_config.get('dof_props', {}))
                dof_props.update(dof_overrides)
                actor_config['dof_props'] = dof_props
            if friction_override is not None:
                actor_config['friction'] = float(friction_override)
            print(
                '[SimulationEvaluator] Allegro overrides '
                f'dof_props={dof_overrides or {}} '
                f'friction={actor_config.get("friction")}',
                flush=True,
            )
        simulator_class = get_simulator(self._config['simulator_type'])
        t = time.time()
        self._simulator = simulator_class(
            config=simulator_config,
            num_envs=num_envs,
            headless=self._config['headless'],
            device_id=int(str(self._device).split(':')[-1]))
        
        # register assets
        t = time.time()
        self._robot_info = self._simulator.register_asset(
            asset_name='robot', 
            asset_root='', 
            asset_path=self._config['robot_name'] + '_free', 
            asset_config={}, 
        )
        self._object_info_dict = {}
        if dataset == 'graspnet':
            mesh_root = self._config.get('mesh_root', '/data/meshdata')
            for object_code in object_pose_dict:
                self._object_info_dict[object_code] = self._simulator.register_asset(
                    asset_name=f'object_{object_code}', 
                    asset_root=os.path.join(mesh_root, object_code), 
                    asset_path='nontextured_simplified.urdf', 
                    asset_config=None, 
                )
        elif dataset == 'acronym':
            for object_code in object_pose_dict:
                self._object_info_dict[object_code] = self._simulator.register_asset(
                    asset_name=f'object_{object_code}', 
                    asset_root=os.path.join('data/acronym/meshes/models', object_code), 
                    asset_path='collision.urdf', 
                    asset_config=None, 
                )
        
        # create environments
        for i in range(num_envs):
            # create env
            self._simulator.create_env()
            # create actors
            self._simulator.create_actor(
                actor_name='robot',
                asset_name='robot',
                actor_config=self._config['robot_name'] + '_free', 
            )
            for object_index, object_code in enumerate(object_pose_dict):
                self._simulator.create_actor(
                    actor_name=f'object_{object_code}',
                    asset_name=f'object_{object_code}',
                    actor_config=dict(
                        no_collision=False, 
                        filter=2, 
                        segmentation_id=0, 
                        friction=1, 
                        dof_force_sensors=False, 
                        mass=0.1)
                )
        
        # prepare simulator
        self._simulator.prepare_sim()
        if self._debug_state and not self._debug_printed_asset:
            print(
                '[SimulationEvaluator][debug] robot asset '
                f'dofs={self._robot_info["num_dofs"]} '
                f'dof_names={self._robot_info["dof_names"]}',
                flush=True,
            )
            print(
                '[SimulationEvaluator][debug] robot bodies '
                f'{self._robot_info.get("body_names", [])}',
                flush=True,
            )
            self._debug_printed_asset = True

    def _build_pregrasp_qpos_dict(
        self,
        grasp_qpos_dict: dict,
        batch_size: int,
        dof_names,
    ) -> dict:
        """Build finger qpos for pregrasp/cover waypoints."""
        mode = str(self._config.get('pregrasp_qpos_mode', 'relative')).lower()
        pregrasp_open_delta = float(self._config.get('pregrasp_open_delta', -0.025))
        if mode == 'canonical':
            canonical_qpos = self._width_mapper._robot_meta.get(
                'canonical_pose', {}
            ).get('qpos', {})
            pregrasp_qpos_dict = {}
            for joint_name in dof_names:
                value = float(canonical_qpos.get(joint_name, 0.0))
                pregrasp_qpos_dict[joint_name] = torch.full(
                    (batch_size,),
                    value,
                    dtype=torch.float,
                    device=self._device,
                )
            return pregrasp_qpos_dict
        open_mode = str(
            self._config.get(
                "pregrasp_open_mode",
                self._config.get("squeeze_mode", "tip"),
            )
        ).lower()
        # Allegro tip-normal "open" mostly changes abduction. Prefer reducing
        # flexion so the hand keeps Leap-like opposed cradle shape.
        if open_mode in ("flex", "flexion", "joint"):
            flex_delta = float(
                self._config.get(
                    "pregrasp_flex_delta",
                    max(abs(float(pregrasp_open_delta)) * 6.0, 0.20),
                )
            )
            thumb_flex_delta = float(
                self._config.get("pregrasp_flex_thumb_delta", flex_delta)
            )
            return self._squeeze_qpos_by_flexion(
                grasp_qpos_dict,
                flex_delta=-abs(flex_delta),
                thumb_flex_delta=-abs(thumb_flex_delta),
            )
        return self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, pregrasp_open_delta, pregrasp_open_delta
        )[0]

    def _squeeze_qpos_by_flexion(
        self,
        grasp_qpos_dict: dict,
        flex_delta: float,
        thumb_flex_delta=None,
    ) -> dict:
        """Close Allegro fingers by increasing flexion joints (not tip-normal IK).

        ``width_mapper.squeeze_fingers`` moves tips along local normals and, on
        ROS-V5 Allegro, mainly changes abduction (``*_joint_0``), which looks
        like a sideways nudge toward the ring finger.  This helper keeps
        abduction fixed and adds flexion on ``joint_1/2`` (and thumb ``1/2/3``).
        """
        if thumb_flex_delta is None:
            thumb_flex_delta = float(flex_delta)
        out = {
            name: value.clone() if torch.is_tensor(value) else value
            for name, value in grasp_qpos_dict.items()
        }
        finger_specs = (
            ("index", (1, 2, 3), float(flex_delta)),
            ("middle", (1, 2, 3), float(flex_delta)),
            ("ring", (1, 2, 3), float(flex_delta)),
            ("thumb", (1, 2, 3), float(thumb_flex_delta)),
        )
        for finger, joint_ids, delta in finger_specs:
            for joint_id in joint_ids:
                name = f"{finger}_joint_{joint_id}"
                if name not in out:
                    continue
                urdf_i = self._robot_model.joint_names.index(name)
                lo = float(self._robot_model.joints_lower[urdf_i])
                hi = float(self._robot_model.joints_upper[urdf_i])
                out[name] = torch.clamp(out[name] + delta, min=lo, max=hi)
        return out

    def _reinforce_allegro_thumb_qpos(self, grasp_qpos_dict: dict) -> dict:
        """Keep Allegro thumb opposed (CMC) without claw-curling the chain.

        Leap-like grasps rotate ``thumb_joint_0`` toward the fingers while
        leaving ``thumb_joint_1/2/3`` relatively straight.  Only opposition is
        reinforced here; extra flexion is off by default.
        """
        if "thumb_joint_0" not in grasp_qpos_dict:
            return grasp_qpos_dict
        names = list(self._robot_model.movable_joint_names)
        qpos = torch.stack(
            [grasp_qpos_dict[name] for name in names], dim=1
        )
        qpos = apply_allegro_thumb_shape_prior(qpos, self._robot_model)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        q0_boost = float(self._config.get("allegro_thumb_extra_oppose", 0.12))
        if q0_boost != 0.0 and "thumb_joint_0" in name_to_idx:
            col = name_to_idx["thumb_joint_0"]
            urdf_i = self._robot_model.joint_names.index("thumb_joint_0")
            lo = float(self._robot_model.joints_lower[urdf_i])
            hi = float(self._robot_model.joints_upper[urdf_i])
            qpos[:, col] = torch.clamp(qpos[:, col] + q0_boost, min=lo, max=hi)
        # Soft-cap thumb chain flex (Leap j13/j14/j15 stay nearly straight).
        flex_caps = (
            ("thumb_joint_1", float(self._config.get("allegro_thumb_j1_max", 0.60))),
            ("thumb_joint_2", float(self._config.get("allegro_thumb_j2_max", 0.75))),
            ("thumb_joint_3", float(self._config.get("allegro_thumb_j3_max", 0.35))),
        )
        for name, ceiling in flex_caps:
            if name not in name_to_idx:
                continue
            col = name_to_idx[name]
            urdf_i = self._robot_model.joint_names.index(name)
            lo = float(self._robot_model.joints_lower[urdf_i])
            hi = float(self._robot_model.joints_upper[urdf_i])
            target = min(max(ceiling, lo), hi)
            qpos[:, col] = torch.minimum(qpos[:, col], torch.full_like(qpos[:, col], target))
        extra = float(self._config.get("allegro_thumb_extra_flex", 0.0))
        if extra != 0.0:
            for joint_id in (1, 2, 3):
                name = f"thumb_joint_{joint_id}"
                if name not in name_to_idx:
                    continue
                col = name_to_idx[name]
                urdf_i = self._robot_model.joint_names.index(name)
                lo = float(self._robot_model.joints_lower[urdf_i])
                hi = float(self._robot_model.joints_upper[urdf_i])
                qpos[:, col] = torch.clamp(qpos[:, col] + extra, min=lo, max=hi)
        return {
            name: qpos[:, idx] for idx, name in enumerate(names)
        }

    def _resolve_allegro_finger_bias(
        self,
        pose: torch.Tensor,
        qpos_dict: dict,
        canonical_frame_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Per-grasp lateral bias (m). Positive = toward three fingers.

        Adaptive mode uses open-hand geometry from ``qpos_dict``:

        ``ratio = thumb_lead_along_approach / thumb–finger_lateral_span``

        Low ratio → toward-fingers bias; high ratio (thumb-stab risk) →
        toward-thumb bias. Continuous lerp avoids collapsing every leaplike
        open grasp to a single sign (raw lead is ~6–7cm for almost all).
        """
        batch = pose.shape[0]
        device = pose.device
        dtype = pose.dtype
        fixed = float(self._config.get("allegro_descend_finger_bias", 0.0))
        adaptive = bool(
            int(self._config.get("allegro_descend_finger_bias_adaptive", 0))
        )
        if not adaptive:
            return torch.full((batch,), fixed, device=device, dtype=dtype)

        # Scene whitelist: keep historical fixed bias (e.g. small-object 0101).
        scene_id = str(self._config.get("scene_id", "") or "").strip()
        fixed_scenes = self._config.get(
            "allegro_descend_finger_bias_fixed_scenes", []
        ) or []
        fixed_scene_set = {str(s).strip() for s in fixed_scenes if str(s).strip()}
        if scene_id and scene_id in fixed_scene_set:
            if not getattr(self, "_logged_adaptive_finger_bias", False):
                print(
                    "[SimulationEvaluator] Allegro finger_bias FIXED override "
                    f"scene={scene_id} bias={fixed:+.3f} "
                    f"(whitelist={sorted(fixed_scene_set)})",
                    flush=True,
                )
                self._logged_adaptive_finger_bias = True
            return torch.full((batch,), fixed, device=device, dtype=dtype)

        tip_names = list(self._robot_model.fingertip_link_names)
        if len(tip_names) < 2:
            return torch.full((batch,), fixed, device=device, dtype=dtype)

        bias_low = float(
            self._config.get("allegro_descend_finger_bias_low_lead", 0.03)
        )
        bias_high = float(
            self._config.get("allegro_descend_finger_bias_high_lead", -0.01)
        )
        ratio_lo = float(
            self._config.get("allegro_descend_lead_span_ratio_low", 1.0)
        )
        ratio_hi = float(
            self._config.get("allegro_descend_lead_span_ratio_high", 1.8)
        )
        if ratio_hi <= ratio_lo:
            raise ValueError(
                "allegro_descend_lead_span_ratio_high must exceed "
                "allegro_descend_lead_span_ratio_low"
            )

        link_t, _ = self._robot_model.forward_kinematics(dict(qpos_dict))
        rotation = pose[:, :3, :3]
        thumb_local = link_t[tip_names[0]]
        finger_local = torch.stack(
            [link_t[name] for name in tip_names[1:]], dim=1
        ).mean(dim=1)
        approach = (rotation @ canonical_frame_rotation.T)[:, :, 0]
        approach = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r_t_approach = torch.einsum("bji,bj->bi", rotation, approach)
        delta_local = thumb_local - finger_local
        lead = (delta_local * r_t_approach).sum(dim=-1)
        # Lateral span in the plane orthogonal to approach (local).
        lat = delta_local - lead.unsqueeze(-1) * r_t_approach
        span = lat.norm(dim=-1).clamp_min(1e-6)
        ratio = lead / span

        alpha = ((ratio - ratio_lo) / (ratio_hi - ratio_lo)).clamp(0.0, 1.0)
        bias = (1.0 - alpha) * bias_low + alpha * bias_high
        if not getattr(self, "_logged_adaptive_finger_bias", False):
            print(
                "[SimulationEvaluator] Allegro adaptive finger_bias "
                f"ratio[{ratio_lo:.2f},{ratio_hi:.2f}]→"
                f"[{bias_low:+.3f},{bias_high:+.3f}] | "
                f"batch mean_lead={float(lead.mean()):.4f}m "
                f"mean_span={float(span.mean()):.4f}m "
                f"mean_ratio={float(ratio.mean()):.3f} "
                f"mean_bias={float(bias.mean()):+.4f}",
                flush=True,
            )
            self._logged_adaptive_finger_bias = True
        return bias

    def _bias_allegro_descend_pose(
        self,
        grasp_pose: torch.Tensor,
        grasp_qpos_dict: dict,
        canonical_frame_rotation: torch.Tensor,
        finger_bias: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """Shift Allegro grasp pose: lateral finger/thumb bias + optional deepen.

        ``finger_bias``: scalar or ``(B,)``. Positive moves toward three-finger
        midpoint; negative toward thumb. Adaptive callers pass a per-grasp
        tensor resolved from open-hand thumb lead.
        """
        approach_extra = float(
            self._config.get("allegro_descend_approach_extra", 0.0)
        )
        if finger_bias is None:
            finger_bias = self._resolve_allegro_finger_bias(
                grasp_pose, grasp_qpos_dict, canonical_frame_rotation
            )
        if not torch.is_tensor(finger_bias):
            finger_bias = torch.full(
                (grasp_pose.shape[0],),
                float(finger_bias),
                device=grasp_pose.device,
                dtype=grasp_pose.dtype,
            )
        else:
            finger_bias = finger_bias.to(
                device=grasp_pose.device, dtype=grasp_pose.dtype
            )

        if float(finger_bias.abs().max()) == 0.0 and approach_extra == 0.0:
            return grasp_pose

        tip_names = list(self._robot_model.fingertip_link_names)
        if len(tip_names) < 2:
            return grasp_pose

        link_t, _ = self._robot_model.forward_kinematics(dict(grasp_qpos_dict))
        translation = grasp_pose[:, :3, 3]
        rotation = grasp_pose[:, :3, :3]
        thumb_world = translation + torch.einsum(
            "bij,bj->bi", rotation, link_t[tip_names[0]]
        )
        finger_world = translation + torch.einsum(
            "bij,bj->bi",
            rotation,
            torch.stack(
                [link_t[name] for name in tip_names[1:]], dim=1
            ).mean(dim=1),
        )
        delta = torch.zeros_like(translation)
        if float(finger_bias.abs().max()) != 0.0:
            sep = finger_world - thumb_world
            sep = sep / sep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = delta + finger_bias.unsqueeze(-1) * sep
        if approach_extra != 0.0:
            # Grasp approach axis in world: (R @ C.T)[:, :, 0]
            approach = (rotation @ canonical_frame_rotation.T)[:, :, 0]
            delta = delta + float(approach_extra) * approach

        biased = grasp_pose.clone()
        biased[:, :3, 3] = translation + delta
        return biased

    def _world_collision_min_z(
        self,
        pose: torch.Tensor,
        qpos_dict: dict,
    ):
        """Return whole-hand, palm, and fingertip minimum world z values."""
        local_translations, local_rotations = self._robot_model.forward_kinematics(
            dict(qpos_dict)
        )
        global_translation = pose[:, :3, 3]
        global_rotation = pose[:, :3, :3]
        whole_min = torch.full(
            (pose.shape[0],),
            float("inf"),
            dtype=pose.dtype,
            device=pose.device,
        )
        palm_min = whole_min.clone()
        fingertip_min = whole_min.clone()
        fingertip_names = set(self._robot_model.fingertip_link_names)
        for link_name, geometry in self._robot_model._geometry.items():
            vertices = geometry.get("collision_vertices")
            if vertices is None or len(vertices) == 0:
                continue
            vertices = torch.as_tensor(
                vertices, dtype=pose.dtype, device=pose.device
            )
            local_points = (
                vertices.unsqueeze(0)
                @ local_rotations[link_name].transpose(1, 2)
                + local_translations[link_name].unsqueeze(1)
            )
            world_points = (
                local_points @ global_rotation.transpose(1, 2)
                + global_translation.unsqueeze(1)
            )
            link_min = world_points[:, :, 2].min(dim=1).values
            whole_min = torch.minimum(whole_min, link_min)
            if link_name == "palm_link":
                palm_min = torch.minimum(palm_min, link_min)
            if link_name in fingertip_names:
                fingertip_min = torch.minimum(fingertip_min, link_min)
        return whole_min, palm_min, fingertip_min

    @staticmethod
    def _format_clearance_stats(values):
        values = values.detach().cpu().numpy()
        return (
            f"mean={values.mean():.4f} "
            f"min={values.min():.4f} max={values.max():.4f}"
        )

    def _debug_waypoint_clearance(self, name, pose, qpos_dict):
        if not self._debug_state:
            return
        whole_min, palm_min, fingertip_min = self._world_collision_min_z(
            pose, qpos_dict
        )
        print(
            f"[SimulationEvaluator][clearance] waypoint={name} "
            f"palm_min_z({self._format_clearance_stats(palm_min)}) "
            f"hand_min_z({self._format_clearance_stats(whole_min)}) "
            f"tip_min_z({self._format_clearance_stats(fingertip_min)})",
            flush=True,
        )

    def _compute_waypoints(
        self,
        grasps: dict, 
    ):
        """
        compute waypoints: pregrasp, cover, grasp, squeeze, lift
        
        Args:
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        """
        
        batch_size = len(grasps['translation'])
        assert batch_size <= self._num_envs, \
            'batch size should be less than or equal to number of environments'
        
        self._waypoint_pose_list = []
        self._waypoint_qpos_dict_list = []
        self._waypoint_qpos_list = []
        dof_names = self._robot_info['dof_names'][6:]
        canonical_frame_rotation = torch.tensor(
            self._config['canonical_frame_rotation'], dtype=torch.float, device=self._device)
        
        # get grasp pose and qpos (final grasp / squeeze target)
        grasp_pose = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pose[:, :3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self._device)
        grasp_pose[:, :3, :3] = torch.tensor(grasps['rotation'], 
            dtype=torch.float, device=self._device)
        grasp_qpos_dict = {
            joint_name: torch.tensor(
                grasps[joint_name], dtype=torch.float, device=self._device
            )
            for joint_name in dof_names
            if joint_name in grasps
        }
        robot_name = str(self._config.get("robot_name", "")).lower()

        # Optional stage-1 coarse fields: keep open-hand approach from coarse,
        # and only use the (stage-2) main fields at grasp/squeeze.
        use_coarse_pregrasp = (
            robot_name.startswith("allegro_hand")
            and bool(int(self._config.get("allegro_coarse_pregrasp", 1)))
            and ("coarse_translation" in grasps)
            and ("coarse_rotation" in grasps)
        )
        if use_coarse_pregrasp:
            coarse_pose = torch.eye(4, dtype=torch.float, device=self._device
                ).unsqueeze(0).repeat(batch_size, 1, 1)
            coarse_pose[:, :3, 3] = torch.tensor(
                grasps["coarse_translation"], dtype=torch.float, device=self._device
            )
            coarse_pose[:, :3, :3] = torch.tensor(
                grasps["coarse_rotation"], dtype=torch.float, device=self._device
            )
            coarse_qpos_dict = {}
            for joint_name in dof_names:
                coarse_key = f"coarse_{joint_name}"
                if coarse_key in grasps:
                    coarse_qpos_dict[joint_name] = torch.tensor(
                        grasps[coarse_key], dtype=torch.float, device=self._device
                    )
                elif joint_name in grasps:
                    coarse_qpos_dict[joint_name] = torch.tensor(
                        grasps[joint_name], dtype=torch.float, device=self._device
                    )
            print(
                "[SimulationEvaluator] Allegro coarse pregrasp enabled "
                "(open approach from stage-1, grasp/squeeze from stage-2)",
                flush=True,
            )
        else:
            coarse_pose = None
            coarse_qpos_dict = None

        # Prefer OPEN coarse qpos for adaptive lead/span when present, even if
        # allegro_coarse_pregrasp=0 (approach still uses stage-2 + preopen).
        finger_bias_vec = None
        if robot_name.startswith("allegro_hand"):
            lead_qpos = grasp_qpos_dict
            lead_pose = grasp_pose
            if (
                "coarse_translation" in grasps
                and "coarse_rotation" in grasps
            ):
                lead_pose = torch.eye(4, dtype=torch.float, device=self._device
                    ).unsqueeze(0).repeat(batch_size, 1, 1)
                lead_pose[:, :3, 3] = torch.tensor(
                    grasps["coarse_translation"],
                    dtype=torch.float,
                    device=self._device,
                )
                lead_pose[:, :3, :3] = torch.tensor(
                    grasps["coarse_rotation"],
                    dtype=torch.float,
                    device=self._device,
                )
                lead_qpos = {}
                for joint_name in dof_names:
                    coarse_key = f"coarse_{joint_name}"
                    if coarse_key in grasps:
                        lead_qpos[joint_name] = torch.tensor(
                            grasps[coarse_key],
                            dtype=torch.float,
                            device=self._device,
                        )
                    elif joint_name in grasp_qpos_dict:
                        lead_qpos[joint_name] = grasp_qpos_dict[joint_name]
            elif coarse_qpos_dict is not None:
                lead_qpos = coarse_qpos_dict
                lead_pose = coarse_pose
            finger_bias_vec = self._resolve_allegro_finger_bias(
                lead_pose, lead_qpos, canonical_frame_rotation
            )

        if robot_name.startswith("allegro_hand") and bool(
            int(self._config.get("allegro_reinforce_thumb", 1))
        ):
            grasp_qpos_dict = self._reinforce_allegro_thumb_qpos(grasp_qpos_dict)
            if coarse_qpos_dict is not None:
                coarse_qpos_dict = self._reinforce_allegro_thumb_qpos(coarse_qpos_dict)
        # Allegro-only: apply the SAME per-grasp lateral/depth bias to grasp +
        # coarse so approach/final stay aligned.
        if robot_name.startswith("allegro_hand"):
            grasp_pose = self._bias_allegro_descend_pose(
                grasp_pose,
                grasp_qpos_dict,
                canonical_frame_rotation,
                finger_bias=finger_bias_vec,
            )
            if coarse_pose is not None and coarse_qpos_dict is not None:
                coarse_pose = self._bias_allegro_descend_pose(
                    coarse_pose,
                    coarse_qpos_dict,
                    canonical_frame_rotation,
                    finger_bias=finger_bias_vec,
                )
        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        trajectory_mode = str(
            self._config.get("trajectory_mode", "legacy")
        ).lower()
        use_allegro_hover_trajectory = trajectory_mode == "allegro_hover"
        use_allegro_safe_trajectory = trajectory_mode in (
            "allegro_safe",
            "allegro_hover",
        )
        
        pregrasp_open_delta = float(self._config.get('pregrasp_open_delta', -0.025))
        squeeze_delta = float(self._config.get('squeeze_delta', 0.03))
        pregrasp_offset = float(self._config.get('pregrasp_offset', -0.1))

        # waypoint 1 (pregrasp): open hand back along approach.
        # Prefer stage-1 coarse so stage-2 does not rewrite the open approach.
        pregrasp_source_qpos = (
            coarse_qpos_dict if coarse_qpos_dict is not None else grasp_qpos_dict
        )
        pregrasp_source_pose = (
            coarse_pose if coarse_pose is not None else grasp_pose
        )
        pregrasp_qpos_dict = self._build_pregrasp_qpos_dict(
            pregrasp_source_qpos, batch_size, dof_names
        )
        # Allegro-only: after opening fingers for approach, pull thumb tip back
        # along canonical approach so descent is not thumb-first. Soft open-hand
        # caps are OFF here — they freeze lead ~70mm; grasp/squeeze still use
        # stage-2 qpos, not this approach thumb curl.
        if robot_name.startswith("allegro_hand") and self._config.get(
            "allegro_pregrasp_thumb_lead_m", None
        ) is not None:
            target_lead = float(self._config["allegro_pregrasp_thumb_lead_m"])
            q_stack = torch.stack(
                [pregrasp_qpos_dict[name] for name in dof_names], dim=1
            )
            q_fix, lead = equalize_allegro_thumb_approach_lead(
                q_stack,
                self._robot_model,
                target_lead=target_lead,
                open_hand_soft_caps=False,
                steps=int(self._config.get("allegro_pregrasp_thumb_lead_steps", 80)),
            )
            for index, name in enumerate(dof_names):
                pregrasp_qpos_dict[name] = q_fix[:, index]
            if not getattr(self, "_logged_thumb_lead", False):
                print(
                    "[SimulationEvaluator] Allegro pregrasp thumb lead equalize "
                    f"target={target_lead:.3f}m mean_lead={float(lead.mean()):.4f}m "
                    "(fingers keep approach open; grasp uses stage-2)",
                    flush=True,
                )
                self._logged_thumb_lead = True
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        pregrasp_pose_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([pregrasp_offset, 0.0, 0.0], dtype=torch.float, device=self._device)
        pregrasp_pose = pregrasp_source_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)

        # Keep the open hand above the table while it moves to the object.
        # The per-grasp correction makes the actual mesh, rather than the
        # free-root origin, determine the minimum safe height.
        # Adaptive clearance is an Allegro-only experiment. A nonzero default
        # here would silently change the validated Leap baseline.
        approach_lift = float(self._config.get("approach_lift", 0.0))
        approach_clearance = float(
            self._config.get(
                "approach_clearance",
                0.015 if use_allegro_safe_trajectory else 0.0,
            )
        )
        pregrasp_z_offset = float(
            self._config.get("pregrasp_z_offset", 0.0)
        )
        if pregrasp_z_offset != 0.0:
            # This affects only the open-hand approach. The final grasp pose
            # stays unchanged. The safe-trajectory branch below adds any
            # clearance correction needed after applying this signed offset.
            pregrasp_pose = pregrasp_pose.clone()
            pregrasp_pose[:, 2, 3] += pregrasp_z_offset
        if use_allegro_safe_trajectory and (
            approach_lift > 0.0
            or approach_clearance > 0.0
            or pregrasp_z_offset != 0.0
        ):
            open_min_z, _, _ = self._world_collision_min_z(
                pregrasp_pose, pregrasp_qpos_dict
            )
            required_lift = torch.clamp(
                approach_clearance - open_min_z, min=0.0
            )
            effective_lift = torch.maximum(
                required_lift, torch.full_like(required_lift, approach_lift)
            )
            pregrasp_pose = pregrasp_pose.clone()
            pregrasp_pose[:, :3, 3] += torch.stack(
                [
                    torch.zeros_like(effective_lift),
                    torch.zeros_like(effective_lift),
                    effective_lift,
                ],
                dim=1,
            )
            cover_pose = grasp_pose.clone()
            cover_pose[:, :3, 3] += torch.stack(
                [
                    torch.zeros_like(effective_lift),
                    torch.zeros_like(effective_lift),
                    torch.full_like(
                        effective_lift, pregrasp_z_offset
                    ) + effective_lift,
                ],
                dim=1,
            )
        else:
            effective_lift = torch.zeros(
                batch_size, dtype=grasp_pose.dtype, device=grasp_pose.device
            )
            # Open-hand cover stays on the stage-1 coarse pose when provided.
            cover_pose = pregrasp_source_pose

        self._waypoint_pose_list.append(pregrasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        self._debug_waypoint_clearance(
            "pregrasp", pregrasp_pose, pregrasp_qpos_dict
        )
        
        # waypoint 2 (cover): stay above the object with relaxed fingers.
        self._waypoint_pose_list.append(cover_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        self._debug_waypoint_clearance(
            "cover", cover_pose, pregrasp_qpos_dict
        )
        
        # waypoint 3 (grasp pose): the validated Leap trajectory reaches the
        # final pose and target qpos here. Allegro can opt into an open-hand
        # descent because its larger collision meshes behave differently.
        grasp_waypoint_qpos_dict = (
            pregrasp_qpos_dict
            if use_allegro_safe_trajectory
            else grasp_qpos_dict
        )
        grasp_waypoint_qpos = torch.stack(
            [grasp_waypoint_qpos_dict[joint_name] for joint_name in dof_names],
            dim=1,
        )
        if use_allegro_hover_trajectory:
            hover_lift = float(self._config.get("hover_lift", 0.04))
            if hover_lift <= 0.0:
                raise ValueError("allegro_hover requires hover_lift > 0")
            hover_pose = grasp_pose.clone()
            hover_pose[:, :3, 3] += torch.tensor(
                [0.0, 0.0, hover_lift],
                dtype=grasp_pose.dtype,
                device=grasp_pose.device,
            )
            grasp_waypoint_pose = hover_pose
        else:
            grasp_waypoint_pose = grasp_pose
        self._waypoint_pose_list.append(grasp_waypoint_pose)
        self._waypoint_qpos_dict_list.append(grasp_waypoint_qpos_dict.copy())
        self._waypoint_qpos_list.append(grasp_waypoint_qpos)
        self._debug_waypoint_clearance(
            "hover_open"
            if use_allegro_hover_trajectory
            else ("grasp_open" if use_allegro_safe_trajectory else "grasp"),
            grasp_waypoint_pose,
            grasp_waypoint_qpos_dict,
        )

        # waypoint 4 (squeeze): close at the final pose, without translation.
        squeeze_mode = str(self._config.get("squeeze_mode", "tip")).lower()
        if squeeze_mode in ("flex", "flexion", "joint"):
            flex_delta = float(
                self._config.get(
                    "squeeze_flex_delta",
                    max(float(squeeze_delta), 0.25),
                )
            )
            thumb_flex_delta = float(
                self._config.get("squeeze_flex_thumb_delta", flex_delta)
            )
            target_qpos_dict = self._squeeze_qpos_by_flexion(
                grasp_qpos_dict,
                flex_delta=flex_delta,
                thumb_flex_delta=thumb_flex_delta,
            )
        else:
            target_qpos_dict = self._width_mapper.squeeze_fingers(
                grasp_qpos_dict, squeeze_delta, squeeze_delta, keep_z=True
            )[0]
        target_qpos = torch.stack([target_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        self._waypoint_pose_list.append(
            grasp_waypoint_pose if use_allegro_hover_trajectory else grasp_pose
        )
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        self._debug_waypoint_clearance(
            "hover_closed" if use_allegro_hover_trajectory else "squeeze",
            grasp_waypoint_pose if use_allegro_hover_trajectory else grasp_pose,
            target_qpos_dict,
        )

        if use_allegro_hover_trajectory:
            # Lower only after the fingers are closed. This avoids sweeping the
            # thumb through neighboring clutter while keeping the final grasp
            # pose and qpos unchanged.
            self._waypoint_pose_list.append(grasp_pose)
            self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
            self._waypoint_qpos_list.append(target_qpos)
            self._debug_waypoint_clearance(
                "hover_lower", grasp_pose, target_qpos_dict
            )
        
        # waypoint 5 (lift): squeeze fingers and lift
        # case 1 (top grasp): if gripper x-axis and gravity direction spans less than 60 degrees,
        # move back along gripper x-axis for 20cm
        # case 2 (side grasp): otherwise, move up along world z-axis for 20cm
        # case 1
        lift_pose_top_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        lift_pose_top_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float, device=self._device)
        lift_pose_top = grasp_pose @ lift_pose_top_local
        # case 2
        lift_pose_side = grasp_pose.clone()
        lift_pose_side[:, :3, 3] += torch.tensor([0.0, 0.0, 0.2], 
            dtype=torch.float, device=self._device)
        # compose
        gripper_x_axis = (grasp_pose[:, :3, :3] @ canonical_frame_rotation.T)[:, :, 0]
        gravity_direction = torch.tensor([0.0, 0.0, -1.0], 
            dtype=torch.float, device=self._device)
        top_mask = (gripper_x_axis * gravity_direction).sum(dim=1) > np.cos(np.pi / 3)
        lift_pose = torch.where(top_mask.unsqueeze(1).unsqueeze(1),
            lift_pose_top, lift_pose_side)
        self._waypoint_pose_list.append(lift_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        self._debug_waypoint_clearance("lift", lift_pose, target_qpos_dict)
        
        # pad waypoints
        if batch_size < self._num_envs:
            for i in range(len(self._waypoint_pose_list)):
                self._waypoint_pose_list[i] = torch.cat([
                    self._waypoint_pose_list[i], 
                    self._waypoint_pose_list[i][-1:].repeat(self._num_envs - batch_size, 1, 1)
                ], dim=0)
                for joint in self._waypoint_qpos_dict_list[i]:
                    self._waypoint_qpos_dict_list[i][joint] = torch.cat([
                        self._waypoint_qpos_dict_list[i][joint], 
                        self._waypoint_qpos_dict_list[i][joint][-1:]\
                            .repeat(self._num_envs - batch_size)
                    ], dim=0)
                self._waypoint_qpos_list[i] = torch.cat([
                    self._waypoint_qpos_list[i], 
                    self._waypoint_qpos_list[i][-1:].repeat(self._num_envs - batch_size, 1)
                ], dim=0)
        
        # compose pose and dof pos
        self._waypoint_qpos_all_list = []
        for i in range(len(self._waypoint_pose_list)):
            root_pos = self._waypoint_pose_list[i][:, :3, 3]
            root_rot = self._waypoint_pose_list[i][:, :3, :3]
            root_rot = matrix_to_euler_angles(root_rot, 'XYZ')  # convention equivalent to 'rxyz'
            dof_qpos = self._waypoint_qpos_list[i]
            dof_qpos_all = torch.cat([root_pos, root_rot, dof_qpos], dim=1)
            self._waypoint_qpos_all_list.append(dof_qpos_all)
    
    def _check_collision(self):
        """
        check collision between pregrasp and scene
        
        Returns:
        - pregrasp_valid: np.ndarray[num_envs, np.bool], pregrasp validity
        """
        grasps = {}
        grasps.update(self._waypoint_qpos_dict_list[0])
        grasps['translation'] = self._waypoint_pose_list[0][:, :3, 3]
        grasps['rotation'] = self._waypoint_pose_list[0][:, :3, :3]
        scene_point_cloud = torch.cat([
            torch.tensor(self._object_surface_points_dict[object_code], 
            dtype=torch.float, device=self._device) 
            for object_code in self._object_pose_dict], dim=0)
        scene_pen_distance, table_pen_distance = self._collision_checker.check_collision_batch(
            grasps, scene_point_cloud)
        scene_pen_distance = scene_pen_distance.cpu().numpy()
        table_pen_distance = table_pen_distance.cpu().numpy()
        scene_pen_valid = scene_pen_distance < self._config['scene_pen_threshold']
        table_pen_valid = table_pen_distance < self._config['table_pen_threshold']
        pregrasp_valid = scene_pen_valid & table_pen_valid
        return pregrasp_valid
    
    def _reset_environments(self):
        """
        reset environments
        """
        
        # set object states
        for object_code in self._object_pose_dict:
            object_pose = self._object_pose_dict[object_code]
            self._simulator.set_actor_states(
                actor_name=f'object_{object_code}',
                actor_states=dict(
                    root_pos=torch.tensor(object_pose[:3, 3], dtype=torch.float, 
                        device=self._device).unsqueeze(0).repeat(self._num_envs, 1),
                    root_rot=torch.tensor(object_pose[:3, :3],dtype=torch.float, 
                        device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1),
                    root_linvel=torch.zeros([self._num_envs, 3], 
                        dtype=torch.float, device=self._device),
                    root_angvel=torch.zeros([self._num_envs, 3], dtype=torch.float, 
                        device=self._device),
                )
            )
        
        # set robot states
        dof_pos_all = self._waypoint_qpos_all_list[0]
        self._simulator.set_actor_states(
            actor_name='robot',
            actor_states=dict(
                root_pos=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device), 
                root_rot=torch.eye(3, dtype=torch.float, device=self._device
                    ).unsqueeze(0).repeat(self._num_envs, 1, 1), 
                root_linvel=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device),
                root_angvel=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device),
                dof_pos=dof_pos_all, 
                dof_vel=torch.zeros_like(dof_pos_all),
            )
        )
        self._simulator.set_actor_actions(
            actor_name='robot',
            actor_actions=dof_pos_all, 
        )

    @staticmethod
    def _format_debug_vector(values):
        return np.array2string(
            values.detach().cpu().numpy(),
            precision=4,
            suppress_small=True,
            floatmode='fixed',
        )

    def _debug_robot_state(self, stage, target_qpos_all=None):
        """Print measured Allegro joint and link states for one real grasp env."""
        if not self._debug_state:
            return
        states = self._simulator.get_actor_states('robot')
        env_id = 1 if self._num_envs > 1 else 0
        dof_pos = states['dof_pos'][env_id]
        dof_names = list(self._robot_info['dof_names'])
        print(
            f'[SimulationEvaluator][debug] stage={stage} env={env_id} '
            f'root_qpos={self._format_debug_vector(dof_pos[:6])}',
            flush=True,
        )
        for finger in ('index', 'middle', 'ring', 'thumb'):
            names = [f'{finger}_joint_{i}' for i in range(4)]
            indices = [dof_names.index(name) for name in names]
            actual = dof_pos[indices]
            text = f'{finger}_actual={self._format_debug_vector(actual)}'
            if target_qpos_all is not None:
                target = target_qpos_all[env_id, indices]
                error = (actual - target).abs().max()
                text += (
                    f' target={self._format_debug_vector(target)}'
                    f' maxerr={float(error):.5f}'
                )
            print(f'[SimulationEvaluator][debug] {text}', flush=True)

        body_names = list(self._robot_info.get('body_names', []))
        body_pos = states.get('body_pos')
        if body_pos is None:
            return
        for body_name in (
            'index_link_1', 'index_link_2', 'index_link_3',
            'middle_link_1', 'middle_link_2', 'middle_link_3',
            'ring_link_1', 'ring_link_2', 'ring_link_3',
            'thumb_link_1', 'thumb_link_2', 'thumb_link_3',
        ):
            if body_name in body_names:
                body_index = body_names.index(body_name)
                print(
                    f'[SimulationEvaluator][debug] {body_name}_pos='
                    f'{self._format_debug_vector(body_pos[env_id, body_index])}',
                    flush=True,
                )

        # Isaac Gym collapses fixed Biotac links into link_3, so the
        # simulator body tensor does not expose *_biotac_tip bodies. Recompute
        # those FK points from the measured root pose and DOF state instead.
        dof_names = list(self._robot_info['dof_names'])
        full_dof_pos = states['dof_pos'][env_id]
        # The *_free URDF represents the world pose with six articulation
        # DOFs. Isaac's actor root state remains at the asset origin.
        root_pos = full_dof_pos[:3]
        root_rot = euler_angles_to_matrix(
            full_dof_pos[3:6].unsqueeze(0), "XYZ"
        )[0]
        qpos = full_dof_pos[6:].unsqueeze(0)
        qpos_dict = {
            name: qpos[:, index]
            for index, name in enumerate(dof_names[6:])
        }
        local_translations, _ = self._robot_model.forward_kinematics(qpos_dict)
        tip_names = (
            'thumb_biotac_tip',
            'index_biotac_tip',
            'middle_biotac_tip',
            'ring_biotac_tip',
        )
        tip_positions = {}
        for tip_name in tip_names:
            tip_position = (
                torch.matmul(root_rot, local_translations[tip_name][0])
                + root_pos
            )
            tip_positions[tip_name] = tip_position
            print(
                f'[SimulationEvaluator][debug] {tip_name}_fk_pos='
                f'{self._format_debug_vector(tip_position)}',
                flush=True,
            )

        # Match the measured FK points against the same world/table-frame
        # surface samples used to build the Isaac object actors. This avoids
        # confusing a bad contact target with a bad object-ID mapping.
        nearest = []
        for object_code in self._object_pose_dict:
            object_points = torch.as_tensor(
                self._object_surface_points_dict[object_code],
                dtype=root_pos.dtype,
                device=root_pos.device,
            )
            object_distances = []
            for tip_name in tip_names:
                distance = torch.linalg.norm(
                    object_points - tip_positions[tip_name].unsqueeze(0),
                    dim=-1,
                ).min()
                object_distances.append(distance)
            mean_distance = torch.stack(object_distances).mean()
            nearest.append((float(mean_distance), object_code, object_distances))
        nearest.sort(key=lambda item: item[0])
        for mean_distance, object_code, distances in nearest[:3]:
            print(
                f'[SimulationEvaluator][debug] {stage} tip_nn object={object_code} '
                f'mean={mean_distance:.4f} '
                f'thumb/index/middle/ring='
                f'{self._format_debug_vector(torch.stack(distances))}',
                flush=True,
            )

        for object_code in self._object_pose_dict:
            object_states = self._simulator.get_actor_states(
                f'object_{object_code}'
            )
            object_pos = object_states['root_pos'][env_id]
            object_vel = object_states['root_linvel'][env_id]
            print(
                f'[SimulationEvaluator][debug] {stage} object={object_code} '
                f'root_pos={self._format_debug_vector(object_pos)} '
                f'root_z={float(object_pos[2]):.5f} '
                f'root_linvel={self._format_debug_vector(object_vel)}',
                flush=True,
            )

    def _debug_contact_state(self, stage, env_id=1):
        """Print actual rigid contact pairs for one debug environment."""
        if not self._debug_contacts:
            return
        env_id = env_id if self._num_envs > 1 else 0
        # GPU PhysX exposes per-rigid-body contact forces through the tensor
        # API. The raw contact-list API is unavailable after simulation starts
        # with the GPU pipeline.
        try:
            robot_forces = self._simulator.get_actor_net_contact_forces(
                'robot'
            )[env_id]
            robot_body_names = list(self._robot_info.get('body_names', []))
            robot_force_norms = torch.linalg.norm(robot_forces, dim=-1)
            active_robot = [
                (
                    name,
                    float(robot_force_norms[index]),
                    self._format_debug_vector(robot_forces[index]),
                )
                for index, name in enumerate(robot_body_names)
                if float(robot_force_norms[index]) > 1e-4
            ]
            active_robot.sort(key=lambda row: row[1], reverse=True)
            object_rows = []
            for object_code in self._object_pose_dict:
                forces = self._simulator.get_actor_net_contact_forces(
                    f'object_{object_code}'
                )[env_id]
                force_sum = forces.sum(dim=0)
                force_norm = torch.linalg.norm(force_sum)
                body_norm = torch.linalg.norm(forces, dim=-1).max()
                object_rows.append(
                    f'object_{object_code}:sum_norm={float(force_norm):.5f} '
                    f'max_body_norm={float(body_norm):.5f} '
                    f'sum={self._format_debug_vector(force_sum)}'
                )
            active_text = ', '.join(
                f'{name}={force_norm:.5f}'
                for name, force_norm, _ in active_robot
            )
            print(
                f'[SimulationEvaluator][contact_tensor] stage={stage} '
                f'active_robot={active_text or "none"} '
                f'objects={" | ".join(object_rows)}',
                flush=True,
            )
            for name, force_norm, vector in active_robot[:20]:
                print(
                    f'[SimulationEvaluator][contact_tensor_body] '
                    f'stage={stage} body={name} norm={force_norm:.5f} '
                    f'force={vector}',
                    flush=True,
                )
            return
        except Exception as exc:
            print(
                f'[SimulationEvaluator][contact_tensor] stage={stage} '
                f'error={type(exc).__name__}: {exc}',
                flush=True,
            )
        try:
            contacts = self._simulator.get_env_rigid_contacts(env_id)
            body_lookup = self._simulator.get_env_body_lookup(env_id)
        except Exception as exc:
            print(
                f'[SimulationEvaluator][contacts] stage={stage} '
                f'error={type(exc).__name__}: {exc}',
                flush=True,
            )
            return

        rows = []
        for contact in contacts:
            body0 = int(getattr(contact, 'body0', -1))
            body1 = int(getattr(contact, 'body1', -1))
            actor0, name0 = body_lookup.get(body0, ('ground', 'ground'))
            actor1, name1 = body_lookup.get(body1, ('ground', 'ground'))
            if actor0 != 'robot' and actor1 != 'robot':
                continue
            normal = getattr(contact, 'normal', None)
            normal_xyz = (
                float(normal.x), float(normal.y), float(normal.z)
            ) if normal is not None else (0.0, 0.0, 0.0)
            rows.append({
                'pair': f'{actor0}:{name0}<->{actor1}:{name1}',
                'other': actor1 if actor0 == 'robot' else actor0,
                'body': name0 if actor0 == 'robot' else name1,
                'lambda': float(getattr(contact, 'lambda', 0.0)),
                'lambda_friction': float(
                    getattr(contact, 'lambda_friction', 0.0)
                ),
                'min_dist': float(getattr(contact, 'min_dist', 0.0)),
                'normal': normal_xyz,
            })

        rows.sort(key=lambda row: row['lambda'], reverse=True)
        object_rows = [row for row in rows if row['other'].startswith('object_')]
        ground_rows = [row for row in rows if row['other'] == 'ground']
        object_summary = {}
        for row in object_rows:
            key = row['other']
            summary = object_summary.setdefault(
                key, {'count': 0, 'lambda': 0.0, 'bodies': set()}
            )
            summary['count'] += 1
            summary['lambda'] += row['lambda']
            summary['bodies'].add(row['body'])
        summaries = []
        for object_name, summary in sorted(object_summary.items()):
            bodies = ','.join(sorted(summary['bodies']))
            summaries.append(
                f'{object_name}:n={summary["count"]} '
                f'lambda={summary["lambda"]:.4f} bodies={bodies}'
            )
        print(
            f'[SimulationEvaluator][contacts] stage={stage} '
            f'total_robot_pairs={len(rows)} '
            f'object_pairs={len(object_rows)} ground_pairs={len(ground_rows)} '
            f'objects={" | ".join(summaries) if summaries else "none"}',
            flush=True,
        )
        for row in rows[:20]:
            print(
                f'[SimulationEvaluator][contact] stage={stage} '
                f'pair={row["pair"]} lambda={row["lambda"]:.5f} '
                f'fric={row["lambda_friction"]:.5f} '
                f'min_dist={row["min_dist"]:.5f} '
                f'normal={np.asarray(row["normal"])}',
                flush=True,
            )
    
    def _execute_waypoints(self):
        """
        execute waypoints
        """
        
        # disable gravity before the first waypoint (pregrasp)
        for object_code in self._object_pose_dict:
            self._simulator.disable_gravity(f'object_{object_code}')
        # execute waypoints
        for i in range(1, len(self._waypoint_pose_list)):
            start_qpos_all = self._waypoint_qpos_all_list[i - 1]
            end_qpos_all = self._waypoint_qpos_all_list[i]
            steps = self._config['waypoint_steps'][i - 1]
            for step in range(steps):
                target_qpos = start_qpos_all + (end_qpos_all - start_qpos_all) * (step + 1) / steps
                self._simulator.set_actor_actions(
                    actor_name='robot',
                    actor_actions=target_qpos,
                )
                self._simulator.step()
            self._debug_robot_state(f'after_waypoint_{i}', end_qpos_all)
            self._debug_contact_state(f'after_waypoint_{i}')
            # enable gravity after the fourth waypoint (squeeze)
            if i == 3:
                for object_code in self._object_pose_dict:
                    self._simulator.enable_gravity(f'object_{object_code}')
    
    def _get_sim_successes(self):
        """
        get successful environments
        
        Returns:
        - successes: np.ndarray[num_envs, np.bool], successes
        """
        
        sim_successes = np.zeros(self._num_envs, dtype=bool)
        
        # successful if any object is lifted by 3cm
        for object_code in self._object_pose_dict:
            object_height_init = self._object_pose_dict[object_code][2, 3]
            object_height_final = self._simulator.get_actor_states(
                f'object_{object_code}')['root_pos'][:, 2].cpu().numpy()
            sim_successes |= (object_height_final > object_height_init + 0.03)
        
        return sim_successes

    def evaluate_data(
        self, 
        grasps: dict,
    ):
        """
        evaluate a batch of grasps on one cluttered scene
        
        Args:
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        
        Returns:
        - successes: np.ndarray[batch_size, np.bool], successes
        """
        
        # compute waypoints
        self._compute_waypoints(grasps)
        
        # check collision
        pregrasp_valid = self._check_collision()[:len(grasps['translation'])]
        
        # reset environments
        self._reset_environments()
        self._simulator.step()
        self._simulator.step()
        self._reset_environments()
        self._debug_robot_state(
            'after_reset',
            self._waypoint_qpos_all_list[0],
        )
        
        # execute waypoints
        self._execute_waypoints()
        
        # get sim_successes
        sim_successes = self._get_sim_successes()[:len(grasps['translation'])]
        self._debug_robot_state(
            'final',
            self._waypoint_qpos_all_list[-1],
        )
        self._debug_contact_state('final')

        successes = pregrasp_valid & sim_successes
        print(
            f"[SimulationEvaluator] batch={len(successes)} "
            f"pregrasp_valid={int(pregrasp_valid.sum())}/{len(pregrasp_valid)} "
            f"sim_success={int(sim_successes.sum())}/{len(sim_successes)} "
            f"final={int(successes.sum())}/{len(successes)}",
            flush=True,
        )

        return successes
