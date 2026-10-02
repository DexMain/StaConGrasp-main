import os
import sys

from isaacgym import gymapi, gymtorch
import yaml
import argparse
import numpy as np
import transforms3d
import torch
import xml.etree.ElementTree as ET
from pytorch3d.transforms import matrix_to_euler_angles
from pytorch3d import transforms as pttf
from typing import Union

from utils.util import set_seed
from utils.data_evaluator.data_evaluator import get_evaluator


def _apply_allegro_sim_overrides(args, evaluator_config):
    """Apply optional simulation overrides only to the Allegro evaluator."""
    is_allegro_robot = args.robot_name.startswith("allegro_hand")
    overrides = {}
    trajectory_mode = args.allegro_trajectory_mode
    if args.allegro_waypoint_steps is not None:
        if len(args.allegro_waypoint_steps) != 4 or any(
            int(value) <= 0 for value in args.allegro_waypoint_steps
        ):
            raise ValueError("--allegro_waypoint_steps requires four positive integers")
        evaluator_config["waypoint_steps"] = [
            int(value) for value in args.allegro_waypoint_steps
        ]
    if args.allegro_hover_waypoint_steps is not None:
        if len(args.allegro_hover_waypoint_steps) != 5 or any(
            int(value) <= 0 for value in args.allegro_hover_waypoint_steps
        ):
            raise ValueError(
                "--allegro_hover_waypoint_steps requires five positive integers"
            )
        evaluator_config["waypoint_steps"] = [
            int(value) for value in args.allegro_hover_waypoint_steps
        ]
    if args.allegro_pregrasp_offset is not None:
        evaluator_config["pregrasp_offset"] = float(args.allegro_pregrasp_offset)
    if args.allegro_pregrasp_qpos_mode is not None:
        evaluator_config["pregrasp_qpos_mode"] = str(
            args.allegro_pregrasp_qpos_mode
        )
    if args.allegro_trajectory_mode is not None:
        evaluator_config["trajectory_mode"] = {
            "safe": "allegro_safe",
            "hover": "allegro_hover",
            "legacy": "legacy",
        }[args.allegro_trajectory_mode]
    if args.allegro_hover_lift is not None:
        if float(args.allegro_hover_lift) <= 0.0:
            raise ValueError("--allegro_hover_lift must be positive")
        evaluator_config["hover_lift"] = float(args.allegro_hover_lift)
    if trajectory_mode == "hover" and args.allegro_hover_waypoint_steps is None:
        base_steps = evaluator_config.get("waypoint_steps", [10, 20, 40, 80])
        if len(base_steps) == 4:
            evaluator_config["waypoint_steps"] = [
                int(base_steps[0]),
                int(base_steps[1]),
                int(base_steps[2]),
                int(base_steps[2]),
                int(base_steps[3]),
            ]
    if args.allegro_squeeze_delta is not None:
        evaluator_config["squeeze_delta"] = float(args.allegro_squeeze_delta)
    if getattr(args, "allegro_squeeze_mode", None) is not None:
        evaluator_config["squeeze_mode"] = str(args.allegro_squeeze_mode)
    if getattr(args, "allegro_squeeze_flex_delta", None) is not None:
        evaluator_config["squeeze_flex_delta"] = float(
            args.allegro_squeeze_flex_delta
        )
    if getattr(args, "allegro_squeeze_flex_thumb_delta", None) is not None:
        evaluator_config["squeeze_flex_thumb_delta"] = float(
            args.allegro_squeeze_flex_thumb_delta
        )
    if args.allegro_approach_lift is not None:
        evaluator_config["approach_lift"] = float(args.allegro_approach_lift)
    if args.allegro_approach_clearance is not None:
        evaluator_config["approach_clearance"] = float(
            args.allegro_approach_clearance
        )
    if args.allegro_pregrasp_z_offset is not None:
        evaluator_config["pregrasp_z_offset"] = float(
            args.allegro_pregrasp_z_offset
        )
    if args.allegro_debug_state:
        evaluator_config["debug_state"] = True
    if args.allegro_debug_contacts:
        evaluator_config["debug_contacts"] = True

    for name in ("stiffness", "damping", "effort"):
        value = getattr(args, f"allegro_drive_{name}")
        if value is not None:
            overrides[name] = float(value)
    if args.allegro_hand_friction is not None:
        evaluator_config["simulator_friction_override"] = float(
            args.allegro_hand_friction
        )
    if overrides:
        evaluator_config["simulator_dof_props_overrides"] = overrides

    if not is_allegro_robot and (
        args.allegro_waypoint_steps is not None
        or args.allegro_hover_waypoint_steps is not None
        or args.allegro_pregrasp_offset is not None
        or args.allegro_pregrasp_qpos_mode is not None
        or args.allegro_trajectory_mode is not None
        or args.allegro_hover_lift is not None
        or args.allegro_squeeze_delta is not None
        or args.allegro_approach_lift is not None
        or args.allegro_approach_clearance is not None
        or args.allegro_pregrasp_z_offset is not None
        or overrides
        or args.allegro_hand_friction is not None
        or args.allegro_debug_contacts
    ):
        raise ValueError(
            "Allegro simulation overrides require an Allegro --robot_name; "
            "Leap configuration was not changed"
        )
    if is_allegro_robot:
        print(
            "[SimulationEvaluator] Allegro trajectory config "
            f"waypoint_steps={evaluator_config.get('waypoint_steps')} "
            f"pregrasp_offset={evaluator_config.get('pregrasp_offset')} "
            f"pregrasp_qpos_mode={evaluator_config.get('pregrasp_qpos_mode', 'relative')} "
            f"pregrasp_open_mode={evaluator_config.get('pregrasp_open_mode')} "
            f"trajectory_mode={evaluator_config.get('trajectory_mode')} "
            f"squeeze_mode={evaluator_config.get('squeeze_mode')} "
            f"squeeze_flex_delta={evaluator_config.get('squeeze_flex_delta')} "
            f"squeeze_flex_thumb_delta={evaluator_config.get('squeeze_flex_thumb_delta')} "
            f"allegro_reinforce_thumb={evaluator_config.get('allegro_reinforce_thumb')} "
            f"allegro_thumb_extra_flex={evaluator_config.get('allegro_thumb_extra_flex')} "
            f"squeeze_delta={evaluator_config.get('squeeze_delta')} "
            f"pregrasp_z_offset={evaluator_config.get('pregrasp_z_offset', 0.0)} "
            f"approach_lift={evaluator_config.get('approach_lift')} "
            f"approach_clearance={evaluator_config.get('approach_clearance')} "
            f"hover_lift={evaluator_config.get('hover_lift')} "
            f"drive_overrides={overrides or {}}",
            flush=True,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_list', type=str, nargs='*', 
        default=[
            'experiments/dex_ours/ckpt/ckpt_50000.pth', 
            ])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--robot_name', type=str,
        default='leap_hand',
        choices=['leap_hand', 'allegro_hand', 'allegro_hand_ros_v5_right_A'])
    parser.add_argument('--scene_id', type=str, 
        default='scene_0100')
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--evaluator', type=str, default='SimulationEvaluator', 
        choices=['SimulationEvaluator'])
    parser.add_argument(
        '--evaluator_config',
        type=str,
        default='',
        help=(
            'Optional path to SimulationEvaluator yaml. '
            'Empty = configs/data_evaluator/<robot_name>/<evaluator>.yaml'
        ),
    )
    parser.add_argument('--headless', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--split', type=str, default='')
    parser.add_argument('--dataset', type=str,
        default='graspnet', choices=['graspnet', 'acronym', 'combined'])
    parser.add_argument('--strategy', type=str,
        default='ours', choices=['ours', 'top10', 'graspness','logprob','random'])
    parser.add_argument('--mesh_root', type=str, default='/data/meshdata',
        help='GraspNet 物体 mesh/URDF 根目录（须含 <obj_id>/nontextured_simplified.urdf 与 surface_points_1000.npy）')
    parser.add_argument(
        '--result_subdir',
        type=str,
        default='results_sdf_hybrid_v8_ann_meta',
        help='与 predict 输出目录一致，如 results_sdf_hybrid_v7_ann_meta',
    )
    parser.add_argument(
        '--result_exp_root',
        type=str,
        default='',
        help='grasps.npz / sim_success.npy 根目录；默认取 ckpt 的上两级目录',
    )
    parser.add_argument(
        '--grasp_result_subdir',
        type=str,
        default='',
        help='输入 grasps.npz 所在目录；为空时使用 result_subdir',
    )
    parser.add_argument(
        '--allegro_waypoint_steps',
        type=int,
        nargs=4,
        default=None,
        metavar=('PRE', 'COVER', 'GRASP', 'LIFT'),
        help='仅 Allegro 生效：覆盖四段仿真 waypoint 步数',
    )
    parser.add_argument(
        '--allegro_hover_waypoint_steps',
        type=int,
        nargs=5,
        default=None,
        metavar=('PRE', 'COVER', 'HOVER', 'CLOSE', 'LIFT'),
        help='仅 Allegro hover 生效：覆盖五段仿真 waypoint 步数',
    )
    parser.add_argument(
        '--allegro_pregrasp_offset',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 pregrasp 沿手部 x 轴的退让距离（米）',
    )
    parser.add_argument(
        '--allegro_pregrasp_qpos_mode',
        choices=['canonical', 'relative'],
        default=None,
        help='仅 Allegro 生效：选择 canonical 或相对最终姿态的预抓取手指姿态',
    )
    parser.add_argument(
        '--allegro_trajectory_mode',
        choices=['safe', 'hover', 'legacy'],
        default=None,
        help='仅 Allegro 生效：选择 safe、hover 或 Leap 使用的 legacy 轨迹',
    )
    parser.add_argument(
        '--allegro_hover_lift',
        type=float,
        default=None,
        help='仅 Allegro hover 生效：闭合阶段相对最终 pose 的世界 z 抬升距离（米）',
    )
    parser.add_argument(
        '--allegro_debug_state',
        type=int,
        default=0,
        help='仅 Allegro 生效：打印第一条真实抓取环境的目标/实际 qpos 和指节位置',
    )
    parser.add_argument(
        '--allegro_debug_contacts',
        type=int,
        default=0,
        help='仅 Allegro 生效：打印第一条真实抓取环境的刚体接触对和法向冲量',
    )
    parser.add_argument(
        '--allegro_allow_model_mismatch',
        type=int,
        default=0,
        help=(
            '仅 ROS-V5 Allegro 诊断：允许读取没有 ROS-V5 target URDF '
            '元数据的旧 grasps.npz；默认拒绝模型不匹配输入'
        ),
    )
    parser.add_argument(
        '--allegro_only_object',
        type=str,
        default='',
        help='仅 Allegro 诊断：只在场景中保留指定三位物体编号，例如 029',
    )
    parser.add_argument(
        '--allegro_squeeze_delta',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 tip-mode squeeze 的指尖收紧距离；0 表示不额外 squeeze',
    )
    parser.add_argument(
        '--allegro_squeeze_mode',
        choices=['tip', 'flex'],
        default=None,
        help='仅 Allegro：tip=width_mapper 法向 IK；flex=直接增加屈曲关节（推荐 ROS-V5）',
    )
    parser.add_argument(
        '--allegro_squeeze_flex_delta',
        type=float,
        default=None,
        help='仅 Allegro + flex mode：index/middle/ring joint_1/2 增加的弧度',
    )
    parser.add_argument(
        '--allegro_squeeze_flex_thumb_delta',
        type=float,
        default=None,
        help='仅 Allegro + flex mode：thumb joint_1/2/3 增加的弧度',
    )
    parser.add_argument(
        '--allegro_approach_lift',
        type=float,
        default=None,
        help='仅 Allegro 生效：预抓取/cover 相对最终抓取位姿沿世界 z 轴抬高的最小距离（米）',
    )
    parser.add_argument(
        '--allegro_approach_clearance',
        type=float,
        default=None,
        help='仅 Allegro 生效：预抓取全手碰撞网格相对桌面的最小安全间隙（米）',
    )
    parser.add_argument(
        '--allegro_pregrasp_z_offset',
        type=float,
        default=None,
        help=(
            '仅 Allegro 生效：预抓取/cover 相对最终抓取位姿沿世界 z '
            '轴偏移（米）；负值降低初始开手，桌面 clearance 自动兜底'
        ),
    )
    parser.add_argument(
        '--allegro_drive_stiffness',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 position drive stiffness',
    )
    parser.add_argument(
        '--allegro_drive_damping',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 position drive damping',
    )
    parser.add_argument(
        '--allegro_drive_effort',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 position drive effort',
    )
    parser.add_argument(
        '--allegro_hand_friction',
        type=float,
        default=None,
        help='仅 Allegro 生效：覆盖 Allegro 手部摩擦系数',
    )
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # load scene annotation
    if args.dataset == 'graspnet':
        scene_path = os.path.join('/data/scenes', args.scene_id)
        extrinsics_path = os.path.join(scene_path, 'realsense/cam0_wrt_table.npy')
        extrinsics = np.load(extrinsics_path)
        annotation_path = os.path.join(scene_path, 'realsense/annotations/0000.xml')
        annotation = ET.parse(annotation_path)
    
        # parse scene annotation
        object_pose_dict = {}
        for obj in annotation.findall('obj'):
            object_code = str(int(obj.find('obj_id').text)).zfill(3)
            translation = np.array([float(x) for x in obj.find('pos_in_world').text.split()])
            rotation = np.array([float(x) for x in obj.find('ori_in_world').text.split()])
            rotation = transforms3d.quaternions.quat2mat(rotation)
            object_pose = np.eye(4)
            object_pose[:3, :3] = rotation
            object_pose[:3, 3] = translation
            object_pose = extrinsics @ object_pose
            object_pose_dict[object_code] = object_pose
        
        # load object surface points
        object_surface_points_dict = {}
        for object_code in object_pose_dict:
            object_surface_points_path = os.path.join(
                args.mesh_root, object_code, f'surface_points_1000.npy')
            object_surface_points = np.load(object_surface_points_path)
            object_pose = object_pose_dict[object_code]
            object_surface_points = object_surface_points @ object_pose[:3, :3].T + object_pose[:3, 3]
            object_surface_points_dict[object_code] = object_surface_points
    elif args.dataset == 'acronym':
        split = args.scene_id.split('_')[1]
        root = f'/data/acronym_test_scenes/test_acronym_{split}'
        annotation_path = os.path.join(root,args.scene_id+'.npz')
        annotation = np.load(annotation_path,allow_pickle=True)['arr_0'][None][0]
        
        # parse scene annotation
        object_pose_dict = {}
        for obj in list(annotation.keys()):
            object_code = obj
            translation = annotation[object_code]['rest_pose_trans']
            translation[..., 2] -= 0.05
            rotation_quat = annotation[object_code]['rest_pose_quat']
            rotation_mat = pttf.quaternion_to_matrix(torch.tensor(rotation_quat[[3, 0, 1, 2]])).numpy()
            object_pose = np.eye(4)
            object_pose[:3, :3] = rotation_mat
            object_pose[:3, 3] = translation
            object_pose_dict[object_code] = object_pose
            
         # load object surface points
        object_surface_points_dict = {}
        for object_code in object_pose_dict:
            object_surface_points_path = os.path.join('/data/acronym/meshes/pc', object_code, f'surface_points_1000.npy')
            object_surface_points = np.load(object_surface_points_path)
            object_pose = object_pose_dict[object_code]
            object_surface_points = object_surface_points @ object_pose[:3, :3].T + object_pose[:3, 3]
            object_surface_points_dict[object_code] = object_surface_points

    if args.allegro_only_object:
        if not args.robot_name.startswith('allegro_hand'):
            raise ValueError(
                '--allegro_only_object is an Allegro-only diagnostic option'
            )
        target_code = str(args.allegro_only_object).zfill(3)
        if target_code not in object_pose_dict:
            raise ValueError(
                f'--allegro_only_object={target_code} is not present in '
                f'{args.scene_id}; available={sorted(object_pose_dict)}'
            )
        object_pose_dict = {target_code: object_pose_dict[target_code]}
        object_surface_points_dict = {
            target_code: object_surface_points_dict[target_code]
        }
        print(
            f'[SimulationEvaluator] Allegro target-only diagnostic '
            f'object={target_code}',
            flush=True,
        )
    # create data evaluator
    if str(getattr(args, "evaluator_config", "") or "").strip():
        evaluator_config_path = str(args.evaluator_config).strip()
    else:
        evaluator_config_path = os.path.join(
            'configs/data_evaluator', args.robot_name, f'{args.evaluator}.yaml')
    print(f'[SimulationEvaluator] config={evaluator_config_path}', flush=True)
    evaluator_config = yaml.safe_load(open(evaluator_config_path, 'r'))
    evaluator_config['headless'] = args.headless
    evaluator_config['mesh_root'] = args.mesh_root
    # Needed for per-scene eval overrides (e.g. fixed finger_bias whitelist).
    evaluator_config['scene_id'] = str(args.scene_id)
    _apply_allegro_sim_overrides(args, evaluator_config)
    evaluator_class = get_evaluator(args.evaluator)
    data_evaluator = evaluator_class(evaluator_config, args.device)
    
    # set environments
    data_evaluator.set_environments(object_pose_dict, object_surface_points_dict, args.batch_size + 1, dataset=args.dataset)
    
    # evaluate networks
    for ckpt_path in args.ckpt_path_list:
        # Prefer an explicit result root so final experiments can live outside ckpt dirs.
        exp_root = (
            args.result_exp_root
            if args.result_exp_root
            else os.path.dirname(os.path.dirname(ckpt_path))
        )
        save_path = os.path.join(
            exp_root, args.result_subdir, args.scene_id, 'sim_success.npy'
        )
        grasp_result_subdir = args.grasp_result_subdir or args.result_subdir
        load_path = os.path.join(
            exp_root, grasp_result_subdir, args.scene_id, 'grasps.npz'
        )
        if os.path.exists(save_path) and not args.overwrite:
            continue

        # Prediction metadata contains object/string arrays such as the
        # target Allegro URDF path.  These are local experiment artifacts
        # written by the predictor, so allow_pickle is required here.
        grasps = np.load(load_path, allow_pickle=True)
        if args.robot_name.startswith('allegro_hand_ros_v5_right_A'):
            target_paths = getattr(grasps, 'files', [])
            if 'allegro_target_urdf_path' not in target_paths:
                message = (
                    '[SimulationEvaluator] input grasps.npz has no '
                    'allegro_target_urdf_path metadata; it may have been '
                    'generated for the simplified Allegro URDF. Regenerate '
                    'with --urdf_path '
                    'robot_models/urdf/allegro_hand_ros_v5_right_A/'
                    'allegro_hand_ros_v5_right_A.urdf.'
                )
                if not int(args.allegro_allow_model_mismatch):
                    raise ValueError(
                        message
                        + ' Use --allegro_allow_model_mismatch 1 only for '
                        'an intentional legacy compatibility check.'
                    )
                print('[SimulationEvaluator][warning] ' + message, flush=True)
            else:
                target_path = str(grasps['allegro_target_urdf_path'][0])
                if 'allegro_hand_ros_v5_right_A' not in target_path:
                    message = (
                        '[SimulationEvaluator] grasp target URDF '
                        f'is {target_path!r}, but evaluator robot is '
                        'allegro_hand_ros_v5_right_A; qpos/pose may be '
                        'kinematically incompatible.'
                    )
                    if not int(args.allegro_allow_model_mismatch):
                        raise ValueError(
                            message
                            + ' Regenerate the grasp file with the ROS-V5 '
                            'URDF or explicitly pass '
                            '--allegro_allow_model_mismatch 1.'
                        )
                    print('[SimulationEvaluator][warning] ' + message, flush=True)
        joint_names = data_evaluator._robot_info['dof_names'][6:]
        eval_keys = ['translation', 'rotation'] + list(joint_names)
        # Keep stage-1 coarse approach fields for Allegro (open-hand pregrasp /
        # adaptive descend bias). Without these, coarse_* is silently dropped.
        if 'coarse_translation' in grasps.files and 'coarse_rotation' in grasps.files:
            eval_keys.extend(['coarse_translation', 'coarse_rotation'])
            for joint_name in joint_names:
                coarse_key = f'coarse_{joint_name}'
                if coarse_key in grasps.files:
                    eval_keys.append(coarse_key)
            print(
                '[SimulationEvaluator] passing coarse_* fields into evaluator '
                f'({sum(1 for k in eval_keys if k.startswith("coarse_"))} keys)',
                flush=True,
            )

        # evaluate grasps by batch
        successes = []
        for i in range(0, len(grasps['translation']), args.batch_size):
            end = min(i + args.batch_size, len(grasps['translation']))
            grasps_batch = {
                key: grasps[key][i:end] for key in eval_keys if key in grasps.files
            }
            # pad the first grasp to avoid isaac gym bug
            grasps_batch = { joint: np.concatenate([grasps_batch[joint][:1], grasps_batch[joint]]) 
                for joint in grasps_batch }
            successes_batch = data_evaluator.evaluate_data(grasps_batch)
            successes.append(successes_batch[1:])
        
        # save results
        successes = np.concatenate(successes)
        print(np.mean(successes))
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        np.save(save_path, successes)
