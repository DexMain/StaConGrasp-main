import os
import yaml
from os.path import dirname, join

# use this so that one can use config.x.y.z instead of config['x']['y']['z']
class DotDict(dict):
    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        return None
    
    def __setattr__(self, key, value):
        self[key] = value

def to_dot_dict(dic):
    for k in dic.keys():
        if type(dic[k]) == dict:
            dic[k] = to_dot_dict(dic[k])
    return DotDict(dic)

def to_dict(args):
    result = dict()
    for k, v in args.items():
        if isinstance(v, dict):
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result

def add_argparse(parser, arg_mapping):
    for raw_key, (_, arg_type, default) in arg_mapping:
        parser.add_argument('--' + raw_key, type=arg_type, default=default)
    return parser

# combine config from yaml file and argument
# priority: args in console > default args > yaml file
def load_config(yaml_file, arg_mapping=None, args=None):
    with open(yaml_file, 'r') as f:
        dic = yaml.load(f, Loader=yaml.FullLoader)
    if args is not None:
        for raw_key, (new_key, _, _) in arg_mapping:
            value = eval(f'args.{raw_key}')
            if value is None:
                continue
            temp = dic
            for k in new_key.split('/')[:-1]:
                if not k in temp.keys():
                    temp[k] = dict()
                elif not type(temp[k]) == dict:
                    raise ValueError
                temp = temp[k]
            temp[new_key.split('/')[-1]] = value
    return to_dot_dict(dic)

def merge_inference_model_config(config, source_config):
    """
    微调实验 config 可能误设推理字段（如 diffusion.log_prob_type=null）。
    从源预训练 config 补回影响 sample/score 的字段。
    """
    src = source_config.model
    dst = config.model
    if getattr(dst.diffusion, "log_prob_type", None) in (None, "null") and getattr(
        src.diffusion, "log_prob_type", None
    ):
        dst.diffusion.log_prob_type = src.diffusion.log_prob_type
        print(
            "[ckpt_to_config] patched model.diffusion.log_prob_type="
            f"{dst.diffusion.log_prob_type!r} from source ckpt config"
        )
    for key in ("num_inference_timesteps", "ode", "clip_sample", "scheduler_type"):
        src_val = getattr(src.diffusion, key, None)
        dst_val = getattr(dst.diffusion, key, None)
        if src_val is not None and dst_val != src_val:
            dst.diffusion[key] = src_val
            print(f"[ckpt_to_config] patched model.diffusion.{key}={src_val!r}")
    for key in ("trans_scale", "joint_scale", "joint_num", "dist_joint", "type"):
        src_val = getattr(src, key, None)
        dst_val = getattr(dst, key, None)
        if src_val is not None and dst_val != src_val:
            dst[key] = src_val
            print(f"[ckpt_to_config] patched model.{key}={src_val!r}")
    return config


def ckpt_to_config(ckpt_path, fallback_yaml=None):
    """
    从 ckpt 路径推断实验 config.yaml：{exp_root}/config.yaml。
    微调实验若缺少 config，回退到 fallback_yaml 或 train_dex_ours.yaml。
    """
    exp_root = dirname(dirname(ckpt_path))
    config_path = join(exp_root, "config.yaml")
    if os.path.isfile(config_path):
        config = load_config(config_path)
        source_ckpt = getattr(config, "ckpt", None)
        if source_ckpt and os.path.isfile(source_ckpt):
            try:
                source_config = ckpt_to_config(source_ckpt, fallback_yaml=fallback_yaml)
                config = merge_inference_model_config(config, source_config)
            except FileNotFoundError:
                pass
        return config

    candidates = []
    if fallback_yaml:
        candidates.append(fallback_yaml)
    candidates.extend(
        [
            join(exp_root, "config.yaml"),
            join("configs", "network", "train_dex_ours.yaml"),
            "/data/DexGraspNet2.0-ckpts/OURS/config.yaml",
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            print(
                f"[ckpt_to_config] warning: missing {config_path}, "
                f"fallback to {path}"
            )
            return load_config(path)

    raise FileNotFoundError(
        f"Cannot find config for ckpt {ckpt_path}. "
        f"Tried {config_path} and fallbacks {candidates}. "
        f"Pass --config_yaml explicitly."
    )