"""Shared CLI + JSON config loading for `train_ppo.py` / `train_sac.py` / `evaluate.py`.

Mirrors the `--config` JSON-overrides-defaults pattern used by
`data_collection/carla_collector/config.py`, flattened into one dict since the env/agent
code here just does plain keyword lookups (`cfg.get("w_lane_offset", 1.0)`) instead of
argparse groups.

`COMMON_DEFAULTS` covers everything algorithm-agnostic (CARLA connection, camera, env
dynamics, reward weights, warm-start/IL-checkpoint settings). `ALGO_DEFAULTS["ppo"|"sac"]`
adds the on-policy/off-policy-specific hyperparameters on top, so `ppo_config.json` and
`sac_config.json` only need to *override* what's actually different — see either file.
"""

import argparse
import json
from pathlib import Path

COMMON_DEFAULTS = {
    # connection
    "host": "127.0.0.1", "port": 2000, "timeout": 20.0,
    # camera / env
    "width": 160, "height": 128, "fov": 90.0, "fps": 10.0,
    "camera_x": 1.5, "camera_y": 0.0, "camera_z": 2.4, "camera_pitch": -5.0,
    "vehicle_filter": "vehicle.lincoln.mkz2017",
    "max_episode_steps": 1000, "off_lane_patience_steps": 20,
    "warmup_ticks": 4, "frame_timeout": 5.0, "no_rendering": False, "seed": 42,
    # reward weights (docs/csv_fields_by_task.md — "DRL" section)
    "w_speed": 1.0, "w_lane_offset": 1.0, "w_heading": 0.5,
    "w_steer_delta": 1.0, "w_long_delta": 0.5, "w_yaw_rate": 0.1,
    "off_lane_penalty": 5.0, "collision_penalty": 50.0, "lane_invasion_penalty": 1.0,
    # shared training settings
    "il_checkpoint": "../behavior_cloning/best_il_model.pth",
    "warm_start": True, "device": "cuda", "gamma": 0.99,
}

ALGO_DEFAULTS = {
    "ppo": {
        "output": "./runs/ppo_lane_keep",
        "learning_rate": 3e-4, "gae_lambda": 0.95, "clip_range": 0.2,
        "value_clip_range": 0.2, "entropy_coef": 0.0, "value_coef": 0.5,
        "max_grad_norm": 0.5, "epochs": 10, "batch_size": 256, "target_kl": 0.02,
        "total_steps": 2000000, "n_steps": 2048,
        "save_every_updates": 5, "eval_every_updates": 10, "eval_episodes": 3,
    },
    "sac": {
        "output": "./runs/sac_lane_keep",
        "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
        "tau": 0.005, "target_entropy": None,  # None -> auto = -action_dim (Haarnoja et al.)
        "batch_size": 256, "buffer_capacity": 100000,
        "learning_starts": 5000, "train_freq": 1, "gradient_steps": 1,
        "max_grad_norm": 0.5,
        "total_steps": 500000, "save_every_steps": 10000,
        "eval_every_steps": 20000, "eval_episodes": 3,
    },
}


def _flatten(data, output=None):
    output = {} if output is None else output
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten(value, output)
        else:
            output[key] = value
    return output


def peek_algorithm(argv=None, default="ppo"):
    """Lightweight pre-parse used by `evaluate.py`, which doesn't know which algorithm's
    defaults to load until it has read `--algorithm` off the command line — same
    two-pass-parse trick as `data_collection/carla_collector/config.py`'s `--config`
    pre-parser."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--algorithm", default=default, choices=list(ALGO_DEFAULTS))
    known, _ = pre_parser.parse_known_args(argv)
    return known.algorithm


def load_config(algorithm, argv=None):
    if algorithm not in ALGO_DEFAULTS:
        raise ValueError("algorithm phai la %s, nhan duoc '%s'." % (list(ALGO_DEFAULTS), algorithm))

    parser = argparse.ArgumentParser(description="DRL fine-tuning (%s) cho lane-keeping tren CARLA" % algorithm.upper())
    parser.add_argument("--config", default=None, help="File JSON cau hinh, vd %s_config.json" % algorithm)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--il-checkpoint", default=None, help="Checkpoint IL (.pth) de warm-start actor")
    parser.add_argument("--output", default=None, help="Thu muc luu checkpoint + log CSV")
    parser.add_argument("--resume", default=None, help="Checkpoint DRL (.pt) de resume train / dung de eval")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false", default=None,
                         help="Bo qua warm-start IL — actor khoi tao ngau nhien (chi de doi chung)")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None, help="[train_ppo.py] so buoc moi rollout/update")
    parser.add_argument("--buffer-capacity", type=int, default=None,
                         help="[train_sac.py] so transition toi da trong replay buffer — "
                              "giam gia tri nay truoc tien neu thieu RAM (xem README.md)")
    parser.add_argument("--width", type=int, default=None, help="Do rong camera/observation (px)")
    parser.add_argument("--height", type=int, default=None, help="Do cao camera/observation (px)")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Kich thuoc minibatch update — dat truc tiep VRAM can dung "
                              "(anh one-hot 13 lop o 480x384 nang hon nhieu 160x128, giam "
                              "gia tri nay truoc tien neu OOM, xem README.md)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--episodes", type=int, default=None, help="[evaluate.py] so episode danh gia")
    parser.add_argument("--deterministic", action="store_true",
                         help="[evaluate.py] dung mean action thay vi sample (tat exploration)")
    parser.add_argument("--algorithm", default=None, choices=["ppo", "sac"],
                         help="[evaluate.py] thuat toan cua checkpoint --resume")
    args = parser.parse_args(argv)

    config = dict(COMMON_DEFAULTS)
    config.update(ALGO_DEFAULTS[algorithm])
    if args.config:
        path = Path(args.config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            config.update(_flatten(json.load(handle)))

    for key in ("host", "port", "il_checkpoint", "output", "total_steps", "n_steps",
                "buffer_capacity", "width", "height", "batch_size", "device"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.warm_start is not None:
        config["warm_start"] = args.warm_start

    config["_resume"] = args.resume
    config["_episodes"] = args.episodes
    config["_deterministic"] = args.deterministic
    config["_algorithm"] = args.algorithm or algorithm
    return config
