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
    # width/height  = do phan giai CAMERA segmentation (khop collector: 480x384).
    # obs_width/obs_height = do phan giai OBSERVATION dua vao mang, sau khi
    #   resize_class_map() ha mau. Phai khop IMAGE_WIDTH/IMAGE_HEIGHT cua
    #   train_il.ipynb (240x192) thi actor warm-start moi nhin thay dung thang do
    #   dac trung nhu luc train IL. Ha o day cung giam 4x bo nho rollout/replay
    #   (SAC 50k transition: 9.2GB o 480x384 -> 2.2GB o 240x192).
    "width": 480, "height": 384, "obs_width": 240, "obs_height": 192,
    "fov": 90.0, "fps": 20.0,
    "camera_x": 1.5, "camera_y": 0.0, "camera_z": 2.4, "camera_pitch": -5.0,
    "vehicle_filter": "vehicle.lincoln.mkz2017",
    # fps 20 (0.05s/tick) x action_repeat 4 = 0.2s moi QUYET DINH cua policy = dung
    # `control_dt` ma checkpoint IL da train (5 FPS). Khong the thay bang fps=5 truc tiep:
    # CARLA khuyen cao fixed_delta_seconds <= 0.05s, tren muc do vat ly bat dau sai (xe
    # rung, va cham gia). CarlaLaneKeepEnv._check_control_rate() canh bao neu hai ben lech.
    "action_repeat": 4,
    # Tinh theo QUYET DINH (khong phai tick): 500 x 0.2s = 100s moi episode.
    "max_episode_steps": 500,
    # 10 quyet dinh = 2s lien tuc ngoai lan thi ket thuc episode (truoc day 20 tick = 1s).
    "off_lane_patience_steps": 10,
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
        # LR TACH RIENG actor/critic + `critic_warmup_updates`: xem docstring
        # ppo/ppo_agent.py.__init__ va §12 cua behavior_cloning/train_il_v4.ipynb.
        "actor_lr": 2e-5, "critic_lr": 3e-4, "critic_warmup_updates": 10,
        # [steer, longitudinal]. std(steer) = e^-3 = 0.05, du de tham do quanh mot lenh lai
        # co bien do dien hinh 0.005-0.03 ma khong lang xe ra khoi lan ngay rollout dau.
        "log_std_init": (-3.0, -1.5),
        "gae_lambda": 0.95, "clip_range": 0.2,
        "value_clip_range": 0.2, "entropy_coef": 0.0, "value_coef": 0.5,
        "max_grad_norm": 0.5, "epochs": 10, "batch_size": 128, "target_kl": 0.02,
        # total_steps dem QUYET DINH: 200k x 0.2s = 11 gio mo phong. Con so cu (2 trieu)
        # duoc dat khi 1 step = 1 tick 0.1s; giu nguyen se thanh 111 gio mo phong.
        "total_steps": 200000, "n_steps": 1024,
        "save_every_updates": 5, "eval_every_updates": 10, "eval_episodes": 3,
    },
    "sac": {
        "output": "./runs/sac_lane_keep",
        "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
        "tau": 0.005,
        # -action_dim = -2.0 (Haarnoja et al.) la mac dinh cho tac vu dieu khien tong quat.
        # O day no qua CAO: entropy muc do dat buoc alpha giu do lech lon tren CHIEU STEER,
        # trong khi lenh lai dien hinh chi 0.005-0.03. Ha xuong -4.0 cho phep policy nhon
        # hon ma van con tham do o chieu longitudinal.
        "target_entropy": -4.0,
        "log_std_init": -2.5,
        "batch_size": 128, "buffer_capacity": 50000,
        "learning_starts": 2000, "train_freq": 1, "gradient_steps": 1,
        "max_grad_norm": 0.5,
        # Cung ly do doi don vi nhu PPO: 100k quyet dinh x 0.2s = 5.6 gio mo phong.
        "total_steps": 100000, "save_every_steps": 5000,
        "eval_every_steps": 10000, "eval_episodes": 3,
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
    parser.add_argument("--action-repeat", type=int, default=None,
                         help="So tick vat ly moi quyet dinh cua policy. fps/action_repeat "
                              "phai bang 1/control_dt cua checkpoint IL (mac dinh 20/4 = 5 Hz)")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                         help="So QUYET DINH toi da moi episode (khong phai so tick)")
    parser.add_argument("--n-steps", type=int, default=None, help="[train_ppo.py] so buoc moi rollout/update")
    parser.add_argument("--buffer-capacity", type=int, default=None,
                         help="[train_sac.py] so transition toi da trong replay buffer — "
                              "giam gia tri nay truoc tien neu thieu RAM (xem README.md)")
    parser.add_argument("--width", type=int, default=None,
                        help="Do rong CAMERA segmentation (px) — khong phai observation")
    parser.add_argument("--height", type=int, default=None,
                        help="Do cao CAMERA segmentation (px) — khong phai observation")
    parser.add_argument("--obs-width", type=int, default=None,
                        help="Do rong OBSERVATION dua vao mang (mac dinh 240 = khop IL)")
    parser.add_argument("--obs-height", type=int, default=None,
                        help="Do cao OBSERVATION dua vao mang (mac dinh 192 = khop IL)")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Kich thuoc minibatch update — dat truc tiep VRAM can dung "
                              "(anh one-hot 4 lop o 240x192 nang hon nhieu 160x128, giam "
                              "gia tri nay truoc tien neu OOM, xem README.md)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--episodes", type=int, default=None, help="[evaluate.py] so episode danh gia")
    parser.add_argument("--deterministic", action="store_true",
                         help="[evaluate.py] dung mean action thay vi sample (tat exploration)")
    parser.add_argument("--algorithm", default=None, choices=["ppo", "sac"],
                         help="[evaluate.py] thuat toan cua checkpoint --resume")
    parser.add_argument("--eval-csv-out", default=None,
                         help="[evaluate.py] neu dat, ghi ket qua tung episode ra file CSV nay "
                              "(vd runs/ppo_lane_keep/eval_results.csv) de plot_metrics.py doc lai")
    args = parser.parse_args(argv)

    config = dict(COMMON_DEFAULTS)
    config.update(ALGO_DEFAULTS[algorithm])
    if args.config:
        path = Path(args.config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            config.update(_flatten(json.load(handle)))

    for key in ("host", "port", "il_checkpoint", "output", "total_steps", "n_steps",
                "buffer_capacity", "width", "height", "obs_width", "obs_height",
                "batch_size", "device", "action_repeat", "max_episode_steps"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.warm_start is not None:
        config["warm_start"] = args.warm_start

    config["_resume"] = args.resume
    config["_episodes"] = args.episodes
    config["_deterministic"] = args.deterministic
    config["_algorithm"] = args.algorithm or algorithm
    config["_eval_csv_out"] = args.eval_csv_out
    return config
