#!/usr/bin/env python3
"""Carla Dashboard Bridge Server — entrypoint.

Run with the carla_rl conda env (Python 3.7, has the CARLA 0.9.10 PythonAPI installed):

    C:\\Users\\dloc\\miniconda3\\envs\\carla_rl\\python.exe run_server.py

CARLA (CarlaUE4.exe) must already be running and a world loaded before starting this.
See README.md for the full walkthrough and what each phase (0/1/2) currently covers.
"""

import argparse
import asyncio
import logging

from bridge.config import BridgeConfig
from bridge.server import run


def parse_args():
    parser = argparse.ArgumentParser(description="Carla Dashboard Bridge Server")
    parser.add_argument("--ws-host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--town", default="", help="Vd: Town03 — de trong = giu world hien tai")
    parser.add_argument("--vehicle-filter", default="vehicle.lincoln.mkz2017")
    parser.add_argument("--sim-fps", type=float, default=20.0)
    parser.add_argument("--publish-fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--deep-rl-carla-root", default="",
                         help="Duong dan toi Deep_RL_Carla neu khong nam canh CarlaDashBoard (Phase 3, A*)")
    parser.add_argument("--astar-target-speed-kmh", type=float, default=30.0)
    parser.add_argument("--astar-graph-resolution-m", type=float, default=2.0)
    parser.add_argument("--il-checkpoint-path", default="",
                         help="Checkpoint IL .pth (Phase 4). Mac dinh: "
                              "{deep-rl-carla-root}/behavior_cloning/best_il_model.pth")
    parser.add_argument("--drl-checkpoint-path", default="",
                         help="Checkpoint DRL .pt (Phase 4). Mac dinh: "
                              "{deep-rl-carla-root}/drl_training/runs/best/{algo}_latest.pt")
    parser.add_argument("--drl-algorithm", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--learned-autopilot-device", default="cuda", choices=["cuda", "cpu"],
                         help="Tu dong ve cpu neu khong co CUDA (Phase 4, IL/DRL Autopilot)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    cfg = BridgeConfig(
        ws_host=args.ws_host, ws_port=args.ws_port,
        carla_host=args.carla_host, carla_port=args.carla_port,
        town=args.town, vehicle_filter=args.vehicle_filter,
        sim_fps=args.sim_fps, publish_fps=args.publish_fps, jpeg_quality=args.jpeg_quality,
        deep_rl_carla_root=args.deep_rl_carla_root,
        astar_target_speed_kmh=args.astar_target_speed_kmh,
        astar_graph_resolution_m=args.astar_graph_resolution_m,
        il_checkpoint_path=args.il_checkpoint_path,
        drl_checkpoint_path=args.drl_checkpoint_path,
        drl_algorithm=args.drl_algorithm,
        learned_autopilot_device=args.learned_autopilot_device,
    )
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
