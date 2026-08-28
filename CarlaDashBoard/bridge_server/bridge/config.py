"""Bridge Server configuration — one dataclass, overridable from CLI flags in run_server.py.

Camera defaults mirror data_collection/carla_collector/config.py so the live preview looks
the same as what the offline collector records (width/height/fov/pitch/mount point).
"""

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    # --- Bridge Server networking (what the WPF client connects to) ---
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # --- CARLA connection ---
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    carla_timeout_s: float = 20.0
    sim_fps: float = 20.0                    # world.tick() rate in synchronous mode

    # --- World / vehicle ---
    town: str = ""                            # "" = keep whatever world is already loaded
    vehicle_filter: str = "vehicle.lincoln.mkz2017"
    role_name: str = "hero"

    # --- Camera (mirrors carla_collector/config.py defaults) ---
    camera_width: int = 800
    camera_height: int = 450
    # Segmentation camera resolution — deliberately NOT camera_width/height. This camera is
    # not just a preview: its raw class-id buffer is the IL/DRL policy's only image input
    # (see carla_session._on_seg_frame -> modes/learned_autopilot.py). The IL checkpoint was
    # trained on 480x384 at fov 90, i.e. a 5:4 frame whose vertical FOV is ~77 deg. Feeding
    # it a 16:9 frame (800x450, vertical FOV ~59 deg) and letting resize_class_map() squash
    # that to 240x192 changes where the horizon and the lane lines land in the image —
    # geometrically a different camera, with no error anywhere to say so. Keep this equal to
    # the collector's camera.width/height (data_collection/collector_config.json).
    seg_width: int = 480
    seg_height: int = 384
    camera_fov: float = 90.0
    camera_x: float = 1.5
    camera_y: float = 0.0
    camera_z: float = 2.4
    camera_pitch: float = -5.0

    # --- Streaming (independent of sim_fps — this is what goes out over /stream) ---
    publish_fps: float = 15.0
    jpeg_quality: int = 75

    # --- Data Collection mode (lightweight live recorder, see modes/data_collection.py) ---
    record_output_dir: str = "dataset_live"

    # --- A* Autopilot / route planning (Phase 3 — wraps Deep_RL_Carla/router_plan) ---
    deep_rl_carla_root: str = ""          # "" = auto-detect as ../../Deep_RL_Carla (see route_planning.py)
    astar_graph_resolution_m: float = 2.0
    astar_lane_change_cost: float = 3.0
    astar_route_tolerance_m: float = 3.0
    astar_target_speed_kmh: float = 30.0

    # --- IL / DRL Autopilot (Phase 4 — wraps Deep_RL_Carla/drl_training) ---
    il_checkpoint_path: str = ""      # "" = {deep_rl_carla_root}/behavior_cloning/best_il_model.pth
    drl_checkpoint_path: str = ""     # "" = {deep_rl_carla_root}/drl_training/runs/{algo}_lane_keep/{algo}_latest.pt
    drl_algorithm: str = "ppo"        # "ppo" | "sac" — overridden if the checkpoint itself says otherwise
    learned_autopilot_device: str = "cuda"   # falls back to "cpu" automatically if CUDA isn't available


WEATHER_PRESETS = [
    # (preset name, group) — group matches the design doc's §05 grouping.
    ("ClearNoon", "Quang đãng"), ("ClearSunset", "Quang đãng"),
    ("CloudyNoon", "Quang đãng"), ("CloudySunset", "Quang đãng"),
    ("Default", "Quang đãng"),
    ("WetNoon", "Ướt không mưa"), ("WetSunset", "Ướt không mưa"),
    ("WetCloudyNoon", "Ướt không mưa"), ("WetCloudySunset", "Ướt không mưa"),
    ("SoftRainNoon", "Mưa nhẹ → to"), ("SoftRainSunset", "Mưa nhẹ → to"),
    ("MidRainyNoon", "Mưa nhẹ → to"), ("MidRainSunset", "Mưa nhẹ → to"),
    ("HardRainNoon", "Mưa nhẹ → to"), ("HardRainSunset", "Mưa nhẹ → to"),
]

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05"]
