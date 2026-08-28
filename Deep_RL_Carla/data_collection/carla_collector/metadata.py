"""Session metadata serialization."""

import json
import math

import carla

from .geometry import utc_now, waypoint_record
from .schema import ROUTE_FIELDS, SEG_CLASS_COLORS, SEG_CLASS_NAMES


def write_metadata(path, world, world_map, ego, args, session_id,
                   goal_waypoint, map_graph_stats):
    fx = args.width / (2.0 * math.tan(math.radians(args.fov) / 2.0))
    weather = world.get_weather()
    settings = world.get_settings()
    weather_fields = [
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "wetness", "fog_falloff"]
    metadata = {
        "created_utc": utc_now(),
        "schema_version": "2.0",
        "carla_version": getattr(carla, "__version__", "0.9.10"),
        "session_id": session_id,
        "episode_id": getattr(world, "id", ""),
        "map": world_map.name,
        # Chiec ego DAU TIEN cua session. Voi `--rebind-ego` (mac dinh bat), mot
        # session co the di qua nhieu chiec xe khi xe dang thu bi ket: xem
        # summary.json -> ego_segments, va cot `vehicle_id` cua tung dong trong
        # states.csv, de biet dong nao thuoc chiec nao.
        "vehicle_id": ego.id,
        "vehicle_type": ego.type_id,
        "role_name": ego.attributes.get("role_name", ""),
        "ego_rebind": {
            "enabled": bool(getattr(args, "rebind_ego", False)),
            "wait_s": getattr(args, "rebind_wait_s", 0.0),
            "max_rebinds": getattr(args, "max_rebinds", 0),
        },
        # Nguong watchdog cua luc THU: mot session it mau hay dut doan chi doc
        # duoc neu biet luc do collector duoc phep cho bao lau.
        "watchdog": {
            "stall_timeout_s": getattr(args, "stall_timeout_s", 0.0),
            "stationary_timeout_s": getattr(args, "stationary_timeout_s", 0.0),
            "camera_timeout_s": getattr(args, "camera_timeout_s", 0.0),
            "ego_missing_timeout_s": getattr(args, "ego_missing_timeout_s", 0.0),
            "stall_speed_mps": getattr(args, "stall_speed", 0.0),
            "min_wheels": getattr(args, "min_wheels", 0),
        },
        "collector_mode": "passive_async_client_no_world_tick",
        # The collector does not own the world, so it cannot set the timestep. On
        # an asynchronous, variable-timestep world the cameras' `sensor_tick` is
        # not honoured (a 5 FPS session was measured writing 22.7 samples per
        # simulated second), so the rate is enforced client-side by
        # synchronizer.SampleRateLimiter. Record what the world was actually doing
        # - `camera.fps` below is the REQUESTED rate; `summary.json` records the
        # rate that was achieved.
        "world_settings": {
            "synchronous_mode": bool(getattr(settings, "synchronous_mode", False)),
            "fixed_delta_seconds": getattr(settings, "fixed_delta_seconds", None),
            "no_rendering_mode": bool(getattr(settings, "no_rendering_mode", False)),
        },
        "sampling_rate_enforced_by": "client_side_sim_time_decimation",
        # Dedup xoa nguyen mau khoi states.csv, nen `throughput_fps` cua mot session
        # co dedup thap hon fps yeu cau mot cach hop le. Ghi lai nguong o day de doc
        # so lieu session ma khong phai doan xem luc thu da bat gi.
        "dedup": {
            "stationary_speed_mps": args.dedup_stationary_speed,
            "action_eps": args.dedup_action_eps,
            "min_interval_s": args.dedup_min_interval_s,
            "enabled": args.dedup_stationary_speed > 0.0,
        },
        "modalities": {
            "semantic_label": True,
            "rgb": args.image_mode == "seg-rgb",
            "semantic_color_preview": args.save_seg_color,
        },
        "camera": {
            "width": args.width, "height": args.height,
            "fov_deg": args.fov, "fps": args.fps,
            "transform": {
                "x": args.camera_x, "y": args.camera_y,
                "z": args.camera_z, "pitch": args.camera_pitch},
            "intrinsics": {
                "fx": fx, "fy": fx, "cx": args.width / 2.0,
                "cy": args.height / 2.0},
        },
        "weather": {name: getattr(weather, name, None) for name in weather_fields},
        "semantic_class_names": list(SEG_CLASS_NAMES),
        "semantic_colors": {str(k): v for k, v in SEG_CLASS_COLORS.items()},
        "lookahead_m": args.lookahead_m,
        "route_lookaheads_m": args.route_lookaheads,
        "goal": waypoint_record(goal_waypoint) if goal_waypoint is not None else None,
        "map_graph": map_graph_stats,
        "route_fields": {
            "status": "reserved_until_astar_planner_is_connected",
            "fields": ROUTE_FIELDS,
        },
        "training_contract": {
            "imitation_observation": [
                "seg_label_or_seg_color", "speed_mps", "yaw_rate_rps",
                "previous_steer", "previous_longitudinal",
                "traffic_light_state"],
            "action": ["steer", "longitudinal"],
            "reward_or_metrics_only": [
                "normalized_lane_offset", "heading_error_rad", "off_lane",
                "steer_delta", "longitudinal_delta"],
        },
        "notes": (
            "seg_label stores RAW CARLA class IDs (0-22) and is the training input; the "
            "4-class lane-keeping scheme (semantic_class_names) is applied by the training "
            "pipelines via schema.RAW_TO_TRAIN_LANE_LUT, so a different scheme can be "
            "re-derived from this dataset without recollecting. seg_color is a preview "
            "already folded down to those 4 classes."),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
