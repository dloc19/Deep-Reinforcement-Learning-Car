"""Session metadata serialization."""

import json
import math

import carla

from .geometry import utc_now, waypoint_record
from .schema import CITYSCAPES_COLORS, ROUTE_FIELDS


def write_metadata(path, world, world_map, ego, args, session_id,
                   goal_waypoint, map_graph_stats):
    fx = args.width / (2.0 * math.tan(math.radians(args.fov) / 2.0))
    weather = world.get_weather()
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
        "vehicle_id": ego.id,
        "vehicle_type": ego.type_id,
        "role_name": ego.attributes.get("role_name", ""),
        "collector_mode": "passive_async_client_no_world_tick",
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
        "semantic_colors": {str(k): v for k, v in CITYSCAPES_COLORS.items()},
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
                "previous_steer", "previous_longitudinal"],
            "action": ["steer", "longitudinal"],
            "reward_or_metrics_only": [
                "normalized_lane_offset", "heading_error_rad", "off_lane",
                "steer_delta", "longitudinal_delta"],
        },
        "notes": (
            "seg_label stores raw class IDs and is the resource-efficient training input; "
            "seg_color may also be trained as a 3-channel categorical image when its palette "
            "is kept unchanged."),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
