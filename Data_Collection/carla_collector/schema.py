"""Dataset column definitions and CARLA 0.9.10 semantic colors."""

CSV_FIELDS = [
    "session_id", "episode_id", "sample_id", "frame", "sim_time_s", "delta_seconds",
    "sample_delta_seconds",
    "wall_time_utc", "map_name",
    "vehicle_id", "vehicle_type", "rgb_path", "seg_label_path", "seg_color_path",
    "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
    "velocity_x", "velocity_y", "velocity_z", "speed_mps", "speed_kmh",
    "forward_speed_mps", "lateral_speed_mps", "distance_travelled_m",
    "accel_x", "accel_y", "accel_z", "accel_mps2",
    "longitudinal_accel_mps2", "lateral_accel_mps2",
    "angular_x_deg_s", "angular_y_deg_s", "angular_z_deg_s",
    "yaw_rate_rps",
    "steer", "throttle", "brake", "hand_brake", "reverse", "manual_gear_shift", "gear",
    "longitudinal", "previous_steer", "previous_longitudinal",
    "steer_delta", "longitudinal_delta",
    "speed_limit_kmh", "traffic_light_state",
    "waypoint_id", "road_id", "section_id", "lane_id", "waypoint_s", "lane_width_m",
    "is_junction", "junction_id", "lane_type", "lane_change",
    "lane_offset_m", "normalized_lane_offset", "heading_error_deg", "heading_error_rad",
    "off_lane",
    "waypoint_x", "waypoint_y", "waypoint_z", "waypoint_yaw_deg",
    "waypoint_local_x", "waypoint_local_y",
    "next_waypoint_x", "next_waypoint_y", "next_waypoint_z", "next_waypoint_yaw_deg",
    "next_waypoint_id", "next_waypoint_road_id", "next_waypoint_section_id",
    "next_waypoint_lane_id", "next_waypoint_s", "next_waypoint_local_x",
    "next_waypoint_local_y", "next_waypoint_distance_m", "next_candidate_count",
    "successor_waypoints_json", "lookahead_waypoints_json",
    "left_waypoint_id", "left_road_id", "left_section_id", "left_lane_id", "left_lane_type",
    "right_waypoint_id", "right_road_id", "right_section_id", "right_lane_id", "right_lane_type",
    "goal_waypoint_id", "goal_road_id", "goal_section_id", "goal_lane_id", "goal_s",
    "goal_x", "goal_y", "goal_z", "goal_yaw_deg", "goal_local_x", "goal_local_y",
    "goal_euclidean_distance_m",
    "route_id", "route_target_index", "route_target_waypoint_id", "route_target_local_x",
    "route_target_local_y", "route_command", "route_progress_m", "route_remaining_m",
    "route_total_m", "route_completed",
    "collision_count", "lane_invasion_count",
]


CITYSCAPES_COLORS = {
    0: (0, 0, 0), 1: (70, 70, 70), 2: (100, 40, 40),
    3: (55, 90, 80), 4: (220, 20, 60), 5: (153, 153, 153),
    6: (157, 234, 50), 7: (128, 64, 128), 8: (244, 35, 232),
    9: (107, 142, 35), 10: (0, 0, 142), 11: (102, 102, 156),
    12: (220, 220, 0),
}


ROUTE_FIELDS = [
    "route_id", "route_target_index", "route_target_waypoint_id",
    "route_target_local_x", "route_target_local_y", "route_command",
    "route_progress_m", "route_remaining_m", "route_total_m", "route_completed",
]
