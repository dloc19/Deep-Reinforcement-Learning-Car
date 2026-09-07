"""Dataset column definitions and the project-wide CARLA semantic label scheme."""

import numpy as np

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


# Canonical CARLA 0.9.10 raw semantic tag (0-22) -> the project's 4-class lane-keeping
# scheme ("lane4" in behavior_cloning/train-segment-lane.ipynb). This project has three independent
# consumers of a raw semantic-segmentation frame that each need to fold it down to the same
# classes: train-segment-lane.ipynb's `LABEL_LUT`, train_il.ipynb's `SEG_LABEL_LUT`, and
# drl_training/policy/observation.py's `resize_class_map()` (used both by the DRL env and the
# Carla Console Bridge Server's IL/DRL Autopilot mode for *live* camera frames). All of them
# must stay byte-for-byte identical to this table: a training-time class index has to mean
# the same set of raw tags as whatever the live camera/collector emits, or a checkpoint is
# being fed a different observation than it was trained on. This dict is the source of truth
# the other three are copied from (notebooks run on Kaggle and can't import this module, so
# they carry their own literal copy — check them if you ever change a mapping here).
#
# Only four raw tags are named; EVERY other tag - including the sky (raw 0 on CARLA 0.9.10,
# raw 13 from 0.9.11 on) and vehicles (raw 10) - collapses into Background. Two deliberate
# choices behind that, both reversals of an earlier 13-class scheme:
#
#   Vehicles -> Background. The dataset is collected with no NPC traffic spawned, so raw 10
#   is ~0% of pixels. An always-empty class still costs a one-hot channel in the IL/DRL
#   backbone and still drags the reported mIoU down with an IoU of ~0. Obstacle avoidance is
#   a later step: it needs data collected WITH traffic, at which point add ("Vehicle", [10])
#   back here AND in both notebooks AND bump drl_training/policy/backbone.py's NUM_CLASSES,
#   then retrain segmentation -> IL -> DRL together (old checkpoints will not load).
#
#   Sky -> Background, NOT ignore_index. The sky is ~26% of every frame; leaving it
#   unsupervised (the old IGNORE_UNLABELED=True) means the model is free to paint sidewalk
#   over the top half of the image and no metric ever notices. It answers the same question
#   as a wall - "not drivable" - so it belongs in Background rather than in a class of its
#   own that would inflate mIoU with an easy ~0.99.
SEG_CLASS_NAMES = ["Background", "Road", "RoadLine", "Sidewalk"]
NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

RAW_TO_TRAIN_LANE = {
     6: 2,   # RoadLine
     7: 1,   # Road
     8: 3,   # Sidewalk
    16: 1,   # RailTrack -> Road (tram rails sit flush in the road surface in Town03)
}

# 256-entry LUT: train_class = RAW_TO_TRAIN_LANE_LUT[raw_tag]. Every unlisted tag - and any
# tag a future CARLA version might add (> 22) - lands on Background (0) instead of producing
# an out-of-range class index that would blow up F.one_hot on the model side.
RAW_TO_TRAIN_LANE_LUT = np.zeros(256, dtype=np.uint8)
for _raw_id, _train_id in RAW_TO_TRAIN_LANE.items():
    RAW_TO_TRAIN_LANE_LUT[_raw_id] = _train_id
del _raw_id, _train_id



# Classes that are only a few pixels wide and would be DELETED by a plain nearest-neighbour
# downscale. RoadLine is 2-3 px wide near the car and 1 px near the horizon, yet it is the
# single most important signal for lane keeping; at 384x480 -> 192x240 a pure INTER_NEAREST
# resize drops ~31% of its pixels (measured), which means a model fed those masks at
# inference sees a thinner lane marking than it ever saw during training.
#
# Everywhere a class-id map is downscaled - `train-segment-lane.ipynb`'s `resize_mask_raw`,
# `train_il.ipynb`'s `downscale_labels`, `drl_training/policy/observation.py`'s
# `resize_class_map` - the fix is the same: nearest for the bulk, then restore any output
# cell whose *area coverage* by a thin class exceeds THIN_COVER_THRESH. Keeping the id list
# and the threshold here means the live DRL/bridge path and the two training notebooks
# cannot drift apart on it.
THIN_SEG_CLASS_NAMES = ("RoadLine", "Pedestrian")
THIN_SEG_CLASS_IDS = tuple(
    i for i, name in enumerate(SEG_CLASS_NAMES) if name in THIN_SEG_CLASS_NAMES)
THIN_COVER_THRESH = 0.25

# Preview colors, keyed by REMAPPED training id - same palette as the notebooks' `COLORS`.
SEG_CLASS_COLORS = {
    0: (60, 60, 60),      # Background (sky, buildings, vegetation, poles, vehicles, ...)
    1: (128, 64, 128),    # Road
    2: (157, 234, 50),    # RoadLine
    3: (244, 35, 232),    # Sidewalk
}

# 256 entries so any uint8 class_id indexes safely. Colors come from the REMAPPED training
# id (via RAW_TO_TRAIN_LANE_LUT), not the raw tag directly, so `seg_color/*.png` always
# shows the class the model is actually trained to predict for that pixel. The full raw
# tags are never lost: `seg_label/*.png` stores them unmodified, so a different class
# scheme can always be re-derived from an already-collected dataset without recollecting.
SEG_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for _raw_id in range(256):
    SEG_COLOR_LUT[_raw_id] = SEG_CLASS_COLORS[int(RAW_TO_TRAIN_LANE_LUT[_raw_id])]
del _raw_id


ROUTE_FIELDS = [
    "route_id", "route_target_index", "route_target_waypoint_id",
    "route_target_local_x", "route_target_local_y", "route_command",
    "route_progress_m", "route_remaining_m", "route_total_m", "route_completed",
]
