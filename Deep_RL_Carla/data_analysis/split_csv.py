#!/usr/bin/env python3
"""
split_csv.py — Tách states.csv thành 3 file theo bài toán: IL, DRL, A*

Cách dùng:
    python split_csv.py <session_dir>
    python split_csv.py D:/CARLA_DATA/Town01_20260715_230000_123456

Đầu ra (cùng thư mục với states.csv):
    il_fields.csv   — observation + label cho Imitation Learning
    drl_fields.csv  — observation + label + reward components cho DRL
    astar_fields.csv — waypoint/goal/route fields cho A*
"""

import argparse
import csv
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Định nghĩa trường cho từng bài toán
# ──────────────────────────────────────────────────────────

# Khoá chính — luôn có trong mọi file để join với nhau
KEY_FIELDS = ["session_id", "sample_id", "frame"]

# IL: observation + label
#
# QUAN TRỌNG: lane_offset_m/heading_error_rad/is_junction KHÔNG phải observation của model —
# chúng chỉ dùng để tính reward (DRL) hoặc chẩn đoán (đánh giá IL theo mức lệch làn). Đưa
# vào input model là lỗi rò rỉ nhãn (leakage) đã sửa trong
# `behavior_cloning/train_il_kaggle.ipynb` (xem `docs/csv_fields_by_task.md`). Cột này vẫn có
# mặt trong file split ra vì file CSV này phục vụ cả huấn luyện lẫn đánh giá/phân tích, không
# phải input tensor trực tiếp — notebook tự chọn đúng tập cột nó cần khi đọc CSV.
IL_FIELDS = KEY_FIELDS + [
    # Observation — ảnh
    "seg_label_path",
    "seg_color_path",
    # Observation — trạng thái xe (đúng SCALAR_FEATURE_ORDER trong checkpoint IL)
    "speed_mps",
    "yaw_rate_rps",
    "previous_steer",
    "previous_longitudinal",
    "speed_limit_kmh",
    "traffic_light_state",
    # Label — hành động chuyên gia
    "steer",
    "longitudinal",
    # Aux — CHỈ để đánh giá/chẩn đoán, KHÔNG đưa vào model (xem ghi chú ở trên)
    "lane_offset_m",
    "heading_error_rad",
    "is_junction",
]

# DRL: IL + các trường tính reward + route (khi A* nối vào)
DRL_FIELDS = KEY_FIELDS + [
    # Observation (giống IL — không gồm lane_offset_m/heading_error_rad/is_junction)
    "seg_label_path",
    "seg_color_path",
    "speed_mps",
    "yaw_rate_rps",
    "previous_steer",
    "previous_longitudinal",
    "speed_limit_kmh",
    "traffic_light_state",
    # Observation bổ sung — route (sau khi A* nối vào, hiện tại để trống)
    "route_target_local_x",
    "route_target_local_y",
    "route_command",
    # Action
    "steer",
    "longitudinal",
    # Reward components
    "forward_speed_mps",
    "speed_limit_kmh",
    "lane_offset_m",
    "normalized_lane_offset",
    "heading_error_rad",
    "steer_delta",
    "longitudinal_delta",
    "yaw_rate_rps",    # đã có trong obs, dùng lại cho reward
    "off_lane",
    "collision_count",
    "lane_invasion_count",
    # Done condition
    "route_completed",
]

# A*: waypoint hiện tại + goal + route output
ASTAR_FIELDS = KEY_FIELDS + [
    # Vị trí xe — snap start node
    "waypoint_id",
    "waypoint_x",
    "waypoint_y",
    "road_id",
    "section_id",
    "lane_id",
    "waypoint_s",
    # Làn kề — đổi làn
    "left_waypoint_id",
    "left_lane_id",
    "left_lane_type",
    "right_waypoint_id",
    "right_lane_id",
    "right_lane_type",
    # Successor để expand
    "successor_waypoints_json",
    "lookahead_waypoints_json",
    "next_candidate_count",
    # Đích — snap goal node
    "goal_waypoint_id",
    "goal_x",
    "goal_y",
    "goal_road_id",
    "goal_section_id",
    "goal_lane_id",
    "goal_s",
    "goal_euclidean_distance_m",
    # Route output (điền bởi planner)
    "route_id",
    "route_target_index",
    "route_target_waypoint_id",
    "route_target_local_x",
    "route_target_local_y",
    "route_command",
    "route_progress_m",
    "route_remaining_m",
    "route_total_m",
    "route_completed",
]

# Bỏ trùng nhưng giữ thứ tự
def deduplicate(fields):
    seen = set()
    result = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result

IL_FIELDS    = deduplicate(IL_FIELDS)
DRL_FIELDS   = deduplicate(DRL_FIELDS)
ASTAR_FIELDS = deduplicate(ASTAR_FIELDS)

# ──────────────────────────────────────────────────────────

def split(session_dir: Path):
    states_path = session_dir / "states.csv"
    if not states_path.is_file():
        sys.exit(f"Không tìm thấy: {states_path}")

    # Đọc header để kiểm tra trường nào thực sự có
    with states_path.open(newline="", encoding="utf-8") as f:
        actual_fields = set(csv.DictReader(f).fieldnames or [])

    outputs = {
        "il":    (session_dir / "il_fields.csv",    IL_FIELDS),
        "drl":   (session_dir / "drl_fields.csv",   DRL_FIELDS),
        "astar": (session_dir / "astar_fields.csv", ASTAR_FIELDS),
    }

    # Cảnh báo trường bị thiếu
    for name, (_, fields) in outputs.items():
        missing = [f for f in fields if f not in actual_fields]
        if missing:
            print(f"[WARN] {name}: thiếu {len(missing)} trường (sẽ để trống): {missing}")

    # Ghi 3 file song song
    handles, writers = {}, {}
    try:
        for name, (path, fields) in outputs.items():
            h = path.open("w", newline="", encoding="utf-8")
            handles[name] = h
            writers[name] = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            writers[name].writeheader()

        with states_path.open(newline="", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            for row in reader:
                for name, (_, fields) in outputs.items():
                    # Điền "" cho trường không có trong row
                    out_row = {f: row.get(f, "") for f in fields}
                    writers[name].writerow(out_row)
    finally:
        for h in handles.values():
            h.close()

    print("\nĐã tạo:")
    for name, (path, fields) in outputs.items():
        print(f"  {path.name:25s}  {len(fields)} cột")


def main():
    parser = argparse.ArgumentParser(description="Tách states.csv thành il/drl/astar CSV")
    parser.add_argument("session", help="Thư mục session (chứa states.csv)")
    args = parser.parse_args()
    split(Path(args.session).expanduser().resolve())


if __name__ == "__main__":
    main()
