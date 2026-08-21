# Trường CSV theo bài toán

---

## IL — Imitation Learning

### Observation (đầu vào)
| Trường | Ghi chú |
|---|---|
| `seg_label_path` | Ảnh 1 kênh **raw tag CARLA 0–22** → remap 4 lớp bám làn (`Background, Road, RoadLine, Sidewalk`) qua `schema.RAW_TO_TRAIN_LANE_LUT` → one-hot 4 lớp |
| `seg_color_path` | Ảnh 3 kênh màu → normalize, không ColorJitter |
| `speed_mps` | Chuẩn hóa theo speed limit |
| `yaw_rate_rps` | Clip ÷ 1.5 |
| `previous_steer` | Đã chuẩn hóa [-1, 1] |
| `previous_longitudinal` | Đã chuẩn hóa [-1, 1] |

### Label (đầu ra)
| Trường | Ghi chú |
|---|---|
| `steer` | [-1, 1] |
| `longitudinal` | `throttle - brake` [-1, 1] |

### Chỉ dùng để đánh giá (không train)
`lane_offset_m`, `heading_error_rad`, `off_lane`, `is_junction`, `steer_delta`, `longitudinal_delta`

---

## DRL — Deep Reinforcement Learning

### Observation (giống IL, thêm sau khi A* nối vào)
| Trường | Ghi chú |
|---|---|
| *(toàn bộ IL observation)* | |
| `route_target_local_x` | Tọa độ mục tiêu trong hệ xe (trục trước) |
| `route_target_local_y` | Tọa độ mục tiêu trong hệ xe (trục phải) |
| `route_command` | One-hot 6 chiều: LANEFOLLOW / LEFT / RIGHT / STRAIGHT / CHANGELANELEFT / CHANGELANERIGHT |

### Action (giống IL)
`steer`, `longitudinal`

### Reward
| Trường | Dấu | Công thức |
|---|---|---|
| `forward_speed_mps` | `+` | clip tới speed limit |
| `lane_offset_m` | `−` | `|lane_offset|` |
| `heading_error_rad` | `−` | `|heading_error|` |
| `steer_delta` | `−` | bình phương |
| `longitudinal_delta` | `−` | bình phương |
| `yaw_rate_rps` | `−` | bình phương |
| `off_lane == 1` | `−` | phạt cố định |
| `collision_count` tăng | `−` | phạt nặng |
| `lane_invasion_count` tăng | `−` | phạt nhẹ |

### Done
| Điều kiện | Loại |
|---|---|
| `collision_count` tăng | Terminate (fail) |
| `off_lane == 1` kéo dài | Terminate (fail) |
| `route_completed == 1` | Terminate (success) |

---

## A* — Global Path Planning

### Input: vị trí hiện tại (snap start node)
`waypoint_id`, `road_id`, `section_id`, `lane_id`, `waypoint_s`

### Input: đích (snap goal node)
`goal_waypoint_id`, `goal_road_id`, `goal_section_id`, `goal_lane_id`, `goal_s`

### File đồ thị (từ `--map-export`)

**`map_nodes.csv`** — đỉnh đồ thị:
`node_id`, `road_id`, `section_id`, `lane_id`, `s`, `x`, `y`, `z`, `yaw_deg`, `left_node_id`, `right_node_id`

**`map_edges.csv`** — cạnh có hướng:
`from_node_id`, `to_node_id`, `edge_type`, `cost_m`, `yaw_delta_deg`

Loại cạnh: `LANE_FOLLOW` · `JUNCTION_BRANCH` · `LANE_CHANGE_LEFT` · `LANE_CHANGE_RIGHT`

Chi phí: `g(n) + h(n)` với `h = Euclidean(node → goal)`

### Output: route tracker điền vào
`route_target_local_x`, `route_target_local_y`, `route_command`, `route_progress_m`, `route_remaining_m`, `route_completed`
