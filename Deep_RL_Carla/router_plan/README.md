# router_plan — A* global route planning

## Trạng thái

**Đã triển khai** phần lập kế hoạch + lái theo lộ trình (bước 1–3 trong "Các bước triển khai"
bên dưới), độc lập với `behavior_cloning/`/`drl_training/` — dùng một controller
pure-pursuit/PID tự viết để lái, **chưa** phải policy IL/DRL đã học. Xem
`docs/manual_dieu_huong_astar.md` cho hướng dẫn sử dụng đầy đủ (cài đặt, CLI, xử lý sự cố).

```
router_plan/
├── graph_builder.py          # build graph live tu CARLA API (node/edge)
├── astar.py                   # A* thuan (g/h/f theo ASTAR_SCHEMA.md)
├── route_tracker.py            # theo doi tien do, sinh cac truong route_*
├── goal_selection.py           # chon dich: --goal-spawn-index / --goal-x,y / --goal-interactive
├── controller.py               # pure-pursuit + PID — lai xe theo route (khong phai policy IL/DRL)
├── Global_Route_Planner.py      # API cong khai, gop 4 file tren lai (GlobalRoutePlanner)
└── drive_to_goal.py             # CLI: chon dich -> A* -> lai xe
```

**Chưa làm** (bước 4–5 bên dưới): route chưa được đưa vào observation contract của IL/DRL —
policy đã train ở baseline vẫn chỉ biết bám làn, chưa biết rẽ theo lộ trình. Đây là việc độc
lập, làm sau khi cần đến navigation thật sự thay vì chỉ demo/validate A*.

## Kiến trúc (đúng theo `../data_collection/ASTAR_SCHEMA.md`)

- **A\*** lo lập kế hoạch toàn cục: chọn chuỗi node/lane ngắn nhất từ vị trí xe hiện tại đến
  điểm đích do người dùng chọn.
- **Điều khiển cục bộ**: hiện tại `controller.py` (pure-pursuit + PID tự viết, xem
  `docs/manual_dieu_huong_astar.md`) — dùng để demo/validate route A* ngay mà không cần đợi
  bước 4 bên dưới. Về lâu dài, đúng tinh thần đồ án, việc này nên do **policy IL/DRL** đảm
  nhiệm (bám làn mượt trong từng đoạn) sau khi route được đưa vào observation contract. A*
  **không** trực tiếp sinh `steer`/`throttle`/`brake` trong cả hai trường hợp.
- **Route tracker**: mỗi bước, tính `route_target_local_x/y` (điểm mục tiêu kế tiếp trong hệ
  toạ độ xe), `route_command` (`LANEFOLLOW`/`LEFT`/`RIGHT`/`STRAIGHT`/`CHANGELANELEFT`/
  `CHANGELANERIGHT`), `route_progress_m`, `route_remaining_m`, `route_completed` — đúng các
  cột đã dành sẵn (để trống) trong `carla_collector/schema.py`.

## Quyết định đã chốt

- **Đồ thị**: build **live từ CARLA API** mỗi lần chạy (`world.get_map().generate_waypoints(...)`
  + `waypoint.next()`/`get_left_lane()`/`get_right_lane()`) — không cần bước `--map-export`
  riêng trước, luôn khớp đúng map đang load. (Đã cân nhắc phương án đọc
  `map_nodes.csv`/`map_edges.csv` do `data_collection --map-export` xuất sẵn — không chọn vì
  cần một bước export riêng cho từng map và dễ lệch nếu map thay đổi.)
- **Chọn đích**: giữ đúng convention đã có ở `data_collection/carla_collector/config.py`
  (`--goal-spawn-index` / `--goal-x`/`--goal-y`/`--goal-z`) — tái dùng, không phát minh lại.

## Các bước triển khai

1. ✅ **`router_plan/graph_builder.py`** — build đồ thị node/edge live từ CARLA: node =
   waypoint theo `graph_resolution` (m/node, cùng ý nghĩa tham số với `data_collection`), edge
   = `LANE_FOLLOW`/`JUNCTION_BRANCH`/`LANE_CHANGE_LEFT`/`LANE_CHANGE_RIGHT` kèm `cost_m`. Cùng
   quy ước `node_id`/cost với `data_collection/carla_collector/map_export.py` (CSV exporter)
   nhưng giữ trong bộ nhớ, không ghi CSV.
2. ✅ **`router_plan/astar.py`** — A* thuần theo đúng công thức trong `ASTAR_SCHEMA.md`:
   `g(next) = g(current) + edge.cost_m`, `h(node) = Euclidean(node, goal)`, `f = g + h`.
3. ✅ **`router_plan/route_tracker.py`** — theo dõi tiến độ xe dọc route đã tìm, sinh đúng các
   trường `route_*` mỗi step. Tái dùng `world_to_ego()`/`normalize_angle()` từ
   `data_collection/carla_collector/geometry.py` (cùng cách bootstrap `sys.path` như
   `drl_training/envs/carla_lane_keep_env.py`).
   Ngoài kế hoạch ban đầu, đã thêm 4 file để có một công cụ chạy được ngay:
   `goal_selection.py` (chọn đích, tái dùng convention `--goal-spawn-index`/`--goal-x/y` +
   thêm chế độ hỏi qua console `--goal-interactive`), `controller.py` (pure-pursuit + PID tự
   viết để lái theo route — không phải policy IL/DRL, xem giải thích ở mục "Kiến trúc"),
   `Global_Route_Planner.py` (API gộp 3 file đầu, `class GlobalRoutePlanner`), và
   `drive_to_goal.py` (CLI hoàn chỉnh: chọn đích → A* → lái xe). Xem
   `docs/manual_dieu_huong_astar.md` cho hướng dẫn dùng đầy đủ.
4. **Mở rộng observation contract** (bước bắt buộc, không thể bỏ qua nếu muốn policy thực sự
   rẽ đúng theo route thay vì chỉ đi thẳng ở ngã ba/ngã tư):
   - `behavior_cloning/train_il.ipynb`: thêm `route_target_local_x`, `route_target_local_y`
     (2 số liên tục, z-score) + `route_command` one-hot (6 lớp) vào scalar vector — cùng chỗ
     đang định nghĩa `CONTINUOUS_COLS`/`TRAFFIC_LIGHT_VOCAB`. Cần **dataset mới** có sẵn các
     cột này không rỗng — tức là thu thập lại với `--map-export` bật + gắn
     `router_plan/route_tracker.py` vào collector (hoặc một script thu thập riêng có random
     goal mỗi session) để `states.csv` có route hợp lệ.
   - Train lại IL trên Kaggle với observation mới → checkpoint IL mới (`best_il_model.pth`
     mới, tự động có `scalar_feature_dim` lớn hơn).
   - `drl_training/policy/observation.py::ObservationContract` **không cần sửa gì** — nó đã tự
     đọc `continuous_cols`/`scalar_feature_dim` từ checkpoint, tự thích ứng với contract mới.
   - `drl_training/envs/carla_lane_keep_env.py` (khả năng cần tách thành
     `carla_route_follow_env.py` mới, giữ `carla_lane_keep_env.py` làm baseline không đích):
     gắn `route_tracker` vào `_build_state()`, thêm `route_completed` vào điều kiện `done`
     (Terminate — success, khác với các lý do fail hiện có), cân nhắc thêm reward theo tiến độ
     route (`route_progress_m` tăng → thưởng nhẹ, khuyến khích tiến về đích thay vì chỉ vòng
     vòng giữ làn).
5. **Warm-start lại DRL** (PPO/SAC) từ checkpoint IL mới — `drl_training/policy/il_compat.py`
   không cần sửa (remap theo tên layer, không phụ thuộc số chiều observation).

## Việc chưa cần làm ngay

Không có gì ở baseline (`behavior_cloning/`, `drl_training/` hiện tại) cần sửa để dùng được
`router_plan/` như hiện tại (bước 1–3, `drive_to_goal.py`). Baseline bám làn train/dùng bình
thường, độc lập hoàn toàn cho tới bước 4 ở trên.
