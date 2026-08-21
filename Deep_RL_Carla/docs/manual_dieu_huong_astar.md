# 📘 Tài liệu Hướng dẫn Module Điều hướng A* (`router_plan/`) — CARLA 0.9.10

> Tài liệu này hướng dẫn dùng `router_plan/`: chọn một điểm đến trên bản đồ, tự động tính
> đường đi **ngắn nhất** (A*) từ vị trí xe hiện tại đến điểm đó, và lái xe đi theo lộ trình
> vừa tính — tất cả trong một lệnh (`drive_to_goal.py`). Module này **độc lập** với
> `behavior_cloning/` và `drl_training/`: nó dùng một bộ điều khiển hình học tự viết
> (pure-pursuit + PID), không phải policy IL/DRL đã học — xem [mục 1](#1-tổng-quan-hệ-thống)
> để hiểu vì sao và mối quan hệ với phần còn lại của đồ án.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu trúc thư mục module](#3-cấu-trúc-thư-mục-module)
4. [Cách hoạt động](#4-cách-hoạt-động)
5. [Chọn điểm đến](#5-chọn-điểm-đến)
6. [Chạy thử](#6-chạy-thử)
7. [Tham số dòng lệnh đầy đủ](#7-tham-số-dòng-lệnh-đầy-đủ)
8. [Theo dõi quá trình chạy](#8-theo-dõi-quá-trình-chạy)
9. [Dùng `GlobalRoutePlanner` trong code khác](#9-dùng-globalrouteplanner-trong-code-khác)
10. [Giới hạn đã biết](#10-giới-hạn-đã-biết)
11. [Xử lý sự cố thường gặp](#11-xử-lý-sự-cố-thường-gặp)

---

## 1. Tổng quan hệ thống

### Vị trí trong đồ án

```
data_collection/       Thu thập dữ liệu bám làn (passive client)
behavior_cloning/       [Kaggle] Segmentation → Imitation Learning
drl_training/            [Local] Fine-tune PPO/SAC — bám làn, KHÔNG có điều hướng theo tuyến
router_plan/              [Local] ← TÀI LIỆU NÀY — A*: chọn đích, tính đường đi, lái theo lộ trình
```

`router_plan/` trả lời câu hỏi **"đi đường nào để tới đích?"** (lập kế hoạch toàn cục — global
planning), tách biệt hoàn toàn khỏi câu hỏi **"lái thế nào để không lệch làn?"** (điều khiển
cục bộ — local control, việc của `behavior_cloning/`/`drl_training/`). Đây là nguyên tắc kiến
trúc xuyên suốt đã ghi trong `data_collection/ASTAR_SCHEMA.md`.

### Vì sao dùng bộ điều khiển tự viết thay vì policy IL/DRL đã train?

`drl_training/` hiện tại (xem `docs/manual_train_drl.md`) chỉ biết **bám làn**, chưa biết
"rẽ theo lộ trình" — observation của nó không có `route_target_local_x/y`/`route_command`
(xem `router_plan/README.md` mục "Các bước triển khai", bước 4, việc **chưa làm**). Nếu chờ
tích hợp đầy đủ vào IL/DRL mới cho chạy A*, sẽ không kiểm chứng được thuật toán A*/route
tracker có đúng hay không. Vì vậy `router_plan/controller.py` cung cấp một bộ điều khiển
pure-pursuit (lái) + PID (ga/phanh) **tự viết, độc lập, không dùng mạng nơ-ron** — đủ để lái
xe đi hết một lộ trình A* ngay bây giờ, dùng để:

- Kiểm chứng graph/A*/route tracker đúng trước khi đầu tư vào bước tích hợp IL/DRL.
- Có một baseline "điều hướng bằng thuật toán cổ điển" để so sánh khi báo cáo đồ án.
- Trực quan hoá route (`--draw-debug`) độc lập với việc train bất kỳ model nào.

> [!NOTE]
> Đây **không phải** `agents.navigation` có sẵn của CARLA (thư mục
> `PythonAPI/carla/agents/`, tách biệt khỏi gói `carla` cài qua `.egg`/`.whl`) — controller ở
> đây được viết từ đầu để không phụ thuộc thư mục đó có nằm trên `sys.path` hay không.

### Kiến trúc 3 lớp

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. graph_builder.RouteGraph                                       │
│    Build graph node/edge LIVE từ world.get_map() mỗi lần chạy     │
│    (không cần --map-export trước, luôn khớp map đang load)        │
├──────────────────────────────────────────────────────────────────┤
│ 2. astar.find_path(graph, start_id, goal_id)                      │
│    A* thuần: g/h/f theo data_collection/ASTAR_SCHEMA.md           │
│    → danh sách node_id có thứ tự (route)                          │
├──────────────────────────────────────────────────────────────────┤
│ 3. route_tracker.RouteTracker(graph, route)                       │
│    Mỗi tick: .update(vehicle_transform) → route_target_local_x/y, │
│    route_command, route_progress_m, route_remaining_m,            │
│    route_completed — đúng ROUTE_FIELDS trong schema.py            │
├──────────────────────────────────────────────────────────────────┤
│ 4. controller.RoutePurePursuitController                          │
│    Pure-pursuit (lái) + PID (ga/phanh) bám theo route_target_*    │
│    → carla.VehicleControl mỗi tick                                │
└──────────────────────────────────────────────────────────────────┘
```

`goal_selection.py` (chọn điểm đến) và `Global_Route_Planner.GlobalRoutePlanner` (gộp lớp 1–3
thành một API) hỗ trợ 4 lớp trên. `drive_to_goal.py` là CLI nối tất cả lại với nhau.

---

## 2. Cài đặt môi trường

Dùng chung môi trường với `data_collection/`/`drl_training/` — không có thư viện nào thêm
(chỉ cần `carla`, và toàn bộ code trong `router_plan/` chỉ dùng thư viện chuẩn Python: không
cần `numpy`/`torch`).

| Thành phần | Phiên bản |
|---|---|
| CARLA Simulator | 0.9.10 |
| Python | 3.7 |
| carla (Python API) | Đi kèm CARLA 0.9.10 |
| OS | Windows 10/11 |

```powershell
# Neu chua cai CARLA Python API (xem docs/manual_thu_thap_du_lieu.md muc 2)
cd C:\CARLA_0.9.10\PythonAPI\carla\dist
pip install carla-0.9.10-py3.7-win-amd64.egg

# Kiem tra
python -c "import carla; print('CARLA OK')"
```

Không cần checkpoint IL/DRL nào — `router_plan/` không phụ thuộc `behavior_cloning/` hay
`drl_training/`.

---

## 3. Cấu trúc thư mục module

```
router_plan/
├── README.md                  # thiết kế, quyết định đã chốt, roadmap tích hợp IL/DRL
├── __init__.py
├── graph_builder.py           # RouteGraph — build graph live tu CARLA
├── astar.py                   # find_path() — A* thuan
├── route_tracker.py            # RouteTracker — sinh cac truong route_* moi tick
├── goal_selection.py           # resolve_goal() — chon diem den (3 cach, xem muc 5)
├── controller.py               # RoutePurePursuitController — pure-pursuit + PID
├── Global_Route_Planner.py      # GlobalRoutePlanner — API gop graph+astar+goal+tracker
└── drive_to_goal.py             # CLI — entrypoint chinh, dung file nay de chay
```

### Tra cứu nhanh: sửa ở đâu?

| Muốn thay đổi | File cần sửa |
|---|---|
| Độ chi tiết graph, cách nối edge | `graph_builder.py` |
| Thuật toán tìm đường | `astar.py` |
| Ngưỡng rẽ trái/phải, dung sai tới đích | `route_tracker.py` (`TURN_YAW_THRESHOLD_DEG`), CLI `--route-tolerance-m` |
| Cách chọn điểm đến | `goal_selection.py` |
| Tốc độ, độ mượt lái, hằng số PID | `controller.py` |
| Luồng chạy chính, log, CSV output, vẽ debug | `drive_to_goal.py` |

---

## 4. Cách hoạt động

### 4.1 Xây dựng graph (`graph_builder.RouteGraph`)

- **Node**: mỗi waypoint CARLA (chỉ lane `Driving`) lấy theo `world_map.generate_waypoints(resolution_m)`
  — mặc định `resolution_m=2.0` (2 mét/node, cùng ý nghĩa tham số `--graph-resolution` của
  `data_collection`).
- **Edge**: với mỗi node, dò các hướng đi hợp lệ:
  - `LANE_FOLLOW`: đi thẳng trong lane (waypoint.next() khi không phải junction/không rẽ nhánh).
  - `JUNCTION_BRANCH`: một trong các nhánh hợp lệ tại giao lộ.
  - `LANE_CHANGE_LEFT`/`LANE_CHANGE_RIGHT`: đổi làn hợp lệ (kiểm tra `lane_change` cho phép và
    lane bên cạnh cùng hướng đi — `lane_id` cùng dấu).
  - Chi phí (`cost_m`) = khoảng cách Euclid, nhân thêm `lane_change_cost` (mặc định 3.0) cho
    2 loại edge đổi làn — khuyến khích A* ưu tiên đi thẳng hơn đổi làn không cần thiết.
- Build **lại từ đầu mỗi lần chạy `drive_to_goal.py`**, mất vài giây trên một Town đầy đủ —
  không có bước export riêng, luôn khớp đúng map đang load (kể cả sau khi đổi map).

### 4.2 Tìm đường (`astar.find_path`)

Đúng công thức trong `data_collection/ASTAR_SCHEMA.md`:

```
g(next) = g(current) + edge.cost_m
h(node) = EuclideanDistance(node, goal)
f(node) = g(node) + h(node)
```

Input: `node_id` bắt đầu và đích (đã snap vào graph qua `graph.nearest_node()`). Output: danh
sách `node_id` có thứ tự từ start đến goal, hoặc `None` nếu không có đường đi (ví dụ xe và
đích ở hai lane một chiều ngược hướng, không có lane-change hợp lệ nối chúng).

### 4.3 Theo dõi tiến độ (`route_tracker.RouteTracker`)

Mỗi tick, `tracker.update(vehicle.get_transform())` trả về đúng các cột đã dành sẵn trong
`data_collection/carla_collector/schema.py::ROUTE_FIELDS`:

| Cột | Ý nghĩa |
|---|---|
| `route_id` | ID ngẫu nhiên (12 ký tự hex) sinh ra khi tạo `RouteTracker`, giữ nguyên suốt route |
| `route_target_index` | Index node đang nhắm tới trong danh sách route |
| `route_target_waypoint_id` | `node_id` đang nhắm tới |
| `route_target_local_x/y` | Vị trí node mục tiêu trong hệ toạ độ xe (m, forward/right) |
| `route_command` | `LANEFOLLOW`, `LEFT`, `RIGHT`, `STRAIGHT`, `CHANGELANELEFT`, `CHANGELANERIGHT` |
| `route_progress_m` | Quãng đường đã đi dọc route đã tính (tới node cuối cùng đã "đi qua") |
| `route_remaining_m` | `route_total_m - route_progress_m` |
| `route_total_m` | Tổng chiều dài route |
| `route_completed` | `1` khi xe vào trong bán kính `--route-tolerance-m` của node cuối |

**Quy ước `route_command` khi rẽ tại giao lộ** (`JUNCTION_BRANCH`): yaw tăng → `RIGHT`, yaw
giảm → `LEFT` (ngưỡng ±20°), nằm giữa → `STRAIGHT`. Suy ra trực tiếp từ quy ước "phải dương"
đã dùng xuyên suốt repo (`lane_offset_m > 0` = phải, xem `geometry.py::world_to_ego`) — xem
comment trong `route_tracker.py` cho chứng minh chi tiết.

### 4.4 Lái xe theo route (`controller.RoutePurePursuitController`)

- **Lái (steer)**: pure-pursuit — nhắm tới một điểm phía trước trên route, cách xe một khoảng
  "lookahead" tỉ lệ với tốc độ hiện tại (`lookahead_m = max(min_lookahead_m, lookahead_gain *
  speed_mps)`), tính độ cong `kappa = 2y/L²` rồi suy ra góc lái qua mô hình xe đạp
  (bicycle model, `wheelbase_m=2.85` mặc định).
- **Ga/phanh**: PID đơn giản bám theo `target_speed_kmh` (mặc định 30 km/h, tự động giảm nếu
  thấp hơn giới hạn tốc độ CARLA tại vị trí hiện tại).
- Controller đọc `tracker.target_index` trực tiếp — luôn nhắm đúng node mà `RouteTracker` coi
  là "tiếp theo", tránh lệch nhau giữa lái và theo dõi tiến độ.

---

## 5. Chọn điểm đến

`goal_selection.resolve_goal()` hỗ trợ đúng 3 cách — **chỉ chọn 1**:

| Cách | Cờ CLI | Khi nào dùng |
|---|---|---|
| Theo spawn point | `--goal-spawn-index N` | Đã biết index spawn point muốn tới (giống `data_collection --goal-spawn-index`) |
| Theo toạ độ world | `--goal-x X --goal-y Y [--goal-z Z]` | Đã biết toạ độ (x, y) chính xác |
| Chọn tương tác qua console | `--goal-interactive` | Chưa biết toạ độ nào cả — liệt kê hết spawn point rồi hỏi index |

Điểm đến (dù chọn cách nào) luôn được chiếu (`project_to_road`) vào waypoint gần nhất trên
lane `Driving` — nếu chọn toạ độ giữa không trung hay ngoài đường, sẽ báo lỗi rõ ràng thay vì
âm thầm lấy một node ngẫu nhiên.

### Ví dụ chế độ tương tác

```
Chon diem den — cac spawn point co san:
[  0] x=  -64.64 y=   24.47 z=   0.60 yaw=   0.00
[  1] x=  -64.64 y=   30.47 z=   0.60 yaw=   0.00
...
Nhap index spawn point lam diem den: 42
```

---

## 6. Chạy thử

### Yêu cầu: 1–2 terminal

```
Cửa sổ 1: CARLA Server
Cửa sổ 2 (tuỳ chọn): script đổi map/thời tiết trước khi chạy
Cửa sổ 3: drive_to_goal.py
```

Khác với `data_collection/`, **không cần** `automatic_control.py` — `drive_to_goal.py` tự
spawn xe của chính nó (giống `drl_training/`).

> [!CAUTION]
> `drive_to_goal.py` là **active client**: tự spawn xe và tự `world.tick()` (đặt
> `synchronous_mode=True`). Không chạy cùng lúc với `data_collection/` (passive collector),
> `automatic_control.py`, hoặc `drl_training/train_ppo.py`/`train_sac.py` trên cùng world —
> tất cả sẽ tranh giành quyền điều khiển xe/world settings.

### Bước 1 — Khởi động CARLA Server

```powershell
cd C:\CARLA_0.9.10\WindowsNoEditor
CarlaUE4.exe -quality-level=Low
```

### Bước 2 — Chạy

```powershell
cd C:\Users\dloc\Desktop\Do_An\Deep_RL_Carla\router_plan

# Chon dich qua console (de nhat khi moi dung lan dau)
python drive_to_goal.py --goal-interactive --draw-debug

# Hoac chi thang spawn index
python drive_to_goal.py --goal-spawn-index 42 --draw-debug

# Hoac toa do world cu the
python drive_to_goal.py --goal-x 120.5 --goal-y -34.2 --draw-debug
```

`--draw-debug` vẽ đường route màu xanh lá + chữ "GOAL" màu đỏ trực tiếp trong cửa sổ CARLA —
rất nên bật khi chạy thử lần đầu để xác nhận A* chọn đúng đường.

### Output mẫu

```
Dang build graph A* (resolution=2.0m)...
Graph: 8531 node, xong sau 3.2s
Tim thay duong A*: 214 node.
Bat dau lai xe theo lo trinh...
tick=20 progress=38.4m/612.7m command=LANEFOLLOW
tick=40 progress=79.1m/612.7m command=LANEFOLLOW
tick=60 progress=118.3m/612.7m command=RIGHT
...
Da den dich sau 640 tick (~32.0s).
```

---

## 7. Tham số dòng lệnh đầy đủ

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--host` | `127.0.0.1` | Địa chỉ CARLA server |
| `--port` | `2000` | Cổng CARLA |
| `--timeout` | `20.0` | Giây chờ kết nối |
| `--fps` | `20.0` | Tần số tick đồng bộ (Hz) — cũng là `dt` cho PID |
| `--vehicle-filter` | `vehicle.lincoln.mkz2017` | Blueprint xe spawn |
| `--graph-resolution` | `2.0` | m/node cho graph A* |
| `--lane-change-cost` | `3.0` | Hệ số nhân chi phí khi A* chọn đổi làn |
| `--route-tolerance-m` | `3.0` | Bán kính (m) coi là "đã tới" một node route / điểm đích |
| `--target-speed-kmh` | `30.0` | Tốc độ mục tiêu (tự động giảm nếu vượt giới hạn tốc độ tại chỗ) |
| `--max-duration-s` | `600.0` | Tự dừng nếu chưa tới đích sau từng này giây |
| `--draw-debug` | tắt | Vẽ đường route + điểm đích trong CARLA |
| `--output PATH` | không ghi | Ghi CSV mỗi tick: `tick, x, y, speed_kmh` + toàn bộ `ROUTE_FIELDS` |
| `--goal-spawn-index N` | `-1` | Chọn đích theo spawn point (xem [mục 5](#5-chọn-điểm-đến)) |
| `--goal-x`, `--goal-y`, `--goal-z` | `None`, `None`, `0.0` | Chọn đích theo toạ độ world |
| `--goal-interactive` | tắt | Hỏi điểm đến qua console |
| `--start-spawn-index N` | `-1` (ngẫu nhiên) | Spawn point xuất phát |

Phải chọn **đúng một** trong `--goal-spawn-index` / `--goal-x`+`--goal-y` / `--goal-interactive`
— thiếu hoặc chọn nhiều hơn một sẽ báo lỗi argparse rõ ràng ngay khi khởi động.

---

## 8. Theo dõi quá trình chạy

### Console

Cứ mỗi giây thực (≈`--fps` tick) in một dòng `tick=... progress=.../...m command=...` — theo
dõi `command` để xác nhận xe đang rẽ đúng chỗ dự kiến theo route đã vẽ debug.

### CSV (`--output`)

| Cột | Ý nghĩa |
|---|---|
| `tick` | Số tick kể từ lúc bắt đầu lái |
| `x`, `y` | Vị trí world hiện tại của xe |
| `speed_kmh` | Tốc độ hiện tại |
| *(toàn bộ `ROUTE_FIELDS`)* | Xem bảng ở [mục 4.3](#43-theo-dõi-tiến-độ-route_trackerroutetracker) |

Dùng để vẽ biểu đồ `route_progress_m` theo thời gian, hoặc kiểm tra `route_command` có đổi
đúng lúc xe thực sự rẽ hay không — hữu ích khi tinh chỉnh `TURN_YAW_THRESHOLD_DEG` trong
`route_tracker.py` nếu thấy nhãn `STRAIGHT`/`LEFT`/`RIGHT` không khớp trực giác.

### Kết thúc

- **Tới đích**: in `Da den dich sau N tick (~X.Xs).` rồi dừng vòng lặp, destroy xe, khôi phục
  `synchronous_mode` cũ của world.
- **Hết giờ** (`--max-duration-s`): in cảnh báo `[!] Het thoi gian toi da...` kèm
  `route_remaining_m` còn lại — tăng `--max-duration-s` hoặc `--target-speed-kmh` nếu route dài.
- **Ctrl+C** hoặc lỗi bất kỳ: khối `finally` trong `drive_to_goal.py` luôn destroy xe và khôi
  phục world settings trước khi thoát — không cần dọn tay.

---

## 9. Dùng `GlobalRoutePlanner` trong code khác

`Global_Route_Planner.GlobalRoutePlanner` là API công khai, dùng được ngoài `drive_to_goal.py`
(ví dụ trong một notebook, hay khi tích hợp route vào `drl_training/` ở bước 4 tương lai —
xem `router_plan/README.md`):

```python
import sys
sys.path.insert(0, r"C:\Users\dloc\Desktop\Do_An\Deep_RL_Carla")

from router_plan.Global_Route_Planner import GlobalRoutePlanner, RouteNotFoundError

planner = GlobalRoutePlanner(world.get_map(), resolution_m=2.0, lane_change_cost=3.0)
try:
    route = planner.plan(vehicle.get_location(), goal_location)
except RouteNotFoundError as exc:
    print("Khong tim duoc duong:", exc)
else:
    tracker = planner.tracker_for(route, target_tolerance_m=3.0)
    # moi tick:
    route_state = tracker.update(vehicle.get_transform())
```

> [!TIP]
> Build `GlobalRoutePlanner` **một lần cho mỗi map** rồi gọi `.plan()` nhiều lần với các đích
> khác nhau — build graph mất vài giây, không cần lặp lại cho mỗi lần đổi đích trên cùng map.

---

## 10. Giới hạn đã biết

- **Controller là bộ điều khiển hình học, không phải policy đã học** — dùng để demo/validate
  route A*, không phải kết quả nghiên cứu chính của đồ án (đó là `behavior_cloning/` +
  `drl_training/`). Đừng dùng số liệu lái của `controller.py` để so sánh với IL/DRL trong báo
  cáo — chúng giải quyết hai bài toán khác nhau (điều hướng cổ điển vs. học tăng cường).
- **Route chưa được đưa vào observation contract của IL/DRL** — `drl_training/` hiện chỉ biết
  bám làn, chưa biết rẽ theo lộ trình A*. Xem `router_plan/README.md` mục "Các bước triển
  khai", bước 4, cho kế hoạch tích hợp đầy đủ (đã chốt nhưng chưa làm).
- **1 CARLA instance / 1 xe** — không hỗ trợ nhiều xe chạy nhiều route song song trong cùng
  script; muốn vậy cần mở rộng `drive_to_goal.py` spawn nhiều `vehicle`/`RouteTracker` cùng
  lúc trong một vòng lặp tick chung.
- **Graph build lại mỗi lần chạy** — chấp nhận được cho một map (vài giây), nhưng nếu cần
  chạy nhiều route liên tiếp, dùng `GlobalRoutePlanner` trực tiếp ([mục 9](#9-dùng-globalrouteplanner-trong-code-khác))
  thay vì gọi lại `drive_to_goal.py` nhiều lần (mỗi lần chạy lại build graph từ đầu).

---

## 11. Xử lý sự cố thường gặp

### `RouteNotFoundError`: A* không tìm thấy đường đi

**Nguyên nhân:** xe và đích nằm trên hai lane không liên thông — thường gặp nhất là đích nằm
trên lane một chiều ngược hướng, và không có `lane_change` hợp lệ nối hai bên.

**Giải pháp:**
1. Thử `--draw-debug` với một đích khác gần đó để xác nhận graph/A* hoạt động đúng.
2. Kiểm tra `--lane-change-cost` không bị đặt quá cao (dù về lý thuyết không ảnh hưởng tính
   khả thi, chỉ ảnh hưởng ưu tiên — vẫn nên kiểm tra nếu nghi ngờ cấu hình sai).
3. Chọn đích trên cùng lane/road hoặc lane liền kề cùng hướng với vị trí xe.

---

### Không snap được vị trí xe/đích vào graph

**Triệu chứng:** `"Khong chieu duoc vi tri bat dau/dich len graph..."`

**Giải pháp:** tăng `--graph-resolution` (giá trị nhỏ hơn = graph dày hơn, vd `1.0`) — vị trí
quá gần rìa lane hoặc gần giao lộ phức tạp đôi khi không snap được ở resolution thô.

---

### Xe lái loạng choạng / lắc vô-lăng liên tục

**Nguyên nhân:** `--target-speed-kmh` quá cao so với độ cong đường, hoặc `--fps` quá thấp
(dt của PID quá lớn, phản ứng trễ).

**Giải pháp:**
1. Giảm `--target-speed-kmh` (thử 15–20 km/h ở khu vực nhiều khúc cua).
2. Tăng `--fps` (thử 30) để controller phản ứng mượt hơn.
3. Nếu vẫn lắc, chỉnh hằng số pure-pursuit trong `controller.py`
   (`lookahead_gain`/`min_lookahead_m` — tăng lên nếu xe bám quá sát, giảm nếu xe cắt cua quá rộng).

---

### Xe không bao giờ đạt `route_completed=1` dù đã tới gần đích

**Nguyên nhân:** `--route-tolerance-m` quá nhỏ so với sai số bám làn của controller ở tốc độ
hiện tại (xe lượn quanh node cuối mà không lọt vào bán kính).

**Giải pháp:** tăng `--route-tolerance-m` (thử 5.0), hoặc giảm `--target-speed-kmh` khi gần
tới đích.

---

### `"Khong spawn duoc xe tai spawn point da chon..."`

**Nguyên nhân:** spawn point đang bị chiếm bởi actor khác (xe từ session trước chưa được dọn,
NPC traffic manager, v.v).

**Giải pháp:** thử `--start-spawn-index` khác, hoặc bỏ cờ này để chọn ngẫu nhiên
(`drive_to_goal.py` không tự retry qua nhiều spawn point như `drl_training/envs/carla_lane_keep_env.py`
— nếu cần độ bền cao hơn, có thể thêm logic thử nhiều điểm tương tự file đó).

---

### Treo lâu ở bước "Dang build graph A*..."

**Nguyên nhân:** map lớn + `--graph-resolution` quá nhỏ (quá nhiều node).

**Giải pháp:** tăng `--graph-resolution` (thử 3.0–4.0 cho map lớn) — đánh đổi độ mượt của
route lấy tốc độ build.

---

## Tóm tắt quy trình nhanh

```
[1] Khởi động CARLA Server
        ↓
[2] (Tuỳ chọn) Đổi map/thời tiết
        ↓
[3] python drive_to_goal.py --goal-interactive --draw-debug
        (hoặc --goal-spawn-index N / --goal-x X --goal-y Y)
        ↓
[4] Quan sát đường route vẽ trong CARLA (nếu --draw-debug) — xác nhận hợp lý
        ↓
[5] Theo dõi console: tick=... progress=.../...m command=...
        ↓
[6] "Da den dich" → hoàn tất. Hết giờ/lỗi → xem mục 11 (Xử lý sự cố)
        ↓
[7] (Tuỳ chọn) Dùng --output route.csv để vẽ biểu đồ progress/route_command cho báo cáo
```

---

*Tài liệu được tạo tự động từ mã nguồn tại `router_plan/` — 2026-08-13.*
