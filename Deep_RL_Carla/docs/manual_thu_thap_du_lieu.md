# 📘 Tài liệu Hướng dẫn Thu thập Dữ liệu CARLA 0.9.10

> Tài liệu này mô tả toàn bộ quy trình thu thập dữ liệu từ simulator CARLA 0.9.10 phục vụ huấn luyện mô hình Imitation Learning (IL) và Deep Reinforcement Learning (DRL) trong dự án xe tự lái.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Cấu hình collector](#4-cấu-hình-collector)
5. [Quy trình chạy thu thập](#5-quy-trình-chạy-thu-thập)
6. [Dữ liệu đầu ra](#6-dữ-liệu-đầu-ra)
7. [Kiểm tra và xác thực session](#7-kiểm-tra-và-xác-thực-session)
8. [Tạo manifest và chia tập dữ liệu](#8-tạo-manifest-và-chia-tập-dữ-liệu)
9. [Sử dụng dữ liệu cho huấn luyện](#9-sử-dụng-dữ-liệu-cho-huấn-luyện)
10. [Mở rộng A*](#10-mở-rộng-a)
11. [Tham số tham khảo](#11-tham-số-tham-khảo)
12. [Xử lý sự cố thường gặp](#12-xử-lý-sự-cố-thường-gặp)

---

## 1. Tổng quan hệ thống

### Kiến trúc pipeline

Pipeline thu thập dữ liệu hoạt động theo mô hình **client thụ động** — nó **không** tạo xe, **không** bật autopilot và **không** thay đổi bản đồ. Thay vào đó, nó:

1. Kết nối vào CARLA server đang chạy.
2. Tìm xe `hero` đã được `automatic_control.py` tạo ra.
3. Gắn semantic camera lên xe `hero`.
4. Ghi dữ liệu đồng bộ theo `frame` của CARLA.

```
┌─────────────────────────────────────────────────────────┐
│                      CARLA Server                       │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  World/Map   │    │  Xe hero (automatic_control) │   │
│  └──────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          ↑                         ↑
          │ get_world()             │ find_ego()
          │                         │ spawn sensors
┌─────────────────────────────────────────────────────────┐
│                  CarlaCollector (client 2)               │
│                                                         │
│  FrameSynchronizer → DatasetWriter → states.csv + PNG   │
└─────────────────────────────────────────────────────────┘
```

### Nguyên tắc đồng bộ dữ liệu

- Collector **không gọi** `world.tick()` vì `automatic_control.py` là client riêng biệt.
- Dữ liệu ghép theo `image.frame` và `WorldSnapshot.frame` — chỉ ghi khi ảnh segmentation và state cùng frame.
- Tránh ghép nhầm ảnh và trạng thái dù camera GPU có độ trễ.

---

## 2. Cài đặt môi trường

### Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| CARLA Simulator | 0.9.10 |
| Python | 3.7 |
| numpy | ≥1.18, <2.0 |
| Pillow | ≥7.0 |
| OS | Windows 10/11 |

### Cài đặt thư viện Python

Mở **PowerShell** hoặc **Anaconda Prompt** với Python đúng phiên bản đang dùng với CARLA:

```powershell
# Bước 1: Cài CARLA Python API
cd C:\CARLA_0.9.10\PythonAPI\carla\dist
pip install carla-0.9.10-py3.7-win-amd64.egg

# Bước 2: Cài thư viện phụ thuộc
pip install numpy==1.19.5 Pillow

# Bước 3: Kiểm tra cài đặt
python -c "import carla; print('CARLA Python API OK')"
```

> [!NOTE]
> Nếu bản CARLA của bạn cung cấp file `.whl` thay vì `.egg`, cài file `.whl` đó thay thế.

### Cài thư viện từ requirements.txt

```powershell
cd <thu_muc_pipeline>\Data_Collection
pip install -r requirements.txt
```

---

## 3. Cấu trúc thư mục dự án

```
Data_Collection/
├── collect_data.py              # Entry point — chạy file này để bắt đầu thu thập
├── collector_config.json        # File cấu hình chính
├── run_collector.bat            # Script chạy nhanh trên Windows
├── verify_dataset.py            # Kiểm tra tính toàn vẹn một session
├── build_manifest.py            # Gộp và chia nhiều session thành train/val/test
├── ASTAR_SCHEMA.md              # Data contract cho planner A*
├── requirements.txt             # Thư viện Python cần thiết
└── carla_collector/             # Package chứa toàn bộ logic
    ├── __init__.py
    ├── collector.py             # Điều phối toàn bộ pipeline
    ├── config.py                # Đọc tham số CLI và JSON config
    ├── schema.py                # Định nghĩa cột CSV và màu semantic
    ├── geometry.py              # Helper tọa độ, góc, waypoint
    ├── events.py                # Đếm va chạm và vượt làn
    ├── synchronizer.py          # Ghép camera/state theo frame
    ├── writer.py                # Lưu PNG và states.csv
    ├── sensors.py               # Tạo/hủy sensor
    ├── state_builder.py         # Xây dựng trạng thái xe, làn, đích
    ├── map_export.py            # Xuất OpenDRIVE và graph A*
    └── metadata.py              # Ghi metadata.json
```

### Tra cứu nhanh: sửa ở đâu?

| Muốn thay đổi | File cần sửa |
|---|---|
| Thêm/bớt cột CSV | `carla_collector/schema.py` |
| Đổi camera, vị trí camera | `config.py`, `sensors.py` |
| Đổi cách lưu segmentation/RGB | `writer.py` |
| Thêm trạng thái hoặc nhãn học | `state_builder.py` |
| Đổi graph/cost A* | `map_export.py` |
| Đổi cách đồng bộ frame | `synchronizer.py` |
| Đổi trình tự kết nối/chạy/dừng | `collector.py` |

---

## 4. Cấu hình collector

### File cấu hình: `collector_config.json`

```json
{
  "connection": {
    "host": "127.0.0.1",
    "port": 2000,
    "timeout": 30.0,
    "role_name": "hero",
    "vehicle_id": 0,
    "wait_vehicle_timeout": 120.0
  },
  "dataset": {
    "output": "D:/CARLA_DATA"
  },
  "camera": {
    "image_mode": "seg-only",
    "save_seg_color": true,
    "width": 192,
    "height": 108,
    "fov": 90.0,
    "fps": 10.0,
    "camera_x": 1.5,
    "camera_y": 0.0,
    "camera_z": 2.4,
    "camera_pitch": -5.0
  },
  "collection": {
    "max_samples": 50000,
    "duration": 0.0,
    "queue_size": 64,
    "no_event_sensors": false
  },
  "astar": {
    "lookahead_m": 5.0,
    "route_lookaheads": "5,10,20,30",
    "graph_resolution": 2.0,
    "lane_change_cost": 3.0,
    "no_map_export": true,
    "goal_spawn_index": -1,
    "goal_x": null,
    "goal_y": null,
    "goal_z": 0.0
  }
}
```

### Giải thích chi tiết từng nhóm tham số

#### Nhóm `connection` — Kết nối CARLA

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `host` | `"127.0.0.1"` | Địa chỉ IP của CARLA server |
| `port` | `2000` | Cổng kết nối CARLA |
| `timeout` | `30.0` | Thời gian chờ kết nối (giây) |
| `role_name` | `"hero"` | Tên role của xe để tìm (do `automatic_control.py` đặt) |
| `vehicle_id` | `0` | Gắn trực tiếp theo actor ID; `0` = tự tìm theo `role_name` |
| `wait_vehicle_timeout` | `120.0` | Thời gian tối đa chờ xe hero xuất hiện (giây) |

#### Nhóm `dataset` — Đầu ra

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `output` | `"D:/CARLA_DATA"` | Thư mục gốc lưu toàn bộ dataset |

#### Nhóm `camera` — Camera và hình ảnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `image_mode` | `"seg-only"` | `seg-only`: chỉ semantic; `seg-rgb`: thêm RGB |
| `save_seg_color` | `true` | Lưu thêm ảnh semantic màu 3 kênh |
| `width` | `192` | Chiều rộng ảnh (pixel) |
| `height` | `108` | Chiều cao ảnh (pixel) |
| `fov` | `90.0` | Field of View theo chiều ngang (độ) |
| `fps` | `10.0` | Tần suất chụp ảnh (frame/giây) |
| `camera_x` | `1.5` | Vị trí camera trước/sau xe (m) |
| `camera_y` | `0.0` | Vị trí camera trái/phải xe (m) |
| `camera_z` | `2.4` | Chiều cao camera (m) |
| `camera_pitch` | `-5.0` | Góc nghiêng camera xuống (độ âm = nhìn xuống) |

> [!TIP]
> Tham số `width: 192, height: 108` trong config là resolution nhỏ để tiết kiệm dung lượng. Để huấn luyện CNN tốt hơn, nên nâng lên `800x450` (xem [Tham số tham khảo](#11-tham-số-tham-khảo)).

#### Nhóm `collection` — Điều kiện dừng

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `max_samples` | `50000` | Tự dừng khi ghi đủ N mẫu; `0` = không giới hạn |
| `duration` | `0.0` | Tự dừng sau N giây thực tế; `0` = không giới hạn |
| `queue_size` | `64` | Kích thước hàng đợi nội bộ giữa camera và writer |
| `no_event_sensors` | `false` | `true` = không spawn sensor va chạm/vượt làn |

> [!IMPORTANT]
> Nếu cả `max_samples > 0` lẫn `duration > 0`, điều kiện nào đạt trước sẽ dừng trước.

#### Nhóm `astar` — A* và waypoint

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `lookahead_m` | `5.0` | Khoảng cách nhìn trước để chọn waypoint tiếp theo |
| `route_lookaheads` | `"5,10,20,30"` | Các khoảng cách lookahead ghi vào CSV (m) |
| `graph_resolution` | `2.0` | Mật độ waypoint trong graph A* (m/node) |
| `lane_change_cost` | `3.0` | Hệ số nhân chi phí khi A* chuyển làn |
| `no_map_export` | `true` | `true` = không xuất graph A* (giai đoạn IL/DRL hiện tại) |
| `goal_spawn_index` | `-1` | Index spawn point làm đích; `-1` = không dùng |
| `goal_x/y/z` | `null` | Tọa độ world làm đích; `null` = không dùng |

---

## 5. Quy trình chạy thu thập

### Yêu cầu: 3 terminal (cửa sổ) riêng biệt

```
Cửa sổ 1: CARLA Server
Cửa sổ 2: Jupyter Notebook (đổi map)
Cửa sổ 3: automatic_control.py (tạo xe hero)
Cửa sổ 4: collect_data.py (thu thập dữ liệu)
```

---

### Bước 1 — Khởi động CARLA Server

```powershell
# Cửa sổ 1
cd C:\CARLA_0.9.10\WindowsNoEditor
CarlaUE4.exe -quality-level=Low
```

> [!NOTE]
> `-quality-level=Low` giảm tải GPU, không ảnh hưởng semantic segmentation. Giữ cửa sổ này mở suốt toàn bộ quá trình.

---

### Bước 2 — Tải map qua Jupyter Notebook

```python
# Server_Connect/Server_Carla.ipynb
import carla
import time

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

# Tải map mong muốn (Town01, Town02, Town03, Town04)
world = client.load_world("Town01")
time.sleep(3)  # Đợi map tải xong

# Đặt thời tiết SAU load_world()
world.set_weather(carla.WeatherParameters.ClearNoon)
print("Map hiện tại:", world.get_map().name)
```

> [!CAUTION]
> **Không** chạy `client.load_world()` khi collector đang chạy — lệnh này hủy toàn bộ world, actor và sensor của map cũ ngay lập tức.

---

### Bước 3 — Chạy automatic_control.py (tạo xe hero)

```powershell
# Cửa sổ 3
cd C:\CARLA_0.9.10\PythonAPI\examples
python automatic_control.py --host 127.0.0.1 --port 2000
```

**Đợi cho đến khi xe xuất hiện trên màn hình CARLA và bắt đầu di chuyển.** Script này tạo xe với `role_name=hero`, collector sẽ tự tìm xe này.

---

### Bước 4 — Bắt đầu thu thập dữ liệu

#### Cách 1: Dùng file `.bat` (khuyến nghị)

```powershell
# Cửa sổ 4
cd C:\Users\dloc\Desktop\Do_An\Data_Collection
run_collector.bat
```

#### Cách 2: Chạy trực tiếp với config file

```powershell
python collect_data.py --config collector_config.json
```

#### Cách 3: Ghi đè tham số tạm thời (không sửa JSON)

```powershell
# Chạy thử 60 giây
python collect_data.py --config collector_config.json --duration 60

# Thu đúng 5.000 mẫu
python collect_data.py --config collector_config.json --max-samples 5000

# Thêm ảnh RGB để đối chiếu thị giác
python collect_data.py --config collector_config.json --image-mode seg-rgb --save-seg-color

# Xuất graph A* (khi cần)
python collect_data.py --config collector_config.json --map-export
```

**Nhấn `Ctrl+C` để dừng an toàn.** Collector chỉ hủy các sensor do nó tạo; xe và autopilot ở cửa sổ 3 vẫn hoạt động.

---

### Bước 5 — Kiểm tra session sau khi thu

```powershell
python verify_dataset.py D:\CARLA_DATA\Town01_20260715_230000_123456 --strict
```

---

### Bước 6 — Lặp lại cho các map tiếp theo

```
1.  [Notebook] client.load_world("Town02")
2.  [Đợi] Map tải xong
3.  [Cửa sổ 3] Dừng automatic_control.py (Ctrl+C) → chạy lại
4.  [Cửa sổ 4] Chạy lại run_collector.bat
5.  [Sau khi đủ mẫu] Kiểm tra verify_dataset.py
6.  Lặp lại với Town03, Town04
```

> [!IMPORTANT]
> Mỗi lần đổi map, **phải** dừng và khởi động lại `automatic_control.py` để tạo xe hero mới trên map mới.

---

### Sơ đồ luồng thu thập

```
┌─────────────────────────────────────────────────────────────┐
│                  Thu thập một vòng lặp                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CarlaCollector.run()                                       │
│       │                                                     │
│       ├─ Kết nối CARLA server                               │
│       ├─ Tìm xe ego (find_ego)                              │
│       ├─ Tạo thư mục session                               │
│       ├─ Xuất graph A* (nếu bật)                           │
│       ├─ Spawn sensors (semantic cam + event sensors)       │
│       ├─ Ghi metadata.json                                  │
│       └─ Vòng lặp chính ──────────────────────────────┐    │
│                                                        │    │
│  WorldSnapshot ──on_tick──> capture_state()            │    │
│  SemanticCamera ──listen──> synchronizer.put("seg")    │    │
│                                                        │    │
│  FrameSynchronizer: khi đủ ("seg", "state")            │    │
│       └─> packet_queue.put(packet)                     │    │
│                                                        │    │
│  DatasetWriter (thread riêng):                         │    │
│       ├─ Giải mã ảnh BGRA → label PNG                  │    │
│       ├─ Lưu seg_color PNG (nếu bật)                   │    │
│       └─ Ghi dòng vào states.csv                       │    │
│                                                        │    │
│  [Ctrl+C / đủ mẫu / hết thời gian] ────────────────────┘    │
│       └─> cleanup(): hủy sensor → flush CSV → summary.json  │
└─────────────────────────────────────────────────────────────┘
```

---

### Tùy chọn nâng cao: Chọn xe theo ID hoặc role khác

```powershell
# Nếu không tìm thấy "hero", chỉ định actor ID trực tiếp
python collect_data.py --vehicle-id 123 --output D:\CARLA_DATA

# Hoặc dùng role name khác
python collect_data.py --role-name ego_vehicle --output D:\CARLA_DATA
```

### Tùy chọn nâng cao: Đặt điểm đích cho A*

```powershell
# Dùng spawn point index làm đích
python collect_data.py --config collector_config.json --goal-spawn-index 12

# Dùng tọa độ world làm đích
python collect_data.py --config collector_config.json --goal-x 85.0 --goal-y 12.5 --goal-z 0

# Tinh chỉnh graph A*
python collect_data.py --map-export --graph-resolution 2.0 --lane-change-cost 3.0
```

---

## 6. Dữ liệu đầu ra

### Cấu trúc thư mục một session

```
D:\CARLA_DATA\
└── Town01_20260715_230000_123456/     ← tên = TownXX + timestamp
    ├── metadata.json                  ← thông tin session
    ├── summary.json                   ← thống kê sau khi dừng
    ├── states.csv                     ← dữ liệu số cho tất cả frame
    ├── seg_label/
    │   ├── 00001234.png               ← ảnh 1 kênh, giá trị = class ID (0..12)
    │   ├── 00001235.png
    │   └── ...
    ├── seg_color/                     ← chỉ khi save_seg_color=true
    │   ├── 00001234.png               ← ảnh 3 kênh màu theo bảng Cityscapes
    │   └── ...
    └── rgb/                           ← chỉ khi image_mode=seg-rgb
        ├── 00001234.png
        └── ...
```

> [!NOTE]
> Tên file ảnh là số frame 8 chữ số (`%08d`), khớp chính xác với cột `frame` trong `states.csv`.

---

### File `metadata.json`

Ghi lại toàn bộ thông tin session:

```json
{
  "created_utc": "2026-07-15T23:00:00.000000Z",
  "schema_version": "2.0",
  "carla_version": "0.9.10",
  "session_id": "Town01_20260715_230000_123456",
  "map": "/Game/Carla/Maps/Town01",
  "vehicle_type": "vehicle.lincoln.mkz2017",
  "collector_mode": "passive_async_client_no_world_tick",
  "camera": {
    "width": 192,
    "height": 108,
    "fov_deg": 90.0,
    "fps": 10.0,
    "transform": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": -5.0},
    "intrinsics": {"fx": 96.0, "fy": 96.0, "cx": 96.0, "cy": 54.0}
  },
  "weather": {
    "cloudiness": 0.0,
    "precipitation": 0.0,
    "sun_altitude_angle": 45.0,
    ...
  },
  "semantic_colors": {"0": [0,0,0], "7": [128,64,128], ...},
  "training_contract": {
    "imitation_observation": ["seg_label_or_seg_color", "speed_mps", ...],
    "action": ["steer", "longitudinal"],
    "reward_or_metrics_only": ["normalized_lane_offset", ...]
  }
}
```

---

### File `states.csv` — Mô tả các cột

#### Nhóm: Định danh

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `session_id` | str | ID của session hiện tại |
| `episode_id` | str | ID world của CARLA |
| `sample_id` | int | Số thứ tự mẫu trong session (0-indexed) |
| `frame` | int | Số frame CARLA (dùng để khớp tên file ảnh) |
| `sim_time_s` | float | Thời gian simulation (giây) |
| `delta_seconds` | float | Bước thời gian simulation |
| `sample_delta_seconds` | float | Khoảng thời gian giữa 2 mẫu liên tiếp |
| `wall_time_utc` | str | Thời điểm ghi (UTC) |
| `map_name` | str | Tên map đầy đủ |
| `vehicle_id` | int | Actor ID của xe hero |
| `vehicle_type` | str | Loại xe (blueprint ID) |

#### Nhóm: Đường dẫn ảnh

| Cột | Ý nghĩa |
|---|---|
| `rgb_path` | Đường dẫn tương đối ảnh RGB (rỗng nếu không thu) |
| `seg_label_path` | Đường dẫn ảnh class ID 1 kênh |
| `seg_color_path` | Đường dẫn ảnh semantic màu 3 kênh (rỗng nếu không thu) |

#### Nhóm: Vị trí và hướng xe

| Cột | Đơn vị | Ý nghĩa |
|---|---|---|
| `x`, `y`, `z` | m | Tọa độ world |
| `roll_deg`, `pitch_deg`, `yaw_deg` | độ | Góc Euler |

#### Nhóm: Vận tốc và gia tốc

| Cột | Đơn vị | Ý nghĩa |
|---|---|---|
| `velocity_x/y/z` | m/s | Vector vận tốc theo các trục world |
| `speed_mps` | m/s | Tốc độ tuyến tính (độ lớn) |
| `speed_kmh` | km/h | Tốc độ theo km/h |
| `forward_speed_mps` | m/s | Vận tốc theo hướng trước xe |
| `lateral_speed_mps` | m/s | Vận tốc theo hướng bên xe |
| `distance_travelled_m` | m | Tổng quãng đường đã đi trong session |
| `accel_x/y/z` | m/s² | Vector gia tốc |
| `accel_mps2` | m/s² | Gia tốc tuyến tính (độ lớn) |
| `longitudinal_accel_mps2` | m/s² | Gia tốc dọc theo hướng xe |
| `lateral_accel_mps2` | m/s² | Gia tốc ngang |
| `angular_x/y/z_deg_s` | độ/s | Vận tốc góc theo các trục |
| `yaw_rate_rps` | rad/s | Tốc độ xoay yaw (rad/giây) |

#### Nhóm: Điều khiển xe (nhãn chuyên gia)

| Cột | Phạm vi | Ý nghĩa |
|---|---|---|
| `steer` | [-1, 1] | Lái (âm = trái, dương = phải) |
| `throttle` | [0, 1] | Ga |
| `brake` | [0, 1] | Phanh |
| `hand_brake` | 0/1 | Phanh tay |
| `reverse` | 0/1 | Lùi xe |
| `gear` | int | Số (gear) hiện tại |
| `longitudinal` | [-1, 1] | `throttle - brake` — nhãn dọc tổng hợp |
| `previous_steer` | [-1, 1] | Steer của frame trước |
| `previous_longitudinal` | [-1, 1] | Longitudinal của frame trước |
| `steer_delta` | float | Thay đổi steer so với frame trước |
| `longitudinal_delta` | float | Thay đổi longitudinal so với frame trước |

#### Nhóm: Thông tin môi trường

| Cột | Ý nghĩa |
|---|---|
| `speed_limit_kmh` | Giới hạn tốc độ tại vị trí hiện tại |
| `traffic_light_state` | Trạng thái đèn giao thông (`Green`, `Red`, `Yellow`, `Unknown`) |

#### Nhóm: Trạng thái làn đường

| Cột | Đơn vị | Ý nghĩa |
|---|---|---|
| `waypoint_id` | str | ID waypoint CARLA hiện tại |
| `road_id` | int | ID đường OpenDRIVE |
| `section_id` | int | ID section OpenDRIVE |
| `lane_id` | int | ID làn đường (âm = bên phải theo quy tắc CARLA) |
| `waypoint_s` | m | Tọa độ `s` dọc theo đường |
| `lane_width_m` | m | Chiều rộng làn |
| `is_junction` | 0/1 | Xe đang ở giao lộ hay không |
| `junction_id` | int | ID giao lộ (nếu có) |
| `lane_type` | str | Loại làn (`Driving`, `Sidewalk`, v.v.) |
| `lane_change` | str | Hướng cho phép đổi làn (`Left`, `Right`, `Both`, `None`) |
| `lane_offset_m` | m | Khoảng lệch ngang so với tâm làn (+= phải, -= trái) |
| `normalized_lane_offset` | [-1, 1] | `lane_offset_m / (lane_width/2)` |
| `heading_error_deg` | độ | Góc lệch hướng xe so với hướng làn |
| `heading_error_rad` | rad | Tương tự, đơn vị radian |
| `off_lane` | 0/1 | Xe đang ra ngoài làn (|offset| > half_width) |

#### Nhóm: Waypoint nhìn trước (lookahead)

| Cột | Ý nghĩa |
|---|---|
| `waypoint_x/y/z` | Tọa độ world của waypoint hiện tại |
| `waypoint_yaw_deg` | Hướng của waypoint hiện tại |
| `waypoint_local_x/y` | Tọa độ waypoint trong hệ tọa độ xe |
| `next_waypoint_*` | Thông tin waypoint tiếp theo (cách `lookahead_m`) |
| `next_candidate_count` | Số nhánh khả thi tại waypoint hiện tại |
| `successor_waypoints_json` | JSON list các waypoint kế tiếp (ở graph resolution) |
| `lookahead_waypoints_json` | JSON list waypoints tại 5m, 10m, 20m, 30m phía trước |

#### Nhóm: Làn kề

| Cột | Ý nghĩa |
|---|---|
| `left_waypoint_id` | ID waypoint làn trái |
| `left_road/section/lane_id` | OpenDRIVE coords làn trái |
| `left_lane_type` | Loại làn trái |
| `right_*` | Tương tự cho làn phải |

#### Nhóm: Đích (goal) — khi dùng A*

| Cột | Ý nghĩa |
|---|---|
| `goal_waypoint_id` | ID waypoint đích |
| `goal_x/y/z` | Tọa độ world đích |
| `goal_local_x/y` | Tọa độ đích trong hệ tọa độ xe |
| `goal_euclidean_distance_m` | Khoảng cách Euclidean từ xe đến đích |

#### Nhóm: Route A* (dành sẵn, để trống đến khi planner được nối vào)

| Cột | Ý nghĩa |
|---|---|
| `route_id` | ID route/goal selection |
| `route_target_index` | Index nút đang nhắm đến trong route |
| `route_target_waypoint_id` | Waypoint đang nhắm đến |
| `route_target_local_x/y` | Vị trí mục tiêu trong hệ tọa độ xe |
| `route_command` | Lệnh điều hướng: `LANEFOLLOW`, `LEFT`, `RIGHT`, `STRAIGHT`, `CHANGELANELEFT`, `CHANGELANERIGHT` |
| `route_progress_m` | Quãng đường đã đi theo route |
| `route_remaining_m` | Quãng đường còn lại theo route |
| `route_total_m` | Tổng độ dài route |
| `route_completed` | 1 khi đã đến đích |

#### Nhóm: Chẩn đoán (không dùng làm observation)

| Cột | Ý nghĩa |
|---|---|
| `collision_count` | Số va chạm tích lũy trong session |
| `lane_invasion_count` | Số lần vượt vạch làn tích lũy |

---

### Semantic class ID và màu sắc

| Class ID | Nhãn | Màu RGB |
|---|---|---|
| 0 | Unlabeled | (0, 0, 0) |
| 1 | Building | (70, 70, 70) |
| 2 | Fence | (100, 40, 40) |
| 3 | Other | (55, 90, 80) |
| 4 | Pedestrian | (220, 20, 60) |
| 5 | Pole | (153, 153, 153) |
| 6 | RoadLine | (157, 234, 50) |
| 7 | Road | (128, 64, 128) |
| 8 | SideWalk | (244, 35, 232) |
| 9 | Vegetation | (107, 142, 35) |
| 10 | Vehicles | (0, 0, 142) |
| 11 | Wall | (102, 102, 156) |
| 12 | TrafficSign | (220, 220, 0) |

> [!WARNING]
> Không dùng `ColorJitter`, `hue`, hay `saturation` augmentation trên ảnh `seg_color`. Bảng màu này là cố định và mang ý nghĩa semantic — biến đổi màu sẽ phá vỡ thông tin class.

---

## 7. Kiểm tra và xác thực session

### Kiểm tra một session

```powershell
python verify_dataset.py D:\CARLA_DATA\Town01_20260715_230000_123456 --strict
```

**Kết quả tốt:**
```json
{
  "session": "D:\\CARLA_DATA\\Town01_20260715_230000_123456",
  "rows": 50000,
  "unique_frames": 50000,
  "first_frame": 1000,
  "last_frame": 68000,
  "astar_graph_exported": false,
  "astar_nodes": 0,
  "astar_edges": 0,
  "errors": []
}
```

Điều kiện "kết quả tốt":
- `rows == unique_frames` — không có frame trùng lặp
- `errors: []` — không có lỗi
- Khi bật `--map-export`: `astar_nodes > 0` và `astar_edges > 0`

### Những gì `verify_dataset.py` kiểm tra

1. Sự tồn tại của `metadata.json`
2. Toàn bộ file A* artifacts (nếu graph đã được xuất)
3. Tất cả các cột training fields có trong CSV
4. Không có frame trùng lặp
5. File ảnh `seg_label` tồn tại và khớp với CSV
6. File `rgb` và `seg_color` tồn tại (nếu có đường dẫn)
7. Ảnh `seg_label` là 1 kênh (grayscale)
8. Class ID trong phạm vi [0, 12] (khi `--strict`)
9. `waypoint_id` không rỗng
10. `successor_waypoints_json` và `lookahead_waypoints_json` là JSON list hợp lệ

---

## 8. Tạo manifest và chia tập dữ liệu

Sau khi thu đủ tất cả các map, tạo manifest để chia train/val/test:

### Chia theo tỉ lệ

```powershell
python build_manifest.py D:\CARLA_DATA --train-ratio 0.70 --val-ratio 0.15 --seed 42
```

### Giữ toàn bộ một map làm test (cross-map generalization)

```powershell
python build_manifest.py D:\CARLA_DATA --holdout-map Town03 --seed 42
```

### Kết quả tạo ra

```
D:\CARLA_DATA\
├── manifest.csv            ← toàn bộ dataset, thêm cột "split" và "session_dir"
├── dataset_summary.json    ← thống kê tổng quan
├── train_sessions.txt      ← danh sách session dùng để train
├── val_sessions.txt        ← danh sách session validation
└── test_sessions.txt       ← danh sách session test
```

`dataset_summary.json` chứa:
- Tổng số session và phân bổ theo split
- Số mẫu theo split và map
- Phân bố hành động: `steer_negative`, `steer_positive`, `steer_near_zero`, `braking`, `junction`

> [!IMPORTANT]
> Việc chia tập phải **theo session** (không random từng frame). Các frame trong cùng một session liên tiếp về thời gian — chia ngẫu nhiên từng frame sẽ gây rò rỉ thông tin từ train vào validation.

---

## 9. Sử dụng dữ liệu cho huấn luyện

### 9.1 Imitation Learning (IL)

**Observation đầu vào:**
```python
observation = {
    "image": seg_label,          # one-hot 13 lớp hoặc seg_color 3 kênh
    "speed_mps": ...,
    "yaw_rate_rps": ...,
    "previous_steer": ...,
    "previous_longitudinal": ...,
}
```

**Nhãn (action):**
```python
action = [steer, longitudinal]   # longitudinal = throttle - brake
```

> [!WARNING]
> **Không** đưa class ID của `seg_label` vào CNN như cường độ xám có thứ tự — class ID là nhãn rời rạc, không có mối quan hệ thứ tự số học. Hãy one-hot 13 lớp, dùng embedding, hoặc gom thành các nhóm semantic.

### 9.2 Fine-tune DRL

Khởi tạo actor từ checkpoint IL, giữ nguyên observation/action. Dữ liệu IL không cần reward — `CarlaEnv.step()` tính reward trực tiếp từ trạng thái môi trường:

```
r = + tốc độ tiến hợp lệ
    - |lane_offset_m|
    - |heading_error_rad|
    - |steer_delta|
    - |longitudinal_delta|
    - |yaw_rate_rps|
```

> [!IMPORTANT]
> Không đưa `lane_offset_m` và `heading_error_rad` vào policy observation. Chỉ dùng chúng làm **reward**, auxiliary target và metric — nếu policy thấy chúng, nó học cách minimize metric thay vì học quan sát từ ảnh segmentation.

---

## 10. Mở rộng A*

### Bước 1: Xuất graph

```powershell
# Chạy collector một lần trên mỗi map với --map-export
python collect_data.py --config collector_config.json --map-export --max-samples 100
```

Tạo ra:
- `map.xodr` — bản đồ OpenDRIVE đầy đủ
- `spawn_points.csv` — các điểm xuất phát
- `map_nodes.csv` — đỉnh của đồ thị (mỗi node là một waypoint)
- `map_edges.csv` — cạnh có hướng của đồ thị

### Bước 2: Hiểu cấu trúc đồ thị

**`map_nodes.csv`:**

| Cột | Ý nghĩa |
|---|---|
| `node_id` | Khóa duy nhất của node |
| `road_id/section_id/lane_id/s` | OpenDRIVE coordinates |
| `x/y/z/yaw_deg` | Vị trí và hướng trong world |
| `is_junction/junction_id` | Có phải giao lộ không |
| `left_node_id/right_node_id` | Node làn kề |

**`map_edges.csv`:**

| Loại cạnh | Ý nghĩa |
|---|---|
| `LANE_FOLLOW` | Đi thẳng trong làn |
| `JUNCTION_BRANCH` | Nhánh tại giao lộ |
| `LANE_CHANGE_LEFT` | Đổi sang làn trái |
| `LANE_CHANGE_RIGHT` | Đổi sang làn phải |

### Bước 3: Thuật toán A*

```
g(next) = g(current) + edge.cost_m
h(node) = EuclideanDistance(node, goal)
f(node) = g(node) + h(node)
```

- Tìm node bắt đầu gần `waypoint_id` hiện tại trong `states.csv`
- Tìm node đích từ `goal_waypoint_id` hoặc `goal.json`
- Kết quả: danh sách `node_id` theo thứ tự

> [!NOTE]
> A* chịu trách nhiệm **lập kế hoạch đường toàn cục**; policy IL/DRL chịu trách nhiệm **điều khiển cục bộ mượt**. Không dùng A* để trực tiếp sinh `steer/throttle/brake`.

---

## 11. Tham số tham khảo

### Cấu hình khuyến nghị cho training

```
Resolution camera : 800 × 450
FPS               : 10
FOV               : 90°
Camera offset     : x=1.5 m, z=2.4 m, pitch=-5°
Lookahead         : 5 m
Route lookaheads  : 5, 10, 20, 30 m
Graph resolution  : 2 m
Lane-change cost  : 3.0×
Max samples/map   : 25.000 – 50.000
```

### Mục tiêu thu thập gợi ý

| Map | Số session | Tổng mẫu |
|---|---|---|
| Town01 | ≥ 3 | ~50.000 |
| Town02 | ≥ 3 | ~50.000 |
| Town03 | ≥ 3 | ~50.000 |
| Town04 | ≥ 3 | ~50.000 |
| **Tổng** | **≥ 12** | **~200.000** |

> [!TIP]
> Tối thiểu **3 session mỗi map** để map đó xuất hiện trong cả train/validation/test. Nên có nhiều session ngắn thay vì một session rất dài.

### Đa dạng hóa dữ liệu

Để tăng tính tổng quát, thu thập ở nhiều điều kiện:

```python
# Các preset thời tiết nên dùng
carla.WeatherParameters.ClearNoon
carla.WeatherParameters.CloudyNoon
carla.WeatherParameters.WetNoon
carla.WeatherParameters.MidRainSunset
carla.WeatherParameters.SoftRainNoon
```

Tình huống cần có trong dataset:
- ✅ Đường thẳng dài
- ✅ Đường cong
- ✅ Giao lộ (nhiều nhánh)
- ✅ Nhiều mức tốc độ
- ✅ Nhiều thời tiết và thời điểm trong ngày
- ✅ Một phần phục hồi khi xe lệch khỏi tâm làn (DAgger — giai đoạn sau)

---

## 12. Xử lý sự cố thường gặp

### Không tìm thấy xe hero

**Triệu chứng:**
```
Khong tim thay ego vehicle. Chay automatic_control.py truoc...
```

**Giải pháp:**
1. Chắc chắn `automatic_control.py` đang chạy và xe đã xuất hiện trong CARLA
2. Kiểm tra `role_name` trong config khớp với script (`hero` mặc định)
3. Thử tăng `wait_vehicle_timeout` lên 180 giây
4. Thử chỉ định trực tiếp: `--vehicle-id <actor_id>`

---

### Collector kết nối được nhưng không ghi dữ liệu

**Nguyên nhân:** Camera chưa nhận được frame hoặc state builder lỗi.

**Giải pháp:**
1. Kiểm tra `queue_size` không quá nhỏ (tăng lên 128)
2. Giảm `fps` xuống 5 để camera kịp đồng bộ
3. Xem log: `samples=0 | queue=0 | dropped=N` → N frame dropped lớn = bottleneck I/O

---

### File `summary.json` báo `samples_written: 0`

**Nguyên nhân:** Session bị hủy trước khi bất kỳ frame nào được ghi đồng bộ.

**Giải pháp:**
1. Đảm bảo xe hero đang di chuyển khi collector bắt đầu
2. Thử `--duration 60` và quan sát output log

---

### Map tải xong nhưng xe không xuất hiện

**Giải pháp:**
1. Dừng `automatic_control.py` hoàn toàn (Ctrl+C, chờ terminal thoát hẳn)
2. Chạy lại `automatic_control.py` sau khi đổi map
3. Đợi ít nhất 3-5 giây để xe spawn xong

---

### verify_dataset.py báo lỗi `frame trùng`

**Nguyên nhân:** Hiếm gặp — CARLA tái sử dụng frame ID sau khi reset.

**Giải pháp:** Bỏ session đó và thu lại; đây là dữ liệu không đáng tin cậy.

---

### Lỗi khi chạy `build_manifest.py`

```
Schema khac nhau tai .../states.csv
```

**Nguyên nhân:** Session được thu với phiên bản pipeline khác nhau.

**Giải pháp:** Chỉ gộp các session thu bằng cùng một phiên bản code. Xóa session cũ hoặc tách riêng manifest.

---

## Tóm tắt quy trình nhanh

```
[1] Khởi động CARLA Server
        ↓
[2] Notebook: load_world("TownXX") + đặt thời tiết
        ↓
[3] Chạy automatic_control.py — đợi xe chạy
        ↓
[4] Chạy run_collector.bat — đợi đủ mẫu
        ↓
[5] verify_dataset.py <session_dir> --strict
        ↓
[6] Lặp lại từ [2] cho map tiếp theo
        ↓
[7] build_manifest.py D:\CARLA_DATA --train-ratio 0.70 --val-ratio 0.15 --seed 42
```

---

*Tài liệu được tạo tự động từ mã nguồn tại `Data_Collection/` — 2026-07-24*
