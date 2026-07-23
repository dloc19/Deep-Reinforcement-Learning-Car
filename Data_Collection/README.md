# Pipeline thu thập dữ liệu CARLA 0.9.10

Pipeline này **không spawn xe, không bật autopilot và không đổi map**. Nó là một client thụ động: tìm chiếc xe `hero` đã được `automatic_control.py` tạo, gắn semantic camera và ghi dữ liệu theo cùng `frame` của CARLA. Mặc định `seg-only` không spawn camera RGB để giảm tải GPU và dung lượng.

## Cấu trúc source code theo module

```text
carla_data_pipeline/
├── collect_data.py                 # entrypoint, chạy file này
├── collector_config.json           # cấu hình dùng chung cho từng map
├── run_collector.bat               # chạy nhanh trên Windows
├── verify_dataset.py               # kiểm tra một session
├── build_manifest.py               # gộp và chia nhiều session
└── carla_collector/
    ├── config.py                   # tham số CLI và validation
    ├── schema.py                   # schema CSV cho IL, DRL và trường A* dự phòng
    ├── geometry.py                 # tọa độ, góc, waypoint helper
    ├── events.py                   # collision/lane-invasion counter
    ├── synchronizer.py             # ghép camera/state theo frame
    ├── writer.py                   # lưu PNG và states.csv
    ├── sensors.py                  # spawn/destroy sensor
    ├── state_builder.py            # trạng thái xe, làn, goal
    ├── map_export.py               # OpenDRIVE và graph A*
    ├── metadata.py                 # metadata.json
    └── collector.py                # điều phối toàn bộ pipeline
```

Vị trí cần sửa thường gặp:

| Muốn thay đổi | File cần sửa |
| --- | --- |
| Thêm/bớt cột CSV | `carla_collector/schema.py` |
| Đổi camera, vị trí camera | `config.py`, `sensors.py` |
| Đổi cách lưu segmentation/RGB | `writer.py` |
| Thêm trạng thái hoặc nhãn học | `state_builder.py` |
| Đổi graph/cost A* | `map_export.py` |
| Đổi cách đồng bộ frame | `synchronizer.py` |
| Đổi trình tự kết nối/chạy/dừng | `collector.py` |

Không cần sửa `collect_data.py` trừ khi muốn đổi cách khởi động chương trình.

## Chạy bằng file cấu hình

File `collector_config.json` mặc định lưu đồng thời `seg_label` và `seg_color`,
thu đúng **25.000 mẫu hợp lệ mỗi map** rồi tự dừng. Collector vẫn chỉ spawn một
semantic camera, không spawn RGB camera. Ở 10 FPS, thời gian lý thuyết khoảng
41 phút 40 giây/map. Bốn map sẽ cho khoảng 100.000 mẫu trước khi lọc.
Graph A* mặc định chưa được xuất vì A* là giai đoạn mở rộng sau IL và DRL.

Những dòng thường cần chỉnh:

```json
{
  "dataset": {
    "output": "D:/CARLA_DATA"
  },
  "camera": {
    "image_mode": "seg-only",
    "save_seg_color": true,
    "width": 800,
    "height": 450,
    "fps": 10.0
  },
  "collection": {
    "max_samples": 25000,
    "duration": 0.0
  },
  "astar": {
    "no_map_export": true
  }
}
```

- `max_samples > 0`: tự dừng khi đã ghi đủ số mẫu đồng bộ.
- `duration > 0`: tự dừng theo số giây thực tế.
- Nếu cả hai lớn hơn 0, điều kiện nào đạt trước sẽ dừng trước.
- JSON không hỗ trợ comment; không chèn dòng bắt đầu bằng `#` hoặc `//`.

Chạy từ Anaconda Prompt/PowerShell đã cài CARLA Python API:

```powershell
cd <duong_dan>\carla_data_pipeline
run_collector.bat
```

Hoặc chạy trực tiếp:

```powershell
python collect_data.py --config collector_config.json
```

Có thể ghi đè tạm thời mà không sửa JSON:

```powershell
python collect_data.py --config collector_config.json --max-samples 10000
```

### Chu kỳ chạy từng map

Giữ CARLA server mở trong toàn bộ quá trình. Với mỗi map, thực hiện đúng thứ tự:

```text
1. Notebook: chạy duy nhất cell client.load_world("Town01").
2. Đợi map tải xong.
3. Terminal A: chạy automatic_control.py, đợi hero bắt đầu chạy.
4. Terminal B: chạy run_collector.bat.
5. Đủ 25.000 mẫu, collector tự cleanup sensor và dừng.
6. Kiểm tra session bằng verify_dataset.py.
7. Dừng automatic_control.py bằng Ctrl+C.
8. Notebook: chạy cell client.load_world("Town02").
9. Chạy lại automatic_control.py để tạo hero mới.
10. Chạy lại run_collector.bat với cùng config.
```

Tiếp tục tương tự với Town03 và Town04. Không chạy cell đổi map khi collector còn
hoạt động, vì `client.load_world()` hủy world và toàn bộ actor/sensor của map cũ.

Cell notebook nên viết như sau để nhìn rõ map hiện tại:

```python
import time

world = client.load_world("Town01")
time.sleep(3)
world.set_weather(carla.WeatherParameters.ClearNoon)  # đổi preset nếu cần
print("Map hiện tại:", world.get_map().name)
```

Phải đặt thời tiết **sau** `load_world()`, vì đổi map tạo một `World` mới. Collector
tự đọc thời tiết hiện tại và ghi đầy đủ vào `metadata.json`; không tự thay đổi thời tiết.

## 1. Dữ liệu được lưu

Mỗi lần chạy tạo một session riêng:

```text
dataset/
└── Town01_20260715_230000_123456/
    ├── metadata.json
    ├── summary.json
    ├── states.csv
    ├── seg_label/00001234.png
    ├── rgb/00001234.png             # chỉ khi --image-mode seg-rgb
    └── seg_color/00001234.png       # chỉ khi --save-seg-color
```

- `rgb`: ảnh RGB tùy chọn để đối chiếu.
- `seg_label`: ảnh một kênh, mỗi pixel là semantic class ID `0..12`; đây là input
  tiết kiệm dung lượng và tính toán nhất.
- `seg_color`: ảnh semantic ba kênh theo bảng màu cố định; có thể dùng để train
  backbone CNN hoặc dùng kiểm tra trực quan. Không được dùng ColorJitter/hue/saturation.
- `states.csv`: mỗi dòng khớp đúng một bộ ảnh qua cột `frame`.
- `metadata.json`: map, xe, thời tiết, cấu hình camera, camera intrinsics và thông tin graph.
- Khi bật `--map-export`, session có thêm `map.xodr`, `spawn_points.csv`,
  `map_nodes.csv`, `map_edges.csv` và `map_graph_metadata.json` cho A* sau này.

Xem `ASTAR_SCHEMA.md` trong gói mã nguồn để biết data contract của planner và route tracker.

CSV gồm:

- Nhãn điều khiển chuyên gia: `steer`, `throttle`, `brake` và
  `longitudinal = throttle - brake`.
- Hành động trước và độ thay đổi: `previous_steer`, `previous_longitudinal`,
  `steer_delta`, `longitudinal_delta`.
- Trạng thái xe: vị trí, góc quay, vận tốc, gia tốc, vận tốc góc, giới hạn tốc độ và đèn giao thông.
- Trạng thái bám làn: `lane_offset_m`, `normalized_lane_offset`,
  `heading_error_rad`, `off_lane`, `road_id`, `section_id`, `lane_id`,
  `waypoint_s` và độ rộng làn.
- Dữ liệu route/A*: ID waypoint hiện tại, OpenDRIVE `(road_id, section_id, lane_id, s)`,
  junction, làn trái/phải, mọi successor ở bước graph, các waypoint nhìn trước 5/10/20/30 m,
  tọa độ local theo xe và destination tùy chọn.
- `collision_count` và `lane_invasion_count` chỉ là thông tin chẩn đoán tùy chọn;
  pipeline không triển khai tránh vật cản và không đưa chúng vào observation.

Các cột `route_*` đã được dành sẵn cho output của A*. Trước khi planner được nối vào,
chúng để trống có chủ ý; collector không tự đoán nhánh rẽ tại giao lộ.

## Dữ liệu dùng để huấn luyện

Imitation learning nên sử dụng:

```text
Observation:
    seg_label one-hot/embedding hoặc seg_color 3 kênh
    speed_mps
    yaw_rate_rps
    previous_steer
    previous_longitudinal

Expert action:
    steer
    longitudinal
```

Có thể đổi nhãn dọc thành một biến liên tục để dùng chung actor head với DRL:

```text
longitudinal = throttle - brake
action = [steer, longitudinal]
```

Không đưa class ID của `seg_label` vào CNN như cường độ xám có thứ tự. Hãy one-hot
13 lớp, dùng embedding, hoặc gom thành các nhóm semantic. Nếu dùng `seg_color`,
giữ nguyên bảng màu và chuẩn hóa ba kênh nhất quán.

Fine-tune DRL online không cần reward lưu trong tập imitation. `CarlaEnv.step()`
sau này tính reward từ tốc độ tiến, lane offset, heading error, steer delta,
longitudinal delta và yaw rate; đồng thời trả `next_observation` và `done`.
Không đưa `lane_offset_m` và `heading_error_rad` vào policy nếu muốn mạng thực sự
học quan sát làn từ segmentation; chỉ dùng chúng làm reward, auxiliary target và metric.

## 2. Cài môi trường

Mở PowerShell/Anaconda Prompt bằng đúng Python dùng với CARLA:

```powershell
cd C:\CARLA_0.9.10\PythonAPI\carla\dist
pip install carla-0.9.10-py3.7-win-amd64.egg
pip install numpy==1.19.5 Pillow
```

Nếu bản CARLA của bạn cung cấp `.whl`, cài file `.whl` thay cho `.egg`. Kiểm tra:

```powershell
python -c "import carla; print('CARLA Python API OK')"
```

## 3. Thứ tự chạy đúng theo yêu cầu

### Cửa sổ 1 — mở CARLA server

```powershell
cd C:\CARLA_0.9.10\WindowsNoEditor
CarlaUE4.exe -quality-level=Low
```

Bạn có thể chọn map từ notebook như đang làm:

```python
import carla
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
client.load_world("Town01")
```

Sau khi `load_world`, chờ map tải xong rồi mới chạy bước tiếp theo.

### Cửa sổ 2 — spawn xe và bật autopilot

Trong thư mục `PythonAPI/examples`:

```powershell
python automatic_control.py --host 127.0.0.1 --port 2000
```

Đợi xe xuất hiện và bắt đầu chạy. Bản `automatic_control.py` chuẩn đặt `role_name=hero`, collector sẽ tự tìm xe này.

### Cửa sổ 3 — bắt đầu thu thập

```powershell
cd <thu_muc_pipeline>
python collect_data.py --output D:\CARLA_DATA --image-mode seg-only --fps 10 --width 800 --height 450
```

Đây là lệnh tối giản: chỉ một semantic camera và class-mask PNG một kênh. Cấu hình
JSON mặc định còn bật `save_seg_color=true`, nên khi chạy bằng `run_collector.bat`
sẽ lưu đồng thời label một kênh và ảnh semantic màu. Nếu cần thêm RGB để đối chiếu:

```powershell
python collect_data.py --output D:\CARLA_DATA --image-mode seg-rgb --save-seg-color
```

Trong giai đoạn IL/DRL hiện tại, `collector_config.json` đặt
`"no_map_export": true`. Khi bắt đầu phần A*, chỉ cần đổi thành `false` hoặc chạy:

```powershell
python collect_data.py --output D:\CARLA_DATA --map-export
```

Nếu muốn ghi sẵn một đích là spawn point 12 trong giai đoạn A*:

```powershell
python collect_data.py --output D:\CARLA_DATA --goal-spawn-index 12
```

Hoặc chọn đích theo tọa độ CARLA; collector sẽ chiếu nó lên Driving waypoint gần nhất:

```powershell
python collect_data.py --output D:\CARLA_DATA --goal-x 85.0 --goal-y 12.5 --goal-z 0
```

Điều chỉnh mật độ graph và chi phí chuyển làn:

```powershell
python collect_data.py --map-export --graph-resolution 2.0 --lane-change-cost 3.0
```

`lane-change-cost=3.0` khiến A* ưu tiên đi tiếp trong làn, chỉ chuyển làn khi route cần.

Nhấn `Ctrl+C` để dừng. Collector chỉ hủy các sensor do chính nó tạo; xe/autopilot vẫn thuộc cửa sổ 2.

Chạy thử 60 giây:

```powershell
python collect_data.py --output D:\CARLA_DATA --duration 60 --fps 10
```

Thu đúng 5.000 mẫu:

```powershell
python collect_data.py --output D:\CARLA_DATA --max-samples 5000 --fps 10
```

Nếu không tìm thấy xe `hero`, lấy actor ID hoặc dùng role khác:

```powershell
python collect_data.py --vehicle-id 123 --output D:\CARLA_DATA
python collect_data.py --role-name ego_vehicle --output D:\CARLA_DATA
```

### Lặp lại qua nhiều map

```text
1. Mở CARLA server một lần.
2. Notebook: client.load_world("Town01").
3. Chạy automatic_control.py, đợi xe bắt đầu chạy.
4. Chạy collect_data.py, thu một session rồi Ctrl+C.
5. Dừng automatic_control.py.
6. Notebook: client.load_world("Town02") (world cũ và actor cũ bị hủy).
7. Chạy lại automatic_control.py để tạo hero mới.
8. Chạy lại collect_data.py.
9. Tiếp tục Town03/Town04 và các weather khác.
```

Mỗi lần chạy collector tạo một thư mục `TownXX_timestamp`, không ghi đè session cũ.
Nên có nhiều session ngắn thay vì một session rất dài; tối thiểu 3 session cho mỗi
map nếu muốn map đó xuất hiện trong cả train/validation/test.

## 4. Kiểm tra session sau khi thu

```powershell
python verify_dataset.py D:\CARLA_DATA\Town01_20260715_230000_123456 --strict
```

Kết quả tốt phải có `rows == unique_frames` và `errors: []`. Khi bật xuất graph
A*, kết quả cũng phải có `astar_nodes > 0` và `astar_edges > 0`.

Sau khi đã thu đủ tất cả map, tạo manifest và chia theo **session**, không chia ngẫu
nhiên từng frame:

```powershell
python build_manifest.py D:\CARLA_DATA --train-ratio 0.70 --val-ratio 0.15 --seed 42
```

Kết quả:

```text
D:\CARLA_DATA\manifest.csv
D:\CARLA_DATA\dataset_summary.json
D:\CARLA_DATA\train_sessions.txt
D:\CARLA_DATA\val_sessions.txt
D:\CARLA_DATA\test_sessions.txt
```

Để đánh giá khả năng tổng quát sang map chưa thấy, có thể giữ toàn bộ Town03 làm test:

```powershell
python build_manifest.py D:\CARLA_DATA --holdout-map Town03 --seed 42
```

## 5. Cách dùng cho các giai đoạn sau

### Imitation learning ban đầu

Đầu vào là `seg_label` one-hot/embedding hoặc `seg_color`, cùng `speed_mps`,
`yaw_rate_rps`, `previous_steer` và `previous_longitudinal`. Nhãn chính là
`[steer, longitudinal]`. Nên chia train/val/test **theo session**, không random
từng frame, để tránh các frame liên tiếp rò rỉ sang validation.

Không nên chỉ thu autopilot chạy hoàn hảo ở giữa làn. Hãy thu nhiều session với:

- Town01, Town02, Town03; nhiều thời tiết và thời gian trong ngày.
- Đường thẳng, đường cong, giao lộ và nhiều mức tốc độ khác nhau.
- Một phần dữ liệu phục hồi khi xe lệch nhẹ khỏi tâm làn. Phần này cần can thiệp có kiểm soát hoặc DAgger về sau; không tự làm lệch xe trên tập thu đầu tiên.

### Fine-tune DRL

Khởi tạo actor từ mô hình imitation và giữ nguyên observation/action. Dùng
`lane_offset_m` và `heading_error_rad` để tính reward, không đưa chúng vào observation.
Reward gợi ý:

```text
r = + tốc độ tiến hợp lệ
    - |lane_offset|
    - |heading_error|
    - độ giật steer
    - độ giật ga/phanh
    - |yaw_rate|
```

### Mở rộng A*

Khi chuyển sang A*, chạy collector một lần trên mỗi map với `--map-export` để xuất:

```text
map_nodes.csv
    node_id
    road_id, section_id, lane_id, s
    x, y, z, yaw
    junction_id
    left_node_id, right_node_id

map_edges.csv
    from_node_id, to_node_id
    edge_type
    distance_m
    cost_m
    yaw_delta_deg
```

A* tìm node bắt đầu gần `waypoint_id` hiện tại và node đích từ `goal_waypoint_id`,
dùng `cost_m` làm `g(n)` và khoảng cách Euclidean đến đích làm heuristic `h(n)`.
Kết quả planner là danh sách `node_id`. Module route-tracker sau này cập nhật:

```text
route_id
route_target_index
route_target_waypoint_id
route_target_local_x, route_target_local_y
route_command = LANEFOLLOW | LEFT | RIGHT | STRAIGHT | CHANGELANELEFT | CHANGELANERIGHT
route_progress_m
route_remaining_m
route_total_m
route_completed
```

Không đưa trực tiếp tọa độ world `x/y` vào policy. Hãy dùng `route_target_local_x/y`
đã biến đổi sang hệ tọa độ xe để mô hình tổng quát giữa các map.

Lưu ý: A* chịu trách nhiệm **lập kế hoạch đường toàn cục**; policy imitation/DRL chịu trách nhiệm **điều khiển cục bộ mượt**. Không dùng A* để trực tiếp sinh `steer/throttle/brake`.

## 6. Lưu ý đồng bộ

Pipeline không gọi `world.tick()` vì `automatic_control.py` là client khác. Dữ liệu được ghép bằng chính `image.frame` và `WorldSnapshot.frame`, nên chỉ ghi khi đủ RGB + segmentation + state cùng frame. Điều này tránh ghép nhầm ảnh và trạng thái dù camera GPU có trễ vài frame.

Không bật synchronous mode riêng trong collector. Nếu sau này muốn thu hoàn toàn deterministic ở synchronous mode, phải chuyển sang một script điều phối duy nhất làm tick cho cả Traffic Manager, ego và sensor; CARLA khuyến cáo chỉ một client được quyền tick.

## 7. Tham số nên dùng ban đầu

```text
resolution : 800 x 450
fps        : 10
FOV        : 90 độ
camera     : x=1.5 m, z=2.4 m, pitch=-5 độ
lookahead  : 5 m
route lookahead: 5, 10, 20, 30 m
graph resolution: 2 m
```

Ước lượng dung lượng thực tế phụ thuộc cảnh và mức nén PNG. Hãy chạy thử 5 phút, xem `summary.json` và dung lượng session rồi mới thu hàng giờ.
