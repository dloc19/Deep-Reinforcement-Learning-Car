# Pipeline thu thập dữ liệu CARLA 0.9.10

Pipeline này **không spawn xe, không bật autopilot và không đổi map**. Nó là một client thụ động: tìm chiếc xe `hero` đã được `automatic_control.py` tạo, gắn semantic camera và ghi dữ liệu theo cùng `frame` của CARLA. Mặc định CLI (`seg-only`) không spawn camera RGB để giảm tải GPU và dung lượng; `collector_config.json` dùng để thu dữ liệu thật lại đặt `seg-rgb`, spawn thêm camera RGB để lưu đối chiếu.

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

File `collector_config.json` mặc định lưu đồng thời `rgb`, `seg_label` và
`seg_color` (vì `image_mode: seg-rgb` và `save_seg_color: true`), ở độ phân giải
480×384, thu đúng **10.000 mẫu hợp lệ mỗi map** rồi tự dừng. Collector spawn cả
semantic camera lẫn RGB camera trong chế độ này. Ở 5 FPS, thời gian lý thuyết
khoảng 33 phút 20 giây/map. Town01–04 cho 40.000 mẫu train, Town05 cho 10.000 mẫu
val — đúng con số `EXPECT_TRAIN`/`EXPECT_VAL` mà hai notebook train kiểm tra.

> **5 FPS là hợp đồng, không phải tuỳ chọn hiệu năng.** `CONTROL_DT = 1/5 = 0.2 s`;
> `previous_steer` nghĩa là "lệnh điều khiển của 1 bước trước", nên đổi FPS là đổi ý
> nghĩa của chính đặc trưng đó. Ở 20 FPS vô-lăng gần như không kịp đổi trong 50 ms →
> `previous_steer ≈ steer` và model IL đạt loss thấp bằng cách chép lại nó, bỏ qua
> hoàn toàn ảnh segmentation (copycat/causal confusion — không hiện ra trong val loss).
> Khi train DRL, đặt `fixed_delta_seconds` của CARLA đúng bằng 0.2 s.
Graph A* mặc định chưa được xuất vì A* là giai đoạn mở rộng sau IL và DRL.

Nội dung `collector_config.json` hiện tại:

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
    "image_mode": "seg-rgb",
    "save_seg_color": true,
    "width": 480,
    "height": 384,
    "fov": 90.0,
    "fps": 5.0,
    "camera_x": 1.5,
    "camera_y": 0.0,
    "camera_z": 2.4,
    "camera_pitch": -5.0
  },
  "collection": {
    "max_samples": 10000,
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

Những dòng thường cần chỉnh: `dataset.output` (ổ đĩa/thư mục lưu), `camera.image_mode`
và `camera.save_seg_color` (đổi luồng ảnh lưu ra), `camera.width/height/fps`,
`collection.max_samples`/`duration` (điều kiện dừng) và `astar.no_map_export`
(bật khi bắt đầu giai đoạn A*). Mục `dedup` lọc bớt các mẫu trùng lặp ngay lúc thu
(xem mục "Giảm trùng lặp / mất cân bằng dữ liệu" bên dưới).

- `max_samples > 0`: tự dừng khi đã ghi đủ số mẫu đồng bộ.
- `duration > 0`: tự dừng theo số giây thực tế.
- Nếu cả hai lớn hơn 0, điều kiện nào đạt trước sẽ dừng trước.
- `collection.queue_size = 128`: hàng đợi packet giữa sensor callback và writer
  thread; tăng giá trị này nếu log báo `dropped` tăng do writer (ghi PNG/CSV)
  theo không kịp tốc độ camera.
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
5. Đủ 10.000 mẫu, collector tự cleanup sensor và dừng.
6. Kiểm tra session bằng verify_dataset.py --strict.
7. Sinh il_fields.csv / drl_fields.csv bằng split_csv.py  <-- BẮT BUỘC
8. Dừng automatic_control.py bằng Ctrl+C.
9. Notebook: chạy cell client.load_world("Town02").
10. Chạy lại automatic_control.py để tạo hero mới.
11. Chạy lại run_collector.bat với cùng config.
```

Tiếp tục tương tự với Town03, Town04 và **Town05** (Town05 là tập validation của cả
hai notebook train — thiếu nó thì không train được). Không chạy cell đổi map khi
collector còn hoạt động, vì `client.load_world()` hủy world và toàn bộ actor/sensor
của map cũ.

**Bước 6–7 cho mỗi session vừa thu xong:**

```powershell
python verify_dataset.py D:\CARLA_DATA\Town01_<timestamp> --strict
python ..\data_analysis\split_csv.py D:\CARLA_DATA\Town01_<timestamp>
```

`collect_data.py` **chỉ** ghi `states.csv`; `train_il.ipynb` đọc `il_fields.csv` trong
từng thư mục Town, và file đó do `split_csv.py` sinh ra. Bỏ qua bước này thì notebook IL
sẽ dừng ngay ở mục 3 với `FileNotFoundError: il_fields.csv`. Notebook segmentation không
cần file này (nó đọc thẳng `rgb/` + `seg_label/`).

Sau khi xong cả 5 map, thư mục upload lên Kaggle phải có dạng — tên `CARLA_DATA` là thứ
`resolve_dataset_root()` của cả hai notebook dò tìm:

```text
CARLA_DATA/
├── Town01_<timestamp>/   rgb/  seg_label/  seg_color/  states.csv  il_fields.csv  ...
├── Town02_<timestamp>/
├── Town03_<timestamp>/
├── Town04_<timestamp>/
└── Town05_<timestamp>/   ← validation
```

> Nếu dung lượng chạm hạn ngạch 20 GB của Kaggle: `seg_color` chỉ là ảnh xem trước, không
> notebook nào đọc nó. Đặt `"save_seg_color": false` trong `collector_config.json` để bỏ
> hẳn luồng ảnh này. **Không** bỏ `rgb` (notebook segmentation train RGB → mask) và không
> bỏ `seg_label` (nhãn của cả hai notebook).

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
- `seg_label`: ảnh một kênh, mỗi pixel là **raw semantic tag CARLA `0..22`** (giữ
  nguyên, không remap) — đây là input tiết kiệm dung lượng và tính toán nhất, và giữ
  raw tag nghĩa là đổi label scheme sau này không cần thu thập lại.
- `seg_color`: ảnh semantic ba kênh, **đã gộp sẵn về 4 lớp bám làn** theo
  `schema.SEG_CLASS_COLORS` (Background xám / Road tím / RoadLine vàng-xanh /
  Sidewalk hồng) — dùng để kiểm tra trực quan xem model sẽ *nhìn thấy* gì. Không được
  dùng ColorJitter/hue/saturation.
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

`seg_label` lưu **raw tag CARLA 0–22**. Pipeline train gộp chúng về 4 lớp bám làn
(`Background, Road, RoadLine, Sidewalk`) bằng `schema.RAW_TO_TRAIN_LANE_LUT` — bảng này
là nguồn sự thật duy nhất, hai notebook và `drl_training/policy/observation.py` đều
phải khớp byte-for-byte với nó. Giữ raw tag trong file nghĩa là đổi scheme sau này
không phải thu thập lại dữ liệu.

Không đưa class ID vào CNN như cường độ xám có thứ tự. Hãy one-hot theo số lớp sau
khi remap, dùng embedding, hoặc gom thành các nhóm semantic. Nếu dùng `seg_color`
(đã được gộp sẵn về 4 lớp), giữ nguyên bảng màu và chuẩn hóa ba kênh nhất quán.

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
python collect_data.py --output D:\CARLA_DATA --image-mode seg-only
```

Đây là lệnh tối giản dùng tham số CLI mặc định (`config.py`: 480×384 @ 5 FPS, giống
`collector_config.json`): chỉ một semantic camera và class-mask PNG một kênh, không
lưu RGB/seg_color. Khi chạy bằng
`run_collector.bat` (tức dùng `collector_config.json`), collector spawn thêm RGB
camera và lưu đồng thời cả ba luồng ảnh (`rgb`, `seg_label`, `seg_color`) ở độ
phân giải 480×384, vì file cấu hình đặt `image_mode: seg-rgb` và
`save_seg_color: true`. Muốn chạy trực tiếp bằng CLI với đúng các luồng ảnh đó:

```powershell
python collect_data.py --output D:\CARLA_DATA --image-mode seg-rgb --save-seg-color --width 480 --height 384
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
python collect_data.py --output D:\CARLA_DATA --duration 60
```

Thu đúng 5.000 mẫu:

```powershell
python collect_data.py --output D:\CARLA_DATA --max-samples 5000
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

## Giảm trùng lặp / mất cân bằng dữ liệu

Dataset thu bằng autopilot có hai loại "trùng lặp" khác nhau, xử lý khác nhau:

1. **Trùng lặp thật** (frame gần như y hệt frame trước): xảy ra khi xe **đứng yên**
   và hành động không đổi trong nhiều giây liên tiếp — chủ yếu là lúc dừng đèn đỏ
   hoặc kẹt xe. Ảnh gần như tĩnh, không mang thêm thông tin nên có thể loại bớt
   một cách an toàn.
2. **Mất cân bằng phân bố** (không phải trùng lặp thật): xe **đang di chuyển**
   nhưng đi thẳng rất lâu (`steer≈0`). Ảnh vẫn đổi thật theo từng frame (cảnh vật
   trôi qua), nên **không được xoá** — chỉ nên giảm tần suất xuất hiện lúc train
   để mô hình không học lệch về "giữ nguyên vô-lăng".

Ba lớp lọc/cân bằng tương ứng, dùng độc lập hoặc kết hợp:

### 1) Lọc lúc thu thập (collector)

`collect_data.py` / `collector_config.json` có 3 tham số (mặc định tắt qua CLI,
đã **bật sẵn** trong `collector_config.json`):

```text
--dedup-stationary-speed 0.3   # m/s; duoi nguong nay coi la dung yen. 0 = tat
--dedup-action-eps 0.02        # nguong |steer_delta|/|longitudinal_delta|
--dedup-min-interval-s 1.0     # giay toi thieu giua 2 mau dung yen duoc giu lai
```

Một mẫu chỉ bị bỏ khi **đồng thời**: tốc độ dưới `dedup-stationary-speed`, hành
động gần như không đổi so với mẫu đã ghi gần nhất (`< dedup-action-eps`), và
chưa đủ `dedup-min-interval-s` giây kể từ mẫu dừng yên đã giữ gần nhất. Không
bao giờ loại mẫu lúc xe đang di chuyển. Ảnh/CSV của các mẫu bị bỏ hoàn toàn
không được ghi ra đĩa (đỡ cả dung lượng lẫn thời gian ghi), và số mẫu bị bỏ
được in ra log (`dup_skip=`) và lưu vào `summary.json`
(`duplicate_frames_skipped`).

### 2) Lọc hậu xử lý cho dữ liệu đã thu trước đó

Có 2 công cụ hậu xử lý dùng cùng tiêu chí và cùng 3 tham số
(`--dedup-stationary-speed`, `--dedup-action-eps`, `--dedup-min-interval-s`,
mặc định **tắt** để không đổi hành vi cũ) — chọn đúng cái khớp với luồng bạn
dùng để train:

- **`data_analysis/split_csv.py`** — đây là script thực sự sinh
  `il_fields.csv`/`drl_fields.csv`/`astar_fields.csv` mà
  `behavior_cloning/train_il.ipynb` đọc trực tiếp theo từng Town. Muốn
  lọc trùng lặp cho dữ liệu **đã thu trước khi collector có filter này**, chạy
  lại lệnh này cho từng session trước khi mở notebook:

  ```powershell
  python data_analysis\split_csv.py D:\CARLA_DATA\Town01_20260715_230000_123456 --dedup-stationary-speed 0.3 --dedup-min-interval-s 1.0
  ```

- **`build_manifest.py`** — chỉ ảnh hưởng `manifest.csv`/`dataset_summary.json`
  (khoá `duplicate_frames_filtered`), dùng cho các luồng đọc `manifest.csv`
  trực tiếp (hiện notebook IL **không** dùng file này, nó tự gộp Town và tự
  chia session).

  ```powershell
  python build_manifest.py D:\CARLA_DATA --dedup-stationary-speed 0.3 --dedup-min-interval-s 1.0
  ```

Cả hai đều không xoá PNG hay sửa `states.csv` gốc — chỉ bớt dòng ở file CSV đầu
ra tương ứng.

### 3) Cân bằng lúc train (không xoá dữ liệu)

`behavior_cloning/train_il.ipynb` dùng `WeightedRandomSampler` để mỗi
epoch lấy mẫu đều hơn theo `traffic_light_state` (đã có sẵn) **và** theo mức độ
`|steer|` (đi thẳng / lái nhẹ / cua vừa / cua gấp). Đây là lựa chọn an toàn nhất
vì không mất bất kỳ mẫu recovery/cua gấp hiếm gặp nào — chỉ đổi tần suất được
lấy ra trong một epoch.

### 4) Đa dạng hoá route/session

Nếu nhiều session autopilot lặp lại gần đúng một quỹ đạo, ba lớp lọc trên không
giải quyết được (chúng không trùng nhau về mặt kỹ thuật, chỉ trùng nhau về mặt
hành vi lái). Giảm bằng cách:

- Ưu tiên nhiều session **ngắn** (đã khuyến nghị ở mục "Lặp lại qua nhiều map")
  thay vì một session dài — mỗi lần chạy lại `automatic_control.py`, xe spawn
  lại ở điểm ngẫu nhiên nên route cũng đổi theo.
- Đổi thời tiết (`world.set_weather(...)`) và/hoặc thời điểm trong ngày giữa
  các session, kể cả cùng map.
- Khi build manifest, xem `dataset_summary.json` → `samples_by_map_and_split`
  để kiểm tra map nào đang chiếm tỉ trọng quá lớn so với các map khác.

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

Giá trị đang dùng thật trong `collector_config.json`. Mặc định CLI của `config.py`
đã được chỉnh cho TRÙNG các giá trị này, nên chạy `collect_data.py` không kèm
`--config` cũng ra cùng một loại dữ liệu (trước đây CLI mặc định `800 x 450 @ 10 FPS`
— khác cả độ phân giải lẫn bước thời gian so với bộ dữ liệu đang có, mà không có
cảnh báo nào):

```text
resolution : 480 x 384
fps        : 5   (CONTROL_DT = 0.2 s — hợp đồng với IL/DRL)
FOV        : 90 độ
camera     : x=1.5 m, z=2.4 m, pitch=-5 độ
lookahead  : 5 m
route lookahead: 5, 10, 20, 30 m
graph resolution: 2 m
```

Ước lượng dung lượng thực tế phụ thuộc cảnh và mức nén PNG. Hãy chạy thử 5 phút, xem `summary.json` và dung lượng session rồi mới thu hàng giờ.
