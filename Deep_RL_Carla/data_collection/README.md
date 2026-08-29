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
    ├── ego_watch.py                # tìm ego / chờ một chiếc xe dùng được
    ├── runner.py                   # chuỗi session của cả buổi thu thập
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
| Đổi điều kiện nhận một chiếc xe là "chạy được" | `ego_watch.py` |
| Đổi cách đổi xe / mở session mới khi kẹt | `collector.py`, `runner.py` |
| Đổi trình tự kết nối/chạy/dừng | `collector.py` |

Không cần sửa `collect_data.py` trừ khi muốn đổi cách khởi động chương trình.

## Chạy bằng file cấu hình

File `collector_config.json` mặc định lưu đồng thời `rgb`, `seg_label` và
`seg_color` (vì `image_mode: seg-rgb` và `save_seg_color: true`), ở độ phân giải
480×384, thu đúng **10.000 mẫu hợp lệ mỗi map** rồi tự dừng — cộng dồn qua nhiều
session (`total_samples: 10000`, tối đa `max_samples: 2500` mỗi session) để một
lần watchdog nổ không làm mất cả buổi. Collector spawn cả
semantic camera lẫn RGB camera trong chế độ này. Ở 5 FPS, thời gian lý thuyết
khoảng 33 phút 20 giây/map. Town01–04 cho 40.000 mẫu train, Town05 cho 10.000 mẫu
val — đúng con số `EXPECT_TRAIN`/`EXPECT_VAL` mà hai notebook train kiểm tra.

> **5 FPS là hợp đồng, không phải tuỳ chọn hiệu năng.** `CONTROL_DT = 1/5 = 0.2 s`;
> `previous_steer` nghĩa là "lệnh điều khiển của 1 bước trước", nên đổi FPS là đổi ý
> nghĩa của chính đặc trưng đó. Ở 20 FPS vô-lăng gần như không kịp đổi trong 50 ms →
> `previous_steer ≈ steer` và model IL đạt loss thấp bằng cách chép lại nó, bỏ qua
> hoàn toàn ảnh segmentation (copycat/causal confusion — không hiện ra trong val loss).
> Khi train DRL, đặt `fixed_delta_seconds` của CARLA đúng bằng 0.2 s.

> **`sensor_tick` một mình KHÔNG giữ được 5 FPS.** Collector là client thụ động trên
> world async, variable-timestep do `automatic_control.py` sở hữu; ở đó CARLA không
> tôn trọng `sensor_tick`. Một session Town01 đặt 5 FPS đã đo được **22.7 mẫu/giây
> sim**, khoảng cách giữa hai mẫu dao động 0.011–1.027 s. Vì vậy nhịp lấy mẫu được
> ép ở phía client bằng `synchronizer.SampleRateLimiter` (chặn theo `sim_time_s` của
> chính packet đã ghép). `metadata.json` ghi FPS *yêu cầu*, `summary.json` ghi FPS
> *đo được*, và `verify_dataset.py` báo lỗi nếu hai con số lệch quá 10%.

> **Collector tự dừng khi session hỏng (watchdog).** Trước đây vòng lặp chính chỉ
> thoát khi ego bị destroy, hết `duration`, hoặc writer lỗi — không cái nào bắt được
> hai kiểu hỏng đã gặp thật:
>
> 1. **Camera ngừng gửi ảnh** trong khi ego vẫn `is_alive`. Một session Town03 chạy
>    **15 phút thực** nhưng chỉ ghi được **102 giây đầu**; 99.967 frame chỉ có state,
>    không có ảnh nào ghép cùng frame.
> 2. **Agent phanh khẩn cấp rồi không bao giờ nhả.** Một session Town03 khác đi được
>    **76 m** rồi đứng yên **344 giây liên tục (96% session)** với
>    `throttle=0, brake=1.0, steer=0.0`, không đèn đỏ, không va chạm.
>
> `stall_timeout_s` bắt (1) theo giây **thực**, `stationary_timeout_s` bắt (2) theo
> giây **sim**. Lý do dừng được ghi vào `summary.json` → `stop_reason`.

> **Chờ 60 giây mới nhận ra hỏng là quá đắt.** Đo trên session
> `Town03_20260824_082122_987768` (4906/10000 mẫu, 24 phút thực): 8 lần đổi xe,
> **7 lần vì `no_samples_timeout`**, và mỗi lần để lại đúng một lỗ **61–70 giây
> sim** trong `states.csv` — tổng cộng **~8 phút chết trên 24 phút chạy (33%)**.
> Cả 8 lỗ đều dài xấp xỉ `stall_timeout_s`, tức là thời gian đó không phải chờ xe
> mới mà là chờ **phát hiện ra** rằng đã hỏng. Hai kiểu hỏng đó giờ có watchdog
> riêng, nhanh hơn một bậc:
>
> - `ego_missing_timeout_s` (mặc định 3 s). `actor.is_alive` là cờ của **riêng
>   client này**: khi `automatic_control.py` hủy chiếc hero cũ để spawn chiếc mới,
>   cờ đó vẫn `True` mãi mãi và collector không hề biết xe đã biến mất. Bằng chứng
>   thật là ego **vắng mặt khỏi `WorldSnapshot`** — vắng liên tục quá ngưỡng này
>   thì coi như đã bị hủy và đổi xe ngay.
> - `camera_timeout_s` (mặc định 10 s). Camera im trong khi world **vẫn tick** là
>   hỏng của riêng camera, không phải của xe: trong session trên, hai lần "đổi xe"
>   tìm ra đúng chiếc xe cũ (id 124 → 124, 137 → 137) vì chiếc xe đó chưa bao giờ
>   hỏng cả. Giờ collector **thay bộ camera ngay trên chiếc xe đang bám**, giữ
>   nguyên ego và bộ đếm quãng đường, mất vài chục mili giây thay vì hơn một phút.
>   Thay hai lần liên tiếp mà vẫn không ra mẫu nào thì mới leo thang thành
>   `camera_dead` và đổi xe thật. `summary.json` → `camera_restarts` đếm số lần.
>
> `stall_timeout_s` vẫn còn, nhưng lùi về vai trò lưới an toàn cuối cùng.

> **Lọc xe 2 bánh (`min_wheels`, mặc định 4).** `automatic_control.py` bốc
> blueprint ngẫu nhiên trong `vehicle.*`, nên chiếc "hero" nó tạo ra có thể là xe
> đạp hoặc mô tô. Trong session trên, **992/4906 mẫu (20%)** quay từ
> `vehicle.gazelle.omafiets` (xe đạp) và `vehicle.kawasaki.ninja` (mô tô): động
> học khác hẳn ô tô, camera đặt ở `z=2.4` không còn nằm trên mui xe, và IL/DRL
> phải học lẫn lộn hai kiểu điều khiển. Collector giờ bỏ qua mọi ứng viên có
> `number_of_wheels < min_wheels`. Chắc ăn hơn nữa thì chạy
> `automatic_control.py --filter vehicle.tesla.model3` để hero luôn là cùng một xe.

> **Chạy `automatic_control.py` với `--loop`.** Không có cờ đó, script thoát ngay
> khi agent tới đích (`automatic_control.py` dòng 749: *"Target reached, mission
> accomplished..."*) và chiếc hero bị hủy theo — đúng 7 lần trong session
> `Town03_20260824_082122_987768`, mỗi lần collector mất một phút mới nhận ra rồi
> lại chờ bạn khởi động script bằng tay. Với `-l/--loop`, agent tự chọn đích mới
> (`agent.reroute`) và chạy liên tục. Lệnh nên dùng:
>
> ```bash
> python automatic_control.py --loop --filter vehicle.tesla.model3
> ```
>
> Thêm `--behavior aggressive` nếu muốn xe dừng ít hơn (khoảng cách an toàn ngắn
> hơn, bám tốc độ giới hạn sát hơn) — nhưng nhớ rằng đó chính là kiểu lái mà IL sẽ
> học theo, nên đừng trộn nhiều `--behavior` khác nhau trong cùng một bộ dữ liệu.

> **Watchdog nổ chỉ kết thúc một session, không kết thúc cả buổi thu thập.**
> Collector là client *thụ động* — nó không lái xe, `automatic_control.py`
> (BehaviorAgent) mới lái, và agent đó kẹt thường xuyên. Đo trên một session Town03
> thật: các quãng đứng yên 12 s, 3 s, **39 s**, 25 s, 5 s, 17 s xen kẽ nhau, không
> quãng nào liên quan tới đèn đỏ (`traffic_light_state = Unknown` suốt cả quãng).
> Sớm muộn sẽ có một quãng vượt 60 s và watchdog giết session — đúng, nhưng ngày
> trước điều đó làm mất luôn phần còn lại của buổi thu thập: 4/10 session trong
> `D:/CARLA_DATA_V2` chết ở mức 485–1826 mẫu trên mục tiêu 10000.
>
> Có **hai lớp** cứu buổi thu thập, lớp trong chạy trước:
>
> **Lớp 1 — đổi xe ngay trong session đang chạy (`--rebind-ego`, mặc định bật).**
> Watchdog nổ không còn đóng session nữa. Collector gỡ camera khỏi chiếc xe kẹt,
> **chờ một chiếc xe khác xuất hiện** (bạn Ctrl+C `automatic_control.py` rồi chạy
> lại nó để tạo hero mới), gắn camera vào chiếc xe đó và **ghi tiếp vào đúng
> session cũ** — cùng thư mục, cùng `states.csv`, `sample_id` chạy tiếp, không
> sinh thêm session vụn. Đây là thứ thay đổi nhiều nhất trong thực tế: mục tiêu
> 10.000 mẫu giờ nằm trong **một** thư mục thay vì rải ra 5–6 thư mục 500–2000 mẫu.
>
> **Lớp 2 — mở session mới (`SessionRunner` trong `carla_collector/runner.py`).**
> Chỉ vào cuộc khi lớp 1 bó tay: hết `rebind_wait_s` mà không có xe nào
> (`rebind_timeout`), world bị `load_world()` (`world_reloaded`, frame counter về
> 0 nên không ghi chung thư mục được nữa), hoặc lỗi lúc khởi động session. Nó chờ
> ego chạy lại rồi mở session mới và cộng dồn số mẫu cho tới khi đủ
> `total_samples`. Bật bằng `collection.total_samples` + `auto_restart.auto_restart`.
>
> Điều kiện nhận một chiếc xe (chung cho cả hai lớp, xem `ego_watch.pick_ego`):
> frame number của world phải **tăng** giữa hai lần poll, và
>
> - xe có `id` **khác** chiếc vừa kẹt → nhận ngay (xe mới spawn thì đứng yên vài
>   frame đầu là bình thường). Bước lọc theo `id` này là bắt buộc: chiếc hero cũ
>   còn nằm đó thêm vài giây sau khi bạn dừng `automatic_control.py`, không lọc
>   thì collector gắn camera vào lại đúng chiếc xe vừa kẹt.
> - vẫn là chiếc xe cũ (hoặc `--vehicle-id` ghim cứng một id) → phải thấy tốc độ
>   vượt `stall_speed` mới nhận.
>
> Phải kiểm tra frame number chứ không chỉ nhìn tốc độ: khi world dừng hẳn
> (trường hợp `no_samples_timeout`) thì `get_velocity()` vẫn trả về giá trị cũ của
> tick cuối, và collector sẽ tưởng xe đang chạy.

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
    "wait_vehicle_timeout": 120.0,
    "min_wheels": 4
  },
  "dataset": {
    "output": "D:/CARLA_DATA_V2"
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
    "total_samples": 10000,
    "duration": 0.0,
    "queue_size": 64,
    "no_event_sensors": false
  },
  "rebind": {
    "rebind_ego": true,
    "rebind_wait_s": 300.0,
    "max_rebinds": 0
  },
  "auto_restart": {
    "auto_restart": true,
    "max_restarts": 20,
    "restart_wait_s": 300.0,
    "restart_settle_s": 3.0
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
  },
  "dedup": {
    "dedup_stationary_speed": 0.0,
    "dedup_action_eps": 0.02,
    "dedup_min_interval_s": 1.0
  },
  "watchdog": {
    "stall_timeout_s": 60.0,
    "stationary_timeout_s": 60.0,
    "camera_timeout_s": 10.0,
    "ego_missing_timeout_s": 3.0,
    "stall_speed": 0.3
  }
}
```

Những dòng thường cần chỉnh: `dataset.output` (ổ đĩa/thư mục lưu), `camera.image_mode`
và `camera.save_seg_color` (đổi luồng ảnh lưu ra), `camera.width/height/fps`,
`collection.max_samples`/`total_samples`/`duration` (điều kiện dừng) và
`astar.no_map_export` (bật khi bắt đầu giai đoạn A*). Mục `dedup` lọc bớt các mẫu
trùng lặp ngay lúc thu — **mặc định vẫn tắt** (`dedup_stationary_speed: 0.0`), xem mục
"Giảm trùng lặp / mất cân bằng dữ liệu" bên dưới. Mục `watchdog` tự dừng session
khi hỏng:

- `stall_timeout_s` (giây **thực**): không có mẫu mới nào trong ngần này giây thì
  dừng. Lưới an toàn cuối cùng — hai watchdog dưới đây bắt sớm hơn nhiều.
- `camera_timeout_s` (giây **thực**): camera không gửi ảnh nào trong ngần này giây
  **trong khi world vẫn tick** thì gắn lại camera vào chính chiếc xe đó, không đổi
  xe. Hỏng của camera không phải hỏng của xe.
- `ego_missing_timeout_s` (giây **thực**): ego vắng mặt khỏi `WorldSnapshot` liên
  tục ngần này giây thì coi như đã bị hủy. Đây là cách **duy nhất** thấy được việc
  `automatic_control.py` hủy hero, vì `actor.is_alive` chỉ là cờ của client này.
- `stationary_timeout_s` (giây **sim**): xe đứng yên liên tục ngần này giây thì dừng.
  Bắt trường hợp agent phanh khẩn cấp vĩnh viễn. **Phải đặt lớn hơn lần chờ đèn đỏ
  lâu nhất của bạn** — đo được trên Town01/Town02 là tối đa 31 s, nên 60 s là an toàn.
  Nếu bạn kéo dài pha đèn thì phải tăng giá trị này theo, không thì sẽ dừng nhầm.
- `stall_speed` (m/s): dưới ngưỡng này watchdog coi là đứng yên. Cố ý tách khỏi
  `dedup_stationary_speed` để watchdog vẫn chạy khi dedup đã tắt.
- Đặt `0` cho bất kỳ timeout nào để tắt riêng watchdog đó.
- `min_wheels` (mục `connection`): số bánh tối thiểu để một chiếc xe được chọn làm
  ego; `4` bỏ qua xe đạp/mô tô, `0` nhận tất cả. Không áp dụng khi đã ghim
  `--vehicle-id`.

Mục `rebind` quyết định chuyện gì xảy ra **ngay khi** watchdog nổ — trước khi
nghĩ đến chuyện đóng session:

- `rebind_ego` (mặc định `true`): gỡ camera khỏi chiếc xe kẹt/bị hủy, chờ một
  chiếc xe khác rồi gắn camera vào và **ghi tiếp vào session đang chạy**. Chỉ bốn
  lý do được đổi xe: `vehicle_stationary_timeout`, `no_samples_timeout`,
  `ego_destroyed`, `camera_dead` — cả bốn đều là chuyện của riêng chiếc xe đó
  (`camera_dead` chỉ đến sau khi thay camera tại chỗ hai lần vẫn không cứu được),
  thư mục session và
  số mẫu đã ghi vẫn còn nguyên. Đặt `false` để quay lại hành vi cũ (watchdog nổ
  là đóng session, để `auto_restart` lo).
- `rebind_wait_s` (giây **thực**): chờ chiếc xe khác tối đa ngần này giây rồi mới
  đóng session với `stop_reason = rebind_timeout`. Để rộng, đủ cho bạn Ctrl+C
  `automatic_control.py` và chạy lại nó.
- `max_rebinds`: số lần đổi xe tối đa trong **một** session; `0` = không giới hạn.

Trong lúc chờ xe mới, mọi thứ khác giữ nguyên: writer, `states.csv`, `sample_id`,
`distance_travelled_m` (cộng dồn tiếp, không cộng thêm quãng nhảy từ chỗ xe cũ kẹt
tới chỗ xe mới spawn). Hai thứ bị **cắt** ở ranh giới đổi xe, cố ý:

- `previous_steer` / `previous_longitudinal` của mẫu đầu tiên thuộc xe mới lấy
  theo chính mẫu đó và `sample_delta_seconds = 0`, y như mẫu đầu session — lệnh
  cuối của chiếc xe trước không phải "một bước trước" của chiếc xe sau.
- Quãng chờ (có thể vài phút sim) **không** tính vào `sim_time_span_s`, nên
  `measured_fps` vẫn là nhịp thật của dữ liệu chứ không bị kéo tụt.

Cột `vehicle_id` trong `states.csv` cho biết mỗi dòng thuộc chiếc xe nào, và
`summary.json` → `ego_segments` liệt kê từng chiếc: `vehicle_id`, số mẫu đóng góp
và `released_reason` (vì sao nó bị gỡ ra).

Mục `auto_restart` quyết định chuyện gì xảy ra **sau khi** session đã đóng hẳn:

- `auto_restart`: bật thì mở session mới khi session vừa rồi chết vì
  `rebind_timeout`, `world_reloaded`, `startup_error` (chưa thấy ego / world đang
  reload / rớt kết nối), hoặc — khi `rebind_ego = false` —
  `vehicle_stationary_timeout`, `no_samples_timeout`, `ego_destroyed`. Các lý do
  `duration`, `writer_error`, `user_interrupt` **không** khởi động lại — chúng
  không phải thứ thử lại sẽ sửa được. Cần `total_samples > 0` mới bật được, vì
  không có mục tiêu tổng thì không biết khi nào là đủ.
- `max_restarts`: số lần khởi động lại **sau lỗi** tối đa; `0` = không giới hạn.
  Session lăn sang session mới vì đã đầy `max_samples` không tính vào đây.
- `restart_wait_s` (giây **thực**): chờ ego chạy lại tối đa ngần này giây rồi mới
  bỏ cuộc. Các quãng kẹt của BehaviorAgent đo được dài 25–45 s nên để rộng.
- `restart_settle_s`: nghỉ giữa hai session cho sensor cũ kịp gỡ xuống.

Kết quả cả buổi được ghi ra `<output>/run_<YYYYmmdd_HHMMSS>.json`: tổng số mẫu,
số session, số session bị watchdog ngã, tổng số lần đổi xe (`ego_rebinds`), và
danh sách `stop_reason` từng session.

- `max_samples > 0`: dừng **một session** khi đã ghi đủ số mẫu đồng bộ.
- `total_samples > 0`: mục tiêu cho **cả buổi**, cộng dồn qua nhiều session. Session
  cuối tự cắt bớt hạn mức để không vượt quá mục tiêu. `0` = chỉ chạy đúng một
  session rồi thoát (hành vi cũ).
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
4b. Nếu collector báo "Cho xe moi ... de thu tiep vao session nay": xe đã kẹt.
    Sang Terminal A, Ctrl+C rồi chạy lại automatic_control.py. KHÔNG đụng vào
    Terminal B — collector tự tìm hero mới và ghi tiếp vào cùng session.
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

`collect_data.py` **chỉ** ghi `states.csv`; `train_il_v9.ipynb` đọc `il_fields.csv` trong
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

`collect_data.py` / `collector_config.json` có 3 tham số, **mặc định tắt ở cả CLI
lẫn `collector_config.json`**:

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

> **Nên để tắt lúc thu, lọc lại ở bước hậu xử lý.** Lọc ở collector là **không đảo
> ngược được**: ảnh PNG của mẫu bị bỏ không bao giờ được ghi ra đĩa. Đo trên dữ liệu
> thật, xe đứng yên **28–41% thời gian session** (chờ đèn đỏ, trung vị 15–19 s/lần),
> và dedup ở ngưỡng 0.3 m/s + 1.0 s đã **xoá 81% số mẫu xe đứng yên** — mẫu đứng yên
> chỉ còn chiếm 8.6% dataset thay vì ~34%. Tệ hơn, các mẫu đứng yên còn sót lại có
> `sample_delta_seconds` là 1.0–1.2 s thay vì 0.2 s, tức là `previous_steer` /
> `previous_longitudinal` **sai thang đo** đúng ở tình huống model IL cần học nhất
> (giữ phanh ở đèn đỏ). Cách (2) bên dưới cho cùng tác dụng lọc mà vẫn giữ được
> dữ liệu gốc để đổi ý sau.

### 2) Lọc hậu xử lý cho dữ liệu đã thu trước đó

Có 2 công cụ hậu xử lý dùng cùng tiêu chí và cùng 3 tham số
(`--dedup-stationary-speed`, `--dedup-action-eps`, `--dedup-min-interval-s`,
mặc định **tắt** để không đổi hành vi cũ) — chọn đúng cái khớp với luồng bạn
dùng để train:

- **`data_analysis/split_csv.py`** — đây là script thực sự sinh
  `il_fields.csv`/`drl_fields.csv`/`astar_fields.csv` mà
  `behavior_cloning/train_il_v9.ipynb` đọc trực tiếp theo từng Town. Muốn
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

`behavior_cloning/train_il_v9.ipynb` dùng `WeightedRandomSampler` để mỗi
epoch lấy mẫu đều hơn theo `traffic_light_state` (đã có sẵn) **và** theo mức độ
`|steer|` (đi thẳng / lái nhẹ / cua vừa / cua gấp). Đây là lựa chọn an toàn nhất
vì không mất bất kỳ mẫu recovery/cua gấp hiếm gặp nào — chỉ đổi tần suất được
lấy ra trong một epoch.

### 4) Đa dạng hoá route/session

Nếu nhiều session autopilot lặp lại gần đúng một quỹ đạo, ba lớp lọc trên không
giải quyết được (chúng không trùng nhau về mặt kỹ thuật, chỉ trùng nhau về mặt
hành vi lái). Giảm bằng cách:

- Mỗi lần chạy lại `automatic_control.py`, xe spawn lại ở điểm ngẫu nhiên nên
  route cũng đổi theo. Từ khi có `rebind_ego`, việc đó không còn bắt buộc phải
  cắt session: một session dài đã đi qua nhiều chiếc xe (`ego_segments`) chính là
  đi qua nhiều route khác nhau. Nếu vẫn muốn chia nhỏ để chia train/val/test theo
  session thì dùng `max_samples` nhỏ hơn `total_samples` — `SessionRunner` tự lăn
  sang session mới khi đầy hạn mức.
- Đổi thời tiết (`world.set_weather(...)`) và/hoặc thời điểm trong ngày giữa
  các session, kể cả cùng map.
- Khi build manifest, xem `dataset_summary.json` → `samples_by_map_and_split`
  để kiểm tra map nào đang chiếm tỉ trọng quá lớn so với các map khác.

## 4. Kiểm tra session sau khi thu

```powershell
python verify_dataset.py D:\CARLA_DATA\Town01_20260715_230000_123456 --strict
```

Trước hết mở `summary.json` và xem `stop_reason`. Chỉ **`max_samples`**,
**`duration`** và **`user_interrupt`** là kết thúc bình thường; `rebind_timeout`
(hết giờ chờ một chiếc xe khác), `world_reloaded` (map bị load lại giữa chừng),
`writer_error`, và — khi tắt `rebind_ego` — `no_samples_timeout` (camera chết),
`vehicle_stationary_timeout` (xe kẹt/agent phanh vĩnh viễn), `ego_destroyed`
nghĩa là session đã hỏng giữa chừng — dữ liệu vẫn dùng được nhưng ngắn hơn dự
kiến, và nên tìm nguyên nhân trước khi thu tiếp.
`vehicle_stationary_s_at_stop` cho biết xe đứng yên bao lâu tại thời điểm dừng.

`ego_rebinds > 0` nghĩa là session này đã đi qua nhiều chiếc xe; `ego_segments`
cho biết từng chiếc đóng góp bao nhiêu mẫu và bị gỡ ra vì lý do gì. Đây **không**
phải lỗi — đó chính là cơ chế giữ cho một buổi thu nằm gọn trong một session. Chỉ
để ý khi một chiếc chỉ đóng góp vài chục mẫu rồi kẹt ngay: lúc đó vấn đề nằm ở map
hoặc ở BehaviorAgent, không phải ở collector.

Kết quả tốt phải có `rows == unique_frames`, `errors: []`, và trong `measured`:

- `cadence_fps` ≈ `requested_fps` (`gap_seconds.median` ≈ 0.2 s) — đây là nhịp thật.
- `off_grid_gap_fraction` ≈ 0 — mọi khoảng cách mẫu là bội số nguyên của 0.2 s.

`throughput_fps` (= `rows / span`) **thấp hơn `requested_fps` là bình thường** khi bật
dedup: bộ lọc xoá hẳn các mẫu xe đứng yên nên gap trở thành 1.0–1.2 s. Một session
Town01 thật đo được `throughput_fps` 3.57 trong khi limiter vẫn cho qua đúng 5.000
mẫu/giây — đừng dùng con số đó để kết luận nhịp bị sai. Chỉ `cadence_fps` và
`off_grid_gap_fraction` mới phân biệt được "dedup đang làm việc" với "sensor_tick đã
tuột". Khi bật xuất graph
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

Vì world chạy async, `sensor_tick` không phải là thứ quyết định nhịp lấy mẫu — xem
`carla_collector/synchronizer.py::SampleRateLimiter`. Bộ này chặn theo `sim_time_s`,
nên `CONTROL_DT` giữ đúng 1/fps dù world tick nhanh chậm thế nào; nó cũng giữ cho
hàng đợi writer không bị tràn (trước đây writer phải nén 3 PNG ở 22.7 Hz, gây ra
chính những khoảng trống 1 s trong dữ liệu).

Việc chặn đó nay xảy ra **ngay tại đầu vào** chứ không phải sau khi packet đã ghép
xong, qua `synchronizer.FrameGate`: cả hai camera và world tick cùng hỏi một câu
"frame này có định làm mẫu không" và nhận cùng một câu trả lời (cache theo `frame`,
vì ba nguồn đó chạy trên ba thread khác nhau). Lý do là chi phí: session
`Town03_20260824_082122_987768` ghép xong **79449** packet để giữ lại **4906** mẫu —
94% số ảnh `bytes(image.raw_data)` (~737 KB mỗi ảnh) và 94% số state đầy đủ (~15 truy
vấn waypoint + 4 RPC mỗi cái) được dựng lên chỉ để vứt đi, ngay trên chính thread mà
CARLA dùng để đẩy ảnh sang client. `SampleRateLimiter.wants()` **không** dịch hạn kỳ;
chỉ `commit()` — chạy khi một packet thực sự ghép xong — mới dịch, nên một frame được
chọn mà cuối cùng thiếu ảnh không làm mất nhịp: các frame kế tiếp vẫn được nhận cho
tới khi có một mẫu thật ra đời. Phải nhận rộng như vậy vì **chỉ ~50% frame có đủ cả
hai camera** (mỗi camera bắn ~70% số frame); chặn các frame kế bên ngay ở đầu vào thì
mỗi lần frame được chọn không ghép xong lại phải chờ nguyên một nhịp, và nhịp lấy mẫu
tụt từ 5.0 xuống **3.9 FPS** (đo bằng mô phỏng).

Cái giá của việc nhận rộng là thỉnh thoảng **hai** frame kế nhau cùng ghép xong — mẫu
sau cách mẫu trước 10 ms, ảnh gần như y hệt, và `previous_steer` của nó là "lệnh của
10 ms trước" chứ không phải 200 ms. Đo trên session `Town03_20260824_092246_845374`:
67/10000 mẫu (0.67%). Vì vậy việc chặn trùng nằm ở **đầu ra**, trong
`SampleRateLimiter.is_duplicate`: packet nào ghép xong mà cách mẫu vừa ghi dưới **nửa
chu kỳ** thì bị bỏ, không ghi vào `states.csv`. Nhịp giữ nguyên 5.0, số mẫu bị bỏ nằm
ở `summary.json` → `near_duplicate_packets_dropped`. Đo lại trên mô phỏng cùng tỉ lệ: 5.000 mẫu/giây sim,
khoảng cách trung bình 0.2000 s, số ảnh phải copy giảm ~11 lần.

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
