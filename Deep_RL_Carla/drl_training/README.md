# drl_training — bước 3/3: fine-tune DRL cho bám làn (warm-start từ IL)

Pipeline của đồ án (xem `../README.md` cho bức tranh tổng): **1) segmentation** →
**2) Imitation Learning** (`../behavior_cloning/`, train trên Kaggle) → **3) DRL fine-tune**
(module này, chạy **local** vì cần một CARLA 0.9.10 server đang sống). Hỗ trợ **2 thuật
toán độc lập** — **PPO** (`train_ppo.py`) và **SAC** (`train_sac.py`) — cùng warm-start từ
một checkpoint IL, cùng dùng chung env/reward/hợp đồng quan sát, khác nhau ở cách fine-tune.

> ⚠️ **Tôi (Claude) không có CARLA/Python trong môi trường viết code này** nên chưa tự chạy
> được module này với server thật. Cả hai thuật toán được viết cẩn thận theo đúng quy ước đã
> có trong repo (`data_collection/carla_collector`, hợp đồng quan sát trong
> `docs/csv_fields_by_task.md`) và rà soát logic kỹ, nhưng hãy coi lần chạy đầu tiên là một
> smoke test (mục "Kiểm thử lần đầu" bên dưới) trước khi chạy train dài.

## Phạm vi

Chỉ **bám làn** (lane-keeping), không điều hướng theo tuyến A*: `../router_plan/Global_Route_Planner.py`
hiện chưa được cài (trống), và các cột `route_*` trong schema vẫn để trống theo đúng thiết kế
(`../data_collection/ASTAR_SCHEMA.md`). Không có A*, `route_command`/`route_target_*` không
nằm trong observation ở bước này — đúng với tên đồ án "điều khiển bám làn". Khi Router Plan
được cài, mở rộng observation/action ở đây là việc thêm trường, không phải viết lại.

## PPO hay SAC?

| | **PPO** (`train_ppo.py`) | **SAC** (`train_sac.py`) |
|---|---|---|
| Kiểu | On-policy | Off-policy |
| Bộ nhớ | Thấp — chỉ giữ 1 rollout (`n_steps=2048` × 480×384 ≈ 377MB) trong RAM, độ phân giải nào cũng rẻ | Cao hơn — replay buffer sống suốt quá trình train (`buffer_capacity=50000` × 480×384 ≈ 9.2GB mặc định, xem `sac/replay_buffer.py`) |
| Mẫu hiệu quả | Thấp hơn — cần nhiều bước môi trường hơn để hội tụ | Cao hơn — tái sử dụng transition nhiều lần qua replay buffer |
| Độ ổn định / dễ tune | Cao — clipped surrogate + early-stop theo KL tự bảo vệ | Nhạy hơn với learning rate/tau, nhưng auto temperature-tuning giảm bớt việc chỉnh tay |
| Khuyến nghị | Máy đơn, CARLA chạy chậm hơn training GPU, muốn kết quả dễ debug trước | Đủ RAM (máy thuê GPU), muốn tận dụng tối đa từng bước môi trường (CARLA step đắt hơn nhiều so với 1 gradient step) |

Cả hai đáng đưa vào báo cáo đồ án như một so sánh thực nghiệm (đúng tinh thần "nghiên cứu ứng
dụng học tăng cường sâu") — cùng warm-start, cùng reward, khác thuật toán fine-tune.

## Độ phân giải camera — vì sao mặc định khớp 480×384

`CarlaLaneKeepEnv` tự spawn camera segmentation **sống** trong lúc train (`width`/`height`
trong config) — **không** đọc lại dữ liệu đã thu ở `../data_collection/`, nên đổi độ phân
giải DRL không cần thu thập lại dữ liệu hay train lại segmentation/IL.

`PolicyBackbone` dùng `AdaptiveAvgPool2d((1,1))` nên nạp checkpoint IL vào actor DRL không
lỗi shape ở bất kỳ độ phân giải nào — nhưng có rủi ro thật nếu hạ độ phân giải quá mạnh: vạch
kẻ làn (`RoadLine`) chỉ rộng vài pixel, CARLA **render trực tiếp ở độ phân giải cấu hình**
(không phải downsample từ ảnh lớn), nên ở độ phân giải quá thấp vạch kẻ có thể biến mất khỏi
mask ở nhiều khung hình — đúng tín hiệu quan trọng nhất cho bài toán bám làn. Reward/điều
kiện done không bị ảnh hưởng (lấy thẳng từ API waypoint của CARLA, không suy ra từ ảnh), nên
hậu quả chỉ là giới hạn trần hiệu năng chính sách, không phải lỗi huấn luyện.

Vì vậy mặc định `width=480, height=384` — khớp **chính xác** độ phân giải notebook IL đã
train, loại bỏ hoàn toàn nguy cơ lệch phân phối quan sát. Chỉ cân nhắc hạ xuống (vd 240×192)
nếu thật sự thiếu RAM cho SAC ngay cả sau khi đã giảm `buffer_capacity` — và nên trực quan
hoá vài khung `seg_label` ở độ phân giải định dùng (giống cell trực quan hoá trong notebook
segmentation) để xác nhận vạch kẻ làn còn nhìn thấy được trước khi train dài.

## Cấu trúc

```
drl_training/
├── ppo_config.json / sac_config.json   # cấu hình mặc định (connection/camera/env/reward + thuật toán)
├── config.py                            # nạp --config JSON + override CLI, dùng chung cho cả 2 thuật toán
├── csv_logger.py                        # CSV logger nhỏ dùng chung train_ppo.py/train_sac.py
├── train_ppo.py / train_sac.py           # entrypoint train — chọn 1 trong 2
├── evaluate.py                            # chạy checkpoint đã train (PPO hoặc SAC), đo reward/va chạm/lệch làn
├── policy/
│   ├── backbone.py           # CNN(seg one-hot)+MLP(scalar) trunk + trunk_head dùng chung — KHỚP KIẾN TRÚC SteeringNet (IL)
│   ├── il_compat.py          # helper remap checkpoint IL -> state_dict, dùng chung PPO + SAC
│   ├── actor_critic.py       # PPO: GaussianActor + ValueCritic + loader warm-start IL
│   ├── observation.py        # dựng scalar vector đúng chuẩn hoá đã lưu trong checkpoint IL
│   └── checkpoint_io.py      # load + validate checkpoint IL
├── envs/
│   └── carla_lane_keep_env.py  # active client CARLA, đồng bộ (sync mode), reward/done — dùng chung cả 2 thuật toán
├── ppo/
│   ├── rollout_buffer.py     # buffer on-policy + GAE(λ)
│   └── ppo_agent.py           # clipped surrogate PPO update
└── sac/
    ├── networks.py            # GaussianPolicy (squashed-tanh, warm-start IL) + TwinQNetwork
    ├── replay_buffer.py        # replay buffer off-policy, tối ưu bộ nhớ (obs lưu 1 lần, không lưu next_obs riêng)
    └── sac_agent.py             # twin-Q update, auto temperature tuning, Polyak target update
```

## Hợp đồng quan sát/khớp checkpoint IL — dùng chung cho cả PPO và SAC

`policy/observation.py::ObservationContract` đọc `continuous_cols`, `raw_action_cols`,
`norm_stats`, `traffic_light_vocab`, `scalar_feature_dim` **trực tiếp từ checkpoint IL**
(`best_il_model.pth`) thay vì hard-code — vì notebook IL chạy trên Kaggle và module này chạy
local, hai bên không thể `import` chung code Python, nên checkpoint tự mang theo "hợp đồng"
của chính nó. Nếu bạn đổi tập đặc trưng ở notebook IL, **train lại IL trước**, checkpoint mới
sẽ tự động được cả `train_ppo.py` lẫn `train_sac.py` đọc đúng — không cần sửa gì ở đây.

`policy/il_compat.py` chứa các hàm remap dùng chung: `PPO GaussianActor` và
`SAC GaussianPolicy` đều gồm `backbone` (giống `SteeringNet.conv/cnn_fc/scalar_mlp`) +
`trunk_head` (giống 2 lớp đầu của `SteeringNet.head`) + một lớp cuối 32→2 tên khác nhau
(`mean_head` ở cả hai — SAC có thêm `log_std_head` không warm-start được vì IL không có khái
niệm noise scale). Nhờ vậy cả hai thuật toán nạp checkpoint IL qua **cùng một đường remap**,
không phải hai cách viết riêng dễ lệch nhau theo thời gian. Cả hai đều **báo lỗi rõ ràng**
(không âm thầm bỏ qua) nếu shape lệch — ví dụ khi checkpoint IL cũ (trước bản sửa rò rỉ
observation) bị dùng nhầm; gặp lỗi này thì train lại IL bằng notebook đã sửa trong
`../behavior_cloning/`.

## Observation / Action / Reward / Done

Theo đúng `docs/csv_fields_by_task.md` (mục DRL) và `docs/manual_thu_thap_du_lieu.md` §9.2,
thu hẹp về phạm vi bám làn — giống hệt nhau cho cả PPO và SAC vì cùng dùng
`envs/carla_lane_keep_env.py`:

- **Observation**: `seg` (ảnh class-ID, one-hot hoá bên trong `PolicyBackbone`) +
  scalar = `[speed_mps, yaw_rate_rps, speed_limit_kmh]` (z-score theo `norm_stats` của
  checkpoint IL) + `[previous_steer, previous_longitudinal]` (raw, đã ∈[-1,1]) +
  one-hot `traffic_light_state` (4 lớp). **Không** có `lane_offset_m`/`heading_error_rad`
  (chỉ dùng cho reward) — xem lý do trong `../behavior_cloning/train_il_kaggle.ipynb`.
- **Action**: `[steer, longitudinal] ∈ [-1,1]`, `longitudinal≥0`→throttle, `<0`→brake
  (giống hệt cách `../data_collection/carla_collector/writer.py` mã hoá `longitudinal`). SAC
  đạt khoảng này bằng tanh-squash trong `GaussianPolicy.sample`; PPO sample rồi clip (xem
  docstring `policy/actor_critic.py::GaussianActor.act` — khác biệt chuẩn giữa 2 thuật toán).
- **Reward** (trọng số cấu hình được trong `*_config.json["reward"]`):
  `+ w_speed·clip(forward_speed, 0, speed_limit) − w_lane_offset·|lane_offset_m| −
  w_heading·|heading_error_rad| − w_steer_delta·Δsteer² − w_long_delta·Δlongitudinal² −
  w_yaw_rate·yaw_rate² − off_lane_penalty (mỗi bước còn lệch làn) −
  lane_invasion_penalty·(số lần vượt vạch mới) − collision_penalty (khi va chạm)`.
- **Done**: `terminated=True` khi va chạm mới, hoặc lệch làn liên tục ≥
  `off_lane_patience_steps` bước (mặc định 20 bước ≈ 2s @10Hz). `truncated=True` khi đạt
  `max_episode_steps` (time-limit). PPO bootstrap giá trị qua critic ngay lúc thu thập (xem
  comment trong `train_ppo.py`); SAC đơn giản hơn — **không lưu** transition time-limit vào
  replay buffer thay vì cố bootstrap nó (xem docstring `sac/replay_buffer.py` — lý do kỹ
  thuật: buffer tối ưu bộ nhớ suy ra `next_obs` từ ô kế tiếp, mà sau truncation ô kế tiếp lại
  thuộc episode mới do `env.reset()`).

> 🎛️ **Trọng số reward là siêu tham số cần tinh chỉnh thực nghiệm** — giá trị mặc định trong
> `ppo_config.json`/`sac_config.json` là điểm khởi đầu hợp lý (đúng thứ tự độ lớn theo tài
> liệu), không phải số đã được kiểm chứng bằng thực nghiệm trên môi trường của bạn. Theo dõi
> `episode_log.csv` để tinh chỉnh — nếu policy học cách "chấp nhận" va chạm để tránh phạt
> lệch làn dồn dập, tăng `collision_penalty` hoặc giảm `off_lane_penalty`; nếu xe không chịu
> tăng tốc, giảm `w_lane_offset` tương đối so với `w_speed`.

## Chạy trên máy yếu (RAM 16GB / VRAM 4GB)

Có 2 nguồn tranh chấp tài nguyên khác nhau, cần xử lý riêng:

**VRAM — giới hạn chặt nhất.** Bản thân CARLA (UE4) đã khuyến nghị ≥6GB VRAM; 4GB là mức
biên ngay cả khi không có gì khác dùng GPU. Nếu để PyTorch cũng chiếm VRAM cùng lúc (mặc định
`device=cuda`), rất dễ OOM khi CARLA cần cấp phát thêm lúc đổi cảnh/thời tiết. **Khuyến nghị
mạnh: `--device cpu`.** `PolicyBackbone` rất nhỏ (5 lớp conv, 24–64 kênh) và điểm nghẽn tốc độ
luôn là bước tick CARLA (vật lý + render), không phải vài phép nhân ma trận của mạng này — nên
train trên CPU hầu như không làm chậm thông lượng tổng thể, trong khi giải phóng toàn bộ VRAM
cho một mình CARLA:

```powershell
CarlaUE4.exe -quality-level=Low -RenderOffScreen
python train_ppo.py --config ppo_config.json --device cpu
python train_sac.py --config sac_config.json --device cpu --buffer-capacity 15000
```

**Trước khi quyết định `cpu` hay `cuda`**: bật CARLA một mình (`-quality-level=Low
-RenderOffScreen`) theo đúng flag trên, để yên ~30s cho map load xong, rồi xem VRAM còn trống:
```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
Trên GTX 1650 (4GB), CARLA ở `Low` thường chiếm khoảng 1.5–3GB tuỳ map — nếu còn dư rõ rệt
(≥1GB), có thể thử `--device cuda --batch-size 16` cho actor/critic thay vì CPU, tăng tốc
gradient step; nếu còn dưới ~500MB, giữ `--device cpu` để tránh OOM giữa chừng (đặc biệt lúc
đổi thời tiết/spawn xe — VRAM CARLA cần có thể tăng đột ngột). Theo dõi `nvidia-smi` (thêm `-l
2` để lặp mỗi 2s) trong lúc train để phát hiện sớm nếu VRAM tiến sát 4GB.

**`batch_size` cũng ăn VRAM/RAM trực tiếp — độc lập với độ phân giải camera.** Ảnh one-hot 13
lớp ở 480×384 nặng hơn 160×128 khoảng 9 lần; mặc định trong `ppo_config.json`/`sac_config.json`
đã hạ xuống 64/32 (SAC thấp hơn vì có 2 mạng Q chạy song song) — neo theo đúng `BATCH_SIZE=32`
notebook IL dùng ổn định ở cùng độ phân giải trên GPU 16GB. Nếu vẫn OOM (kể cả khi đã
`--device cpu`, lúc đó là RAM chứ không phải VRAM), hạ tiếp qua `--batch-size 16`.

**RAM 16GB** — chủ yếu ảnh hưởng SAC (`buffer_capacity`, xem công thức ở mục "Giới hạn đã
biết" bên dưới); PPO gần như không đáng lo vì rollout buffer chỉ giữ `n_steps=2048` frame.

Với cấu hình này, máy 16GB/4GB **đủ để smoke-test và debug** toàn bộ vòng lặp trước khi thuê
GPU — không đủ để chạy train dài hàng trăm nghìn bước với tốc độ tốt, vì khi đó điểm nghẽn là
tốc độ tick/render của CARLA (được cải thiện đáng kể bởi GPU khoẻ hơn), không phải mạng nơ-ron.

## Chạy train

**⚠️ Active client**: cả 2 script tự spawn xe và tick world (`synchronous_mode=True`).
**Không** chạy cùng lúc với `../data_collection/` (passive collector) hay
`automatic_control.py` trên cùng world — cả hai sẽ tranh giành quyền điều khiển xe/world
settings.

```powershell
# Terminal 1
CarlaUE4.exe -quality-level=Low

# Terminal 2 (tuỳ chọn — đổi map/thời tiết trước khi train, giống quy trình data_collection)
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(30); c.load_world('Town02')"

# Terminal 3
cd drl_training
pip install -r requirements.txt

# PPO
python train_ppo.py --config ppo_config.json
# hoặc SAC
python train_sac.py --config sac_config.json
```

Checkpoint IL mặc định trỏ tới `../behavior_cloning/best_il_model.pth` — tải file này về máy
local sau khi train xong trên Kaggle (hoặc đổi `il_checkpoint` trong file config /
`--il-checkpoint`).

### Kiểm thử lần đầu (smoke test)

**PPO:**
```powershell
python train_ppo.py --config ppo_config.json --total-steps 4096 --n-steps 512
```
Kỳ vọng: log `update=0 step=512 ...` sau khi xe chạy được ~512 bước.

**SAC:** sửa tạm `learning_starts`/`total_steps` nhỏ trong `sac_config.json` (hoặc tạo bản
copy `sac_smoke_config.json`), ví dụ `learning_starts=200, total_steps=1000`, rồi:
```powershell
python train_sac.py --config sac_config.json
```
Kỳ vọng: sau bước 200, `update_log.csv` bắt đầu có `critic_loss`/`actor_loss` không phải NaN.

Cả hai: nếu bị treo ở bước tick đầu tiên (`_get_seg_frame` timeout) — kiểm tra CARLA server
còn sống và không có client passive nào khác giữ `synchronous_mode`. `episode_log.csv`/
`update_log.csv` được tạo trong thư mục `output` (mặc định `runs/ppo_lane_keep/` hoặc
`runs/sac_lane_keep/`).

### Resume

```powershell
python train_ppo.py --config ppo_config.json --resume runs/ppo_lane_keep/ppo_latest.pt
python train_sac.py --config sac_config.json --resume runs/sac_lane_keep/sac_latest.pt
```

## Đánh giá

Dùng chung một script cho cả hai thuật toán:

```powershell
python evaluate.py --algorithm ppo --config ppo_config.json ^
    --resume runs/ppo_lane_keep/ppo_latest.pt --episodes 10 --deterministic

python evaluate.py --algorithm sac --config sac_config.json ^
    --resume runs/sac_lane_keep/sac_latest.pt --episodes 10 --deterministic
```

In ra reward trung bình, tỷ lệ va chạm, độ lệch làn trung bình theo `|lane_offset_m|` — số
liệu này dùng để so sánh trực tiếp với MAE của bước IL (mục 11 trong notebook IL) **và giữa
PPO với SAC**, đưa vào báo cáo đồ án.

## Giới hạn đã biết / hướng mở rộng

- **Chưa test với CARLA thật** (xem cảnh báo đầu file) — validate kỹ ở quy mô nhỏ trước.
- **1 CARLA instance / 1 env** — không có song song hoá nhiều world để tăng thông lượng thu
  thập rollout; nếu cần, chạy nhiều `CarlaUE4.exe` trên các port khác nhau và mở rộng
  `train_ppo.py`/`train_sac.py` thành vòng lặp nhiều `CarlaLaneKeepEnv` (mã hiện tại được
  viết theo hướng dễ mở rộng: mọi state đều local trong instance `CarlaLaneKeepEnv`, không
  có biến toàn cục).
- **`sac/replay_buffer.py`'s bộ nhớ**: RAM ≈ `buffer_capacity × width × height` byte (uint8,
  1 kênh, không one-hot — one-hot hoá diễn ra trong `PolicyBackbone.forward`, ngay trước
  conv; chỉ tốn RAM host, không tốn VRAM vì chỉ 1 batch nhỏ được chuyển lên GPU mỗi lần
  update). Mặc định `width=480, height=384` (khớp đúng độ phân giải IL đã train — xem mục
  "Độ phân giải" bên dưới) + `buffer_capacity=50000` ≈ 9.2GB, nhắm tới máy thuê GPU có
  ≥32GB RAM. Đang smoke-test trên máy cá nhân RAM nhỏ hơn? Hạ tạm bằng
  `--buffer-capacity 15000` (≈2.7GB) — không cần sửa file JSON. `train_sac.py` tự in ước
  lượng GB thực tế lúc khởi động, dựa trên `width`/`height`/`buffer_capacity` hiện tại.
- **Chưa có A*/route** — khi `../router_plan/Global_Route_Planner.py` được cài, thêm
  `route_target_local_x/y` + one-hot `route_command` vào `ObservationContract` và
  `CarlaLaneKeepEnv._build_state`, rồi train lại IL với các cột này trước khi warm-start DRL
  (giữ đúng nguyên tắc: IL và DRL luôn cùng một observation contract).
