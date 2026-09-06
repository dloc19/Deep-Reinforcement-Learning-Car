# drl_training_sac — fine-tune SAC (off-policy) cho bám làn

Pipeline của đồ án (xem `../README.md` cho bức tranh tổng): **1) segmentation** →
**2) Imitation Learning** (`../behavior_cloning/`, train trên Kaggle) → **3) DRL fine-tune**.
Thư mục này là **nhánh SAC** của bước 3: một thuật toán, một entrypoint, một file config.

> **Bản PPO (on-policy) nằm ở `../drl_training/`.** Hai thư mục dùng chung kiến trúc mạng và
> hợp đồng quan sát nhưng độc lập về code. Mọi tham chiếu tới `train_ppo.py` / `runs/ppo_v*`
> trong chú thích ở đây là **bằng chứng đo được** bên đó, không phải code trong thư mục này.

## Cấu trúc

```
drl_training_sac/
├── sac_config.json          # cấu hình DUY NHẤT cho lần train thật (60k step, ~6 giờ)
├── sac_config_smoke.json    # bản rút gọn 8k step (~15 phút) để kiểm thử trước
├── config.py                # nạp --config JSON + override CLI (ENV_DEFAULTS + SAC_DEFAULTS)
├── csv_logger.py            # CSV logger nhỏ
├── train_sac.py             # ENTRYPOINT train
├── evaluate.py              # chạy checkpoint đã train, đo reward/va chạm/lệch làn
├── policy/
│   ├── backbone.py          # CNN(seg one-hot)+MLP(scalar) — KHỚP KIẾN TRÚC SteeringNet (IL)
│   ├── il_compat.py         # remap checkpoint IL -> state_dict
│   ├── observation.py       # dựng scalar vector đúng chuẩn hoá đã lưu trong checkpoint IL
│   └── checkpoint_io.py     # load + validate checkpoint IL
├── envs/
│   └── carla_lane_keep_env.py   # active client CARLA (sync mode), reward/done
├── sac/
│   ├── networks.py          # GaussianPolicy (squashed-tanh, warm-start IL) + TwinQNetwork
│   ├── replay_buffer.py     # replay buffer off-policy, tối ưu bộ nhớ
│   └── sac_agent.py         # twin-Q update, auto temperature, Polyak target, ràng buộc BC
├── collect_probe.py         # thu bộ quan sát THẬT (probe_obs.npz) làm chuẩn đo độ trôi
├── drift.py                 # đo policy đã trôi bao xa khỏi IL trên chính bộ probe đó
├── check_weather_invariance.py  # kiểm chứng ảnh seg bất biến với thời tiết
└── runs/                    # checkpoint + log (bị .gitignore chặn)
```

## Chạy

```powershell
# Terminal 1 — server
CarlaUE4.exe -quality-level=Low

# Terminal 2
conda activate carla_rl
cd Deep_RL_Carla\drl_training_sac

# 1) smoke test 8k step (~15 phút) — LUÔN chạy cái này trước
python train_sac.py --config sac_config_smoke.json

# 2) lần train thật 60k step (~3 giờ, ghi vào runs/sac_f/)
python train_sac.py --config sac_config.json
```

**Đọc log thế nào.** `critic_loss` **không** dùng để đánh giá sức khoẻ critic: giá trị ghi
log là *minibatch cuối cùng* của cửa sổ 1000 step, nên nó là biến đuôi nặng phụ thuộc việc có
transition va chạm rơi vào batch đó hay không — trên thang reward chuẩn hoá,
`collision_penalty` 75 cho MSE ≈ 75²/32 ≈ 176. Đo được ở `runs/smoke`: loss vọt 0.75 → 183
trong khi `ev` giữ nguyên 0.57 → 0.50, rồi lập tức về 0.95. **Chỉ số cần theo dõi là `ev`**
(`explained_variance`): ~1.0 tốt, ~0 vô dụng, <0 hỏng thật.

Trong 60k step, actor bị **đóng băng tới ~env step 18000** (`critic_warmup_steps` 4000 gradient
step × `train_freq` 4 + `learning_starts` 2000) = 30% ngân sách. Suốt giai đoạn đó log in
`[critic-warmup]` và reward **không** cải thiện — đó là thiết kế, không phải hỏng.

**⚠️ Active client**: script tự spawn xe và tick world (`synchronous_mode=True`). **Không**
chạy cùng lúc với `../data_collection/` hay `automatic_control.py` trên cùng một world.

Kỳ vọng của smoke test:

- in `Da warm-start actor tu: ...best_il_model.pth` và `log_std_head khoi tao [...]`;
- sau bước 500 (`learning_starts`), `runs/smoke/update_log.csv` có `critic_loss` khác NaN;
- 1000 step đầu in `[critic-warmup]` — actor đang bị đóng băng, đúng thiết kế;
- nếu treo ngay tick đầu (`_get_seg_frame` timeout): server chết, hoặc còn một client passive
  khác đang giữ `synchronous_mode`.

### Resume

```powershell
python train_sac.py --config sac_config.json --resume runs/sac_f/sac_latest.pt
```

Replay buffer **không** nằm trong checkpoint (2.2GB), nên mỗi lần resume SAC phải nạp lại
`learning_starts` transition trước khi học tiếp (~2 phút). Đổi `--seed` giữa các phiên nối
tiếp nhau, nếu không mỗi phiên sẽ gặp y hệt một chuỗi kịch bản.

### Đánh giá

Mỗi lô đánh giá chạy trên **một** bản đồ (nếu để nguyên danh sách 4 town, env sẽ xoay vòng và
trộn kết quả của nhiều bản đồ vào một con số trung bình không ứng với bản đồ nào):

```powershell
python evaluate.py --config sac_config.json --town Town05 ^
    --resume runs/sac_f/sac_latest.pt --episodes 30 --deterministic ^
    --eval-csv-out runs/sac_f/eval_Town05.csv
```

`--episodes 30`, không phải 10: ở n=10 tỉ lệ va chạm chỉ là đếm 1–2 vụ nên không phân biệt
được hai checkpoint bất kỳ. Town05 là tập held-out (IL chưa từng train trên đó).

### Đo độ trôi khỏi IL

Vấn đề trung tâm của SAC ở bài này (xem "Kết quả đã đo"): policy trôi khỏi warm-start IL mà
chỉ số train không báo. Đo trực tiếp trên quan sát thật:

```powershell
python collect_probe.py 256 Town01,Town04     # thu probe_obs.npz (cần CARLA, chạy 1 lần)
python drift.py runs/sac_f/sac_step_*.pt      # so từng checkpoint với policy IL gốc
```

## Cấu hình đang dùng và vì sao

`sac_config.json` = bộ tham số của `runs/sac_d` — lần chạy duy nhất trong 5 lần cải thiện
đồng thời cả ba chỉ số, nhưng bị cắt ở step 39154/60000.

| tham số | giá trị | lý do |
|---|---|---|
| `gamma` | **0.95** | 0.99 cho chân trời ~100 bước = 20s, một hành động chỉ chiếm 1% → `dQ/da` quá nhỏ để actor học. 0.95 → ~20 bước = 4s, đủ dài cho bám làn. |
| `actor_lr` | 1e-5 | thấp hơn cả PPO (2e-5): SAC không có vùng tin cậy nào chặn actor nhảy xa, mà Adam thì chuẩn hoá độ lớn gradient. |
| `bc_coef` | **1.0** | ràng buộc BC kiểu TD3+BC neo `mean_action` về IL. `bc_coef=0` (sac_e) làm lệch làn xấu đi 47%. |
| `freeze_log_std` | true | đóng băng nhiễu thăm dò ở đúng `action_std` của IL. |
| `critic_warmup_steps` | 4000 | 4000 gradient step đầu chỉ train critic, actor đóng băng — không thể phá warm-start. |
| `explore_epsilon` | 0.25 | trong giai đoạn warmup, 25% bước lấy hành động ngẫu nhiên toàn dải để critic học được *sự phụ thuộc hành động* của Q. Chỉ an toàn vì actor đang đóng băng. |
| `batch_size` / `train_freq` | 32 / 4 | đo trên GTX 1650 Max-Q 4GB: 0.98s mỗi gradient step, VRAM 0.60GB. `train_freq=1` sẽ mất 45 giờ và gần như chắc chắn OOM cạnh CARLA. |
| `total_steps` | 60000 | 60k quyết định × 0.2s = 3.3 giờ mô phỏng. Đo thật ngày 6/9/2026 trên GTX 1650 Max-Q: 6.2 steps/s khi actor+critic cùng train → ≈ 2.7 giờ + chi phí nạp bản đồ ≈ **3–3.5 giờ**. |

Mọi con số còn lại (reward, camera, env) giữ nguyên như PPO v4 để hai thuật toán so sánh được
— lý do chi tiết của từng trọng số nằm trong chú thích của `config.py`.

## Kết quả đã đo (5 lần chạy trước)

Đo trên `runs/*/episode_log.csv`, so 1/3 số episode đầu với 1/3 cuối của mỗi lần chạy:

| run | khác biệt | va chạm (đầu→cuối) | lệch làn, đường thẳng | reward |
|---|---|---|---|---|
| sac_a | bc_coef 2.5 | 73.7% → 79.2% | 0.51 → 0.48 | 5.0 → −2.3 |
| sac_b | bc_coef 1.0 | 71.6% → 65.3% | 0.44 → 0.53 | 6.0 → 11.6 |
| sac_c | + critic_warmup 4000 | 93.9% → 82.0% | 0.73 → 0.62 | dừng ở 21k |
| **sac_d** | **+ gamma 0.95** | **71.6% → 62.7%** | **0.76 → 0.51** | **−115 → +20** |
| sac_e | bc_coef 0.0 (SAC thuần) | 80.0% → 83.1% | 0.74 → **1.09** | xấu đi |

Đánh giá 30 episode/bản đồ trên checkpoint SAC tốt nhất (sac_b) so với PPO v4:

| | Town01 | Town03 | Town04 | Town05 |
|---|---|---|---|---|
| PPO va chạm | **0.0%** | **16.7%** | **33.3%** | **13.3%** |
| SAC va chạm | 83.3% | 43.3% | 36.7% | 43.3% |
| PPO lệch làn (m) | 0.129 | 0.375 | 0.099 | 0.148 |
| SAC lệch làn (m) | 0.145 | **0.187** | **0.081** | 0.191 |

Đọc bảng này: **SAC bám làn tốt hơn PPO nhưng va chạm nhiều hơn hẳn**. Nguyên nhân đã xác
định được là chuỗi nhân quả critic → actor: quét lệnh lái toàn dải −1..+1 chỉ làm Q đổi
0.3–1.7% so với biến thiên giữa các trạng thái, tức critic học được `Q(s,a) ≈ V(s)` — nó biết
đang ở tình huống nào nhưng không phân biệt *làm gì* trong tình huống đó. Actor chỉ học qua
`dQ/da`, nên không có gì để học. `gamma 0.95` + `explore_epsilon` là hai can thiệp nhắm thẳng
vào chỗ này; sac_d là lần đầu chúng có tác dụng, và nó chưa chạy hết.

Các bảng so sánh đầy đủ PPO-vs-SAC (`compare_*.csv`) và script sinh ra chúng đã được gỡ khỏi
thư mục này để pipeline chỉ còn SAC; lấy lại bằng `git checkout aac7fed -- <đường dẫn file>`.

## Độ phân giải — camera 480×384, observation 240×192

`CarlaLaneKeepEnv` tự spawn camera segmentation **sống** trong lúc train — **không** đọc lại
dữ liệu đã thu ở `data_collection/`. Nhưng ở đây có **hai** con số, không phải một:

| Tham số | Giá trị | Là gì |
|---|---|---|
| `width` / `height` | **480 × 384** | độ phân giải CAMERA render ra, khớp `collector_config.json` |
| `obs_width` / `obs_height` | **240 × 192** | độ phân giải OBSERVATION đưa vào mạng, sau `resize_class_map()` |

**Vì sao observation phải là 240×192.** Đó chính xác là `IMAGE_WIDTH`/`IMAGE_HEIGHT` mà
`SteeringNet` đã được train trong `train_il_v9.ipynb`. `PolicyBackbone` dùng
`AdaptiveAvgPool2d((1,1))` nên **mọi** kích thước đều chạy được và **không có lỗi nào báo** —
đó chính là cái bẫy: actor warm-start ở 480×384 vẫn chạy ngon lành nhưng nhìn thấy cấu trúc
lớn gấp 2× so với lúc học, khiến phần warm-start mất giá trị trong im lặng.

**Vì sao camera vẫn giữ 480×384.** CARLA render **trực tiếp** ở độ phân giải cấu hình. Render
thẳng ở 240×192 thì vạch kẻ làn (1–3 px) biến mất ngay lúc rasterize. Render 480×384 rồi hạ
mẫu bằng `resize_class_map()` thì vạch được bảo tồn theo độ phủ diện tích (giống hệt
`downscale_labels()` của notebook IL, đã kiểm chứng byte-identical). Đo trên mask phối cảnh:
nearest thuần chỉ giữ 69% số pixel `RoadLine` so với cách này.

## Hợp đồng quan sát / khớp checkpoint IL

`policy/observation.py::ObservationContract` đọc `continuous_cols`, `raw_action_cols`,
`norm_stats`, `traffic_light_vocab`, `scalar_feature_dim` **trực tiếp từ checkpoint IL**
(`best_il_model.pth`) thay vì hard-code — vì notebook IL chạy trên Kaggle còn module này chạy
local, hai bên không thể `import` chung code Python, nên checkpoint tự mang theo "hợp đồng"
của chính nó. Đổi tập đặc trưng ở notebook IL → **train lại IL trước**, checkpoint mới sẽ tự
được đọc đúng, không cần sửa gì ở đây.

`policy/il_compat.py` remap `SteeringNet` → `GaussianPolicy`: `backbone` (giống
`SteeringNet.conv/cnn_fc/scalar_mlp`) + `trunk_head` (2 lớp đầu của `SteeringNet.head`) +
`mean_head`. Riêng `log_std_head` không warm-start được vì IL không có khái niệm noise scale —
nó được khởi tạo từ `action_std` mà notebook IL ghi lại. Việc nạp **báo lỗi rõ ràng** (không
âm thầm bỏ qua) nếu shape lệch.

## Observation / Action / Reward / Done

Theo đúng `docs/csv_fields_by_task.md` (mục DRL) và `docs/manual_thu_thap_du_lieu.md` §9.2,
thu hẹp về phạm vi bám làn:

- **Observation**: `seg` (ảnh class-ID, one-hot hoá bên trong `PolicyBackbone`) +
  scalar = `[speed_mps, yaw_rate_rps, speed_limit_kmh]` (z-score theo `norm_stats` của
  checkpoint IL) + `[previous_steer, previous_longitudinal]` (raw, đã ∈[-1,1]) + one-hot
  `traffic_light_state` (4 lớp). **Không** có `lane_offset_m`/`heading_error_rad` (chỉ dùng
  cho reward) — xem lý do trong `../behavior_cloning/train_il_v9.ipynb`.
- **Action**: `[steer, longitudinal] ∈ [-1,1]`, `longitudinal≥0`→throttle, `<0`→brake (giống
  hệt cách `../data_collection/carla_collector/writer.py` mã hoá `longitudinal`). SAC đạt
  khoảng này bằng tanh-squash trong `GaussianPolicy.sample`.
- **Reward** (`reward_mode: "normalized"` — mỗi số hạng chia cho thang tự nhiên của nó, nhờ
  vậy `w_*` mới thật sự là trọng số tương đối và reward nhất quán giữa các town):
  `+ w_speed·(v/target_speed) − w_lane_offset·(|lane_offset|/nửa bề rộng làn) −
  w_heading·(|heading_error|/45°) − w_steer_delta·Δsteer² − w_long_delta·Δlong² −
  w_yaw_rate·yaw_rate² − off_lane_penalty − lane_invasion_penalty·(số vạch mới vượt) −
  collision_penalty`.
- **Done**: `terminated=True` khi va chạm mới, hoặc lệch làn liên tục ≥
  `off_lane_patience_steps` bước (10 quyết định = 2s). `truncated=True` khi đạt
  `max_episode_steps`. Transition bị cắt vì hết giờ **được lưu nhưng không bao giờ được
  sample** (xem docstring `sac/replay_buffer.py`).

## Chạy trên máy yếu (RAM 16GB / VRAM 4GB)

**VRAM là giới hạn chặt nhất.** CARLA (UE4) khuyến nghị ≥6GB; 4GB đã là mức biên. Trước khi
quyết `cpu` hay `cuda`: bật CARLA một mình, đợi ~30s cho map load xong, rồi

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

CARLA ở `Low` thường chiếm 1.5–3GB tuỳ map. Còn dư ≥1GB → `--device cuda` với `batch_size=32`
(đo được: 0.60GB VRAM). Còn dưới ~500MB → `--device cpu`, và thêm `-RenderOffScreen` cho
server. Theo dõi `nvidia-smi -l 2` trong lúc train.

**RAM**: replay buffer ≈ `buffer_capacity × obs_width × obs_height` byte (uint8, 1 kênh —
one-hot hoá diễn ra trong `PolicyBackbone.forward`, ngay trước conv, nên chỉ tốn RAM host).
Mặc định 50000 × 240 × 192 ≈ **2.2GB**, vừa máy 16GB. Thiếu RAM thì hạ bằng
`--buffer-capacity 15000` (≈0.7GB) — không cần sửa file JSON. `train_sac.py` in ước lượng GB
thực tế lúc khởi động.

## Giới hạn đã biết / hướng mở rộng

- **Va chạm còn cao** — xem "Kết quả đã đo". Nếu sac_f (60k step đầy đủ) vẫn không hạ được va
  chạm xuống dưới mức PPO, hướng tiếp theo là làm critic phân biệt được hành động: tăng
  `explore_epsilon`, kéo dài `critic_warmup_steps`, hoặc thêm n-step return.
- **1 CARLA instance / 1 env** — không song song hoá nhiều world. Nếu cần, chạy nhiều
  `CarlaUE4.exe` trên các port khác nhau và mở rộng `train_sac.py` thành vòng lặp nhiều
  `CarlaLaneKeepEnv` (mọi state đều local trong instance, không có biến toàn cục).
- **Chưa có A*/route** — chỉ bám làn. Khi `../router_plan/Global_Route_Planner.py` được cài,
  thêm `route_target_local_x/y` + one-hot `route_command` vào `ObservationContract` và
  `CarlaLaneKeepEnv._build_state`, rồi **train lại IL** với các cột này trước khi warm-start
  DRL (nguyên tắc: IL và DRL luôn cùng một observation contract).
- **Vẽ biểu đồ**: `../drl_training/plot_metrics.py` đọc được `episode_log.csv`/`update_log.csv`
  của thư mục này (`--sac-dir ../drl_training_sac/runs/sac_f`).
