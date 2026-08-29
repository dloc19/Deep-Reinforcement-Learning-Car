# 📘 Tài liệu Hướng dẫn Huấn luyện DRL (Fine-tune PPO/SAC) — CARLA 0.9.10

> Tài liệu này mô tả quy trình chạy **bước 3/3** của pipeline đồ án: fine-tune một policy
> Deep Reinforcement Learning (PPO hoặc SAC) cho bài toán bám làn, warm-start từ checkpoint
> Imitation Learning (`best_il_model.pth`) đã train ở bước 2 (`../behavior_cloning/`). Khác
> với 2 bước đầu (chạy trên Kaggle), bước này chạy **local**, cần một **CARLA 0.9.10 server
> đang sống** vì `drl_training/` là một **active client** — tự spawn xe và tick world, chứ
> không đọc lại dữ liệu đã thu như `data_collection/`.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [PPO hay SAC?](#4-ppo-hay-sac)
5. [Observation / Action / Reward / Done](#5-observation--action--reward--done)
6. [Cấu hình](#6-cấu-hình)
7. [Quy trình chạy huấn luyện](#7-quy-trình-chạy-huấn-luyện)
8. [Theo dõi quá trình train](#8-theo-dõi-quá-trình-train)
9. [Resume training](#9-resume-training)
10. [Đánh giá checkpoint](#10-đánh-giá-checkpoint)
11. [Chạy trên máy yếu (RAM 16GB / VRAM 4GB)](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb)
12. [Tham số tham khảo](#12-tham-số-tham-khảo)
13. [Xử lý sự cố thường gặp](#13-xử-lý-sự-cố-thường-gặp)

---

## 1. Tổng quan hệ thống

### Vị trí trong pipeline

```
1) data_collection/    Thu thập dữ liệu passive-client  → states.csv + seg_label/*.png
        │
        ▼
2) behavior_cloning/   [Kaggle] Segmentation → Imitation Learning
        │                       → best_carla_segmentation.pth, best_il_model.pth
        ▼
3) drl_training/        [Local, cần CARLA server sống]  ← TÀI LIỆU NÀY
                         Warm-start actor từ best_il_model.pth,
                         fine-tune bằng PPO hoặc SAC → policy DRL cuối cùng
```

### Kiến trúc: active client, KHÔNG phải passive collector

Đây là điểm khác biệt quan trọng nhất so với `data_collection/`:

| | `data_collection/` (bước 1) | `drl_training/` (bước 3) |
|---|---|---|
| Vai trò | Passive client — chỉ **quan sát** xe do `automatic_control.py` lái | Active client — **tự spawn xe, tự lái, tự `world.tick()`** |
| `synchronous_mode` | Không set (client khác lo) | **Tự set `True`** trong `CarlaLaneKeepEnv` |
| Chạy cùng lúc với client khác trên cùng world? | Được (đó là mục đích) | **KHÔNG** — tranh giành quyền điều khiển xe/world settings |

> [!CAUTION]
> **Không** chạy `train_ppo.py`/`train_sac.py`/`evaluate.py` cùng lúc với `data_collection/`
> hoặc `automatic_control.py` trên cùng một CARLA world. Cả hai sẽ tranh giành
> `synchronous_mode` và quyền điều khiển xe, gây lỗi khó chẩn đoán (đứng hình, timeout, hoặc
> xe bị 2 client cùng điều khiển).

### Phạm vi: chỉ bám làn (lane-keeping)

Không điều hướng theo tuyến A* — `router_plan/Global_Route_Planner.py` hiện **chưa được cài**
(trống), các cột `route_*` trong schema vẫn để trống theo đúng thiết kế
(`../data_collection/ASTAR_SCHEMA.md`). Observation ở bước này **không** có
`route_command`/`route_target_*`. Khi Router Plan được cài, mở rộng observation/action ở đây
là việc thêm trường, không phải viết lại.

### Hợp đồng quan sát dùng chung với IL

`policy/observation.py::ObservationContract` đọc `continuous_cols`, `raw_action_cols`,
`norm_stats`, `traffic_light_vocab`, `scalar_feature_dim` **trực tiếp từ checkpoint IL**
(`best_il_model.pth`) thay vì hard-code trong code DRL — vì notebook IL chạy trên Kaggle còn
`drl_training/` chạy local, hai bên không thể `import` chung code Python. Checkpoint IL tự
mang theo "hợp đồng" của chính nó.

> [!IMPORTANT]
> Nếu bạn đổi tập đặc trưng ở notebook IL, **train lại IL trước** rồi mới fine-tune DRL —
> checkpoint mới sẽ tự động được cả `train_ppo.py` lẫn `train_sac.py` đọc đúng, không cần sửa
> gì ở `drl_training/`. Ngược lại, nếu gặp lỗi shape lúc warm-start (thường do dùng nhầm
> checkpoint IL cũ), train lại IL bằng notebook đã sửa trong `../behavior_cloning/`.

---

## 2. Cài đặt môi trường

### Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| CARLA Simulator | 0.9.10 |
| Python | 3.7 (giới hạn của CARLA 0.9.10 client API trên Windows) |
| numpy | ≥1.18, <2.0 |
| torch | ≥1.9, ≤1.13.1 (1.13.1 là bản cuối chính thức hỗ trợ Python 3.7) |
| opencv-python-headless | ≥4.5 |
| OS | Windows 10/11 |
| GPU (khuyến nghị) | ≥6GB VRAM nếu chạy CARLA + PyTorch cùng `cuda`; xem [mục 11](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb) nếu máy yếu hơn |

### Cài đặt thư viện Python

```powershell
# Bước 1: Cài CARLA Python API (giống hệt bước 1 của manual_thu_thap_du_lieu.md)
cd C:\CARLA_0.9.10\PythonAPI\carla\dist
pip install carla-0.9.10-py3.7-win-amd64.egg

# Bước 2: Cài thư viện của drl_training
cd C:\Users\dloc\Desktop\Do_An\Deep_RL_Carla\drl_training
pip install -r requirements.txt

# Bước 3: Kiểm tra
python -c "import carla, torch; print('CARLA OK | torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Checkpoint IL bắt buộc phải có trước

Tải `best_il_model.pth` từ Kaggle (output của `../behavior_cloning/train_il_v9.ipynb`)
về `../behavior_cloning/best_il_model.pth` trên máy local — đây là đường dẫn mặc định
`il_checkpoint` trong cả `ppo_config.json` lẫn `sac_config.json`. Không có file này thì
`train_ppo.py`/`train_sac.py` không khởi động được (ngay cả khi `--no-warm-start`, checkpoint
vẫn được đọc để lấy `ObservationContract`).

---

## 3. Cấu trúc thư mục dự án

```
drl_training/
├── ppo_config.json / sac_config.json   # cấu hình mặc định (connection/camera/env/reward + thuật toán)
├── config.py                            # nạp --config JSON + override CLI, dùng chung cho cả 2 thuật toán
├── csv_logger.py                        # CSV logger nhỏ dùng chung train_ppo.py/train_sac.py
├── train_ppo.py / train_sac.py          # entrypoint train — chọn 1 trong 2
├── evaluate.py                          # chạy checkpoint đã train (PPO hoặc SAC), đo reward/va chạm/lệch làn
├── requirements.txt
├── policy/
│   ├── backbone.py           # CNN(seg one-hot)+MLP(scalar) trunk — KHỚP KIẾN TRÚC SteeringNet (IL)
│   ├── il_compat.py          # helper remap checkpoint IL -> state_dict, dùng chung PPO + SAC
│   ├── actor_critic.py       # PPO: GaussianActor + ValueCritic + loader warm-start IL
│   ├── observation.py        # dựng scalar vector đúng chuẩn hoá đã lưu trong checkpoint IL
│   └── checkpoint_io.py      # load + validate checkpoint IL
├── envs/
│   └── carla_lane_keep_env.py  # active client CARLA, đồng bộ (sync mode), reward/done
├── ppo/
│   ├── rollout_buffer.py     # buffer on-policy + GAE(λ)
│   └── ppo_agent.py          # clipped surrogate PPO update
└── sac/
    ├── networks.py           # GaussianPolicy (squashed-tanh, warm-start IL) + TwinQNetwork
    ├── replay_buffer.py      # replay buffer off-policy, tối ưu bộ nhớ
    └── sac_agent.py          # twin-Q update, auto temperature tuning, Polyak target update
```

### Tra cứu nhanh: sửa ở đâu?

| Muốn thay đổi | File cần sửa |
|---|---|
| Trọng số reward, điều kiện dừng episode | `*_config.json` mục `reward`/`env`, hoặc trực tiếp `envs/carla_lane_keep_env.py` |
| Kiến trúc mạng (backbone/actor/critic) | `policy/backbone.py`, `policy/actor_critic.py`, `sac/networks.py` |
| Cách warm-start từ IL | `policy/il_compat.py` |
| Đặc trưng scalar đưa vào observation | `policy/observation.py` (đọc từ checkpoint IL — **không sửa ở đây**, sửa ở notebook IL rồi train lại IL) |
| Thuật toán update PPO | `ppo/ppo_agent.py`, `ppo/rollout_buffer.py` |
| Thuật toán update SAC | `sac/sac_agent.py`, `sac/replay_buffer.py` |
| Vòng lặp train chính, log, checkpoint | `train_ppo.py` / `train_sac.py` |

---

## 4. PPO hay SAC?

| | **PPO** (`train_ppo.py`) | **SAC** (`train_sac.py`) |
|---|---|---|
| Kiểu | On-policy | Off-policy |
| Bộ nhớ | Thấp — chỉ giữ 1 rollout (`n_steps=2048` × 240×192 ≈ 90MB) trong RAM | Cao hơn — replay buffer sống suốt quá trình train (`buffer_capacity=50000` × 240×192 ≈ 2.2GB mặc định) |
| Mẫu hiệu quả | Thấp hơn — cần nhiều bước môi trường hơn để hội tụ | Cao hơn — tái sử dụng transition nhiều lần qua replay buffer |
| Độ ổn định / dễ tune | Cao — clipped surrogate + early-stop theo KL tự bảo vệ | Nhạy hơn với learning rate/tau, nhưng auto temperature-tuning giảm bớt việc chỉnh tay |
| Khuyến nghị | Máy đơn, CARLA chạy chậm hơn training GPU, muốn kết quả dễ debug trước | Đủ RAM (máy thuê GPU), muốn tận dụng tối đa từng bước môi trường |

Cả hai đáng đưa vào báo cáo đồ án như một so sánh thực nghiệm — cùng warm-start, cùng reward,
khác thuật toán fine-tune. Xem [mục 12](#12-tham-số-tham-khảo) cho lý do độ phân giải camera
mặc định là 480×384 (khớp đúng notebook IL).

---

## 5. Observation / Action / Reward / Done

Giống hệt nhau cho cả PPO và SAC vì cùng dùng `envs/carla_lane_keep_env.py`:

- **Observation**: `seg` (ảnh class-ID, one-hot hoá bên trong `PolicyBackbone`) + scalar =
  `[speed_mps, yaw_rate_rps, speed_limit_kmh]` (z-score theo `norm_stats` của checkpoint IL) +
  `[previous_steer, previous_longitudinal]` (raw, đã ∈[-1,1]) + one-hot `traffic_light_state`
  (4 lớp). **Không** có `lane_offset_m`/`heading_error_rad` trong observation — hai giá trị
  này chỉ dùng cho reward, tránh policy "học tắt" từ chính metric đánh giá.
- **Action**: `[steer, longitudinal] ∈ [-1,1]`, `longitudinal≥0`→throttle, `<0`→brake.
- **Reward** (trọng số cấu hình ở `*_config.json["reward"]`):

  ```
  r = + w_speed · clip(forward_speed, 0, speed_limit)
      − w_lane_offset · |lane_offset_m|
      − w_heading · |heading_error_rad|
      − w_steer_delta · Δsteer²
      − w_long_delta · Δlongitudinal²
      − w_yaw_rate · yaw_rate²
      − off_lane_penalty         (mỗi bước còn lệch làn)
      − lane_invasion_penalty · (số lần vượt vạch mới)
      − collision_penalty        (khi va chạm)
  ```

- **Done**: `terminated=True` khi va chạm mới, hoặc lệch làn liên tục ≥
  `off_lane_patience_steps` bước (mặc định 20 bước ≈ 2s @10Hz). `truncated=True` khi đạt
  `max_episode_steps` (time-limit; PPO bootstrap giá trị qua critic, SAC không lưu transition
  này vào replay buffer — xem chi tiết kỹ thuật trong docstring `train_ppo.py`/`sac/replay_buffer.py`).

> [!TIP]
> Trọng số reward là **siêu tham số cần tinh chỉnh thực nghiệm** — giá trị mặc định là điểm
> khởi đầu hợp lý, không phải số đã kiểm chứng trên môi trường của bạn. Theo dõi
> `episode_log.csv` ([mục 8](#8-theo-dõi-quá-trình-train)) để tinh chỉnh: nếu policy "chấp
> nhận" va chạm để tránh phạt lệch làn dồn dập → tăng `collision_penalty` hoặc giảm
> `off_lane_penalty`; nếu xe không chịu tăng tốc → giảm `w_lane_offset` tương đối so với
> `w_speed`.

---

## 6. Cấu hình

### File cấu hình: `ppo_config.json` / `sac_config.json`

Cùng cấu trúc, khác mục thuật toán cuối (`ppo` hoặc `sac`):

```json
{
  "connection": { "host": "127.0.0.1", "port": 2000, "timeout": 20.0 },
  "camera": {
    "width": 480, "height": 384, "fov": 90.0, "fps": 10.0,
    "camera_x": 1.5, "camera_y": 0.0, "camera_z": 2.4, "camera_pitch": -5.0
  },
  "env": {
    "vehicle_filter": "vehicle.lincoln.mkz2017",
    "max_episode_steps": 1000, "off_lane_patience_steps": 20,
    "warmup_ticks": 4, "frame_timeout": 5.0, "no_rendering": false, "seed": 42
  },
  "reward": {
    "w_speed": 1.0, "w_lane_offset": 1.0, "w_heading": 0.5,
    "w_steer_delta": 1.0, "w_long_delta": 0.5, "w_yaw_rate": 0.1,
    "off_lane_penalty": 5.0, "collision_penalty": 50.0, "lane_invasion_penalty": 1.0
  },
  "ppo": {
    "il_checkpoint": "../behavior_cloning/best_il_model.pth",
    "output": "./runs/ppo_lane_keep",
    "warm_start": true, "device": "cuda", "gamma": 0.99,
    "learning_rate": 0.0003, "gae_lambda": 0.95, "clip_range": 0.2,
    "value_clip_range": 0.2, "entropy_coef": 0.0, "value_coef": 0.5,
    "max_grad_norm": 0.5, "epochs": 10, "batch_size": 64, "target_kl": 0.02,
    "total_steps": 2000000, "n_steps": 2048,
    "save_every_updates": 5, "eval_every_updates": 10, "eval_episodes": 3
  }
}
```

### Giải thích các nhóm tham số

#### Nhóm `connection` / `camera` / `env` — dùng chung PPO & SAC

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `host`, `port`, `timeout` | `127.0.0.1`, `2000`, `20.0` | Kết nối CARLA server |
| `width`, `height` | `480`, `384` | Độ phân giải camera **sống** — spawn trực tiếp lúc train, không đọc dữ liệu đã thu; khớp đúng notebook IL, xem [mục 12](#12-tham-số-tham-khảo) |
| `fov`, `fps` | `90.0`, `10.0` | Field of view, tần suất camera |
| `camera_x/y/z`, `camera_pitch` | `1.5, 0.0, 2.4, -5.0` | Vị trí/góc camera trên xe — khớp `data_collection/collector_config.json` |
| `vehicle_filter` | `vehicle.lincoln.mkz2017` | Blueprint xe spawn |
| `max_episode_steps` | `1000` | Time-limit mỗi episode (≈100s @10Hz) |
| `off_lane_patience_steps` | `20` | Số bước lệch làn liên tục trước khi `terminated=True` (≈2s @10Hz) |
| `warmup_ticks` | `4` | Số tick chờ sau `reset()` trước khi lấy observation đầu tiên |
| `frame_timeout` | `5.0` | Giây chờ tối đa 1 frame camera trước khi coi là lỗi |
| `no_rendering` | `false` | `true` = tắt render (nhanh hơn, cần `-RenderOffScreen` phía CARLA) |
| `seed` | `42` | Seed cho reproducibility |

#### Nhóm `reward`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `w_speed` | `1.0` | Thưởng tốc độ tiến hợp lệ (clip theo `speed_limit`) |
| `w_lane_offset` | `1.0` | Phạt `\|lane_offset_m\|` |
| `w_heading` | `0.5` | Phạt `\|heading_error_rad\|` |
| `w_steer_delta` | `1.0` | Phạt thay đổi steer đột ngột (Δsteer²) |
| `w_long_delta` | `0.5` | Phạt thay đổi ga/phanh đột ngột |
| `w_yaw_rate` | `0.1` | Phạt xoay yaw nhanh |
| `off_lane_penalty` | `5.0` | Phạt cố định mỗi bước còn lệch làn |
| `collision_penalty` | `50.0` | Phạt khi va chạm |
| `lane_invasion_penalty` | `1.0` | Phạt mỗi lần vượt vạch mới |

#### Nhóm thuật toán (`ppo` hoặc `sac`) — dùng chung

| Tham số | PPO mặc định | SAC mặc định | Ý nghĩa |
|---|---|---|---|
| `il_checkpoint` | `../behavior_cloning/best_il_model.pth` | như PPO | Checkpoint IL để warm-start actor |
| `output` | `./runs/ppo_lane_keep` | `./runs/sac_lane_keep` | Thư mục lưu checkpoint + CSV log |
| `warm_start` | `true` | `true` | Warm-start actor từ IL; `false` = khởi tạo ngẫu nhiên (chỉ để đối chứng) |
| `device` | `cuda` | `cuda` | `cuda` hoặc `cpu` — xem [mục 11](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb) |
| `gamma` | `0.99` | `0.99` | Hệ số chiết khấu |
| `batch_size` | `64` | `32` | Kích thước minibatch update — ăn trực tiếp VRAM/RAM |
| `total_steps` | `2,000,000` | `500,000` | Tổng số bước môi trường |
| `max_grad_norm` | `0.5` | `0.5` | Gradient clipping |

#### Riêng PPO (on-policy)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `learning_rate` | `3e-4` | LR chung cho actor + critic |
| `gae_lambda` | `0.95` | Hệ số GAE(λ) |
| `clip_range` | `0.2` | Ngưỡng clip surrogate objective |
| `value_clip_range` | `0.2` | Ngưỡng clip value loss |
| `entropy_coef` | `0.0` | Hệ số thưởng entropy (khuyến khích exploration) |
| `value_coef` | `0.5` | Trọng số value loss trong tổng loss |
| `epochs` | `10` | Số epoch update trên mỗi rollout |
| `target_kl` | `0.02` | Early-stop update nếu approx-KL vượt ngưỡng này |
| `n_steps` | `2048` | Số bước môi trường mỗi rollout/update |
| `save_every_updates` | `5` | Lưu checkpoint mỗi N update |
| `eval_every_updates`, `eval_episodes` | `10`, `3` | (dành sẵn) tần suất & số episode eval xen kẽ |

#### Riêng SAC (off-policy)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `actor_lr`, `critic_lr`, `alpha_lr` | `3e-4` mỗi loại | LR riêng cho actor / twin-Q critic / temperature α |
| `tau` | `0.005` | Hệ số Polyak update target network |
| `target_entropy` | `null` | `null` = tự động = `-action_dim` (Haarnoja et al.) |
| `buffer_capacity` | `50000` | Số transition tối đa trong replay buffer — **ăn RAM trực tiếp**, xem [mục 11](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb) |
| `learning_starts` | `5000` | Số bước warmup trước khi bắt đầu update (dùng sample stochastic của actor đã warm-start, không random thuần vì đã có IL) |
| `train_freq` | `1` | Update sau mỗi N bước môi trường |
| `gradient_steps` | `1` | Số gradient step mỗi lần update |
| `save_every_steps` | `10000` | Lưu checkpoint mỗi N bước |
| `eval_every_steps`, `eval_episodes` | `20000`, `3` | (dành sẵn) tần suất & số episode eval xen kẽ |

### Override qua CLI (không cần sửa JSON)

`config.py` áp dụng đúng pattern `--config` JSON-overrides-defaults rồi CLI-overrides-JSON
(giống `data_collection/carla_collector/config.py`):

```powershell
python train_ppo.py --config ppo_config.json --host 127.0.0.1 --port 2000 ^
    --il-checkpoint ../behavior_cloning/best_il_model.pth --output ./runs/ppo_test ^
    --total-steps 100000 --n-steps 1024 --width 320 --height 256 ^
    --batch-size 32 --device cpu
```

Các cờ CLI có sẵn: `--config`, `--host`, `--port`, `--il-checkpoint`, `--output`, `--resume`,
`--no-warm-start`, `--total-steps`, `--n-steps` (PPO), `--buffer-capacity` (SAC), `--width`,
`--height`, `--batch-size`, `--device {cuda,cpu}`, `--episodes` (evaluate.py),
`--deterministic` (evaluate.py), `--algorithm {ppo,sac}` (evaluate.py).

---

## 7. Quy trình chạy huấn luyện

### Yêu cầu: 2–3 terminal (cửa sổ) riêng biệt

```
Cửa sổ 1: CARLA Server
Cửa sổ 2 (tuỳ chọn): script/notebook đổi map + thời tiết trước khi train
Cửa sổ 3: train_ppo.py hoặc train_sac.py
```

Khác với `data_collection/`, **không cần** cửa sổ chạy `automatic_control.py` — script train
tự spawn xe của chính nó.

---

### Bước 1 — Khởi động CARLA Server

```powershell
# Cửa sổ 1
cd C:\CARLA_0.9.10\WindowsNoEditor
CarlaUE4.exe -quality-level=Low
```

> [!NOTE]
> Giữ cửa sổ này mở suốt toàn bộ quá trình train. Nếu máy yếu VRAM, thêm `-RenderOffScreen`
> — xem [mục 11](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb).

---

### Bước 2 (tuỳ chọn) — Đổi map/thời tiết trước khi train

```powershell
# Cửa sổ 2
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(30); c.load_world('Town02')"
```

> [!CAUTION]
> Không gọi `client.load_world()` trong lúc `train_ppo.py`/`train_sac.py` đang chạy — lệnh
> này hủy toàn bộ world, actor và sensor hiện có, kể cả xe do script train vừa spawn.

---

### Bước 3 — Chạy training

```powershell
# Cửa sổ 3
cd C:\Users\dloc\Desktop\Do_An\Deep_RL_Carla\drl_training

# PPO
python train_ppo.py --config ppo_config.json

# hoặc SAC
python train_sac.py --config sac_config.json
```

Checkpoint IL mặc định trỏ tới `../behavior_cloning/best_il_model.pth` (xem
[mục 2](#2-cài-đặt-môi-trường)). Nhấn `Ctrl+C` để dừng an toàn — cả hai script bắt
`KeyboardInterrupt`, lưu checkpoint `*_interrupted.pt` trước khi thoát.

---

### Bước 4 — Kiểm thử lần đầu (smoke test) TRƯỚC KHI train dài

> [!IMPORTANT]
> Module này chưa được chạy với CARLA thật lúc viết code (xem cảnh báo trong
> `drl_training/README.md`). **Luôn** smoke-test quy mô nhỏ trước khi chạy train hàng giờ.

**PPO:**
```powershell
python train_ppo.py --config ppo_config.json --total-steps 4096 --n-steps 512
```
Kỳ vọng: log `update=0 step=512 policy_loss=... value_loss=... kl=... mean_ep_reward=...`
sau khi xe chạy được ~512 bước, không có exception.

**SAC:** sửa tạm `learning_starts`/`total_steps` nhỏ trong `sac_config.json` (hoặc tạo bản
copy `sac_smoke_config.json`), ví dụ `learning_starts=200, total_steps=1000`, rồi:
```powershell
python train_sac.py --config sac_config.json
```
Kỳ vọng: sau bước 200, `update_log.csv` bắt đầu có `critic_loss`/`actor_loss` **không phải
NaN**.

Nếu bị treo ở bước tick đầu tiên (`_get_seg_frame` timeout) — kiểm tra CARLA server còn sống
và không có client passive nào khác (data_collection, automatic_control.py) đang giữ
`synchronous_mode`.

---

## 8. Theo dõi quá trình train

Cả hai script ghi 2 file CSV vào thư mục `output` (mặc định `runs/ppo_lane_keep/` hoặc
`runs/sac_lane_keep/`) — cùng shape cho cả 2 thuật toán để dễ dùng chung 1 script vẽ biểu đồ.

### `episode_log.csv` — mỗi dòng = 1 episode kết thúc

| Cột (PPO) | Cột (SAC) | Ý nghĩa |
|---|---|---|
| `update` | — | Chỉ số update PPO đang chạy khi episode kết thúc |
| — | `step` | Global step khi episode kết thúc |
| `global_step` | (= `step`) | Tổng số bước môi trường đã chạy |
| `episode_reward` | `episode_reward` | Tổng reward tích lũy của episode |
| `episode_len` | `episode_len` | Số bước của episode |
| `terminate_reason` | `terminate_reason` | `collision`, `off_lane`, hoặc `time_limit` |

### `update_log.csv` — mỗi dòng = 1 lần update mạng

**PPO** (mỗi dòng = 1 rollout/update, mặc định mỗi 2048 bước):

| Cột | Ý nghĩa |
|---|---|
| `update`, `global_step` | Định danh |
| `policy_loss`, `value_loss`, `entropy` | Các thành phần loss PPO |
| `approx_kl` | KL xấp xỉ giữa policy cũ/mới — theo dõi so với `target_kl=0.02` |
| `clip_fraction` | Tỉ lệ sample bị clip surrogate objective |
| `steps_per_sec` | Thông lượng (điểm nghẽn thường là CARLA tick, không phải gradient step) |
| `mean_episode_reward` | Trung bình reward 20 episode gần nhất |

**SAC** (mỗi dòng = log định kỳ mỗi 1000 bước):

| Cột | Ý nghĩa |
|---|---|
| `step` | Global step |
| `critic_loss`, `actor_loss`, `alpha_loss` | Các thành phần loss SAC |
| `alpha` | Temperature hiện tại (auto-tuned) |
| `mean_q` | Giá trị Q trung bình — theo dõi để phát hiện overestimation |
| `entropy` | Entropy chính sách hiện tại |
| `steps_per_sec`, `mean_episode_reward` | Như PPO |

> [!TIP]
> Dấu hiệu cần dừng và tinh chỉnh: `mean_episode_reward` không tăng sau nhiều update/step,
> `critic_loss`/`policy_loss` là `NaN`, hoặc `terminate_reason` gần như luôn là `collision`.
> Xem gợi ý tinh chỉnh reward ở [mục 5](#5-observation--action--reward--done).

### Checkpoint

| File | Khi nào tạo | Ghi chú |
|---|---|---|
| `ppo_update_NNNNNN.pt` / `sac_step_NNNNNNNN.pt` | Mỗi `save_every_updates`/`save_every_steps` | Checkpoint có đánh số, giữ lại lịch sử |
| `ppo_latest.pt` / `sac_latest.pt` | Cùng lúc với checkpoint đánh số | Dùng cho `--resume` hoặc `evaluate.py` |
| `ppo_interrupted.pt` / `sac_interrupted.pt` | Khi `Ctrl+C` | Checkpoint khẩn cấp lúc dừng giữa chừng |

---

## 9. Resume training

```powershell
python train_ppo.py --config ppo_config.json --resume runs/ppo_lane_keep/ppo_latest.pt
python train_sac.py --config sac_config.json --resume runs/sac_lane_keep/sac_latest.pt
```

PPO resume theo `update`, SAC resume theo `step` (global_step) — cả hai đọc lại từ trường
tương ứng đã lưu trong checkpoint.

---

## 10. Đánh giá checkpoint

Dùng chung một script (`evaluate.py`) cho cả hai thuật toán — chỉ cần
`agent.select_action(seg, scalar, deterministic)`, giao diện giống nhau ở cả `PPOAgent` và
`SACAgent`:

```powershell
python evaluate.py --algorithm ppo --config ppo_config.json ^
    --resume runs/ppo_lane_keep/ppo_latest.pt --episodes 10 --deterministic

python evaluate.py --algorithm sac --config sac_config.json ^
    --resume runs/sac_lane_keep/sac_latest.pt --episodes 10 --deterministic
```

Output mẫu:

```
episode=0 reward=123.45 len=850 collided=False mean|lane_offset|=0.182m reason=time_limit
episode=1 reward=98.20  len=612 collided=True  mean|lane_offset|=0.301m reason=collision
...
=== Tong ket 10 episode (PPO) ===
Reward trung binh: 110.32 +/- 15.67
Ty le va cham: 20.0%
Lech lan trung binh (|m|): 0.221
```

Số liệu này (reward trung bình, tỉ lệ va chạm, `mean|lane_offset|`) dùng để so sánh trực tiếp
với MAE của bước IL (mục 11 trong notebook IL) **và giữa PPO với SAC** — đưa vào báo cáo đồ
án. `--deterministic` dùng mean action (tắt exploration) — luôn bật cờ này khi đánh giá cuối
cùng để so sánh công bằng.

> [!NOTE]
> `evaluate.py` tự đọc trường `algorithm` đã lưu trong checkpoint và cảnh báo nếu bạn truyền
> `--algorithm` không khớp — checkpoint PPO không nạp được vào `SACAgent` và ngược lại.

---

## 11. Chạy trên máy yếu (RAM 16GB / VRAM 4GB)

Có 2 nguồn tranh chấp tài nguyên khác nhau, cần xử lý riêng.

### VRAM — giới hạn chặt nhất

Bản thân CARLA (UE4) đã khuyến nghị ≥6GB VRAM; 4GB là mức biên ngay cả khi không có gì khác
dùng GPU. Nếu để PyTorch cũng chiếm VRAM cùng lúc (mặc định `device=cuda`), rất dễ OOM khi
CARLA cần cấp phát thêm lúc đổi cảnh/thời tiết.

**Khuyến nghị mạnh: `--device cpu`.** `PolicyBackbone` rất nhỏ (5 lớp conv, 24–64 kênh) và
điểm nghẽn tốc độ luôn là bước tick CARLA (vật lý + render), không phải vài phép nhân ma trận
của mạng này — train trên CPU hầu như không làm chậm thông lượng tổng thể, trong khi giải
phóng toàn bộ VRAM cho một mình CARLA:

```powershell
CarlaUE4.exe -quality-level=Low -RenderOffScreen
python train_ppo.py --config ppo_config.json --device cpu
python train_sac.py --config sac_config.json --device cpu --buffer-capacity 15000
```

**Trước khi quyết định `cpu` hay `cuda`**: bật CARLA một mình (`-quality-level=Low
-RenderOffScreen`), đợi ~30s cho map load xong, rồi xem VRAM còn trống:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Trên GTX 1650 (4GB), CARLA ở `Low` thường chiếm khoảng 1.5–3GB tuỳ map:
- Còn dư rõ rệt (≥1GB) → có thể thử `--device cuda --batch-size 16` cho actor/critic, tăng
  tốc gradient step.
- Còn dưới ~500MB → giữ `--device cpu` để tránh OOM giữa chừng (đặc biệt lúc đổi thời
  tiết/spawn xe — VRAM CARLA cần có thể tăng đột ngột).

Theo dõi `nvidia-smi -l 2` (lặp mỗi 2s) trong lúc train để phát hiện sớm nếu VRAM tiến sát 4GB.

### `batch_size` — ăn VRAM/RAM trực tiếp, độc lập với độ phân giải camera

Ảnh one-hot 4 lớp ở 240×192 nặng hơn 160×128 khoảng 2,25 lần. Mặc định trong
`ppo_config.json`/`sac_config.json` đã hạ xuống 64/32 (SAC thấp hơn vì có 2 mạng Q chạy song
song). Nếu vẫn OOM (kể cả khi đã `--device cpu`, lúc đó là RAM chứ không phải VRAM), hạ tiếp:

```powershell
python train_ppo.py --config ppo_config.json --batch-size 16
```

### RAM 16GB — chủ yếu ảnh hưởng SAC

`sac/replay_buffer.py`: RAM ≈ `buffer_capacity × obs_width × obs_height` byte (uint8, 1 kênh,
không one-hot — one-hot hoá diễn ra trong `PolicyBackbone.forward` ngay trước conv, chỉ tốn
RAM host). Mặc định `obs_width=240, obs_height=192` + `buffer_capacity=50000` ≈ 2.2GB, vừa
với máy 16GB RAM. Muốn nhẹ hơn nữa? Hạ tạm:

```powershell
python train_sac.py --config sac_config.json --buffer-capacity 15000   # ≈ 0.7GB
```

`train_sac.py` tự in ước lượng GB thực tế lúc khởi động, dựa trên `obs_width`/`obs_height`/
`buffer_capacity` hiện tại — kiểm tra dòng in này trước khi để train chạy lâu.

PPO gần như không đáng lo về RAM vì rollout buffer chỉ giữ `n_steps=2048` frame (~90MB ở độ
phân giải observation mặc định).

### Kết luận cho máy yếu

Cấu hình 16GB RAM / 4GB VRAM **đủ để smoke-test và debug** toàn bộ vòng lặp trước khi thuê
GPU — **không đủ** để chạy train dài hàng trăm nghìn bước với tốc độ tốt, vì khi đó điểm
nghẽn là tốc độ tick/render của CARLA (được cải thiện đáng kể bởi GPU khoẻ hơn), không phải
mạng nơ-ron.

---

## 12. Tham số tham khảo

### Vì sao camera 480×384 nhưng observation 240×192

`CarlaLaneKeepEnv` tự spawn camera segmentation **sống** trong lúc train — **không** đọc lại
dữ liệu đã thu ở `data_collection/`, nên đổi độ phân giải DRL không cần thu thập lại dữ liệu
hay train lại segmentation/IL. Ở đây có **hai** con số:

| Tham số | Giá trị | Là gì |
|---|---|---|
| `width` / `height` | **480 × 384** | camera render ra, khớp `collector_config.json` |
| `obs_width` / `obs_height` | **240 × 192** | đưa vào mạng, sau khi `resize_class_map()` hạ mẫu |

**Observation phải là 240×192** vì đó đúng là `IMAGE_WIDTH`/`IMAGE_HEIGHT` mà `SteeringNet`
đã học trong `train_il_v9.ipynb`. `PolicyBackbone` dùng `AdaptiveAvgPool2d((1,1))` nên mọi kích
thước đều chạy và **không lỗi gì** — đó là cái bẫy: actor warm-start ở 480×384 vẫn chạy nhưng
nhìn cấu trúc lớn gấp 2× so với lúc học, làm hỏng warm-start trong im lặng.

**Camera vẫn 480×384** vì CARLA render **trực tiếp** ở độ phân giải cấu hình chứ không
downsample từ ảnh lớn: render thẳng 240×192 thì vạch kẻ làn (1–3 px) mất ngay lúc rasterize.
Render 480×384 rồi hạ mẫu bằng `resize_class_map()` thì vạch được bảo tồn theo **độ phủ diện
tích**, giống hệt `downscale_labels()` của notebook IL (đã kiểm chứng: hai hàm cho kết quả
byte-identical). Đo trên mask phối cảnh, nearest thuần chỉ giữ **69%** số pixel `RoadLine` so
với cách này.

Reward/điều kiện done không phụ thuộc độ phân giải (lấy thẳng từ API waypoint của CARLA).

> [!TIP]
> Muốn smoke-test nhanh trên máy yếu: `--obs-width 160 --obs-height 128`. Nhớ rằng lúc đó
> observation không còn khớp IL nữa, nên chỉ dùng để kiểm tra đường ống chạy được, không dùng
> để đo hiệu năng chính sách.

### Semantic class ID (dùng chung với `data_collection`/`behavior_cloning`)

| Class ID | Nhãn | Class ID | Nhãn |
|---|---|---|---|
| 0 | Unlabeled | 7 | Road |
| 1 | Building | 8 | SideWalk |
| 2 | Fence | 9 | Vegetation |
| 3 | Other | 10 | Vehicles |
| 4 | Pedestrian | 11 | Wall |
| 5 | Pole | 12 | TrafficSign |
| 6 | RoadLine | | |

Chi tiết màu RGB xem `docs/manual_thu_thap_du_lieu.md` §6.

---

## 13. Xử lý sự cố thường gặp

### Không kết nối được CARLA / treo ở bước tick đầu tiên

**Triệu chứng:** treo ở `_get_seg_frame` timeout, hoặc lỗi kết nối ngay khi khởi động.

**Giải pháp:**
1. Kiểm tra CARLA server (`CarlaUE4.exe`) đang chạy và đúng `host`/`port`.
2. Kiểm tra **không có client passive nào khác** (`data_collection/`, `automatic_control.py`)
   đang giữ `synchronous_mode` trên cùng world — chỉ được một active client mỗi lúc.
3. Tăng `frame_timeout`/`timeout` trong config nếu máy chậm khởi động map.

---

### Lỗi shape khi warm-start actor từ checkpoint IL

**Nguyên nhân:** checkpoint IL cũ (trước khi sửa observation contract) hoặc checkpoint IL bị
train với tập đặc trưng khác.

**Giải pháp:** train lại IL bằng notebook hiện tại trong `../behavior_cloning/`, tải checkpoint
mới về, trỏ `il_checkpoint` (config hoặc `--il-checkpoint`) tới file mới. Không sửa
`policy/observation.py` bằng tay — nó đọc hợp đồng trực tiếp từ checkpoint.

---

### `mean_episode_reward` không tăng / policy không học được gì

**Giải pháp theo thứ tự ưu tiên:**
1. Xác nhận warm-start đã chạy đúng (log `"Da warm-start actor tu: ..."` lúc khởi động, không
   phải `"[!] Bo qua warm-start"`).
2. Kiểm tra `terminate_reason` trong `episode_log.csv` — nếu gần như luôn `collision` ở đầu
   train, thử giảm `max_episode_steps` tạm thời để debug nhanh hơn, hoặc kiểm tra checkpoint
   IL có thực sự lái được trước (đánh giá riêng ở notebook IL).
3. Xem lại trọng số reward — [mục 5](#5-observation--action--reward--done).
4. Với PPO: theo dõi `approx_kl` — nếu liên tục sát/gần `target_kl=0.02` mỗi update, learning
   rate có thể đang quá cao.
5. Với SAC: theo dõi `alpha` — nếu tăng không kiểm soát, `mean_q` bất thường (rất lớn hoặc
   âm sâu) là dấu hiệu overestimation, cân nhắc giảm `critic_lr`.

---

### `critic_loss`/`policy_loss` là `NaN`

**Nguyên nhân thường gặp:** learning rate quá cao, reward scale quá lớn (không được
clip/chuẩn hoá), hoặc `batch_size` quá nhỏ gây gradient nhiễu mạnh.

**Giải pháp:**
1. Dừng ngay (`Ctrl+C`) — checkpoint từ điểm `NaN` không dùng được để resume tiếp tục học có
   ích.
2. Giảm `learning_rate`/`actor_lr`/`critic_lr` xuống một nửa, thử lại từ checkpoint trước đó
   (`ppo_update_*.pt`/`sac_step_*.pt` đánh số, không phải `*_interrupted.pt` nếu nó chính là
   điểm NaN).
3. Kiểm tra trọng số reward không quá lớn (vd `collision_penalty=50.0` là hợp lý, không nên
   để hàng nghìn).

---

### OOM (CUDA hoặc RAM)

Xem đầy đủ [mục 11](#11-chạy-trên-máy-yếu-ram-16gb--vram-4gb) — tóm tắt nhanh:

```powershell
# OOM VRAM (CARLA + PyTorch tranh chấp)
python train_ppo.py --config ppo_config.json --device cpu

# Vẫn OOM (RAM, không phải VRAM)
python train_ppo.py --config ppo_config.json --device cpu --batch-size 16

# SAC — giảm replay buffer trước tiên
python train_sac.py --config sac_config.json --buffer-capacity 15000
```

---

### `evaluate.py` báo lỗi thuật toán không khớp checkpoint

**Triệu chứng:**
```
[!] --algorithm=ppo nhung checkpoint duoc luu boi thuat toan 'sac' — dung 'sac'.
```

**Giải pháp:** đây là cảnh báo tự phục hồi — `evaluate.py` tự đổi sang đúng thuật toán và
nạp lại config tương ứng. Nếu muốn tránh cảnh báo, truyền đúng `--algorithm` khớp với
checkpoint (`ppo_latest.pt` → `--algorithm ppo`, `sac_latest.pt` → `--algorithm sac`).

---

## Tóm tắt quy trình nhanh

```
[1] Đảm bảo đã có checkpoint IL: ../behavior_cloning/best_il_model.pth
        ↓
[2] Khởi động CARLA Server (thêm -RenderOffScreen nếu máy yếu VRAM)
        ↓
[3] (Tuỳ chọn) Đổi map/thời tiết qua script nhỏ
        ↓
[4] Smoke test: python train_ppo.py --config ppo_config.json --total-steps 4096 --n-steps 512
        (hoặc SAC với learning_starts/total_steps nhỏ tạm thời)
        ↓
[5] Kiểm tra episode_log.csv/update_log.csv không có NaN, xe chạy được
        ↓
[6] Chạy train dài: python train_ppo.py --config ppo_config.json
        (hoặc python train_sac.py --config sac_config.json)
        ↓
[7] Theo dõi episode_log.csv/update_log.csv, tinh chỉnh reward nếu cần → quay lại [6]
        ↓
[8] Đánh giá: python evaluate.py --algorithm ppo --config ppo_config.json \
        --resume runs/ppo_lane_keep/ppo_latest.pt --episodes 10 --deterministic
        ↓
[9] So sánh PPO vs SAC vs MAE của bước IL → đưa vào báo cáo đồ án
```

---

*Tài liệu được tạo tự động từ mã nguồn tại `drl_training/` — 2026-08-13.*
