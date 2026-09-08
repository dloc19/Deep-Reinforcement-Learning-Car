# Carla Dashboard — Bridge Server

Python WebSocket server that sits between CARLA and the WPF **Carla Dashboard** dashboard. Full
rationale and protocol spec: see the design doc artifact ("Carla Dashboard",
`§01`/`§02`/`§07`). This package currently implements **Phase 0–4** of the roadmap (`§12`):

| Phase | What's here |
|---|---|
| 0 | `websockets` server, `/stream` binary framing, one RGB camera published live |
| 1 | Segmentation camera channel, telemetry envelope, **Data Collection** mode (Traffic Manager autopilot + optional live CSV/JPEG recorder) |
| 2 | `/control` command router, `ServerInfo` on connect, Town/weather switching, mode state machine (with clean `NOT_IMPLEMENTED` stubs for A*/IL/DRL, which are Phase 3/4) |
| 3 | `GET /maps/{town}` REST endpoint (road graph, built once per town and cached — see `route_planning.py`/`map_graph.py`), `SetDestination` command, real **A* Autopilot** mode wrapping `router_plan`'s own `RoutePurePursuitController` |
| 4 | Real **IL Autopilot** and **DRL Autopilot** (PPO/SAC) modes — see `il_drl_bridge.py`/`modes/learned_autopilot.py` |

## Run it

1. Start CARLA and load a world (`CarlaUE4.exe` from `CARLA_0.9.10/WindowsNoEditor/`, quality
   `-quality-level=Low` is fine for a demo).
2. Install dependencies once, into the **carla_rl** conda env** (Python 3.7, already has
   `carla==0.9.10`):

   ```
   C:\Users\dloc\miniconda3\envs\carla_rl\python.exe -m pip install -r requirements.txt
   ```
3. Run the server:

   ```
   C:\Users\dloc\miniconda3\envs\carla_rl\python.exe run_server.py --town Town03
   ```

   It listens on `ws://0.0.0.0:8765` — `/stream` and `/control` (see `--help` for every flag:
   host/port, sim fps, publish fps, JPEG quality, vehicle blueprint...).
4. No vehicle exists yet at this point — a WPF client (or any WebSocket test client) must send
   `{"type":"StartSession"}` on `/control` first. Then `{"type":"SetMode","mode":"DATA_COLLECTION"}`
   hands the car to CARLA's Traffic Manager and starts publishing telemetry + camera frames.

## Bộ test end-to-end (`tools/e2e_test.py`)

Kiểm tra toàn bộ Bridge Server với **CarlaUE4 đang chạy thật** — không cần app WPF. Script tự
khởi động `run_server.py` ở cổng 8799 (không đụng server thật ở 8765), nối cả `/control` lẫn
`/stream`, chạy 25 test case rồi in bảng kết quả; exit code ≠ 0 nếu có case FAIL.

```
cd bridge_server
C:\Users\dloc\miniconda3\envs\carla_rl\python.exe tools\e2e_test.py          # đầy đủ (~4 phút)
C:\Users\dloc\miniconda3\envs\carla_rl\python.exe tools\e2e_test.py --quick  # bỏ phần đổi Town
```

Bao phủ: `ServerInfo`, `GET /maps/{town}` (kể cả 503 khi chưa sẵn sàng), mọi nhánh lỗi
(`BAD_JSON`/`BAD_COMMAND`/`UNKNOWN_COMMAND`/`UNKNOWN_MODE`/`NO_SESSION`/`BAD_WEATHER`/
`BAD_DESTINATION`/`NOT_RECORDABLE`), `StartSession`/`StopSession`, nhịp + định dạng của kênh
RGB (JPEG) và Seg (PNG), nhịp telemetry, `SetWeather`, `SetCameraParams`, cả **năm** mode lái
(có kiểm tra xe thực sự chạy và tiến trên tuyến), ghi dữ liệu ra CSV + ảnh, và `SetTown` qua
lại giữa hai bản đồ. Log đầy đủ của server nằm ở `tools/e2e_server.log` (script đếm luôn số
traceback trong đó).

Script tự trả CARLA về chế độ bất đồng bộ khi kết thúc, kể cả khi nó phải giết tiến trình
server con — nếu không, world sẽ nằm lại ở `synchronous_mode=True` mà không ai tick và cửa sổ
CarlaUE4 trông y như bị treo.

## Manual smoke test without the WPF client

`websockets` ships a CLI you can use from the same conda env while the server is running:

```
C:\Users\dloc\miniconda3\envs\carla_rl\python.exe -m websockets ws://127.0.0.1:8765/control
```

Then type e.g. `{"type":"StartSession"}` and press Enter — you should immediately get a
`ServerInfo` message on connect, and events as you send more commands.

## Phase 3 notes

- `router_plan` lives in the sibling `Deep_RL_Carla` repo folder, not inside `bridge_server` —
  `bridge/route_planning.py` puts it on `sys.path` (auto-detects `../../Deep_RL_Carla`
  relative to `bridge_server/`; override with `--deep-rl-carla-root` if you've moved things).
- The A* graph for the current Town is built **once**, right after connecting (and again on
  every `SetTown`) — that's the "vài giây" cost `router_plan/README.md` already documents, paid
  up front instead of on the first `SetDestination`. `GET /maps/{town}` just reads the cached,
  pre-serialized JSON; it never touches CARLA itself, so it can't block the sim loop.
- `SetDestination` computes the route immediately and replies with `RouteComputed`; it does
  **not** start driving. Driving only starts once `SetMode {mode:"ASTAR_AUTOPILOT"}` is sent
  (normally right after the WPF Route & Map screen's "Bắt đầu lái" button) — matching the
  wireframe flow in design doc Hình 03/Hình 06.

## Phase 4 notes — IL / DRL Autopilot

- `bridge/il_drl_bridge.py` is the sibling-repo bridge to `Deep_RL_Carla/drl_training` (same
  shape as `route_planning.py` for `router_plan`) plus the two live-inference factories:
  `build_il_predictor()` (an IL checkpoint from `behavior_cloning/train_il.ipynb`) and
  `build_drl_predictor()` (a PPO/SAC checkpoint from `drl_training/train_ppo.py` /
  `train_sac.py`, loaded the same way `drl_training/evaluate.py` does — including reading the
  observation contract off the **IL** checkpoint, since that's the only place
  `scalar_feature_dim`/`continuous_cols`/`norm_stats`/etc. are saved).
- Both modes are **lane-keeping only** — no destination, they just drive off the live camera +
  scalar state every tick (`bridge/modes/learned_autopilot.py::LearnedAutopilotMode`, shared by
  both, since only "how one (seg, scalar) observation turns into an action" differs).
- `CarlaSession` keeps the raw semantic-segmentation class-id map (`last_seg_class_map`) around
  each frame — that map is the policy's image input, and the Live Drive stream is drawn from
  the same map, so both consumers share the one sensor.
- The Live Drive **Seg** channel is drawn with this project's own **4-class** palette
  (`Background / Road / RoadLine / Sidewalk`, from `carla_collector/schema.py::SEG_COLOR_LUT`),
  not CARLA's 13+-class `CityScapesPalette` — the point of that panel is to show what the model
  actually sees. It is encoded as **PNG, not JPEG**: this is a class-index image, and JPEG's 8x8
  quantisation smears flat colour regions (measured on a Town03 frame: only 37% of pixels kept
  an exact palette colour, and `RoadLine` — 1-3 px wide — lost essentially all of its pixels).
  PNG is both lossless and smaller here (16.8 KB vs 26.1 KB). See `bridge/seg_palette.py`.
- Checkpoints are configured on the Bridge Server (CLI flags below), not from the WPF app —
  `--il-checkpoint-path` / `--drl-checkpoint-path` / `--drl-algorithm` / `--learned-autopilot-device`.
  Defaults, if not given: `{deep-rl-carla-root}/behavior_cloning/best_il_model.pth` and
  `{deep-rl-carla-root}/drl_training/runs/best/{algo}_latest.pt` — `runs/best/` is the stable
  pointer to the checkpoint chosen for deployment (with a `README.txt` saying which run it came
  from and why); see the comment on `drl_checkpoint_path` in `bridge/config.py` for why the
  default is no longer `runs/{algo}_lane_keep/`. A missing
  checkpoint fails the `SetMode` call with a clear `MODE_START_FAILED` error (path it looked
  for included) instead of hanging or crashing the sim loop.
- Loaded models are cached in `SimLoop` (keyed by resolved checkpoint path, same idea as the
  A* graph cache) so switching modes back and forth doesn't reload weights from disk each time.

## What's intentionally NOT here yet

- IL/DRL training itself (that's `behavior_cloning/train_il.ipynb` and
  `drl_training/train_ppo.py`/`train_sac.py`, both in the sibling `Deep_RL_Carla` repo) — this
  package only runs an already-trained checkpoint live.
- This is **not** a drop-in replacement for `data_collection/collect_data.py` — see the
  module docstring in `bridge/modes/data_collection.py` for why the two stay separate.

## Layout

```
bridge_server/
├── run_server.py           entrypoint (argparse + asyncio.run)
└── bridge/
    ├── config.py            BridgeConfig, TOWNS, WEATHER_PRESETS
    ├── protocol.py           message/channel constants + JSON envelope helpers
    ├── hub.py                thread-safe client registry + broadcast (sim thread -> asyncio)
    ├── carla_session.py      CARLA client/world/ego/camera lifecycle
    ├── cameras.py             camera blueprint + JPEG encode
    ├── telemetry.py           telemetry JSON builder
    ├── route_planning.py       sys.path bridge to Deep_RL_Carla/router_plan (Phase 3)
    ├── map_graph.py            RouteGraph -> JSON for GET /maps/{town} (Phase 3)
    ├── il_drl_bridge.py         sys.path bridge to Deep_RL_Carla/drl_training + IL/DRL predictor factories (Phase 4)
    ├── sim_loop.py            the ONE thread that ticks CARLA + drains /control commands
    ├── server.py              websockets routing for /stream, /control + /maps/{town} REST
    └── modes/
        ├── base.py            ModeRuntime lifecycle (start/tick/stop/status_extra)
        ├── data_collection.py  Phase 1 mode
        ├── astar_autopilot.py  Phase 3 mode (RouteTracker + RoutePurePursuitController)
        └── learned_autopilot.py Phase 4 mode (IL Autopilot + DRL Autopilot, shared tick loop)
```
