# Carla Dashboard

Implementation of the "Carla Dashboard" design doc (published artifact — see project memory /
ask for the link) — a WPF dashboard for the CARLA graduation project at
`Do_An/Deep_RL_Carla`. Two independent pieces, talking WebSocket:

```
bridge_server/       Python (run with the carla_rl conda env) — see bridge_server/README.md
CarlaDashBoard.Wpf/     WPF .NET 8 client — see below
CarlaDashBoard.slnx
```

**Status: Phase 0–4 of the roadmap are built and each half compiles/runs on its own.** They
have **not** been tested end-to-end together against a live CARLA instance in this session —
that needs `CarlaUE4.exe` running, which wasn't started here. See "What's verified" below.

## Run it for real

1. Start CARLA (`CARLA_0.9.10/WindowsNoEditor/CarlaUE4.exe`), load a Town.
2. Start the bridge server — see `bridge_server/README.md` for the one-time `pip install`:
   ```
   C:\Users\dloc\miniconda3\envs\carla_rl\python.exe bridge_server\run_server.py --town Town03
   ```
3. Run the WPF app (Visual Studio, or):
   ```
   dotnet run --project CarlaDashBoard.Wpf
   ```
4. In the app's **Settings** tab: enter host/port (defaults `127.0.0.1:8765`) → **Kết nối** →
   **Bắt đầu phiên** → pick **Data Collection** → **Áp dụng mode**. Switch to **Live Drive** to
   see the camera stream and toggle RGB/Seg/Split.
5. To try A* Autopilot: go to **Route & Map**, wait for the graph to load, click a point on the
   road (or pick a spawn point / type x,y) → route appears with distance/ETA → **Bắt đầu lái**.
6. To try IL/DRL Autopilot: train a checkpoint first (`Deep_RL_Carla/behavior_cloning/train_il_v9.ipynb`
   for IL, then `Deep_RL_Carla/drl_training/train_ppo.py`/`train_sac.py` for DRL — see that
   repo's own docs), point the bridge server at it with `--il-checkpoint-path` /
   `--drl-checkpoint-path` (defaults assume the standard output paths next to
   `Deep_RL_Carla/`), then pick **IL Autopilot** / **DRL Autopilot** → **Áp dụng mode** in
   Settings, same as any other mode. No destination needed — both are lane-keeping only.

## What's verified in this session (no live CARLA available here)

- **Bridge Server**: every module imports cleanly under the `carla_rl` env's Python 3.7 with
  the real `carla` package present, **including** the `router_plan` bridge (Phase 3) and the
  new `drl_training` bridge (Phase 4, `il_drl_bridge.py`) — both resolve their real classes
  from the sibling `Deep_RL_Carla` repo folder (see `bridge_server/README.md`).
- **IL/DRL Autopilot (Phase 4)**: exercised end-to-end with a synthetic checkpoint (matching
  `train_il_v9.ipynb`'s exact key names/shapes) and fake CARLA state objects, offline —
  `build_il_predictor()`/`build_drl_predictor()` load the checkpoint, remap IL weights onto
  `GaussianActor` (`policy/il_compat.py`), and `LearnedAutopilotMode.tick()` runs the full
  observation → forward pass → `VehicleControl` path and returns a valid, in-range action.
  What this does **not** cover: a real trained checkpoint (none has been trained yet — no
  `.pth`/`.pt` file exists in the repo), and the actual driving behaviour against a live
  simulator — the missing-checkpoint path was only confirmed to fail *cleanly*
  (`MODE_START_FAILED` with the path it looked for), not the driving itself.
- **A\* Autopilot (Phase 3)**: not run against a live simulator, so `GlobalRoutePlanner`'s
  graph build, `SetDestination`, and the drive loop have not been exercised with real CARLA
  data.
- **WPF app**: `dotnet build`/`dotnet build CarlaDashBoard.slnx` both succeed with 0
  warnings/errors on `net8.0-windows` for all three screens, including the Settings screen
  now offering IL/DRL Autopilot as selectable (no longer "Phase sau"). Rebranded to
  **CarlaDashBoard** (project/namespace/folder rename) with a redesigned sidebar — app
  mark + vector window icon, labelled nav items with an active-screen highlight driven by
  `ShellViewModel.IsLiveDriveActive`/etc., and a persistent connection-status row (previously
  only a tiny dot, and missing entirely from Route & Map) — plus a highlighted RGB/Seg/Split
  toggle on Live Drive and a responsive (WrapPanel, not a fixed 4-column grid) mode-tile
  layout on Settings so it no longer overflows at the window's stated `MinWidth="900"`.
  Verified by actually launching the built exe and screenshotting all three screens after
  driving the nav rail via UI Automation (not just eyeballing the default screen) — sidebar
  active-state, camera-mode highlight, and mode-tile wrapping all confirmed visually, process
  stayed alive with no exceptions. Since no bridge server was running, the Settings → Connect
  flow, the camera stream, and every mode's actual drive loop were **not** exercised live.

## Known gaps going into Phase 5 (polish)

- No trained IL/DRL checkpoint exists yet in the repo — Phase 4's code path is verified with
  synthetic weights only (see above); train `best_il_model.pth` and a PPO/SAC checkpoint via
  `Deep_RL_Carla/behavior_cloning/` and `Deep_RL_Carla/drl_training/` before relying on IL/DRL
  Autopilot for a demo.
- No automated tests on either side — validate manually against a running CARLA session
  before relying on this for a demo, especially the A* Autopilot drive loop and the
  `/maps/{town}` payload size on a full Town (thousands of nodes — untested for real).
