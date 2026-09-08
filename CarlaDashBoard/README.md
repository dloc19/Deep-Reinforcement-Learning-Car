# Carla Dashboard

Implementation of the "Carla Dashboard" design doc (published artifact — see project memory /
ask for the link) — a WPF dashboard for the CARLA graduation project at
`Do_An/Deep_RL_Carla`. Two independent pieces, talking WebSocket:

```
bridge_server/       Python (run with the carla_rl conda env) — see bridge_server/README.md
CarlaDashBoard.Wpf/     WPF .NET 8 client — see below
CarlaDashBoard.slnx
```

**Status: Phase 0–5 của roadmap đã dựng xong, và đã chạy end-to-end với CarlaUE4 thật.**
Bộ test `bridge_server/tools/e2e_test.py` chạy 25 case trên một phiên CARLA sống (Town03 +
Town02): 25/25 PASS, lặp lại hai lần. Xem "Đã kiểm chứng những gì" bên dưới.

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
6. To try IL/DRL Autopilot: train a checkpoint first (`Deep_RL_Carla/behavior_cloning/train_il.ipynb`
   for IL, then `Deep_RL_Carla/drl_training/train_ppo.py`/`train_sac.py` for DRL — see that
   repo's own docs), point the bridge server at it with `--il-checkpoint-path` /
   `--drl-checkpoint-path` (defaults assume the standard output paths next to
   `Deep_RL_Carla/`), then pick **IL Autopilot** / **DRL Autopilot** → **Áp dụng mode** in
   Settings, same as any other mode. No destination needed — both are lane-keeping only.

## Đã kiểm chứng những gì

Chạy `bridge_server/tools/e2e_test.py` với CarlaUE4 0.9.10 đang mở (Town03, rồi đổi sang
Town02 và quay lại) — **25/25 case PASS, chạy lại lần hai vẫn 25/25**, log server không có
traceback nào. Số đo lấy từ chính lần chạy đó:

- **Kênh dữ liệu**: RGB 15.2 fps (JPEG ~57 KB/khung), Seg 15.2 fps (PNG ~4 KB/khung),
  telemetry 15.0 Hz — đúng bằng `--publish-fps 15`.
- **`GET /maps/Town03`**: 7092 node / 11258 edge, 2.2 MB, dựng đồ thị hết 0.2 s. Town02:
  1505 node. Tên trường khớp `Models/MapGraph.cs`.
- **A\* Autopilot**: tuyến 170 m, xe tiến 158 m trong 20 s, tự phanh khi tới đích.
- **IL Autopilot**: |lệch làn| trung bình **0.03 m** (max 0.05), chạy ở 5 Hz (`action_repeat=4`
  suy từ `control_dt=0.2s` của checkpoint).
- **DRL Autopilot** (`runs/best/ppo_latest.pt`, update=195, reward_mode=normalized): |lệch làn|
  trung bình 0.15–0.17 m (max ~1.7) — kém IL rõ rệt, đúng như ghi nhận trước đó.
- **Route + DRL Autopilot**: đi 215 m trong 30 s trên tuyến 653 m, bàn giao qua lại
  policy ↔ planner, `RouteCompleted` bắn đúng lúc tới đích.
- **Data Collection**: Traffic Manager lái 17 km/h, ghi 123 dòng `states.csv` + 25 ảnh RGB
  (JPEG) + 25 ảnh Seg (PNG) trong 6 s.
- **Đổi Town giữa phiên**: Town03 → Town02 → Town03, mỗi lần đều spawn lại xe và lái tiếp được.
- **WPF app**: `dotnet build CarlaDashBoard.slnx` — 0 warning, 0 error trên `net8.0-windows`.
  Phía WPF chưa có test tự động (xem "Known gaps").

## Known gaps going into Phase 5 (polish)

- Checkpoint DRL đang dùng (`drl_training/runs/best/ppo_latest.pt`) bám làn **kém hơn** IL
  gốc (0.15 m so với 0.03 m, đo ở bảng trên). Muốn demo bám làn đẹp nhất thì chọn
  **IL Autopilot**; DRL để so sánh.
- Bộ test end-to-end (`bridge_server/tools/e2e_test.py`) phủ phía Bridge Server; phía WPF
  vẫn chưa có test tự động — vẫn phải mở app bấm tay trước khi demo.
- `ROUTE_DRL_AUTOPILOT` bàn giao vô-lăng cho pure-pursuit trước ngã tư ~24 m (12 node đồ
  thị). Trên bản đồ nhiều ngã tư sát nhau như Town03, tỷ lệ thời gian policy thật sự cầm lái
  vì thế thấp hơn trên đường trường — muốn quay video khoe policy thì chọn đoạn đường dài.
