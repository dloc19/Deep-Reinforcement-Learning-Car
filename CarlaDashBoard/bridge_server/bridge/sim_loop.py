"""The one thread that ever calls into the CARLA client: ticks the world, runs the active
ModeRuntime, and applies every state-mutating command (§13 risk: "CARLA synchronous mode chỉ
chịu một client chủ động"). Everything from /control lands in `self.commands` (thread-safe
queue.Queue) and is drained here — since only this thread ever starts/stops a mode, a
SetMode can never race with another SetMode; the queue's FIFO order already gives us the
"transaction" semantics the design doc asks for, no extra lock needed.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path

import carla

from . import il_drl_bridge, map_graph, modes, protocol, route_planning, telemetry
from .carla_session import CarlaSession
from .modes.astar_autopilot import AstarAutopilotMode, RouteContext
from .modes.route_learned_autopilot import RouteLearnedAutopilotMode
from .modes.learned_autopilot import LearnedAutopilotMode

logger = logging.getLogger("bridge.sim_loop")


class SimLoop:
    def __init__(self, cfg, hub):
        self.cfg = cfg
        self.hub = hub
        self.session = CarlaSession(cfg, hub)
        self.commands = queue.Queue()
        self.current_mode = None
        self._stop = threading.Event()
        self._thread = None

        # --- Phase 3: route planning / map graph (see route_planning.py, map_graph.py) ---
        self._router_plan = None                 # lazy-loaded router_plan namespace
        self._planner_cache = {}                 # town short name -> GlobalRoutePlanner
        self.map_graph_json_cache = {}            # town short name -> pre-encoded JSON bytes
        self.route_context = None                 # RouteContext from the last SetDestination
        self._route_completed_notified = False
        self._route_stalled_notified = False
        self._route_progress_mark = None
        self._route_progress_since = 0.0

        # Doc san tren SIM THREAD de server.py._server_info_payload() (chay tren luong
        # asyncio) khong phai goi vao API CARLA — xem chu thich o ham do.
        self.spawn_points_cache = []              # list[dict] da san sang serialize
        self.current_town_cache = ""

        # --- Phat hien CarlaUE4 chet/khoi dong lai (xem _handle_tick_failure) ---
        self._tick_failures = 0
        self._carla_lost_notified = False
        self._next_reconnect_at = 0.0

        # --- Phase 4: IL / DRL Autopilot (see il_drl_bridge.py, modes/learned_autopilot.py) ---
        self._drl_training = None                 # lazy-loaded drl_training namespace
        self._il_predictor_cache = None            # (checkpoint_path, contract, predict)
        self._drl_predictor_cache = None            # (cache_key, contract, predict)

    def start(self):
        self.session.connect()
        self._refresh_spawn_points()
        self._ensure_town_graph()  # build once up front so /maps/{town} works immediately
        self._thread = threading.Thread(target=self._run, name="sim-loop", daemon=True)
        self._thread.start()

    def _refresh_spawn_points(self):
        """Doc lai danh sach spawn point + ten town cua ban do dang load. Chi duoc goi tu
        luong dieu khien CARLA (luc start(), va tu _cmd_SetTown tren sim thread)."""
        self.current_town_cache = self.session.current_town_short()
        spawn_points = self.session.map.get_spawn_points() if self.session.map else []
        self.spawn_points_cache = [
            {"index": i, "x": round(t.location.x, 1), "y": round(t.location.y, 1),
             "yaw": round(t.rotation.yaw, 1)}
            for i, t in enumerate(spawn_points)
        ]

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        self._stop_current_mode()
        self.session.shutdown()

    def _stop_current_mode(self):
        """Dung mode dang chay, KHONG bao gio de ngoai le tu no thoat ra ngoai.

        `ModeRuntime.stop()` co hop dong la "must not raise", nhung no goi vao API CARLA
        nen loi van co the tu duoi bay len — do that: `set_autopilot(False)` cua Data
        Collection nem `IndexError: invalid unordered_map<K, T> key` khi Traffic Manager
        khong con giu dang ky chiec xe. Ngoai le do lam ca `SetMode` that bai SAU KHI mode
        moi da start() xong, tuc bo roi mode moi va giu lai mot mode cu da huy het sensor.
        Don mode cu la viec "don dep", khong duoc phep quyet dinh so phan cua lenh dang chay.
        """
        if self.current_mode is None:
            return
        try:
            self.current_mode.stop()
        except Exception:                                          # noqa: BLE001
            logger.exception("Mode %s stop() raised — bo qua de tiep tuc", self.current_mode.name)
        self.current_mode = None

    def submit(self, command: dict):
        self.commands.put(command)

    # ------------------------------------------------------------------ main loop
    def _run(self):
        logger.info("Sim loop running (target %.0f Hz)", self.cfg.sim_fps)
        # Moc cho lan phat telemetry KE TIEP, cong don theo boi so cua chu ky. Ban truoc so
        # `now - last_publish >= publish_interval`, ma vong lap nay chi kiem tra moi tick
        # (20 Hz): voi publish_fps=15 (chu ky 0.0667s) thi tick o 0.05s luon "chua den han",
        # nen thuc te chi phat duoc moi tick thu hai — 10 Hz, do duoc that trong bo test
        # end-to-end. Cong don giu dung pha va cho ra dung 3 khung moi 4 tick.
        next_publish = 0.0
        publish_interval = 1.0 / self.cfg.publish_fps
        # GHIM THEO THOI GIAN THUC. `fixed_delta_seconds` chi noi mot tick dai bao nhieu
        # trong the gioi mo phong; no khong ep vong lap nay cho. Khong ghim thi world.tick()
        # chay het toc do may (~175 Hz do duoc tren may nay voi Town03 quality Low), tuc
        # mo phong chay nhanh gap ~9 lan thuc te: nguoi xem thay xe tua nhanh, va camera
        # segmentation ban ~175 khung/giay vao client. Ca hai deu khong phai thu ta muon
        # trong mot ban demo.
        tick_interval = 1.0 / self.cfg.sim_fps
        next_tick = time.time()
        while not self._stop.is_set():
            self._drain_commands()
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
            # Tut lai neu da tre nhieu hon mot tick (vd vua load town xong) — khong thi
            # vong lap se "duoi" cho kip bang mot loat tick lien tuc khong ghim.
            next_tick = max(next_tick + tick_interval, time.time() - tick_interval)
            if self._carla_lost_notified:
                # Da biet CARLA mat roi thi DUNG goi world.tick() nua: moi lan goi se treo
                # het `carla_tick_timeout_s` truoc khi bao hong, tuc vong lap tut xuong con
                # vai nhip mot phut va telemetry gan nhu dung han. Bo qua tick, chi thu noi
                # lai theo nhip rieng, de telemetry (connected=false) van chay deu cho
                # giao dien biet duong ma hien "mat ket noi".
                tick_ok = False
                self._attempt_reconnect()
            else:
                try:
                    self.session.world.tick()
                    tick_ok = True
                except RuntimeError as exc:
                    # KHONG `continue` o day. Ban truoc nhay thang sang vong sau, nen khi
                    # CarlaUE4 khoi dong lai (moi handle world/ego thanh rac) vong lap quay
                    # mai ma KHONG con phat telemetry nao: dashboard van bao "Da ket noi",
                    # camera dung hinh, va khong cho nao noi ra rang CARLA da mat. Bay gio
                    # van phat telemetry voi connected=false, va thu noi lai.
                    tick_ok = False
                    self._handle_tick_failure(exc)

            if tick_ok:
                self._tick_failures = 0
                snapshot = self.session.world.get_snapshot()

                if self.current_mode is not None and self.session.ego is not None:
                    try:
                        control = self.current_mode.tick(snapshot)
                        if control is not None:
                            self.session.ego.apply_control(control)
                    except Exception:
                        logger.exception("Mode %s tick() raised", self.current_mode.name)
                    self._check_route_completion()

            now = time.time()
            if now >= next_publish:
                next_publish = max(now, next_publish + publish_interval)
                try:
                    payload = telemetry.build(self.session, self.current_mode, connected=tick_ok)
                except RuntimeError:
                    # ego/world da chet giua chung — telemetry.build doc `ego.is_alive`.
                    payload = {"type": "telemetry", "t": now, "mode": "IDLE",
                               "connected": False, "ego_alive": False}
                self.hub.publish_stream_text(json.dumps(payload))

    # ------------------------------------------------------------------ mat ket noi CARLA
    _RECONNECT_INTERVAL_S = 3.0

    def _handle_tick_failure(self, exc):
        """Mot lan `world.tick()` hong: phan biet su co nhat thoi voi CarlaUE4 da chet."""
        self._tick_failures += 1
        # Hoi thang simulator bang mot RPC 2 giay thay vi dem so lan tick hong. Dem thi
        # phai cho `carla_timeout_s` (20s) MOI LAN hong, tuc vai chuc giay den vai phut moi
        # dam ket luan — trong khi cau hoi "CarlaUE4 con song khong?" tra loi duoc ngay.
        if self.session.simulator_reachable(2.0):
            logger.warning("world.tick() hong (%d) nhung simulator van tra loi: %s",
                           self._tick_failures, exc)
            time.sleep(0.5)
            return

        self._carla_lost_notified = True
        self._next_reconnect_at = time.time() + self._RECONNECT_INTERVAL_S
        logger.error("Mat ket noi toi CARLA (%s). Dang thu noi lai moi %.0fs — neu ban vua "
                     "khoi dong lai CarlaUE4 thi KHONG can tat Bridge Server.",
                     exc, self._RECONNECT_INTERVAL_S)
        # Mode dang chay bam vao mot chiec xe khong con ton tai — bo no truoc khi bao ra.
        self.current_mode = None
        self.route_context = None
        self.hub.publish_control_text(protocol.error(
            "CARLA_LOST", "Mất kết nối tới CARLA (CarlaUE4 có thể đã đóng hoặc đang khởi "
                          "động lại). Bridge Server sẽ tự nối lại, không cần tắt."))

    def _attempt_reconnect(self):
        """Thu noi lai CARLA theo nhip `_RECONNECT_INTERVAL_S`. Chi goi khi da biet mat."""
        now = time.time()
        if now < self._next_reconnect_at:
            return
        self._next_reconnect_at = now + self._RECONNECT_INTERVAL_S
        if not self.session.simulator_reachable(2.0):
            return                       # CarlaUE4 chua boot xong — im lang cho tiep
        try:
            self.session.reconnect()
        except Exception as exc:
            logger.warning("Noi lai CARLA that bai, se thu tiep: %s", exc)
            return

        town = self.session.current_town_short()
        logger.info("Da noi lai CARLA (map=%s). Phien cu da mat — bam \"Bat dau phien\" de "
                    "spawn xe moi.", town)
        self._tick_failures = 0
        self._carla_lost_notified = False
        # World moi = do thi/spawn point moi. Xoa cache theo town de khong phuc vu do thi
        # cua mot world da khong con.
        self._planner_cache.pop(town, None)
        self.map_graph_json_cache.pop(town, None)
        self._refresh_spawn_points()
        self._ensure_town_graph()
        self.hub.publish_control_text(protocol.dumps("CarlaReconnected", town=town))

    def _drain_commands(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(command)
            except Exception as exc:
                logger.exception("Command %s failed", command.get("type"))
                self.hub.publish_control_text(protocol.error("COMMAND_FAILED", str(exc)))

    # ------------------------------------------------------------------ commands
    def _handle_command(self, command):
        kind = command.get("type")
        handler = getattr(self, "_cmd_" + kind, None)
        if handler is None:
            self.hub.publish_control_text(
                protocol.error("UNKNOWN_COMMAND", "Không rõ lệnh: %s" % kind))
            return
        handler(command)

    def _cmd_StartSession(self, command):
        spawn_index = command.get("spawn_index", -1)
        # Xe MOI thi moi thu bam theo xe cu deu het hieu luc. Mode dang chay om mot
        # RouteTracker dung cho vi tri cu: sau khi respawn, tracker thay xe "da toi noi"
        # ngay tu tick dau (route_completed=1) va mode phanh cung 1.0 vinh vien — xe dung
        # im, khong loi nao hien ra o dau ca. Quan sat duoc that: brake=1.00, speed=0,
        # active_controller="arrived". Nen dung mode va bo tuyen cu TRUOC khi spawn.
        self._stop_current_mode()
        self.route_context = None
        self._reset_route_watch()
        self.session.spawn_ego(spawn_index)
        self.hub.publish_control_text(protocol.dumps("SessionStarted"))

    def _cmd_StopSession(self, command):
        self._stop_current_mode()
        self.session.destroy_cameras()
        self.session.destroy_ego()
        self.hub.publish_control_text(protocol.dumps("SessionStopped"))

    def _cmd_SetMode(self, command):
        mode_name = command.get("mode")
        if mode_name not in protocol.KNOWN_MODES:
            self.hub.publish_control_text(
                protocol.error("UNKNOWN_MODE", "Mode không hợp lệ: %s" % mode_name))
            return
        if self.session.ego is None:
            self.hub.publish_control_text(
                protocol.error("NO_SESSION", "Chưa có xe — bấm \"Bắt đầu phiên\" ở màn hình Settings trước."))
            return

        self.hub.publish_control_text(protocol.dumps("ModeChanging", mode=mode_name))
        try:
            if mode_name == protocol.MODE_ASTAR_AUTOPILOT:
                new_mode = AstarAutopilotMode(self.cfg, self.route_context, self._get_router_plan())
            elif mode_name == protocol.MODE_IL_AUTOPILOT:
                new_mode = self._build_il_mode()
            elif mode_name == protocol.MODE_DRL_AUTOPILOT:
                new_mode = self._build_drl_mode()
            elif mode_name == protocol.MODE_ROUTE_DRL_AUTOPILOT:
                new_mode = self._build_route_drl_mode()
            else:
                new_mode = modes.build(mode_name, self.cfg)
            new_mode.start(self.session)
        except NotImplementedError as exc:
            self.hub.publish_control_text(protocol.error("NOT_IMPLEMENTED", str(exc)))
            return
        except Exception as exc:
            logger.exception("Mode %s start() failed", mode_name)
            self.hub.publish_control_text(protocol.error("MODE_START_FAILED", str(exc)))
            return

        self._stop_current_mode()
        self.current_mode = new_mode
        self._reset_route_watch()
        self.hub.publish_control_text(protocol.dumps("ModeChanged", mode=mode_name))

    def _cmd_SetWeather(self, command):
        preset = command.get("preset")
        try:
            self.session.set_weather(preset)
            self.hub.publish_control_text(protocol.dumps("WeatherChanged", preset=preset))
        except ValueError as exc:
            self.hub.publish_control_text(protocol.error("BAD_WEATHER", str(exc)))

    def _cmd_SetTown(self, command):
        town = command.get("town")
        self._stop_current_mode()
        self.route_context = None  # route was planned on the old town's graph — no longer valid
        self._reset_route_watch()
        self.session.load_town(town)
        self._refresh_spawn_points()
        self._ensure_town_graph()
        self.hub.publish_control_text(protocol.dumps("TownChanged", town=town))

    def _cmd_SetDestination(self, command):
        if self.session.ego is None:
            self.hub.publish_control_text(
                protocol.error("NO_SESSION", "Chưa có xe — bấm \"Bắt đầu phiên\" ở màn hình Settings trước."))
            return
        town = self.session.current_town_short()
        planner = self._planner_cache.get(town)
        if planner is None:
            self.hub.publish_control_text(
                protocol.error("MAP_NOT_READY", "Đồ thị A* của bản đồ này chưa sẵn sàng, thử lại sau ít giây."))
            return

        try:
            goal_location = self._resolve_goal_location(command)
        except ValueError as exc:
            self.hub.publish_control_text(protocol.error("BAD_DESTINATION", str(exc)))
            return

        router_plan = self._get_router_plan()
        # Truyen CA HUONG xe, khong chi vi tri. Trong nga tu, phep chieu chi theo khoang
        # cach hay bat vao mot lan cat ngang, va tuyen se bat dau di huong khac han huong
        # xe dang chay: RouteTracker khong bao gio toi duoc node muc tieu nen tien do dung
        # yen o 0 m trong khi xe van chay — khong loi nao hien ra. Xem docstring cua
        # `GlobalRoutePlanner.snap_to_graph` (router_plan) de biet so lieu do duoc.
        ego_transform = self.session.ego.get_transform()
        try:
            route = planner.plan(ego_transform.location, goal_location,
                                 start_heading_deg=ego_transform.rotation.yaw)
        except router_plan.RouteNotFoundError as exc:
            self.hub.publish_control_text(protocol.error("ROUTE_NOT_FOUND", str(exc)))
            return

        tracker = planner.tracker_for(route, target_tolerance_m=self.cfg.astar_route_tolerance_m)
        polyline = [{"x": round(loc.x, 2), "y": round(loc.y, 2)}
                    for loc in (planner.graph.location_of(node_id) for node_id in route)]
        target_speed_mps = max(self.cfg.astar_target_speed_kmh / 3.6, 0.1)

        self.route_context = RouteContext(planner=planner, route=route)
        self._reset_route_watch()
        self.hub.publish_control_text(protocol.dumps(
            "RouteComputed",
            polyline=polyline,
            distance_m=round(tracker.total_m, 1),
            eta_s=round(tracker.total_m / target_speed_mps, 1),
            node_count=len(route),
        ))

    def _resolve_goal_location(self, command):
        if "spawn_index" in command:
            index = int(command["spawn_index"])
            spawn_points = self.session.map.get_spawn_points()
            if not 0 <= index < len(spawn_points):
                raise ValueError("spawn_index %d không hợp lệ (0..%d)" % (index, len(spawn_points) - 1))
            return spawn_points[index].location
        if "x" in command and "y" in command:
            return carla.Location(
                x=float(command["x"]), y=float(command["y"]), z=float(command.get("z", 0.0)))
        raise ValueError("Cần 'spawn_index' hoặc cả 'x' và 'y'.")

    def _get_router_plan(self):
        if self._router_plan is None:
            self._router_plan = route_planning.load(self.cfg.deep_rl_carla_root)
        return self._router_plan

    # ------------------------------------------------------------------ Phase 4: IL/DRL Autopilot
    def _get_drl_training(self):
        if self._drl_training is None:
            self._drl_training = il_drl_bridge.load(self.cfg.deep_rl_carla_root)
        return self._drl_training

    def _resolve_checkpoint_path(self, configured_path, default_relative):
        root = il_drl_bridge.resolve_root(self.cfg.deep_rl_carla_root)
        path = Path(configured_path) if configured_path else (root / default_relative)
        path = path.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(
                "Không tìm thấy checkpoint: %s (đặt --il-checkpoint-path / --drl-checkpoint-path "
                "nếu file nằm nơi khác, hoặc train model trước)." % path)
        return path

    def _build_il_mode(self):
        ns = self._get_drl_training()
        checkpoint_path = self._resolve_checkpoint_path(
            self.cfg.il_checkpoint_path, "behavior_cloning/best_il_model.pth")
        if self._il_predictor_cache is None or self._il_predictor_cache[0] != checkpoint_path:
            logger.info("Dang nap IL checkpoint: %s", checkpoint_path)
            contract, predict = il_drl_bridge.build_il_predictor(
                ns, checkpoint_path, self.cfg.learned_autopilot_device)
            self._il_predictor_cache = (checkpoint_path, contract, predict)
        _, contract, predict = self._il_predictor_cache
        return LearnedAutopilotMode(protocol.MODE_IL_AUTOPILOT, ns, contract, predict,
                                    action_repeat=self._learned_action_repeat(contract))

    def _drl_predictor(self):
        """(namespace, contract, predict, action_repeat) cho checkpoint DRL dang cau hinh.

        Tach rieng vi HAI mode dung chung no: DRL_AUTOPILOT (bam lan thuan) va
        ROUTE_DRL_AUTOPILOT (bam lan theo tuyen A*). Nap checkpoint hai lan cho hai mode se
        ton VRAM gap doi ma khong duoc gi — cache o `_drl_predictor_cache` ben duoi lo phan
        do, ham nay chi lo cho hai noi goi cung mot duong."""
        ns = self._get_drl_training()
        il_checkpoint_path = self._resolve_checkpoint_path(
            self.cfg.il_checkpoint_path, "behavior_cloning/best_il_model.pth")
        algo = self.cfg.drl_algorithm
        drl_checkpoint_path = self._resolve_checkpoint_path(
            self.cfg.drl_checkpoint_path, "drl_training/runs/best/%s_latest.pt" % algo)
        cache_key = (algo, il_checkpoint_path, drl_checkpoint_path)
        if self._drl_predictor_cache is None or self._drl_predictor_cache[0] != cache_key:
            logger.info("Dang nap DRL checkpoint (%s): %s", algo, drl_checkpoint_path)
            # In ky lai chinh checkpoint dang nap. Mot checkpoint SAI khong bao loi gi —
            # shape khop, xe van chay, chi la chay bang mot policy khac voi y dinh. Truong
            # `update` la chot chan re nhat: neu no nho hon `critic_warmup_updates` cua lan
            # train do thi actor con dang bi dong bang, tuc trong so van la IL nguyen ban.
            for line in il_drl_bridge.describe_checkpoint(drl_checkpoint_path):
                logger.info("  %s", line)
            contract, resolved_algo, predict = il_drl_bridge.build_drl_predictor(
                ns, algo, il_checkpoint_path, drl_checkpoint_path, self.cfg.learned_autopilot_device)
            if resolved_algo != algo:
                logger.info("Checkpoint tu khai bao thuat toan '%s' (khac --drl-algorithm='%s')",
                            resolved_algo, algo)
            self._drl_predictor_cache = (cache_key, contract, predict)
        _, contract, predict = self._drl_predictor_cache
        return ns, contract, predict, self._learned_action_repeat(contract)

    def _build_drl_mode(self):
        ns, contract, predict, action_repeat = self._drl_predictor()
        return LearnedAutopilotMode(protocol.MODE_DRL_AUTOPILOT, ns, contract, predict,
                                    action_repeat=action_repeat)

    def _build_route_drl_mode(self):
        """Lai theo tuyen A* + bam lan bang policy. Khong tu nap gi: dung lai predictor da
        cache va `_get_router_plan()` da co, chi dieu phoi giua chung."""
        ns, contract, predict, action_repeat = self._drl_predictor()
        return RouteLearnedAutopilotMode(
            protocol.MODE_ROUTE_DRL_AUTOPILOT, self.cfg, self.route_context,
            self._get_router_plan(), ns, contract, predict, action_repeat=action_repeat)

    def _learned_action_repeat(self, contract):
        """So world tick giu nguyen mot lenh de policy chay dung nhip `control_dt` cua
        checkpoint IL. Vd sim_fps=20 + control_dt=0.2 -> 4 tick moi quyet dinh."""
        control_dt = getattr(contract, "control_dt", None)
        if not control_dt:
            logger.warning("Checkpoint IL khong co 'control_dt' — chay policy moi tick "
                            "(%.0f Hz). `previous_steer/longitudinal` se lech y nghia so "
                            "voi luc train.", self.cfg.sim_fps)
            return 1
        repeat = max(1, int(round(float(control_dt) * self.cfg.sim_fps)))
        actual_hz = self.cfg.sim_fps / repeat
        logger.info("IL/DRL Autopilot: control_dt=%.2fs, sim_fps=%.1f -> action_repeat=%d "
                    "(policy chay o %.2f Hz)", control_dt, self.cfg.sim_fps, repeat, actual_hz)
        if abs(actual_hz - 1.0 / float(control_dt)) > 0.01:
            logger.warning("sim_fps=%.1f khong chia het cho nhip %.2f Hz cua checkpoint — "
                            "policy se chay o %.2f Hz. Dat --sim-fps la boi so cua %.2f.",
                            self.cfg.sim_fps, 1.0 / float(control_dt), actual_hz,
                            1.0 / float(control_dt))
        return repeat

    def _ensure_town_graph(self):
        town = self.session.current_town_short()
        if town in self._planner_cache:
            return
        try:
            router_plan = self._get_router_plan()
            logger.info("Building A* graph for %s (co the mat vai giay)...", town)
            t0 = time.time()
            planner = router_plan.GlobalRoutePlanner(
                self.session.map, self.cfg.astar_graph_resolution_m, self.cfg.astar_lane_change_cost)
            logger.info("Graph %s: %d node, xong sau %.1fs", town, len(planner.graph.nodes), time.time() - t0)
            self._planner_cache[town] = planner
            payload = map_graph.build_payload(town, planner.graph)
            self.map_graph_json_cache[town] = json.dumps(payload).encode("utf-8")
        except Exception:
            logger.exception("Khong build duoc A* graph cho %s — /maps/%s se tra 503 den khi thu lai.", town, town)

    # Xe chay nhanh hon nguong nay ma tien do tuyen khong nhuc nhich qua ngan ay giay thi
    # coi nhu da lac khoi tuyen. 15s la du dai de khong bao nham luc dung den do hay nhuong
    # duong (xe dung han thi toc do duoi nguong, dong ho khong chay).
    _ROUTE_STALL_SECONDS = 15.0
    _ROUTE_STALL_MIN_SPEED_KMH = 5.0

    def _reset_route_watch(self):
        self._route_completed_notified = False
        self._route_stalled_notified = False
        self._route_progress_mark = None
        self._route_progress_since = 0.0

    def _check_route_completion(self):
        """Theo doi tuyen dang chay: bao khi toi dich, va bao khi KET DINH.

        Phan "ket dinh" quan trong khong kem phan "toi dich": RouteTracker chi tien khi xe
        di gan node muc tieu, nen mot lan lac tuyen (policy bo lo khuc re, hoac tuyen tinh
        tu mot vi tri xe da di qua) lam tien do dung yen VINH VIEN trong khi xe van chay —
        khong ngoai le, khong log, dashboard chi hien mot con so khong doi. Do dung la thu
        lam nguoi xem demo tuong dashboard bi treo. Bao mot lan, ro rang, roi thoi.
        """
        route_state = getattr(self.current_mode, "route_state", None)
        if not route_state:
            return

        if route_state.get("route_completed"):
            if not self._route_completed_notified:
                self._route_completed_notified = True
                self.hub.publish_control_text(protocol.dumps("RouteCompleted", success=True))
            return

        progress = route_state.get("route_progress_m")
        if progress is None or self._route_stalled_notified:
            return
        now = time.time()
        if self._route_progress_mark is None or progress > self._route_progress_mark + 1.0:
            self._route_progress_mark = progress
            self._route_progress_since = now
            return

        velocity = self.session.ego.get_velocity()
        speed_kmh = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        if speed_kmh < self._ROUTE_STALL_MIN_SPEED_KMH:
            self._route_progress_since = now      # dung cho den do — khong tinh la ket dinh
            return
        if now - self._route_progress_since < self._ROUTE_STALL_SECONDS:
            return

        self._route_stalled_notified = True
        logger.warning("Tuyen ket dinh: tien do dung o %.1f m suot %.0fs trong khi xe chay "
                       "%.0f km/h — xe da lac khoi tuyen.", progress,
                       self._ROUTE_STALL_SECONDS, speed_kmh)
        self.hub.publish_control_text(protocol.error(
            "ROUTE_STALLED",
            "Xe đã lạc khỏi tuyến: tiến độ đứng yên ở %.0f m suốt %.0f giây trong khi xe vẫn "
            "chạy. Hãy chọn lại điểm đến (Route & Map) để tính tuyến mới từ vị trí hiện tại."
            % (progress, self._ROUTE_STALL_SECONDS)))

    def _cmd_SetCameraParams(self, command):
        self.session.apply_camera_params(
            width=command.get("width"), height=command.get("height"),
            fov=command.get("fov"), fps=command.get("fps"))
        self.hub.publish_control_text(protocol.dumps(
            "CameraParamsChanged",
            width=self.cfg.camera_width, height=self.cfg.camera_height,
            fov=self.cfg.camera_fov, fps=self.cfg.publish_fps))

    def _cmd_RecordStart(self, command):
        if getattr(self.current_mode, "start_recording", None) is None:
            self.hub.publish_control_text(
                protocol.error("NOT_RECORDABLE", "Mode hiện tại không hỗ trợ ghi dữ liệu."))
            return
        self.current_mode.start_recording()
        self.hub.publish_control_text(protocol.dumps("RecordingStarted"))

    def _cmd_RecordStop(self, command):
        if getattr(self.current_mode, "stop_recording", None) is None:
            return
        self.current_mode.stop_recording()
        self.hub.publish_control_text(protocol.dumps("RecordingStopped"))
