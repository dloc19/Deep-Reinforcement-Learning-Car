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

        # --- Phase 4: IL / DRL Autopilot (see il_drl_bridge.py, modes/learned_autopilot.py) ---
        self._drl_training = None                 # lazy-loaded drl_training namespace
        self._il_predictor_cache = None            # (checkpoint_path, contract, predict)
        self._drl_predictor_cache = None            # (cache_key, contract, predict)

    def start(self):
        self.session.connect()
        self._ensure_town_graph()  # build once up front so /maps/{town} works immediately
        self._thread = threading.Thread(target=self._run, name="sim-loop", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        if self.current_mode:
            self.current_mode.stop()
        self.session.shutdown()

    def submit(self, command: dict):
        self.commands.put(command)

    # ------------------------------------------------------------------ main loop
    def _run(self):
        logger.info("Sim loop running (target %.0f Hz)", self.cfg.sim_fps)
        last_publish = 0.0
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
            try:
                self.session.world.tick()
            except RuntimeError as exc:
                logger.warning("world.tick() failed: %s", exc)
                time.sleep(0.5)
                continue
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
            if now - last_publish >= publish_interval:
                payload = telemetry.build(self.session, self.current_mode)
                self.hub.publish_stream_text(json.dumps(payload))
                last_publish = now

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
                protocol.error("UNKNOWN_COMMAND", "Khong ro lenh: %s" % kind))
            return
        handler(command)

    def _cmd_StartSession(self, command):
        spawn_index = command.get("spawn_index", -1)
        self.session.spawn_ego(spawn_index)
        self.hub.publish_control_text(protocol.dumps("SessionStarted"))

    def _cmd_StopSession(self, command):
        if self.current_mode is not None:
            self.current_mode.stop()
            self.current_mode = None
        self.session.destroy_cameras()
        self.session.destroy_ego()
        self.hub.publish_control_text(protocol.dumps("SessionStopped"))

    def _cmd_SetMode(self, command):
        mode_name = command.get("mode")
        if mode_name not in protocol.KNOWN_MODES:
            self.hub.publish_control_text(
                protocol.error("UNKNOWN_MODE", "Mode khong hop le: %s" % mode_name))
            return
        if self.session.ego is None:
            self.hub.publish_control_text(
                protocol.error("NO_SESSION", "Chua co xe — goi StartSession truoc."))
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

        if self.current_mode is not None:
            self.current_mode.stop()
        self.current_mode = new_mode
        self._route_completed_notified = False
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
        if self.current_mode is not None:
            self.current_mode.stop()
            self.current_mode = None
        self.route_context = None  # route was planned on the old town's graph — no longer valid
        self.session.load_town(town)
        self._ensure_town_graph()
        self.hub.publish_control_text(protocol.dumps("TownChanged", town=town))

    def _cmd_SetDestination(self, command):
        if self.session.ego is None:
            self.hub.publish_control_text(
                protocol.error("NO_SESSION", "Chua co xe — goi StartSession truoc."))
            return
        town = self.session.current_town_short()
        planner = self._planner_cache.get(town)
        if planner is None:
            self.hub.publish_control_text(
                protocol.error("MAP_NOT_READY", "Do thi A* cua ban do nay chua san sang, thu lai sau it giay."))
            return

        try:
            goal_location = self._resolve_goal_location(command)
        except ValueError as exc:
            self.hub.publish_control_text(protocol.error("BAD_DESTINATION", str(exc)))
            return

        router_plan = self._get_router_plan()
        try:
            route = planner.plan(self.session.ego.get_location(), goal_location)
        except router_plan.RouteNotFoundError as exc:
            self.hub.publish_control_text(protocol.error("ROUTE_NOT_FOUND", str(exc)))
            return

        tracker = planner.tracker_for(route, target_tolerance_m=self.cfg.astar_route_tolerance_m)
        polyline = [{"x": round(loc.x, 2), "y": round(loc.y, 2)}
                    for loc in (planner.graph.location_of(node_id) for node_id in route)]
        target_speed_mps = max(self.cfg.astar_target_speed_kmh / 3.6, 0.1)

        self.route_context = RouteContext(planner=planner, route=route)
        self._route_completed_notified = False
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
                raise ValueError("spawn_index %d khong hop le (0..%d)" % (index, len(spawn_points) - 1))
            return spawn_points[index].location
        if "x" in command and "y" in command:
            return carla.Location(
                x=float(command["x"]), y=float(command["y"]), z=float(command.get("z", 0.0)))
        raise ValueError("Can 'spawn_index' hoac ca 'x' va 'y'.")

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
                "Khong tim thay checkpoint: %s (dat --il-checkpoint-path / --drl-checkpoint-path "
                "neu file nam noi khac, hoac train model truoc)." % path)
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
            self.cfg.drl_checkpoint_path, "drl_training/runs/%s_lane_keep/%s_latest.pt" % (algo, algo))
        cache_key = (algo, il_checkpoint_path, drl_checkpoint_path)
        if self._drl_predictor_cache is None or self._drl_predictor_cache[0] != cache_key:
            logger.info("Dang nap DRL checkpoint (%s): %s", algo, drl_checkpoint_path)
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

    def _check_route_completion(self):
        if not isinstance(self.current_mode, AstarAutopilotMode) or self._route_completed_notified:
            return
        if self.current_mode.route_state.get("route_completed"):
            self._route_completed_notified = True
            self.hub.publish_control_text(protocol.dumps("RouteCompleted", success=True))

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
                protocol.error("NOT_RECORDABLE", "Mode hien tai khong ho tro ghi du lieu."))
            return
        self.current_mode.start_recording()
        self.hub.publish_control_text(protocol.dumps("RecordingStarted"))

    def _cmd_RecordStop(self, command):
        if getattr(self.current_mode, "stop_recording", None) is None:
            return
        self.current_mode.stop_recording()
        self.hub.publish_control_text(protocol.dumps("RecordingStopped"))
