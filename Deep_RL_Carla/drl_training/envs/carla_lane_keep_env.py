"""Active-client CARLA lane-keeping environment for DRL fine-tuning (PPO or SAC).

Unlike `data_collection/` (a *passive* client that attaches to a vehicle already spawned by
`automatic_control.py`), this environment is an *active* client: it spawns its own ego
vehicle, drives the simulation in synchronous mode, and owns the world tick. Do NOT run
this alongside the passive data-collection pipeline or `automatic_control.py` against the
same CARLA server/world — both would fight over vehicle control and world settings.

Observation, reward and termination follow `docs/csv_fields_by_task.md` ("DRL" section) and
`docs/manual_thu_thap_du_lieu.md` §9.2, scoped to lane-keeping only: no A* route fields,
since `router_plan/Global_Route_Planner.py` is not implemented yet and the thesis itself
targets lane-keeping control, not full route navigation.
"""

import queue
import random
import sys
import time
from pathlib import Path

import numpy as np

# Ngu den truoc moc thoi gian bao nhieu roi moi quay tay (xem `_tick`). 3ms du rong de che
# do phan giai timer 15.6ms cua Windows ma khong quay tay lau.
_SPIN_MARGIN = 0.003
# Tut lai qua muc nay thi bo qua phan no thay vi co duoi.
_MAX_LAG_SECONDS = 0.5

_ENV_FILE = Path(__file__).resolve()
_DRL_TRAINING_DIR = _ENV_FILE.parents[1]          # "drl_training/"
_REPO_ROOT = _ENV_FILE.parents[2]                  # "Deep_RL_Carla/"
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
for _path in (str(_DRL_TRAINING_DIR), str(_DATA_COLLECTION_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from carla_collector.events import EventCounters
except ImportError as exc:
    raise ImportError(
        "Khong import duoc carla_collector tu '%s'. Dam bao thu muc 'data_collection' van "
        "nam canh 'drl_training' trong repo (khong doi ten/di chuyen thu muc goc)." %
        _DATA_COLLECTION_DIR
    ) from exc

from policy.observation import build_vehicle_state, resize_class_map

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc khi chay "
        "module DRL (xem docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc


class CarlaLaneKeepEnv(object):
    """Duck-typed Gym-like env: `reset() -> (obs, info)`, `step(action) -> (obs, reward,
    terminated, truncated, info)` — the modern Gymnasium 5-tuple `step()` contract.

    Deliberately NOT a `gymnasium.Env` subclass: this project targets the CARLA 0.9.10
    Python 3.7 client API, and recent `gymnasium` releases have dropped 3.7 support. The
    interface matches Gymnasium closely enough that wrapping this in a real `gymnasium.Env`
    later (on a newer Python + a rebuilt CARLA wheel) is a thin shim, not a rewrite.

    `observation_contract` (a `policy.observation.ObservationContract`, built once from the
    IL checkpoint in `train_ppo.py`) is required: it is the single source of truth for how
    raw CARLA state turns into the scalar feature vector, shared with the actor/critic that
    consume this env's observations.
    """

    def __init__(self, config, observation_contract):
        self.cfg = config
        self.contract = observation_contract
        self.client = carla.Client(config["host"], config.get("port", 2000))
        self.client.set_timeout(config.get("timeout", 20.0))
        self._load_town(config.get("town"))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self._apply_synchronous_mode()
        self.spectator = (self.world.get_spectator()
                          if self.cfg.get("spectator_follow", False) else None)
        self._tick_seconds = 1.0 / float(self.cfg.get("fps", 20.0))
        self._realtime = bool(self.cfg.get("realtime", False))
        self._next_tick_wall = None

        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("Map hien tai (%s) khong co spawn point nao." % self.map.name)

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.events = None
        self._seg_queue = None

        self.action_repeat = max(1, int(config.get("action_repeat", 1)))
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self.off_lane_streak = 0
        self.step_count = 0
        self._rng = random.Random(config.get("seed"))
        self._check_control_rate()

    # ------------------------------------------------------------------ setup / teardown
    def _load_town(self, town):
        """Nap ban do neu `town` co dat va khac ban do dang chay.

        O day chu khong o tung entrypoint: train_ppo.py, train_sac.py, evaluate.py va
        demo_il.py deu dung env nay, nen chi can mot ban cai dat. Bo qua khi ban do da
        dung — `load_world()` dung lai TOAN BO the gioi, dat tien va khong can thiet.
        """
        if not town:
            return
        world = self.client.get_world()
        current = world.get_map().name.replace("\\", "/").split("/")[-1]
        if current.lower() == str(town).lower():
            return
        # Neu the gioi hien tai con ket o synchronous mode (lan chay truoc bi Ctrl+C hoac
        # crash truoc khi close() kip khoi phuc), load_world() se TREO vinh vien: server
        # cho mot tick ma khong con client nao goi. Tra ve async truoc da.
        settings = world.get_settings()
        if settings.synchronous_mode:
            print("[!] The gioi dang ket o synchronous mode (lan chay truoc chua don sach) "
                  "— tra ve async truoc khi nap ban do.")
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print("Dang nap ban do %s (dang chay: %s) ..." % (town, current))
        self.client.load_world(town)

    def _tick(self):
        """Mot tick vat ly, kem hai thu chi phuc vu NGUOI XEM.

        Camera duoc doi o day — moi TICK, khong phai moi quyet dinh. Do duoc o lan chay
        demo: 14.6 quyet dinh/s nghia la camera chi nhay 14.6 lan/giay trong khi the gioi
        di 58 tick/s, nen chuyen dong giat. Doi moi tick dua no len 58 lan/giay.

        `realtime` ghim vong lap ve dung toc do that. O sync mode, `fixed_delta_seconds`
        chi noi MOT tick dai bao nhieu trong the gioi mo phong — no khong ep vong lap cho.
        Client keo duoc bao nhieu thi CARLA chay bay nhieu: do duoc 58 tick/s x 0.05s =
        nhanh gap 2.9 lan thuc te. Khong sai ve vat ly, nhung nhin thi nhu tua nhanh.
        """
        self.world.tick()
        self._update_spectator()
        if not self._realtime:
            return
        now = time.perf_counter()
        if self._next_tick_wall is None:
            self._next_tick_wall = now
        self._next_tick_wall += self._tick_seconds
        if self._next_tick_wall - now < -_MAX_LAG_SECONDS:
            # Da tut lai qua xa (vd vua nap xong ban do, hoac mot tick bi treo): bo phan no
            # thay vi co duoi bang mot loat tick khong nghi.
            self._next_tick_wall = now + self._tick_seconds
        # KHONG ngu thang bang time.sleep(): tren Windows + Python 3.7 no dung do phan giai
        # timer he thong 15.6ms, do duoc tren may nay sleep(0.033) that ra ngu 0.047s
        # (+43%) va sleep(0.010) ngu 0.016s (+61%). Ngu thang thi mo phong chay CHAM hon
        # thuc te va giat — dung thu dang muon sua. Ngu den truoc muc tieu _SPIN_MARGIN roi
        # quay tay not phan con lai: sai so con duoi mot mili giay, doi lai vai phan tram
        # mot nhan CPU va CHI khi `--realtime` duoc bat de xem.
        remaining = self._next_tick_wall - time.perf_counter()
        if remaining > _SPIN_MARGIN:
            time.sleep(remaining - _SPIN_MARGIN)
        while time.perf_counter() < self._next_tick_wall:
            pass

    def _update_spectator(self):
        """Dat camera cua so CARLA phia sau + tren cao xe ego, nhin cung huong xe.

        Goi mot lan moi QUYET DINH (5 Hz) chu khong moi tick: mat du muot cho nguoi xem, va
        set_transform() la mot lenh RPC — goi 20 lan/giay chi de nhin thi khong dang.
        """
        if self.spectator is None or self.vehicle is None:
            return
        tf = self.vehicle.get_transform()
        yaw = np.radians(tf.rotation.yaw)
        self.spectator.set_transform(carla.Transform(
            carla.Location(x=tf.location.x - 8.0 * np.cos(yaw),
                           y=tf.location.y - 8.0 * np.sin(yaw),
                           z=tf.location.z + 4.5),
            carla.Rotation(pitch=-16.0, yaw=tf.rotation.yaw)))

    def _apply_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.cfg.get("fps", 20.0))
        settings.no_rendering_mode = bool(self.cfg.get("no_rendering", False))
        self.world.apply_settings(settings)

    def _check_control_rate(self):
        """Doi chieu nhip ra quyet dinh cua env voi `control_dt` ghi trong checkpoint IL.

        Day la loai lech IM LANG: shape van khop, khong gi bao loi, chi co policy hanh xu
        vo nghia. In canh bao thay vi raise vi co truong hop co tinh (vd doi chung 10 Hz).
        """
        control_dt = getattr(self.contract, "control_dt", None)
        if not control_dt:
            return
        env_dt = self.action_repeat / float(self.cfg.get("fps", 20.0))
        if abs(env_dt - control_dt) > 1e-6:
            print("[!] Nhip dieu khien env = %.3fs (fps=%.1f x action_repeat=%d) nhung "
                  "checkpoint IL train o control_dt = %.3fs. `previous_steer`/"
                  "`previous_longitudinal` se mang y nghia khac luc train. Dat "
                  "fps=%.1f + action_repeat=%d de khop." %
                  (env_dt, float(self.cfg.get("fps", 20.0)), self.action_repeat, control_dt,
                   1.0 / 0.05, int(round(control_dt / 0.05))))

    def _camera_blueprint(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self.cfg.get("width", 160)))
        bp.set_attribute("image_size_y", str(self.cfg.get("height", 128)))
        bp.set_attribute("fov", str(self.cfg.get("fov", 90.0)))
        return bp

    def _on_camera_image(self, image):
        # Same decode as `data_collection/carla_collector/writer.py`: raw class ID lives in
        # the red channel of the semantic segmentation camera's BGRA output.
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        labels = array[:, :, 2].copy()
        if self._seg_queue.full():
            try:
                self._seg_queue.get_nowait()
            except queue.Empty:
                pass
        self._seg_queue.put_nowait(labels)

    def _spawn_actors(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.cfg.get("vehicle_filter", "vehicle.lincoln.mkz2017"))[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "drl_ego")

        spawn_points = list(self.spawn_points)
        self._rng.shuffle(spawn_points)
        vehicle = None
        for transform in spawn_points:
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Khong spawn duoc xe o bat ky spawn point nao (co the dang bi chiem).")
        self.vehicle = vehicle

        camera_tf = carla.Transform(
            carla.Location(
                x=self.cfg.get("camera_x", 1.5),
                y=self.cfg.get("camera_y", 0.0),
                z=self.cfg.get("camera_z", 2.4)),
            carla.Rotation(pitch=self.cfg.get("camera_pitch", -5.0)),
        )
        self._seg_queue = queue.Queue(maxsize=1)
        self.camera = self.world.spawn_actor(
            self._camera_blueprint(), camera_tf, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)
        self.camera.listen(self._on_camera_image)

        self.events = EventCounters()
        self.collision_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.collision"), carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self.events.collision)
        self.lane_invasion_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self.events.lane_invasion)

    def _destroy_actors(self):
        actors = (self.camera, self.collision_sensor, self.lane_invasion_sensor, self.vehicle)
        for actor in actors:
            if actor is not None and hasattr(actor, "stop"):
                try:
                    actor.stop()
                except RuntimeError:
                    pass
        for actor in actors:
            if actor is not None:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except RuntimeError:
                    pass
        self.vehicle = self.camera = self.collision_sensor = self.lane_invasion_sensor = None
        self._seg_queue = None

    def close(self):
        self._destroy_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------------- Gym API
    def reset(self):
        self._destroy_actors()
        self._spawn_actors()
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self.off_lane_streak = 0
        self.step_count = 0

        seg = None
        self._next_tick_wall = None      # reset ton thoi gian -> dung co duoi cho bu
        for _ in range(self.cfg.get("warmup_ticks", 4)):
            self._tick()
            seg = self._get_seg_frame()
        state = self._build_state(seg)
        return self._make_observation(state), {"state": state}

    def step(self, action):
        """Mot STEP = mot QUYET DINH cua policy, keo dai `action_repeat` tick vat ly.

        Ly do phai co action_repeat: hop dong IL (`control_dt` trong checkpoint, 0.2s = 5 Hz)
        dinh nghia `previous_steer`/`previous_longitudinal` la "lenh cua 0.2s TRUOC". Neu env
        goi policy moi tick 0.05s thi cung mot o dac trung mang y nghia khac han luc train ->
        warm-start lech ma khong co exception nao. Nhung ha thang `fixed_delta_seconds` len
        0.2s de bu lai thi vat ly CARLA vo (khuyen cao chinh thuc: <= 0.05s) — xe rung, va
        cham gia. Nen: vat ly chay 0.05s, policy quyet dinh moi 4 tick (`action_repeat`).
        `_check_control_rate()` doi chieu fps*action_repeat voi `control_dt` cua checkpoint.
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        throttle, brake = (longitudinal, 0.0) if longitudinal >= 0.0 else (0.0, -longitudinal)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        seg = None
        for _ in range(self.action_repeat):
            # apply_control() lai o MOI tick: CARLA giu lenh cuoi cung nen mot lan la du,
            # nhung goi lai la vo hai va chong truong hop mode/agent khac chen ngang.
            self.vehicle.apply_control(control)
            self._tick()
            seg = self._get_seg_frame()
        self.step_count += 1

        state = self._build_state(seg)
        reward, terminated, info = self._compute_reward(state, steer, longitudinal)
        truncated = self.step_count >= self.cfg.get("max_episode_steps", 1000)

        self.previous_steer = steer
        self.previous_longitudinal = longitudinal
        info["state"] = state
        return self._make_observation(state), reward, terminated, truncated, info

    # ------------------------------------------------------------------------- internals
    def _get_seg_frame(self):
        timeout = self.cfg.get("frame_timeout", 5.0)
        try:
            return self._seg_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                "Khong nhan duoc frame camera segmentation trong %.1fs — kiem tra CARLA "
                "server con chay, fps cau hinh, va khong co client passive nao khac dang "
                "giu synchronous_mode." % timeout)

    def _build_state(self, seg):
        state = build_vehicle_state(
            self.vehicle, self.map, seg, self.previous_steer, self.previous_longitudinal)

        collisions, lane_invasions = self.events.snapshot()
        new_collision = collisions > self.previous_collisions
        new_lane_invasions = max(0, lane_invasions - self.previous_lane_invasions)
        self.previous_collisions, self.previous_lane_invasions = collisions, lane_invasions

        state.update({
            "collisions": collisions,
            "lane_invasions": lane_invasions,
            "new_collision": bool(new_collision),
            "new_lane_invasions": int(new_lane_invasions),
        })
        return state

    def _compute_reward(self, state, steer, longitudinal):
        cfg = self.cfg
        speed_limit_mps = max(state["speed_limit_kmh"] / 3.6, 1e-6)
        speed_term = float(np.clip(state["forward_speed_mps"], 0.0, speed_limit_mps))
        steer_delta = steer - self.previous_steer
        long_delta = longitudinal - self.previous_longitudinal

        # NGA TU: tat cac so hang doc tu waypoint. `lane_offset_m`, `heading_error_rad` va
        # `off_lane` deu suy ra tu "lan duong gan nhat", ma trong nga tu cac nhanh cat nhau
        # nen lan do doi sang nhanh vuong goc/nguoc chieu chi sau vai met (xem chu thich o
        # policy/observation.py). Giu chung lai la dua vao PPO mot gradient RAC o dung nhung
        # buoc kho nhat: do tren runs/il_demo_v9, ca 3 episode deu chet trong ~13 buoc sau
        # khi tham chieu nhay, va bi tru diem vi mot do lech vo nghia (-2.4, -4.0/buoc).
        #
        # Cac so hang KHONG bi tat: `speed_term` (di tiep van tot), do muot cua
        # steer/longitudinal va va cham — chung do tren chinh chiec xe, khong qua ban do,
        # nen van dung trong nga tu. `w_yaw_rate` cung giu: no la so hang uu tien nhe, khong
        # phai mot phep do bi hong, va rieng viec vao cua thi quay la dung.
        #
        # Con thieu, va co y de lai: khong co `route_command` trong observation nen policy
        # VAN khong biet nen re huong nao trong nga tu. Vo hieu hoa reward chi ngung day no
        # hoc nham; day la viec cua Router Plan (xem README.md, muc "Pham vi").
        in_junction = bool(state.get("is_junction", 0)) and cfg.get("junction_mask_lane_terms", True)
        w_lane = 0.0 if in_junction else cfg.get("w_lane_offset", 1.0)
        w_head = 0.0 if in_junction else cfg.get("w_heading", 0.5)

        reward = (
            cfg.get("w_speed", 1.0) * speed_term
            - w_lane * abs(state["lane_offset_m"])
            - w_head * abs(state["heading_error_rad"])
            - cfg.get("w_steer_delta", 1.0) * steer_delta ** 2
            - cfg.get("w_long_delta", 0.5) * long_delta ** 2
            - cfg.get("w_yaw_rate", 0.1) * state["yaw_rate_rps"] ** 2
        )

        if in_junction:
            # DONG BANG streak, khong phai xoa. Xe dang ra ngoai lan ma di vao nga tu thi van
            # dang ra ngoai lan; xoa ve 0 se tang cho no mot lan "an xa" moi lan qua nga tu.
            pass
        elif state["off_lane"]:
            reward -= cfg.get("off_lane_penalty", 5.0)
            self.off_lane_streak += 1
        else:
            self.off_lane_streak = 0

        if state["new_lane_invasions"]:
            reward -= cfg.get("lane_invasion_penalty", 1.0) * state["new_lane_invasions"]

        terminated = False
        info = {}
        if state["new_collision"]:
            reward -= cfg.get("collision_penalty", 50.0)
            terminated = True
            info["terminate_reason"] = "collision"
        elif self.off_lane_streak >= cfg.get("off_lane_patience_steps", 20):
            terminated = True
            info["terminate_reason"] = "off_lane"

        return float(reward), terminated, info

    def _make_observation(self, state):
        # Camera chay o 480x384 (khop collector, de dung chung mot ham voi duong
        # camera that sau nay), nhung OBSERVATION ha xuong 240x192 = dung do phan
        # giai `IMAGE_WIDTH`/`IMAGE_HEIGHT` ma train_il_v9.ipynb da train. Hai ly do:
        #   1. Actor warm-start tu IL nhin thay dung thang do dac trung no da hoc.
        #      AdaptiveAvgPool2d khien moi kich thuoc deu CHAY duoc nen sai lech nay
        #      khong bao loi gi - no chi lam warm-start kem hieu qua trong im lang.
        #   2. Giam 4x bo nho rollout/replay (SAC 50k: 9.2GB -> 2.3GB).
        # resize_class_map() bao ton class manh (RoadLine) giong het downscale_labels()
        # cua notebook IL - xem docstring cua no.
        seg = resize_class_map(
            state["seg"],
            self.cfg.get("obs_height", self.cfg.get("height", 128)),
            self.cfg.get("obs_width", self.cfg.get("width", 160)),
        )
        scalar = self.contract.build_scalar_vector(state)
        return {"seg": seg.astype(np.uint8), "scalar": scalar}
