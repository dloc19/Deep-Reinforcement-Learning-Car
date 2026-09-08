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

        # `town` nhan MOT ten hoac MOT DANH SACH ten. Danh sach = xoay vong ban do trong
        # luc train, doi ban do sau moi `town_rotate_episodes` episode.
        #
        # Vi sao can: checkpoint IL (behavior_cloning/best_il_model.pth) ghi
        # `train_towns = ['Town01','Town02','Town03','Town04']` — no da hoc tren BON ban do.
        # Fine-tune PPO chi tren MOT ban do lam policy quen ba ban do kia: do duoc tren
        # runs/ppo_v3, lech lan tren Town01 di tu 0.138 m (IL) len 0.382 m (PPO) sau 250
        # update chi chay Town04, trong khi tren chinh Town04 thi PPO tot len. Day la quen
        # tham hoa (catastrophic forgetting) kinh dien, khong phai loi thuat toan.
        towns = config.get("town")
        if isinstance(towns, (list, tuple)):
            self.town_list = [t for t in towns if t]
        elif towns:
            self.town_list = [towns]
        else:
            self.town_list = []
        self.town_rotate_episodes = int(config.get("town_rotate_episodes", 25))
        self._town_index = 0
        self._episodes_on_town = 0
        self._load_town(self.town_list[0] if self.town_list else None)
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
            # Hai truong hop rat khac nhau cung roi vao day, dung mot thong bao cho ca hai
            # thi doc log se hieu nham: (a) XOAY BAN DO giua lan train — sync mode la cua
            # CHINH ta, hoan toan binh thuong, va `_rebind_world()` se bat lai ngay sau khi
            # nap xong; (b) lan chay TRUOC bi Ctrl+C/crash truoc khi close() kip khoi phuc —
            # luc do khong con client nao tick va load_world() se treo vinh vien neu khong
            # tra ve async. `self.vehicle` con song = ta dang lai = truong hop (a).
            if getattr(self, "vehicle", None) is not None or getattr(self, "_town_index", 0) or self.town_list[1:]:
                print("Tra the gioi ve async de nap ban do (sync mode dang do lan train nay giu).")
            else:
                print("[!] The gioi dang ket o synchronous mode (lan chay truoc chua don sach) "
                      "— tra ve async truoc khi nap ban do.")
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print("Dang nap ban do %s (dang chay: %s) — co the mat vai phut ..." % (town, current))
        # `load_world()` phai chay duoi mot timeout RIENG, dai hon nhieu timeout dieu khien
        # thuong ngay. Do tren may nay: `timeout` mac dinh 20s KHONG du de server dung xong
        # Town04 (33.8 km lan). Het 20s thi client bo cuoc voi "failed to connect to newly
        # created map" — va server ket lai o trang thai nap do, khong tra loi bat ky client
        # nao nua (thu lai voi timeout 120s van khong vao duoc), phai tat CarlaUE4 di bat
        # lai. Tuc mot timeout dat qua ngan o day khong chi lam that bai lenh nay, no GIET
        # ca server. Tra ve timeout cu ngay sau do: 300s cho mot lenh dieu khien thuong
        # ngay thi qua dai, se bien mot server treo thanh mot lan train treo im lang.
        control_timeout = float(self.cfg.get("timeout", 20.0))
        self.client.set_timeout(float(self.cfg.get("map_load_timeout", 300.0)))
        try:
            self.client.load_world(town)
        finally:
            self.client.set_timeout(control_timeout)

    def _rebind_world(self):
        """Doc lai moi tham chieu phu thuoc world sau khi `load_world()` thay the no.

        `client.load_world()` dung LEN mot world moi — moi handle cu (world, map,
        spectator, spawn point) tro toi world da chet. Quen mot cai la loi im lang:
        vd `self.map` cu van tra ve waypoint cua ban do truoc, nen `lane_offset_m` duoc
        tinh tren mot ban do khac han ban do xe dang chay.
        """
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self._apply_synchronous_mode()
        self.spectator = (self.world.get_spectator()
                          if self.cfg.get("spectator_follow", False) else None)
        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("Map %s khong co spawn point nao." % self.map.name)
        self._next_tick_wall = None

    def _maybe_rotate_town(self):
        """Doi sang ban do ke tiep khi da chay du `town_rotate_episodes` episode.

        Xoay theo EPISODE chu khong theo tick: `load_world()` mat hang phut, nen doi qua
        day thi phan lon thoi gian la nap ban do. 25 episode ~ 8-10 phut lai xe cho mot
        lan nap ~2 phut.
        """
        if len(self.town_list) < 2:
            return None
        if self._episodes_on_town < self.town_rotate_episodes:
            return None
        self._town_index = (self._town_index + 1) % len(self.town_list)
        self._episodes_on_town = 0
        target = self.town_list[self._town_index]
        self._load_town(target)
        self._rebind_world()
        return target

    @property
    def current_town(self):
        name = self.map.name.replace("\\", "/").split("/")[-1]
        return name

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
        frame_id = self.world.tick()
        self._update_spectator()
        if not self._realtime:
            return frame_id
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
        return frame_id

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
        #
        # Kem theo `image.frame`: `_get_seg_frame()` PHAI ghep dung anh cua dung tick ma
        # `_tick()` vua tinh (xem docstring o do). Hang doi khong gioi han kich thuoc va
        # KHONG bo frame nao: o sync mode moi tick sinh dung mot anh va vong lap tieu thu
        # dung mot anh, nen do sau hang doi luon ~1. Ban cu dung maxsize=1 + bo cai cu nhat,
        # tuc no vut di dung cai frame ma tick ke tiep dang cho -> `get()` treo het
        # `frame_timeout` roi nem RuntimeError, giet ca lan train.
        camera_queue = self._seg_queue
        if camera_queue is None:
            return      # callback con bay giua chung sau khi _destroy_actors() da chay
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        camera_queue.put((image.frame, array[:, :, 2].copy()))

    def _spawn_actors(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.cfg.get("vehicle_filter", "vehicle.lincoln.mkz2017"))[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "drl_ego")

        spawn_points = list(self.spawn_points)
        # `fixed_spawn_index` (mac dinh None = giu nguyen hanh vi cu: xao ngau nhien) cho
        # phep ep xe xuat phat tai DUNG mot spawn point. Can cho `evaluate_route.py`: de so
        # sanh "policy don" voi "policy + A*" mot cach cong bang thi hai lan chay phai di
        # CUNG mot tuyen, ma tuyen bat dau tu diem xuat phat. Neu de ngau nhien, chenh lech
        # do duoc se lan voi chenh lech do tuyen de/kho khac nhau.
        #
        # Van giu ca danh sach lam du phong (chi day diem duoc chon len dau) chu khong chi
        # thu mot diem duy nhat: spawn point co the dang bi xe khac chiem, va khi do that bai
        # cung nen roi ve diem khac hon la hong ca lo danh gia.
        fixed = self.cfg.get("fixed_spawn_index")
        if fixed is None:
            self._rng.shuffle(spawn_points)
        else:
            idx = int(fixed) % len(spawn_points)
            spawn_points = [spawn_points[idx]] + spawn_points[:idx] + spawn_points[idx + 1:]
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
        self._seg_queue = queue.Queue()
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
        rotated = self._maybe_rotate_town()
        if rotated:
            print("Doi ban do -> %s (sau %d episode)" % (rotated, self.town_rotate_episodes))
        self._episodes_on_town += 1
        self._spawn_actors()
        # Thong ke lech lan cua episode — de train_ppo.py/train_sac.py ghi vao episode_log.csv.
        # Truoc day chi `evaluate.py` do dai luong nay, nen mot lan train nham vao viec cai
        # thien bam lan (vd tang w_lane_offset) khong the theo doi duoc gi cho toi tan buoc
        # eval cuoi cung. Tach rieng "tren duong thuong" vi trong nga tu `lane_offset_m` la
        # phep do rac (xem _compute_reward).
        self._ep_offsets = []
        self._ep_offsets_road = []
        self._ep_junction_steps = 0
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self.off_lane_streak = 0
        self.step_count = 0

        seg = None
        self._next_tick_wall = None      # reset ton thoi gian -> dung co duoi cho bu
        for _ in range(self.cfg.get("warmup_ticks", 4)):
            seg = self._get_seg_frame(self._tick())
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
            seg = self._get_seg_frame(self._tick())
        self.step_count += 1

        state = self._build_state(seg)
        reward, terminated, info = self._compute_reward(state, steer, longitudinal)
        truncated = self.step_count >= self.cfg.get("max_episode_steps", 1000)

        offset = abs(state["lane_offset_m"])
        self._ep_offsets.append(offset)
        if state.get("is_junction"):
            self._ep_junction_steps += 1
        else:
            self._ep_offsets_road.append(offset)

        self.previous_steer = steer
        self.previous_longitudinal = longitudinal
        info["state"] = state
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats()
        return self._make_observation(state), reward, terminated, truncated, info

    def episode_stats(self):
        """Tom tat episode vua ket thuc — cung dinh nghia voi `evaluate.py` de hai nguon so
        lieu (train va eval) doc duoc tren cung mot thang."""
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
        return {
            "mean_abs_lane_offset": mean(self._ep_offsets),
            "mean_abs_lane_offset_road": mean(self._ep_offsets_road),
            "junction_steps": self._ep_junction_steps,
            "town": self.current_town,
        }

    # ------------------------------------------------------------------------- internals
    def _get_seg_frame(self, frame_id):
        """Anh segmentation cua DUNG tick `frame_id` (gia tri `world.tick()` tra ve).

        Khop frame la bat buoc, khong phai cho chac. `data_collection` — bo du lieu ma
        checkpoint IL duoc train tren do — khop chat theo `image.frame`
        (carla_collector/sensors.py: `gate.wants(image.frame, ...)` roi
        `synchronizer.put(image.frame, ...)`), nen moi mau IL hoc la mot cap (anh tick N,
        trang thai xe tick N). Ban truoc cua ham nay chi lay "cai gi dang nam trong hang
        doi" roi ghep voi transform xe doc o tick hien tai: mot do lech mot tick khong bao
        loi gi, chi lam policy warm-start nhin thay mot cap (anh, trang thai) khac loai voi
        cap no da hoc. O 8 m/s, mot tick 0.05s = 0.4 m sai lech.

        Callback camera chay tren thread rieng nen anh ve SAU khi `world.tick()` da tra ve;
        cho o day la dung, va la dung cach vi du sync chinh thuc cua CARLA lam.
        """
        timeout = self.cfg.get("frame_timeout", 5.0)
        deadline = time.time() + timeout
        while True:
            try:
                frame, labels = self._seg_queue.get(timeout=max(deadline - time.time(), 0.0))
            except queue.Empty:
                raise RuntimeError(
                    "Khong nhan duoc frame camera segmentation cho tick %s trong %.1fs — "
                    "kiem tra CARLA server con chay, fps cau hinh, va khong co client "
                    "passive nao khac dang giu synchronous_mode." % (frame_id, timeout))
            if frame >= frame_id:
                # `>` thay vi `==` chi xay ra neu mot anh bi mat hoan toan; lay anh moi nhat
                # van dung hon la treo cho mot frame khong bao gio den.
                return labels
            # frame < frame_id: anh cua tick da qua (vd con sot lai tu warmup) -> bo, doi tiep

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
        steer_delta = steer - self.previous_steer
        long_delta = longitudinal - self.previous_longitudinal

        # ------------------------------------------------------------- thang do cua reward
        # `reward_mode`:
        #   "normalized" (mac dinh) — moi so hang khong thu nguyen, nam trong [0, 1].
        #   "raw"                   — cong thuc cu (speed tinh bang m/s tho). Giu lai DUY NHAT
        #                             de tai lap runs/ppo_v1..v3; khong dung cho lan chay moi.
        #
        # Vi sao phai doi: ban "raw" tra `w_speed * forward_speed_mps`, tuc thuong theo m/s
        # THO. Do tren runs/ppo_v2 (Town04, co doan cao toc 90 km/h), so hang nay dat 13.47
        # moi buoc, trong khi phat lech lan toi da chi 1.75 -> ti le 14:1. Hai he qua do duoc:
        #   1. Policy hoc "chay nhanh, giu tim lan gan nhu mien phi": lech lan tren duong
        #      thang di tu 0.098 m (IL) len 0.233-0.341 m (PPO) tren CHINH ban do da train,
        #      te hon co y nghia thong ke (Welch t = 2.22).
        #   2. Cung mot hanh vi lai duoc thuong khac nhau 3 LAN giua pho 30 km/h va cao toc
        #      90 km/h, nen mot ham gia tri hoc o town nay sai thang o town kia — dieu do
        #      lam viec train nhieu ban do (xem `town_rotate_episodes`) kho hon nhieu.
        # Ban "normalized" chia moi dai luong cho thang tu nhien cua no: toc do cho gioi han
        # toc do, lech lan cho nua be rong lan, goc lech cho 45 do. Sau do `w_*` moi thuc su
        # la "trong so tuong doi", va return mot episode 500 buoc ~ 100 thay vi ~800.
        if cfg.get("reward_mode", "normalized") == "raw":
            speed_term = float(np.clip(state["forward_speed_mps"], 0.0, speed_limit_mps))
            lane_term = abs(state["lane_offset_m"])
            heading_term = abs(state["heading_error_rad"])
        else:
            # Mau so la TOC DO MUC TIEU, khong phai gioi han toc do hop phap.
            #
            # Ban dau to chia cho `speed_limit_mps` va do la mot loi: tren doan cao toc
            # Town04 (90 km/h = 25 m/s), lai dung nhu IL — 7.56 m/s, chinh la trung binh
            # cua tap train IL — chi duoc 7.56/25 = 0.30 diem toc do, nen sau khi tru phat
            # lech lan thi tong con AM (-0.18 do duoc voi lech 0.85 m). Tuc reward noi voi
            # policy rang DUNG YEN (0.00) tot hon lai xe. Do dung la che do hong "xe dung
            # im" ma bang chan doan da canh bao.
            #
            # `min(target, limit)` giu ca hai tinh chat: thuong day du o toc do ma he thong
            # thuc su lai duoc (nhat quan giua pho va cao toc), va van ha muc tieu xuong o
            # nhung doan gioi han thap hon target.
            speed_ref = min(float(cfg.get("target_speed_mps", 8.0)), speed_limit_mps)
            speed_term = float(np.clip(state["forward_speed_mps"] / max(speed_ref, 1e-6), 0.0, 1.0))
            half_width = max(float(state.get("lane_half_width_m", 1.75)), 1e-6)
            lane_term = float(np.clip(abs(state["lane_offset_m"]) / half_width, 0.0, 1.0))
            heading_term = float(np.clip(abs(state["heading_error_rad"]) / (np.pi / 4.0), 0.0, 1.0))

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
            - w_lane * lane_term
            - w_head * heading_term
            - cfg.get("w_steer_delta", 1.0) * steer_delta ** 2
            - cfg.get("w_long_delta", 0.5) * long_delta ** 2
            - cfg.get("w_yaw_rate", 0.1) * state["yaw_rate_rps"] ** 2
        )

        if in_junction:
            # Trong nga tu, `off_lane` binh thuong la phep do RAC (waypoint tham chieu nhay
            # sang nhanh vuong goc sau vai met), nen khong dung no de dem streak. Nhung
            # "dong bang hoan toan" — ban truoc — la mot lo hong: no vo hieu hoa luon chot
            # an toan, va reward cung da tat so hang bam lan o day, nen ben trong nga tu
            # KHONG con bat ky ap luc nao giu xe tren mat duong.
            #
            # Do duoc tren runs/ppo_v4, Town04: 3/3 va cham deu xay ra trong nga tu, deu
            # dam vao static.guardrail/static.fence, mot lan o lech lan +4.72 m — tuc xe da
            # ra khoi mat duong hoan toan ma episode van khong bi dung. Town04 la ban do duy
            # nhat co NUT GIAO CAO TOC: vung danh dau junction rat rong nen xe troi rat xa
            # ma van "dang trong nga tu". Town01/Town02 (giao lo pho nho) khong bi: 0% va cham.
            #
            # Nguong rong o day giu dung y dinh ban dau (bo qua nhieu do vai chuc cm) nhung
            # van bat duoc do lech LON, thu khong the la nhieu: 3 x nua be rong lan ~ 5.2 m.
            factor = float(cfg.get("junction_off_lane_factor", 3.0))
            half_width = max(float(state.get("lane_half_width_m", 1.75)), 1e-6)
            if abs(state["lane_offset_m"]) > factor * half_width:
                self.off_lane_streak += 1
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
        # giai `IMAGE_WIDTH`/`IMAGE_HEIGHT` ma train_il.ipynb da train. Hai ly do:
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
