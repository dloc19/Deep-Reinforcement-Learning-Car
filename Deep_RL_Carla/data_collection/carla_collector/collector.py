"""High-level orchestration; domain details live in dedicated modules."""

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import carla

from .ego_watch import find_ego_actor, wait_for_ego  # noqa: F401 (re-export)
from .events import EventCounters
from .geometry import utc_now
from .map_export import MapArtifacts
from .metadata import write_metadata
from .sensors import SensorSuite
from .state_builder import StateBuilder
from .synchronizer import FrameGate, FrameSynchronizer, SampleRateLimiter
from .writer import DatasetWriter

# Nhung ly do dung ma DOI SANG MOT CHIEC XE KHAC co the sua duoc: xe ket, camera
# chet, xe bi huy. Tat ca deu la chuyen cua RIENG chiec ego do, khong phai cua
# session - session (thu muc, states.csv, so mau da ghi) van tot nguyen.
REBINDABLE_REASONS = frozenset((
    "vehicle_stationary_timeout",
    "no_samples_timeout",
    "ego_destroyed",
    "camera_dead",
))


class CarlaCollector:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        self.world = None
        self.map = None
        self.ego = None
        # Id cua chiec ego gan nhat, giu lai ca sau khi no bi huy: dung de KHONG
        # gan sensor vao lai dung chiec xe vua ket.
        self.last_ego_id = 0
        self.goal_waypoint = None
        self.session_dir = None
        self.session_id = None
        self.tick_callback_id = None
        self.writer = None
        self.sensor_suite = None
        self.state_builder = None
        self.stop_event = threading.Event()
        # Vi sao session ket thuc. Ghi vao summary.json de mot session hong khong
        # con trong y het mot session tot chi it mau hon.
        self.stop_reason = None
        # Moi lan sensor duoc gan vao mot chiec xe = mot doan. Mot session co the
        # co nhieu doan; cot `vehicle_id` trong states.csv cho biet moi dong
        # thuoc doan nao.
        self.ego_history = []
        self.rebinds = 0
        # Lan gan lai RIENG camera vao dung chiec xe do (khong doi xe, khong cat
        # session): xem respawn_sensors.
        self.camera_restarts = 0
        # Moc de bat hai kieu hong ma `ego.is_alive` khong bao gio thay:
        #  - ego bien khoi world snapshot (client KHAC huy no; `is_alive` la co
        #    cua rieng client nay nen no van True mai mai),
        #  - camera ngung gui anh trong khi world van tick.
        # Truoc day ca hai deu phai doi het `--stall-timeout-s` (60 s) moi lo ra.
        self.ego_missing_since = None
        self.last_tick_wall = time.time()
        self.packet_queue = queue.Queue(maxsize=args.queue_size)
        required = ("seg", "state", "rgb") if args.image_mode == "seg-rgb" else ("seg", "state")
        # sensor_tick alone does not hold the sampling rate on an async world that
        # another client owns; see SampleRateLimiter. Every downstream consumer
        # assumes a constant CONTROL_DT = 1/fps, so enforce it here on sim_time.
        self.rate_limiter = SampleRateLimiter(args.fps)
        # Loc nhip lay mau TRUOC khi mot frame kip ton chi phi gi: camera khong
        # copy anh cua frame bi bo, world tick khong dung state day du cho no.
        self.gate = FrameGate(self.rate_limiter)
        self.sync = FrameSynchronizer(
            self.packet_queue, required, on_emit=self.gate.commit_packet)
        self.events = EventCounters()

    def set_stop_reason(self, reason):
        """Ly do DAU TIEN thang: cac buoc don dep phia sau khong duoc ghi de len no."""
        if self.stop_reason is None:
            self.stop_reason = reason

    def find_ego(self):
        deadline = time.time() + self.args.wait_vehicle_timeout
        printed = False
        while not self.stop_event.is_set() and time.time() < deadline:
            ego = find_ego_actor(self.world, self.args)
            if ego is not None:
                return ego
            if not printed:
                print("Dang cho xe ego (role_name=%s)..." % self.args.role_name)
                printed = True
            time.sleep(0.5)
        raise RuntimeError(
            "Khong tim thay ego vehicle. Chay automatic_control.py truoc, "
            "hoac truyen --vehicle-id / --role-name.")

    def make_session(self):
        map_short = self.map.name.split("/")[-1]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_id = "%s_%s" % (map_short, stamp)
        self.session_dir = Path(self.args.output).expanduser().resolve() / self.session_id
        directories = ["seg_label"]
        if self.args.image_mode == "seg-rgb":
            directories.append("rgb")
        if self.args.save_seg_color:
            directories.append("seg_color")
        for index, name in enumerate(directories):
            (self.session_dir / name).mkdir(
                parents=True, exist_ok=False if index == 0 else True)

    def capture_state(self, world_snapshot):
        builder = self.state_builder
        if builder is None:
            # Dang o giua hai chiec xe (xem unbind_ego): khong co ego de mo ta.
            return
        self.last_tick_wall = time.time()
        actor_snapshot = world_snapshot.find(self.ego.id)
        if actor_snapshot is None:
            # Ego khong xuat hien trong snapshot nay. Mot frame le la chuyen binh
            # thuong; vang mat LIEN TUC thi la no da bi huy - va `ego.is_alive`
            # KHONG bao duoc dieu do vi client khac moi la ben goi destroy().
            # Vong lap chinh doc `ego_missing_since` de biet.
            if self.ego_missing_since is None:
                self.ego_missing_since = time.time()
            return
        self.ego_missing_since = None
        # Odometer chay o moi tick, con state day du (~15 truy van waypoint +
        # 4 RPC) chi dung cho nhung frame that su duoc chon lam mau.
        builder.track_odometry(actor_snapshot)
        if not self.gate.wants(
                world_snapshot.frame, world_snapshot.timestamp.elapsed_seconds):
            return
        state = builder.build(world_snapshot, actor_snapshot)
        if state is None:
            return
        self.sync.put(world_snapshot.frame, "state", state)

    def bind_ego(self, ego, distance_travelled_m=0.0):
        """Gan camera + state builder vao mot chiec xe. Session giu nguyen."""
        self.ego = ego
        self.last_ego_id = ego.id
        self.sensor_suite = SensorSuite(
            self.world, ego, self.args, self.sync, self.events, self.gate)
        self.sensor_suite.spawn()
        self.ego_missing_since = None
        self.last_tick_wall = time.time()
        self.sync.mark_camera_alive()
        # Quang duong cong don theo CA SESSION nen khong reset ve 0 khi doi xe,
        # nhung `last_location` thi phai reset - neu khong doan tu cho xe cu ket
        # den cho xe moi spawn se bi cong vao nhu mot buoc nhay hang tram met.
        self.state_builder = StateBuilder(
            self.world, self.map, ego, self.args, self.events,
            self.session_id, self.goal_waypoint,
            distance_travelled_m=distance_travelled_m)
        self.ego_history.append({
            "vehicle_id": ego.id,
            "vehicle_type": ego.type_id,
            "bound_utc": utc_now(),
            "samples_at_bind": self.writer.samples if self.writer else 0,
            "samples": 0,
            "released_reason": None,
        })

    def unbind_ego(self, reason):
        """Go sensor khoi ego hien tai, giu lai writer/session/so mau da ghi."""
        self.state_builder = None  # capture_state thanh no-op ngay lap tuc
        if self.sensor_suite is not None:
            self.sensor_suite.destroy()
            self.sensor_suite = None
        self.drain_queue()
        self.sync.reset()  # bo cac frame do dang cua ego cu
        if self.writer is not None:
            # Mau dau tien cua chiec xe sau KHONG duoc lay `previous_steer` tu
            # lenh cuoi cua chiec xe truoc, va khoang cach giua hai mau do khong
            # phai mot buoc dieu khien - do la mot vet cat, khong phai mot buoc.
            self.writer.begin_new_segment()
        if self.ego_history:
            entry = self.ego_history[-1]
            entry["released_reason"] = reason
            entry["samples"] = (
                (self.writer.samples if self.writer else 0) - entry["samples_at_bind"])

    def drain_queue(self, timeout_s=5.0):
        """Cho writer ghi not nhung packet da ghep xong cua ego cu."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.writer is None or not self.writer.is_alive():
                return
            if self.packet_queue.empty():
                return
            time.sleep(0.05)

    def world_changed(self):
        """True neu world hien tai KHONG con la world dang thu.

        `client.load_world()` huy toan bo actor va dat lai frame counter ve 0.
        Ghi tiep vao cung session luc do se de len chinh cac file anh da ghi
        (`seg_label/%08d.png` dat ten theo frame), va `map` trong metadata.json
        co the khong con dung - phai mo session moi thay vi gan sensor vao.
        """
        try:
            world = self.client.get_world()
            snapshot = world.get_snapshot()
        except RuntimeError:
            return True
        if getattr(world, "id", None) != getattr(self.world, "id", None):
            return True
        last_frame = self.writer.last_frame if self.writer else 0
        return bool(last_frame) and snapshot.frame < last_frame

    def respawn_sensors(self):
        """Gan lai RIENG camera vao dung chiec xe dang bam. True neu thanh cong.

        Camera co the im trong khi xe van song va van chay: hai lan trong mot
        session Town03 that, collector phai doi het 60 s `--stall-timeout-s` roi
        doi xe, va chiec "xe moi" no tim thay chinh la chiec cu (id 124 -> 124,
        137 -> 137). Doi xe trong truong hop do vua cham vua thua: no bat chiec
        xe phai DANG CHAY moi nhan lai (xem pick_ego), nen neu camera chet dung
        luc xe dung den do thi con phai cho them. Thay ca bo camera mat vai chuc
        mili giay va giu nguyen ca ego lan bo dem quang duong.
        """
        if self.ego is None or not self.ego.is_alive or self.ego_missing_since:
            return False
        try:
            if self.sensor_suite is not None:
                self.sensor_suite.destroy()
            self.sensor_suite = None
            self.drain_queue()
            self.sync.reset()
            if self.writer is not None:
                # Quang camera im la mot vet cat, khong phai mot buoc dieu khien:
                # mau dau tien sau do khong duoc noi `previous_steer` voi mau cuoi
                # truoc do.
                self.writer.begin_new_segment()
            self.sensor_suite = SensorSuite(
                self.world, self.ego, self.args, self.sync, self.events, self.gate)
            self.sensor_suite.spawn()
        except RuntimeError as exc:
            print("Khong gan lai duoc camera: %s" % exc)
            return False
        self.sync.mark_camera_alive()
        self.camera_restarts += 1
        print("Da gan lai camera vao xe id=%d (lan thu %d). Van la session cu, "
              "van la chiec xe cu." % (self.ego.id, self.camera_restarts))
        return True

    def try_rebind(self, reason):
        """Doi sang mot chiec xe khac ma KHONG dong session. True neu thanh cong.

        Day la ly do collector khong con chet theo chiec xe no dang bam: khi
        `automatic_control.py` ket cung, nguoi dung chay lai no de tao hero moi,
        con session dang chay thi tu tim chiec hero do va thu tiep vao dung thu
        muc cu - khong sinh them session vun.
        """
        if not self.args.rebind_ego:
            self.set_stop_reason(reason)
            return False
        if self.args.max_rebinds and self.rebinds >= self.args.max_rebinds:
            print("Da doi xe %d lan (--max-rebinds). Dong session."
                  % self.args.max_rebinds)
            self.set_stop_reason(reason)
            return False

        previous_id = self.last_ego_id
        distance = (self.state_builder.distance_travelled_m
                    if self.state_builder is not None else 0.0)
        self.unbind_ego(reason)
        print("Da go sensor khoi xe id=%d (%s). Da ghi %d mau, session van mo."
              % (previous_id, reason, self.writer.samples if self.writer else 0))
        if self.args.vehicle_id:
            print("CHU Y: --vehicle-id=%d ghim cung mot xe, chi nhan lai dung xe do."
                  % self.args.vehicle_id)

        ego, _ = wait_for_ego(
            self.client, self.args, self.args.rebind_wait_s,
            lambda: self.stop_event.is_set(), previous_id=previous_id,
            message=("Cho xe moi (chay lai automatic_control.py) de thu tiep vao "
                     "session nay, toi da %.0f s..."))
        if ego is None:
            if not self.stop_event.is_set():
                print("Het %.0f s ma chua co xe nao chay. Dong session."
                      % self.args.rebind_wait_s)
                self.set_stop_reason("rebind_timeout")
            return False
        if self.world_changed():
            print("World da duoc load lai (frame counter va actor deu moi). "
                  "Khong ghi tiep vao session cu duoc; dong session.")
            self.set_stop_reason("world_reloaded")
            return False

        self.rebinds += 1
        self.bind_ego(ego, distance_travelled_m=distance)
        print("Da gan sensor vao xe moi: id=%d, type=%s (lan doi xe thu %d). "
              "Thu tiep vao %s" % (ego.id, ego.type_id, self.rebinds, self.session_dir))
        return True

    def run(self):
        print("Ket noi CARLA %s:%d..." % (self.args.host, self.args.port))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        ego = self.find_ego()
        print("Da tim thay ego: id=%d, type=%s" % (ego.id, ego.type_id))
        self.make_session()

        map_artifacts = MapArtifacts(self.map, self.args, self.session_dir)
        graph_stats = map_artifacts.export()
        self.goal_waypoint = map_artifacts.resolve_goal()
        write_metadata(
            self.session_dir / "metadata.json", self.world, self.map, ego,
            self.args, self.session_id, self.goal_waypoint, graph_stats)

        self.writer = DatasetWriter(
            self.session_dir, self.packet_queue, self.stop_event,
            save_rgb=self.args.image_mode == "seg-rgb",
            save_seg_color=self.args.save_seg_color,
            max_samples=self.args.max_samples,
            dedup_stationary_speed=self.args.dedup_stationary_speed,
            dedup_action_eps=self.args.dedup_action_eps,
            dedup_min_interval_s=self.args.dedup_min_interval_s,
            stall_speed=self.args.stall_speed)
        self.writer.start()
        self.tick_callback_id = self.world.on_tick(self.capture_state)
        self.bind_ego(ego)
        print("Dang thu thap tai: %s" % self.session_dir)
        print("Nhan Ctrl+C de dung an toan.")

        self.main_loop()

    def main_loop(self):
        """Vong lap giam sat. Tach khoi run() de test duoc khong can server CARLA."""
        start = time.time()
        last_report = start
        # Watchdog. Truoc day vong lap chi thoat khi ego bi destroy, het --duration,
        # hoac writer loi - khong cai nao bat duoc hai kieu hong that su da gap:
        # camera ngung gui anh (ego van alive) va agent phanh khan cap vinh vien.
        # Ca hai deu de session chay hang chuc phut roi moi lo ra o summary.json.
        # Gio ba kieu hong do khong con dong session nua ma chi dong CHIEC XE:
        # xem try_rebind.
        last_sample_count = self.writer.samples
        last_progress = start
        evicted_at_progress = self.sync.incomplete_evicted
        warned_no_samples = False
        warned_stationary = False
        # So lan gan lai camera lien tiep ma van khong ra duoc mau nao: gan lai
        # lan thu hai ma van im thi khong phai loi cua camera nua, doi xe.
        camera_restarts_without_progress = 0
        while not self.stop_event.is_set():
            now = time.time()
            # Ly do phai doi xe. None = moi thu binh thuong.
            fault = None
            if not self.ego.is_alive:
                print("Ego vehicle da bi huy.")
                fault = "ego_destroyed"
            missing_since = self.ego_missing_since
            if (fault is None and missing_since is not None
                    and self.args.ego_missing_timeout_s > 0
                    and now - missing_since >= self.args.ego_missing_timeout_s):
                # `ego.is_alive` la co cua RIENG client nay: khi
                # `automatic_control.py` huy chiec hero cu de spawn chiec moi, co
                # do van True mai mai va truoc day phai doi het 60 s stall
                # watchdog moi lo ra. Vang mat khoi world snapshot moi la bang
                # chung that.
                print("WATCHDOG: ego id=%d vang mat khoi world snapshot %.0f s "
                      "- coi nhu da bi huy." % (self.ego.id, now - missing_since))
                fault = "ego_destroyed"
            if self.args.duration > 0 and now - start >= self.args.duration:
                self.set_stop_reason("duration")
                break
            if self.writer.error:
                self.set_stop_reason("writer_error")
                raise self.writer.error

            if self.writer.samples != last_sample_count:
                last_sample_count = self.writer.samples
                last_progress = now
                evicted_at_progress = self.sync.incomplete_evicted
                warned_no_samples = False
                camera_restarts_without_progress = 0
            idle = now - last_progress

            # Camera im trong khi world VAN tick: hong rieng cua camera, khong
            # phai cua xe. Bat o day (mac dinh 10 s) thay vi de stall watchdog
            # bat sau 60 s, va sua bang cach thay camera chu khong phai doi xe.
            if (fault is None and self.args.camera_timeout_s > 0
                    and self.sensor_suite is not None
                    and now - self.last_tick_wall < 2.0):
                silence = now - self.sync.last_image_wall
                if silence >= self.args.camera_timeout_s:
                    print("WATCHDOG: camera khong gui anh nao trong %.0f s "
                          "(world van tick)." % silence)
                    if (camera_restarts_without_progress < 2
                            and self.respawn_sensors()):
                        camera_restarts_without_progress += 1
                        last_sample_count = self.writer.samples
                        last_progress = time.time()
                        evicted_at_progress = self.sync.incomplete_evicted
                        warned_no_samples = False
                        warned_stationary = False
                        last_report = time.time()
                        continue
                    fault = "camera_dead"
            if (fault is None and self.args.stall_timeout_s > 0
                    and idle >= self.args.stall_timeout_s):
                # `incomplete_evicted` tang trong khi khong co mau nao = state van
                # ve deu nhung anh thi khong, tuc la camera da im. Neu ca hai deu
                # dung yen thi la world khong tick hoac ego bien khoi snapshot.
                evicted = self.sync.incomplete_evicted - evicted_at_progress
                cause = ("camera ngung gui anh (%d frame chi co state)" % evicted
                         if evicted > 0
                         else "world khong tick, hoac ego bien mat khoi snapshot")
                print("WATCHDOG: %.0f s khong co mau moi - %s." % (idle, cause))
                fault = "no_samples_timeout"
            if (fault is None and self.args.stall_timeout_s > 0 and not warned_no_samples
                    and idle >= 0.5 * self.args.stall_timeout_s):
                print("CANH BAO: %.0f s khong co mau moi (se doi xe o %.0f s)."
                      % (idle, self.args.stall_timeout_s))
                warned_no_samples = True

            parked = self.writer.stationary_seconds()
            if fault is None and self.args.stationary_timeout_s > 0:
                if parked >= self.args.stationary_timeout_s:
                    print("WATCHDOG: xe dung yen %.0f s sim lien tuc (nguong %.0f s). "
                          "Kiem tra xem agent co dang phanh khan cap khong."
                          % (parked, self.args.stationary_timeout_s))
                    fault = "vehicle_stationary_timeout"
                elif (not warned_stationary
                        and parked >= 0.5 * self.args.stationary_timeout_s):
                    print("CANH BAO: xe da dung yen %.0f s sim (se doi xe o %.0f s)."
                          % (parked, self.args.stationary_timeout_s))
                    warned_stationary = True
                elif parked < 0.5 * self.args.stationary_timeout_s:
                    warned_stationary = False

            if fault is not None:
                if not self.try_rebind(fault):
                    break
                # Xe moi = dong ho watchdog phai dem lai tu dau, neu khong thi
                # chinh quang cho xe moi se lam no nga ngay o vong lap ke tiep.
                last_sample_count = self.writer.samples
                last_progress = time.time()
                evicted_at_progress = self.sync.incomplete_evicted
                warned_no_samples = False
                warned_stationary = False
                last_report = time.time()
                continue

            if now - last_report >= 2.0:
                extra = ""
                if self.rebinds:
                    extra += " | doi_xe=%d" % self.rebinds
                if self.camera_restarts:
                    extra += " | gan_lai_camera=%d" % self.camera_restarts
                print("samples=%d | fps=%.2f | queue=%d | dropped=%d | dup_skip=%d "
                      "| dung_yen=%.0fs%s" % (
                          self.writer.samples, self.writer.measured_fps(),
                          self.packet_queue.qsize(), self.sync.dropped,
                          self.writer.duplicates_skipped, parked, extra))
                last_report = now
            time.sleep(0.1)
        self.stop_event.set()

    def cleanup(self):
        self.stop_event.set()
        if self.tick_callback_id is not None and self.world is not None:
            try:
                self.world.remove_on_tick(self.tick_callback_id)
            except RuntimeError:
                pass
        self.state_builder = None
        if self.sensor_suite is not None:
            self.sensor_suite.destroy()
            self.sensor_suite = None
        if self.writer is not None:
            self.writer.join(timeout=20.0)
        if self.stop_reason is None and self.writer is not None:
            self.stop_reason = ("max_samples"
                                if self.args.max_samples
                                and self.writer.samples >= self.args.max_samples
                                else "unknown")
        if self.ego_history and self.ego_history[-1]["released_reason"] is None:
            entry = self.ego_history[-1]
            entry["released_reason"] = self.stop_reason or "unknown"
            entry["samples"] = (
                (self.writer.samples if self.writer else 0) - entry["samples_at_bind"])
        if self.session_dir is not None:
            summary = {
                "samples_written": self.writer.samples if self.writer else 0,
                # Phan biet session dung dung han voi session chet giua chung.
                # Chi "max_samples", "duration" va "user_interrupt" moi la binh thuong.
                "stop_reason": self.stop_reason or "unknown",
                "vehicle_stationary_s_at_stop": (
                    round(self.writer.stationary_seconds(), 1) if self.writer else 0.0),
                "requested_fps": self.args.fps,
                # Rate actually achieved, from the sim_time span of the samples on
                # disk. Compare it with requested_fps before training on a session:
                # metadata.json records what was ASKED for, this records what came
                # out, and they used to be able to differ by 4.5x without a warning.
                "measured_fps": round(self.writer.measured_fps(), 3) if self.writer else 0.0,
                "sim_time_span_s": round(self.writer.sim_time_span(), 3) if self.writer else 0.0,
                # Mot session co the di qua nhieu chiec xe: `ego_segments` cho biet
                # moi chiec gop bao nhieu mau va vi sao no bi go ra.
                "ego_rebinds": self.rebinds,
                # Gan lai camera vao CHINH chiec xe do (khong doi xe): tach khoi
                # `ego_rebinds` de biet lop nao dang phai lam viec.
                "camera_restarts": self.camera_restarts,
                "ego_segments": self.ego_history,
                "complete_frames_dropped_queue_full": self.sync.dropped,
                "incomplete_frames_evicted": self.sync.incomplete_evicted,
                "frames_skipped_by_rate_limiter": self.rate_limiter.skipped,
                # Packet ghep xong nhung qua sat mau vua ghi (<0.5 chu ky) nen bi
                # bo: hai anh gan nhu y het nhau va `previous_steer` cua mau sau
                # se sai thang do. Xem SampleRateLimiter.is_duplicate.
                "near_duplicate_packets_dropped": self.rate_limiter.duplicates,
                "camera_images_received": self.sync.images_seen,
                "duplicate_frames_skipped": self.writer.duplicates_skipped if self.writer else 0,
                "finished_utc": utc_now(),
            }
            try:
                with (self.session_dir / "summary.json").open(
                        "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
            except OSError:
                pass
            print("Da dung [%s]. Tong so mau: %d (fps do duoc: %.2f, yeu cau: %.2f)" % (
                summary["stop_reason"], summary["samples_written"],
                summary["measured_fps"], self.args.fps))
            if self.rebinds:
                print("Session nay da doi xe %d lan, di qua %d chiec ego."
                      % (self.rebinds, len(self.ego_history)))
            if self.sync.dropped:
                print("CANH BAO: %d frame hoan chinh bi mat vi writer khong kip. "
                      "Tang --queue-size hoac giam --fps." % self.sync.dropped)
