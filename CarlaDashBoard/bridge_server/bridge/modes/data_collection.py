"""Data Collection mode — design doc §07: "Giám sát ghi dữ liệu (passive client bám xe do
Traffic Manager lái)".

Deliberately NOT a wrapper around data_collection/collect_data.py's `CarlaCollector`: that
class is a *passive* client — it waits for a "hero" vehicle some other process already spawned
(automatic_control.py) and drives, and it spawns its own rgb+seg sensor pair. The Bridge
Server's CarlaSession already owns the one ego + one camera pair shared by every mode (so
Live Drive keeps working no matter which mode is active), so wiring in a second, independent
sensor pair here would just double up cameras on the same car for no benefit.

What this mode does instead: hands the wheel to CARLA's own Traffic Manager
(`vehicle.set_autopilot(True)`) and, only while recording is explicitly turned on (the
"⏺ Bắt đầu ghi" control in Hình 02), appends one row per tick to `states.csv` and periodically
saves the camera frames CarlaSession already decoded — reusing them instead of re-encoding.
This is a live-monitoring counterpart, not a byte-identical dataset generator; for an actual
training dataset with map export + goal metadata, keep using `collect_data.py` directly.
"""

import csv
import logging
import time
from pathlib import Path

import carla

from .base import ModeRuntime

logger = logging.getLogger("bridge.modes.data_collection")

CSV_FIELDS = [
    "frame", "t", "x", "y", "z", "yaw", "speed_kmh",
    "steer", "throttle", "brake", "collision_count", "lane_invasion_count",
]

SAVE_IMAGE_EVERY_N_FRAMES = 5  # ~4 img/s at sim_fps=20 — plenty for a live "sanity" preview

# .../CarlaDashBoard — moc de neo `record_output_dir` khi no la duong dan tuong doi.
DASHBOARD_ROOT = Path(__file__).resolve().parents[3]


def resolve_output_dir(record_output_dir):
    """Duong dan tuong doi duoc neo vao thu muc CarlaDashBoard, KHONG phai thu muc lam viec.

    `Path("dataset_live").resolve()` phu thuoc vao cho nguoi dung dung luc go lenh: chay
    `python bridge_server\\run_server.py` tu CarlaDashBoard thi du lieu vao
    `CarlaDashBoard/dataset_live`, con `cd bridge_server` roi chay `python run_server.py`
    (dung cach README bao) lai tao ra mot thu muc dataset_live THU HAI trong bridge_server/.
    Quan sat duoc that khi chay bo test end-to-end: ban ghi khong nam cho nguoi dung tuong.
    Duong dan tuyet doi thi van duoc ton trong nguyen ven.
    """
    path = Path(record_output_dir).expanduser()
    if not path.is_absolute():
        path = DASHBOARD_ROOT / path
    return path.resolve()


class DataCollectionMode(ModeRuntime):
    name = "DATA_COLLECTION"

    def __init__(self, cfg):
        self.cfg = cfg
        self.session = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.recording = False
        self.csv_writer = None
        self.csv_file = None
        self.session_dir = None
        self.frames_recorded = 0
        self._record_started_at = None
        self._tick_counter = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self, session):
        self.session = session
        ego = session.ego
        world = session.world
        ego.set_autopilot(True)

        blueprint_library = world.get_blueprint_library()
        self.collision_sensor = world.spawn_actor(
            blueprint_library.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        self.collision_sensor.listen(lambda _e: self._bump("collision_count"))
        self.lane_sensor = world.spawn_actor(
            blueprint_library.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
        self.lane_sensor.listen(lambda _e: self._bump("lane_invasion_count"))
        logger.info("Data Collection mode started (autopilot ON)")

    def _bump(self, field):
        setattr(self, field, getattr(self, field) + 1)

    def tick(self, snapshot):
        self._tick_counter += 1
        if self.recording:
            self._write_row(snapshot)
        return None  # Traffic Manager already drives — nothing for SimLoop to apply

    def stop(self):
        self.stop_recording()
        for sensor in (self.collision_sensor, self.lane_sensor):
            if sensor is None:
                continue
            try:
                sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
        self.collision_sensor = None
        self.lane_sensor = None
        if self.session and self.session.ego is not None:
            try:
                self.session.ego.set_autopilot(False)
            except (RuntimeError, IndexError):
                # IndexError("invalid unordered_map<K, T> key") la cach Traffic Manager bao
                # "toi khong con giu dang ky cua chiec xe nay" — gap that khi doi mode luc
                # dang chay Data Collection. No KHONG phai RuntimeError nen ban truoc lot
                # qua except, bay len tan `_cmd_SetMode` va lam ca lenh SetMode that bai
                # (COMMAND_FAILED) SAU KHI mode moi da start() xong: mode moi bi bo roi,
                # con mode cu thi da huy het sensor — dashboard ket o mot trang thai khong
                # ai lai ma van bao dang o Data Collection. Bo autopilot khong duoc thi
                # cung khong sao: mode tiep theo se tu ap lenh dieu khien cua no moi tick.
                pass

    def status_extra(self):
        extra = {
            "recording": self.recording,
            "collision_count": self.collision_count,
            "lane_invasion_count": self.lane_invasion_count,
        }
        if self.recording:
            extra["frames_recorded"] = self.frames_recorded
            extra["record_elapsed_s"] = round(time.time() - self._record_started_at, 1)
        return extra

    # ------------------------------------------------------------------ recording toggle
    def start_recording(self):
        if self.recording:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        town = self.session.current_town_short()
        self.session_dir = resolve_output_dir(self.cfg.record_output_dir) / f"{town}_{stamp}"
        (self.session_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "seg").mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.session_dir / "states.csv").open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()
        self.frames_recorded = 0
        self._record_started_at = time.time()
        self.recording = True
        logger.info("Recording started -> %s", self.session_dir)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.csv_file:
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        logger.info("Recording stopped (%d frames -> %s)", self.frames_recorded, self.session_dir)

    def _write_row(self, snapshot):
        ego = self.session.ego
        if ego is None or not ego.is_alive:
            return
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        speed_kmh = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        control = ego.get_control()
        self.csv_writer.writerow({
            "frame": snapshot.frame,
            "t": round(snapshot.timestamp.elapsed_seconds, 3),
            "x": round(transform.location.x, 3),
            "y": round(transform.location.y, 3),
            "z": round(transform.location.z, 3),
            "yaw": round(transform.rotation.yaw, 2),
            "speed_kmh": round(speed_kmh, 2),
            "steer": round(control.steer, 4),
            "throttle": round(control.throttle, 4),
            "brake": round(control.brake, 4),
            "collision_count": self.collision_count,
            "lane_invasion_count": self.lane_invasion_count,
        })
        self.frames_recorded += 1

        if self._tick_counter % SAVE_IMAGE_EVERY_N_FRAMES == 0:
            if self.session.last_rgb_jpeg:
                (self.session_dir / "rgb" / f"{snapshot.frame}.jpg").write_bytes(
                    self.session.last_rgb_jpeg)
            # .png, khong phai .jpg: khung segmentation duoc ma hoa PNG (anh chi so lop 4
            # mau — xem cameras.class_map_to_png). Ghi byte PNG vao file ten .jpg thi mo
            # bang thu vien nao cung phai doan lai dinh dang.
            if self.session.last_seg_image:
                (self.session_dir / "seg" / f"{snapshot.frame}.png").write_bytes(
                    self.session.last_seg_image)
