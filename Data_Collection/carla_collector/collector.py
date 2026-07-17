"""High-level orchestration; domain details live in dedicated modules."""

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import carla

from .events import EventCounters
from .geometry import utc_now
from .map_export import MapArtifacts
from .metadata import write_metadata
from .sensors import SensorSuite
from .state_builder import StateBuilder
from .synchronizer import FrameSynchronizer
from .writer import DatasetWriter


class CarlaCollector:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        self.world = None
        self.map = None
        self.ego = None
        self.session_dir = None
        self.session_id = None
        self.tick_callback_id = None
        self.writer = None
        self.sensor_suite = None
        self.state_builder = None
        self.stop_event = threading.Event()
        self.packet_queue = queue.Queue(maxsize=args.queue_size)
        required = ("seg", "state", "rgb") if args.image_mode == "seg-rgb" else ("seg", "state")
        self.sync = FrameSynchronizer(self.packet_queue, required)
        self.events = EventCounters()

    def find_ego(self):
        deadline = time.time() + self.args.wait_vehicle_timeout
        printed = False
        while not self.stop_event.is_set() and time.time() < deadline:
            vehicles = list(self.world.get_actors().filter("vehicle.*"))
            if self.args.vehicle_id:
                matches = [item for item in vehicles if item.id == self.args.vehicle_id]
            else:
                matches = [item for item in vehicles
                           if item.attributes.get("role_name", "") == self.args.role_name]
                if not matches and len(vehicles) == 1:
                    matches = vehicles
            if matches:
                return matches[0]
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
        state = self.state_builder.build(world_snapshot)
        if state is None:
            self.stop_event.set()
            return
        self.sync.put(world_snapshot.frame, "state", state)

    def run(self):
        print("Ket noi CARLA %s:%d..." % (self.args.host, self.args.port))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = self.find_ego()
        print("Da tim thay ego: id=%d, type=%s" % (self.ego.id, self.ego.type_id))
        self.make_session()

        map_artifacts = MapArtifacts(self.map, self.args, self.session_dir)
        graph_stats = map_artifacts.export()
        goal_waypoint = map_artifacts.resolve_goal()
        self.sensor_suite = SensorSuite(
            self.world, self.ego, self.args, self.sync, self.events)
        self.sensor_suite.spawn()
        self.state_builder = StateBuilder(
            self.world, self.map, self.ego, self.args, self.events,
            self.session_id, goal_waypoint)
        write_metadata(
            self.session_dir / "metadata.json", self.world, self.map, self.ego,
            self.args, self.session_id, goal_waypoint, graph_stats)

        self.writer = DatasetWriter(
            self.session_dir, self.packet_queue, self.stop_event,
            save_rgb=self.args.image_mode == "seg-rgb",
            save_seg_color=self.args.save_seg_color,
            max_samples=self.args.max_samples)
        self.writer.start()
        self.tick_callback_id = self.world.on_tick(self.capture_state)
        print("Dang thu thap tai: %s" % self.session_dir)
        print("Nhan Ctrl+C de dung an toan.")

        start = time.time()
        last_report = start
        while not self.stop_event.is_set():
            if not self.ego.is_alive:
                print("Ego vehicle da bi huy; dung collector.")
                break
            if self.args.duration > 0 and time.time() - start >= self.args.duration:
                break
            if self.writer.error:
                raise self.writer.error
            if time.time() - last_report >= 2.0:
                print("samples=%d | queue=%d | dropped=%d" % (
                    self.writer.samples, self.packet_queue.qsize(), self.sync.dropped))
                last_report = time.time()
            time.sleep(0.1)
        self.stop_event.set()

    def cleanup(self):
        self.stop_event.set()
        if self.tick_callback_id is not None and self.world is not None:
            try:
                self.world.remove_on_tick(self.tick_callback_id)
            except RuntimeError:
                pass
        if self.sensor_suite is not None:
            self.sensor_suite.destroy()
        if self.writer is not None:
            self.writer.join(timeout=20.0)
        if self.session_dir is not None:
            summary = {
                "samples_written": self.writer.samples if self.writer else 0,
                "frames_dropped_or_incomplete": self.sync.dropped,
                "finished_utc": utc_now(),
            }
            try:
                with (self.session_dir / "summary.json").open(
                        "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
            except OSError:
                pass
            print("Da dung. Tong so mau: %d" % summary["samples_written"])
