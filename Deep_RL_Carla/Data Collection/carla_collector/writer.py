"""Background writer for images and frame-aligned CSV rows."""

import csv
import queue
import threading

import numpy as np
from PIL import Image

from .schema import CITYSCAPES_COLORS, CSV_FIELDS


class DatasetWriter(threading.Thread):
    def __init__(self, session_dir, packet_queue, stop_event, save_rgb,
                 save_seg_color, max_samples=0):
        super().__init__(daemon=True)
        self.session_dir = session_dir
        self.packet_queue = packet_queue
        self.stop_event = stop_event
        self.save_rgb = save_rgb
        self.save_seg_color = save_seg_color
        self.max_samples = max_samples
        self.samples = 0
        self.error = None
        self.previous_action = None
        self.previous_sim_time = None

    @staticmethod
    def decode_bgra(raw_data, width, height):
        return np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))

    def run(self):
        csv_path = self.session_dir / "states.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                while not self.stop_event.is_set() or not self.packet_queue.empty():
                    try:
                        packet = self.packet_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        self.write_packet(packet, writer)
                        handle.flush()
                    finally:
                        self.packet_queue.task_done()
                    if self.max_samples and self.samples >= self.max_samples:
                        self.stop_event.set()
                        break
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def write_packet(self, packet, csv_writer):
        state = packet["state"]
        frame = int(state["frame"])
        stem = "%08d" % frame

        current_steer = float(state.get("steer", 0.0))
        current_longitudinal = float(state.get("longitudinal", 0.0))
        current_sim_time = float(state.get("sim_time_s", 0.0))
        if self.previous_action is None:
            previous_steer = current_steer
            previous_longitudinal = current_longitudinal
            sample_delta_seconds = 0.0
        else:
            previous_steer, previous_longitudinal = self.previous_action
            sample_delta_seconds = max(
                0.0, current_sim_time - float(self.previous_sim_time))
        state.update({
            "previous_steer": previous_steer,
            "previous_longitudinal": previous_longitudinal,
            "steer_delta": current_steer - previous_steer,
            "longitudinal_delta": current_longitudinal - previous_longitudinal,
            "sample_delta_seconds": sample_delta_seconds,
        })

        seg_data, seg_w, seg_h = packet["seg"]
        seg_bgra = self.decode_bgra(seg_data, seg_w, seg_h)
        labels = seg_bgra[:, :, 2].copy()  # Raw class ID is in BGRA red.
        label_rel = "seg_label/%s.png" % stem
        Image.fromarray(labels, mode="L").save(
            str(self.session_dir / label_rel), compress_level=3)

        rgb_rel = ""
        if self.save_rgb:
            rgb_data, rgb_w, rgb_h = packet["rgb"]
            rgb_bgra = self.decode_bgra(rgb_data, rgb_w, rgb_h)
            rgb = rgb_bgra[:, :, :3][:, :, ::-1].copy()
            rgb_rel = "rgb/%s.png" % stem
            Image.fromarray(rgb, mode="RGB").save(
                str(self.session_dir / rgb_rel), compress_level=3)

        color_rel = ""
        if self.save_seg_color:
            color = np.zeros((seg_h, seg_w, 3), dtype=np.uint8)
            for class_id, class_color in CITYSCAPES_COLORS.items():
                color[labels == class_id] = class_color
            color_rel = "seg_color/%s.png" % stem
            Image.fromarray(color, mode="RGB").save(
                str(self.session_dir / color_rel), compress_level=3)

        state.update({
            "sample_id": self.samples,
            "rgb_path": rgb_rel,
            "seg_label_path": label_rel,
            "seg_color_path": color_rel,
        })
        csv_writer.writerow({field: state.get(field, "") for field in CSV_FIELDS})
        self.previous_action = (current_steer, current_longitudinal)
        self.previous_sim_time = current_sim_time
        self.samples += 1
