"""Background writer for images and frame-aligned CSV rows."""

import csv
import json
import queue
import threading

import numpy as np
from PIL import Image

from .schema import CSV_FIELDS, SEG_COLOR_LUT


class DatasetWriter(threading.Thread):
    def __init__(self, session_dir, packet_queue, stop_event, save_rgb,
                 save_seg_color, max_samples=0, dedup_stationary_speed=0.0,
                 dedup_action_eps=0.02, dedup_min_interval_s=1.0,
                 stall_speed=0.3):
        super().__init__(daemon=True)
        self.session_dir = session_dir
        self.packet_queue = packet_queue
        self.stop_event = stop_event
        self.save_rgb = save_rgb
        self.save_seg_color = save_seg_color
        self.max_samples = max_samples
        self.dedup_stationary_speed = dedup_stationary_speed
        self.dedup_action_eps = dedup_action_eps
        self.dedup_min_interval_s = dedup_min_interval_s
        # Watchdog nguong rieng, KHONG dung chung voi dedup: tat dedup van phai
        # phat hien duoc xe dung yen vinh vien.
        self.stall_speed = stall_speed
        self.latest_sim_time = None
        self.last_moving_sim_time = None
        self.samples = 0
        self.duplicates_skipped = 0
        self.error = None
        self.previous_action = None
        self.previous_sim_time = None
        self.last_kept_sim_time = None
        self.first_sim_time = None
        # So doan (moi chiec ego mot doan) va tong span cua cac doan da dong lai;
        # xem begin_new_segment.
        self.segments = 1
        self.span_accum = 0.0
        # Frame CARLA cua mau cuoi cung da ghi. Collector doc no de nhan ra world
        # da bi load lai (frame counter tut ve 0) truoc khi ghi de len anh cu.
        self.last_frame = 0
        # Bat len khi collector doi sang mot chiec xe khac; chinh writer thread
        # ap dung o packet ke tiep, de mot packet cua xe cu con dang duoc ghi
        # khong kip ghi de len cac moc vua reset.
        self.segment_break = False

    def begin_new_segment(self):
        """Bao rang cac mau sau day den tu MOT CHIEC XE KHAC.

        Goi khi collector go sensor khoi ego cu va gan sang ego moi trong cung
        mot session. Cat chuoi lien tuc cua `previous_steer`/`previous_longitudinal`
        va cua dong ho dung yen: hai chiec xe khac nhau khong noi thanh mot chuoi
        dieu khien, va quang cho xe moi khong phai la quang xe dung yen.
        """
        self.segment_break = True

    def _apply_segment_break(self):
        if self.first_sim_time is not None and self.last_kept_sim_time is not None:
            # Nhip lay mau chi co nghia TRONG mot doan; quang cho xe moi (co the
            # hang phut) khong duoc tinh vao span, neu khong measured_fps se tut
            # xuong ma khong co mau nao that su bi mat.
            self.span_accum += max(0.0, self.last_kept_sim_time - self.first_sim_time)
            self.segments += 1
        self.first_sim_time = None
        self.last_kept_sim_time = None
        self.previous_action = None
        self.previous_sim_time = None
        self.latest_sim_time = None
        self.last_moving_sim_time = None
        self.segment_break = False

    def sim_time_span(self):
        """Simulated seconds covered by the samples actually written."""
        span = self.span_accum
        if self.first_sim_time is not None and self.last_kept_sim_time is not None:
            span += max(0.0, self.last_kept_sim_time - self.first_sim_time)
        return span

    def stationary_seconds(self):
        """So giay SIM ke tu lan cuoi ego con di chuyen.

        Do o day chu khong o collector vi writer la cho duy nhat nhin thay MOI
        packet da ghep - ke ca packet bi dedup bo di. Neu do bang `samples` thi
        mot xe dung yen vinh vien voi dedup bat van sinh ~1 mau/giay va watchdog
        se khong bao gio kich hoat.
        """
        if self.segment_break:
            # Dang doi xe: dong ho cua chiec xe cu khong con y nghia gi nua.
            return 0.0
        if self.latest_sim_time is None or self.last_moving_sim_time is None:
            return 0.0
        return max(0.0, self.latest_sim_time - self.last_moving_sim_time)

    def measured_fps(self):
        """Samples per SIMULATED second, as actually written to disk.

        The collector cannot set the world's timestep, so `--fps` is a request,
        not a guarantee - report what was achieved so a bad session is visible
        while it is still running instead of at training time.
        """
        span = self.sim_time_span()
        # Moi doan (moi chiec xe) chi dong gop `so_mau - 1` khoang cach, nen tru
        # theo so doan chu khong phai tru 1.
        intervals = self.samples - self.segments
        if span <= 0.0 or intervals < 1:
            return 0.0
        return intervals / span

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
                        if self.write_packet(packet, writer):
                            handle.flush()
                    finally:
                        self.packet_queue.task_done()
                    if self.max_samples and self.samples >= self.max_samples:
                        self.stop_event.set()
                        break
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def is_redundant(self, speed_mps, steer_delta, longitudinal_delta, current_sim_time):
        """True neu mau nay la ban sao gan nhu y het mau da giu gan nhat.

        Chi kich hoat khi xe DUNG YEN (vd. cho den do, ket xe) VA hanh dong khong
        doi so voi mau da ghi truoc do - day la truong hop anh gan nhu tinh, khong
        mang them thong tin. Khong bao gio bo mau luc xe dang di chuyen: luc do
        canh vat xung quanh doi that theo tung frame nen khong phai trung lap.
        """
        if self.dedup_stationary_speed <= 0.0 or self.previous_action is None:
            return False
        if speed_mps >= self.dedup_stationary_speed:
            return False
        if abs(steer_delta) >= self.dedup_action_eps:
            return False
        if abs(longitudinal_delta) >= self.dedup_action_eps:
            return False
        if self.last_kept_sim_time is None:
            return False
        return (current_sim_time - self.last_kept_sim_time) < self.dedup_min_interval_s

    def write_packet(self, packet, csv_writer):
        """Ghi mot mau (anh + dong CSV). Tra ve False neu mau bi bo qua vi trung lap."""
        if self.segment_break:
            self._apply_segment_break()

        state = packet["state"]
        frame = int(state["frame"])
        stem = "%08d" % frame

        current_steer = float(state.get("steer", 0.0))
        current_longitudinal = float(state.get("longitudinal", 0.0))
        current_sim_time = float(state.get("sim_time_s", 0.0))
        speed_mps = float(state.get("speed_mps", 0.0))

        # Cap nhat TRUOC khi kiem tra trung lap, de mau bi dedup bo van tinh vao
        # watchdog. Packet dau tien lam moc, neu khong thi mot ego dung yen ngay
        # tu dau se giu last_moving_sim_time = None mai mai.
        self.latest_sim_time = current_sim_time
        if self.last_moving_sim_time is None or speed_mps >= self.stall_speed:
            self.last_moving_sim_time = current_sim_time

        if self.previous_action is None:
            previous_steer = current_steer
            previous_longitudinal = current_longitudinal
            sample_delta_seconds = 0.0
        else:
            previous_steer, previous_longitudinal = self.previous_action
            sample_delta_seconds = max(
                0.0, current_sim_time - float(self.previous_sim_time))
        steer_delta = current_steer - previous_steer
        longitudinal_delta = current_longitudinal - previous_longitudinal

        if self.is_redundant(speed_mps, steer_delta, longitudinal_delta, current_sim_time):
            self.duplicates_skipped += 1
            return False

        state.update({
            "previous_steer": previous_steer,
            "previous_longitudinal": previous_longitudinal,
            "steer_delta": steer_delta,
            "longitudinal_delta": longitudinal_delta,
            "sample_delta_seconds": sample_delta_seconds,
            # StateBuilder leaves these as plain lists; serialize only for the
            # samples actually kept, instead of on every world tick.
            "successor_waypoints_json": json.dumps(
                state.get("successor_waypoints_json", []), separators=(",", ":")),
            "lookahead_waypoints_json": json.dumps(
                state.get("lookahead_waypoints_json", []), separators=(",", ":")),
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
            color = SEG_COLOR_LUT[labels]  # vectorized class_id -> RGB lookup
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
        if self.first_sim_time is None:
            self.first_sim_time = current_sim_time
        self.last_kept_sim_time = current_sim_time
        self.last_frame = frame
        self.samples += 1
        return True
