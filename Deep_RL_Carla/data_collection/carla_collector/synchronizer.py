"""Join asynchronous camera and vehicle-state packets by CARLA frame."""

import queue
import threading
import time
from collections import OrderedDict


class SampleRateLimiter:
    """Enforce the configured sampling period in SIMULATION time.

    `sensor_tick` is NOT authoritative here. The collector is a passive client on
    a world that another script (`automatic_control.py`) owns and does not run in
    synchronous mode, so the world steps at a variable rate and the cameras hand
    us frames far faster than `1/fps`. Measured on a real Town01 session that was
    configured for 5 FPS: 461 samples over 20.30 simulated seconds (22.7 FPS),
    with the gap between consecutive samples ranging 0.011 s to 1.027 s.

    That is not a cosmetic problem. `previous_steer` / `previous_longitudinal`
    mean "the control command one step earlier", so the step size IS the scale of
    those features - the training contract in README ("hop dong buoc thoi gian",
    `CONTROL_DT = 1/COLLECT_FPS`) and in both notebooks assumes it is constant.
    Gate on the frame's own `sim_time_s` instead of trusting the server, and the
    rate holds whatever the world happens to be doing.

    Tach lam hai buoc - `wants()` hoi, `commit()` xac nhan - de FrameGate co the
    hoi TRUOC khi mot frame kip ton chi phi gi (xem FrameGate). `wants()` khong
    dich han, nen mot frame duoc chon ma cuoi cung khong ghep xong (camera kia
    khong ban o frame do, state khong ve kip) se khong lam mat nhip: cac frame ke
    tiep van duoc chon cho den khi co MOT mau that su ra doi.
    """

    EMA_ALPHA = 0.1

    def __init__(self, fps):
        self.period = 1.0 / fps
        self.next_deadline = None
        self.last_seen = None
        self.last_committed = None
        self.mean_gap = None
        self.skipped = 0
        # Packet ghep xong nhung qua sat mau vua ghi nen bi bo; xem is_duplicate.
        self.duplicates = 0

    def _observe(self, sim_time):
        """Cap nhat khoang cach trung binh giua hai frame lien tiep."""
        if self.last_seen is None:
            self.last_seen = sim_time
            return
        if sim_time < self.last_seen:
            if self.last_seen - sim_time <= max(1.0, 10.0 * self.period):
                # Lui mot chut: `SensorData.timestamp` cua camera va
                # `WorldSnapshot.timestamp.elapsed_seconds` la hai duong vao khac
                # nhau cua cung mot dong ho, va frame den khong hoan toan dung
                # thu tu. Bo qua, dung coi la world reload - neu khong thi moi
                # lan lech vai mili giay se xoa sach nhip lay mau.
                return
            # World reloaded: sim_time restarted, moi moc cu deu vo nghia.
            self.last_seen = sim_time
            self.next_deadline = None
            self.last_committed = None
            self.mean_gap = None
            return
        gap = sim_time - self.last_seen
        if gap > 0.0:
            self.mean_gap = gap if self.mean_gap is None else (
                (1.0 - self.EMA_ALPHA) * self.mean_gap + self.EMA_ALPHA * gap)
            self.last_seen = sim_time

    def wants(self, sim_time):
        """True neu frame nay du dieu kien lam mau. KHONG dich han - xem commit."""
        self._observe(sim_time)
        if self.next_deadline is None:
            # Chua co mau nao: dang "doi", nhan bat ky frame nao.
            return True
        # Frames arrive quantized at roughly `mean_gap`, so one landing exactly on
        # the deadline is the exception. Accepting only frames at-or-after it would
        # bias the mean period up by half an arrival interval (0.22 s instead of
        # 0.20 s at the rates measured above); allowing the same slack on the early
        # side centres the error on zero instead.
        slack = min(0.5 * (self.mean_gap or 0.0), 0.25 * self.period)
        return sim_time >= self.next_deadline - slack

    def is_duplicate(self, sim_time):
        """True neu packet nay ghep xong qua sat mau vua ghi -> khong nen ghi.

        Mot frame duoc nhan roi van phai doi ca hai camera ve du moi ghep xong;
        trong khoang do `commit()` chua chay nen han ky chua dich, va frame ke ben
        (cach 10 ms) cung duoc nhan. Phai nhan nhu vay: chi ~50% frame co du ca
        hai camera, nen neu chan cac frame ke ben o dau vao thi moi lan frame duoc
        chon khong ghep xong lai phai cho nguyen mot nhip, va nhip lay mau tut tu
        5.0 xuong 3.9 FPS (do bang mo phong). Chan o day - sau khi da biet packet
        nao THAT SU ghep xong - vua giu nguyen nhip vua khong ghi mau trung.

        Do tren session Town03_20260824_092246_845374: 67/10000 mau (0.67%) cach
        mau truoc do duoi 0.05 s. Hai anh gan nhu y het nhau, va `previous_steer`
        cua mau sau la "lenh cua 10 ms truoc" chu khong phai 200 ms - sai hop dong
        CONTROL_DT. Nua chu ky la nguong an toan: mot mau dung nhip luon cach mau
        truoc it nhat `period - slack` >= 0.75 * period.
        """
        return (self.last_committed is not None
                and sim_time - self.last_committed < 0.5 * self.period)

    def commit(self, sim_time):
        """Mot mau da thuc su ghep xong: dat han cho mau ke tiep."""
        self.last_committed = sim_time
        if self.next_deadline is None:
            self.next_deadline = sim_time + self.period
            return
        self.next_deadline += self.period
        if self.next_deadline <= sim_time:
            # Fell a whole period behind (camera hiccup, writer stall). Re-anchor
            # instead of emitting a burst of back-to-back samples to catch up.
            self.next_deadline = sim_time + self.period


class FrameGate:
    """Quyet dinh MOT LAN cho moi frame CARLA: frame nay co dang lam mau khong.

    Truoc day viec loc nhip lay mau chi chay SAU khi packet da ghep xong - tuc la
    sau khi ca hai camera da copy xong ~737 KB anh cua frame do va state day du
    (~15 truy van waypoint + 4 RPC toi server) da duoc dung xong. Do tren mot
    session Town03 that: 79449 packet ghep xong / 4906 mau duoc giu -> 94% khoi
    luong do la de vut di, va no bi vut di NGAY TREN thread ma CARLA dung de day
    anh sang client. Hoi truoc, o day, thi 94% do khong con ton gi ca.

    Cache theo `frame` vi ba nguon (seg, rgb, world tick) hoi cung mot frame tu
    ba thread khac nhau va PHAI nhan cung mot cau tra loi, neu khong packet se
    khong bao gio ghep du.
    """

    def __init__(self, limiter, max_cached=256):
        self.limiter = limiter
        self.max_cached = max_cached
        self.lock = threading.Lock()
        self.decisions = OrderedDict()

    def wants(self, frame, sim_time):
        with self.lock:
            if frame in self.decisions:
                return self.decisions[frame]
            verdict = self.limiter.wants(sim_time)
            if not verdict:
                self.limiter.skipped += 1
            self.decisions[frame] = verdict
            while len(self.decisions) > self.max_cached:
                self.decisions.popitem(last=False)
            return verdict

    def commit_packet(self, packet):
        """Packet da ghep xong: True neu nen ghi, False neu no trung mau vua ghi."""
        state = packet.get("state")
        if state is None:
            return True
        sim_time = float(state.get("sim_time_s", 0.0))
        with self.lock:
            if self.limiter.is_duplicate(sim_time):
                self.limiter.duplicates += 1
                return False
            self.limiter.commit(sim_time)
            return True


class FrameSynchronizer:
    def __init__(self, output_queue, required_keys, max_pending=512, on_emit=None):
        self.output_queue = output_queue
        self.required_keys = tuple(required_keys)
        self.max_pending = max_pending
        # Optional callable(packet) run when a packet is complete; see
        # FrameGate.commit_packet.
        self.on_emit = on_emit
        self.lock = threading.Lock()
        self.pending = OrderedDict()
        # Complete packets lost because the writer could not keep up. This is the
        # only counter that means real data loss.
        self.dropped = 0
        # Frames evicted from `pending` without ever completing: frame chi co
        # state (world tick ma camera khong ban), hoac chi co mot trong hai
        # camera. Ke tu khi FrameGate loc tu dau vao, con so nay nho hon han
        # truoc kia (49015 tren mot session Town03) nen no lai dung duoc lam dau
        # hieu hong.
        self.incomplete_evicted = 0
        # Nhip tim cua camera: dem MOI anh nhan duoc, KE CA anh bi FrameGate bo.
        # Watchdog camera doc no de phan biet "camera im" voi "camera van ban
        # nhung khong frame nao thanh mau".
        self.images_seen = 0
        self.last_image_wall = time.time()

    def note_image(self):
        """Goi ngay khi mot anh ve toi, truoc moi buoc loc."""
        self.images_seen += 1
        self.last_image_wall = time.time()

    def mark_camera_alive(self):
        """Dat lai nhip tim camera khi vua gan/gan lai sensor."""
        self.last_image_wall = time.time()

    def reset(self):
        """Bo cac frame dang cho ghep.

        Goi khi collector doi sang mot chiec xe khac: nhung frame do dang trong
        `pending` la anh cua camera vua bi huy ghep voi state cua chiec xe cu,
        khong bao gio hoan chinh duoc nua, va neu de lai thi chung se bi dem vao
        `incomplete_evicted` nhu mot dau hieu camera hong.
        """
        with self.lock:
            self.pending.clear()

    def put(self, frame, key, value):
        with self.lock:
            packet = self.pending.setdefault(frame, {})
            packet[key] = value
            if all(name in packet for name in self.required_keys):
                complete = self.pending.pop(frame)
                # Dich han TRUOC khi thu day vao queue: neu queue day thi mau do
                # mat, nhung nhip lay mau van phai giu - "doi" lien tuc dung luc
                # writer dang qua tai chi lam no qua tai them. `on_emit` tra ve
                # False cho packet ghep xong qua sat mau truoc (xem is_duplicate).
                if self.on_emit is None or self.on_emit(complete):
                    try:
                        self.output_queue.put_nowait(complete)
                    except queue.Full:
                        self.dropped += 1
            while len(self.pending) > self.max_pending:
                self.pending.popitem(last=False)
                self.incomplete_evicted += 1
