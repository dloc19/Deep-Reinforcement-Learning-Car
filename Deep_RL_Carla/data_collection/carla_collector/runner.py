"""Chay nhieu session lien tiep va tu khoi dong lai khi watchdog nga session.

Ly do ton tai: collector la client THU DONG, no khong lai xe. Xe do
`automatic_control.py` (BehaviorAgent) lai, va agent do ket thuong xuyen -
do dac tren D:/CARLA_DATA_V2 cho thay cac quang dung yen 25 s, 39 s, 42 s,
45 s xen ke nhau trong cung mot session Town03, khong lien quan den den do
(`traffic_light_state = Unknown` suot ca quang). Som muon se co mot quang
vuot `--stationary-timeout-s` va watchdog giet session - dung, nhung the la
mat luon phan du lieu con lai cua buoi thu thap.

Truoc day mot lan watchdog no = het buoi. Gio watchdog chi ket thuc mot
SESSION: runner cho den khi xe chay lai roi mo session moi, va cong don so
mau cho den khi du `--total-samples`. Nho vay muc tieu 10000 mau khong con
doi hoi 2000 giay sim chay lien tuc khong bao gio dung qua 60 s - mot dieu
kien Town01 dat duoc (7% frame dung yen) nhung Town03 thi khong (44-63%).

Day la lop NGOAI CUNG, va gio la lop thu hai chu khong con la lop dau: khi xe
ket, `CarlaCollector.try_rebind` doi sang mot chiec xe khac ngay trong session
dang chay (xem `--rebind-ego`), khong dong session lai. Runner chi vao cuoc khi
ca viec doi xe cung that bai - het gio cho xe moi (`rebind_timeout`), world bi
load lai (`world_reloaded`), hoac loi luc khoi dong session.
"""

import copy
import json
import time
from datetime import datetime
from pathlib import Path

import carla

from .collector import CarlaCollector
from .ego_watch import wait_for_ego
from .geometry import utc_now

# Nhung ly do dung do MOI TRUONG ben ngoai gay ra: session sau van co the tot.
FAULT_REASONS = frozenset((
    "vehicle_stationary_timeout",  # agent phanh khan cap roi khong nha
    "no_samples_timeout",          # world ngung tick / camera im
    "ego_destroyed",               # automatic_control.py respawn xe
    "startup_error",               # chua thay ego, world dang reload, ket noi loi
    "camera_dead",                 # camera im ca sau khi da gan lai
    "rebind_timeout",              # het gio cho xe khac NGAY TRONG session
    "world_reloaded",              # client.load_world(): phai sang session moi
))
# Khong phai loi: session day nhung tong muc tieu chua dat -> sang session ke.
ROLLOVER_REASONS = frozenset(("max_samples",))


def _wait_for_driving_ego(args, timeout_s, should_stop, previous_id=0):
    """Cho den khi co mot ego dung duoc roi moi mo session moi.

    Dieu kien "dung duoc" nam trong `ego_watch.wait_for_ego`: world phai tick VA
    (co mot chiec xe khac `previous_id`, hoac chinh chiec xe cu da chay lai).
    """
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    ego, _ = wait_for_ego(
        client, args, timeout_s, should_stop, previous_id=previous_id,
        message="Cho ego chay lai truoc khi mo session moi (toi da %.0f s)...")
    if ego is None and not should_stop():
        print("Het %.0f s ma ego van chua chay lai. Dung buoi thu thap." % timeout_s)
    return ego is not None


class SessionRunner:
    """Chuoi session cua mot buoi thu thap, cong don ve `--total-samples`."""

    def __init__(self, args):
        self.args = args
        self.collector = None
        self.interrupted = False
        self.total_samples = 0
        self.sessions = []
        self.faults = 0
        self.summary_written = False
        # Id chiec xe cua session vua roi: de lan cho ke tiep nhan ra ngay mot
        # chiec hero MOI thay vi doi dung chiec cu chay lai.
        self.last_ego_id = 0

    def request_stop(self, reason="user_interrupt"):
        """Goi tu signal handler: dung session dang chay VA khong mo session moi."""
        self.interrupted = True
        collector = self.collector
        if collector is not None:
            collector.set_stop_reason(reason)
            collector.stop_event.set()

    def _session_budget(self):
        """`max_samples` cho session sap chay, da tru so mau da thu duoc."""
        if not self.args.total_samples:
            return self.args.max_samples
        remaining = self.args.total_samples - self.total_samples
        if self.args.max_samples:
            return min(self.args.max_samples, remaining)
        return remaining

    def _target_reached(self):
        return (bool(self.args.total_samples)
                and self.total_samples >= self.args.total_samples)

    def _run_one_session(self):
        """Chay tron mot session. Tra ve (stop_reason, so_mau, session_id, so_lan_doi_xe).

        Mot session gio da tu doi sang chiec xe khac khi xe dang thu bi ket
        (`CarlaCollector.try_rebind`), nen ham nay chi tra ve khi ngay ca viec
        doi xe cung khong cuu duoc buoi thu.
        """
        session_args = copy.copy(self.args)
        session_args.max_samples = self._session_budget()
        collector = CarlaCollector(session_args)
        self.collector = collector
        try:
            try:
                collector.run()
            except KeyboardInterrupt:
                collector.set_stop_reason("user_interrupt")
                collector.stop_event.set()
                self.interrupted = True
            except RuntimeError as exc:
                # find_ego het gio, world dang reload, ket noi rot: deu la loi
                # tam thoi cua phia server nen van cho phep thu lai session sau.
                collector.set_stop_reason("startup_error")
                print("LOI khoi dong session: %s" % exc)
            finally:
                collector.cleanup()
        finally:
            self.collector = None
        written = collector.writer.samples if collector.writer is not None else 0
        self.last_ego_id = collector.last_ego_id
        return (collector.stop_reason or "unknown", written, collector.session_id,
                collector.rebinds)

    def run(self):
        # finally: mot loi khong luong truoc van phai de lai tong ket, neu khong
        # nguoi dung chi con mot dong traceback va vai thu muc session roi rac.
        try:
            self._run_sessions()
        finally:
            self.write_run_summary()

    def _run_sessions(self):
        while not self.interrupted:
            if self.args.total_samples and self._session_budget() <= 0:
                break
            reason, written, session_id, rebinds = self._run_one_session()
            self.total_samples += written
            self.sessions.append({
                "session_id": session_id,
                "samples": written,
                "stop_reason": reason,
                "ego_rebinds": rebinds,
            })
            if self.args.total_samples:
                print("Tong cong: %d/%d mau sau %d session."
                      % (self.total_samples, self.args.total_samples,
                         len(self.sessions)))

            if self.interrupted or reason == "user_interrupt":
                break
            if self._target_reached():
                break
            if reason in ROLLOVER_REASONS:
                if not self.args.total_samples:
                    break
                # Session day nhung chua du tong muc tieu: xe van dang chay,
                # khong can cho gi ngoai mot nhip de sensor cu kip go xuong.
                time.sleep(self.args.restart_settle_s)
                continue
            if not self.args.auto_restart:
                break
            if reason not in FAULT_REASONS:
                # duration, writer_error, unknown: khong phai thu khoi dong lai
                # se sua duoc, dung lai de nguoi dung nhin thay.
                print("Ly do dung '%s' khong tu khoi dong lai duoc." % reason)
                break
            self.faults += 1
            if self.args.max_restarts and self.faults > self.args.max_restarts:
                print("Da khoi dong lai %d lan (--max-restarts). Dung buoi thu thap."
                      % self.args.max_restarts)
                break
            print("Session hong (%s). Khoi dong lai lan %d/%s..."
                  % (reason, self.faults, self.args.max_restarts or "vo han"))
            if not _wait_for_driving_ego(self.args, self.args.restart_wait_s,
                                         lambda: self.interrupted,
                                         previous_id=self.last_ego_id):
                break
            time.sleep(self.args.restart_settle_s)

    def write_run_summary(self):
        if not self.sessions or self.summary_written:
            return
        self.summary_written = True
        faulty = [item for item in self.sessions
                  if item["stop_reason"] in FAULT_REASONS]
        summary = {
            "finished_utc": utc_now(),
            "total_samples": self.total_samples,
            "requested_total_samples": self.args.total_samples,
            "target_reached": self._target_reached(),
            "sessions": len(self.sessions),
            "sessions_ended_by_fault": len(faulty),
            "restarts": self.faults,
            # Doi xe TRONG session (khong sinh session moi) va mo session moi la
            # hai lop khac nhau; tach ra de biet lop nao dang phai lam viec.
            "ego_rebinds": sum(item.get("ego_rebinds", 0) for item in self.sessions),
            "session_list": self.sessions,
        }
        print("BUOI THU THAP KET THUC: %d mau / %d session (%d session bi watchdog nga)."
              % (self.total_samples, len(self.sessions), len(faulty)))
        try:
            root = Path(self.args.output).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = root / ("run_%s.json" % stamp)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print("Tong ket buoi thu thap: %s" % path)
        except OSError as exc:
            print("Khong ghi duoc tong ket buoi thu thap: %s" % exc)
