"""Route + DRL Autopilot — lai theo tuyen A* nhung dung policy da hoc de bam lan.

VI SAO PHAI LAI HAI BO DIEU KHIEN, khong phai chi dung policy:

Observation cua checkpoint IL/DRL la `[speed_mps, speed_limit_kmh, traffic_light_*]` + anh
segmentation. KHONG co truong nao mang y dinh di lai — khong `route_command`, khong huong
re. Nghia la o mot nga tu, policy khong the biet nen re trai hay phai; no chi thay mot vung
duong mo ra ba huong. Do khong phai loi huan luyen ma la gioi han cua hop dong quan sat
(xem `drl_training/README.md` muc "Pham vi"), va do duoc truc tiep: chay
`drl_training/demo_il.py` tren Town03, CA BA episode deu chet trong ~13 buoc sau khi xe vao
nga tu, trong khi tren duong thuong no bam lan o 0.109 m.

Nen mode nay chia viec theo dung the manh cua tung ben:

    duong thuong  -> POLICY hoc duoc  (bam lan: 0.109 m tren duong thuong, do that)
    nga tu / doi lan -> PURE-PURSUIT theo tuyen A*  (biet phai re huong nao)

`RouteTracker` chay MOI tick bat ke ai dang lai, vi tien do tuyen va `route_command` la thu
quyet dinh chuyen giao — de no lech mot nhip la chuyen giao sai cho.

Day khong phai giai phap tam: tach "hoach dinh topology" khoi "dieu khien ngang hoc duoc"
la kien truc pho bien trong cac he tu lai that. Muon policy tu di het tuyen thi phai dua
`route_command` vao observation va huan luyen lai tu buoc thu thap du lieu — viec khac han,
khong phai sua o day.
"""

import numpy as np
import carla

from .base import ModeRuntime

# Lenh tuyen buoc phai giao cho pure-pursuit: chung mang Y DINH ma policy khong the doan tu
# anh. "STRAIGHT" khong nam trong danh sach — di thang qua nga tu chinh la thu policy lam
# tot, va giao cho pure-pursuit chi lam mat do muot.
_PLANNER_COMMANDS = ("LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT")

# Giu pure-pursuit them bao nhieu tick sau khi tin hieu nga tu tat. Khong co do tre nay thi
# `is_junction` nhap nhay o ranh gioi lam hai bo dieu khien tranh nhau tung tick, va vo-lang
# giat. 20 tick = 1 giay o sim_fps 20 — du de ra han khoi nga tu roi moi tra lai cho policy.
_HANDOFF_HOLD_TICKS = 20


class RouteLearnedAutopilotMode(ModeRuntime):
    """`predict` va `contract` giong het LearnedAutopilotMode (do
    `il_drl_bridge.build_drl_predictor` dung nen); `route_context` giong het
    AstarAutopilotMode (do `SetDestination` dung nen). Mode nay chi dieu phoi giua hai ben,
    khong tu dung lai phan nao cua chung."""

    def __init__(self, mode_name, cfg, route_context, router_plan, drl_training,
                 contract, predict, action_repeat=1):
        self.name = mode_name
        self.cfg = cfg
        self.route_context = route_context
        self.router_plan = router_plan
        self.drl_training = drl_training
        self.contract = contract
        self.predict = predict
        self.action_repeat = max(1, int(action_repeat))

        self.session = None
        self.tracker = None
        self.controller = None
        self.route_state = {}
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.last_state = {}
        self._hold_control = None
        self._ticks_since_decision = 0
        self._planner_hold = 0
        self._driver = "policy"

    def start(self, session):
        if self.route_context is None or not self.route_context.route:
            raise RuntimeError(
                "Chưa có tuyến đường — chọn điểm đến ở màn hình Route & Map trước (SetDestination).")
        self.session = session
        self.tracker = self.route_context.planner.tracker_for(
            self.route_context.route, target_tolerance_m=self.cfg.astar_route_tolerance_m)
        self.controller = self.router_plan.RoutePurePursuitController(
            target_speed_kmh=self.cfg.astar_target_speed_kmh, dt=1.0 / self.cfg.sim_fps)
        self.route_state = {}
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.last_state = {}
        self._hold_control = None
        self._ticks_since_decision = 0
        self._planner_hold = 0
        self._driver = "policy"

    # ------------------------------------------------------------------ chuyen giao
    def _planner_should_drive(self, state, route_state):
        """Pure-pursuit cam lai khi (a) tuyen yeu cau re/doi lan, hoac (b) xe dang o trong
        nga tu. Dieu kien (b) can rieng vi `route_command` doi ngay tai canh nga tu con xe
        thi con o giua no them mot doan."""
        command = route_state.get("route_command", "LANEFOLLOW")
        if command in _PLANNER_COMMANDS or state.get("is_junction"):
            self._planner_hold = _HANDOFF_HOLD_TICKS
        elif self._planner_hold > 0:
            self._planner_hold -= 1
        return self._planner_hold > 0

    def _policy_control(self, seg):
        """Chay policy o dung nhip `control_dt` cua checkpoint, giu nguyen lenh giua cac
        nhip — giong LearnedAutopilotMode. Tra ve None neu chua den nhip quyet dinh moi."""
        if self._hold_control is not None and self._ticks_since_decision < self.action_repeat:
            self._ticks_since_decision += 1
            return self._hold_control

        vehicle = self.session.ego
        height = self.contract.image_height or seg.shape[0]
        width = self.contract.image_width or seg.shape[1]
        seg_model = self.drl_training.resize_class_map(seg, height, width)
        state = self.drl_training.build_vehicle_state(
            vehicle, self.session.map, seg_model,
            self.previous_steer, self.previous_longitudinal)
        scalar = self.contract.build_scalar_vector(state)
        action = self.predict(seg_model, scalar)

        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        throttle, brake = (longitudinal, 0.0) if longitudinal >= 0.0 else (0.0, -longitudinal)
        self.last_state = state
        self._hold_control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._ticks_since_decision = 1
        return self._hold_control

    # ------------------------------------------------------------------ vong tick
    def tick(self, snapshot):
        vehicle = self.session.ego
        seg = self.session.last_seg_class_map
        if vehicle is None or not vehicle.is_alive:
            return None

        # Tracker chay TRUOC va MOI tick, bat ke ai lai: `target_index` cua no vua la dau
        # vao cua pure-pursuit vua la can cu chuyen giao. Cap nhat le nhip la lai sai cho.
        self.route_state = self.tracker.update(vehicle.get_transform())

        if self.route_state.get("route_completed"):
            self._driver = "arrived"
            self._hold_control = None
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        # `is_junction` lay tu chinh ham dung chung voi DRL env, nen dinh nghia "dang o trong
        # nga tu" o day giong het luc train (drl_training/policy/observation.py).
        state = self.last_state
        if seg is not None:
            height = self.contract.image_height or seg.shape[0]
            width = self.contract.image_width or seg.shape[1]
            state = self.drl_training.build_vehicle_state(
                vehicle, self.session.map, self.drl_training.resize_class_map(seg, height, width),
                self.previous_steer, self.previous_longitudinal)
            self.last_state = state

        if self._planner_should_drive(state or {}, self.route_state):
            self._driver = "planner"
            # Reset nhip policy: khi tra lai quyen lai, policy phai quyet dinh ngay chu khong
            # phat lai mot lenh cu tu truoc khi vao nga tu.
            self._hold_control = None
            self._ticks_since_decision = 0
            control = self.controller.compute_control(
                self.route_context.planner.graph, self.route_context.route,
                self.tracker.target_index, vehicle, speed_limit_kmh=vehicle.get_speed_limit())
        else:
            if seg is None:
                return None          # chua nhan duoc frame segmentation nao
            self._driver = "policy"
            control = self._policy_control(seg)

        if control is not None:
            self.previous_steer = float(control.steer)
            self.previous_longitudinal = float(control.throttle - control.brake)
        return control

    def stop(self):
        self.session = None
        self.tracker = None
        self.controller = None
        self.last_state = {}
        self._hold_control = None

    def status_extra(self):
        extra = {
            # De giao dien hien duoc ai dang cam lai — day cung la thu can quay lai khi lam
            # video demo: nguoi xem phai thay duoc luc nao policy lai, luc nao planner lai.
            "active_controller": self._driver,
        }
        if self.route_state:
            extra.update({
                "route_command": self.route_state.get("route_command"),
                "route_progress_m": round(self.route_state.get("route_progress_m", 0.0), 1),
                "route_remaining_m": round(self.route_state.get("route_remaining_m", 0.0), 1),
                "route_total_m": round(self.route_state.get("route_total_m", 0.0), 1),
                "route_completed": bool(self.route_state.get("route_completed", 0)),
            })
        if self.last_state:
            extra.update({
                "lane_offset_m": round(float(self.last_state.get("lane_offset_m", 0.0)), 3),
                "is_junction": int(bool(self.last_state.get("is_junction", 0))),
            })
        return extra
