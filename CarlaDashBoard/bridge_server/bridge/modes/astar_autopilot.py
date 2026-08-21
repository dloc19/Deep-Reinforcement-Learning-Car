"""A* Autopilot — design doc §07: wraps router_plan/drive_to_goal.py's own drive loop
(RouteTracker + RoutePurePursuitController) as a long-running ModeRuntime instead of a
CLI script that spawns its own vehicle and exits when done. The route itself is computed
ahead of time by SetDestination (sim_loop._cmd_SetDestination) — this mode only needs a
RouteContext that already has a route to drive.
"""

from collections import namedtuple

from .base import ModeRuntime

RouteContext = namedtuple("RouteContext", ["planner", "route"])


class AstarAutopilotMode(ModeRuntime):
    name = "ASTAR_AUTOPILOT"

    def __init__(self, cfg, route_context, router_plan):
        self.cfg = cfg
        self.route_context = route_context
        self.router_plan = router_plan
        self.session = None
        self.tracker = None
        self.controller = None
        self.route_state = {}

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

    def tick(self, snapshot):
        vehicle = self.session.ego
        if vehicle is None or not vehicle.is_alive:
            return None
        control = self.controller.compute_control(
            self.route_context.planner.graph, self.route_context.route,
            self.tracker.target_index, vehicle, speed_limit_kmh=vehicle.get_speed_limit())
        self.route_state = self.tracker.update(vehicle.get_transform())
        return control

    def stop(self):
        self.tracker = None
        self.controller = None

    def status_extra(self):
        if not self.route_state:
            return {}
        return {
            "route_command": self.route_state.get("route_command"),
            "route_progress_m": round(self.route_state.get("route_progress_m", 0.0), 1),
            "route_remaining_m": round(self.route_state.get("route_remaining_m", 0.0), 1),
            "route_total_m": round(self.route_state.get("route_total_m", 0.0), 1),
            "route_completed": bool(self.route_state.get("route_completed", 0)),
        }
