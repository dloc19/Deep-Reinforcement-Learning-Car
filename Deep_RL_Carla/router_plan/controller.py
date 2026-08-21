"""Self-contained lateral (pure-pursuit) + longitudinal (PID) controller — drives the vehicle
along the sequence of route waypoints produced by `router_plan`, without depending on CARLA's
`agents.navigation` package. That package lives under `PythonAPI/carla/agents/`, a separate
folder from the pip-installed `carla` client egg/wheel used everywhere else in this repo
(`data_collection/`, `drl_training/`) — not guaranteed to be on `sys.path`, so this module is
a small from-scratch controller instead.

This is a geometry-based demo/validation controller for `router_plan`'s A* route, NOT the
project's learned lane-keeping policy. `behavior_cloning/` (Imitation Learning) and
`drl_training/` (PPO/SAC) remain the actual subject of the thesis; this controller exists so
the route the planner computes can be driven and inspected end-to-end before route awareness
is wired into the learned policy (see `router_plan/README.md`, step 4).
"""

import math

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc (xem "
        "docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc


class PIDController(object):
    """Textbook discrete PID — used here for the longitudinal (speed) loop."""

    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class RoutePurePursuitController(object):
    """Pure-pursuit steering + PID throttle/brake, targeting a speed-scaled lookahead point
    walked forward from the route node `RouteTracker` currently reports as
    `route_target_index` — so the steering target always comes from whatever the tracker
    considers "next", not a separately-maintained path index.
    """

    def __init__(self, wheelbase_m=2.85, target_speed_kmh=30.0, min_lookahead_m=3.0,
                 lookahead_gain=0.5, max_steer_deg=70.0, dt=0.05):
        self.wheelbase_m = wheelbase_m
        self.target_speed_kmh = target_speed_kmh
        self.min_lookahead_m = min_lookahead_m
        self.lookahead_gain = lookahead_gain
        self.max_steer_rad = math.radians(max_steer_deg)
        self._speed_pid = PIDController(kp=0.35, ki=0.05, kd=0.05, dt=dt)

    def _lookahead_point(self, graph, route, target_index, speed_mps, vehicle_transform):
        """Walk forward along the route from `target_index` until the speed-scaled lookahead
        distance is covered — keeps the steering target stable at low speed instead of
        jittering onto whatever the very next (possibly too-close) node is."""
        lookahead_m = max(self.min_lookahead_m, self.lookahead_gain * speed_mps)
        vehicle_loc = vehicle_transform.location
        index = target_index
        last_index = len(route) - 1
        while index < last_index and self._distance(vehicle_loc, graph.location_of(route[index])) < lookahead_m:
            index += 1
        return graph.location_of(route[index])

    @staticmethod
    def _distance(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def compute_control(self, graph, route, target_index, vehicle, speed_limit_kmh=None):
        """Return a `carla.VehicleControl` for this tick. `target_index` should be
        `RouteTracker.target_index` (call `tracker.update()` once per tick and reuse its
        `target_index`, so steering and progress tracking never disagree about where "next" is).
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2)

        target_loc = self._lookahead_point(graph, route, target_index, speed_mps, transform)
        dx = target_loc.x - transform.location.x
        dy = target_loc.y - transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

        lookahead_dist = max(math.sqrt(local_x ** 2 + local_y ** 2), 1e-3)
        # Standard pure-pursuit curvature law: kappa = 2*y / L^2, y = lateral offset of the
        # target point in the vehicle frame (right-positive, same convention as
        # `geometry.py::world_to_ego`), L = distance to that target point.
        curvature = 2.0 * local_y / (lookahead_dist ** 2)
        steer_rad = math.atan(curvature * self.wheelbase_m)
        steer_norm = max(-1.0, min(1.0, steer_rad / self.max_steer_rad))

        target_speed_kmh = self.target_speed_kmh
        if speed_limit_kmh:
            target_speed_kmh = min(target_speed_kmh, speed_limit_kmh)
        target_speed_mps = target_speed_kmh / 3.6

        speed_error = target_speed_mps - speed_mps
        throttle_brake = self._speed_pid.step(speed_error)
        throttle = max(0.0, min(1.0, throttle_brake))
        brake = max(0.0, min(1.0, -throttle_brake))

        return carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake)
