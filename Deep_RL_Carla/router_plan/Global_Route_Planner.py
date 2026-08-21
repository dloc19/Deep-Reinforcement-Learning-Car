"""Public entrypoint for `router_plan` — orchestrates `graph_builder` + `astar` +
`goal_selection` + `route_tracker` behind one small API.

This is the file other modules in the repo already refer to by name as the "not implemented
yet" marker for A* routing (`drl_training/README.md`, `docs/manual_train_drl.md`,
`drl_training/envs/carla_lane_keep_env.py`) — it is now implemented. See
`router_plan/README.md` for the full design rationale and
`docs/manual_dieu_huong_astar.md` for a step-by-step usage manual. `drive_to_goal.py` is the
ready-to-run CLI built on top of this class.
"""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc (xem "
        "docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc

from router_plan.astar import find_path
from router_plan.goal_selection import resolve_goal
from router_plan.graph_builder import RouteGraph
from router_plan.route_tracker import RouteTracker


class RouteNotFoundError(RuntimeError):
    """Raised when a start/goal location cannot be snapped onto the graph, or no A* path
    connects them (e.g. opposite one-way lanes with no legal lane-change edge between them)."""


class GlobalRoutePlanner(object):
    """Build a live A* graph for the current CARLA map, resolve a start/goal pair, find the
    shortest route between them, and hand back a `RouteTracker` ready to drive with.

    Usage::

        planner = GlobalRoutePlanner(world.get_map(), resolution_m=2.0, lane_change_cost=3.0)
        route = planner.plan(vehicle.get_location(), goal_waypoint.transform.location)
        tracker = planner.tracker_for(route, target_tolerance_m=3.0)
        # each tick:
        route_state = tracker.update(vehicle.get_transform())

    Building the graph (`generate_waypoints()` + walking `.next()`/`.get_left_lane()`/
    `.get_right_lane()` for every node) takes a few seconds on a full town — build one
    `GlobalRoutePlanner` per map/session and reuse it for `plan()` calls with different goals,
    rather than constructing a new one per trip.
    """

    def __init__(self, world_map, resolution_m=2.0, lane_change_cost=3.0):
        self.world_map = world_map
        self.graph = RouteGraph(resolution_m, lane_change_cost).build(world_map)

    def snap_to_graph(self, location):
        """Return the `node_id` closest to `location` (a `carla.Location`), or `None` if it
        cannot be projected onto a `Driving` lane at all."""
        waypoint = self.world_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return None
        return self.graph.nearest_node(waypoint)

    def plan(self, start_location, goal_location):
        """Return the ordered `node_id` list for the shortest route from `start_location` to
        `goal_location` (both `carla.Location`). Raises `RouteNotFoundError` if either
        endpoint cannot be snapped to the graph, or if A* finds no connecting path."""
        start_id = self.snap_to_graph(start_location)
        goal_id = self.snap_to_graph(goal_location)
        if start_id is None or goal_id is None:
            raise RouteNotFoundError(
                "Khong chieu duoc vi tri bat dau/dich len graph (khong nam gan lane Driving nao). "
                "Thu diem gan duong hon, hoac tang resolution_m.")
        route = find_path(self.graph, start_id, goal_id)
        if route is None:
            raise RouteNotFoundError(
                "A* khong tim thay duong di giua hai vi tri — co the o hai lane/road khong lien "
                "thong (vd duong 1 chieu nguoc huong, khong co lane-change hop le). Thu diem dich khac.")
        return route

    def resolve_goal_waypoint(self, args):
        """Thin re-export of `goal_selection.resolve_goal` so `drive_to_goal.py` only needs
        to import `GlobalRoutePlanner` for the common case."""
        return resolve_goal(self.world_map, args)

    def tracker_for(self, route, target_tolerance_m=3.0):
        return RouteTracker(self.graph, route, target_tolerance_m=target_tolerance_m)
