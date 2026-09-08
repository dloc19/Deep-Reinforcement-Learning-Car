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

# Import SAU `graph_builder`: chinh no la cho dat data_collection/ vao sys.path (xem dau
# file do), nen thu tu hai dong nay khong doi cho nhau duoc.
from carla_collector.geometry import location_distance, normalize_angle


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

    # Ban kinh quet node khi biet huong xe, va do lech huong toi da con chap nhan duoc.
    # 8 m du de phu ca lan ke ben tren duong nhieu lan; 60 do du rong de khong loai nham
    # mot doan duong cong, nhung du hep de loai han lan cat ngang trong nga tu (~90 do).
    START_SNAP_RADIUS_M = 8.0
    START_SNAP_MAX_YAW_DIFF_DEG = 60.0

    def snap_to_graph(self, location, heading_deg=None):
        """Return the `node_id` closest to `location` (a `carla.Location`), or `None` if it
        cannot be projected onto a `Driving` lane at all.

        `heading_deg` (yaw cua xe, do) la TUY CHON nhung nen truyen khi diem can chieu la vi
        tri hien tai cua mot chiec xe dang chay. Ly do: `map.get_waypoint(project_to_road)`
        tra ve tam lan GAN NHAT VE KHOANG CACH, khong quan tam lan do di huong nao. Trong
        nga tu — noi nhieu lan chong len nhau — no thuong xuyen chon mot lan CAT NGANG.

        Do that tren Town03: xe o (-84.1, -3.7) yaw -178 do (dang chay theo huong -x), thi
        waypoint chieu duoc co yaw 56 do — lech 127 do so voi xe. Tuyen A* vi vay bat dau
        bang mot lan di huong khac han, con xe thi chay thang tiep; `RouteTracker` chi tien
        khi xe toi gan node muc tieu nen no KET DINH o 0 m, khong loi, khong canh bao:
        dashboard hien "tien do 0 m" trong khi xe chay toi luc dam vao dau do. Tai hien
        3/3 lan trong bo test end-to-end cua CarlaDashBoard khi dat dich luc xe dang chay.

        Khi co `heading_deg`: quet cac node trong ban kinh `START_SNAP_RADIUS_M`, chi giu
        node lech huong duoi `START_SNAP_MAX_YAW_DIFF_DEG` so voi xe, roi lay node gan nhat
        trong so do. Khong co node nao hop huong thi quay ve cach chieu cu (vd xe dang o
        giua bai dat trong, hoac dang do nguoc dau) — van co tuyen de di con hon khong co.
        """
        waypoint = self.world_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        fallback_id = self.graph.nearest_node(waypoint) if waypoint is not None else None
        if heading_deg is None:
            return fallback_id

        best_id, best_distance = None, float("inf")
        for node_id, transform in self.graph.nodes.items():
            distance = location_distance(transform.location, location)
            if distance > self.START_SNAP_RADIUS_M or distance >= best_distance:
                continue
            yaw_diff = abs(normalize_angle(transform.rotation.yaw - heading_deg))
            if yaw_diff > self.START_SNAP_MAX_YAW_DIFF_DEG:
                continue
            best_id, best_distance = node_id, distance
        return best_id if best_id is not None else fallback_id

    def plan(self, start_location, goal_location, start_heading_deg=None):
        """Return the ordered `node_id` list for the shortest route from `start_location` to
        `goal_location` (both `carla.Location`). Raises `RouteNotFoundError` if either
        endpoint cannot be snapped to the graph, or if A* finds no connecting path.

        Truyen `start_heading_deg` = yaw cua xe khi diem xuat phat la mot chiec xe dang chay
        — xem `snap_to_graph()` de biet vi sao no quan trong. Diem DICH khong can (va khong
        nen) co huong: nguoi dung chi mot cho tren ban do, khong chi chieu di toi no.
        """
        start_id = self.snap_to_graph(start_location, heading_deg=start_heading_deg)
        goal_id = self.snap_to_graph(goal_location)
        if start_id is None or goal_id is None:
            raise RouteNotFoundError(
                "Không chiếu được vị trí bắt đầu/đích lên graph (không nằm gần lane Driving nào). "
                "Thử điểm gần đường hơn, hoặc tăng resolution_m.")
        route = find_path(self.graph, start_id, goal_id)
        if route is None:
            raise RouteNotFoundError(
                "A* không tìm thấy đường đi giữa hai vị trí — có thể ở hai lane/road không liên "
                "thông (vd đường 1 chiều ngược hướng, không có lane-change hợp lệ). Thử điểm đích khác.")
        return route

    def resolve_goal_waypoint(self, args):
        """Thin re-export of `goal_selection.resolve_goal` so `drive_to_goal.py` only needs
        to import `GlobalRoutePlanner` for the common case."""
        return resolve_goal(self.world_map, args)

    def tracker_for(self, route, target_tolerance_m=3.0):
        return RouteTracker(self.graph, route, target_tolerance_m=target_tolerance_m)
