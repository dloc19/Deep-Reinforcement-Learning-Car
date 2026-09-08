"""Route progress tracker — walks a resolved A* route (the ordered `node_id` list returned
by `router_plan.astar.find_path`) and, on every `update()` call, emits exactly the `route_*`
fields already reserved for this purpose in
`data_collection/carla_collector/schema.py::ROUTE_FIELDS` and defined in
`data_collection/ASTAR_SCHEMA.md` ("Planner output contract").
"""

import sys
import uuid
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
if str(_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECTION_DIR))

try:
    from carla_collector.geometry import location_distance, normalize_angle, world_to_ego
except ImportError as exc:
    raise ImportError(
        "Khong import duoc carla_collector tu '%s'. Dam bao thu muc 'data_collection' van "
        "nam canh 'router_plan' trong repo (khong doi ten/di chuyen thu muc goc)." %
        _DATA_COLLECTION_DIR
    ) from exc

from .astar import path_edge_types

# Nguong (do) de phan biet re trai/phai voi di thang tai JUNCTION_BRANCH.
TURN_YAW_THRESHOLD_DEG = 20.0


def _route_command(edge_type, yaw_delta_deg):
    if edge_type == "LANE_CHANGE_LEFT":
        return "CHANGELANELEFT"
    if edge_type == "LANE_CHANGE_RIGHT":
        return "CHANGELANERIGHT"
    if edge_type == "JUNCTION_BRANCH":
        # Quy uoc dau: yaw tang = re PHAI. Suy ra truc tiep tu `world_to_ego()` trong
        # geometry.py (da dung xuyen suot repo, vd lane_offset_m: "+= phai, -= trai"):
        # forward = (cos(yaw), sin(yaw)), right = (-sin(yaw), cos(yaw)). Tai yaw=0, forward
        # huong +X; yaw tang mot luong nho epsilon dua forward ve phia (cos(eps), sin(eps))
        # ~ (1, eps), tuc la dich chuyen ve phia +Y — cung huong voi vector "right" o yaw=0
        # ((-sin0, cos0) = (0,1) = +Y). Vay yaw tang = xoay forward ve phia "right" = re phai.
        if yaw_delta_deg > TURN_YAW_THRESHOLD_DEG:
            return "RIGHT"
        if yaw_delta_deg < -TURN_YAW_THRESHOLD_DEG:
            return "LEFT"
        return "STRAIGHT"
    return "LANEFOLLOW"


class RouteTracker(object):
    """Track vehicle progress along a resolved route and emit `ROUTE_FIELDS`-shaped dicts.

    `graph` is the `RouteGraph` the route was planned on. `route` is the ordered `node_id`
    list from `astar.find_path`. `target_tolerance_m` is both (a) how close the vehicle must
    get to a route node before the tracker advances its target to the next node, and (b) the
    arrival tolerance used for `route_completed`.
    """

    def __init__(self, graph, route, target_tolerance_m=3.0):
        if not route:
            raise ValueError("route rong — khong the tao RouteTracker")
        self.graph = graph
        self.route = route
        self.tolerance_m = target_tolerance_m
        self.route_id = uuid.uuid4().hex[:12]
        self.edge_types = path_edge_types(graph, route) if len(route) > 1 else []

        self._locations = [graph.location_of(node_id) for node_id in route]
        # graph.nodes[node_id] la carla.Transform (khong phai Waypoint day du — xem
        # RouteGraph.nodes trong graph_builder.py), nen doc truc tiep .rotation.yaw.
        self._yaw_deg = [graph.nodes[node_id].rotation.yaw for node_id in route]
        self._segment_len = [
            location_distance(self._locations[i], self._locations[i + 1])
            for i in range(len(route) - 1)
        ]
        self._cum_dist = [0.0]
        for seg in self._segment_len:
            self._cum_dist.append(self._cum_dist[-1] + seg)
        self.total_m = self._cum_dist[-1]

        self.target_index = 1 if len(route) > 1 else 0
        self.completed = len(route) == 1  # a route that is just the start node is already "there"

    def resync_to(self, vehicle_transform):
        """Dat lai node muc tieu ve node dau tien NAM PHIA TRUOC xe, thay vi luon la node 1.

        Goi mot lan ngay truoc khi bat dau lai, khi tuyen co the da duoc tinh tu truoc do
        mot luc. `update()` chi tien muc tieu khi xe DEN GAN node hien tai (duoi
        `tolerance_m`, mac dinh 3 m); neu xe da vuot qua node do roi thi dieu kien do khong
        bao gio con dung nua, va tracker KET DINH VINH VIEN o node 1: `route_progress_m`
        dung yen o 0 trong khi xe van chay, con pure-pursuit thi bam mot node o phia sau.

        Do that tren Town03 (CarlaDashBoard, bo test end-to-end): dat dich luc xe dang chay
        28 km/h roi bam "Bat dau lai" sau 2 giay — xe da di 16.4 m, tuc vuot node muc tieu
        13 m — tien do dung o 0.0 m suot 12 giay va xe chay thang toi ria ban do. Voi do tre
        ~0 giay thi khong sao (tien 93 m trong cung 12 giay), nen loi chi lo ra khi nguoi
        dung thao tac o toc do nguoi that: chon dich, nhin tuyen, roi moi bam lai.

        Chi bao gio TIEN VE PHIA TRUOC, khong bao gio lui: mot tuyen di vong lai gan chinh
        no (rat hay gap trong pho o Town03) co the co node cu nam gan xe hon node sap toi,
        va lui lai la tu xoa tien do da di duoc.
        """
        loc = vehicle_transform.location
        last_index = len(self.route) - 1
        nearest_index = min(range(len(self._locations)),
                            key=lambda i: location_distance(loc, self._locations[i]))
        # Node gan nhat coi nhu DA di qua — muc tieu la node ke tiep.
        self.target_index = max(self.target_index, min(nearest_index + 1, last_index))
        if (self.target_index == last_index and
                location_distance(loc, self._locations[last_index]) < self.tolerance_m):
            self.completed = True

    def update(self, vehicle_transform):
        """Call once per tick with the vehicle's current `carla.Transform`. Returns a dict
        with exactly the keys in `carla_collector.schema.ROUTE_FIELDS`."""
        loc = vehicle_transform.location
        last_index = len(self.route) - 1
        old_target_index = self.target_index

        while (self.target_index < last_index and
               location_distance(loc, self._locations[self.target_index]) < self.tolerance_m):
            self.target_index += 1

        if (self.target_index == last_index and
                location_distance(loc, self._locations[last_index]) < self.tolerance_m):
            self.completed = True

        passed_index = max(0, self.target_index - 1)
        progress_m = self._cum_dist[passed_index]
        remaining_m = max(self.total_m - progress_m, 0.0)

        target_loc = self._locations[self.target_index]
        local_x, local_y = world_to_ego(target_loc, vehicle_transform)

        # target_tolerance_m (mac dinh 3.0m) co the lon hon khoang cach giua cac node tren
        # graph (mac dinh graph_resolution=2.0m) — vong lap tren co the nhay qua 2+ node
        # trong DUNG MOT lan goi update(), thuong gap nhat o khuc cua gap (JUNCTION_BRANCH).
        # Neu chi xet canh ngay truoc target_index MOI (nhu ban cu), canh JUNCTION_BRANCH
        # nam giua khoang bi nhay qua se bi bo sot hoan toan -> mat lenh LEFT/RIGHT. Quet lai
        # TOAN BO cac canh da "di qua" ke tu lan update() truoc (tu old_target_index-1 den
        # passed_index) va uu tien tin hieu quan trong nhat: JUNCTION_BRANCH > LANE_CHANGE_* >
        # LANEFOLLOW mac dinh.
        first_skipped = max(0, old_target_index - 1)
        edge_type = "LANE_FOLLOW"
        for i in range(first_skipped, passed_index + 1):
            if i >= len(self.edge_types):
                continue
            candidate = self.edge_types[i]
            if candidate == "JUNCTION_BRANCH":
                edge_type = candidate
                break
            if candidate in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT") and edge_type == "LANE_FOLLOW":
                edge_type = candidate

        if edge_type == "JUNCTION_BRANCH":
            # Tinh yaw_delta tren TOAN BO khoang bi nhay qua (first_skipped -> target_index
            # moi), khong chi 1 canh cuoi cung — dam bao dung voi muc do re thuc te du bi
            # nhay qua bao nhieu node.
            yaw_delta = normalize_angle(self._yaw_deg[self.target_index] - self._yaw_deg[first_skipped])
        else:
            yaw_delta = 0.0

        return {
            "route_id": self.route_id,
            "route_target_index": self.target_index,
            "route_target_waypoint_id": self.route[self.target_index],
            "route_target_local_x": local_x,
            "route_target_local_y": local_y,
            "route_command": _route_command(edge_type, yaw_delta),
            "route_progress_m": progress_m,
            "route_remaining_m": remaining_m,
            "route_total_m": self.total_m,
            "route_completed": int(self.completed),
        }
