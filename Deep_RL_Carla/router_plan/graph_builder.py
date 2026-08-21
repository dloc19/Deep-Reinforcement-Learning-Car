"""Live A* graph builder — builds an in-memory node/edge graph directly from the CARLA map
API each time this module runs (no separate `--map-export` step first, so it always matches
whatever map is currently loaded). Node/edge conventions mirror
`data_collection/carla_collector/map_export.py` (the CSV exporter used during data
collection) so the two representations stay consistent, but this version keeps everything in
memory instead of writing CSV files — see `router_plan/README.md` ("Quyết định đã chốt") for
why this was chosen over reading `map_nodes.csv`/`map_edges.csv`.

Node/edge classification (driving-lane filter, nearest-node snap, junction-branch and
same-direction lane-change detection) lives in `carla_collector.graph_geometry`, shared with
`map_export.py`, instead of being reimplemented here — see that module's docstring for why.
"""

import sys
from collections import namedtuple
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
if str(_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECTION_DIR))

try:
    from carla_collector.geometry import enum_text, location_distance, normalize_angle, waypoint_id
    from carla_collector.graph_geometry import (
        build_lane_index, driving_lane, is_junction_branch, nearest_node, same_direction_lane_change)
except ImportError as exc:
    raise ImportError(
        "Khong import duoc carla_collector tu '%s'. Dam bao thu muc 'data_collection' van "
        "nam canh 'router_plan' trong repo (khong doi ten/di chuyen thu muc goc)." %
        _DATA_COLLECTION_DIR
    ) from exc


# (to_node_id, edge_type, cost_m, distance_m, yaw_delta_deg) — same columns as
# `map_edges.csv` in `carla_collector/map_export.py`, minus the from/road/lane columns which
# the caller already knows (it is looking this edge up from `graph.edges[from_node_id]`).
Edge = namedtuple("Edge", ["to_id", "edge_type", "cost_m", "distance_m", "yaw_delta_deg"])


class RouteGraph(object):
    """In-memory node/edge graph for A*.

    `nodes[node_id]` is a `carla.Transform` (location + rotation only, NOT the full live
    `carla.Waypoint`) — every consumer (`location_of()`, A*, `route_tracker.py`) only ever
    reads position/yaw once `build()` is done, so there is no reason to keep the heavier
    Waypoint object (which also carries OpenDRIVE query machinery: `.road_id`, `.next()`,
    `.get_left_lane()`, ...) resident for the whole drive session — at 2m resolution a full
    town is several thousand nodes.
    `edges[node_id]` is a list of `Edge` namedtuples — a directed adjacency list, exactly the
    same edge types as `data_collection/ASTAR_SCHEMA.md`: `LANE_FOLLOW`, `JUNCTION_BRANCH`,
    `LANE_CHANGE_LEFT`, `LANE_CHANGE_RIGHT`.
    """

    def __init__(self, resolution_m, lane_change_cost):
        if resolution_m <= 0:
            raise ValueError("resolution_m phai > 0")
        if lane_change_cost < 1.0:
            raise ValueError("lane_change_cost nen >= 1 (mac dinh 3.0, xem router_plan/README.md)")
        self.resolution_m = resolution_m
        self.lane_change_cost = lane_change_cost
        self.nodes = {}          # node_id -> carla.Transform (khong giu ca Waypoint)
        self.edges = {}          # node_id -> list[Edge]
        self._lane_index = {}    # (road_id, section_id, lane_id) -> sorted [(s, node_id), ...]

    def location_of(self, node_id):
        return self.nodes[node_id].location

    def build(self, world_map):
        """Populate `nodes`/`edges` from `world_map` (a `carla.Map`, e.g.
        `world.get_world().get_map()`). Safe to call once per `RouteGraph` instance — build a
        new instance if the map changes (e.g. after `client.load_world()`)."""
        waypoints = [wp for wp in world_map.generate_waypoints(self.resolution_m) if driving_lane(wp)]
        if not waypoints:
            raise RuntimeError(
                "generate_waypoints() khong tra ve Driving waypoint nao — map co the chua load xong.")

        for wp in waypoints:
            node_id = waypoint_id(wp)
            self.nodes[node_id] = wp.transform  # chi giu Transform, khong giu ca Waypoint
            self.edges.setdefault(node_id, [])
        self._lane_index = build_lane_index(waypoints, value_of=waypoint_id)

        for wp in waypoints:
            node_id = waypoint_id(wp)
            successors = wp.next(self.resolution_m)
            edge_type = "JUNCTION_BRANCH" if is_junction_branch(wp, successors) else "LANE_FOLLOW"
            for candidate in successors:
                target_id = nearest_node(self._lane_index, candidate)
                self._add_edge(node_id, wp.transform, target_id, edge_type)

            lane_change = enum_text(wp.lane_change).lower()
            left = wp.get_left_lane()
            if lane_change in ("left", "both") and driving_lane(left) and same_direction_lane_change(wp, left):
                self._add_edge(node_id, wp.transform, nearest_node(self._lane_index, left),
                                "LANE_CHANGE_LEFT", self.lane_change_cost)
            right = wp.get_right_lane()
            if lane_change in ("right", "both") and driving_lane(right) and same_direction_lane_change(wp, right):
                self._add_edge(node_id, wp.transform, nearest_node(self._lane_index, right),
                                "LANE_CHANGE_RIGHT", self.lane_change_cost)
        return self

    def _add_edge(self, from_id, from_transform, to_id, edge_type, cost_multiplier=1.0):
        if to_id is None or to_id == from_id:
            return
        existing = self.edges[from_id]
        for edge in existing:
            if edge.to_id == to_id and edge.edge_type == edge_type:
                return  # generate_waypoints()/next() can overlap near junctions — no dupes
        target_transform = self.nodes[to_id]
        distance = location_distance(from_transform.location, target_transform.location)
        yaw_delta = normalize_angle(target_transform.rotation.yaw - from_transform.rotation.yaw)
        existing.append(Edge(to_id, edge_type, distance * cost_multiplier, distance, yaw_delta))

    def nearest_node(self, waypoint):
        """Public snap-to-graph helper — maps an arbitrary `carla.Waypoint` (vehicle
        position, resolved goal) onto a graph node before calling `astar.find_path`."""
        return nearest_node(self._lane_index, waypoint)
