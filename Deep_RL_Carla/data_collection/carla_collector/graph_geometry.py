"""Shared node/edge-classification primitives for the dense driving-lane graph built from
`world_map.generate_waypoints(resolution)`.

Two independent places build this same kind of graph from the same waypoints and MUST
classify nodes/edges identically, or they silently drift apart:
- `map_export.py::MapArtifacts` — offline exporter, writes `map_nodes.csv`/`map_edges.csv`
  during data collection (labels training data with graph context).
- `router_plan/graph_builder.py::RouteGraph` — in-memory graph actually used to plan and
  drive routes with A* (`router_plan/astar.py`).

Both used to reimplement the driving-lane filter, the nearest-node bisect snap, the
junction-branch heuristic and the same-direction lane-change check independently. Import
these instead of re-deriving them, so a future edge-case fix only has to happen once.
"""

import bisect

from .geometry import enum_text


def driving_lane(waypoint):
    """True if `waypoint` sits on a `Driving`-type lane — the only lane type this graph
    includes (no sidewalks/shoulders/parking)."""
    return waypoint is not None and enum_text(waypoint.lane_type).lower() == "driving"


def is_junction_branch(waypoint, successors):
    """True if `waypoint` sits at a fork/junction. `is_junction` alone under-detects: some
    forks (e.g. a lane about to split into two) are not always flagged as a junction by
    CARLA even though `.next(resolution)` already returns more than one candidate — treat
    that as a branch too."""
    return bool(getattr(
        waypoint, "is_junction", getattr(waypoint, "is_intersection", False))) or len(successors) > 1


def same_direction_lane_change(waypoint, neighbor_waypoint):
    """True if `neighbor_waypoint` (the left/right lane of `waypoint`) runs the SAME
    direction as `waypoint`. CARLA encodes driving direction in the *sign* of `lane_id`
    relative to the road's reference line — opposite sign means the neighbor lane carries
    oncoming traffic, not a valid same-direction lane-change target."""
    return waypoint.lane_id * neighbor_waypoint.lane_id > 0


def build_lane_index(waypoints, value_of=lambda wp: wp):
    """Group `waypoints` by `(road_id, section_id, lane_id)`, each bucket sorted by `.s` —
    the index `nearest_node()` binary-searches. `value_of(wp)` controls what gets stored
    alongside each `.s` (the waypoint itself, a derived node_id string, ...) so callers can
    keep only as much per node as they actually need."""
    index = {}
    for wp in waypoints:
        key = (wp.road_id, wp.section_id, wp.lane_id)
        index.setdefault(key, []).append((wp.s, value_of(wp)))
    for lane_values in index.values():
        lane_values.sort(key=lambda item: item[0])
    return index


def nearest_node(index, waypoint):
    """Snap an arbitrary `waypoint` (from `.next()`, `.get_left_lane()`, a vehicle's current
    position, or a resolved goal) to the closest indexed value on the SAME
    `(road_id, section_id, lane_id)` — `index` built by `build_lane_index()`. Returns `None`
    if that lane isn't indexed (no Driving waypoint there) or `waypoint` is `None`."""
    if waypoint is None:
        return None
    key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
    lane_values = index.get(key)
    if not lane_values:
        return None
    s_values = [item[0] for item in lane_values]
    pos = bisect.bisect_left(s_values, waypoint.s)
    choices = lane_values[max(0, pos - 1):min(len(lane_values), pos + 2)]
    return min(choices, key=lambda item: abs(item[0] - waypoint.s))[1]
