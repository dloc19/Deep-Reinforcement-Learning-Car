"""A* search over a `router_plan.graph_builder.RouteGraph`, per the cost/heuristic definition
in `data_collection/ASTAR_SCHEMA.md`:

    g(next) = g(current) + edge.cost_m
    h(node) = EuclideanDistance(node, goal)
    f(node) = g(node) + h(node)

Pure stdlib (`heapq`), no CARLA import needed — this module only ever touches the plain
Python data already extracted into `RouteGraph` (node locations, edge costs).
"""

import heapq
import math


def _heuristic(graph, node_id, goal_location):
    loc = graph.location_of(node_id)
    return math.sqrt((loc.x - goal_location.x) ** 2 + (loc.y - goal_location.y) ** 2)


def find_path(graph, start_id, goal_id):
    """Return an ordered list of `node_id` from `start_id` to `goal_id` (both endpoints
    included), or `None` if no path exists — e.g. `start_id`/`goal_id` sit on lanes with no
    legal connecting edge (opposite one-way road with no lane-change permitted between them).
    """
    if start_id not in graph.nodes or goal_id not in graph.nodes:
        raise KeyError("start_id/goal_id phai la node_id da co trong graph (dung graph.nearest_node truoc).")
    if start_id == goal_id:
        return [start_id]

    goal_location = graph.location_of(goal_id)
    g_score = {start_id: 0.0}
    came_from = {}
    open_heap = [(_heuristic(graph, start_id, goal_location), start_id)]
    open_set = {start_id}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue  # stale heap entry — a cheaper path to `current` was already processed
        open_set.discard(current)
        if current == goal_id:
            return _reconstruct(came_from, current)
        closed.add(current)

        for edge in graph.edges.get(current, []):
            if edge.to_id is None or edge.to_id in closed:
                continue
            tentative_g = g_score[current] + edge.cost_m
            if tentative_g < g_score.get(edge.to_id, math.inf):
                g_score[edge.to_id] = tentative_g
                came_from[edge.to_id] = current
                f_score = tentative_g + _heuristic(graph, edge.to_id, goal_location)
                heapq.heappush(open_heap, (f_score, edge.to_id))
                open_set.add(edge.to_id)

    return None


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_edge_types(graph, path):
    """Return the `edge_type` used between `path[i]` and `path[i + 1]` for each consecutive
    pair (length `len(path) - 1`) — needed by `route_tracker.py` to derive `route_command`
    (distinguish a lane-change from a junction turn from a straight lane-follow)."""
    types = []
    for from_id, to_id in zip(path, path[1:]):
        edge_type = "LANE_FOLLOW"
        for edge in graph.edges.get(from_id, []):
            if edge.to_id == to_id:
                edge_type = edge.edge_type
                break
        types.append(edge_type)
    return types
