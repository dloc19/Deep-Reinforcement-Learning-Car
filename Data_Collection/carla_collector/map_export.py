"""OpenDRIVE, spawn-point and dense waypoint-graph export for future A*."""

import bisect
import csv
import json

import carla

from .geometry import (
    enum_text, location_distance, normalize_angle, waypoint_id, waypoint_record)


def driving_lane(waypoint):
    return waypoint is not None and enum_text(waypoint.lane_type).lower() == "driving"


def junction_id(waypoint):
    value = getattr(waypoint, "junction_id", "")
    if getattr(waypoint, "is_junction", getattr(waypoint, "is_intersection", False)):
        try:
            junction = waypoint.get_junction()
            value = getattr(junction, "id", value) if junction else value
        except AttributeError:
            pass
    return value


class MapArtifacts:
    def __init__(self, world_map, args, session_dir):
        self.map = world_map
        self.args = args
        self.session_dir = session_dir

    def resolve_goal(self):
        """Resolve an optional destination; A* itself is intentionally separate."""
        spawn_points = self.map.get_spawn_points()
        if self.args.goal_spawn_index >= 0:
            if self.args.goal_spawn_index >= len(spawn_points):
                raise ValueError(
                    "--goal-spawn-index=%d khong hop le; map co %d spawn points" %
                    (self.args.goal_spawn_index, len(spawn_points)))
            location = spawn_points[self.args.goal_spawn_index].location
        elif self.args.goal_x is not None and self.args.goal_y is not None:
            location = carla.Location(
                x=self.args.goal_x, y=self.args.goal_y, z=self.args.goal_z)
        else:
            return None

        goal_waypoint = self.map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if goal_waypoint is None:
            raise RuntimeError("Khong chieu duoc diem dich len Driving waypoint")
        goal = waypoint_record(goal_waypoint)
        with (self.session_dir / "goal.json").open("w", encoding="utf-8") as handle:
            json.dump(goal, handle, indent=2, ensure_ascii=False)
        print("Dich A* da chon: waypoint=%s, road=%s, lane=%s" % (
            goal["waypoint_id"], goal["road_id"], goal["lane_id"]))
        return goal_waypoint

    @staticmethod
    def nearest_graph_node(index, waypoint):
        if waypoint is None:
            return None
        key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
        lane_nodes = index.get(key)
        if not lane_nodes:
            return None
        s_values = [item[0] for item in lane_nodes]
        pos = bisect.bisect_left(s_values, waypoint.s)
        choices = lane_nodes[max(0, pos - 1):min(len(lane_nodes), pos + 2)]
        return min(choices, key=lambda item: abs(item[0] - waypoint.s))[1]

    def export(self):
        if self.args.no_map_export:
            return {"exported": False}
        print("Dang xuat OpenDRIVE va graph A* (chi mot lan cho session)...")
        (self.session_dir / "map.xodr").write_text(
            self.map.to_opendrive(), encoding="utf-8")
        self._write_spawn_points()

        nodes = [wp for wp in self.map.generate_waypoints(self.args.graph_resolution)
                 if driving_lane(wp)]
        node_index = self._build_node_index(nodes)
        self._write_nodes(nodes, node_index)
        edge_count = self._write_edges(nodes, node_index)
        stats = {
            "exported": True,
            "resolution_m": self.args.graph_resolution,
            "node_count": len(nodes),
            "edge_count": edge_count,
            "lane_change_cost_multiplier": self.args.lane_change_cost,
            "files": ["map.xodr", "spawn_points.csv", "map_nodes.csv", "map_edges.csv"],
        }
        with (self.session_dir / "map_graph_metadata.json").open(
                "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print("Graph A*: %d nodes, %d edges" % (len(nodes), edge_count))
        return stats

    def _write_spawn_points(self):
        fields = [
            "spawn_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
            "waypoint_id", "road_id", "section_id", "lane_id", "s"]
        with (self.session_dir / "spawn_points.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, transform in enumerate(self.map.get_spawn_points()):
                wp = self.map.get_waypoint(
                    transform.location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                row = {
                    "spawn_index": index,
                    "x": transform.location.x, "y": transform.location.y,
                    "z": transform.location.z,
                    "roll_deg": transform.rotation.roll,
                    "pitch_deg": transform.rotation.pitch,
                    "yaw_deg": transform.rotation.yaw,
                }
                if wp is not None:
                    row.update({
                        "waypoint_id": waypoint_id(wp), "road_id": wp.road_id,
                        "section_id": wp.section_id, "lane_id": wp.lane_id, "s": wp.s})
                writer.writerow(row)

    @staticmethod
    def _build_node_index(nodes):
        index = {}
        for wp in nodes:
            key = (wp.road_id, wp.section_id, wp.lane_id)
            index.setdefault(key, []).append((wp.s, wp))
        for lane_nodes in index.values():
            lane_nodes.sort(key=lambda item: item[0])
        return index

    def _write_nodes(self, nodes, node_index):
        fields = [
            "node_id", "road_id", "section_id", "lane_id", "s", "x", "y", "z",
            "yaw_deg", "lane_width_m", "is_junction", "junction_id", "lane_change",
            "left_node_id", "right_node_id"]
        with (self.session_dir / "map_nodes.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for wp in nodes:
                tf = wp.transform
                left_lane = wp.get_left_lane()
                right_lane = wp.get_right_lane()
                left = self.nearest_graph_node(node_index, left_lane) \
                    if driving_lane(left_lane) else None
                right = self.nearest_graph_node(node_index, right_lane) \
                    if driving_lane(right_lane) else None
                writer.writerow({
                    "node_id": waypoint_id(wp), "road_id": wp.road_id,
                    "section_id": wp.section_id, "lane_id": wp.lane_id, "s": wp.s,
                    "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                    "yaw_deg": tf.rotation.yaw, "lane_width_m": wp.lane_width,
                    "is_junction": int(getattr(
                        wp, "is_junction", getattr(wp, "is_intersection", False))),
                    "junction_id": junction_id(wp), "lane_change": enum_text(wp.lane_change),
                    "left_node_id": waypoint_id(left) if left else "",
                    "right_node_id": waypoint_id(right) if right else "",
                })

    def _write_edges(self, nodes, node_index):
        fields = [
            "from_node_id", "to_node_id", "edge_type", "distance_m", "cost_m",
            "yaw_delta_deg", "from_road_id", "to_road_id", "from_lane_id", "to_lane_id"]
        edge_keys = set()
        edge_count = 0
        with (self.session_dir / "map_edges.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

            def add_edge(source, target, edge_type, cost_multiplier=1.0):
                nonlocal edge_count
                if target is None:
                    return
                key = (waypoint_id(source), waypoint_id(target), edge_type)
                if key in edge_keys or key[0] == key[1]:
                    return
                edge_keys.add(key)
                distance = location_distance(
                    source.transform.location, target.transform.location)
                writer.writerow({
                    "from_node_id": key[0], "to_node_id": key[1], "edge_type": edge_type,
                    "distance_m": distance, "cost_m": distance * cost_multiplier,
                    "yaw_delta_deg": normalize_angle(
                        target.transform.rotation.yaw - source.transform.rotation.yaw),
                    "from_road_id": source.road_id, "to_road_id": target.road_id,
                    "from_lane_id": source.lane_id, "to_lane_id": target.lane_id,
                })
                edge_count += 1

            for wp in nodes:
                successors = wp.next(self.args.graph_resolution)
                for candidate in successors:
                    target = self.nearest_graph_node(node_index, candidate)
                    is_branch = (
                        getattr(wp, "is_junction", getattr(wp, "is_intersection", False)) or
                        len(successors) > 1)
                    add_edge(wp, target, "JUNCTION_BRANCH" if is_branch else "LANE_FOLLOW")

                lane_change = enum_text(wp.lane_change).lower()
                left = wp.get_left_lane()
                if (lane_change in ("left", "both") and driving_lane(left) and
                        wp.lane_id * left.lane_id > 0):
                    add_edge(wp, self.nearest_graph_node(node_index, left),
                             "LANE_CHANGE_LEFT", self.args.lane_change_cost)
                right = wp.get_right_lane()
                if (lane_change in ("right", "both") and driving_lane(right) and
                        wp.lane_id * right.lane_id > 0):
                    add_edge(wp, self.nearest_graph_node(node_index, right),
                             "LANE_CHANGE_RIGHT", self.args.lane_change_cost)
        return edge_count
