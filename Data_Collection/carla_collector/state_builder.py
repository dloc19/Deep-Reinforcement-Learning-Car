"""Build one vehicle/lane/route-ready state dictionary per world snapshot."""

import json
import math

import carla

from .geometry import (
    enum_text, location_distance, magnitude, normalize_angle, utc_now,
    waypoint_id, waypoint_record, world_to_ego)
from .map_export import junction_id


class StateBuilder:
    def __init__(self, world, world_map, ego, args, events, session_id,
                 goal_waypoint=None):
        self.world = world
        self.map = world_map
        self.ego = ego
        self.args = args
        self.events = events
        self.session_id = session_id
        self.goal_waypoint = goal_waypoint
        self.last_location = None
        self.distance_travelled_m = 0.0

    def select_next_waypoint(self, waypoint):
        candidates = waypoint.next(self.args.lookahead_m)
        if not candidates:
            return None, 0
        chosen = min(
            candidates,
            key=lambda item: abs(normalize_angle(
                item.transform.rotation.yaw - waypoint.transform.rotation.yaw)))
        return chosen, len(candidates)

    def lookahead_records(self, waypoint, ego_transform):
        records = []
        for distance in self.args.route_lookaheads:
            candidates = waypoint.next(distance)
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda item: abs(normalize_angle(
                    item.transform.rotation.yaw - waypoint.transform.rotation.yaw)))
            record = waypoint_record(chosen, ego_transform)
            record["lookahead_m"] = distance
            record["candidate_count"] = len(candidates)
            records.append(record)
        return records

    def build(self, world_snapshot):
        actor_snapshot = world_snapshot.find(self.ego.id)
        if actor_snapshot is None:
            return None
        transform = actor_snapshot.get_transform()
        location = transform.location
        velocity = actor_snapshot.get_velocity()
        acceleration = actor_snapshot.get_acceleration()
        angular = actor_snapshot.get_angular_velocity()
        control = self.ego.get_control()
        collisions, invasions = self.events.snapshot()

        if self.last_location is not None:
            self.distance_travelled_m += location_distance(location, self.last_location)
        self.last_location = carla.Location(x=location.x, y=location.y, z=location.z)
        yaw = math.radians(transform.rotation.yaw)
        forward_speed = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)
        lateral_speed = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)
        longitudinal_accel = acceleration.x * math.cos(yaw) + acceleration.y * math.sin(yaw)
        lateral_accel = -acceleration.x * math.sin(yaw) + acceleration.y * math.cos(yaw)
        longitudinal = float(control.throttle) - float(control.brake)

        state = {
            "session_id": self.session_id,
            "episode_id": getattr(self.world, "id", ""),
            "frame": world_snapshot.frame,
            "sim_time_s": world_snapshot.timestamp.elapsed_seconds,
            "delta_seconds": world_snapshot.timestamp.delta_seconds,
            "wall_time_utc": utc_now(),
            "map_name": self.map.name,
            "vehicle_id": self.ego.id,
            "vehicle_type": self.ego.type_id,
            "x": location.x, "y": location.y, "z": location.z,
            "roll_deg": transform.rotation.roll,
            "pitch_deg": transform.rotation.pitch,
            "yaw_deg": transform.rotation.yaw,
            "velocity_x": velocity.x, "velocity_y": velocity.y, "velocity_z": velocity.z,
            "speed_mps": magnitude(velocity), "speed_kmh": 3.6 * magnitude(velocity),
            "forward_speed_mps": forward_speed, "lateral_speed_mps": lateral_speed,
            "distance_travelled_m": self.distance_travelled_m,
            "accel_x": acceleration.x, "accel_y": acceleration.y, "accel_z": acceleration.z,
            "accel_mps2": magnitude(acceleration),
            "longitudinal_accel_mps2": longitudinal_accel,
            "lateral_accel_mps2": lateral_accel,
            "angular_x_deg_s": angular.x, "angular_y_deg_s": angular.y,
            "angular_z_deg_s": angular.z,
            "yaw_rate_rps": math.radians(angular.z),
            "steer": control.steer, "throttle": control.throttle, "brake": control.brake,
            "longitudinal": longitudinal,
            "hand_brake": int(control.hand_brake), "reverse": int(control.reverse),
            "manual_gear_shift": int(control.manual_gear_shift), "gear": control.gear,
            "speed_limit_kmh": self.ego.get_speed_limit(),
            "traffic_light_state": enum_text(self.ego.get_traffic_light_state()),
            "collision_count": collisions, "lane_invasion_count": invasions,
        }
        self._add_lane_state(state, transform, location)
        self._add_goal_state(state, transform, location)
        return state

    def _add_lane_state(self, state, transform, location):
        waypoint = self.map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return
        wp_tf = waypoint.transform
        dx = location.x - wp_tf.location.x
        dy = location.y - wp_tf.location.y
        wp_yaw = math.radians(wp_tf.rotation.yaw)
        lane_offset = dx * (-math.sin(wp_yaw)) + dy * math.cos(wp_yaw)
        half_lane_width = max(float(waypoint.lane_width) * 0.5, 1e-6)
        normalized_lane_offset = lane_offset / half_lane_width
        heading_error_deg = normalize_angle(
            transform.rotation.yaw - wp_tf.rotation.yaw)
        next_wp, candidate_count = self.select_next_waypoint(waypoint)
        wp_local_x, wp_local_y = world_to_ego(wp_tf.location, transform)
        successors = waypoint.next(self.args.graph_resolution)
        state.update({
            "waypoint_id": waypoint_id(waypoint),
            "road_id": waypoint.road_id, "section_id": waypoint.section_id,
            "lane_id": waypoint.lane_id, "waypoint_s": waypoint.s,
            "lane_width_m": waypoint.lane_width,
            "is_junction": int(getattr(
                waypoint, "is_junction", getattr(waypoint, "is_intersection", False))),
            "junction_id": junction_id(waypoint),
            "lane_type": enum_text(waypoint.lane_type),
            "lane_change": enum_text(waypoint.lane_change),
            "lane_offset_m": lane_offset,
            "normalized_lane_offset": normalized_lane_offset,
            "heading_error_deg": heading_error_deg,
            "heading_error_rad": math.radians(heading_error_deg),
            "off_lane": int(abs(lane_offset) > half_lane_width),
            "waypoint_x": wp_tf.location.x, "waypoint_y": wp_tf.location.y,
            "waypoint_z": wp_tf.location.z, "waypoint_yaw_deg": wp_tf.rotation.yaw,
            "waypoint_local_x": wp_local_x, "waypoint_local_y": wp_local_y,
            "next_candidate_count": candidate_count,
            "successor_waypoints_json": json.dumps(
                [waypoint_record(item, transform) for item in successors],
                separators=(",", ":")),
            "lookahead_waypoints_json": json.dumps(
                self.lookahead_records(waypoint, transform), separators=(",", ":")),
        })
        self._add_adjacent_lanes(state, waypoint)
        if next_wp is not None:
            next_tf = next_wp.transform
            next_local_x, next_local_y = world_to_ego(next_tf.location, transform)
            state.update({
                "next_waypoint_x": next_tf.location.x,
                "next_waypoint_y": next_tf.location.y,
                "next_waypoint_z": next_tf.location.z,
                "next_waypoint_yaw_deg": next_tf.rotation.yaw,
                "next_waypoint_id": waypoint_id(next_wp),
                "next_waypoint_road_id": next_wp.road_id,
                "next_waypoint_section_id": next_wp.section_id,
                "next_waypoint_lane_id": next_wp.lane_id,
                "next_waypoint_s": next_wp.s,
                "next_waypoint_local_x": next_local_x,
                "next_waypoint_local_y": next_local_y,
                "next_waypoint_distance_m": location_distance(next_tf.location, location),
            })

    @staticmethod
    def _add_adjacent_lanes(state, waypoint):
        left = waypoint.get_left_lane()
        if left is not None:
            state.update({
                "left_waypoint_id": waypoint_id(left), "left_road_id": left.road_id,
                "left_section_id": left.section_id, "left_lane_id": left.lane_id,
                "left_lane_type": enum_text(left.lane_type),
            })
        right = waypoint.get_right_lane()
        if right is not None:
            state.update({
                "right_waypoint_id": waypoint_id(right), "right_road_id": right.road_id,
                "right_section_id": right.section_id, "right_lane_id": right.lane_id,
                "right_lane_type": enum_text(right.lane_type),
            })

    def _add_goal_state(self, state, transform, location):
        if self.goal_waypoint is None:
            return
        goal_tf = self.goal_waypoint.transform
        goal_local_x, goal_local_y = world_to_ego(goal_tf.location, transform)
        state.update({
            "goal_waypoint_id": waypoint_id(self.goal_waypoint),
            "goal_road_id": self.goal_waypoint.road_id,
            "goal_section_id": self.goal_waypoint.section_id,
            "goal_lane_id": self.goal_waypoint.lane_id,
            "goal_s": self.goal_waypoint.s,
            "goal_x": goal_tf.location.x, "goal_y": goal_tf.location.y,
            "goal_z": goal_tf.location.z, "goal_yaw_deg": goal_tf.rotation.yaw,
            "goal_local_x": goal_local_x, "goal_local_y": goal_local_y,
            "goal_euclidean_distance_m": location_distance(location, goal_tf.location),
        })
