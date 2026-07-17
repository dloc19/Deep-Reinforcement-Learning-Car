"""Small coordinate and waypoint helpers shared by map/state modules."""

import math
from datetime import datetime


def normalize_angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def magnitude(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def utc_now():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def enum_text(value):
    text = str(value)
    return text.split(".")[-1] if "." in text else text


def waypoint_id(waypoint):
    value = getattr(waypoint, "id", None)
    if value is not None:
        return str(value)
    return "%s:%s:%s:%.2f" % (
        waypoint.road_id, waypoint.section_id, waypoint.lane_id, waypoint.s)


def world_to_ego(point, ego_transform):
    """Return (forward, right) metres for a world point in the ego frame."""
    dx = point.x - ego_transform.location.x
    dy = point.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    return (
        dx * math.cos(yaw) + dy * math.sin(yaw),
        -dx * math.sin(yaw) + dy * math.cos(yaw),
    )


def location_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def waypoint_record(waypoint, ego_transform=None):
    tf = waypoint.transform
    record = {
        "waypoint_id": waypoint_id(waypoint),
        "road_id": waypoint.road_id,
        "section_id": waypoint.section_id,
        "lane_id": waypoint.lane_id,
        "s": waypoint.s,
        "x": tf.location.x,
        "y": tf.location.y,
        "z": tf.location.z,
        "yaw_deg": tf.rotation.yaw,
        "is_junction": bool(getattr(
            waypoint, "is_junction", getattr(waypoint, "is_intersection", False))),
    }
    if ego_transform is not None:
        local_x, local_y = world_to_ego(tf.location, ego_transform)
        record.update({
            "local_x": local_x,
            "local_y": local_y,
            "distance_m": location_distance(tf.location, ego_transform.location),
        })
    return record
