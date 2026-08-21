"""Builds the telemetry JSON envelope published on /stream (design doc §02)."""

import time


def build(session, mode, connected=True):
    ego = session.ego
    if ego is None or not ego.is_alive:
        return {
            "type": "telemetry",
            "t": time.time(),
            "mode": mode.name if mode else "IDLE",
            "connected": connected,
            "ego_alive": False,
        }

    transform = ego.get_transform()
    velocity = ego.get_velocity()
    speed_kmh = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
    control = ego.get_control()

    payload = {
        "type": "telemetry",
        "t": time.time(),
        "mode": mode.name if mode else "IDLE",
        "connected": connected,
        "ego_alive": True,
        "speed_kmh": round(speed_kmh, 2),
        "steer": round(control.steer, 4),
        "throttle": round(control.throttle, 4),
        "brake": round(control.brake, 4),
        "x": round(transform.location.x, 2),
        "y": round(transform.location.y, 2),
        "yaw": round(transform.rotation.yaw, 2),
        "town": session.current_town_short(),
    }
    if mode is not None:
        payload.update(mode.status_extra())
    return payload
