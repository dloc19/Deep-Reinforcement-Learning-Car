"""WebSocket entrypoints — design doc §01/§02: /stream (camera+telemetry, server->client) and
/control (commands + events, both directions). Both paths share one websockets server/port;
routing is by request path, matching the "one Bridge Server" picture in Hình 01.
"""

import asyncio
import http
import json
import logging
from urllib.parse import urlsplit

import websockets

from . import protocol
from .config import TOWNS, WEATHER_PRESETS
from .hub import Hub
from .sim_loop import SimLoop

logger = logging.getLogger("bridge.server")

_JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8"), ("Access-Control-Allow-Origin", "*")]


def _server_info_payload(sim: SimLoop):
    spawn_points = sim.session.map.get_spawn_points() if sim.session.map else []
    return protocol.dumps(
        "ServerInfo",
        towns=TOWNS,
        current_town=sim.session.current_town_short(),
        weather_presets=[{"name": name, "group": group} for name, group in WEATHER_PRESETS],
        vehicle_filter=sim.cfg.vehicle_filter,
        spawn_point_count=len(spawn_points),
        spawn_points=[
            {"index": i, "x": round(t.location.x, 1), "y": round(t.location.y, 1), "yaw": round(t.rotation.yaw, 1)}
            for i, t in enumerate(spawn_points)
        ],
        mode=sim.current_mode.name if sim.current_mode else protocol.MODE_IDLE,
        has_session=sim.session.ego is not None,
    )


async def _maps_process_request(path, request_headers, sim: SimLoop):
    """Phase 3 — GET /maps/{town}: serves the pre-built, pre-cached A* graph as plain JSON
    over HTTP, intercepted before the WebSocket handshake (design doc §01: static/cacheable
    map data doesn't belong on the real-time channels). Returning None here falls through to
    the normal /stream and /control WS routing in `_router`.
    """
    request_path = urlsplit(path).path
    if not request_path.startswith("/maps/"):
        return None
    town = request_path[len("/maps/"):].strip("/")
    body = sim.map_graph_json_cache.get(town)
    if body is None:
        error_body = json.dumps({"error": "MAP_NOT_READY", "town": town}).encode("utf-8")
        return http.HTTPStatus.SERVICE_UNAVAILABLE, _JSON_HEADERS, error_body
    return http.HTTPStatus.OK, _JSON_HEADERS, body


async def _stream_handler(websocket, path, hub: Hub):
    hub.add_stream_client(websocket)
    logger.info("stream client connected (%d total)", len(hub.stream_clients))
    try:
        async for _ in websocket:
            pass  # /stream is server -> client only; ignore anything a client sends here
    finally:
        hub.remove_stream_client(websocket)
        logger.info("stream client disconnected (%d total)", len(hub.stream_clients))


async def _control_handler(websocket, path, hub: Hub, sim: SimLoop):
    hub.add_control_client(websocket)
    await websocket.send(_server_info_payload(sim))
    try:
        async for message in websocket:
            _dispatch(message, sim)
    finally:
        hub.remove_control_client(websocket)


def _dispatch(message, sim: SimLoop):
    try:
        command = json.loads(message)
    except json.JSONDecodeError:
        sim.hub.publish_control_text(protocol.error("BAD_JSON", "Khong parse duoc JSON."))
        return
    if not isinstance(command, dict) or "type" not in command:
        sim.hub.publish_control_text(protocol.error("BAD_COMMAND", "Thieu truong 'type'."))
        return
    sim.submit(command)


async def _router(websocket, path, hub: Hub, sim: SimLoop):
    if path.startswith("/stream"):
        await _stream_handler(websocket, path, hub)
    elif path.startswith("/control"):
        await _control_handler(websocket, path, hub, sim)
    else:
        await websocket.close(code=1008, reason="Unknown path — dung /stream hoac /control")


async def run(cfg):
    hub = Hub()
    sim = SimLoop(cfg, hub)
    sim.start()  # connects to CARLA + starts the sim thread (blocking connect, done once)

    hub.bind_loop(asyncio.get_event_loop())

    async def handler(websocket, path):
        await _router(websocket, path, hub, sim)

    async def process_request(path, request_headers):
        return await _maps_process_request(path, request_headers, sim)

    async with websockets.serve(handler, cfg.ws_host, cfg.ws_port, max_size=None,
                                 process_request=process_request):
        logger.info("Bridge Server listening on ws://%s:%d (/stream, /control) "
                    "+ http://%s:%d/maps/{town}", cfg.ws_host, cfg.ws_port, cfg.ws_host, cfg.ws_port)
        await asyncio.Future()  # run forever
