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
    # `spawn_points` doc tu cache ma SIM THREAD dung san (SimLoop._refresh_spawn_points),
    # KHONG goi `map.get_spawn_points()` o day. Ham nay chay tren luong asyncio, va quy tac
    # xuyen suot server la chi mot luong duy nhat cham vao API CARLA (xem docstring cua
    # sim_loop.py, va rui ro §13 "CARLA synchronous mode chi chiu mot client chu dong"):
    # goi tu day vua dua vao mot cuoc dua voi world.tick(), vua chan ca event loop — tuc
    # dung ca /stream cua moi client khac — trong luc cho CARLA tra loi.
    spawn_points = sim.spawn_points_cache
    return protocol.dumps(
        "ServerInfo",
        towns=TOWNS,
        current_town=sim.current_town_cache,
        weather_presets=[{"name": name, "group": group} for name, group in WEATHER_PRESETS],
        vehicle_filter=sim.cfg.vehicle_filter,
        spawn_point_count=len(spawn_points),
        spawn_points=spawn_points,
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
    except websockets.exceptions.ConnectionClosed:
        pass  # xem chu thich o _control_handler
    finally:
        hub.remove_stream_client(websocket)
        logger.info("stream client disconnected (%d total)", len(hub.stream_clients))


async def _control_handler(websocket, path, hub: Hub, sim: SimLoop):
    hub.add_control_client(websocket)
    try:
        await websocket.send(_server_info_payload(sim))
        async for message in websocket:
            _dispatch(message, sim)
    except websockets.exceptions.ConnectionClosed:
        # Client bien mat KHONG phai loi cua server. Truoc day ngoai le nay bay len tan
        # `websockets`, va thu vien in ra mot traceback ~30 dong ("connection handler
        # failed") cho MOI lan mot client rot. Dong CarlaDashBoard.Wpf, chuyen man hinh,
        # hay khoi dong lai server la du de log day traceback trong khi khong co gi hong —
        # chinh la thu lam ban demo trong nhu dang loi rat nang.
        pass
    finally:
        hub.remove_control_client(websocket)
        logger.info("control client disconnected (%d total)", len(hub.control_clients))


def _dispatch(message, sim: SimLoop):
    try:
        command = json.loads(message)
    except json.JSONDecodeError:
        sim.hub.publish_control_text(protocol.error("BAD_JSON", "Không đọc được JSON."))
        return
    if not isinstance(command, dict) or "type" not in command:
        sim.hub.publish_control_text(protocol.error("BAD_COMMAND", "Thiếu trường 'type'."))
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
    # Bind loop TRUOC khi start(): sim thread bat dau publish telemetry ngay tu tick dau,
    # nen neu bind sau thi nhung tick dau tien roi vao khoang trong (hub.loop is None).
    hub.bind_loop(asyncio.get_event_loop())
    sim.start()  # connects to CARLA + starts the sim thread (blocking connect, done once)

    async def handler(websocket, path):
        await _router(websocket, path, hub, sim)

    async def process_request(path, request_headers):
        return await _maps_process_request(path, request_headers, sim)

    try:
        async with websockets.serve(handler, cfg.ws_host, cfg.ws_port, max_size=None,
                                     process_request=process_request):
            logger.info("Bridge Server listening on ws://%s:%d (/stream, /control) "
                        "+ http://%s:%d/maps/{town}", cfg.ws_host, cfg.ws_port, cfg.ws_host, cfg.ws_port)
            await asyncio.Future()  # run forever
    finally:
        # Tra CARLA ve che do bat dong bo va don xe/camera. Khong co doan nay thi tat
        # server (Ctrl+C) de lai world o synchronous_mode=True MA KHONG CON AI TICK:
        # cua so CarlaUE4 dung hinh, trong y het nhu CARLA bi treo, va lan chay sau phai
        # khoi dong lai ca simulator. `SimLoop.stop()` da lo phan khoi phuc settings goc.
        logger.info("Dang dung Bridge Server: tra CARLA ve che do bat dong bo...")
        sim.stop()
