#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bo test end-to-end cho Bridge Server — chay THAT voi CarlaUE4 dang mo.

Chay (tu thu muc bridge_server/, bang env carla_rl):

    C:\\Users\\dloc\\miniconda3\\envs\\carla_rl\\python.exe tools\\e2e_test.py

Script tu khoi dong `run_server.py` nhu mot tien trinh con (log ra
`tools/e2e_server.log`), noi vao ca /control lan /stream, chay lan luot cac test case
va in bang ket qua. Exit code != 0 neu co case FAIL.

Vi sao mot script rieng chu khong phai pytest: moi thu can kiem tra o day deu la hanh vi
CUA MOT PHIEN CHAY THAT (xe co chay khong, frame co ve khong, route co tien khong), nen
chung phai chay TUAN TU tren cung mot server va cung mot chiec xe. Do la mot kich ban, khong
phai mot tap test doc lap — dong goi thanh pytest chi them mot lop trung gian.

Co `--quick` de bo qua cac case doi Town (rat cham) khi chi muon kiem tra nhanh.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import websockets

BRIDGE_SERVER_DIR = Path(__file__).resolve().parents[1]
SERVER_LOG = Path(__file__).resolve().parent / "e2e_server.log"

HOST = "127.0.0.1"
PORT = 8799            # khac 8765 de khong dam vao server that dang chay


# --------------------------------------------------------------------------- ket qua
class Results:
    def __init__(self):
        self.rows = []          # (name, status, detail, seconds)

    def add(self, name, status, detail, seconds):
        self.rows.append((name, status, detail, seconds))
        print("[%s] %-38s %5.1fs  %s" % (status.center(4), name, seconds, detail), flush=True)

    def failed(self):
        return [r for r in self.rows if r[1] == "FAIL"]

    def summary(self):
        n_pass = sum(1 for r in self.rows if r[1] == "PASS")
        n_fail = len(self.failed())
        n_skip = sum(1 for r in self.rows if r[1] == "SKIP")
        print("\n" + "=" * 78)
        print("TONG KET: %d PASS, %d FAIL, %d SKIP (tong %d case)"
              % (n_pass, n_fail, n_skip, len(self.rows)))
        for name, status, detail, _ in self.failed():
            print("  FAIL  %-38s %s" % (name, detail))
        print("=" * 78)


RESULTS = Results()


class CaseFailed(Exception):
    pass


def check(condition, message):
    if not condition:
        raise CaseFailed(message)


# --------------------------------------------------------------------------- bus
class Bus:
    """Gom moi thu server gui ve, de cac case truy nguoc lai duoc."""

    def __init__(self):
        self.control = []       # list[dict]
        self.telemetry = []     # list[dict] (text tren /stream)
        self.frames = []        # list[(channel_tag, bytes, timestamp)]
        self.control_ws = None
        self.stream_ws = None

    async def send(self, command):
        await self.control_ws.send(json.dumps(command))

    async def wait_control(self, predicate, timeout, what):
        """Cho mot message tren /control thoa `predicate`. Tra ve message do."""
        start = len(self.control)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for m in self.control[start:]:
                if predicate(m):
                    return m
            await asyncio.sleep(0.05)
        raise CaseFailed("qua %.0fs khong thay %s (control gan nhat: %s)"
                         % (timeout, what, _tail_types(self.control)))

    async def wait_control_type(self, msg_type, timeout=10.0):
        return await self.wait_control(lambda m: m.get("type") == msg_type, timeout, msg_type)

    async def wait_error(self, code, timeout=10.0):
        return await self.wait_control(
            lambda m: m.get("type") == "Error" and m.get("code") == code,
            timeout, "Error(%s)" % code)

    async def wait_telemetry(self, predicate, timeout, what):
        start = len(self.telemetry)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for m in self.telemetry[start:]:
                if predicate(m):
                    return m
            await asyncio.sleep(0.05)
        raise CaseFailed("qua %.0fs khong thay telemetry %s (cuoi cung: %s)"
                         % (timeout, what, json.dumps(self.telemetry[-1] if self.telemetry else {},
                                                      ensure_ascii=False)[:300]))


def _tail_types(messages, n=6):
    return ", ".join(m.get("type", "?") for m in messages[-n:]) or "(rong)"


async def _control_reader(bus):
    try:
        async for message in bus.control_ws:
            try:
                bus.control.append(json.loads(message))
            except ValueError:
                bus.control.append({"type": "?raw", "raw": message[:200]})
    except Exception:
        pass


async def _stream_reader(bus):
    try:
        async for message in bus.stream_ws:
            if isinstance(message, bytes):
                bus.frames.append((message[0], message[1:], time.time()))
            else:
                try:
                    bus.telemetry.append(json.loads(message))
                except ValueError:
                    pass
    except Exception:
        pass


# --------------------------------------------------------------------------- server con
def wait_port(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), 1.0)
            sock.close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def start_server(extra_args):
    log = SERVER_LOG.open("w", encoding="utf-8", errors="replace")
    cmd = [sys.executable, "run_server.py", "--ws-port", str(PORT),
           "--log-level", "INFO"] + list(extra_args)
    print("Khoi dong Bridge Server: %s" % " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(cmd, cwd=str(BRIDGE_SERVER_DIR), stdout=log,
                               stderr=subprocess.STDOUT, env=env)
    return process, log


def server_log_tail(lines=40):
    try:
        text = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(khong doc duoc log)"
    return "\n".join(text.splitlines()[-lines:])


# --------------------------------------------------------------------------- ho tro
def http_get(path, timeout=15.0):
    url = "http://%s:%d%s" % (HOST, PORT, path)
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        try:
            return response.getcode(), response.read()
        finally:
            response.close()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def is_jpeg(data):
    return len(data) > 4 and data[:2] == b"\xff\xd8"


def is_png(data):
    return len(data) > 8 and data[:8] == b"\x89PNG\r\n\x1a\n"


async def run_case(name, coro_factory, results=RESULTS):
    t0 = time.time()
    try:
        detail = await coro_factory()
        results.add(name, "PASS", detail or "", time.time() - t0)
        return True
    except CaseFailed as exc:
        results.add(name, "FAIL", str(exc), time.time() - t0)
    except Exception as exc:                                       # noqa: BLE001
        results.add(name, "FAIL", "%s: %s" % (type(exc).__name__, exc), time.time() - t0)
        traceback.print_exc()
    return False


async def drive_for(bus, seconds):
    """Cho xe chay `seconds` giay, tra ve list telemetry thu duoc trong khoang do."""
    start = len(bus.telemetry)
    await asyncio.sleep(seconds)
    return bus.telemetry[start:]


async def set_mode(bus, mode, timeout=120.0):
    await bus.send({"type": "SetMode", "mode": mode})
    message = await bus.wait_control(
        lambda m: (m.get("type") == "ModeChanged" and m.get("mode") == mode)
        or m.get("type") == "Error",
        timeout, "ModeChanged(%s)" % mode)
    if message.get("type") == "Error":
        raise CaseFailed("SetMode %s loi: %s %s"
                         % (mode, message.get("code"), message.get("message")))
    return message


def max_speed(rows):
    return max([r.get("speed_kmh", 0.0) for r in rows] or [0.0])


# --------------------------------------------------------------------------- cac case
async def case_server_info(bus):
    check(bus.control, "khong nhan duoc message nao tren /control")
    info = bus.control[0]
    check(info.get("type") == "ServerInfo",
          "message dau tien khong phai ServerInfo: %s" % info.get("type"))
    check(info.get("towns"), "ServerInfo thieu 'towns'")
    check(info.get("current_town"), "ServerInfo thieu 'current_town'")
    check(len(info.get("weather_presets", [])) >= 10, "weather_presets qua it")
    check(info.get("spawn_point_count", 0) > 0, "spawn_point_count = 0")
    check(len(info.get("spawn_points", [])) == info["spawn_point_count"],
          "spawn_points khong khop spawn_point_count")
    check(info.get("mode") == "IDLE", "mode khoi dau phai IDLE, nhan %s" % info.get("mode"))
    check(info.get("has_session") is False, "has_session phai False khi chua StartSession")
    return "town=%s, %d spawn point" % (info["current_town"], info["spawn_point_count"])


async def case_maps_endpoint(bus, town):
    code, body = http_get("/maps/%s" % town)
    check(code == 200, "GET /maps/%s tra ve %d: %s" % (town, code, body[:200]))
    payload = json.loads(body.decode("utf-8"))
    check(payload.get("town") == town, "payload.town = %s" % payload.get("town"))
    check(len(payload.get("nodes", [])) > 100, "graph chi co %d node" % len(payload.get("nodes", [])))
    check(len(payload.get("edges", [])) > 100, "graph chi co %d edge" % len(payload.get("edges", [])))
    # Ten truong phai khop DUNG voi Models/MapGraph.cs ben WPF (JsonPropertyName), khong
    # thi minimap ve ra mot dong node o goc toa do ma khong bao loi gi.
    node0 = payload["nodes"][0]
    check(all(k in node0 for k in ("node_id", "x", "y", "z", "yaw_deg")),
          "node thieu truong: %s" % node0)
    edge0 = payload["edges"][0]
    check(all(k in edge0 for k in ("from_node_id", "to_node_id", "edge_type", "cost_m")),
          "edge thieu truong: %s" % edge0)
    return "%d node / %d edge, %.0f KB" % (
        len(payload["nodes"]), len(payload["edges"]), len(body) / 1024.0)


async def case_maps_unknown_town(bus):
    code, body = http_get("/maps/TownKhongTonTai")
    check(code == 503, "town khong ton tai phai tra 503, nhan %d" % code)
    payload = json.loads(body.decode("utf-8"))
    check(payload.get("error") == "MAP_NOT_READY", "body: %s" % payload)
    return "503 MAP_NOT_READY"


async def case_bad_commands(bus):
    await bus.control_ws.send("{khong phai json")
    await bus.wait_error("BAD_JSON", 5.0)

    await bus.send({"khong_co_type": 1})
    await bus.wait_error("BAD_COMMAND", 5.0)

    await bus.send({"type": "LenhKhongCo"})
    await bus.wait_error("UNKNOWN_COMMAND", 5.0)

    await bus.send({"type": "SetMode", "mode": "MODE_BAY_BONG"})
    await bus.wait_error("UNKNOWN_MODE", 5.0)
    return "BAD_JSON / BAD_COMMAND / UNKNOWN_COMMAND / UNKNOWN_MODE"


async def case_no_session_guards(bus):
    await bus.send({"type": "SetMode", "mode": "DATA_COLLECTION"})
    await bus.wait_error("NO_SESSION", 5.0)
    await bus.send({"type": "SetDestination", "spawn_index": 0})
    await bus.wait_error("NO_SESSION", 5.0)
    return "SetMode + SetDestination deu bao NO_SESSION"


async def case_start_session(bus):
    await bus.send({"type": "StartSession", "spawn_index": 0})
    await bus.wait_control_type("SessionStarted", 30.0)
    telemetry = await bus.wait_telemetry(
        lambda m: m.get("ego_alive") is True, 15.0, "ego_alive=true")
    check(telemetry.get("connected") is True, "telemetry.connected phai True")
    check("x" in telemetry and "y" in telemetry, "telemetry thieu toa do")
    return "ego tai (%.1f, %.1f), town=%s" % (telemetry["x"], telemetry["y"], telemetry.get("town"))


async def case_stream_frames(bus, publish_fps):
    start = len(bus.frames)
    t0 = time.time()
    await asyncio.sleep(4.0)
    frames = bus.frames[start:]
    elapsed = time.time() - t0
    rgb = [f for f in frames if f[0] == 0x01]
    seg = [f for f in frames if f[0] == 0x02]
    check(rgb, "khong nhan duoc frame RGB nao")
    check(seg, "khong nhan duoc frame SEG nao")
    check(all(is_jpeg(f[1]) for f in rgb), "co frame RGB khong phai JPEG")
    check(all(is_png(f[1]) for f in seg), "co frame SEG khong phai PNG")
    rgb_fps = len(rgb) / elapsed
    seg_fps = len(seg) / elapsed
    check(rgb_fps > publish_fps * 0.4, "RGB chi %.1f fps (publish_fps=%.0f)" % (rgb_fps, publish_fps))
    check(seg_fps > publish_fps * 0.4, "SEG chi %.1f fps (publish_fps=%.0f)" % (seg_fps, publish_fps))
    check(rgb_fps < publish_fps * 2.0, "RGB %.1f fps — vuot xa publish_fps=%.0f" % (rgb_fps, publish_fps))
    check(seg_fps < publish_fps * 2.0, "SEG %.1f fps — vuot xa publish_fps=%.0f" % (seg_fps, publish_fps))
    return "RGB %.1f fps (%.0f KB/frame), SEG %.1f fps (%.0f KB/frame)" % (
        rgb_fps, sum(len(f[1]) for f in rgb) / len(rgb) / 1024.0,
        seg_fps, sum(len(f[1]) for f in seg) / len(seg) / 1024.0)


async def case_telemetry_rate(bus, publish_fps):
    start = len(bus.telemetry)
    t0 = time.time()
    await asyncio.sleep(3.0)
    rows = bus.telemetry[start:]
    rate = len(rows) / (time.time() - t0)
    check(rate > publish_fps * 0.4, "telemetry chi %.1f Hz (publish_fps=%.0f)" % (rate, publish_fps))
    return "%.1f Hz" % rate


async def case_weather(bus):
    await bus.send({"type": "SetWeather", "preset": "HardRainNoon"})
    await bus.wait_control_type("WeatherChanged", 10.0)
    await bus.send({"type": "SetWeather", "preset": "KhongCoPreset"})
    await bus.wait_error("BAD_WEATHER", 10.0)
    await bus.send({"type": "SetWeather", "preset": "ClearNoon"})
    await bus.wait_control_type("WeatherChanged", 10.0)
    return "HardRainNoon -> BAD_WEATHER -> ClearNoon"


async def case_camera_params(bus):
    await bus.send({"type": "SetCameraParams", "width": 640, "height": 360})
    message = await bus.wait_control_type("CameraParamsChanged", 15.0)
    check(message.get("width") == 640 and message.get("height") == 360,
          "server bao %sx%s" % (message.get("width"), message.get("height")))
    start = len(bus.frames)
    await asyncio.sleep(2.0)
    rgb = [f for f in bus.frames[start:] if f[0] == 0x01]
    seg = [f for f in bus.frames[start:] if f[0] == 0x02]
    check(rgb, "sau SetCameraParams khong con frame RGB nao")
    check(seg, "sau SetCameraParams khong con frame SEG nao")
    await bus.send({"type": "SetCameraParams", "width": 800, "height": 450})
    await bus.wait_control_type("CameraParamsChanged", 15.0)
    await asyncio.sleep(1.0)
    return "640x360 roi tra ve 800x450, stream khong dut"


async def case_record_not_recordable(bus):
    await bus.send({"type": "RecordStart"})
    await bus.wait_error("NOT_RECORDABLE", 5.0)
    return "RecordStart o mode khong ghi duoc -> NOT_RECORDABLE"


async def case_data_collection(bus):
    await set_mode(bus, "DATA_COLLECTION")
    rows = await drive_for(bus, 12.0)
    speed = max_speed(rows)
    check(speed > 1.0, "Traffic Manager khong lai — toc do cao nhat %.2f km/h" % speed)
    last = rows[-1]
    check("collision_count" in last, "telemetry thieu collision_count (status_extra)")
    check(last.get("recording") is False, "recording phai False khi chua bam ghi")
    return "toc do cao nhat %.1f km/h" % speed


async def case_recording(bus, record_dir):
    before = set(p.name for p in record_dir.glob("*")) if record_dir.is_dir() else set()
    await bus.send({"type": "RecordStart"})
    await bus.wait_control_type("RecordingStarted", 10.0)
    await bus.wait_telemetry(lambda m: m.get("recording") is True, 10.0, "recording=true")
    await asyncio.sleep(6.0)
    telemetry = bus.telemetry[-1]
    check(telemetry.get("frames_recorded", 0) > 10,
          "chi ghi duoc %s frame" % telemetry.get("frames_recorded"))
    await bus.send({"type": "RecordStop"})
    await bus.wait_control_type("RecordingStopped", 10.0)

    after = set(p.name for p in record_dir.glob("*"))
    new_dirs = sorted(after - before)
    check(new_dirs, "khong tao thu muc ghi nao trong %s" % record_dir)
    session_dir = record_dir / new_dirs[-1]
    csv_path = session_dir / "states.csv"
    check(csv_path.is_file(), "khong co states.csv trong %s" % session_dir)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    check(len(lines) > 10, "states.csv chi co %d dong" % len(lines))
    check("speed_kmh" in lines[0] and "steer" in lines[0], "header la: %s" % lines[0])
    rgb_files = list((session_dir / "rgb").glob("*.jpg"))
    seg_files = list((session_dir / "seg").glob("*.png"))
    check(rgb_files, "khong luu anh rgb nao")
    check(seg_files, "khong luu anh seg nao")
    check(is_jpeg(rgb_files[0].read_bytes()), "file rgb khong phai JPEG")
    check(is_png(seg_files[0].read_bytes()), "file seg khong phai PNG")
    return "%s: %d dong CSV, %d rgb, %d seg" % (
        session_dir.name, len(lines) - 1, len(rgb_files), len(seg_files))


async def case_set_destination(bus, spawn_index):
    await bus.send({"type": "SetDestination", "spawn_index": spawn_index})
    message = await bus.wait_control(
        lambda m: m.get("type") in ("RouteComputed", "Error"), 60.0, "RouteComputed")
    check(message["type"] == "RouteComputed",
          "SetDestination loi: %s %s" % (message.get("code"), message.get("message")))
    check(message.get("distance_m", 0) > 5, "tuyen chi dai %.1f m" % message.get("distance_m", 0))
    check(len(message.get("polyline", [])) > 2,
          "polyline chi co %d diem" % len(message.get("polyline", [])))
    check(message.get("eta_s", 0) > 0, "eta_s = %s" % message.get("eta_s"))
    return "%.0f m, %d node, ETA %.0fs" % (
        message["distance_m"], message["node_count"], message["eta_s"])


async def case_bad_destination(bus):
    await bus.send({"type": "SetDestination", "spawn_index": 99999})
    await bus.wait_error("BAD_DESTINATION", 10.0)
    await bus.send({"type": "SetDestination"})
    await bus.wait_error("BAD_DESTINATION", 10.0)
    return "spawn_index ngoai khoang + thieu toa do deu bao BAD_DESTINATION"


async def case_astar_drive(bus):
    await set_mode(bus, "ASTAR_AUTOPILOT")
    rows = await drive_for(bus, 20.0)
    have_route = [r for r in rows if "route_progress_m" in r]
    check(have_route, "telemetry khong co route_progress_m (status_extra cua A*)")
    progress = have_route[-1]["route_progress_m"] - have_route[0]["route_progress_m"]
    speed = max_speed(rows)
    check(speed > 3.0, "xe khong chay — toc do cao nhat %.2f km/h" % speed)
    check(progress > 5.0 or have_route[-1].get("route_completed"),
          "tien do tuyen chi tang %.1f m sau 20s" % progress)
    check(have_route[-1].get("route_command"), "thieu route_command")
    return "tien %.0f m, toc do cao nhat %.0f km/h, lenh=%s" % (
        progress, speed, have_route[-1].get("route_command"))


async def case_learned_drive(bus, mode, seconds=20.0):
    await set_mode(bus, mode)
    rows = await drive_for(bus, seconds)
    have_state = [r for r in rows if "lane_offset_m" in r]
    check(have_state, "telemetry khong co lane_offset_m — policy chua chay tick nao")
    speed = max_speed(rows)
    check(speed > 1.0, "xe khong chay — toc do cao nhat %.2f km/h" % speed)
    steers = [abs(r.get("steer", 0.0)) for r in have_state]
    offsets = [abs(r.get("lane_offset_m", 0.0)) for r in have_state]
    check(max(steers) <= 1.0 + 1e-6, "steer vuot khoang [-1,1]: %.3f" % max(steers))
    return "toc do cao nhat %.0f km/h, |lech lan| tb %.2f m (max %.2f), action_repeat=%s" % (
        speed, sum(offsets) / len(offsets), max(offsets), have_state[-1].get("action_repeat"))


def route_trace(rows, every=8):
    """Vet ngan gon de doc duoc ngay trong dong FAIL: xe di dau, ai lai, tien do bao nhieu."""
    picked = rows[::every][:8]
    return " | ".join("(%.0f,%.0f) v=%.0f p=%s %s" % (
        r.get("x", 0), r.get("y", 0), r.get("speed_kmh", 0), r.get("route_progress_m"),
        r.get("active_controller", "")) for r in picked)


async def case_route_drl_drive(bus, dest_spawn_index, seconds=30.0):
    # Bat dau tu mot phien SACH: case truoc do (DRL_AUTOPILOT) bo lai chiec xe o mot cho
    # bat ky — co the dang keo let vao lan can — nen ket qua se khong lap lai duoc.
    await bus.send({"type": "StartSession", "spawn_index": 0})
    await bus.wait_control_type("SessionStarted", 30.0)
    await bus.wait_telemetry(lambda m: m.get("ego_alive") is True, 15.0, "ego_alive=true")
    await case_set_destination(bus, dest_spawn_index)

    await set_mode(bus, "ROUTE_DRL_AUTOPILOT")
    rows = await drive_for(bus, seconds)
    have_state = [r for r in rows if "active_controller" in r]
    check(have_state, "telemetry khong co active_controller")
    drivers = set(r["active_controller"] for r in have_state)
    speed = max_speed(rows)
    check(speed > 1.0, "xe khong chay — toc do cao nhat %.2f km/h" % speed)
    check("policy" in drivers or "planner" in drivers,
          "active_controller chi thay: %s" % drivers)
    with_route = [r for r in have_state if "route_progress_m" in r]
    check(len(with_route) >= 2, "telemetry khong co route_progress_m")
    progress = with_route[-1]["route_progress_m"] - with_route[0]["route_progress_m"]
    check(progress > 5.0 or with_route[-1].get("route_completed"),
          "xe chay (%.0f km/h) nhung tien do tuyen chi tang %.1f m sau %.0fs — dang lai lac "
          "khoi tuyen. Vet: %s" % (speed, progress, seconds, route_trace(with_route)))
    return "toc do cao nhat %.0f km/h, tien %.0f m, dieu khien: %s" % (
        speed, progress, "/".join(sorted(drivers)))


async def case_route_after_delay(bus, spawn_index, delay_s=3.0):
    """Hoi quy: dat dich LUC XE DANG CHAY roi doi vai giay moi bam "Bat dau lai".

    Do dung nhip thao tac cua nguoi that o man hinh Route & Map (chon dich -> nhin tuyen ->
    bam lai). Truoc khi sua, xe da vuot qua nhung node dau tuyen trong khoang tre do va
    RouteTracker ket dinh vinh vien: `route_progress_m` dung o 0.0 m trong khi xe van chay
    thang toi ria ban do — 3/3 lan tai hien duoc, khong loi nao hien ra.
    """
    await bus.send({"type": "StartSession", "spawn_index": 0})
    await bus.wait_control_type("SessionStarted", 30.0)
    await bus.wait_telemetry(lambda m: m.get("ego_alive") is True, 15.0, "ego_alive=true")
    await set_mode(bus, "DRL_AUTOPILOT")
    rows = await drive_for(bus, 8.0)
    check(max_speed(rows) > 5.0, "xe khong chay truoc khi dat dich (%.1f km/h)" % max_speed(rows))

    detail = await case_set_destination(bus, spawn_index)
    await asyncio.sleep(delay_s)                     # nguoi dung con nhin tuyen vai giay
    await set_mode(bus, "ASTAR_AUTOPILOT")
    rows = await drive_for(bus, 15.0)
    with_route = [r for r in rows if "route_progress_m" in r]
    check(with_route, "telemetry khong co route_progress_m")
    progress = with_route[-1]["route_progress_m"] - with_route[0]["route_progress_m"]
    check(progress > 5.0 or with_route[-1].get("route_completed"),
          "dat dich luc dang chay roi doi %.0fs: tien do chi tang %.1f m (xe chay %.0f km/h) "
          "— tracker ket dinh. Vet: %s"
          % (delay_s, progress, max_speed(rows), route_trace(with_route)))
    return "%s; sau %.0fs tre van tien %.0f m" % (detail, delay_s, progress)


async def case_idle_mode(bus):
    await set_mode(bus, "IDLE")
    telemetry = await bus.wait_telemetry(lambda m: m.get("mode") == "IDLE", 10.0, "mode=IDLE")
    check(telemetry.get("mode") == "IDLE",
          "ten mode trong telemetry la '%s'" % telemetry.get("mode"))
    # IDLE phai lam xe DUNG. CARLA giu nguyen lenh dieu khien cuoi cung, nen mot mode
    # "khong lam gi" se de xe chay tiep voi ga cu (do duoc: 52 km/h sau khi vao IDLE).
    stopped = await bus.wait_telemetry(
        lambda m: m.get("speed_kmh", 99.0) < 1.0, 15.0, "xe dung han sau khi vao IDLE")
    return "mode=IDLE, xe dung han (%.2f km/h)" % stopped.get("speed_kmh", 0.0)


async def case_stop_session(bus):
    await bus.send({"type": "StopSession"})
    await bus.wait_control_type("SessionStopped", 20.0)
    await bus.wait_telemetry(lambda m: m.get("ego_alive") is False, 10.0, "ego_alive=false")
    await asyncio.sleep(1.0)
    start = len(bus.frames)
    await asyncio.sleep(1.5)
    extra = bus.frames[start:]
    check(not extra, "van con frame camera sau khi StopSession (%d frame)" % len(extra))
    return "xe + camera da huy, stream ngung frame"


async def case_set_town(bus, town):
    await bus.send({"type": "SetTown", "town": town})
    message = await bus.wait_control(
        lambda m: m.get("type") in ("TownChanged", "Error"), 300.0, "TownChanged(%s)" % town)
    check(message["type"] == "TownChanged", "SetTown loi: %s" % message)
    await bus.wait_telemetry(lambda m: m.get("connected") is True, 30.0, "connected=true")
    code, body = http_get("/maps/%s" % town, timeout=30.0)
    check(code == 200, "/maps/%s sau khi doi town tra ve %d" % (town, code))
    payload = json.loads(body.decode("utf-8"))
    return "%s, graph %d node" % (town, len(payload.get("nodes", [])))


async def case_session_after_town(bus):
    await bus.send({"type": "StartSession"})
    await bus.wait_control_type("SessionStarted", 30.0)
    await bus.wait_telemetry(lambda m: m.get("ego_alive") is True, 20.0, "ego_alive=true")
    await set_mode(bus, "DATA_COLLECTION")
    rows = await drive_for(bus, 10.0)
    speed = max_speed(rows)
    check(speed > 1.0, "sau khi doi town xe khong lai duoc (max %.2f km/h)" % speed)
    return "spawn + DATA_COLLECTION chay lai duoc (%.0f km/h)" % speed


# --------------------------------------------------------------------------- kich ban
async def main_async(args):
    bus = Bus()
    control_task = stream_task = None
    control_ws = await websockets.connect("ws://%s:%d/control" % (HOST, PORT), max_size=None)
    stream_ws = await websockets.connect("ws://%s:%d/stream" % (HOST, PORT), max_size=None)
    bus.control_ws = control_ws
    bus.stream_ws = stream_ws
    try:
        control_task = asyncio.ensure_future(_control_reader(bus))
        stream_task = asyncio.ensure_future(_stream_reader(bus))
        await asyncio.sleep(1.0)          # doi ServerInfo

        record_dir = BRIDGE_SERVER_DIR.parent / "dataset_live"
        town = bus.control[0].get("current_town") if bus.control else "Town03"

        await run_case("ServerInfo khi ket noi", lambda: case_server_info(bus))
        await run_case("GET /maps/{town}", lambda: case_maps_endpoint(bus, town))
        await run_case("GET /maps/{town la}", lambda: case_maps_unknown_town(bus))
        await run_case("Lenh sai dinh dang", lambda: case_bad_commands(bus))
        await run_case("Chan lenh khi chua co phien", lambda: case_no_session_guards(bus))
        await run_case("StartSession", lambda: case_start_session(bus))
        await run_case("Stream RGB + SEG", lambda: case_stream_frames(bus, args.publish_fps))
        await run_case("Nhip telemetry", lambda: case_telemetry_rate(bus, args.publish_fps))
        await run_case("SetWeather", lambda: case_weather(bus))
        await run_case("SetCameraParams", lambda: case_camera_params(bus))
        await run_case("RecordStart khi mode khong ghi", lambda: case_record_not_recordable(bus))
        await run_case("Mode DATA_COLLECTION", lambda: case_data_collection(bus))
        await run_case("Ghi du lieu (CSV + anh)", lambda: case_recording(bus, record_dir))
        await run_case("SetDestination sai", lambda: case_bad_destination(bus))
        await run_case("SetDestination", lambda: case_set_destination(bus, args.dest_spawn_index))
        await run_case("Mode ASTAR_AUTOPILOT", lambda: case_astar_drive(bus))
        await run_case("Mode IL_AUTOPILOT", lambda: case_learned_drive(bus, "IL_AUTOPILOT"))
        await run_case("Mode DRL_AUTOPILOT", lambda: case_learned_drive(bus, "DRL_AUTOPILOT"))
        await run_case("Mode ROUTE_DRL_AUTOPILOT",
                       lambda: case_route_drl_drive(bus, args.dest_spawn_index))
        await run_case("Dat dich luc dang chay (tre 3s)",
                       lambda: case_route_after_delay(bus, args.dest_spawn_index))
        await run_case("Mode IDLE", lambda: case_idle_mode(bus))
        await run_case("StopSession", lambda: case_stop_session(bus))

        if args.quick:
            RESULTS.add("SetTown", "SKIP", "bo qua vi --quick", 0.0)
            RESULTS.add("Phien sau khi doi town", "SKIP", "bo qua vi --quick", 0.0)
        else:
            other_town = "Town02" if town != "Town02" else "Town01"
            ok = await run_case("SetTown", lambda: case_set_town(bus, other_town))
            if ok:
                await run_case("Phien sau khi doi town", lambda: case_session_after_town(bus))
                await run_case("Doi ve town ban dau", lambda: case_set_town(bus, town))
            else:
                RESULTS.add("Phien sau khi doi town", "SKIP", "SetTown that bai", 0.0)
    finally:
        for task in (control_task, stream_task):
            if task is not None:
                task.cancel()
        await control_ws.close()
        await stream_ws.close()


def restore_carla_async_mode(carla_host="127.0.0.1", carla_port=2000):
    """Tra CARLA ve che do bat dong bo sau khi da giet tien trinh server con.

    Can thiet vi `Popen.terminate()` tren Windows la TerminateProcess: khong co
    KeyboardInterrupt, nen khoi `finally` trong `server.run()` (cho goi `sim.stop()`)
    KHONG chay. World se ket thuc o synchronous_mode=True ma khong con ai tick —
    cua so CarlaUE4 dung hinh y nhu bi treo, va lan chay test sau se timeout.
    """
    try:
        import carla
        client = carla.Client(carla_host, carla_port)
        client.set_timeout(10.0)
        world = client.get_world()
        settings = world.get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("Da tra CARLA ve che do bat dong bo.")
    except Exception as exc:                                       # noqa: BLE001
        print("[!] Khong tra duoc CARLA ve bat dong bo: %s" % exc)


def main():
    parser = argparse.ArgumentParser(description="Test end-to-end Bridge Server voi CARLA that")
    parser.add_argument("--quick", action="store_true", help="bo qua cac case doi Town (cham)")
    parser.add_argument("--publish-fps", type=float, default=15.0)
    parser.add_argument("--sim-fps", type=float, default=20.0)
    parser.add_argument("--dest-spawn-index", type=int, default=20)
    parser.add_argument("--server-args", default="", help="co them truyen thang cho run_server.py")
    args = parser.parse_args()

    extra = ["--publish-fps", str(args.publish_fps), "--sim-fps", str(args.sim_fps)]
    if args.server_args:
        extra += args.server_args.split()
    process, log = start_server(extra)
    try:
        if not wait_port(HOST, PORT, 180.0):
            print("Server khong len duoc cong %d. Log:\n%s" % (PORT, server_log_tail(60)))
            return 2
        time.sleep(1.0)
        asyncio.get_event_loop().run_until_complete(main_async(args))
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        restore_carla_async_mode()

    RESULTS.summary()

    text = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    tracebacks = text.count("Traceback (most recent call last)")
    if tracebacks:
        print("\n[!] Log server co %d traceback — xem %s" % (tracebacks, SERVER_LOG))
    print("Log server day du: %s" % SERVER_LOG)
    return 1 if RESULTS.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
