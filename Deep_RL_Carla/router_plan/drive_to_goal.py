#!/usr/bin/env python3
"""Entrypoint: pick a destination, compute the shortest A* route from the vehicle's current
position, and drive along it with a self-contained pure-pursuit/PID controller.

Usage (see docs/manual_dieu_huong_astar.md for the full walkthrough):
    Terminal 1: CarlaUE4.exe -quality-level=Low
    Terminal 2 (optional): load a Town + weather, same as data_collection's workflow
    Terminal 3:
        python drive_to_goal.py --goal-interactive
        python drive_to_goal.py --goal-spawn-index 42
        python drive_to_goal.py --goal-x 120.5 --goal-y -34.2 --draw-debug

ACTIVE client: spawns its own vehicle and drives the simulation in synchronous mode. Do NOT
run this alongside `data_collection/` (passive collector) or `automatic_control.py` against
the same CARLA world — both would fight over vehicle control and world settings (see
`docs/manual_thu_thap_du_lieu.md` and `drl_training/README.md` for why).
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
for _path in (str(_REPO_ROOT), str(_DATA_COLLECTION_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc (xem "
        "docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc

from carla_collector.geometry import magnitude  # noqa: E402
from carla_collector.schema import ROUTE_FIELDS  # noqa: E402

from router_plan.Global_Route_Planner import GlobalRoutePlanner, RouteNotFoundError  # noqa: E402
from router_plan.controller import RoutePurePursuitController  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chon diem den, tinh duong di ngan nhat (A*), va lai xe theo lo trinh do.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=20.0, help="Tan so tick dong bo (Hz)")
    parser.add_argument("--vehicle-filter", default="vehicle.lincoln.mkz2017")
    parser.add_argument("--graph-resolution", type=float, default=2.0,
                         help="m/node cho graph A* — giong y nghia voi data_collection --graph-resolution")
    parser.add_argument("--lane-change-cost", type=float, default=3.0,
                         help="He so nhan chi phi khi A* chon doi lan")
    parser.add_argument("--route-tolerance-m", type=float, default=3.0,
                         help="Ban kinh (m) coi la 'da toi' mot node route / diem dich")
    parser.add_argument("--target-speed-kmh", type=float, default=30.0)
    parser.add_argument("--max-duration-s", type=float, default=600.0,
                         help="Tu dung neu chua toi dich sau tung nay giay")
    parser.add_argument("--draw-debug", action="store_true",
                         help="Ve duong route mau xanh + diem dich trong CARLA de quan sat")
    parser.add_argument("--output", default=None,
                         help="File CSV ghi lai route_* + vi tri/toc do xe moi tick (tuy chon)")

    goal = parser.add_argument_group("destination (chon dung 1 trong 3 cach)")
    goal.add_argument("--goal-spawn-index", type=int, default=-1)
    goal.add_argument("--goal-x", type=float, default=None)
    goal.add_argument("--goal-y", type=float, default=None)
    goal.add_argument("--goal-z", type=float, default=0.0)
    goal.add_argument("--goal-interactive", action="store_true",
                       help="Liet ke spawn points va hoi index qua console")

    start = parser.add_argument_group("start position")
    start.add_argument("--start-spawn-index", type=int, default=-1, help="-1 = ngau nhien")
    args = parser.parse_args()

    if (args.goal_x is None) != (args.goal_y is None):
        parser.error("Phai truyen dong thoi --goal-x va --goal-y")
    if args.goal_spawn_index >= 0 and args.goal_x is not None:
        parser.error("Chi chon mot trong: --goal-spawn-index hoac --goal-x/--goal-y")
    if args.goal_spawn_index < 0 and args.goal_x is None and not args.goal_interactive:
        parser.error("Phai chon diem den: --goal-spawn-index, --goal-x/--goal-y, hoac --goal-interactive")
    return args


def _draw_route(world, planner, route, goal_id):
    for from_id, to_id in zip(route, route[1:]):
        world.debug.draw_line(
            planner.graph.location_of(from_id) + carla.Location(z=0.5),
            planner.graph.location_of(to_id) + carla.Location(z=0.5),
            thickness=0.15, color=carla.Color(0, 255, 0), life_time=0.0)
    world.debug.draw_string(
        planner.graph.location_of(goal_id) + carla.Location(z=2.0), "GOAL",
        color=carla.Color(255, 0, 0), life_time=0.0)


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_map = world.get_map()

    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    print("Dang build graph A* (resolution=%.1fm)..." % args.graph_resolution)
    t0 = time.time()
    planner = GlobalRoutePlanner(world_map, args.graph_resolution, args.lane_change_cost)
    print("Graph: %d node, xong sau %.1fs" % (len(planner.graph.nodes), time.time() - t0))

    goal_waypoint = planner.resolve_goal_waypoint(args)
    if goal_waypoint is None:
        raise SystemExit(
            "Chua chon diem den — dung --goal-spawn-index, --goal-x/--goal-y, hoac --goal-interactive.")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(args.vehicle_filter)[0]
    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "astar_demo")

    spawn_points = world_map.get_spawn_points()
    if args.start_spawn_index >= 0:
        if args.start_spawn_index >= len(spawn_points):
            raise SystemExit("--start-spawn-index=%d khong hop le; map co %d spawn points" %
                              (args.start_spawn_index, len(spawn_points)))
        start_transform = spawn_points[args.start_spawn_index]
    else:
        start_transform = random.choice(spawn_points)

    vehicle = None
    csv_handle = None
    try:
        vehicle = world.try_spawn_actor(vehicle_bp, start_transform)
        if vehicle is None:
            raise RuntimeError(
                "Khong spawn duoc xe tai spawn point da chon (co the dang bi chiem boi xe khac).")
        world.tick()

        try:
            route = planner.plan(vehicle.get_location(), goal_waypoint.transform.location)
        except RouteNotFoundError as exc:
            raise SystemExit(str(exc))
        print("Tim thay duong A*: %d node." % len(route))

        if len(route) == 1:
            print("Vi tri xuat phat da o trong pham vi diem den (%.1fm) — khong can di chuyen." %
                  args.route_tolerance_m)
            return

        goal_id = planner.snap_to_graph(goal_waypoint.transform.location)
        if args.draw_debug:
            _draw_route(world, planner, route, goal_id)

        tracker = planner.tracker_for(route, target_tolerance_m=args.route_tolerance_m)
        controller = RoutePurePursuitController(target_speed_kmh=args.target_speed_kmh, dt=1.0 / args.fps)

        if args.output:
            csv_handle = open(args.output, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_handle, fieldnames=["tick", "x", "y", "speed_kmh"] + ROUTE_FIELDS)
            csv_writer.writeheader()
        else:
            csv_writer = None

        print("Bat dau lai xe theo lo trinh...")
        start_time = time.time()
        tick = 0
        log_every = max(1, int(args.fps))
        route_state = {}
        while time.time() - start_time < args.max_duration_s:
            control = controller.compute_control(
                planner.graph, route, tracker.target_index, vehicle,
                speed_limit_kmh=vehicle.get_speed_limit())
            vehicle.apply_control(control)
            world.tick()
            tick += 1

            transform = vehicle.get_transform()
            route_state = tracker.update(transform)

            if csv_writer:
                velocity = vehicle.get_velocity()
                speed_kmh = 3.6 * magnitude(velocity)
                row = {"tick": tick, "x": transform.location.x, "y": transform.location.y,
                       "speed_kmh": speed_kmh}
                row.update(route_state)
                csv_writer.writerow(row)

            if tick % log_every == 0:
                print("tick=%d progress=%.1fm/%.1fm command=%s" % (
                    tick, route_state["route_progress_m"], route_state["route_total_m"],
                    route_state["route_command"]))

            if route_state.get("route_completed"):
                print("Da den dich sau %d tick (~%.1fs)." % (tick, tick / args.fps))
                break
        else:
            print("[!] Het thoi gian toi da (%.0fs) ma chua den dich — "
                  "route_remaining_m=%.1f." % (
                      args.max_duration_s, route_state.get("route_remaining_m", float("nan"))))
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError:
                pass
        try:
            world.apply_settings(previous_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
