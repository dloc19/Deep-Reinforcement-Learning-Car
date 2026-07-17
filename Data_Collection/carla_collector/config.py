"""Command-line configuration and validation."""

import argparse
import json
from pathlib import Path


def _flatten_config(data, output=None):
    """Flatten named JSON sections while keeping argparse destination keys."""
    output = {} if output is None else output
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten_config(value, output)
        else:
            if key in output:
                raise ValueError("Config key bi trung: %s" % key)
            output[key] = value
    return output


def _apply_config_file(parser):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _ = pre_parser.parse_known_args()
    if not known.config:
        return
    path = Path(known.config).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        defaults = _flatten_config(data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error("Khong doc duoc config %s: %s" % (path, exc))
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - valid_keys)
    if unknown:
        parser.error("Config co key khong hop le: %s" % ", ".join(unknown))
    parser.set_defaults(**defaults)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thu thap du lieu tu ego vehicle dang chay trong CARLA")
    parser.add_argument("--config", default=None,
                        help="File JSON cau hinh, vi du collector_config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--role-name", default="hero",
                        help="role_name do automatic_control.py tao")
    parser.add_argument("--vehicle-id", type=int, default=0,
                        help="Gan truc tiep vao actor ID; 0=tu tim")
    parser.add_argument("--wait-vehicle-timeout", type=float, default=120.0)

    camera = parser.add_argument_group("camera")
    camera.add_argument("--width", type=int, default=800)
    camera.add_argument("--height", type=int, default=450)
    camera.add_argument("--fov", type=float, default=90.0)
    camera.add_argument("--fps", type=float, default=10.0)
    camera.add_argument(
        "--image-mode", choices=("seg-only", "seg-rgb"), default="seg-only",
        help="seg-only tiet kiem tai nguyen; seg-rgb luu them RGB")
    camera.add_argument("--save-seg-color", action="store_true",
                        help="Luu semantic to mau de xem, khong dung train")
    camera.add_argument("--camera-x", type=float, default=1.5)
    camera.add_argument("--camera-y", type=float, default=0.0)
    camera.add_argument("--camera-z", type=float, default=2.4)
    camera.add_argument("--camera-pitch", type=float, default=-5.0)

    route = parser.add_argument_group("route and A-star")
    route.add_argument("--lookahead-m", type=float, default=5.0)
    route.add_argument("--route-lookaheads", default="5,10,20,30")
    route.add_argument("--graph-resolution", type=float, default=2.0)
    route.add_argument("--lane-change-cost", type=float, default=3.0)
    route.add_argument("--no-map-export", action="store_true")
    route.add_argument("--goal-spawn-index", type=int, default=-1)
    route.add_argument("--goal-x", type=float, default=None)
    route.add_argument("--goal-y", type=float, default=None)
    route.add_argument("--goal-z", type=float, default=0.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--duration", type=float, default=0.0,
                         help="So giay; 0=den khi Ctrl+C")
    runtime.add_argument("--max-samples", type=int, default=0,
                         help="0=khong gioi han")
    runtime.add_argument("--queue-size", type=int, default=32)
    runtime.add_argument("--no-event-sensors", action="store_true")
    _apply_config_file(parser)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps phai > 0")
    if args.graph_resolution <= 0:
        parser.error("--graph-resolution phai > 0")
    if args.lane_change_cost < 1.0:
        parser.error("--lane-change-cost nen >= 1")
    if (args.goal_x is None) != (args.goal_y is None):
        parser.error("Phai truyen dong thoi --goal-x va --goal-y")
    if args.goal_spawn_index >= 0 and args.goal_x is not None:
        parser.error("Chi chon --goal-spawn-index hoac --goal-x/--goal-y")
    try:
        args.route_lookaheads = [
            float(value.strip()) for value in args.route_lookaheads.split(",")
            if value.strip()]
    except ValueError:
        parser.error("--route-lookaheads phai co dang 5,10,20,30")
    if not args.route_lookaheads or any(value <= 0 for value in args.route_lookaheads):
        parser.error("--route-lookaheads phai chua cac so > 0")
    return args
