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
    # Mac dinh CLI phai TRUNG collector_config.json: neu khong, chay
    # `collect_data.py` khong kem --config se sinh ra mot bo du lieu khac do phan
    # giai / khac buoc thoi gian voi bo dang co, va khong gi bao loi ca.
    camera.add_argument("--width", type=int, default=480)
    camera.add_argument("--height", type=int, default=384)
    camera.add_argument("--fov", type=float, default=90.0)
    camera.add_argument("--fps", type=float, default=5.0,
                        help="Tan so thu mau. 5 FPS = CONTROL_DT 0.2s - day la HOP "
                             "DONG voi IL/DRL: previous_steer nghia la 'lenh cua 1 "
                             "buoc truoc', doi FPS la doi y nghia dac trung do "
                             "(xem train_il.ipynb muc 2)")
    camera.add_argument(
        "--image-mode", choices=("seg-only", "seg-rgb"), default="seg-only",
        help="seg-only tiet kiem tai nguyen; seg-rgb luu them RGB")
    color_output = camera.add_mutually_exclusive_group()
    color_output.add_argument(
        "--save-seg-color", dest="save_seg_color", action="store_true",
        help="Luu semantic to mau 3 kenh de xem hoac train")
    color_output.add_argument(
        "--no-seg-color", dest="save_seg_color", action="store_false",
        help="Chi luu seg_label 1 kenh de tiet kiem dung luong")
    camera.set_defaults(save_seg_color=False)
    camera.add_argument("--camera-x", type=float, default=1.5)
    camera.add_argument("--camera-y", type=float, default=0.0)
    camera.add_argument("--camera-z", type=float, default=2.4)
    camera.add_argument("--camera-pitch", type=float, default=-5.0)

    route = parser.add_argument_group("route and A-star")
    route.add_argument("--lookahead-m", type=float, default=5.0)
    route.add_argument("--route-lookaheads", default="5,10,20,30")
    route.add_argument("--graph-resolution", type=float, default=2.0)
    route.add_argument("--lane-change-cost", type=float, default=3.0)
    map_output = route.add_mutually_exclusive_group()
    map_output.add_argument(
        "--map-export", dest="no_map_export", action="store_false",
        help="Xuat OpenDRIVE va graph de dung cho A* sau nay")
    map_output.add_argument(
        "--no-map-export", dest="no_map_export", action="store_true",
        help="Khong xuat graph A* trong session hien tai")
    route.set_defaults(no_map_export=False)
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

    dedup = parser.add_argument_group("dedup")
    dedup.add_argument(
        "--dedup-stationary-speed", type=float, default=0.0,
        help=("m/s; duoi nguong nay coi xe la dung yen. 0 = tat loc trung lap "
              "(mac dinh, giu nguyen hanh vi cu). Chi bo qua mau khi xe dung yen "
              "VA hanh dong khong doi (vd. dung cho den do), khong bao gio bo "
              "mau luc xe dang di chuyen."))
    dedup.add_argument(
        "--dedup-action-eps", type=float, default=0.02,
        help="Nguong |steer_delta|/|longitudinal_delta| de coi hanh dong la khong doi")
    dedup.add_argument(
        "--dedup-min-interval-s", type=float, default=1.0,
        help=("So giay toi thieu giua 2 mau dung yen/hanh dong khong doi lien tiep "
              "duoc GIU LAI; chi co hieu luc khi --dedup-stationary-speed > 0"))
    _apply_config_file(parser)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps phai > 0")
    if args.graph_resolution <= 0:
        parser.error("--graph-resolution phai > 0")
    if args.lane_change_cost < 1.0:
        parser.error("--lane-change-cost nen >= 1")
    if args.dedup_stationary_speed < 0:
        parser.error("--dedup-stationary-speed phai >= 0")
    if args.dedup_action_eps < 0:
        parser.error("--dedup-action-eps phai >= 0")
    if args.dedup_min_interval_s < 0:
        parser.error("--dedup-min-interval-s phai >= 0")
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
