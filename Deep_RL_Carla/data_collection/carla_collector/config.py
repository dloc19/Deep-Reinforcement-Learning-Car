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
    parser.add_argument(
        "--min-wheels", type=int, default=4,
        help=("So banh toi thieu de mot chiec xe duoc chon lam ego. 4 = bo qua "
              "xe dap va moto ma `automatic_control.py` co the boc trung khi no "
              "chon blueprint ngau nhien. 0 = nhan tat ca. Khong ap dung khi da "
              "ghim --vehicle-id."))

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
                         help="So mau toi da cua MOT session; 0=khong gioi han")
    runtime.add_argument("--queue-size", type=int, default=32)
    runtime.add_argument("--no-event-sensors", action="store_true")

    rebind = parser.add_argument_group("ego-rebind")
    rebind_switch = rebind.add_mutually_exclusive_group()
    rebind_switch.add_argument(
        "--rebind-ego", dest="rebind_ego", action="store_true",
        help=("Khi xe dang thu bi ket / bi huy / camera chet: go sensor ra, cho "
              "mot chiec xe KHAC xuat hien (chay lai automatic_control.py) roi "
              "gan sensor vao va thu tiep VAO CHINH SESSION DANG CHAY. Session "
              "chi dong khi het --rebind-wait-s ma khong co xe nao."))
    rebind_switch.add_argument(
        "--no-rebind-ego", dest="rebind_ego", action="store_false",
        help="Xe ket la dong session luon (hanh vi cu, de SessionRunner lo tiep)")
    rebind.set_defaults(rebind_ego=True)
    rebind.add_argument(
        "--rebind-wait-s", type=float, default=300.0,
        help=("Giay THUC cho mot chiec xe khac xuat hien truoc khi dong session. "
              "Du de kip Ctrl+C roi chay lai automatic_control.py."))
    rebind.add_argument(
        "--max-rebinds", type=int, default=0,
        help="So lan doi xe toi da trong MOT session; 0 = khong gioi han")

    restart = parser.add_argument_group("auto-restart")
    restart.add_argument(
        "--total-samples", type=int, default=0,
        help=("Muc tieu so mau cho CA BUOI thu thap, cong don qua nhieu session. "
              "0 = chi chay mot session roi thoi (hanh vi cu). Dat cai nay lon "
              "hon --max-samples de mot session bi watchdog nga khong lam mat "
              "phan con lai cua buoi thu thap."))
    auto = restart.add_mutually_exclusive_group()
    auto.add_argument(
        "--auto-restart", dest="auto_restart", action="store_true",
        help=("Mo session moi khi session hien tai bi watchdog nga "
              "(vehicle_stationary_timeout / no_samples_timeout / ego_destroyed). "
              "Runner cho den khi ego thuc su chay lai roi moi mo session moi."))
    auto.add_argument(
        "--no-auto-restart", dest="auto_restart", action="store_false",
        help="Hong session nao la dung buoi thu thap luon")
    restart.set_defaults(auto_restart=False)
    restart.add_argument(
        "--max-restarts", type=int, default=0,
        help="So lan khoi dong lai sau LOI toi da; 0 = khong gioi han")
    restart.add_argument(
        "--restart-wait-s", type=float, default=300.0,
        help=("Giay THUC cho ego chay lai truoc khi bo cuoc. Cac quang ket cua "
              "BehaviorAgent do duoc dai 25-45 s nen mac dinh de rong."))
    restart.add_argument(
        "--restart-settle-s", type=float, default=3.0,
        help="Giay nghi giua hai session de sensor cu kip go xuong")

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
    watchdog = parser.add_argument_group("watchdog")
    watchdog.add_argument(
        "--stall-timeout-s", type=float, default=60.0,
        help=("Giay THUC khong co mau moi nao thi dung session. Bat truong hop "
              "camera ngung goi callback trong khi ego van song, luc do ego.is_alive "
              "van True nen vong lap chinh khong tu phat hien duoc (mot session "
              "Town03 da chay 15 phut thuc ma chi ghi duoc 102 giay dau). 0 = tat."))
    watchdog.add_argument(
        "--stationary-timeout-s", type=float, default=60.0,
        help=("Giay SIM xe dung yen lien tuc thi dung session. Bat truong hop "
              "BehaviorAgent phanh khan cap roi khong bao gio nha (mot session "
              "Town03 dung yen 344 s = 96%% session, di duoc 76 m). Phai dat LON HON "
              "lan cho den do lau nhat cua ban, neu khong se dung nham. 0 = tat."))
    watchdog.add_argument(
        "--camera-timeout-s", type=float, default=10.0,
        help=("Giay THUC camera khong gui anh nao (trong khi world VAN tick) thi "
              "gan lai camera vao chinh chiec xe do. Bat rieng truong hop camera "
              "chet ma xe van chay - truoc day phai doi het --stall-timeout-s "
              "(60 s) roi doi xe, va chiec 'xe moi' tim duoc thuong la chinh no. "
              "0 = tat."))
    watchdog.add_argument(
        "--ego-missing-timeout-s", type=float, default=3.0,
        help=("Giay THUC ego vang mat khoi world snapshot thi coi nhu da bi huy. "
              "`actor.is_alive` la co cua RIENG client nay nen no van True khi "
              "client khac (automatic_control.py) huy chiec hero; day la cach "
              "duy nhat thay dieu do ma khong phai cho het 60 s. 0 = tat."))
    watchdog.add_argument(
        "--stall-speed", type=float, default=0.3,
        help=("m/s; duoi nguong nay watchdog coi la xe dung yen. Doc lap voi "
              "--dedup-stationary-speed de watchdog van chay khi dedup da tat."))

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
    for flag in ("stall_timeout_s", "stationary_timeout_s", "stall_speed",
                 "camera_timeout_s", "ego_missing_timeout_s",
                 "restart_wait_s", "restart_settle_s", "rebind_wait_s"):
        if getattr(args, flag) < 0:
            parser.error("--%s phai >= 0" % flag.replace("_", "-"))
    if args.min_wheels < 0:
        parser.error("--min-wheels phai >= 0")
    if args.max_rebinds < 0:
        parser.error("--max-rebinds phai >= 0")
    if args.total_samples < 0:
        parser.error("--total-samples phai >= 0")
    if args.max_restarts < 0:
        parser.error("--max-restarts phai >= 0")
    if args.total_samples and args.max_samples > args.total_samples:
        parser.error("--max-samples (%d) khong duoc lon hon --total-samples (%d)"
                     % (args.max_samples, args.total_samples))
    if args.auto_restart and not args.total_samples:
        # Khong co muc tieu tong thi khoi dong lai bao nhieu lan cung khong biet
        # khi nao la du - session moi se chay den khi bi nga lan nua, mai mai.
        parser.error("--auto-restart can --total-samples > 0 de biet khi nao dung")
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
