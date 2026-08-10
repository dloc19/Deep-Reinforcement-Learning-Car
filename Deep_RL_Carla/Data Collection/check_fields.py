#!/usr/bin/env python3
"""
check_fields.py
---------------
Đọc states.csv của một session và kiểm tra xem toàn bộ các trường yêu cầu
cho Imitation Learning, Deep Reinforcement Learning và A* đã có dữ liệu chưa.

Cách dùng:
    python check_fields.py <session_dir>
    python check_fields.py <session_dir> --rows 200   # chỉ quét 200 dòng đầu
    python check_fields.py dataset/Town01_20260726_...
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Định nghĩa các trường cần kiểm tra cho từng module
# ---------------------------------------------------------------------------

IL_FIELDS = [
    "session_id",
    "sample_id",
    "frame",
    "seg_label_path",
    "seg_color_path",
    "speed_mps",
    "yaw_rate_rps",
    "lane_offset_m",
    "heading_error_rad",
    "speed_limit_kmh",
    "traffic_light_state",
    "is_junction",
    "previous_steer",
    "previous_longitudinal",
    "steer",
    "longitudinal",
]

DRL_FIELDS = [
    "session_id",
    "sample_id",
    "frame",
    "seg_label_path",
    "seg_color_path",
    "speed_mps",
    "forward_speed_mps",
    "yaw_rate_rps",
    "lane_offset_m",
    "normalized_lane_offset",
    "heading_error_rad",
    "route_target_local_x",
    "route_target_local_y",
    "route_command",
    "steer",
    "longitudinal",
    "steer_delta",
    "longitudinal_delta",
    "collision_count",
    "lane_invasion_count",
    "off_lane",
    "speed_limit_kmh",
    "traffic_light_state",
    "is_junction",
    "route_completed",
]

ASTAR_FIELDS = [
    "session_id",
    "sample_id",
    "frame",
    "waypoint_id",
    "road_id",
    "section_id",
    "lane_id",
    "waypoint_s",
    "waypoint_x",
    "waypoint_y",
    "left_waypoint_id",
    "right_waypoint_id",
    "successor_waypoints_json",
    "lookahead_waypoints_json",
    "goal_waypoint_id",
    "goal_x",
    "goal_y",
    "route_id",
    "route_target_waypoint_id",
    "route_target_local_x",
    "route_target_local_y",
    "route_command",
    "route_progress_m",
    "route_remaining_m",
    "route_total_m",
    "route_completed",
]

GROUPS = {
    "Imitation Learning (IL)": IL_FIELDS,
    "Deep Reinforcement Learning (DRL)": DRL_FIELDS,
    "A*": ASTAR_FIELDS,
}

# ---------------------------------------------------------------------------
# Màu terminal ANSI (tắt tự động nếu không phải TTY)
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):  return _c("32", t)
def red(t):    return _c("31;1", t)
def yellow(t): return _c("33", t)
def bold(t):   return _c("1", t)
def cyan(t):   return _c("36", t)


# ---------------------------------------------------------------------------
# Hàm kiểm tra
# ---------------------------------------------------------------------------

def check_fields(csv_path, max_rows=0):
    """
    Trả về dict kết quả:
        {
            "total_rows": int,
            "header": [str, ...],
            "groups": {
                group_name: {
                    "missing_from_header": [field, ...],
                    "always_empty":        [field, ...],
                    "sometimes_empty":     {field: count},
                    "ok":                  [field, ...],
                }
            }
        }
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Khong tim thay: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        header_set = set(header)

        empty_counts = defaultdict(int)
        total_rows = 0

        for row in reader:
            total_rows += 1
            for field in header_set:
                val = row.get(field, "")
                if val is None or str(val).strip() == "":
                    empty_counts[field] += 1
            if max_rows and total_rows >= max_rows:
                break

    results = {"total_rows": total_rows, "header": header, "groups": {}}

    for group_name, fields in GROUPS.items():
        missing_from_header = []
        always_empty = []
        sometimes_empty = {}
        ok = []

        for field in fields:
            if field not in header_set:
                missing_from_header.append(field)
            elif total_rows > 0 and empty_counts.get(field, 0) == total_rows:
                always_empty.append(field)
            elif empty_counts.get(field, 0) > 0:
                sometimes_empty[field] = empty_counts[field]
            else:
                ok.append(field)

        results["groups"][group_name] = {
            "missing_from_header": missing_from_header,
            "always_empty": always_empty,
            "sometimes_empty": sometimes_empty,
            "ok": ok,
        }

    return results


# ---------------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------------

def print_report(results, session_dir):
    total = results["total_rows"]
    header_count = len(results["header"])

    print()
    print(bold("=" * 66))
    print(bold(f"  CHECK FIELDS -- {session_dir.name}"))
    print(bold("=" * 66))
    print(f"  Tong so dong du lieu : {cyan(str(total))}")
    print(f"  So cot trong CSV     : {cyan(str(header_count))}")
    print()

    overall_ok = True

    for group_name, group in results["groups"].items():
        missing   = group["missing_from_header"]
        always    = group["always_empty"]
        sometimes = group["sometimes_empty"]
        ok        = group["ok"]

        total_fields = len(missing) + len(always) + len(sometimes) + len(ok)
        has_issue = bool(missing or always or sometimes)
        if has_issue:
            overall_ok = False

        status_icon = red("X LOI") if has_issue else green("OK")
        print(bold(f"+-- [{status_icon}] {group_name}  ({total_fields} truong)"))

        # Trường đủ dữ liệu
        if ok:
            print(f"|  {green('+')} Du du lieu ({len(ok)} truong):")
            for f in ok:
                print(f"|      {green('v')} {f}")

        # Trường có nhưng đôi khi rỗng
        if sometimes:
            print(f"|  {yellow('!')} Doi khi rong ({len(sometimes)} truong):")
            for f, cnt in sometimes.items():
                pct = cnt / total * 100 if total else 0
                print(f"|      {yellow('!')} {f}  ->  {cnt}/{total} dong rong ({pct:.1f}%)")

        # Trường rỗng hoàn toàn
        if always:
            print(f"|  {red('X')} Rong hoan toan ({len(always)} truong):")
            for f in always:
                print(f"|      {red('X')} {f}")

        # Trường không có trong CSV
        if missing:
            print(f"|  {red('X')} Khong co trong CSV ({len(missing)} truong):")
            for f in missing:
                print(f"|      {red('X')} {f}  <- THIEU COT")

        print("|")
        print("+" + "-" * 64)
        print()

    print(bold("=" * 66))
    if overall_ok:
        print(bold(green("  OK  TAT CA CAC TRUONG DEU CO DU LIEU -- pipeline hoan chinh")))
    else:
        print(bold(red("  X   CO TRUONG BI THIEU HOAC RONG -- xem chi tiet o tren")))
    print(bold("=" * 66))
    print()

    return overall_ok


# ---------------------------------------------------------------------------
# Xuất JSON (tuỳ chọn)
# ---------------------------------------------------------------------------

def export_json(results, out_path):
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu ket qua JSON: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kiem tra cac truong IL / DRL / A* trong states.csv cua mot session CARLA.")
    parser.add_argument(
        "session",
        help="Duong dan toi thu muc session (chua states.csv)")
    parser.add_argument(
        "--rows", type=int, default=0, metavar="N",
        help="Chi quet N dong dau tien (0 = quet tat ca, mac dinh)")
    parser.add_argument(
        "--json", default="", metavar="FILE",
        help="Xuat ket qua dang JSON ra file (tuy chon)")
    args = parser.parse_args()

    session_dir = Path(args.session).expanduser().resolve()
    csv_path = session_dir / "states.csv"

    print(f"\nDang doc: {csv_path}")
    if args.rows:
        print(f"Gioi han quet: {args.rows} dong dau tien")

    try:
        results = check_fields(csv_path, max_rows=args.rows)
    except FileNotFoundError as exc:
        print(red(f"\nLOI: {exc}"), file=sys.stderr)
        raise SystemExit(2)

    ok = print_report(results, session_dir)

    if args.json:
        export_json(results, Path(args.json))

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
