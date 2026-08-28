#!/usr/bin/env python3
"""Check that CSV rows and all three image streams are complete and aligned."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# CARLA 0.9.10 dinh nghia raw semantic tag 0..22 (0.9.11+ them mot vai tag moi).
MAX_RAW_SEMANTIC_TAG = 22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="Thu muc TownXX_YYYYMMDD_HHMMSS")
    parser.add_argument("--strict", action="store_true",
                        help="Bao loi neu seg_label chua tag ngoai dai raw CARLA 0..22")
    parser.add_argument("--fps-tolerance", type=float, default=0.10,
                        help="Sai so tuong doi cho phep giua FPS do duoc va camera.fps")
    args = parser.parse_args()

    root = Path(args.session).resolve()
    errors = []
    rows = []
    frames = set()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        errors.append("Thieu metadata.json")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    astar_fields = {
        "waypoint_id", "road_id", "section_id", "lane_id", "waypoint_s",
        "successor_waypoints_json", "lookahead_waypoints_json",
        "left_waypoint_id", "right_waypoint_id", "goal_waypoint_id",
        "route_id", "route_target_waypoint_id", "route_command",
        "route_progress_m", "route_remaining_m"}
    training_fields = {
        "seg_label_path", "seg_color_path", "speed_mps", "yaw_rate_rps",
        "steer", "throttle", "brake", "longitudinal", "previous_steer",
        "previous_longitudinal", "steer_delta", "longitudinal_delta",
        "lane_offset_m", "normalized_lane_offset", "heading_error_rad",
        "lane_width_m", "off_lane"}
    graph = metadata.get("map_graph", {})
    if graph.get("exported"):
        for filename in ("map.xodr", "spawn_points.csv", "map_nodes.csv",
                         "map_edges.csv", "map_graph_metadata.json"):
            if not (root / filename).is_file():
                errors.append("Thieu A* artifact: %s" % filename)
        if graph.get("node_count", 0) <= 0 or graph.get("edge_count", 0) <= 0:
            errors.append("Map graph khong co node/edge")
    with (root / "states.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(astar_fields - set(reader.fieldnames or []))
        if missing_fields:
            errors.append("states.csv thieu A* fields: %s" % ", ".join(missing_fields))
        missing_training = sorted(training_fields - set(reader.fieldnames or []))
        if missing_training:
            errors.append(
                "states.csv thieu training fields: %s" % ", ".join(missing_training))
        for line_no, row in enumerate(reader, start=2):
            frame = int(row["frame"])
            if frame in frames:
                errors.append("Dong %d: frame trung %d" % (line_no, frame))
            frames.add(frame)
            label_rel = row.get("seg_label_path", "")
            if not label_rel or not (root / label_rel).is_file():
                errors.append("Dong %d: thieu seg_label %s" % (line_no, label_rel))
            for key in ("rgb_path", "seg_color_path"):
                relative = row.get(key, "")
                if relative and not (root / relative).is_file():
                    errors.append("Dong %d: thieu %s" % (line_no, relative))
            label_path = root / row["seg_label_path"]
            if label_path.is_file():
                label = np.asarray(Image.open(str(label_path)))
                if label.ndim != 2:
                    errors.append("Dong %d: seg_label khong phai anh 1 kenh" % line_no)
                elif args.strict and label.max() > MAX_RAW_SEMANTIC_TAG:
                    # seg_label giu RAW tag CARLA (0..22), KHONG phai train id 0..3: viec gop
                    # ve 4 lop bam lan do phia train lam qua schema.RAW_TO_TRAIN_LANE_LUT.
                    # Nguong cu la 12 nen moi anh co bau troi/Terrain deu bi bao loi sai.
                    errors.append("Dong %d: raw semantic tag lon nhat=%d (> %d)"
                                  % (line_no, label.max(), MAX_RAW_SEMANTIC_TAG))
            if not row.get("waypoint_id"):
                errors.append("Dong %d: thieu waypoint_id" % line_no)
            for key in ("successor_waypoints_json", "lookahead_waypoints_json"):
                try:
                    value = json.loads(row.get(key, ""))
                    if not isinstance(value, list):
                        raise ValueError("not a list")
                except (TypeError, ValueError, json.JSONDecodeError):
                    errors.append("Dong %d: %s khong phai JSON list hop le" % (line_no, key))
            rows.append(row)

    # Buoc thoi gian giua hai mau LA thang do cua previous_steer /
    # previous_longitudinal, khong phai metadata trang tri. Collector chay nhu mot
    # client thu dong tren world async nen `sensor_tick` KHONG dam bao dieu do: mot
    # session dat 5 FPS da tung ghi ra 22.7 mau/giay sim ma khong co canh bao nao.
    # Nhip that su duoc ep boi synchronizer.SampleRateLimiter, nen o day chi can
    # kiem tra lai ket qua.
    #
    # Do bang CADENCE (trung vi khoang cach giua hai mau), KHONG phai rows/span:
    # bo loc dedup xoa han cac mau dung yen, nen throughput tut xuong duoi fps yeu
    # cau mot cach hop le (do duoc 3.57 thay vi 5.00 tren mot session Town01 that,
    # trong khi limiter van cho qua dung 5.000 mau/giay). Trung vi khong bi anh
    # huong boi nhung khoang trong do.
    measured = {}
    sim_times = []
    for row in rows:
        try:
            sim_times.append(float(row.get("sim_time_s") or 0.0))
        except ValueError:
            pass
    requested_fps = (metadata.get("camera") or {}).get("fps")
    if len(sim_times) >= 3 and sim_times[-1] > sim_times[0]:
        gaps = sorted(b - a for a, b in zip(sim_times, sim_times[1:]))
        median_gap = gaps[len(gaps) // 2]
        span = sim_times[-1] - sim_times[0]
        measured = {
            "cadence_fps": round(1.0 / median_gap, 3) if median_gap > 0 else None,
            "throughput_fps": round((len(sim_times) - 1) / span, 3),
            "gap_seconds": {"min": round(gaps[0], 4),
                            "median": round(median_gap, 4),
                            "max": round(gaps[-1], 4)},
        }
        if requested_fps:
            period = 1.0 / requested_fps
            # Dedup bo NGUYEN mau, nen gap con lai luon la BOI SO nguyen cua chu ky.
            # Gap lech khoi luoi do moi la dau hieu nhip that su khong on dinh - day
            # la thu phan biet "dedup dang lam viec" voi "sensor_tick da tuot".
            off_grid = [g for g in gaps
                        if abs(g / period - round(g / period)) > 0.25]
            measured["off_grid_gap_fraction"] = round(len(off_grid) / len(gaps), 4)
            if median_gap > 0:
                deviation = abs(median_gap - period) / period
                if deviation > args.fps_tolerance:
                    errors.append(
                        "Nhip lay mau lech: metadata ghi %.2f FPS (chu ky %.3f s) "
                        "nhung trung vi khoang cach mau la %.3f s (%.2f FPS, lech "
                        "%.0f%%). previous_steer/previous_longitudinal se sai thang "
                        "do; KHONG dung session nay de train." % (
                            requested_fps, period, median_gap, 1.0 / median_gap,
                            100 * deviation))
            if len(off_grid) > 0.05 * len(gaps):
                errors.append(
                    "%.1f%% khoang cach mau khong roi vao boi so cua %.3f s. Dedup "
                    "chi xoa nguyen mau nen khong gay ra dieu nay; nhip lay mau dang "
                    "that su khong on dinh." % (
                        100.0 * len(off_grid) / len(gaps), period))

    report = {
        "session": str(root), "rows": len(rows), "unique_frames": len(frames),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "requested_fps": requested_fps,
        "measured": measured,
        "astar_graph_exported": bool(graph.get("exported")),
        "astar_nodes": graph.get("node_count", 0),
        "astar_edges": graph.get("edge_count", 0),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
