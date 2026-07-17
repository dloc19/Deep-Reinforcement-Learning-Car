#!/usr/bin/env python3
"""Check that CSV rows and all three image streams are complete and aligned."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="Thu muc TownXX_YYYYMMDD_HHMMSS")
    parser.add_argument("--strict", action="store_true", help="Bao loi neu class ID nam ngoai 0..12")
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
                elif args.strict and label.max() > 12:
                    errors.append("Dong %d: semantic class ID lon nhat=%d" % (line_no, label.max()))
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

    report = {
        "session": str(root), "rows": len(rows), "unique_frames": len(frames),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "astar_graph_exported": bool(graph.get("exported")),
        "astar_nodes": graph.get("node_count", 0),
        "astar_edges": graph.get("edge_count", 0),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
