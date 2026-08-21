#!/usr/bin/env python3
"""Merge repeated CARLA collection sessions without frame-level leakage."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def discover_sessions(root):
    sessions = []
    for states_path in sorted(root.glob("*/states.csv")):
        session_dir = states_path.parent
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sessions.append({
            "path": session_dir,
            "name": session_dir.name,
            "map": metadata.get("map", "unknown").split("/")[-1],
        })
    return sessions


def allocate_sessions(sessions, train_ratio, val_ratio, seed, holdout_map):
    rng = random.Random(seed)
    pool = list(sessions)
    test = []
    if holdout_map:
        test = [item for item in pool if item["map"].lower() == holdout_map.lower()]
        pool = [item for item in pool if item not in test]
        if not test:
            raise ValueError("Khong tim thay session cua holdout map %s" % holdout_map)
    rng.shuffle(pool)

    count = len(pool)
    if count == 0:
        raise ValueError("Khong con session de chia train/validation")
    n_train = max(1, int(round(count * train_ratio)))
    n_val = max(1, int(round(count * val_ratio))) if count >= 2 else 0
    if not holdout_map:
        # Keep at least one independent test session when possible.
        while count >= 3 and count - n_train - n_val < 1:
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
    if n_train + n_val > count:
        n_val = max(0, count - n_train)

    train = pool[:n_train]
    val = pool[n_train:n_train + n_val]
    if not holdout_map:
        test = pool[n_train + n_val:]
    assignments = {}
    for split, items in (("train", train), ("val", val), ("test", test)):
        for item in items:
            assignments[item["name"]] = split
    return assignments


def is_redundant_row(row, last_kept_sim_time, dedup_stationary_speed,
                      dedup_action_eps, dedup_min_interval_s):
    """Hau kiem tra trung lap cho du lieu da thu TRUOC khi collector co filter nay.

    Cung tieu chi voi carla_collector/writer.py: chi coi la trung lap khi xe DUNG
    YEN va hanh dong (steer/longitudinal) khong doi so voi mau da giu gan nhat -
    khong bao gio loai mau luc xe dang di chuyen.
    """
    if dedup_stationary_speed <= 0.0 or last_kept_sim_time is None:
        return False
    speed = float(row.get("speed_mps", 0.0) or 0.0)
    if speed >= dedup_stationary_speed:
        return False
    steer_delta = abs(float(row.get("steer_delta", 0.0) or 0.0))
    long_delta = abs(float(row.get("longitudinal_delta", 0.0) or 0.0))
    if steer_delta >= dedup_action_eps or long_delta >= dedup_action_eps:
        return False
    sim_time = float(row.get("sim_time_s", 0.0) or 0.0)
    return (sim_time - last_kept_sim_time) < dedup_min_interval_s


def main():
    parser = argparse.ArgumentParser(
        description="Gop cac session CARLA va chia train/val/test theo session")
    parser.add_argument("dataset_root", help="Thu muc chua cac TownXX_timestamp")
    parser.add_argument("--output", default="manifest.csv")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-map", default="",
        help="Vi du Town03: toan bo Town03 duoc dung lam test cross-map")
    parser.add_argument(
        "--dedup-stationary-speed", type=float, default=0.0,
        help=("m/s; hau loc cac mau dung yen + hanh dong khong doi (vd. dung cho "
              "den do) khi gop manifest. 0 = tat (mac dinh). Dung cho du lieu da "
              "thu TRUOC khi collector co filter nay; anh PNG khong bi xoa, chi "
              "khong duoc dua vao manifest.csv."))
    parser.add_argument("--dedup-action-eps", type=float, default=0.02)
    parser.add_argument("--dedup-min-interval-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0:
        parser.error("Ti le train/val khong hop le")
    if not args.holdout_map and args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train-ratio + val-ratio phai < 1 khi khong co holdout map")
    if args.dedup_stationary_speed < 0 or args.dedup_action_eps < 0 or args.dedup_min_interval_s < 0:
        parser.error("Cac tham so --dedup-* phai >= 0")

    root = Path(args.dataset_root).expanduser().resolve()
    sessions = discover_sessions(root)
    if not sessions:
        raise SystemExit("Khong tim thay session hop le trong %s" % root)
    assignments = allocate_sessions(
        sessions, args.train_ratio, args.val_ratio, args.seed, args.holdout_map)

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    fieldnames = None
    sample_counts = Counter()
    session_counts = Counter(assignments.values())
    map_counts = defaultdict(Counter)
    action_stats = Counter()
    duplicates_filtered = 0

    with output.open("w", newline="", encoding="utf-8") as output_handle:
        writer = None
        for session in sessions:
            split = assignments.get(session["name"])
            if split is None:
                continue
            states_path = session["path"] / "states.csv"
            with states_path.open("r", newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                if fieldnames is None:
                    fieldnames = ["split", "session_dir"] + list(reader.fieldnames or [])
                    writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames or []) != fieldnames[2:]:
                    raise ValueError(
                        "Schema khac nhau tai %s; khong tron pipeline version cu va moi" % states_path)

                last_kept_sim_time = None
                for row in reader:
                    if is_redundant_row(
                            row, last_kept_sim_time, args.dedup_stationary_speed,
                            args.dedup_action_eps, args.dedup_min_interval_s):
                        duplicates_filtered += 1
                        continue
                    last_kept_sim_time = float(row.get("sim_time_s", 0.0) or 0.0)

                    seg_relative = row.get("seg_label_path", "")
                    seg_path = session["path"] / seg_relative
                    if not seg_relative or not seg_path.is_file():
                        raise FileNotFoundError("Thieu segmentation: %s" % seg_path)
                    row["seg_label_path"] = (
                        Path(session["name"]) / seg_relative).as_posix()
                    for optional_key in ("rgb_path", "seg_color_path"):
                        if row.get(optional_key):
                            row[optional_key] = (
                                Path(session["name"]) / row[optional_key]).as_posix()
                    output_row = {"split": split, "session_dir": session["name"]}
                    output_row.update(row)
                    writer.writerow(output_row)

                    sample_counts[split] += 1
                    map_counts[split][session["map"]] += 1
                    steer = float(row.get("steer", 0.0) or 0.0)
                    if steer < -0.05:
                        action_stats["steer_negative"] += 1
                    elif steer > 0.05:
                        action_stats["steer_positive"] += 1
                    else:
                        action_stats["steer_near_zero"] += 1
                    if float(row.get("brake", 0.0) or 0.0) > 0.05:
                        action_stats["braking"] += 1
                    if str(row.get("is_junction", "0")) == "1":
                        action_stats["junction"] += 1

    for split in ("train", "val", "test"):
        names = sorted(name for name, value in assignments.items() if value == split)
        (root / ("%s_sessions.txt" % split)).write_text(
            "\n".join(names) + ("\n" if names else ""), encoding="utf-8")

    summary = {
        "dataset_root": str(root),
        "manifest": str(output),
        "seed": args.seed,
        "holdout_map": args.holdout_map or None,
        "sessions_total": len(assignments),
        "sessions_by_split": dict(session_counts),
        "samples_by_split": dict(sample_counts),
        "samples_by_map_and_split": {
            split: dict(counts) for split, counts in map_counts.items()},
        "action_distribution": dict(action_stats),
        "duplicate_frames_filtered": duplicates_filtered,
        "warning": (
            None if len(assignments) >= 3 else
            "Can it nhat 3 sessions de co train, validation va test doc lap"),
    }
    summary_path = root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
