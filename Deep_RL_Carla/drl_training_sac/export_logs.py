# -*- coding: utf-8 -*-
"""Xuat log train cua PPO va SAC ra CSV thong nhat de ve bieu do so sanh.

Hai thuat toan ghi log theo don vi khac nhau (PPO theo `update`, SAC theo `step`), va
episode_log cua PPO v1-v3 chua co cot lech lan. Script nay quy ve mot thang chung —
so QUYET DINH cua policy (env step) — va danh dau ro o nao thieu du lieu thay vi
de nguoi doc tu doan.

Sinh ra:
  compare_episodes.csv  moi dong = 1 episode, co cot algo/run/env_step/town/...
  compare_updates.csv   moi dong = 1 moc log update, co cot algo/run/env_step/...
"""
import csv, io, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)

# (algo, ten run, thu muc, cot buoc, he so quy ve env step, ghi chu)
RUNS = [
    ("PPO", "ppo_v1", os.path.join(REPO, "drl_training", "runs", "ppo_v1"), "update", 1024, "reward raw, penalty 300, chi Town04"),
    ("PPO", "ppo_v2", os.path.join(REPO, "drl_training", "runs", "ppo_v2"), "update", 1024, "reward raw, penalty 1000"),
    ("PPO", "ppo_v3", os.path.join(REPO, "drl_training", "runs", "ppo_v3"), "update", 1024, "reward raw, w_lane 8"),
    ("PPO", "ppo_v4", os.path.join(REPO, "drl_training", "runs", "ppo_v4"), "update", 1024, "reward chuan hoa, 4 town — MO HINH DUOC CHON"),
    ("PPO", "ppo_v5", os.path.join(REPO, "drl_training", "runs", "ppo_v5"), "update", 1024, "them junction_off_lane_factor"),
    ("SAC", "sac_a", os.path.join(ROOT, "runs", "sac_a"), "step", 1, "bc_coef 2.5"),
    ("SAC", "sac_b", os.path.join(ROOT, "runs", "sac_b"), "step", 1, "bc_coef 1.0"),
    ("SAC", "sac_d", os.path.join(ROOT, "runs", "sac_d"), "step", 1, "kien truc Q moi + gamma 0.95"),
    ("SAC", "sac_e", os.path.join(ROOT, "runs", "sac_e"), "step", 1, "khong rang buoc — SAC dung ban chat"),
]

EP_OUT = ["algo", "run", "env_step", "town", "terminate_reason", "episode_len",
          "episode_reward", "mean_abs_lane_offset_road", "junction_steps", "ghi_chu"]
UP_OUT = ["algo", "run", "env_step", "explained_variance", "mean_episode_reward",
          "steps_per_sec", "entropy", "ghi_chu"]


def read(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def main():
    eps, ups = [], []
    for algo, run, d, key, mult, note in RUNS:
        for row in read(os.path.join(d, "episode_log.csv")):
            eps.append({
                "algo": algo, "run": run, "env_step": int(row[key]) * mult,
                "town": row.get("town", ""), "terminate_reason": row["terminate_reason"],
                "episode_len": row["episode_len"], "episode_reward": row["episode_reward"],
                "mean_abs_lane_offset_road": row.get("mean_abs_lane_offset_road", ""),
                "junction_steps": row.get("junction_steps", ""), "ghi_chu": note,
            })
        for row in read(os.path.join(d, "update_log.csv")):
            ups.append({
                "algo": algo, "run": run, "env_step": int(row[key]) * mult,
                "explained_variance": row.get("explained_variance", ""),
                "mean_episode_reward": row.get("mean_episode_reward", ""),
                "steps_per_sec": row.get("steps_per_sec", ""),
                "entropy": row.get("entropy", ""), "ghi_chu": note,
            })
    for name, rows, fields in (("compare_episodes.csv", eps, EP_OUT),
                               ("compare_updates.csv", ups, UP_OUT)):
        p = os.path.join(ROOT, name)
        h = io.open(p, "w", encoding="utf-8", newline="")
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r)
        h.close()
        print("%-22s %5d dong" % (name, len(rows)))
    runs_seen = {}
    for r in eps:
        runs_seen.setdefault((r["algo"], r["run"]), 0)
        runs_seen[(r["algo"], r["run"])] += 1
    print()
    for (a, rn), n in sorted(runs_seen.items()):
        print("   %-4s %-8s %4d episode" % (a, rn, n))


if __name__ == "__main__":
    main()
