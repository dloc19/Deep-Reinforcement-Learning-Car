# -*- coding: utf-8 -*-
"""Gop ket qua danh gia cua IL / PPO / SAC tren cung bon ban do thanh MOT bang.

Ba mo hinh nay duoc danh gia boi cung `evaluate.py`, cung so episode, cung mot bo
`terminate_reason` — nen cac cot o day so sanh duoc truc tiep. `mean_abs_lane_offset_road`
(khong phai `mean_abs_lane_offset`) la cot dung de so bam lan: trong nga tu `lane_offset_m`
duoc suy tu "lan gan nhat" nen nhay sang nhanh vuong goc, thanh so rac.

KHONG co cot reward o day, va do la co y. Baseline IL duoc do duoi ham reward cua ppo_v1
(reward tho, cong don theo m/s), con PPO v4 va SAC do duoi `reward_mode: "normalized"` —
hai thang do khac nhau hoan toan (IL Town04 = 3972 vs PPO Town04 = 308 khong co nghia la IL
tot gap 13 lan). Ba cot con lai — ty le va cham, lech lan, do dai episode — la dai luong
vat ly do tu mo phong, khong phu thuoc ham reward, nen so sanh duoc giua ca ba mo hinh.
"""
import csv, io, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
TOWNS = ["Town01", "Town03", "Town04", "Town05"]

SOURCES = [
    ("IL",  "behavior cloning (khoi tao)", {
        "Town01": os.path.join(REPO, "drl_training/runs/ppo_v1/eval_il_town01.csv"),
        "Town04": os.path.join(REPO, "drl_training/runs/ppo_v1/eval_il_Town04.csv"),
        "Town05": os.path.join(REPO, "drl_training/runs/ppo_v1/eval_il_Town05.csv")}),
    ("PPO", "ppo_v4 (mo hinh trien khai)", {
        t: os.path.join(REPO, "drl_training/runs/ppo_v4/eval_%s.csv" % t) for t in TOWNS}),
    ("SAC", "sac_b, bc_coef 1.0 (SAC tot nhat)", {
        t: os.path.join(ROOT, "runs/sac_b/eval_%s.csv" % t) for t in TOWNS}),
]


def doc(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    if not rows:
        return None
    off = [float(r["mean_abs_lane_offset_road"]) for r in rows
           if r["mean_abs_lane_offset_road"] not in ("", "nan")]
    return {
        "n": len(rows),
        "va_cham_pct": 100.0 * sum(1 for r in rows if r["collided"].lower() == "true") / len(rows),
        "lech_lan_m": float(np.mean(off)) if off else float("nan"),
        "dai_buoc": float(np.mean([float(r["length"]) for r in rows])),
    }


def main():
    out = []
    print("%-5s %-8s %4s %10s %12s %10s" % (
        "Model", "Ban do", "n", "va cham%", "lech lan m", "dai buoc"))
    print("-" * 55)
    for algo, ghi_chu, paths in SOURCES:
        for town in TOWNS:
            s = doc(paths.get(town, ""))
            if s is None:
                continue
            out.append(dict(s, thuat_toan=algo, ban_do=town, ghi_chu=ghi_chu))
            print("%-5s %-8s %4d %9.1f%% %12.3f %10.1f" % (
                algo, town, s["n"], s["va_cham_pct"], s["lech_lan_m"], s["dai_buoc"]))
        print()

    dest = os.path.join(ROOT, "compare_eval.csv")
    fields = ["thuat_toan", "ban_do", "n", "va_cham_pct", "lech_lan_m", "dai_buoc",
              "ghi_chu"]
    with io.open(dest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r[k] for k in fields})
    print("Da ghi:", dest, "(%d dong)" % len(out))


if __name__ == "__main__":
    main()
