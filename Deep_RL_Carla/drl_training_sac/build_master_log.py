# -*- coding: utf-8 -*-
"""Gop TOAN BO log cua ca hai thuat toan vao MOT file: `compare_all.csv`.

Ba nguon co ba do phan giai khac nhau, nen cot `loai` cho biet mot dong la gi:

    loai=episode   mot episode huan luyen        (10 lan chay)
    loai=update    mot lan cap nhat mang         (10 lan chay)
    loai=eval      mot lo danh gia 30 episode    (11 dong: IL/PPO/SAC x cac ban do)

Truc x chung la `env_step` = so buoc moi truong da di. PPO ghi log theo `update` nen
duoc nhan voi n_steps=1024; SAC ghi theo `step` nen giu nguyen. Nho vay hai duong hoc
cua PPO va SAC ve duoc len cung mot truc.

BA CAI BAY KHI VE BIEU DO - doc truoc khi ket luan:

1. `reward` KHONG so sanh duoc giua `ham_reward=raw` va `ham_reward=normalized`.
   ppo_v1/v2/v3 chay duoi reward tho (cong don theo m/s, phat va cham 300-1000);
   ppo_v4/v5 va toan bo SAC chay duoi `reward_mode: normalized`. Ve chung mot truc la
   sai - vi du IL Town04 dat 3972 con PPO v4 dat 308 khong co nghia la IL tot gap 13
   lan, chi la hai thuoc do khac nhau. Loc theo `ham_reward` truoc khi ve.

2. `actor_loss` va `critic_loss` KHONG so sanh duoc giua PPO va SAC. Cua PPO la clipped
   surrogate va MSE cua V(s); cua SAC la (alpha*logp - Q) va TD error cua twin-Q. Cung
   ten cot nhung la hai dai luong khac han ve dinh nghia lan thang do. Chi nen ve TUNG
   thuat toan rieng de xem xu huong theo thoi gian, dung dat canh nhau.

3. `explained_variance`, `do_dai`, `lech_lan_m`, `va_cham_pct`, `ket_thuc` thi SO SANH
   DUOC thoai mai - deu la dai luong vat ly do tu mo phong, hoac dinh nghia giong het
   nhau o ca hai thuat toan. Cac bieu do KET LUAN nen dua tren nhung cot nay.

Cac lan chay smoke test (`*_smoke*`) bi loai: chung chi vai chuc buoc de kiem tra code
khong crash, ve len bieu do chi lam nhieu.

Chay:  python build_master_log.py
"""

import csv
import io
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)


def d(*parts):
    return os.path.join(*parts)


# Cot `chuoi` gom cac lan chay NOI TIEP nhau thanh mot duong lien tuc khi ve bieu do.
#
# Vi sao can: `ppo_v2` bat dau o env_step 103424 chu khong phai 0, vi no chay TIEP tu
# checkpoint cua `ppo_v1` — `global_step` cong don qua cac phien. Ve moi lan chay thanh mot
# duong rieng se ra nam doan roi rac thay vi hai duong lien tuc, va nguoi doc bieu do se
# tuong day la nam thi nghiem doc lap.
#
# Gia tri nay duoc GO TAY chu khong suy ra tu du lieu. Ban dau tac gia thu suy tu khoang
# step ("lan chay nao bat dau trong khoang cua lan chay khac thi la con cua no") va no cho
# ket qua SAI: gan `ppo_v5` vao nhanh v3 thay vi v4, va xau ca nam lan chay SAC thanh mot
# chuoi — vi cac lan SAC deu bat dau o step 1000 nen khoang cua chung chong len nhau hoan
# toan. Quan he ke thua khong nam trong con so step; no nam o cho lan chay do duoc khoi
# dong bang --resume tu dau. Ham `kiem_tra_chuoi()` doi chieu lai khai bao nay voi du lieu
# de neu ai them lan chay moi ma dan nham nhan thi script bao loi thay vi ve ra bieu do sai.
#
# Bang chung cho tung nhan (doc tu checkpoint thap nhat + step dau cua log):
#   ppo_v1 1024   moi | ppo_v2 103424 tiep v1 | ppo_v3 165888 tiep v2
#   ppo_v4 1024   moi | ppo_v5 200704 tiep v4 (v4 ket thuc o 199680, dung mot khoang log)
#   sac_a/c/d/e   co checkpoint tu 5000  -> deu chay MOI tu IL
#   sac_b         checkpoint thap nhat 15000, log bat dau 11000 -> NHANH ra tu sac_a@10000
#                 (alpha cua no o step 12000 la 0.00999, gan gia tri khoi tao, trong khi
#                 sac_a cung luc da la 0.00858 — tuc khong ke thua optimizer, chi ke thua
#                 trong so; day la mot nhanh cau hinh khac chu khong phai chay tiep)
#
# (thuat_toan, lan_chay, thu_muc, cot_step, he_so_nhan, ham_reward, chuoi, ghi_chu)
RUNS = [
    ("PPO", "ppo_v1", d(REPO, "drl_training", "runs", "ppo_v1"), "update", 1024, "raw",
     "PPO-A reward tho", "reward tho, phat va cham 300, chi Town04"),
    ("PPO", "ppo_v2", d(REPO, "drl_training", "runs", "ppo_v2"), "update", 1024, "raw",
     "PPO-A reward tho", "reward tho, phat va cham 1000"),
    ("PPO", "ppo_v3", d(REPO, "drl_training", "runs", "ppo_v3"), "update", 1024, "raw",
     "PPO-A reward tho", "reward tho, w_lane_offset 8"),
    ("PPO", "ppo_v4", d(REPO, "drl_training", "runs", "ppo_v4"), "update", 1024, "normalized",
     "PPO-B reward chuan hoa", "reward chuan hoa, xoay 4 ban do - MO HINH DUOC CHON"),
    ("PPO", "ppo_v5", d(REPO, "drl_training", "runs", "ppo_v5"), "update", 1024, "normalized",
     "PPO-B reward chuan hoa", "them junction_off_lane_factor"),
    ("SAC", "sac_a", d(ROOT, "runs", "sac_a"), "step", 1, "normalized",
     "SAC-A bc 2.5", "bc_coef 2.5, gamma 0.99"),
    ("SAC", "sac_b", d(ROOT, "runs", "sac_b"), "step", 1, "normalized",
     "SAC-B bc 1.0", "bc_coef 1.0, nhanh tu SAC-A @10k - SAC TOT NHAT"),
    ("SAC", "sac_c", d(ROOT, "runs", "sac_c"), "step", 1, "normalized",
     "SAC-C explore", "them explore_epsilon 0.25 de da dang hoa replay buffer"),
    ("SAC", "sac_d", d(ROOT, "runs", "sac_d"), "step", 1, "normalized",
     "SAC-D kien truc Q", "kien truc Q moi (action_emb 2 -> 32) + gamma 0.95"),
    ("SAC", "sac_e", d(ROOT, "runs", "sac_e"), "step", 1, "normalized",
     "SAC-E khong rang buoc", "bc_coef 0, log_std tu do - SAC dung ban chat, pha policy"),
]

# Chuoi nao gom NHIEU lan chay thi cac doan step cua chung phai noi duoc voi nhau. Kiem tra
# nay chan dung loi da xay ra mot lan: dan nham `ppo_v5` vao nhanh reward tho.
CHUOI_NHIEU_LAN = {"PPO-A reward tho": ["ppo_v1", "ppo_v2", "ppo_v3"],
                   "PPO-B reward chuan hoa": ["ppo_v4", "ppo_v5"]}

EVAL = [
    ("IL", "il", "raw", "IL khoi tao", {
        "Town01": d(REPO, "drl_training/runs/ppo_v1/eval_il_town01.csv"),
        "Town04": d(REPO, "drl_training/runs/ppo_v1/eval_il_Town04.csv"),
        "Town05": d(REPO, "drl_training/runs/ppo_v1/eval_il_Town05.csv")},
     "behavior cloning - diem khoi tao chung cua ca hai thuat toan"),
    ("PPO", "ppo_v4", "normalized", "PPO-B reward chuan hoa",
     dict((t, d(REPO, "drl_training/runs/ppo_v4/eval_%s.csv" % t))
          for t in ("Town01", "Town03", "Town04", "Town05")),
     "mo hinh trien khai"),
    ("SAC", "sac_b", "normalized", "SAC-B bc 1.0",
     dict((t, d(ROOT, "runs/sac_b/eval_%s.csv" % t))
          for t in ("Town01", "Town03", "Town04", "Town05")),
     "checkpoint SAC tot nhat"),
]

FIELDS = [
    "loai", "thuat_toan", "lan_chay", "chuoi", "env_step", "ham_reward", "ban_do",
    "do_dai", "reward", "lech_lan_m", "junction_steps", "ket_thuc",
    "va_cham_pct", "n",
    "actor_loss", "critic_loss", "entropy", "explained_variance",
    "alpha", "mean_q", "approx_kl", "clip_fraction", "steps_per_sec",
    "ghi_chu",
]


def doc_csv(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def gom_train(rows, dem):
    for algo, run, folder, step_col, mult, reward_mode, chuoi, note in RUNS:
        n_ep = n_up = 0

        for r in doc_csv(d(folder, "episode_log.csv")):
            rows.append({
                "loai": "episode", "thuat_toan": algo, "lan_chay": run,
                "chuoi": chuoi,
                "env_step": int(float(r.get(step_col) or 0)) * mult,
                "ham_reward": reward_mode, "ban_do": r.get("town", ""),
                "do_dai": r.get("episode_len", ""),
                "reward": r.get("episode_reward", ""),
                "lech_lan_m": r.get("mean_abs_lane_offset_road", ""),
                "junction_steps": r.get("junction_steps", ""),
                "ket_thuc": r.get("terminate_reason", ""),
                "ghi_chu": note,
            })
            n_ep += 1

        for r in doc_csv(d(folder, "update_log.csv")):
            # PPO ghi san `global_step` (= update * n_steps), nen dung thang. SAC khong co
            # cot do, `step` cua no da chinh la so buoc moi truong -> he so nhan 1.
            if r.get("global_step"):
                step = int(float(r["global_step"]))
            else:
                step = int(float(r.get(step_col) or 0)) * mult
            rows.append({
                "loai": "update", "thuat_toan": algo, "lan_chay": run,
                "chuoi": chuoi,
                "env_step": step, "ham_reward": reward_mode, "ban_do": "",
                "reward": r.get("mean_episode_reward", ""),
                # PPO goi la policy_loss/value_loss, SAC goi la actor_loss/critic_loss.
                # Gop vao mot cot cho de ve, nhung xem cai bay so 2 o docstring: cung ten
                # KHONG co nghia la cung dai luong.
                "actor_loss": r.get("actor_loss") or r.get("policy_loss", ""),
                "critic_loss": r.get("critic_loss") or r.get("value_loss", ""),
                "entropy": r.get("entropy", ""),
                "explained_variance": r.get("explained_variance", ""),
                "alpha": r.get("alpha", ""),
                "mean_q": r.get("mean_q", ""),
                "approx_kl": r.get("approx_kl", ""),
                "clip_fraction": r.get("clip_fraction", ""),
                "steps_per_sec": r.get("steps_per_sec", ""),
                "ghi_chu": note,
            })
            n_up += 1

        if n_ep or n_up:
            dem[run] = (algo, n_ep, n_up)


def gom_eval(rows):
    for algo, run, reward_mode, chuoi, paths, note in EVAL:
        for town in sorted(paths):
            er = doc_csv(paths[town])
            if not er:
                continue
            off = [float(x["mean_abs_lane_offset_road"]) for x in er
                   if x["mean_abs_lane_offset_road"] not in ("", "nan")]
            va_cham = sum(1 for x in er if x["collided"].lower() == "true")
            rows.append({
                "loai": "eval", "thuat_toan": algo, "lan_chay": run,
                "chuoi": chuoi,
                "env_step": "", "ham_reward": reward_mode, "ban_do": town,
                "n": len(er),
                "va_cham_pct": round(100.0 * va_cham / len(er), 4),
                "lech_lan_m": round(float(np.mean(off)), 6) if off else "",
                "do_dai": round(float(np.mean([float(x["length"]) for x in er])), 4),
                "junction_steps": sum(int(float(x["junction_steps"])) for x in er),
                "ghi_chu": note,
            })


def kiem_tra_chuoi(rows):
    """Doi chieu nhan `chuoi` da go tay voi khoang step thuc te trong du lieu.

    Voi mot chuoi gom nhieu lan chay, cac doan phai NOI duoc: lan sau bat dau ngay sat
    diem ket thuc cua lan truoc (chenh khong qua vai khoang ghi log). Neu lech xa thi hoac
    nhan bi dan nham, hoac lan chay do that ra la mot thi nghiem doc lap — ca hai truong
    hop deu lam bieu do sai, nen dung lai va bao thay vi ghi file.
    """
    khoang = {}
    for r in rows:
        if r["loai"] not in ("episode", "update") or r["env_step"] == "":
            continue
        st = int(r["env_step"])
        lo, hi = khoang.get(r["lan_chay"], (st, st))
        khoang[r["lan_chay"]] = (min(lo, st), max(hi, st))

    for chuoi, ds in CHUOI_NHIEU_LAN.items():
        for truoc, sau in zip(ds, ds[1:]):
            if truoc not in khoang or sau not in khoang:
                continue
            hi_truoc = khoang[truoc][1]
            lo_sau = khoang[sau][0]
            # Cho phep lech mot vai khoang ghi log theo ca hai chieu: checkpoint cuoi cung
            # duoc luu thuong som hon dong log cuoi cung mot chut.
            if abs(lo_sau - hi_truoc) > 4096:
                raise SystemExit(
                    "Chuoi '%s' khong noi duoc: %s ket thuc o step %d nhung %s bat dau o "
                    "step %d (lech %d). Kiem tra lai nhan `chuoi` trong RUNS."
                    % (chuoi, truoc, hi_truoc, sau, lo_sau, lo_sau - hi_truoc))
    return khoang


def main():
    rows, dem = [], {}
    gom_train(rows, dem)
    khoang = kiem_tra_chuoi(rows)
    gom_eval(rows)

    rows.sort(key=lambda r: (
        {"episode": 0, "update": 1, "eval": 2}[r["loai"]],
        r["lan_chay"],
        r["env_step"] if r["env_step"] != "" else 0,
        r.get("ban_do", "")))

    dest = d(ROOT, "compare_all.csv")
    with io.open(dest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, r.get(k, "")) for k in FIELDS))

    print("Da ghi %s" % dest)
    print("  %d dong x %d cot" % (len(rows), len(FIELDS)))
    print()
    print("  %-8s %-6s %8s %8s" % ("lan chay", "thuat", "episode", "update"))
    print("  " + "-" * 34)
    for run in sorted(dem):
        algo, ne, nu = dem[run]
        print("  %-8s %-6s %8d %8d" % (run, algo, ne, nu))
    print()
    print("  Chuoi (ve moi chuoi thanh MOT duong lien tuc tren truc env_step):")
    nhom = {}
    for _a, run, _f, _sc, _m, _rm, chuoi, _n in RUNS:
        if run in khoang:
            nhom.setdefault(chuoi, []).append((khoang[run][0], run, khoang[run][1]))
    for c in sorted(nhom):
        ds = sorted(nhom[c])
        print("    %-24s %-26s step %6d -> %6d" % (
            c, " -> ".join(r for _, r, _h in ds), ds[0][0], ds[-1][2]))
    print()
    for loai in ("episode", "update", "eval"):
        print("  loai=%-8s %5d dong" % (
            loai, sum(1 for r in rows if r["loai"] == loai)))
    print()
    print("  SO SANH DUOC giua PPO va SAC: explained_variance, do_dai, lech_lan_m,")
    print("                                va_cham_pct, ket_thuc, steps_per_sec")
    print("  KHONG so sanh duoc:           reward (loc theo ham_reward truoc),")
    print("                                actor_loss, critic_loss (khac dinh nghia)")


if __name__ == "__main__":
    main()
