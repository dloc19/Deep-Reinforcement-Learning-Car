# -*- coding: utf-8 -*-
"""Gom log cua RIENG PPO thanh bo du lieu de ve bieu do danh gia.

Tach khoi `drl_training_sac/build_master_log.py` (gom ca hai thuat toan) vi hai cai bay
lon nhat cua file gop deu sinh ra tu viec dat PPO canh SAC: `actor_loss`/`critic_loss`
cung ten nhung khac dinh nghia, va nguoi doc de tuong hai duong hoc cung mot thang do.
O day chi co PPO nen giu dung ten goc `policy_loss`/`value_loss`, khong con nhap nhang.

Sinh ra ba file, moi file mot muc dich:

    ppo_all.csv            moi dong = 1 episode / 1 update / 1 lo eval. Cot `loai` phan biet.
                           Dung cho duong hoc theo `env_step`.
    ppo_eval.csv           moi dong = 1 (mo hinh, ban do). Da co san khoang tin cay.
                           Dung cho bieu do cot ket luan.
    ppo_eval_episodes.csv  moi dong = 1 episode danh gia. Dung cho box plot, hoac de tu
                           tinh lai kiem dinh ma khong phai mo lai tung file rieng.

BON CAI BAY - doc truoc khi ket luan:

1. `reward` KHONG so sanh duoc giua `ham_reward=raw` va `ham_reward=normalized`.
   ppo_v1/v2/v3 chay duoi reward tho (cong don theo m/s, phat va cham 300-1000): reward
   episode trung binh 3334-4417. ppo_v4/v5 chay duoi `reward_mode: normalized`: trung binh
   145-287. Hai con so nay do hai thu khac nhau, ve chung mot truc la sai. Loc theo
   `ham_reward` truoc khi ve, hoac tach lam hai do thi.

2. `lech_lan_m` va `ban_do` CHI CO tu ppo_v4 tro di. episode_log cua v1-v3 chua ghi hai cot
   nay (do phu 0%), va `explained_variance` thi ppo_v1 cung chua ghi. Duong hoc lech lan va
   moi phep tach theo ban do bat dau tu v4; duong EV bat dau tu v2. O dau cot rong thi de
   rong, dung noi qua cho lien.

3. MOC KHOI TAO KHONG PHAI IL. Ba file `ppo_v1/eval_il_*.csv` co ten goi y behavior cloning,
   nhung cot `checkpoint` cua chung ghi `ppo_update_000005.pt` - tuc PPO sau 5 lan cap nhat,
   6144 env step, duoi reward tho. `evaluate.py` khong co duong nao nap trong so BC lam
   policy: no bat buoc `--resume` mot checkpoint PPO/SAC, con `il_checkpoint` chi dung de
   dung ObservationContract. Nen o day moc do duoc dan nhan `PPO@6k` chu khong phai `IL`,
   va KHONG duoc dung de ket luan "PPO tot/te hon IL" - muon ket luan do thi phai chay
   danh gia tren dung checkpoint IL. Ham `kiem_tra_checkpoint()` doi chieu lai dieu nay voi
   du lieu de khong ai vo tinh dan lai nhan cu.

4. Moc khoi tao co n=10 tren 3 ban do, con v4/v5 co n=30 tren 4 ban do. Khoang tin cay cua
   no rong hon han va Town03 khong co cot doi chung - cot `n` va `va_cham_ci_*` noi ro dieu
   do, dung ve thanh cac cot trong nhu nhau.

   Tin tot: commit chuan hoa reward (fe90ccb) chi doi TRONG SO reward, khong doi
   `max_episode_steps` (500) hay `off_lane_patience_steps` (10). Nen `va_cham_pct`,
   `lech_lan_m`, `dai_buoc` van so sanh duoc giua hai lo danh gia; chi `reward` la khong.

v1->v2->v3 va v4->v5 la cac lan chay `--resume` noi tiep, KHONG phai seed doc lap. Khong
co thanh sai so theo seed cho duong hoc; chi bieu do cot danh gia moi co khoang tin cay
(o muc episode, n=30).

Chay:  python build_ppo_log.py
"""

import csv
import io
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def d(*parts):
    return os.path.join(*parts)


# Cot `chuoi` gom cac lan chay NOI TIEP nhau thanh mot duong lien tuc khi ve bieu do.
#
# Vi sao can: `ppo_v2` bat dau o env_step 103424 chu khong phai 0, vi no chay TIEP tu
# checkpoint cua `ppo_v1` - `global_step` cong don qua cac phien. Ve moi lan chay thanh mot
# duong rieng se ra nam doan roi rac thay vi hai duong lien tuc, va nguoi doc bieu do se
# tuong day la nam thi nghiem doc lap.
#
# Gia tri nay duoc GO TAY chu khong suy ra tu du lieu: quan he ke thua khong nam trong con
# so step, no nam o cho lan chay do duoc khoi dong bang --resume tu dau. `kiem_tra_chuoi()`
# doi chieu lai khai bao voi du lieu de neu ai them lan chay moi ma dan nham nhan thi
# script bao loi thay vi ve ra bieu do sai.
#
#   ppo_v1 1024 moi | ppo_v2 103424 tiep v1 | ppo_v3 165888 tiep v2
#   ppo_v4 1024 moi | ppo_v5 200704 tiep v4 (v4 ket thuc o 199680, dung mot khoang log)
#
# (lan_chay, thu_muc, ham_reward, chuoi, ghi_chu)
RUNS_TRAIN = [
    ("ppo_v1", d(RUNS, "ppo_v1"), "raw", "PPO-A reward tho",
     "reward tho, phat va cham 300, chi Town04"),
    ("ppo_v2", d(RUNS, "ppo_v2"), "raw", "PPO-A reward tho",
     "reward tho, phat va cham 1000"),
    ("ppo_v3", d(RUNS, "ppo_v3"), "raw", "PPO-A reward tho",
     "reward tho, w_lane_offset 8"),
    ("ppo_v4", d(RUNS, "ppo_v4"), "normalized", "PPO-B reward chuan hoa",
     "reward chuan hoa, xoay 4 ban do - MO HINH DUOC CHON"),
    ("ppo_v5", d(RUNS, "ppo_v5"), "normalized", "PPO-B reward chuan hoa",
     "them junction_off_lane_factor"),
]

CHUOI_NHIEU_LAN = {"PPO-A reward tho": ["ppo_v1", "ppo_v2", "ppo_v3"],
                   "PPO-B reward chuan hoa": ["ppo_v4", "ppo_v5"]}

N_STEPS = 1024  # rollout cua PPO; update_log da co san `global_step` nen chi dung cho episode_log

# Cac lo danh gia. `checkpoint_cho` la ten file checkpoint MONG DOI - `kiem_tra_checkpoint()`
# doi chieu no voi cot `checkpoint` trong file eval. Day la cho da tung sai mot lan (moc
# khoi tao bi dan nhan IL trong khi trong so la PPO@6k), nen kiem tra chu dung tin ten file.
#
# (nhan, lan_chay, ham_reward, chuoi, checkpoint_cho, {ban_do: duong dan}, ghi_chu)
EVAL = [
    ("PPO@6k", "ppo_v1_u5", "raw", "PPO-A reward tho", "ppo_update_000005.pt",
     {"Town01": d(RUNS, "ppo_v1", "eval_il_town01.csv"),
      "Town04": d(RUNS, "ppo_v1", "eval_il_Town04.csv"),
      "Town05": d(RUNS, "ppo_v1", "eval_il_Town05.csv")},
     "moc som: PPO sau 5 update (6144 step), reward tho - KHONG phai IL, xem cai bay 3"),
    ("PPO-v4", "ppo_v4", "normalized", "PPO-B reward chuan hoa", "ppo_latest.pt",
     dict((t, d(RUNS, "ppo_v4", "eval_%s.csv" % t))
          for t in ("Town01", "Town03", "Town04", "Town05")),
     "mo hinh trien khai"),
    ("PPO-v5", "ppo_v5", "normalized", "PPO-B reward chuan hoa", "ppo_latest.pt",
     dict((t, d(RUNS, "ppo_v5", "eval_%s.csv" % t))
          for t in ("Town01", "Town03", "Town04", "Town05")),
     "them junction_off_lane_factor - doi chung cho lua chon v4"),
]

FIELDS = [
    "loai", "nhan", "lan_chay", "chuoi", "env_step", "ham_reward", "ban_do",
    "do_dai", "reward", "lech_lan_m", "junction_steps", "ket_thuc",
    "va_cham_pct", "n",
    "policy_loss", "value_loss", "entropy", "explained_variance",
    "approx_kl", "clip_fraction", "steps_per_sec",
    "ghi_chu",
]

EVAL_FIELDS = [
    "nhan", "lan_chay", "ban_do", "n", "ham_reward",
    "va_cham_pct", "va_cham_ci_lo", "va_cham_ci_hi",
    "lech_lan_m", "lech_lan_sd", "lech_lan_ci_lo", "lech_lan_ci_hi",
    "dai_buoc", "dai_buoc_sd", "off_lane_steps", "junction_steps",
    "checkpoint", "ngay_chay", "ghi_chu",
]

EP_FIELDS = [
    "nhan", "lan_chay", "ban_do", "episode", "reward", "do_dai", "va_cham",
    "off_lane_steps", "junction_steps", "lech_lan_m", "ket_thuc", "ham_reward",
]

# t_{0.975, df} cho vai bac tu do hay gap. n=30 -> df=29 -> 2.045, lech ro so voi 1.96 nen
# dung xap xi chuan o co mau nay se lam khoang tin cay hep gia.
T975 = {9: 2.262, 10: 2.228, 19: 2.093, 29: 2.045, 30: 2.042, 39: 2.023, 59: 2.001}


def t_crit(df):
    if df <= 0:
        return float("nan")
    if df in T975:
        return T975[df]
    for k in sorted(T975):          # bac tu do nho nhat >= df -> khoang rong hon, ben an toan
        if k >= df:
            return T975[k]
    return 1.96


def wilson(k, n, z=1.96):
    """Khoang tin cay Wilson cho ti le, tinh theo %.

    Dung Wilson chu khong phai Wald vi ti le va cham cham day bien: Town01 cua v4/v5 la
    0/30. Wald cho ra khoang [0, 0] - mot khang dinh "chac chan khong bao gio va cham"
    ma 30 episode khong the chung minh. Wilson cho khoang rong hon o bien, dung hon.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = float(k) / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100.0 * max(0.0, ctr - half), 100.0 * min(1.0, ctr + half)


def doc_csv(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def so(v):
    """Doi chuoi sang float, tra None neu rong / nan - de cot rong khong thanh so 0."""
    if v is None or v.strip().lower() in ("", "nan", "none"):
        return None
    return float(v)


def gom_train(rows, dem):
    for run, folder, reward_mode, chuoi, note in RUNS_TRAIN:
        n_ep = n_up = 0

        for r in doc_csv(d(folder, "episode_log.csv")):
            rows.append({
                "loai": "episode", "nhan": run, "lan_chay": run, "chuoi": chuoi,
                "env_step": int(float(r.get("update") or 0)) * N_STEPS,
                "ham_reward": reward_mode,
                "ban_do": r.get("town", ""),                            # rong o v1-v3
                "do_dai": r.get("episode_len", ""),
                "reward": r.get("episode_reward", ""),
                "lech_lan_m": r.get("mean_abs_lane_offset_road", ""),   # rong o v1-v3
                "junction_steps": r.get("junction_steps", ""),
                "ket_thuc": r.get("terminate_reason", ""),
                "ghi_chu": note,
            })
            n_ep += 1

        for r in doc_csv(d(folder, "update_log.csv")):
            rows.append({
                "loai": "update", "nhan": run, "lan_chay": run, "chuoi": chuoi,
                "env_step": int(float(r["global_step"])), "ham_reward": reward_mode,
                "ban_do": "",
                "reward": r.get("mean_episode_reward", ""),
                "policy_loss": r.get("policy_loss", ""),
                "value_loss": r.get("value_loss", ""),
                "entropy": r.get("entropy", ""),
                "explained_variance": r.get("explained_variance", ""),  # rong o v1
                "approx_kl": r.get("approx_kl", ""),
                "clip_fraction": r.get("clip_fraction", ""),
                "steps_per_sec": r.get("steps_per_sec", ""),
                "ghi_chu": note,
            })
            n_up += 1

        if n_ep or n_up:
            dem[run] = (n_ep, n_up)


def gom_eval(rows, eval_rows, ep_rows):
    for nhan, run, reward_mode, chuoi, _ckpt_cho, paths, note in EVAL:
        for town in sorted(paths):
            er = doc_csv(paths[town])
            if not er:
                continue
            n = len(er)
            va_cham = sum(1 for x in er if x["collided"].lower() == "true")
            off = [so(x["mean_abs_lane_offset_road"]) for x in er]
            off = [v for v in off if v is not None]
            L = [float(x["length"]) for x in er]
            olane = sum(int(float(x["off_lane_steps"])) for x in er)
            jsteps = sum(int(float(x["junction_steps"])) for x in er)

            col_lo, col_hi = wilson(va_cham, n)
            if len(off) > 1:
                sd = float(np.std(off, ddof=1))
                off_m = float(np.mean(off))
                half = t_crit(len(off) - 1) * sd / math.sqrt(len(off))
                off_lo, off_hi = off_m - half, off_m + half
            else:
                off_m = float(np.mean(off)) if off else float("nan")
                sd = off_lo = off_hi = float("nan")

            ckpt = sorted(set(os.path.basename(x["checkpoint"]) for x in er))
            ngay = sorted(set(x["eval_run_utc"][:10] for x in er))

            rows.append({
                "loai": "eval", "nhan": nhan, "lan_chay": run, "chuoi": chuoi,
                "env_step": "", "ham_reward": reward_mode, "ban_do": town,
                "n": n, "va_cham_pct": round(100.0 * va_cham / n, 4),
                "lech_lan_m": round(off_m, 6) if off else "",
                "do_dai": round(float(np.mean(L)), 4),
                "junction_steps": jsteps,
                "ghi_chu": note,
            })

            eval_rows.append({
                "nhan": nhan, "lan_chay": run, "ban_do": town, "n": n,
                "ham_reward": reward_mode,
                "va_cham_pct": round(100.0 * va_cham / n, 4),
                "va_cham_ci_lo": round(col_lo, 4), "va_cham_ci_hi": round(col_hi, 4),
                "lech_lan_m": round(off_m, 6), "lech_lan_sd": round(sd, 6),
                "lech_lan_ci_lo": round(off_lo, 6), "lech_lan_ci_hi": round(off_hi, 6),
                "dai_buoc": round(float(np.mean(L)), 4),
                "dai_buoc_sd": round(float(np.std(L, ddof=1)), 4) if n > 1 else "",
                "off_lane_steps": olane, "junction_steps": jsteps,
                "checkpoint": ",".join(ckpt), "ngay_chay": ",".join(ngay),
                "ghi_chu": note,
            })

            for x in er:
                ep_rows.append({
                    "nhan": nhan, "lan_chay": run, "ban_do": town,
                    "episode": x["episode"], "reward": x["reward"],
                    "do_dai": x["length"],
                    "va_cham": 1 if x["collided"].lower() == "true" else 0,
                    "off_lane_steps": x["off_lane_steps"],
                    "junction_steps": x["junction_steps"],
                    "lech_lan_m": x["mean_abs_lane_offset_road"],
                    "ket_thuc": x["terminate_reason"], "ham_reward": reward_mode,
                })


def kiem_tra_chuoi(rows):
    """Doi chieu nhan `chuoi` da go tay voi khoang step thuc te trong du lieu.

    Voi mot chuoi gom nhieu lan chay, cac doan phai NOI duoc: lan sau bat dau ngay sat diem
    ket thuc cua lan truoc. Neu lech xa thi hoac nhan bi dan nham, hoac lan chay do that ra
    la mot thi nghiem doc lap - ca hai deu lam bieu do sai, nen dung lai va bao.
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
            hi_truoc, lo_sau = khoang[truoc][1], khoang[sau][0]
            if abs(lo_sau - hi_truoc) > 4 * N_STEPS:
                raise SystemExit(
                    "Chuoi '%s' khong noi duoc: %s ket thuc o step %d nhung %s bat dau o "
                    "step %d (lech %d). Kiem tra lai nhan `chuoi` trong RUNS_TRAIN."
                    % (chuoi, truoc, hi_truoc, sau, lo_sau, lo_sau - hi_truoc))
    return khoang


def kiem_tra_checkpoint():
    """Doi chieu checkpoint MONG DOI voi cot `checkpoint` thuc te trong tung file eval.

    Chan dung loi da xay ra: ba file `eval_il_*.csv` mang ten IL nhung trong so ben trong
    la `ppo_update_000005.pt`. Ten file khong phai bang chung; cot checkpoint moi la.
    """
    for nhan, _run, _rm, _ch, ckpt_cho, paths, _note in EVAL:
        for town in sorted(paths):
            er = doc_csv(paths[town])
            if not er:
                continue
            thuc = set(os.path.basename(x["checkpoint"]) for x in er)
            if thuc != set([ckpt_cho]):
                raise SystemExit(
                    "Nhan '%s' (%s) khai bao checkpoint '%s' nhung file ghi %s. Hoac sua "
                    "`checkpoint_cho` trong EVAL, hoac doi nhan cho dung trong so."
                    % (nhan, town, ckpt_cho, sorted(thuc)))


def do_phu(rows, loai, run, cot):
    rs = [r for r in rows if r["loai"] == loai and r["lan_chay"] == run]
    if not rs:
        return 0, 0
    co = sum(1 for r in rs if str(r.get(cot, "")).strip() not in ("", "nan"))
    return len(rs), 100 * co // len(rs)


def main():
    rows, eval_rows, ep_rows, dem = [], [], [], {}
    gom_train(rows, dem)
    khoang = kiem_tra_chuoi(rows)
    kiem_tra_checkpoint()
    gom_eval(rows, eval_rows, ep_rows)

    rows.sort(key=lambda r: (
        {"episode": 0, "update": 1, "eval": 2}[r["loai"]],
        r["lan_chay"],
        r["env_step"] if r["env_step"] != "" else 0,
        r.get("ban_do", "")))

    def ghi(ten, fields, ds):
        with io.open(d(ROOT, ten), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in ds:
                w.writerow(dict((k, r.get(k, "")) for k in fields))
        print("  %-24s %5d dong x %2d cot" % (ten, len(ds), len(fields)))

    print("Da ghi vao %s" % ROOT)
    ghi("ppo_all.csv", FIELDS, rows)
    ghi("ppo_eval.csv", EVAL_FIELDS, eval_rows)
    ghi("ppo_eval_episodes.csv", EP_FIELDS, ep_rows)

    print()
    print("  %-8s %8s %8s" % ("lan chay", "episode", "update"))
    print("  " + "-" * 26)
    for run in sorted(dem):
        print("  %-8s %8d %8d" % (run, dem[run][0], dem[run][1]))

    print()
    print("  Chuoi (ve moi chuoi thanh MOT duong lien tuc tren truc env_step):")
    nhom = {}
    for run, _f, _rm, chuoi, _n in RUNS_TRAIN:
        if run in khoang:
            nhom.setdefault(chuoi, []).append((khoang[run][0], run, khoang[run][1]))
    for c in sorted(nhom):
        ds = sorted(nhom[c])
        print("    %-24s %-26s step %6d -> %6d" % (
            c, " -> ".join(r for _, r, _h in ds), ds[0][0], ds[-1][2]))

    print()
    print("  Do phu cot (% dong co gia tri) - o dau 0% thi duong tuong ung khong ve duoc:")
    print("    %-8s %6s %11s %8s %6s %20s" % (
        "lan chay", "n_ep", "lech_lan_m", "ban_do", "n_up", "explained_variance"))
    for run, _f, _rm, _c, _n in RUNS_TRAIN:
        n_ep, p_off = do_phu(rows, "episode", run, "lech_lan_m")
        _, p_town = do_phu(rows, "episode", run, "ban_do")
        n_up, p_ev = do_phu(rows, "update", run, "explained_variance")
        print("    %-8s %6d %10d%% %7d%% %6d %19d%%" % (
            run, n_ep, p_off, p_town, n_up, p_ev))

    print()
    print("  Danh gia cuoi (ppo_eval.csv):")
    print("    %-7s %-7s %3s %20s %26s %8s" % (
        "nhan", "ban do", "n", "va cham % [CI95]", "lech lan m [CI95]", "dai buoc"))
    for r in eval_rows:
        print("    %-7s %-7s %3d %6.1f [%4.1f-%5.1f] %8.4f [%.4f-%.4f] %8.1f" % (
            r["nhan"], r["ban_do"], r["n"], r["va_cham_pct"], r["va_cham_ci_lo"],
            r["va_cham_ci_hi"], r["lech_lan_m"], r["lech_lan_ci_lo"],
            r["lech_lan_ci_hi"], r["dai_buoc"]))

    print()
    print("  NHAC: `PPO@6k` la PPO sau 5 update duoi reward tho, KHONG phai IL (cai bay 3).")
    print("        `reward` khong so sanh duoc giua raw va normalized (cai bay 1).")
    print("        `lech_lan_m` chi co tu ppo_v4, `explained_variance` tu ppo_v2 (cai bay 2).")


if __name__ == "__main__":
    main()
