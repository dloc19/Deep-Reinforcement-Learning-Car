# -*- coding: utf-8 -*-
"""Tong hop BANG KICH BAN TEST + TI LE HOAN THANH TUYEN DUONG tu cac lan danh gia da chay.

KHONG chay lai CARLA, khong can torch: chi doc lai cac file `eval_*.csv` do `evaluate.py`
sinh ra trong `runs/`, nen chay duoc tren may viet bao cao.

Xuat ra:
  - docs/ket_qua_ti_le_hoan_thanh.md   bang Markdown dan thang vao bao cao
  - drl_training/route_scenarios.csv   bang kich ban test (dang du lieu)
  - drl_training/route_completion.csv  bang ti le hoan thanh (dang du lieu)
  - report_figures/10_ti_le_hoan_thanh.{png,pdf}
  - report_figures/11_ly_do_ket_thuc.{png,pdf}
  - report_figures/12_tien_do_tuyen.{png,pdf}

Dinh nghia "hoan thanh tuyen duong" (phai ghi ro trong bao cao): episode chay het
`max_episode_steps` = 500 buoc dieu khien (~100 s mo phong voi action_repeat=4 @ 20 FPS)
MA KHONG va cham va KHONG bi ket thuc som do roi lan. Day chinh la nhan `time_limit` trong
cot `terminate_reason`. Moi tieu chi khac deu suy ra tu cung mot cot nay de bang so khong
mau thuan voi cac hinh da co (03_terminate_reason).

Chay:  python make_route_report.py
"""

import argparse
import sys
from collections import OrderedDict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Deep_RL_Carla/
FIG_DIR = ROOT / "report_figures"
DOC_DIR = ROOT / "docs"

MAX_STEPS = 500          # env.max_episode_steps trong ppo_config_v4.json
ACTION_REPEAT = 4        # env.action_repeat
SIM_FPS = 20.0           # camera.fps
HORIZON_S = MAX_STEPS * ACTION_REPEAT / SIM_FPS   # 100 s mo phong

# ------------------------------------------------------------ bang mau (dong bo plot_metrics.py)
COLOR_PPO = "#2a78d6"
COLOR_PPO_EARLY = "#8fb9e8"
COLOR_PPO_V5 = "#1f4e8c"
COLOR_SAC = "#eb6834"
COLOR_COLLISION = "#d03b3b"
COLOR_OFF_LANE = "#ec835a"
COLOR_TIME_LIMIT = "#0ca30c"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK_PRIMARY, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "grid.color": GRIDLINE, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 10, "figure.titlesize": 13, "figure.titleweight": "bold",
    })


# Hau to them vao ten MOI file hinh, dat boi `--suffix`. Mac dinh rong -> ghi de dung cac
# hinh chinh thuc; co hau to -> ra file rieng, khong dung toi bo hinh da co trong bao cao.
FIG_SUFFIX = ""


def savefig(fig, stem):
    stem = stem + FIG_SUFFIX
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / ("%s.%s" % (stem, ext)), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ->", FIG_DIR / ("%s.png / .pdf" % stem))


# ------------------------------------------------------------------------ kich ban test
# Mo ta dia hinh lay tu chinh ban do CARLA 0.9.10. "Vai tro": Town01-Town04 nam trong
# `env.town` cua config huan luyen -> mo hinh DA thay; Town05 khong nam trong do -> ban do
# GIU LAI, dung de do kha nang tong quat hoa.
SCENARIOS = OrderedDict([
    ("KB-01", dict(town="Town01", vai_tro="Huấn luyện",
                   dia_hinh="Thị trấn nhỏ, đường hai làn, ngã ba chữ T, không có đường cao tốc",
                   kho="Thấp")),
    ("KB-02", dict(town="Town02", vai_tro="Huấn luyện",
                   dia_hinh="Thị trấn nhỏ và gọn, đường hai làn, ngã ba chữ T và ngã tư vuông góc",
                   kho="Thấp")),
    ("KB-03", dict(town="Town03", vai_tro="Huấn luyện",
                   dia_hinh="Đô thị lớn: vòng xuyến, hầm chui, ngã tư nhiều làn, đường dốc",
                   kho="Cao")),
    ("KB-04", dict(town="Town04", vai_tro="Huấn luyện",
                   dia_hinh="Vòng lặp cao tốc quanh núi và thị trấn, nhiều làn, nhánh nhập/tách",
                   kho="Trung bình")),
    ("KB-05", dict(town="Town05", vai_tro="Giữ lại (held-out)",
                   dia_hinh="Lưới ô vuông, đường bốn làn hai chiều, cầu vượt",
                   kho="Trung bình")),
])

# Cac lan danh gia dua vao bao cao.
MODELS = OrderedDict([
    ("PPO@6k (moc som)", dict(color=COLOR_PPO_EARLY,
        note="PPO sau 5 update (6 144 buoc), reward tho",
        run_dir="runs/ppo_v1",
        files={"Town01": "runs/ppo_v1/eval_il_town01.csv",
               "Town04": "runs/ppo_v1/eval_il_Town04.csv",
               "Town05": "runs/ppo_v1/eval_il_Town05.csv"})),
    ("PPO-v4 (trien khai)", dict(color=COLOR_PPO,
        note="reward chuan hoa, mo hinh duoc chon trien khai",
        run_dir="runs/ppo_v4",
        files={"Town01": "runs/ppo_v4/eval_Town01.csv",
               "Town02": "runs/ppo_v4/eval_Town02.csv",
               "Town03": "runs/ppo_v4/eval_Town03.csv",
               "Town04": "runs/ppo_v4/eval_Town04.csv",
               "Town05": "runs/ppo_v4/eval_Town05.csv"})),
    ("PPO-v5 (doi chung)", dict(color=COLOR_PPO_V5,
        note="them junction_off_lane_factor",
        run_dir="runs/ppo_v5",
        files={"Town01": "runs/ppo_v5/eval_Town01.csv",
               "Town03": "runs/ppo_v5/eval_Town03.csv",
               "Town04": "runs/ppo_v5/eval_Town04.csv",
               "Town05": "runs/ppo_v5/eval_Town05.csv"})),
    ("SAC-b (nhanh so sanh)", dict(color=COLOR_SAC,
        note="SAC tot nhat trong sau lan chay",
        run_dir="../drl_training_sac/runs/sac_b",
        files={"Town01": "../drl_training_sac/runs/sac_b/eval_Town01.csv",
               "Town02": "../drl_training_sac/runs/sac_b/eval_Town02.csv",
               "Town03": "../drl_training_sac/runs/sac_b/eval_Town03.csv",
               "Town04": "../drl_training_sac/runs/sac_b/eval_Town04.csv",
               "Town05": "../drl_training_sac/runs/sac_b/eval_Town05.csv"})),
])

# Nhan thoi tiet cho cac lo danh gia chay TRUOC khi `--weather` ton tai. Nhung file CSV do
# khong co cot `weather`, va chung deu chay o thoi tiet mac dinh cua ban do — CARLA 0.9.10
# dung ClearNoon lam mac dinh khi nap map. Ghi ro "(mac dinh)" chu khong ghi tron "ClearNoon"
# de nguoi doc bao cao phan biet duoc "da chu dong dat thoi tiet nay" voi "khong dat gi ca".
WEATHER_LEGACY = "ClearNoon (mac dinh)"

TOWN_TO_KB = dict((v["town"], k) for k, v in SCENARIOS.items())


def wilson_ci(k, n, z=1.96):
    """Khoang tin cay Wilson 95 % cho ti le. Dung Wilson chu khong phai Wald vi n = 30 va ti
    le hay cham 0 %/100 % — khi do Wald cho khoang rong bang 0, tuc la sai."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def load_eval(path):
    df = pd.read_csv(path)
    df["collided"] = df["collided"].astype(str).str.lower().isin(["true", "1"])
    return df


def weather_of(df):
    """Thoi tiet cua mot lo danh gia, doc tu chinh file CSV.

    File cu (chay truoc khi co `--weather`) khong co cot nay -> nhan mac dinh. File moi ghi
    'default' khi nguoi chay khong dat `--weather` -> cung la thoi tiet mac dinh cua ban do.
    """
    if "weather" not in df.columns:
        return WEATHER_LEGACY
    name = str(df["weather"].iloc[0]).strip()
    if not name or name.lower() in ("default", "nan", "none"):
        return WEATHER_LEGACY
    return name


def summarise(df):
    n = len(df)
    reason = df["terminate_reason"]
    done = int(((reason == "time_limit") & (~df["collided"])).sum())
    collision = int((reason == "collision").sum())
    off_lane = int((reason == "off_lane").sum())
    lo, hi = wilson_ci(done, n)
    return OrderedDict([
        ("thoi_tiet", weather_of(df)),
        ("n", n),
        ("hoan_thanh", done),
        ("ti_le_hoan_thanh_pct", 100.0 * done / n),
        ("ci_lo", lo), ("ci_hi", hi),
        ("va_cham_pct", 100.0 * collision / n),
        ("lech_lan_pct", 100.0 * off_lane / n),
        ("tien_do_tb_pct", 100.0 * float(df["length"].mean()) / MAX_STEPS),
        ("tien_do_sd_pct", 100.0 * float(df["length"].std(ddof=1)) / MAX_STEPS if n > 1 else 0.0),
        ("thoi_luong_tb_s", float(df["length"].mean()) * ACTION_REPEAT / SIM_FPS),
        ("do_lech_lan_m", float(df["mean_abs_lane_offset_road"].mean())),
        ("ngay_chay", str(df["eval_run_utc"].iloc[0])[:10]),
        ("checkpoint", Path(str(df["checkpoint"].iloc[0])).name),
    ])


def _add(records, episodes, model, town, df, path):
    row = OrderedDict([("mo_hinh", model), ("kich_ban", TOWN_TO_KB[town]), ("ban_do", town)])
    row.update(summarise(df))
    row["nguon"] = str(path).replace("\\", "/").split("Deep_RL_Carla/")[-1]
    records.append(row)
    ep = df[["episode", "length", "terminate_reason", "mean_abs_lane_offset_road"]].copy()
    ep["mo_hinh"] = model
    ep["kich_ban"] = TOWN_TO_KB[town]
    ep["ban_do"] = town
    ep["thoi_tiet"] = row["thoi_tiet"]
    episodes.append(ep)


def scan_weather_runs(seen):
    """Tim them cac lo danh gia CO dat thoi tiet, sinh ra sau khi them `--weather`.

    Chi nhan file co cot `weather` voi gia tri khac 'default': cac file cu khong co cot do
    va da nam trong danh sach `files` tuong minh, quet lai se dem CHUNG hai lan. Cung vi ly
    do do khong quet bua cac file gop nhu `eval_gop_4town.csv` — chung la ban ghep cua nhung
    file da tinh roi, gop vao se lam ti le hoan thanh sai ma khong bao loi gi.
    """
    found = []
    for model, meta in MODELS.items():
        run_dir = (HERE / meta.get("run_dir", "")).resolve()
        if not run_dir.is_dir():
            continue
        for path in sorted(run_dir.glob("eval_*.csv")):
            if path.resolve() in seen:
                continue
            try:
                df = load_eval(path)
            except Exception as exc:                    # file dang ghi do, hoac hong
                print("[!] khong doc duoc %s (%s)" % (path.name, exc))
                continue
            if "weather" not in df.columns or weather_of(df) == WEATHER_LEGACY:
                continue
            if "town" not in df.columns:
                print("[!] %s thieu cot 'town', bo qua" % path.name)
                continue
            town = str(df["town"].iloc[0])
            if town not in TOWN_TO_KB:
                print("[!] %s chay tren %s — chua co kich ban cho ban do nay, bo qua"
                      % (path.name, town))
                continue
            found.append((model, town, df, path))
    return found


def collect():
    records, episodes, seen = [], [], set()
    for model, meta in MODELS.items():
        for town, rel in meta["files"].items():
            path = (HERE / rel).resolve()
            if not path.exists():
                print("[!] thieu file, bo qua:", path)
                continue
            seen.add(path)
            _add(records, episodes, model, town, load_eval(path), path)

    extra = scan_weather_runs(seen)
    for model, town, df, path in extra:
        _add(records, episodes, model, town, df, path)
    if extra:
        print("Tim them %d lo danh gia co dat thoi tiet." % len(extra))
    else:
        print("Chua co lo danh gia nao dat `--weather`: cot thoi tiet se toan la '%s'."
              % WEATHER_LEGACY)
    return pd.DataFrame(records), pd.concat(episodes, ignore_index=True)


# ------------------------------------------------------------------------------- bieu do
def pick(summary, kb, model=None, weather=None):
    """Mot dong tom tat cho o (kich ban, mo hinh, thoi tiet), hoac None neu khong co.

    Tra ve dong DAU khi co nhieu lo trung dieu kien (vd chay lai cung mot o de kiem tra
    lai): ve chong hai cot cung cho len nhau thi hinh doc ra sai ma khong bao gi."""
    sub = summary[summary["kich_ban"] == kb]
    if model is not None:
        sub = sub[sub["mo_hinh"] == model]
    if weather is not None:
        sub = sub[sub["thoi_tiet"] == weather]
    if sub.empty:
        return None
    if len(sub) > 1:
        print("[!] %s co %d lo trung dieu kien (%s / %s) — hinh dung lo dau tien."
              % (kb, len(sub), model, weather))
    return sub.iloc[0]


def fig_completion(summary, weather):
    """So sanh CAC MO HINH tai mot thoi tiet. Khoa thoi tiet lai chu khong gop moi thoi tiet
    vao mot cot: gop lai thi mot mo hinh chay them lo mua se tut diem so voi mo hinh chi
    chay nang, va chenh lech doc ra nhu la 'mo hinh kem hon'."""
    summary = summary[summary["thoi_tiet"] == weather]
    models = [m for m in MODELS if m in set(summary["mo_hinh"])]
    kbs = list(SCENARIOS)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for j, kb in enumerate(kbs):
            r = pick(summary, kb, model=model)
            if r is None:
                continue
            xs.append(j + (i - (len(models) - 1) / 2.0) * width)
            ys.append(r["ti_le_hoan_thanh_pct"])
            lo.append(max(0.0, r["ti_le_hoan_thanh_pct"] - r["ci_lo"]))
            hi.append(max(0.0, r["ci_hi"] - r["ti_le_hoan_thanh_pct"]))
        ax.bar(xs, ys, width=width * 0.92, color=MODELS[model]["color"], label=model, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.1, capsize=3, zorder=4)
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 2.5, "%.0f" % y, ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(kbs)))
    ax.set_xticklabels(["%s\n%s%s" % (kb, SCENARIOS[kb]["town"],
                        "\n(giu lai)" if "held-out" in SCENARIOS[kb]["vai_tro"] else "")
                        for kb in kbs])
    ax.set_ylabel("Ti le hoan thanh tuyen (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # So episode/o doc TU DU LIEU chu khong ghi cung: khi loc bot mo hinh (--models), phu de
    # cung nhu "30 episode/o (moc som: 10)" se mo ta mot mo hinh khong con trong hinh nua.
    ns = sorted(set(int(x) for x in summary["n"].tolist()))
    n_txt = ("%d" % ns[0]) if len(ns) == 1 else "/".join(str(x) for x in ns)
    ax.set_title("Ti le hoan thanh tuyen duong theo kich ban test — thoi tiet %s\n"
                 "%s episode/o - thanh doc la khoang tin cay Wilson 95 %%"
                 % (weather, n_txt), loc="left")
    ax.legend(ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.xaxis.grid(False)
    savefig(fig, "10_ti_le_hoan_thanh")


def fig_reasons(summary, weather):
    rows = OrderedDict()
    for kb in SCENARIOS:
        r = pick(summary, kb, model=DEPLOY_MODEL, weather=weather)
        if r is not None:
            rows[kb] = r
    kbs = list(rows)
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bottoms = np.zeros(len(kbs))
    parts = [("ti_le_hoan_thanh_pct", "Hoan thanh (het gio, an toan)", COLOR_TIME_LIMIT),
             ("lech_lan_pct", "That bai: roi lan", COLOR_OFF_LANE),
             ("va_cham_pct", "That bai: va cham", COLOR_COLLISION)]
    for key, label, color in parts:
        vals = np.array([rows[kb][key] for kb in kbs], dtype=float)
        ax.barh(range(len(kbs)), vals, left=bottoms, color=color, label=label, height=0.62, zorder=3)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 8:
                ax.text(b + v / 2, i, "%.0f%%" % v, ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals
    ax.set_yticks(range(len(kbs)))
    ax.set_yticklabels(["%s - %s" % (kb, SCENARIOS[kb]["town"]) for kb in kbs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Ti le episode (%)")
    ax.set_title("Ket cuc episode cua mo hinh trien khai PPO-v4 — thoi tiet %s\n"
                 "(30 episode/kich ban)" % weather, loc="left")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.26))
    ax.yaxis.grid(False)
    savefig(fig, "11_ly_do_ket_thuc")


def fig_progress(episodes, weather):
    episodes = episodes[episodes["thoi_tiet"] == weather]
    models = [m for m in MODELS if m in set(episodes["mo_hinh"])]
    kbs = list(SCENARIOS)
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        for j, kb in enumerate(kbs):
            sel = episodes[(episodes["mo_hinh"] == model) & (episodes["kich_ban"] == kb)]
            if sel.empty:
                continue
            x = j + (i - (len(models) - 1) / 2.0) * width
            vals = 100.0 * sel["length"].values / MAX_STEPS
            bp = ax.boxplot([vals], positions=[x], widths=width * 0.78, patch_artist=True,
                            showfliers=False, zorder=3, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=MODELS[model]["color"], alpha=0.85,
                        edgecolor=BASELINE, linewidth=0.8)
            for part in ("whiskers", "caps"):
                for it in bp[part]:
                    it.set(color=INK_MUTED, linewidth=1.0)
            for med in bp["medians"]:
                med.set(color="white", linewidth=1.6)
            jitter = rng.uniform(-width * 0.16, width * 0.16, len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=6, color=INK_PRIMARY,
                       alpha=0.30, zorder=5, linewidths=0)
    ax.set_xticks(range(len(kbs)))
    ax.set_xticklabels(["%s\n%s" % (kb, SCENARIOS[kb]["town"]) for kb in kbs])
    ax.set_ylabel("Tien do tuyen di duoc (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Phan bo tien do tuyen truoc khi episode ket thuc — thoi tiet %s\n"
                 "100 %% = di het %d buoc (~%.0f s mo phong); moi cham la mot episode"
                 % (weather, MAX_STEPS, HORIZON_S), loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODELS[m]["color"]) for m in models]
    ax.legend(handles, models, ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.xaxis.grid(False)
    savefig(fig, "12_tien_do_tuyen")


# Thang do tuan tu cho thoi tiet: day KHONG phai danh muc tuy y ma la mot truc "kho dan"
# (nang -> nhieu may -> uot -> mua to / hoang hon), nen dung mot dai mau chuyen dan thay vi
# bang mau danh muc — nguoi doc thay ngay chieu tang cua do kho.
WEATHER_SCALE = ("#bcd7f2", "#7fb0e0", "#4a86c9", "#2a5d9f", "#173a६6".replace("६", "6"))
DEPLOY_MODEL = "PPO-v4 (trien khai)"


def fig_weather(summary, weathers):
    """Chi ve khi co tu hai thoi tiet tro len — voi mot thoi tiet thi hinh nay chi lap lai
    hinh 10 duoi dang khac."""
    weathers = list(weathers)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    kbs = list(SCENARIOS)
    width = 0.8 / len(weathers)
    for i, w in enumerate(weathers):
        xs, ys, lo, hi = [], [], [], []
        for j, kb in enumerate(kbs):
            r = pick(summary, kb, model=DEPLOY_MODEL, weather=w)
            if r is None:
                continue
            xs.append(j + (i - (len(weathers) - 1) / 2.0) * width)
            ys.append(r["ti_le_hoan_thanh_pct"])
            lo.append(max(0.0, r["ti_le_hoan_thanh_pct"] - r["ci_lo"]))
            hi.append(max(0.0, r["ci_hi"] - r["ti_le_hoan_thanh_pct"]))
        color = WEATHER_SCALE[min(i, len(WEATHER_SCALE) - 1)]
        ax.bar(xs, ys, width=width * 0.92, color=color, label=w, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.1, capsize=3, zorder=4)
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 2.5, "%.0f" % y, ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(kbs)))
    ax.set_xticklabels(["%s\n%s" % (kb, SCENARIOS[kb]["town"]) for kb in kbs])
    ax.set_ylabel("Ti le hoan thanh tuyen (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Anh huong cua thoi tiet len ti le hoan thanh — mo hinh trien khai PPO-v4\n"
                 "thanh doc la khoang tin cay Wilson 95 %", loc="left")
    ax.legend(ncol=min(len(weathers), 4), loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.xaxis.grid(False)
    savefig(fig, "13_thoi_tiet")


# ------------------------------------------------------------------------- bang Markdown
def md_tables(summary):
    L = []
    A = L.append
    A("# Kịch bản kiểm thử và tỉ lệ hoàn thành tuyến đường")
    A("")
    A("> Sinh tự động bởi `drl_training/make_route_report.py` từ các file `eval_*.csv` có sẵn")
    A("> trong `runs/` — **không có số liệu nào nhập tay**. Chạy lại script sau mỗi lần đánh")
    A("> giá mới để bảng và hình luôn khớp nhau.")
    A("")
    A("## 1. Định nghĩa chỉ số")
    A("")
    A("Nhiệm vụ đánh giá là **bám làn liên tục trong một khoảng thời gian cố định**, không phải")
    A("chạy từ điểm A tới điểm B. Vì vậy \"hoàn thành tuyến đường\" được định nghĩa như sau:")
    A("")
    A("| Chỉ số | Định nghĩa vận hành | Cột nguồn trong CSV |")
    A("|---|---|---|")
    A("| **Hoàn thành tuyến** | Xe đi hết %d bước điều khiển (≈ %.0f s mô phỏng, `action_repeat` = %d @ %.0f FPS) mà **không va chạm** và **không bị dừng sớm do rời làn** | `terminate_reason == time_limit` |"
      % (MAX_STEPS, HORIZON_S, ACTION_REPEAT, SIM_FPS))
    A("| **Tiến độ tuyến** | Số bước đi được / %d bước, tính cho *mọi* episode kể cả episode thất bại | `length` |" % MAX_STEPS)
    A("| **Thất bại do va chạm** | Cảm biến va chạm kích hoạt, episode kết thúc ngay | `terminate_reason == collision` |")
    A("| **Thất bại do rời làn** | Lệch khỏi tâm làn quá ngưỡng liên tiếp `off_lane_patience_steps` = 10 bước | `terminate_reason == off_lane` |")
    A("| **Độ lệch làn** | Trung bình trị tuyệt đối độ lệch tâm làn, **chỉ tính ngoài nút giao** | `mean_abs_lane_offset_road` |")
    A("| **KTC 95 %** | Khoảng tin cậy Wilson cho tỉ lệ nhị phân (dùng Wilson vì n = 30 và tỉ lệ hay chạm 0 % hoặc 100 %) | tính từ số episode |")
    A("")
    A("## 2. Bảng kịch bản kiểm thử")
    A("")
    A("Mỗi kịch bản là một bản đồ CARLA. Trong mỗi kịch bản, **điểm xuất phát được bốc ngẫu")
    A("nhiên từ toàn bộ danh sách spawn point của bản đồ** (`_rng.shuffle`, seed = 42) — mỗi")
    A("episode một điểm khác nhau, nên 30 episode phủ 30 vị trí xuất phát khác nhau.")
    A("")
    A("| Mã | Bản đồ | Vai trò | Đặc trưng địa hình | Độ khó | Số episode/mô hình | Điểm xuất phát | Thời tiết | Chính sách |")
    A("|---|---|---|---|---|---|---|---|---|")
    for kb, meta in SCENARIOS.items():
        sub = summary[summary["kich_ban"] == kb]
        ns = sorted(set(int(x) for x in sub["n"].tolist()))
        n_txt = " / ".join(str(x) for x in ns) if ns else "—"
        weathers = sorted(set(sub["thoi_tiet"].tolist())) or [WEATHER_LEGACY]
        A("| %s | %s | %s | %s | %s | %s | Ngẫu nhiên, mỗi episode một điểm | %s | Tất định |"
          % (kb, meta["town"], meta["vai_tro"], meta["dia_hinh"], meta["kho"], n_txt,
             ", ".join(weathers)))
    A("")
    A("**Điều kiện chung cho mọi ô:** xe `vehicle.lincoln.mkz2017`; quan sát là ảnh phân vùng")
    A("ngữ nghĩa 240×192 cộng vector trạng thái; `action_repeat` = %d; giới hạn %d bước/episode;"
      % (ACTION_REPEAT, MAX_STEPS))
    A("chạy ở chế độ tất định (`--deterministic`: lấy kỳ vọng của phân phối hành động, không")
    A("lấy mẫu ngẫu nhiên); không có phương tiện hay người đi bộ nền.")
    A("")
    A("## 3. Tỉ lệ hoàn thành tuyến đường")
    A("")
    A("| Kịch bản | Bản đồ | Thời tiết | Mô hình | n | Hoàn thành | Tỉ lệ hoàn thành | KTC 95 % | Tiến độ TB | Va chạm | Rời làn | Lệch làn (m) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kb in SCENARIOS:
        for model in MODELS:
            sub = summary[(summary["kich_ban"] == kb) & (summary["mo_hinh"] == model)]
            for _, r in sub.iterrows():
                A("| %s | %s | %s | %s | %d | %d | **%.1f %%** | %.1f – %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.3f |"
                  % (kb, r["ban_do"], r["thoi_tiet"], model, r["n"], r["hoan_thanh"],
                     r["ti_le_hoan_thanh_pct"], r["ci_lo"], r["ci_hi"], r["tien_do_tb_pct"],
                     r["va_cham_pct"], r["lech_lan_pct"], r["do_lech_lan_m"]))
    A("")
    A("### Gộp bốn kịch bản (gộp episode, không lấy trung bình của trung bình)")
    A("")
    A("| Mô hình | Thời tiết đã gộp | Tổng episode | Hoàn thành | Tỉ lệ hoàn thành | KTC 95 % | Va chạm | Rời làn |")
    A("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        sub = summary[summary["mo_hinh"] == model]
        if sub.empty:
            continue
        n = int(sub["n"].sum())
        k = int(sub["hoan_thanh"].sum())
        lo, hi = wilson_ci(k, n)
        col = float((sub["va_cham_pct"] * sub["n"]).sum() / n)
        off = float((sub["lech_lan_pct"] * sub["n"]).sum() / n)
        A("| %s | %s | %d | %d | **%.1f %%** | %.1f – %.1f %% | %.1f %% | %.1f %% |"
          % (model, ", ".join(sorted(set(sub["thoi_tiet"].tolist()))), n, k,
             100.0 * k / n, lo, hi, col, off))
    A("")
    A("*(Mốc sớm PPO@6k chỉ chạy trên ba bản đồ nên dòng gộp của nó không so trực tiếp được")
    A("với ba dòng còn lại; đưa vào để thấy mức xuất phát trước khi chuẩn hoá hàm thưởng.)*")
    A("")
    weathers = sorted(set(summary["thoi_tiet"].tolist()))
    if len(weathers) > 1:
        A("### Ảnh hưởng của thời tiết (gộp mọi bản đồ có đủ cặp so sánh)")
        A("")
        A("| Mô hình | Thời tiết | Tổng episode | Hoàn thành | Tỉ lệ hoàn thành | KTC 95 % |")
        A("|---|---|---|---|---|---|")
        for model in MODELS:
            for w in weathers:
                sub = summary[(summary["mo_hinh"] == model) & (summary["thoi_tiet"] == w)]
                if sub.empty:
                    continue
                n = int(sub["n"].sum())
                k = int(sub["hoan_thanh"].sum())
                lo, hi = wilson_ci(k, n)
                A("| %s | %s | %d | %d | **%.1f %%** | %.1f – %.1f %% |"
                  % (model, w, n, k, 100.0 * k / n, lo, hi))
        A("")
        A("Hai dòng chỉ so được với nhau khi chúng phủ **cùng tập bản đồ**; nếu một thời tiết")
        A("mới chỉ chạy trên một phần số bản đồ thì chênh lệch ở bảng này lẫn cả chênh lệch độ")
        A("khó giữa các bản đồ. Dùng bảng chi tiết ở mục 3 để so từng cặp cùng kịch bản.")
        A("")
    A("## 4. Hình minh hoạ")
    A("")
    A("| Hình | File | Nội dung |")
    A("|---|---|---|")
    A("| Hình A | `report_figures/10_ti_le_hoan_thanh.png` | Tỉ lệ hoàn thành theo kịch bản × mô hình, kèm KTC Wilson |")
    A("| Hình B | `report_figures/11_ly_do_ket_thuc.png` | Phân rã kết cục episode của mô hình triển khai PPO-v4 |")
    A("| Hình C | `report_figures/12_tien_do_tuyen.png` | Phân bố tiến độ tuyến của từng episode |")
    A("")
    A("## 5. Giới hạn của bộ số liệu")
    A("")
    A("Cần ghi rõ trong báo cáo, nếu không bảng sẽ bị hiểu sai:")
    A("")
    if len(weathers) > 1:
        A("1. **Số ô thời tiết còn ít.** Bảng đã có %d điều kiện thời tiết (%s), nhưng chỉ những"
          % (len(weathers), ", ".join(weathers)))
        A("   cặp dòng cùng kịch bản mới so trực tiếp được với nhau. Đầu vào của mô hình là ảnh")
        A("   phân vùng ngữ nghĩa nên về nguyên tắc bất biến với thời tiết;")
        A("   `drl_training_sac/check_weather_invariance.py` kiểm chứng giả định đó ở mức tín hiệu.")
    else:
        A("1. **Chưa quét thời tiết.** Toàn bộ episode ở trên chạy với thời tiết mặc định của bản")
        A("   đồ (ClearNoon), nên cột thời tiết hiện chỉ có một giá trị. Đầu vào của mô hình là")
        A("   ảnh phân vùng ngữ nghĩa — trên nguyên tắc bất biến với thời tiết — nhưng đó là")
        A("   *giả định*, và `drl_training_sac/check_weather_invariance.py` được viết ra chính là")
        A("   để kiểm chứng nó. Muốn bảng có nhiều thời tiết thì chạy `evaluate.py` với")
        A("   `--weather <preset>`; script này tự nhận các lô mới và thêm dòng vào bảng.")
    A("2. **Nhiệm vụ là bám làn theo thời lượng, không phải đi theo lộ trình A→B.** \"Hoàn thành")
    A("   tuyến\" ở đây nghĩa là sống sót hết khung thời gian, không phải tới đích do A* hoạch định.")
    A("3. **Không có phương tiện nền.** Tỉ lệ va chạm đo được là va chạm với hạ tầng tĩnh (lề,")
    A("   dải phân cách, cột), không phải va chạm giữa các xe.")
    A("4. **Một seed duy nhất (42).** Chưa lặp lại với nhiều seed nên chưa tách được phương sai")
    A("   do khởi tạo khỏi phương sai do bản đồ.")
    A("")
    return "\n".join(L)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Tong hop bang kich ban + ti le hoan thanh tuyen tu cac file eval_*.csv.")
    ap.add_argument("--models", default=None, metavar="A,B",
                    help="Chi ve nhung mo hinh nay (khop theo tien to ten trong MODELS, vd "
                         "'PPO-v4,SAC-b'). Mac dinh: tat ca. Khi co co nay, script KHONG ghi "
                         "de route_*.csv va docs/ket_qua_ti_le_hoan_thanh.md — nhung bang do "
                         "la ban day du, thu hep mo hinh roi ghi de se lam mat so lieu.")
    ap.add_argument("--suffix", default="", metavar="_hau_to",
                    help="Them hau to vao ten file hinh (vd '_tot_nhat' -> "
                         "10_ti_le_hoan_thanh_tot_nhat.png). Mac dinh: ghi de hinh chinh thuc.")
    ap.add_argument("--figures", default="10,11,12,13", metavar="10,12",
                    help="Danh sach so hieu hinh can ve. Mac dinh: 10,11,12,13.")
    return ap.parse_args()


def main():
    global FIG_SUFFIX, MODELS
    args = parse_args()
    FIG_SUFFIX = args.suffix
    want = set(x.strip() for x in args.figures.split(",") if x.strip())

    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        chon = OrderedDict()
        for k in keys:
            hit = [m for m in MODELS if m.startswith(k) or k in m]
            if not hit:
                sys.exit("Khong co mo hinh nao khop %r. Cac lua chon: %s"
                         % (k, ", ".join(MODELS)))
            for m in hit:
                chon[m] = MODELS[m]
        MODELS = chon
        print("Chi ve cac mo hinh:", ", ".join(MODELS))

    apply_style()
    summary, episodes = collect()
    if summary.empty:
        sys.exit("Khong doc duoc file eval nao.")

    # Thoi tiet dung cho cac hinh so sanh mo hinh. Khoa mot thoi tiet chu khong gop tat ca
    # (xem docstring fig_completion). Uu tien thoi tiet mac dinh vi moi lo danh gia deu co.
    weathers = sorted(set(summary["thoi_tiet"].tolist()))
    primary = WEATHER_LEGACY if WEATHER_LEGACY in weathers else weathers[0]

    if args.models or args.suffix:
        print("(bo qua route_*.csv va docs/ket_qua_ti_le_hoan_thanh.md — chung la ban day du "
              "cua MOI mo hinh, chi sinh lai khi chay khong co --models/--suffix)")
        if "10" in want:
            fig_completion(summary, primary)
        if "11" in want:
            fig_reasons(summary, primary)
        if "12" in want:
            fig_progress(episodes, primary)
        if "13" in want and len(weathers) >= 2:
            fig_weather(summary, weathers)
        print("Xong.")
        return

    summary.to_csv(HERE / "route_completion.csv", index=False, encoding="utf-8-sig")
    print("  ->", HERE / "route_completion.csv")

    scen = pd.DataFrame([dict(kich_ban=kb, **meta) for kb, meta in SCENARIOS.items()])
    scen["so_episode"] = [", ".join(str(int(x)) for x in sorted(set(
        summary[summary["kich_ban"] == kb]["n"].tolist()))) for kb in SCENARIOS]
    scen["thoi_tiet"] = [", ".join(sorted(set(
        summary[summary["kich_ban"] == kb]["thoi_tiet"].tolist())) or [WEATHER_LEGACY])
        for kb in SCENARIOS]
    scen["diem_xuat_phat"] = "ngau nhien tu spawn points, seed 42"
    scen["gioi_han_buoc"] = MAX_STEPS
    scen.to_csv(HERE / "route_scenarios.csv", index=False, encoding="utf-8-sig")
    print("  ->", HERE / "route_scenarios.csv")

    if "10" in want:
        fig_completion(summary, primary)
    if "11" in want:
        fig_reasons(summary, primary)
    if "12" in want:
        fig_progress(episodes, primary)
    if "13" in want and len(weathers) >= 2:
        fig_weather(summary, weathers)

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    out = DOC_DIR / "ket_qua_ti_le_hoan_thanh.md"
    out.write_text(md_tables(summary), encoding="utf-8")
    print("  ->", out)
    print("\nXong.")


if __name__ == "__main__":
    main()
