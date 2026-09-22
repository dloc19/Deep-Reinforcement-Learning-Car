# -*- coding: utf-8 -*-
"""Tong hop TI LE HOAN THANH QUANG DUONG (di tu A den B) tu cac lan chay `evaluate_route.py`.

KHAC voi `make_route_report.py` o cho nao — phan biet duoc hai cai nay moi doc dung bang:

    make_route_report.py       doc `eval_*.csv` cua `evaluate.py`.  "Hoan thanh" = song sot
                               het 500 buoc bam lan, KHONG co dich den.
    make_route_goal_report.py  doc `route_*.csv` cua `evaluate_route.py`. "Hoan thanh" = xe
                               THUC SU toi dich cua tuyen A* (`route_completed`), do tren
                               tuyen dai 80-220 m.

Chi so thu hai moi la thu bao cao goi la "ti le hoan thanh quang duong". No do duoc dieu ma
chi so thu nhat khong do duoc: trong nga tu phan nhanh, observation cua policy khong co
truong nao mang y dinh di lai, nen PPO thuan khong biet re trai hay phai — no di duoc bao xa
la chuyen may rui. Vi vay bang nay tach ba bo dieu khien tren CUNG bo tuyen (ghep cap):

    policy   PPO lai toan bo                      -> gioi han cua HOP DONG QUAN SAT
    astar    pure-pursuit lai toan bo             -> tran hinh hoc, khong hoc gi
    hybrid   PPO lai duong thuong, A* lai nga tu  -> DUNG cau hinh dem trien khai

Xuat ra:
  - docs/ket_qua_hoan_thanh_quang_duong.md
  - drl_training/route_goal_completion.csv
  - report_figures/15_ti_le_hoan_thanh_quang_duong.{png,pdf}
  - report_figures/16_tien_do_quang_duong.{png,pdf}
  - report_figures/17_route_drl_theo_ban_do.{png,pdf}

Chay:  python make_route_goal_report.py
"""

import argparse
from collections import OrderedDict
from math import factorial, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "report_figures"
DOC_DIR = ROOT / "docs"
RUN_DIR = HERE / "runs" / "best"

# Bang mau: cot PPO giu dung mau xanh cua make_route_report.py, hai cot con lai chon rieng.
# Da chay `dataviz/scripts/validate_palette.js` tren bo ba nay — dat CVD separation (cap ke
# nhau te nhat dE 15.5 protan) va chi con WARN o do tuong phan cua mau cam; WARN do duoc go
# bang cach GHI SO TRUC TIEP tren moi cot (xem `ax.text` ben duoi), dung nhu dieu kien
# "visible labels" ma validator doi.
COLOR_POLICY = "#2a78d6"
COLOR_ASTAR = "#f09030"
COLOR_HYBRID = "#0b7a3f"

INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#23211e", "#4a4842", "#75736b"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

MODES = OrderedDict([
    ("policy", dict(nhan="PPO thuan (DRL lai toan bo)", color=COLOR_POLICY)),
    ("astar", dict(nhan="A* + pure-pursuit (khong hoc)", color=COLOR_ASTAR)),
    ("hybrid", dict(nhan="Route+DRL (cau hinh trien khai)", color=COLOR_HYBRID)),
])

SCENARIOS = OrderedDict([
    ("KB-01", dict(town="Town01", vai_tro="Huấn luyện", kho="Thấp")),
    ("KB-02", dict(town="Town02", vai_tro="Huấn luyện", kho="Thấp")),
    ("KB-03", dict(town="Town03", vai_tro="Huấn luyện", kho="Cao")),
    ("KB-04", dict(town="Town04", vai_tro="Huấn luyện", kho="Trung bình")),
    ("KB-05", dict(town="Town05", vai_tro="Giữ lại (held-out)", kho="Trung bình")),
])
TOWN_TO_KB = dict((v["town"], k) for k, v in SCENARIOS.items())


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


def savefig(fig, stem):
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / ("%s.%s" % (stem, ext)), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ->", FIG_DIR / ("%s.png / .pdf" % stem))


def wilson_ci(k, n, z=1.96):
    """Khoang tin cay Wilson 95 %. Dung Wilson chu khong phai Wald vi n nho (20) va ti le
    hay cham dung 100 % — o do Wald cho khoang rong bang 0, tuc la sai."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def mcnemar_p(rows, mode_a, mode_b, khoa="den_dich"):
    """McNemar hai phia cho du lieu GHEP CAP. Ba che do chay dung cung danh sach tuyen nen
    hai nhanh khong doc lap — z-test hai ty le se cho p sai o day."""
    a = dict((r["diem_xuat_phat"], r) for _, r in rows[rows["che_do"] == mode_a].iterrows())
    b = dict((r["diem_xuat_phat"], r) for _, r in rows[rows["che_do"] == mode_b].iterrows())
    chung = sorted(set(a) & set(b))
    n01 = sum(1 for k in chung if bool(a[k][khoa]) and not bool(b[k][khoa]))
    n10 = sum(1 for k in chung if not bool(a[k][khoa]) and bool(b[k][khoa]))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0

    def to_hop(x, y):
        return factorial(x) // (factorial(y) * factorial(x - y))

    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(to_hop(n, i) for i in range(k + 1)) / (2.0 ** n))
    return n01, n10, p


def doc_du_lieu():
    """Gom moi file `runs/best/route_*.csv`. Ten ban do lay tu COT `ban_do` chu khong tu ten
    file: file co the bi doi ten, con cot thi di kem tung dong nen khong lech duoc."""
    frames = []
    for path in sorted(RUN_DIR.glob("route_*.csv")):
        df = pd.read_csv(path)
        df["nguon"] = str(path.relative_to(ROOT)).replace("\\", "/")
        frames.append(df)
    if not frames:
        raise SystemExit("Khong thay file nao khop runs/best/route_*.csv — "
                         "chay evaluate_route.py truoc.")
    df = pd.concat(frames, ignore_index=True)
    for col in ("den_dich", "va_cham"):
        df[col] = df[col].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    df["ty_le_buoc_planner"] = pd.to_numeric(df["ty_le_buoc_planner"], errors="coerce").fillna(0.0)
    df["tien_do_pct"] = pd.to_numeric(df["tien_do_pct"], errors="coerce")
    df["ban_do"] = df["ban_do"].astype(str).str.strip()
    return df


def tom_tat(df):
    recs = []
    for town in sorted(set(df["ban_do"]), key=lambda t: TOWN_TO_KB.get(t, "ZZ")):
        sub_town = df[df["ban_do"] == town]
        for mode in MODES:
            r = sub_town[sub_town["che_do"] == mode]
            if r.empty:
                continue
            n, k = len(r), int(r["den_dich"].sum())
            lo, hi = wilson_ci(k, n)
            recs.append(OrderedDict([
                ("kich_ban", TOWN_TO_KB.get(town, "?")), ("ban_do", town),
                ("che_do", mode), ("nhan", MODES[mode]["nhan"]),
                ("n", n), ("den_dich", k),
                ("ti_le_den_dich_pct", 100.0 * k / n), ("ci_lo", lo), ("ci_hi", hi),
                ("tien_do_tb_pct", float(r["tien_do_pct"].mean())),
                ("tien_do_sd_pct", float(r["tien_do_pct"].std(ddof=1)) if n > 1 else 0.0),
                ("va_cham_pct", 100.0 * float(r["va_cham"].sum()) / n),
                ("lech_lan_m", float(np.nanmean(r["lech_lan_m"].astype(float)))),
                ("buoc_planner_pct", float(r["ty_le_buoc_planner"].mean())),
                ("dai_tuyen_tb_m", float(r["dai_tuyen_m"].mean())),
                ("chay_luc_utc", str(r["chay_luc_utc"].iloc[0])),
                ("nguon", str(r["nguon"].iloc[0])),
            ]))
    return pd.DataFrame(recs)


def fig_completion(summary):
    """Hinh chinh: ti le den dich theo ban do x bo dieu khien."""
    towns = [t for t in (v["town"] for v in SCENARIOS.values())
             if t in set(summary["ban_do"])]
    modes = [m for m in MODES if m in set(summary["che_do"])]
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    width = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        xs, ys, lo, hi = [], [], [], []
        for j, town in enumerate(towns):
            r = summary[(summary["ban_do"] == town) & (summary["che_do"] == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            xs.append(j + (i - (len(modes) - 1) / 2.0) * width)
            ys.append(r["ti_le_den_dich_pct"])
            lo.append(max(0.0, r["ti_le_den_dich_pct"] - r["ci_lo"]))
            hi.append(max(0.0, r["ci_hi"] - r["ti_le_den_dich_pct"]))
        ax.bar(xs, ys, width=width * 0.92, color=MODES[mode]["color"],
               label=MODES[mode]["nhan"], zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.1, capsize=3, zorder=4)
        # Ghi so tren TUNG cot: vua la cach doc chinh xac gia tri, vua la "secondary
        # encoding" de bo ba mau nay khong phai chi dua vao mau de phan biet.
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 2.5, "%.0f" % y, ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(towns)))
    ax.set_xticklabels(
        ["%s\n%s%s" % (TOWN_TO_KB.get(t, "?"), t,
                       "\n(giu lai)"
                       if "held-out" in SCENARIOS[TOWN_TO_KB[t]]["vai_tro"] else "")
         for t in towns])
    ax.set_ylabel("Ti le hoan thanh quang duong (%)")
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ns = sorted(set(int(x) for x in summary["n"].tolist()))
    n_txt = ("%d" % ns[0]) if len(ns) == 1 else "%d-%d" % (ns[0], ns[-1])
    ax.set_title("Ti le hoan thanh quang duong A -> B theo ban do\n"
                 "%s tuyen/o, BA che do chay DUNG cung bo tuyen (ghep cap) - "
                 "thanh doc la KTC Wilson 95 %%" % n_txt, loc="left")
    ax.legend(ncol=len(modes), loc="upper left", bbox_to_anchor=(0, -0.15))
    ax.xaxis.grid(False)
    savefig(fig, "15_ti_le_hoan_thanh_quang_duong")


def fig_progress(df):
    """Hinh phu: PHAN BO tien do tung tuyen. Ti le den dich la chi so nhi phan nen no giau
    mat chuyen "hong o dau": mot che do 0 % van co the di duoc 90 % quang duong roi moi
    lac, con che do khac hong ngay met dau."""
    towns = [t for t in (v["town"] for v in SCENARIOS.values()) if t in set(df["ban_do"])]
    modes = [m for m in MODES if m in set(df["che_do"])]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    width = 0.8 / len(modes)
    rng = np.random.RandomState(7)
    for i, mode in enumerate(modes):
        for j, town in enumerate(towns):
            v = df[(df["ban_do"] == town) &
                   (df["che_do"] == mode)]["tien_do_pct"].astype(float)
            if v.empty:
                continue
            x = j + (i - (len(modes) - 1) / 2.0) * width
            ax.scatter(x + rng.uniform(-width * 0.3, width * 0.3, len(v)), v,
                       s=14, color=MODES[mode]["color"], alpha=0.55,
                       linewidths=0.6, edgecolors=SURFACE, zorder=3,
                       label=MODES[mode]["nhan"] if j == 0 else None)
            ax.plot([x - width * 0.42, x + width * 0.42], [v.mean()] * 2,
                    color=INK_PRIMARY, linewidth=1.8, zorder=5)
    ax.set_xticks(range(len(towns)))
    ax.set_xticklabels(["%s\n%s" % (TOWN_TO_KB.get(t, "?"), t) for t in towns])
    ax.set_ylabel("Tien do di duoc tren tuyen (%)")
    ax.set_ylim(-4, 108)
    ax.set_title("Tien do tung tuyen - moi cham la mot tuyen, gach ngang la trung binh\n"
                 "Cham nam sat 100 % la tuyen ve toi dich", loc="left")
    ax.legend(ncol=len(modes), loc="upper left", bbox_to_anchor=(0, -0.15))
    ax.xaxis.grid(False)
    savefig(fig, "16_tien_do_quang_duong")


def fig_hybrid_only(df, summary):
    """CHI che do trien khai (Route+DRL = PPO ghep A*), Town01 den Town05 tren mot hinh.

    Hinh 15 tra loi "ghep A* vao thi hon PPO thuan bao nhieu"; hinh nay tra loi cau khac:
    "cai dem trien khai chay duoc bao nhieu tren tung ban do". Mot chuoi duy nhat nen KHONG
    co o chu thich — tieu de da goi ten no; va dat them cot GOP o phai, tach bang mot duong
    doc de khong ai doc nham no la ban do thu sau.
    """
    towns = [t for t in (v["town"] for v in SCENARIOS.values())
             if t in set(summary["ban_do"])]
    if not towns:
        return
    fig, ax = plt.subplots(figsize=(9.6, 4.9))

    xs, ys, lo, hi, nhan_x = [], [], [], [], []
    for j, town in enumerate(towns):
        r = summary[(summary["ban_do"] == town) & (summary["che_do"] == "hybrid")]
        if r.empty:
            continue
        r = r.iloc[0]
        xs.append(j)
        ys.append(r["ti_le_den_dich_pct"])
        lo.append(max(0.0, r["ti_le_den_dich_pct"] - r["ci_lo"]))
        hi.append(max(0.0, r["ci_hi"] - r["ti_le_den_dich_pct"]))
        nhan_x.append("%s\n%s%s\n%d/%d tuyen" % (
            TOWN_TO_KB.get(town, "?"), town,
            "\n(giu lai)" if "held-out" in SCENARIOS[TOWN_TO_KB[town]]["vai_tro"] else "",
            int(r["den_dich"]), int(r["n"])))

    # Cot GOP: dat cach mot o trong de mat doc ra ngay day la muc khac, khong phai ban do.
    x_gop = len(towns) + 0.6
    n_g, k_g, pct_g, lo_g, hi_g, _vc, _td = gop(df, "hybrid")

    ax.bar(xs, ys, width=0.58, color=COLOR_HYBRID, zorder=3)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                elinewidth=1.1, capsize=4, zorder=4)
    ax.bar([x_gop], [pct_g], width=0.58, color=COLOR_HYBRID, zorder=3)
    ax.errorbar([x_gop], [pct_g], yerr=[[pct_g - lo_g], [hi_g - pct_g]], fmt="none",
                ecolor=INK_MUTED, elinewidth=1.1, capsize=4, zorder=4)
    ax.axvline(len(towns) - 0.3, color=BASELINE, linewidth=1.0, zorder=2)

    for x, y, h in zip(xs + [x_gop], ys + [pct_g], hi + [hi_g - pct_g]):
        ax.text(x, y + h + 2.0, "%.0f %%" % y, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK_PRIMARY)

    ax.set_xticks(list(xs) + [x_gop])
    ax.set_xticklabels(nhan_x + ["GOP\n%d ban do\n%d/%d tuyen"
                                 % (len(towns), k_g, n_g)])
    ax.set_ylabel("Ti le hoan thanh quang duong (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Route+DRL (PPO ghep A*) - ti le di tron quang duong A -> B\n"
                 "Tuyen 80-220 m, chay tat dinh, thanh doc la KTC Wilson 95 %", loc="left")
    ax.xaxis.grid(False)
    savefig(fig, "17_route_drl_theo_ban_do")


def gop(df, mode):
    r = df[df["che_do"] == mode]
    n, k = len(r), int(r["den_dich"].sum())
    lo, hi = wilson_ci(k, n)
    return (n, k, 100.0 * k / max(n, 1), lo, hi,
            100.0 * float(r["va_cham"].sum()) / max(n, 1), float(r["tien_do_pct"].mean()))


def md(df, summary):
    L = []
    A = L.append
    A("# Tỉ lệ hoàn thành quãng đường (đi từ A đến B)\n")
    A("> Sinh tự động bởi `drl_training/make_route_goal_report.py` từ các file")
    A("> `drl_training/runs/best/route_*.csv` do `evaluate_route.py` ghi ra — **không có số")
    A("> liệu nào nhập tay**. Chạy lại script sau mỗi lô đánh giá mới.\n")
    A("## 1. Chỉ số này khác gì bảng ở `ket_qua_ti_le_hoan_thanh.md`\n")
    A("| | `ket_qua_ti_le_hoan_thanh.md` | Bảng này |")
    A("|---|---|---|")
    A("| Nhiệm vụ | Bám làn tự do, **không có đích** | Đi hết tuyến A* từ A đến B |")
    A("| \"Hoàn thành\" | Sống sót hết 500 bước, không va chạm | Xe **thực sự tới đích** "
      "(`route_completed`) |")
    A("| Script | `evaluate.py` | `evaluate_route.py` |")
    A("| Nguồn | `runs/*/eval_*.csv` | `runs/best/route_*.csv` |\n")
    A("Chỉ bảng này mới trả lời được câu \"xe đi hết quãng đường bao nhiêu phần trăm số "
      "lần\".\n")
    A("## 2. Ba bộ điều khiển chạy trên **cùng một bộ tuyến** (thiết kế ghép cặp)\n")
    A("| Chế độ | Ai cầm lái | Đo cái gì |")
    A("|---|---|---|")
    A("| `policy` | PPO lái toàn bộ, kể cả trong ngã tư | Giới hạn của **hợp đồng quan "
      "sát**: observation không có trường nào mang ý định rẽ, nên trong ngã tư phân nhánh "
      "policy không thể biết rẽ trái hay phải |")
    A("| `astar` | Pure-pursuit lái toàn bộ | Trần hình học thuần tuý, không học gì |")
    A("| `hybrid` | PPO lái đường thường, pure-pursuit lái ngã tư rẽ và đổi làn | **Đúng "
      "cấu hình đem triển khai** trên dashboard |\n")
    dai = float(df["dai_tuyen_m"].mean())
    A("Tuyến dài %.0f m trung bình (lọc 80–220 m), giới hạn 90 s mô phỏng mỗi tuyến, xe "
      "`vehicle.lincoln.mkz2017`, chạy tất định, không có xe nền." % dai)
    A("Checkpoint: `runs/best/ppo_latest.pt` (PPO-v4, update 195).\n")
    A("## 3. Tỉ lệ hoàn thành quãng đường theo bản đồ\n")
    A("| Kịch bản | Bản đồ | Chế độ | n | Tới đích | Tỉ lệ hoàn thành | KTC 95 % | "
      "Tiến độ TB | Va chạm | Lệch làn (m) | A* cầm lái |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        A("| %s | %s | %s | %d | %d | **%.1f %%** | %.1f – %.1f %% | %.1f %% | %.1f %% | "
          "%.3f | %.1f %% |"
          % (r["kich_ban"], r["ban_do"], r["nhan"], r["n"], r["den_dich"],
             r["ti_le_den_dich_pct"], r["ci_lo"], r["ci_hi"], r["tien_do_tb_pct"],
             r["va_cham_pct"], r["lech_lan_m"], r["buoc_planner_pct"]))
    A("\n### Gộp toàn bộ bản đồ (gộp tuyến, không lấy trung bình của trung bình)\n")
    A("| Chế độ | Tổng tuyến | Tới đích | Tỉ lệ hoàn thành | KTC 95 % | Va chạm | "
      "Tiến độ TB |")
    A("|---|---|---|---|---|---|---|")
    for mode in MODES:
        if mode not in set(df["che_do"]):
            continue
        n, k, pct, lo, hi, vc, td = gop(df, mode)
        A("| %s | %d | %d | **%.1f %%** | %.1f – %.1f %% | %.1f %% | %.1f %% |"
          % (MODES[mode]["nhan"], n, k, pct, lo, hi, vc, td))
    A("\n### Kiểm định McNemar ghép cặp — `policy` đối chứng `hybrid`\n")
    A("Dùng McNemar chứ không dùng z-test hai tỉ lệ: hai nhánh chạy **đúng cùng những "
      "tuyến** nên không độc lập; chỉ các cặp bất đồng mới mang thông tin.\n")
    A("| Phạm vi | Chỉ `policy` tới đích | Chỉ `hybrid` tới đích | p (hai phía) | "
      "Kết luận |")
    A("|---|---|---|---|---|")
    for town in [v["town"] for v in SCENARIOS.values() if v["town"] in set(df["ban_do"])]:
        n01, n10, p = mcnemar_p(df[df["ban_do"] == town], "policy", "hybrid")
        A("| %s | %d | %d | %.4f | %s |" % (town, n01, n10, p,
          "khác biệt có ý nghĩa" if p < 0.05 else "chưa đủ để kết luận"))
    n01, n10, p = mcnemar_p(df, "policy", "hybrid")
    A("| **Gộp** | %d | %d | %.6f | %s |" % (n01, n10, p,
      "khác biệt có ý nghĩa" if p < 0.05 else "chưa đủ để kết luận"))
    A("\n## 4. Hình minh hoạ\n")
    A("| Hình | File | Nội dung |")
    A("|---|---|---|")
    A("| Hình D | `report_figures/15_ti_le_hoan_thanh_quang_duong.png` | Tỉ lệ tới đích "
      "theo bản đồ × bộ điều khiển, kèm KTC Wilson |")
    A("| Hình E | `report_figures/16_tien_do_quang_duong.png` | Phân bố tiến độ từng "
      "tuyến — cho thấy hỏng ở đâu, không chỉ hỏng hay không |")
    A("| Hình F | `report_figures/17_route_drl_theo_ban_do.png` | **Chỉ chế độ triển khai "
      "Route+DRL** trên Town01–Town05, kèm cột gộp |")
    A("\n## 5. Giới hạn\n")
    A("1. **Một seed chọn tuyến duy nhất (20260906).** Bộ tuyến cố định để ba chế độ ghép")
    A("   cặp được, nhưng chưa lặp lại với bộ tuyến khác.")
    A("2. **Không có xe nền.** Va chạm đo được là va chạm với hạ tầng tĩnh.")
    A("3. **Tuyến 80–220 m.** Tuyến ngắn hơn không đi qua ngã tư nào nên không đo được gì;")
    A("   tuyến dài hơn làm mỗi lượt tốn vài phút.")
    A("4. **Thời tiết mặc định của bản đồ (ClearNoon).** Chưa quét thời tiết ở bài này.")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fig", action="store_true", help="Chi xuat bang, khong ve hinh")
    args = parser.parse_args()

    apply_style()
    df = doc_du_lieu()
    summary = tom_tat(df)

    out_csv = HERE / "route_goal_completion.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("  ->", out_csv)

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    out_md = DOC_DIR / "ket_qua_hoan_thanh_quang_duong.md"
    out_md.write_text(md(df, summary), encoding="utf-8")
    print("  ->", out_md)

    if not args.no_fig:
        fig_completion(summary)
        fig_progress(df)
        fig_hybrid_only(df, summary)

    print("\n=== Gop toan bo ban do ===")
    for mode in MODES:
        if mode not in set(df["che_do"]):
            continue
        n, k, pct, lo, hi, vc, td = gop(df, mode)
        print("  %-34s %3d tuyen  toi dich %3d (%5.1f%%)  va cham %5.1f%%  tien do %5.1f%%"
              % (MODES[mode]["nhan"], n, k, pct, vc, td))


if __name__ == "__main__":
    main()
