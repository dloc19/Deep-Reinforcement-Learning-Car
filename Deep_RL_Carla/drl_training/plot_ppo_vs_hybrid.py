# -*- coding: utf-8 -*-
"""Ve bieu do so sanh ti le hoan thanh tuyen duong giua PPO thuan va PPO + A* (Hybrid).

Du lieu nguon: drl_training/runs/best/route_*.csv sinh boi evaluate_route.py.
Hai che do so sanh:
  - policy: PPO thuan (lai toan bo tuyen ke ca nga tu)
  - hybrid: PPO + A* (PPO bam lan duong thuong, A* dan huong qua nga tu)

Xuat ra:
  - report_figures/compare_ppo_vs_hybrid_success_rate.{png,pdf}
  - report_figures/compare_ppo_vs_hybrid_progress.{png,pdf}
  - report_figures/compare_ppo_vs_hybrid_combined.{png,pdf}
"""

import os
import shutil
from pathlib import Path
from math import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiet lap duong dan
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN_DIR = HERE / "runs" / "best"
FIG_DIR = ROOT / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Bang mau dong bo voi de tai
COLOR_PPO = "#2a78d6"       # Xanh duong (PPO thuan)
COLOR_HYBRID = "#0b7a3f"    # Xanh la dam (Hybrid PPO + A*)
INK_PRIMARY = "#23211e"
INK_SECONDARY = "#4a4842"
INK_MUTED = "#75736b"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Calibri"],
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10.0,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 10.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })

def wilson_ci(k, n, z=1.96):
    """Khoang tin cay Wilson 95% cho ti le k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)

def load_data():
    csv_files = sorted(RUN_DIR.glob("route_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Khong tim thay file route_*.csv trong {RUN_DIR}")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["den_dich"] = full_df["den_dich"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    full_df["va_cham"] = full_df["va_cham"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    full_df["tien_do_pct"] = pd.to_numeric(full_df["tien_do_pct"], errors="coerce")
    full_df["ban_do"] = full_df["ban_do"].astype(str).str.strip()
    return full_df

def build_comparison_stats(df):
    towns = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    town_roles = {
        "Town01": "Town01\n(Huấn luyện)",
        "Town02": "Town02\n(Huấn luyện)",
        "Town03": "Town03\n(Huấn luyện)",
        "Town04": "Town04\n(Huấn luyện)",
        "Town05": "Town05\n(Held-out)",
    }
    
    data = []
    for town in towns:
        sub = df[df["ban_do"] == town]
        # PPO
        ppo_sub = sub[sub["che_do"] == "policy"]
        ppo_n = len(ppo_sub)
        ppo_success = int(ppo_sub["den_dich"].sum())
        ppo_succ_rate = 100.0 * ppo_success / ppo_n if ppo_n else 0.0
        ppo_ci_lo, ppo_ci_hi = wilson_ci(ppo_success, ppo_n)
        ppo_prog_mean = float(ppo_sub["tien_do_pct"].mean())
        ppo_prog_std = float(ppo_sub["tien_do_pct"].std(ddof=1)) if ppo_n > 1 else 0.0
        
        # Hybrid
        hyb_sub = sub[sub["che_do"] == "hybrid"]
        hyb_n = len(hyb_sub)
        hyb_success = int(hyb_sub["den_dich"].sum())
        hyb_succ_rate = 100.0 * hyb_success / hyb_n if hyb_n else 0.0
        hyb_ci_lo, hyb_ci_hi = wilson_ci(hyb_success, hyb_n)
        hyb_prog_mean = float(hyb_sub["tien_do_pct"].mean())
        hyb_prog_std = float(hyb_sub["tien_do_pct"].std(ddof=1)) if hyb_n > 1 else 0.0
        
        data.append({
            "town": town,
            "town_label": town_roles[town],
            "ppo_n": ppo_n, "ppo_success": ppo_success,
            "ppo_succ_rate": ppo_succ_rate, "ppo_ci_lo": ppo_ci_lo, "ppo_ci_hi": ppo_ci_hi,
            "ppo_prog_mean": ppo_prog_mean, "ppo_prog_std": ppo_prog_std,
            "hyb_n": hyb_n, "hyb_success": hyb_success,
            "hyb_succ_rate": hyb_succ_rate, "hyb_ci_lo": hyb_ci_lo, "hyb_ci_hi": hyb_ci_hi,
            "hyb_prog_mean": hyb_prog_mean, "hyb_prog_std": hyb_prog_std,
        })
        
    # Tong hop (Gop toan bo 100 tuyen)
    ppo_all = df[df["che_do"] == "policy"]
    hyb_all = df[df["che_do"] == "hybrid"]
    
    p_n_all, p_s_all = len(ppo_all), int(ppo_all["den_dich"].sum())
    p_ci_lo_all, p_ci_hi_all = wilson_ci(p_s_all, p_n_all)
    h_n_all, h_s_all = len(hyb_all), int(hyb_all["den_dich"].sum())
    h_ci_lo_all, h_ci_hi_all = wilson_ci(h_s_all, h_n_all)
    
    overall = {
        "town": "Overall",
        "town_label": "Toàn bộ\n(100 tuyến)",
        "ppo_n": p_n_all, "ppo_success": p_s_all,
        "ppo_succ_rate": 100.0 * p_s_all / p_n_all, "ppo_ci_lo": p_ci_lo_all, "ppo_ci_hi": p_ci_hi_all,
        "ppo_prog_mean": float(ppo_all["tien_do_pct"].mean()), "ppo_prog_std": float(ppo_all["tien_do_pct"].std(ddof=1)),
        "hyb_n": h_n_all, "hyb_success": h_s_all,
        "hyb_succ_rate": 100.0 * h_s_all / h_n_all, "hyb_ci_lo": h_ci_lo_all, "hyb_ci_hi": h_ci_hi_all,
        "hyb_prog_mean": float(hyb_all["tien_do_pct"].mean()), "hyb_prog_std": float(hyb_all["tien_do_pct"].std(ddof=1)),
    }
    
    return data, overall

def plot_success_rate(data, overall):
    """Bieu do so sanh ti le hoan thanh toi dich (Success Rate %)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    
    n_towns = len(data)
    width = 0.36
    x_base = np.arange(n_towns)
    
    x_ppo = list(x_base - width / 2)
    x_hyb = list(x_base + width / 2)
    
    y_ppo = [d["ppo_succ_rate"] for d in data]
    y_hyb = [d["hyb_succ_rate"] for d in data]
    
    err_ppo = [
        [d["ppo_succ_rate"] - d["ppo_ci_lo"] for d in data],
        [d["ppo_ci_hi"] - d["ppo_succ_rate"] for d in data]
    ]
    err_hyb = [
        [d["hyb_succ_rate"] - d["hyb_ci_lo"] for d in data],
        [d["hyb_ci_hi"] - d["hyb_succ_rate"] for d in data]
    ]
    
    # Ve cot theo tung town
    b1 = ax.bar(x_ppo, y_ppo, width=width * 0.92, color=COLOR_PPO, label="PPO thuần (Policy chính)", zorder=3)
    ax.errorbar(x_ppo, y_ppo, yerr=err_ppo, fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3.5, zorder=4)
    
    b2 = ax.bar(x_hyb, y_hyb, width=width * 0.92, color=COLOR_HYBRID, label="Hybrid (PPO + A*)", zorder=3)
    ax.errorbar(x_hyb, y_hyb, yerr=err_hyb, fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3.5, zorder=4)
    
    # Ghi so tren tung cot Town
    for i, (xp, yp, xh, yh) in enumerate(zip(x_ppo, y_ppo, x_hyb, y_hyb)):
        ax.text(xp, yp + err_ppo[1][i] + 2.0, f"{yp:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_PPO)
        ax.text(xh, yh + err_hyb[1][i] + 2.0, f"{yh:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_HYBRID)
        
    # Cot Gop (Overall)
    x_gop_base = n_towns + 0.35
    x_gop_ppo = x_gop_base - width / 2
    x_gop_hyb = x_gop_base + width / 2
    
    ax.bar(x_gop_ppo, overall["ppo_succ_rate"], width=width * 0.92, color=COLOR_PPO, zorder=3)
    ax.errorbar(x_gop_ppo, overall["ppo_succ_rate"],
                yerr=[[overall["ppo_succ_rate"] - overall["ppo_ci_lo"]], [overall["ppo_ci_hi"] - overall["ppo_succ_rate"]]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3.5, zorder=4)
    ax.text(x_gop_ppo, overall["ppo_succ_rate"] + (overall["ppo_ci_hi"] - overall["ppo_succ_rate"]) + 2.0,
            f"{overall['ppo_succ_rate']:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_PPO)
    
    ax.bar(x_gop_hyb, overall["hyb_succ_rate"], width=width * 0.92, color=COLOR_HYBRID, zorder=3)
    ax.errorbar(x_gop_hyb, overall["hyb_succ_rate"],
                yerr=[[overall["hyb_succ_rate"] - overall["hyb_ci_lo"]], [overall["hyb_ci_hi"] - overall["hyb_succ_rate"]]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3.5, zorder=4)
    ax.text(x_gop_hyb, overall["hyb_succ_rate"] + (overall["hyb_ci_hi"] - overall["hyb_succ_rate"]) + 2.0,
            f"{overall['hyb_succ_rate']:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_HYBRID)
    
    # Vach ngan cach phan tach Toan bo
    ax.axvline(n_towns - 0.32, color=BASELINE, linestyle="--", linewidth=1.2, zorder=2)
    
    # Ticks & labels
    all_x = list(x_base) + [x_gop_base]
    all_labels = [d["town_label"] for d in data] + [overall["town_label"]]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)
    
    ax.set_ylabel("Tỉ lệ hoàn thành tới đích (%)", fontweight="bold")
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.grid(False)
    
    ax.set_title("So sánh tỉ lệ hoàn thành tuyến đường (Tới đích thành công)\n"
                 "PPO thuần vs. Hybrid (PPO + A*) trên các bản đồ CARLA (Thanh sai số: KTC Wilson 95%)",
                 loc="left", pad=12)
    
    # Dat legend o duoi de khong che so lieu
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"compare_ppo_vs_hybrid_success_rate.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Saved compare_ppo_vs_hybrid_success_rate")

def plot_progress_rate(data, overall):
    """Bieu do so sanh tien do hoan thanh tuyen duong trung binh (Progress %)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    
    n_towns = len(data)
    width = 0.36
    x_base = np.arange(n_towns)
    
    x_ppo = list(x_base - width / 2)
    x_hyb = list(x_base + width / 2)
    
    y_ppo = [d["ppo_prog_mean"] for d in data]
    y_hyb = [d["hyb_prog_mean"] for d in data]
    
    # Ve cot theo tung town
    b1 = ax.bar(x_ppo, y_ppo, width=width * 0.92, color=COLOR_PPO, label="PPO thuần (Policy chính)", zorder=3)
    b2 = ax.bar(x_hyb, y_hyb, width=width * 0.92, color=COLOR_HYBRID, label="Hybrid (PPO + A*)", zorder=3)
    
    # Ghi so tren tung cot Town
    for i, (xp, yp, xh, yh) in enumerate(zip(x_ppo, y_ppo, x_hyb, y_hyb)):
        ax.text(xp, yp + 1.8, f"{yp:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_PPO)
        ax.text(xh, yh + 1.8, f"{yh:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_HYBRID)
        
    # Cot Gop (Overall)
    x_gop_base = n_towns + 0.35
    x_gop_ppo = x_gop_base - width / 2
    x_gop_hyb = x_gop_base + width / 2
    
    ax.bar(x_gop_ppo, overall["ppo_prog_mean"], width=width * 0.92, color=COLOR_PPO, zorder=3)
    ax.text(x_gop_ppo, overall["ppo_prog_mean"] + 1.8, f"{overall['ppo_prog_mean']:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_PPO)
    
    ax.bar(x_gop_hyb, overall["hyb_prog_mean"], width=width * 0.92, color=COLOR_HYBRID, zorder=3)
    ax.text(x_gop_hyb, overall["hyb_prog_mean"] + 1.8, f"{overall['hyb_prog_mean']:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_HYBRID)
    
    # Vach ngan cach phan tach Toan bo
    ax.axvline(n_towns - 0.32, color=BASELINE, linestyle="--", linewidth=1.2, zorder=2)
    
    # Ticks & labels
    all_x = list(x_base) + [x_gop_base]
    all_labels = [d["town_label"] for d in data] + [overall["town_label"]]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)
    
    ax.set_ylabel("Tiến độ tuyến đường trung bình (%)", fontweight="bold")
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.grid(False)
    
    ax.set_title("So sánh tiến độ hoàn thành chiều dài tuyến đường trung bình (%)\n"
                 "PPO thuần vs. Hybrid (PPO + A*) (Độ dài tuyến: 80 - 220 mét)",
                 loc="left", pad=12)
    
    # Dat legend o duoi de khong che so lieu
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"compare_ppo_vs_hybrid_progress.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Saved compare_ppo_vs_hybrid_progress")

def plot_combined_figure(data, overall):
    """Bieu do tong hop 2 subplot (Ti le toi dich & Tien do tuyen duong)."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.5), sharey=False)
    
    n_towns = len(data)
    width = 0.36
    x_base = np.arange(n_towns)
    x_ppo = list(x_base - width / 2)
    x_hyb = list(x_base + width / 2)
    x_gop_base = n_towns + 0.35
    x_gop_ppo = x_gop_base - width / 2
    x_gop_hyb = x_gop_base + width / 2
    all_x = list(x_base) + [x_gop_base]
    all_labels = [d["town_label"] for d in data] + [overall["town_label"]]
    
    # ------------------ SUBPLOT 1: SUCCESS RATE ------------------
    y_ppo_s = [d["ppo_succ_rate"] for d in data]
    y_hyb_s = [d["hyb_succ_rate"] for d in data]
    err_ppo = [
        [d["ppo_succ_rate"] - d["ppo_ci_lo"] for d in data],
        [d["ppo_ci_hi"] - d["ppo_succ_rate"] for d in data]
    ]
    err_hyb = [
        [d["hyb_succ_rate"] - d["hyb_ci_lo"] for d in data],
        [d["hyb_ci_hi"] - d["hyb_succ_rate"] for d in data]
    ]
    
    ax1.bar(x_ppo, y_ppo_s, width=width * 0.92, color=COLOR_PPO, label="PPO thuần (Policy chính)", zorder=3)
    ax1.errorbar(x_ppo, y_ppo_s, yerr=err_ppo, fmt="none", ecolor=INK_MUTED, elinewidth=1.1, capsize=3, zorder=4)
    ax1.bar(x_hyb, y_hyb_s, width=width * 0.92, color=COLOR_HYBRID, label="Hybrid (PPO + A*)", zorder=3)
    ax1.errorbar(x_hyb, y_hyb_s, yerr=err_hyb, fmt="none", ecolor=INK_MUTED, elinewidth=1.1, capsize=3, zorder=4)
    
    for i, (xp, yp, xh, yh) in enumerate(zip(x_ppo, y_ppo_s, x_hyb, y_hyb_s)):
        ax1.text(xp, yp + err_ppo[1][i] + 2.0, f"{yp:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_PPO)
        ax1.text(xh, yh + err_hyb[1][i] + 2.0, f"{yh:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_HYBRID)
        
    ax1.bar(x_gop_ppo, overall["ppo_succ_rate"], width=width * 0.92, color=COLOR_PPO, zorder=3)
    ax1.errorbar(x_gop_ppo, overall["ppo_succ_rate"],
                 yerr=[[overall["ppo_succ_rate"] - overall["ppo_ci_lo"]], [overall["ppo_ci_hi"] - overall["ppo_succ_rate"]]],
                 fmt="none", ecolor=INK_MUTED, elinewidth=1.1, capsize=3, zorder=4)
    ax1.text(x_gop_ppo, overall["ppo_succ_rate"] + (overall["ppo_ci_hi"] - overall["ppo_succ_rate"]) + 2.0,
             f"{overall['ppo_succ_rate']:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_PPO)
    
    ax1.bar(x_gop_hyb, overall["hyb_succ_rate"], width=width * 0.92, color=COLOR_HYBRID, zorder=3)
    ax1.errorbar(x_gop_hyb, overall["hyb_succ_rate"],
                 yerr=[[overall["hyb_succ_rate"] - overall["hyb_ci_lo"]], [overall["hyb_ci_hi"] - overall["hyb_succ_rate"]]],
                 fmt="none", ecolor=INK_MUTED, elinewidth=1.1, capsize=3, zorder=4)
    ax1.text(x_gop_hyb, overall["hyb_succ_rate"] + (overall["hyb_ci_hi"] - overall["hyb_succ_rate"]) + 2.0,
             f"{overall['hyb_succ_rate']:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_HYBRID)
    
    ax1.axvline(n_towns - 0.32, color=BASELINE, linestyle="--", linewidth=1.1, zorder=2)
    ax1.set_xticks(all_x)
    ax1.set_xticklabels(all_labels, fontsize=9)
    ax1.set_ylabel("Tỉ lệ hoàn thành tới đích (%)", fontweight="bold")
    ax1.set_ylim(0, 116)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.xaxis.grid(False)
    ax1.set_title("(a) Tỉ lệ xe hoàn thành tới đích (KTC Wilson 95%)", loc="left", fontsize=11.5)
    
    # ------------------ SUBPLOT 2: PROGRESS RATE ------------------
    y_ppo_p = [d["ppo_prog_mean"] for d in data]
    y_hyb_p = [d["hyb_prog_mean"] for d in data]
    
    ax2.bar(x_ppo, y_ppo_p, width=width * 0.92, color=COLOR_PPO, label="PPO thuần (Policy chính)", zorder=3)
    ax2.bar(x_hyb, y_hyb_p, width=width * 0.92, color=COLOR_HYBRID, label="Hybrid (PPO + A*)", zorder=3)
    
    for i, (xp, yp, xh, yh) in enumerate(zip(x_ppo, y_ppo_p, x_hyb, y_hyb_p)):
        ax2.text(xp, yp + 1.8, f"{yp:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_PPO)
        ax2.text(xh, yh + 1.8, f"{yh:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_HYBRID)
        
    ax2.bar(x_gop_ppo, overall["ppo_prog_mean"], width=width * 0.92, color=COLOR_PPO, zorder=3)
    ax2.text(x_gop_ppo, overall["ppo_prog_mean"] + 1.8, f"{overall['ppo_prog_mean']:.1f}%",
             ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_PPO)
    
    ax2.bar(x_gop_hyb, overall["hyb_prog_mean"], width=width * 0.92, color=COLOR_HYBRID, zorder=3)
    ax2.text(x_gop_hyb, overall["hyb_prog_mean"] + 1.8, f"{overall['hyb_prog_mean']:.1f}%",
             ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLOR_HYBRID)
    
    ax2.axvline(n_towns - 0.32, color=BASELINE, linestyle="--", linewidth=1.1, zorder=2)
    ax2.set_xticks(all_x)
    ax2.set_xticklabels(all_labels, fontsize=9)
    ax2.set_ylabel("Tiến độ tuyến đường trung bình (%)", fontweight="bold")
    ax2.set_ylim(0, 114)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.xaxis.grid(False)
    ax2.set_title("(b) Tiến độ hoàn thành chiều dài tuyến đường trung bình", loc="left", fontsize=11.5)
    
    # Legend chung dat o duoi cung
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10.5)
    
    fig.suptitle("Đánh giá toàn diện: PPO thuần vs. Hybrid (PPO + A*) trên 100 kịch bản tuyến đường CARLA",
                 fontsize=13.0, fontweight="bold", y=1.00)
    
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"compare_ppo_vs_hybrid_combined.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Saved compare_ppo_vs_hybrid_combined")

def copy_to_artifacts():
    artifact_dir = Path(r"C:\Users\dloc\.gemini\antigravity-cli\brain\191d4433-c343-4e25-8a67-dac1d3bf836b")
    if artifact_dir.exists():
        for fname in ["compare_ppo_vs_hybrid_success_rate.png", "compare_ppo_vs_hybrid_progress.png", "compare_ppo_vs_hybrid_combined.png"]:
            src = FIG_DIR / fname
            if src.exists():
                shutil.copy2(src, artifact_dir / fname)
                print(f"Copied {fname} to artifacts directory")

if __name__ == "__main__":
    print("Dang doc du lieu...")
    df = load_data()
    data, overall = build_comparison_stats(df)
    print(f"Da xu ly {len(df)} dong du lieu tu 5 ban do.")
    
    plot_success_rate(data, overall)
    plot_progress_rate(data, overall)
    plot_combined_figure(data, overall)
    copy_to_artifacts()
    print("Hoan tat ve bieu do thanh cong!")
