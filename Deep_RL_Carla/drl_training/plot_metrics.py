"""Ve bieu do tu log huan luyen/danh gia DRL (PPO/SAC) de dua vao bao cao do an.

Doc truc tiep `episode_log.csv` + `update_log.csv` do `train_ppo.py`/`train_sac.py` sinh ra
(cung thu muc `output` cua moi lan train), va tuy chon `eval_results.csv` do `evaluate.py`
sinh ra khi chay voi `--eval-csv-out` (xem `config.py`). KHONG can CARLA/torch — chi can
`numpy`+`matplotlib`, nen chay duoc tren may viet bao cao (khac may train) mien co Python 3.

Moi bieu do xuat ra 2 dinh dang: `.png` (nhung vao Word) va `.pdf` (vector, nhung vao
LaTeX/Overleaf), cung mot bang mau danh muc co dinh xuyen suot: PPO=xanh duong, SAC=cam,
IL (baseline, neu co)=xanh la — mau luon gan voi "thuat toan", khong doi theo bieu do, de
nguoi doc bao cao khong bi nham lan giua cac hinh khac nhau.

Vi du:
    python plot_metrics.py --ppo-dir runs/ppo_lane_keep --sac-dir runs/sac_lane_keep \\
        --eval-ppo-csv runs/ppo_lane_keep/eval_results.csv \\
        --eval-sac-csv runs/sac_lane_keep/eval_results.csv \\
        --il-mae 0.25 --output ./report_figures

Chi co 1 thuat toan (vd chi PPO)? Bo qua --sac-dir/--eval-sac-csv, script tu bo qua cac bieu
do can ca hai (so sanh PPO vs SAC) va chi ve cac bieu do rieng cua PPO.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # khong can man hinh/GUI — chi xuat file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------- bang mau / style
# Bang mau danh muc co dinh (validated categorical palette — xem skill "dataviz" cua repo
# cong cu): mau gan voi THUC THE (thuat toan), giu nguyen xuyen suot moi bieu do.
COLOR_PPO = "#2a78d6"       # slot 1 - blue
COLOR_SAC = "#eb6834"       # slot 2 - orange
COLOR_IL = "#1baf7a"        # slot 3 - aqua (baseline tham chieu, khong phai DRL)

# Mau trang thai (status) danh cho terminate_reason — day la mot thang do "an toan", khong
# phai danh tinh tuy y, nen dung dung bang mau status thay vi danh muc.
COLOR_COLLISION = "#d03b3b"   # critical
COLOR_OFF_LANE = "#ec835a"    # serious
COLOR_TIME_LIMIT = "#0ca30c"  # good — ket thuc an toan (het gio, khong va cham/lech lan)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

TERMINATE_ORDER = ("collision", "off_lane", "time_limit")
TERMINATE_LABEL = {"collision": "Va cham", "off_lane": "Lech lan", "time_limit": "Het gio (an toan)"}
TERMINATE_COLOR = {"collision": COLOR_COLLISION, "off_lane": COLOR_OFF_LANE, "time_limit": COLOR_TIME_LIMIT}


def _apply_style():
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


def _savefig(fig, output_dir, stem, close=True):
    """`close=False` (dung trong notebook, xem plot_metrics.ipynb): giu figure mo de
    `%matplotlib inline` tu hien no ngay duoi cell thay vi bien mat sau khi luu file — CLI
    (`main()` ben duoi) luon dung mac dinh `close=True` vi khong can hien inline."""
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / ("%s.%s" % (stem, ext))
        # bbox_inches="tight": mo rong canvas de chua het legend/nhan xoay thay vi cat —
        # quan trong voi legend dat ngoai truc (vd plot_terminate_reason ben duoi).
        fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    print("  ->", output_dir / ("%s.png / .pdf" % stem))
    if close:
        plt.close(fig)


# ---------------------------------------------------------------------------- doc du lieu
def _load_csv(path):
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if rows else None


def _col(rows, key, cast=float):
    out = []
    for row in rows:
        try:
            out.append(cast(row[key]))
        except (KeyError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float if cast is float else object)


def _step_key(rows):
    return "global_step" if "global_step" in rows[0] else "step"


def _smooth(x, y, window):
    """Rolling mean cua `y` (bo qua NaN trong tung cua so), tra ve (x_smooth, y_smooth) da
    can chinh do dai — dung cho duong hoc (learning curve) vi episode_reward tho dao dong
    manh giua cac episode."""
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(y) == 0:
        return x, y
    window = max(1, min(window, len(y)))
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    x_smooth = x[window - 1:]
    return x_smooth, y_smooth


class Run(object):
    """Mot lan train (PPO hoac SAC) da doc san episode_log.csv/update_log.csv/eval CSV."""

    def __init__(self, name, color, run_dir=None, eval_csv=None):
        self.name = name
        self.color = color
        self.episode_rows = None
        self.update_rows = None
        self.eval_rows = None
        if run_dir is not None:
            run_dir = Path(run_dir).expanduser().resolve()
            self.episode_rows = _load_csv(run_dir / "episode_log.csv")
            self.update_rows = _load_csv(run_dir / "update_log.csv")
            if self.episode_rows is None:
                print("[!] Khong tim thay/rong episode_log.csv trong", run_dir)
            if self.update_rows is None:
                print("[!] Khong tim thay/rong update_log.csv trong", run_dir)
        if eval_csv is not None:
            self.eval_rows = _load_csv(Path(eval_csv).expanduser().resolve())
            if self.eval_rows is None:
                print("[!] Khong tim thay/rong file eval CSV:", eval_csv)


# --------------------------------------------------------------------------- cac bieu do
def plot_learning_curve(runs, metric_key, ylabel, title, stem, output_dir, window, show=False):
    """Duong hoc: gia tri tho (nhat, lam nen) + trung binh truot (net day) theo global step.
    Toi da 2 chuoi (PPO/SAC) tren cung 1 truc — dung 'trend over time' + mau danh muc theo
    thuat toan (dataviz skill: line chart, categorical color theo thuc the)."""
    usable = [r for r in runs if r.episode_rows]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in usable:
        step = _col(run.episode_rows, _step_key(run.episode_rows))
        y = _col(run.episode_rows, metric_key)
        ax.plot(step, y, color=run.color, alpha=0.15, linewidth=1.0, zorder=1)
        x_s, y_s = _smooth(step, y, window)
        ax.plot(x_s, y_s, color=run.color, linewidth=2.0, solid_capstyle="round",
                 label="%s (trung binh truot %d episode)" % (run.name, window), zorder=2)
    ax.set_xlabel("Buoc moi truong (global step)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(usable) >= 1:
        ax.legend(loc="best")
    _savefig(fig, output_dir, stem, close=not show)


def plot_update_diagnostic(runs, panels, title, stem, output_dir, window, show=False):
    """Small multiples: moi panel 1 truc y rieng (khong bao gio dual-axis) — dung cho cac
    dai luong chan doan (loss, KL, alpha, mean_q, ...) cua tung thuat toan. `panels`:
    list (metric_key, ylabel, hline) voi `hline` la (gia_tri, nhan) tuy chon."""
    usable = [r for r in runs if r.update_rows]
    if not usable:
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (metric_key, ylabel, hline) in zip(axes, panels):
        for run in usable:
            step = _col(run.update_rows, _step_key(run.update_rows))
            y = _col(run.update_rows, metric_key)
            ax.plot(step, y, color=run.color, alpha=0.2, linewidth=1.0)
            x_s, y_s = _smooth(step, y, max(1, window // 2))
            ax.plot(x_s, y_s, color=run.color, linewidth=2.0, solid_capstyle="round", label=run.name)
        if hline is not None:
            value, label = hline
            ax.axhline(value, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=0)
            ax.text(0.99, value, label, transform=ax.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=8, color=INK_MUTED)
        ax.set_xlabel("Buoc moi truong")
        ax.set_ylabel(ylabel)
        if len(usable) > 1:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    _savefig(fig, output_dir, stem, close=not show)


def plot_terminate_reason(runs, stem, output_dir, n_bins=10, show=False):
    """Bieu do cot xep chong 100% (part-to-whole) theo ti le terminate_reason trong tung
    'cua so' episode lien tiep doc theo qua trinh train — moi thuat toan 1 subplot rieng
    (small multiples, KHONG xep chung PPO/SAC vao 1 cot vi day la 2 thuc the khac nhau).
    Mau theo NGU NGHIA trang thai (status palette: do=va cham, cam=lech lan, xanh=an toan),
    khong dung bang mau danh muc thuat toan o day."""
    usable = [r for r in runs if r.episode_rows]
    if not usable:
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(5.5 * len(usable), 3.8), squeeze=False)
    axes = axes[0]
    for ax, run in zip(axes, usable):
        step = _col(run.episode_rows, _step_key(run.episode_rows))
        reason = np.array([row.get("terminate_reason", "time_limit") for row in run.episode_rows])
        order = np.argsort(step)
        step, reason = step[order], reason[order]

        edges = np.linspace(step.min(), step.max() + 1e-9, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = np.clip(np.digitize(step, edges) - 1, 0, n_bins - 1)
        width = (edges[1] - edges[0]) * 0.85

        bottoms = np.zeros(n_bins)
        for cat in TERMINATE_ORDER:
            fracs = np.zeros(n_bins)
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    fracs[b] = np.mean(reason[mask] == cat)
            ax.bar(centers, fracs, bottom=bottoms, width=width, color=TERMINATE_COLOR[cat],
                    label=TERMINATE_LABEL[cat] if run is usable[0] else None)
            bottoms += fracs

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Buoc moi truong (theo cua so ~%d episode)" % max(1, len(step) // n_bins))
        ax.set_ylabel("Ti le episode")
        ax.set_title(run.name)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Ly do ket thuc episode theo tien trinh huan luyen")
    _savefig(fig, output_dir, stem, close=not show)


def plot_eval_comparison(runs, il_mae, il_collision_rate, stem, output_dir, show=False):
    """3 subplot (small multiples, 1 truc/subplot) so sanh: reward trung binh +/- std,
    ti le va cham (%), va lech lan trung binh |m| — giua cac thuat toan (+ IL neu co so
    lieu baseline). Khong gop vao 1 truc vi 3 dai luong khac don vi/thang do hoan toan."""
    usable = [r for r in runs if r.eval_rows]
    if not usable and il_mae is None:
        return

    names, colors = [], []
    rewards_mean, rewards_std = [], []
    collision_rate, lane_offset = [], []
    for run in usable:
        rewards = _col(run.eval_rows, "reward")
        collided = np.array([row["collided"] in ("True", "true", "1") for row in run.eval_rows])
        offsets = _col(run.eval_rows, "mean_abs_lane_offset")
        names.append(run.name); colors.append(run.color)
        rewards_mean.append(float(np.mean(rewards))); rewards_std.append(float(np.std(rewards)))
        collision_rate.append(100.0 * float(np.mean(collided)))
        lane_offset.append(float(np.mean(offsets)))

    fig, (ax_r, ax_c, ax_l) = plt.subplots(1, 3, figsize=(12, 4))

    ax_r.bar(names, rewards_mean, yerr=rewards_std, color=colors, width=0.55, capsize=4)
    ax_r.set_title("Reward trung binh / episode")
    ax_r.set_ylabel("Reward")

    c_names, c_colors, c_vals = list(names), list(colors), list(collision_rate)
    if il_collision_rate is not None:
        c_names.append("IL (baseline)"); c_colors.append(COLOR_IL); c_vals.append(il_collision_rate)
    ax_c.bar(c_names, c_vals, color=c_colors, width=0.55)
    ax_c.set_title("Ti le va cham")
    ax_c.set_ylabel("%")

    l_names, l_colors, l_vals = list(names), list(colors), list(lane_offset)
    if il_mae is not None:
        l_names.append("IL (baseline)"); l_colors.append(COLOR_IL); l_vals.append(il_mae)
    ax_l.bar(l_names, l_vals, color=l_colors, width=0.55)
    ax_l.set_title("Lech lan trung binh |lane_offset_m|")
    ax_l.set_ylabel("met")

    for ax in (ax_r, ax_c, ax_l):
        ax.tick_params(axis="x", labelrotation=15)

    fig.suptitle("So sanh danh gia cuoi (evaluate.py --deterministic)")
    _savefig(fig, output_dir, stem, close=not show)


def plot_throughput(runs, stem, output_dir, window, show=False):
    """steps_per_sec theo thoi gian train — bieu do phu tro danh gia hieu nang he thong,
    khong phai chat luong policy (dua vao update_log.csv ca 2 thuat toan)."""
    usable = [r for r in runs if r.update_rows]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run in usable:
        step = _col(run.update_rows, _step_key(run.update_rows))
        y = _col(run.update_rows, "steps_per_sec")
        x_s, y_s = _smooth(step, y, max(1, window // 2))
        ax.plot(x_s, y_s, color=run.color, linewidth=2.0, solid_capstyle="round", label=run.name)
    ax.set_xlabel("Buoc moi truong")
    ax.set_ylabel("Buoc/giay")
    ax.set_title("Thong luong huan luyen (steps/s)")
    ax.legend(loc="best")
    _savefig(fig, output_dir, stem, close=not show)


# --------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Ve bieu do tu log train_ppo.py/train_sac.py/evaluate.py de dua vao bao cao.")
    parser.add_argument("--ppo-dir", default=None, help="Thu muc output cua train_ppo.py (chua episode_log.csv/update_log.csv)")
    parser.add_argument("--sac-dir", default=None, help="Thu muc output cua train_sac.py")
    parser.add_argument("--eval-ppo-csv", default=None, help="File CSV tu 'evaluate.py --eval-csv-out' (PPO)")
    parser.add_argument("--eval-sac-csv", default=None, help="File CSV tu 'evaluate.py --eval-csv-out' (SAC)")
    parser.add_argument("--il-mae", type=float, default=None,
                         help="Lech lan trung binh |m| cua rieng buoc IL (muc 11 notebook IL) — ve them lam baseline")
    parser.add_argument("--il-collision-rate", type=float, default=None,
                         help="Ti le va cham (%%) cua rieng buoc IL, neu co so lieu")
    parser.add_argument("--reward-window", type=int, default=20,
                         help="Do rong cua so trung binh truot cho duong hoc reward/do dai episode (mac dinh 20, khop voi 'recent_episode_rewards' trong train_*.py)")
    parser.add_argument("--output", default="./report_figures", help="Thu muc luu bieu do (.png + .pdf)")
    args = parser.parse_args()

    if not args.ppo_dir and not args.sac_dir:
        parser.error("Can it nhat mot trong --ppo-dir / --sac-dir.")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    runs = [
        Run("PPO", COLOR_PPO, run_dir=args.ppo_dir, eval_csv=args.eval_ppo_csv),
        Run("SAC", COLOR_SAC, run_dir=args.sac_dir, eval_csv=args.eval_sac_csv),
    ]
    runs = [r for r in runs if r.episode_rows or r.update_rows or r.eval_rows]
    if not runs:
        sys.exit("Khong doc duoc du lieu nao tu cac duong dan da cho — kiem tra lai --ppo-dir/--sac-dir.")

    print("Dang ve bieu do vao:", output_dir)

    plot_learning_curve(runs, "episode_reward", "Reward / episode", "Duong hoc — Reward theo episode",
                          "01_reward_curve", output_dir, args.reward_window)
    plot_learning_curve(runs, "episode_len", "So buoc / episode", "Duong hoc — Do dai episode",
                          "02_episode_length", output_dir, args.reward_window)
    plot_terminate_reason(runs, "03_terminate_reason", output_dir)

    ppo_run = next((r for r in runs if r.name == "PPO" and r.update_rows), None)
    if ppo_run:
        plot_update_diagnostic([ppo_run],
            [("policy_loss", "Policy loss", None), ("value_loss", "Value loss", None)],
            "PPO — Policy loss & Value loss", "04_ppo_losses", output_dir, args.reward_window)
        target_kl = 0.02
        plot_update_diagnostic([ppo_run],
            [("approx_kl", "Approx. KL", (target_kl, "target_kl=%.2f" % target_kl)),
             ("clip_fraction", "Ti le bi clip", None),
             ("entropy", "Entropy chinh sach", None)],
            "PPO — Chan doan huan luyen", "05_ppo_diagnostics", output_dir, args.reward_window)

    sac_run = next((r for r in runs if r.name == "SAC" and r.update_rows), None)
    if sac_run:
        plot_update_diagnostic([sac_run],
            [("critic_loss", "Critic loss", None), ("actor_loss", "Actor loss", None)],
            "SAC — Critic loss & Actor loss", "06_sac_losses", output_dir, args.reward_window)
        plot_update_diagnostic([sac_run],
            [("alpha", "Temperature (alpha)", None), ("mean_q", "Mean Q", None),
             ("entropy", "Entropy chinh sach", None)],
            "SAC — Chan doan huan luyen", "07_sac_diagnostics", output_dir, args.reward_window)

    plot_throughput(runs, "08_throughput", output_dir, args.reward_window)
    plot_eval_comparison(runs, args.il_mae, args.il_collision_rate, "09_eval_comparison", output_dir)

    print("Xong. Nhung file .png vao Word hoac .pdf vao LaTeX/Overleaf tuy y.")


if __name__ == "__main__":
    main()
