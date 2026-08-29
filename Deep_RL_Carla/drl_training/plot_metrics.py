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

# So episode toi thieu moi cot cua bieu do 03. Duoi muc nay, mot cot chi phan anh 1-2
# episode nen "ti le" khong con y nghia thong ke — thà it cot ma doc duoc.
_MIN_EPISODES_PER_BIN = 5

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


def _bar_labels(ax, xs, values, fmt="%.2f", dy=0.02):
    """Ghi gia tri ngay tren dau moi cot.

    Bat buoc, khong phai trang tri: `scripts/validate_palette.js` cua skill dataviz bao
    WARN tuong phan cho COLOR_IL (#1baf7a = 2.74:1 tren nen #fcfcfb, duoi nguong 3:1), va
    theo skill thi WARN do "obligates visible labels or a table view — it is not
    dismissable". Nhan cung giup nguoi doc bao cao lay so ma khong phai do bang thuoc.
    Chu dung INK, khong dung mau cua cot — mau la danh tinh cua cot, khong phai cua chu.
    """
    finite = [v for v in values if np.isfinite(v)]
    span = (max(finite) - min(min(finite), 0.0)) if finite else 1.0
    offset = (span or 1.0) * dy
    for x, v in zip(xs, values):
        if not np.isfinite(v) or not fmt:
            continue          # fmt="" = chi noi rong thang, nguoi goi tu ve nhan
        ax.annotate(fmt % v, xy=(x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    if finite:
        # CHI noi rong, khong bao gio thu nho: ham nay duoc goi nhieu lan tren cung mot truc
        # (bieu do cot ghep doi), va neu lan sau dat top theo nhom gia tri nho hon thi nhom
        # lon hon bi CAT CUT cung nhan cua no — dung loi da xay ra o 00b truoc khi sua.
        ax.set_ylim(top=max(ax.get_ylim()[1], max(finite) + offset * 4))


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


def _plot_series(ax, x, y, color, window, label):
    """Ve mot chuoi: gia tri tho (nhat, lam nen) + trung binh truot (net day).

    Cua so truot bi CHAN o ~1/3 so diem. `np.convolve(mode="valid")` tra ve
    len(y)-window+1 diem, nen window=20 tren 13 diem cho DUNG MOT diem — khong ve ra duong
    nao, va hinh chi con lai duong tho mo nhat gan nhu vo hinh. Do la loi da gap that o lan
    chay demo 5120 buoc; no se lap lai o dau MOI lan train, khi so episode/update con it.

    Khi khong du diem de lam muot, ve thang duong tho o do dam DAY DU kem marker — tha hinh
    tho con hon hinh trong. Tra ve nhan da dung de goi y dat legend cho dung.
    """
    n_ok = int(np.sum(~np.isnan(y)))
    eff_window = max(1, min(window, n_ok // 3))
    x_s, y_s = _smooth(x, y, eff_window)
    if len(y_s) >= 2:
        if eff_window > 1:
            # Duong tho chi lam nen khi that su CO lam muot. Cua so 1 thi "trung binh truot"
            # chinh la duong tho, ve chong len chinh no va ghi nhan "TB truot 1" la noi doi.
            ax.plot(x, y, color=color, alpha=0.18, linewidth=1.0, zorder=1)
        ax.plot(x_s, y_s, color=color, linewidth=2.0, solid_capstyle="round",
                label=("%s (TB truot %d)" % (label, eff_window)) if eff_window > 1 else label,
                zorder=2)
    else:
        ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round",
                marker="o", markersize=4, label=label, zorder=2)


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


class IlDemo(object):
    """Mot lan chay `demo_il.py` (baseline IL vong kin, CHUA co DRL) da doc san.

    Day la moc "truoc" cua bao cao: policy IL chay tren CARLA that, khong hoc gi them. Moi
    con so DRL sau nay deu phai doi chieu ve day, neu khong thi khong noi duoc DRL cai
    thien bao nhieu.
    """

    def __init__(self, label, csv_path):
        self.label = label
        self.rows = _load_csv(Path(csv_path).expanduser().resolve())
        if self.rows is None:
            print("[!] Khong doc duoc il_demo_results.csv:", csv_path)

    @property
    def ok(self):
        return bool(self.rows)

    def col(self, key):
        return _col(self.rows, key)

    @property
    def collided(self):
        return np.array([r.get("collided") in ("True", "true", "1") for r in self.rows])

    @property
    def junction_rate(self):
        """Ti le buoc nam trong nga tu. NaN neu file cu (truoc khi them cot junction_steps)."""
        js, st = self.col("junction_steps"), self.col("steps")
        if not np.isfinite(js).any() or np.nansum(st) == 0:
            return float("nan")
        return float(np.nansum(js) / np.nansum(st))


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
        _plot_series(ax, step, y, run.color, window, run.name)

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
            _plot_series(ax, step, y, run.color, max(1, window // 2), run.name)
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

        # So bin PHAI thich ung theo so episode. Voi n_bins co dinh = 10 va 13 episode thi
        # moi bin chua dung 1 episode, nen moi cot la 100% mot mau — hinh thanh mot dai mau
        # chu khong con la bieu do ti le. Bat moi bin co it nhat _MIN_EPISODES_PER_BIN mau.
        bins = int(np.clip(len(step) // _MIN_EPISODES_PER_BIN, 1, n_bins))
        edges = np.linspace(step.min(), step.max() + 1e-9, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = np.clip(np.digitize(step, edges) - 1, 0, bins - 1)
        width = (edges[1] - edges[0]) * 0.88

        bottoms = np.zeros(bins)
        for cat in TERMINATE_ORDER:
            fracs = np.zeros(bins)
            for b in range(bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    fracs[b] = np.mean(reason[mask] == cat)
            # Vien mau NEN giua cac mang xep chong = "spacer" 2px theo chuan dataviz: no
            # tach ranh gioi hai mang cung do dam ma khong them mot mau moi vao hinh.
            ax.bar(centers, fracs, bottom=bottoms, width=width, color=TERMINATE_COLOR[cat],
                    edgecolor=SURFACE, linewidth=1.4,
                    label=TERMINATE_LABEL[cat] if run is usable[0] else None)
            bottoms += fracs

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Buoc moi truong (moi cot ~%d episode)" % max(1, len(step) // bins))
        ax.set_ylabel("Ti le episode")
        ax.set_title(run.name)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Ly do ket thuc episode theo tien trinh huan luyen")
    _savefig(fig, output_dir, stem, close=not show)


def plot_eval_comparison(runs, il_mae, il_collision_rate, il_reward, stem, output_dir, show=False):
    """3 subplot (small multiples, 1 truc/subplot) so sanh: reward trung binh +/- std,
    ti le va cham (%), va lech lan trung binh |m| — giua cac thuat toan (+ IL neu co so
    lieu baseline). Khong gop vao 1 truc vi 3 dai luong khac don vi/thang do hoan toan."""
    usable = [r for r in runs if r.eval_rows]
    if not usable:
        # Day la bieu do SO SANH. Chi co mot cot IL thi panel Reward trong tron (baseline IL
        # khong co reward cua evaluate.py) va cot con lai keo dai het truc — vo nghia va de
        # gay hieu nham. Bo qua, va noi ro can gi de co no.
        print("  (bo qua 09_eval_comparison: chua co eval CSV nao. Chay evaluate.py voi "
              "--eval-csv-out roi truyen --eval-ppo-csv / --eval-sac-csv)")
        return

    names, colors = [], []
    rewards_mean, rewards_std = [], []
    collision_rate, lane_offset = [], []
    for run in usable:
        rewards = _col(run.eval_rows, "reward")
        collided = np.array([row["collided"] in ("True", "true", "1") for row in run.eval_rows])
        # Uu tien so do CHI DUONG THUONG de cung mot thang voi baseline IL (xem
        # il_mae trong main()). File eval cu — sinh truoc khi evaluate.py tach nga tu —
        # khong co cot nay, khi do quay ve cot cu de van ve duoc.
        offsets = _col(run.eval_rows, "mean_abs_lane_offset_road")
        if not np.isfinite(offsets).any():
            offsets = _col(run.eval_rows, "mean_abs_lane_offset")
            print("  [!] %s: eval CSV chua co 'mean_abs_lane_offset_road' (file cu) — dung "
                  "so do LAN NGA TU, khong so sanh truc tiep duoc voi baseline IL." % run.name)
        names.append(run.name); colors.append(run.color)
        rewards_mean.append(float(np.mean(rewards))); rewards_std.append(float(np.std(rewards)))
        collision_rate.append(100.0 * float(np.mean(collided)))
        lane_offset.append(float(np.mean(offsets)))

    fig, (ax_r, ax_c, ax_l) = plt.subplots(1, 3, figsize=(12, 4))

    # IL cung vao panel nay: `il_demo_results.csv` co cot `reward` sinh tu CHINH ham reward
    # cua env, voi cung `max_episode_steps` — nen no so sanh truc tiep duoc, va no chinh la
    # cot "truoc khi co DRL" ma bao cao can de cho thay hieu qua.
    r_names, r_colors = list(names), list(colors)
    r_mean, r_std = list(rewards_mean), list(rewards_std)
    if il_reward is not None:
        r_names.append("IL (baseline)"); r_colors.append(COLOR_IL)
        r_mean.append(il_reward[0]); r_std.append(il_reward[1])
    ax_r.bar(r_names, r_mean, yerr=r_std, color=r_colors, width=0.55, capsize=4)
    # Nhan dat tren DINH thanh sai so, neu khong no de len dau mu cua thanh do.
    _bar_labels(ax_r, np.arange(len(r_names)),
                [m + sd for m, sd in zip(r_mean, r_std)], fmt="", dy=0.02)
    for i, m in enumerate(r_mean):
        ax_r.annotate("%.0f" % m, xy=(i, m + r_std[i]), xytext=(0, 6),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=8, color=INK_SECONDARY)
    ax_r.set_title("Reward trung binh / episode")
    ax_r.set_ylabel("Reward")

    c_names, c_colors, c_vals = list(names), list(colors), list(collision_rate)
    if il_collision_rate is not None:
        c_names.append("IL (baseline)"); c_colors.append(COLOR_IL); c_vals.append(il_collision_rate)
    ax_c.bar(c_names, c_vals, color=c_colors, width=0.55)
    _bar_labels(ax_c, np.arange(len(c_names)), c_vals, fmt="%.0f%%")
    ax_c.set_title("Ti le va cham")
    ax_c.set_ylabel("%")

    l_names, l_colors, l_vals = list(names), list(colors), list(lane_offset)
    if il_mae is not None:
        l_names.append("IL (baseline)"); l_colors.append(COLOR_IL); l_vals.append(il_mae)
    ax_l.bar(l_names, l_vals, color=l_colors, width=0.55)
    _bar_labels(ax_l, np.arange(len(l_names)), l_vals, fmt="%.3f")
    ax_l.set_title("Lech lan trung binh |lane_offset_m|\n(chi duong thuong, bo nga tu)",
                    fontsize=10)
    ax_l.set_ylabel("met")

    # Dem theo danh sach CUA CHINH panel do, khong phai `names`: panel Reward co them cot
    # IL nen dai hon, va lay nham do dai se cat cot cuoi ra ngoai khung.
    for ax, n in ((ax_r, len(r_names)), (ax_c, len(c_names)), (ax_l, len(l_names))):
        ax.tick_params(axis="x", labelrotation=15)
        # Khong de cot phinh ra chiem het truc khi chi co 1-2 danh muc.
        ax.set_xlim(-0.7, max(n - 1, 0) + 0.7)

    fig.suptitle("So sanh danh gia cuoi (evaluate.py --deterministic)")
    _savefig(fig, output_dir, stem, close=not show)


def plot_il_closed_loop(il_demos, stem, output_dir, show=False):
    """Small multiples: moi lan chay mot subplot, moi cot mot episode = quang duong di duoc,
    to mau theo LY DO KET THUC.

    Vi sao cot chu khong phai duong: truc x la danh tinh (episode roi rac, khong co thu tu
    thoi gian y nghia), va so luong it — do la dinh nghia cua bieu do cot. Mau o day la
    thang do TRANG THAI (an toan -> nguy hiem), khong phai danh muc, nen dung bang mau
    status rieng chu khong dung mau thuat toan.
    """
    usable = [d for d in il_demos if d.ok]
    if not usable:
        return
    # sharey BAT BUOC voi small multiples: neu moi panel tu chia thang rieng thi cot 275m
    # cua Town03 trong cao ngang cot 774m cua Town04, va nguoi doc so sanh sai hoan toan.
    # Cai gia phai tra la panel co gia tri nho trong "lun" — dung y do, do la su that.
    fig, axes = plt.subplots(1, len(usable), figsize=(5.2 * len(usable), 4.0),
                             squeeze=False, sharey=True)
    for ax, demo in zip(axes[0], usable):
        dist = demo.col("distance_m")
        reasons = [r.get("terminate_reason", "time_limit") for r in demo.rows]
        xs = np.arange(len(dist))
        colors = [TERMINATE_COLOR.get(r, INK_MUTED) for r in reasons]
        ax.bar(xs, dist, color=colors, width=0.6)
        _bar_labels(ax, xs, dist, fmt="%.0f m")
        ax.set_xticks(xs)
        ax.set_xticklabels(["ep %d" % i for i in range(len(dist))])
        if ax is axes[0][0]:
            ax.set_ylabel("Quang duong (m)")
        jr = demo.junction_rate
        sub = "" if not np.isfinite(jr) else "  |  nga tu %.0f%% thoi gian" % (100 * jr)
        ax.set_title("%s%s" % (demo.label, sub))
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERMINATE_COLOR[k]) for k in TERMINATE_ORDER]
    fig.legend(handles, [TERMINATE_LABEL[k] for k in TERMINATE_ORDER],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Baseline IL vong kin — quang duong va ly do ket thuc moi episode")
    _savefig(fig, output_dir, stem, close=not show)


def plot_il_lane_keeping(il_demos, stem, output_dir, pass_threshold=0.35, show=False):
    """Bam lan cua IL: |lech| tren TOAN BO buoc so voi chi rieng DUONG THUONG.

    Day la bieu do quan trong nhat ve mat phuong phap trong bao cao. Trong nga tu,
    `lane_offset_m` duoc suy ra tu "lan duong gan nhat", ma cac nhanh cat nhau nen tham
    chieu do nhay sang nhanh vuong goc chi sau vai met — con so do vo nghia. Gop no vao
    trung binh lam nang luc bam lan trong te hon THUC TE rat nhieu.

    Cot xam = so do bi nhiem nga tu (cai dang duoc sua), cot xanh = so do that. Ti le thoi
    gian trong nga tu ghi o nhan truc x chu KHONG ve thanh cot thu ba: no la don vi phan
    tram, ve chung truc voi met se thanh bieu do hai truc — loi so mot cua truc quan hoa.
    """
    usable = [d for d in il_demos if d.ok]
    if not usable:
        return
    labels, all_off, road_off = [], [], []
    for demo in usable:
        jr = demo.junction_rate
        suffix = "" if not np.isfinite(jr) else "\n(nga tu %.0f%%)" % (100 * jr)
        labels.append(demo.label + suffix)
        all_off.append(float(np.nanmean(demo.col("mean_abs_offset"))))
        road_off.append(float(np.nanmean(demo.col("mean_abs_offset_road"))))

    xs = np.arange(len(labels))
    width, gap = 0.36, 0.02      # khe giua hai cot cung nhom (spacer, xem marks-and-anatomy)
    off = width / 2 + gap / 2
    fig, ax = plt.subplots(figsize=(2.4 * len(labels) + 3.6, 4.4))
    ax.bar(xs - off, all_off, width=width, color=BASELINE,
           label="Tinh ca nga tu (so do bi nhiem)")
    ax.bar(xs + off, road_off, width=width, color=COLOR_IL,
           label="Chi duong thuong (so do that)")
    ax.axhline(pass_threshold, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=0)
    # Nguong phai nam trong khung nhin, neu khong thi ve xong khong ai thay no.
    ax.set_ylim(top=pass_threshold * 1.12)
    _bar_labels(ax, xs - off, all_off, fmt="%.3f")
    _bar_labels(ax, xs + off, road_off, fmt="%.3f")
    # Dat nhan o MEP PHAI: phia tren duong nguong ben trai thuong co cot cao (chinh la cot
    # dang vuot nguong — thu ta muon nguoi doc nhin), nen de chu o do se de len cot.
    ax.annotate("nguong dat %.2f m" % pass_threshold,
                xy=(len(labels) - 0.45, pass_threshold), xytext=(-2, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("|lech lan| trung binh (m)")
    ax.set_title("Baseline IL — bam lan do dung cach vs do lan nga tu")
    # Legend dat NGOAI vung ve: trong khung no de de len cot cao nhat.
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=2)
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
        _plot_series(ax, step, y, run.color, max(1, window // 2), run.name)
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
    parser.add_argument("--il-demo-csv", action="append", default=None, metavar="NHAN=DUONGDAN",
                         help="Ket qua demo_il.py, dang 'Town04=runs/il_demo_v9/il_demo_results.csv'. "
                              "Lap lai co nay de so sanh nhieu ban do. Cac gia tri baseline IL "
                              "trong bieu do 09 duoc suy ra tu day neu khong truyen --il-mae")
    parser.add_argument("--il-mae", type=float, default=None,
                         help="Lech lan trung binh |m| cua rieng buoc IL (muc 11 notebook IL) — ve them lam baseline")
    parser.add_argument("--il-collision-rate", type=float, default=None,
                         help="Ti le va cham (%%) cua rieng buoc IL, neu co so lieu")
    parser.add_argument("--reward-window", type=int, default=20,
                         help="Do rong cua so trung binh truot cho duong hoc reward/do dai episode (mac dinh 20, khop voi 'recent_episode_rewards' trong train_*.py)")
    parser.add_argument("--output", default="./report_figures", help="Thu muc luu bieu do (.png + .pdf)")
    args = parser.parse_args()

    if not args.ppo_dir and not args.sac_dir and not args.il_demo_csv:
        parser.error("Can it nhat mot trong --ppo-dir / --sac-dir / --il-demo-csv.")

    il_demos = []
    for spec in (args.il_demo_csv or []):
        if "=" not in spec:
            parser.error("--il-demo-csv phai co dang NHAN=DUONGDAN, nhan duoc: %r" % spec)
        label, _, path = spec.partition("=")
        il_demos.append(IlDemo(label.strip(), path.strip()))
    il_demos = [d for d in il_demos if d.ok]

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    runs = [
        Run("PPO", COLOR_PPO, run_dir=args.ppo_dir, eval_csv=args.eval_ppo_csv),
        Run("SAC", COLOR_SAC, run_dir=args.sac_dir, eval_csv=args.eval_sac_csv),
    ]
    runs = [r for r in runs if r.episode_rows or r.update_rows or r.eval_rows]
    if not runs and not il_demos:
        sys.exit("Khong doc duoc du lieu nao tu cac duong dan da cho — kiem tra lai "
                 "--ppo-dir/--sac-dir/--il-demo-csv.")

    # Baseline IL cho bieu do 09: uu tien so do THAT (duong thuong) thay vi so bi nhiem
    # nga tu. --il-mae truyen tay van thang, de con doi chung duoc.
    il_mae, il_collision_rate, il_reward = args.il_mae, args.il_collision_rate, None
    if il_demos:
        if il_mae is None:
            il_mae = float(np.nanmean([np.nanmean(d.col("mean_abs_offset_road")) for d in il_demos]))
        if il_collision_rate is None:
            il_collision_rate = 100.0 * float(np.mean(np.concatenate([d.collided for d in il_demos])))
        _r = np.concatenate([d.col("reward") for d in il_demos])
        il_reward = (float(np.nanmean(_r)), float(np.nanstd(_r)))

    print("Dang ve bieu do vao:", output_dir)

    if il_demos:
        plot_il_closed_loop(il_demos, "00_il_closed_loop", output_dir)
        plot_il_lane_keeping(il_demos, "00b_il_lane_keeping", output_dir)

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
    plot_eval_comparison(runs, il_mae, il_collision_rate, il_reward, "09_eval_comparison", output_dir)

    print("Xong. Nhung file .png vao Word hoac .pdf vao LaTeX/Overleaf tuy y.")


if __name__ == "__main__":
    main()
