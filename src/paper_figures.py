"""按论文版面（正文宽 16 cm）重画插图，保证缩印后字号可读。

运行：.\\.venv\\Scripts\\python.exe -m src.paper_figures
只读取已保存的结果文件（reports/*.json、data/processed/valid_predictions.csv），不重新训练，也不读取测试集数据。
图保存到 reports/figures/paper/。
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data_loader import PROC_DIR, ROOT

OUT = ROOT / "reports" / "figures" / "paper"
W = 6.3  # 英寸，约 16 cm
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.dpi": 300,
})
pct = lambda v, _=None: f"{v:.0%}"
load = lambda name: json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def rate_panel(ax, rows, key, base, title, xlabel):
    g = pd.DataFrame(rows)
    x = np.arange(len(g))
    ax.bar(x, g["rate"], width=0.66, color=BLUE, yerr=g["ci95"], error_kw=dict(ecolor=MUTED, elinewidth=0.7, capsize=1.5))
    ax.axhline(base, color=ORANGE, lw=1, ls="--")
    for i, n in enumerate(g["n"]):
        ax.text(i, -0.025, f"{n:,}", ha="center", va="top", fontsize=6, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [str(v) for v in g[key]]); ax.tick_params(axis="x", pad=9)
    ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 0.82); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=1, ls="--", label=f"训练集平均 {base:.1%}")], loc="upper left")


def fig_eda():
    E = load("eda_tables.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
    rate_panel(axs[0], E["rate_by_pay0"], "k", E["base_rate"], "(a) 按 9 月还款状态", "还款状态编码（正值为逾期月数）")
    rate_panel(axs[1], E["rate_by_late_months"], "k", E["base_rate"], "(b) 按近六个月逾期月份数", "逾期月份数")
    axs[0].set_ylabel("违约率")
    fig.tight_layout(w_pad=1.2); save(fig, "p1_eda")


def fig_roc_pr():
    R = load("model_results.json")
    V = pd.read_csv(PROC_DIR / "valid_predictions.csv"); y = V["target"].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
    for key, name, col in [("lr_none", "逻辑回归", BLUE), ("rf_none", "随机森林", ORANGE)]:
        m = R["valid"][key]
        fpr, tpr, _ = roc_curve(y, V[f"p_{key}"]); axs[0].plot(fpr, tpr, color=col, lw=1.3, label=f"{name}（AUC {m['roc_auc']:.3f}）")
        prec, rec, _ = precision_recall_curve(y, V[f"p_{key}"]); axs[1].plot(rec, prec, color=col, lw=1.3, label=f"{name}（AP {m['ap']:.3f}）")
    axs[0].plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="基准（AUC 0.500）")
    axs[1].axhline(y.mean(), color=GRAY, lw=1, ls="--", label=f"基准（AP {y.mean():.3f}）")
    axs[0].set(xlabel="假阳性率", ylabel="真阳性率"); axs[0].set_title("(a) ROC 曲线", loc="left"); axs[0].legend(loc="lower right")
    axs[1].set(xlabel="召回率", ylabel="精确率", ylim=(0, 1.02)); axs[1].set_title("(b) Precision–Recall 曲线", loc="left"); axs[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5); save(fig, "p2_roc_pr")


def fig_calibration():
    R = load("model_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.15), sharey=True)
    for ax, fam, title in [(axs[0], "lr", "(a) 逻辑回归"), (axs[1], "rf", "(b) 随机森林")]:
        ax.plot([0, 1], [0, 1], color=GRAY, lw=0.9, ls="--", label="完美校准")
        for meth, name, col, mk in [("none", "不校准", BLUE, "o"), ("sigmoid", "Platt 缩放", ORANGE, "s"), ("isotonic", "保序回归", AQUA, "^")]:
            key = f"{fam}_{meth}"; t = pd.DataFrame(R["calibration"][key]); m = R["valid"][key]
            ax.plot(t["mean_p"], t["obs"], color=col, lw=1.2, marker=mk, ms=4, mec="white", mew=0.6,
                    label=f"{name}（Brier {m['brier']:.4f}，ECE {m['ece']:.3f}）")
        ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="平均预测违约概率", aspect="equal")
        ax.set_title(title, loc="left"); ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel("实际违约率")
    fig.tight_layout(w_pad=1.0); save(fig, "p3_calibration")


def fig_thresholds():
    P = load("policy_results.json"); cand = pd.DataFrame(P["candidates"])
    cap, maxdr = P["constraints"]["review_capacity"], P["constraints"]["max_pass_default_rate"]
    labels = [f"({c['t1']:.2f}, {c['t2']:.2f})" for c in P["candidates"]]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.55), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    ax = axs[0]; left = np.zeros(len(cand))
    for key, name, col in [("pass", "通过", BLUE), ("review", "复核", AQUA), ("suspend", "暂缓", ORANGE)]:
        v = cand[f"{key}_rate"].to_numpy()
        ax.barh(labels, v, left=left, color=col, height=0.58, edgecolor="white", linewidth=1, label=name)
        for i, (l, w) in enumerate(zip(left, v)):
            ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=6.5 if w >= 0.13 else 5.5,
                    color="white", fontweight="bold")
        left += v
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(pct); ax.grid(axis="y", visible=False)
    ax.set_title("(a) 各档人数占比", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, columnspacing=1.0, handlelength=1.2)

    def panel(ax, col, limit, title, lab, step):
        v = cand[col].to_numpy()
        ax.bar(range(len(v)), v, width=0.74, color=[BLUE if x <= limit else GRAY for x in v])
        ax.axhline(limit, color=ORANGE, lw=1, ls="--")
        for i, x in enumerate(v):  # 数值写在柱内中部，避免压住约束线
            ax.text(i, x / 2, f"{x:.1%}", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        ax.set_xticks(range(len(v)), [l.replace(", ", ",\n") for l in labels], fontsize=6.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(step)); ax.yaxis.set_major_formatter(pct)
        ax.set_ylim(0, max(v.max(), limit) * 1.15); ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
        ax.legend(handles=[Line2D([], [], color=ORANGE, lw=1, ls="--", label=lab)], loc="upper center", bbox_to_anchor=(0.5, -0.2))

    panel(axs[1], "review_rate", cap, "(b) 复核率", f"复核容量 {cap:.0%}", 0.1)
    panel(axs[2], "pass_default_rate", maxdr, "(c) 通过组违约率", f"上限 {maxdr:.0%}", 0.04)
    fig.tight_layout(w_pad=1.0); save(fig, "p4_thresholds")


def fig_test():
    T = load("test_results.json"); m, pol, t1, t2 = T["metrics"], T["policy"], T["t1"], T["t2"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]; t = pd.DataFrame(T["calibration"])
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.9, ls="--", label="完美校准")
    ax.plot(t["mean_p"], t["obs"], color=BLUE, lw=1.2, marker="o", ms=4, mec="white", mew=0.6, label="逻辑回归（测试集）")
    for x in (t1, t2):
        ax.axvline(x, color=MUTED, lw=0.7, ls=":")
    ax.text(t1 + 0.01, 0.02, f"$t_1$ = {t1:.2f}", fontsize=6.5, color=INK); ax.text(t2 + 0.01, 0.02, f"$t_2$ = {t2:.2f}", fontsize=6.5, color=INK)
    ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="平均预测违约概率", ylabel="实际违约率", aspect="equal")
    ax.set_title("(a) 校准曲线", loc="left"); ax.legend(loc="upper left")

    ax = axs[1]
    names = [f"通过\np < {t1:.2f}", f"复核\n{t1:.2f} ≤ p < {t2:.2f}", f"暂缓\np ≥ {t2:.2f}"]
    keys, cols = ["pass", "review", "suspend"], [BLUE, AQUA, ORANGE]
    v = [pol[f"{k}_default_rate"] for k in keys]; n = [pol[f"n_{k}"] for k in keys]
    err = [1.96 * np.sqrt(x * (1 - x) / k) for x, k in zip(v, n)]
    ax.bar(names, v, color=cols, width=0.58, yerr=err, error_kw=dict(ecolor=MUTED, elinewidth=0.7, capsize=2))
    for i, (x, k, e) in enumerate(zip(v, n, err)):
        ax.text(i, x + e + 0.012, f"{x:.1%}\n{k:,} 人", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(T["default_rate"], color=MUTED, lw=0.9, ls="--")
    ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 0.85); ax.grid(axis="x", visible=False); ax.set_ylabel("实际违约率")
    ax.set_title("(b) 三档客户的实际违约率", loc="left")
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=0.9, ls="--", label=f"测试集平均 {T['default_rate']:.1%}")], loc="upper left")
    fig.tight_layout(w_pad=1.5); save(fig, "p5_test")


if __name__ == "__main__":
    for f in (fig_eda, fig_roc_pr, fig_calibration, fig_thresholds, fig_test):
        f()
    print("已生成：", sorted(p.name for p in OUT.glob("*.png")))
