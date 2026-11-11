"""评价指标：违约为正类。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def pr_auc_trapz(y, p) -> float:
    """Precision–Recall 曲线下面积，梯形积分（与 AP 的阶梯求和不同，两者都报告）。"""
    prec, rec, _ = precision_recall_curve(y, p)
    return float(auc(rec, prec))


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """按预测概率分位数分箱：每箱人数、平均预测概率、实际违约率。"""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean"))
    return t.reset_index(drop=True)


def ece(y, p, n_bins: int = 10) -> float:
    """期望校准误差：各箱 |平均预测 − 实际违约率| 按人数加权。"""
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * (t["mean_p"] - t["obs"]).abs()).sum() / t["n"].sum())


def summarize(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ks": ks_stat(y, p),
        "pr_auc": pr_auc_trapz(y, p),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece(y, p),
        "mean_p": float(p.mean()),
    }


def bootstrap_auc_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """配对 bootstrap：AUC(a) − AUC(b) 的点估计与 95% 百分位区间。"""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(lo), float(hi)], "n_boot": len(diffs)}
