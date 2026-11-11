"""逐行特征工程：只用同一客户 4–9 月的字段计算，不使用任何跨样本统计量，
因此放在 Pipeline 最前面也不会造成泄漏。

依据见 reports/eda_report.md 第 4 节。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import BILL_AMT, PAY_AMT, PAY_STATUS

# 原说明未定义的类别编码并入“其他”
EDU_MAP = {0: 4, 5: 4, 6: 4}
MAR_MAP = {0: 3}

ENGINEERED = [
    "n_late",          # 近六个月 PAY_* > 0 的月份数
    "max_late",        # 最大逾期月数（未逾期记 0）
    "late_recent",     # 9 月是否逾期
    "pay_amt_mean",    # 近六个月平均实还金额
    "n_zero_pay",      # 近六个月实还为 0 的月份数
    "pay_ratio1",      # 9 月实还 / 8 月账单
    "pay_ratio_avg",   # 5–9 月实还合计 / 4–8 月正账单合计
    "util1",           # 9 月账单 / 额度
    "util_mean",       # 六个月平均账单 / 额度
    "bill_le0",        # 9 月账单 ≤ 0
    "bill_trend",      # (9 月账单 − 4 月账单) / 额度
]

RAW_NUMERIC = ["LIMIT_BAL", "AGE"] + BILL_AMT + PAY_AMT
CATEGORICAL = ["EDUCATION", "MARRIAGE"]
EXCLUDED = ["ID", "SEX", "target"]  # SEX 仅用于审计


def _safe_ratio(num: pd.Series, den: pd.Series, fill: float, lo: float, hi: float) -> pd.Series:
    """den ≤ 0（无欠款或溢缴）时记为 fill，再截断到 [lo, hi]。"""
    out = pd.Series(fill, index=num.index, dtype=float)
    pos = den > 0
    out[pos] = num[pos] / den[pos]
    return out.clip(lo, hi)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """返回带派生特征、已合并未定义编码的新 DataFrame（不修改输入）。"""
    X = df.copy()
    X["EDUCATION"] = X["EDUCATION"].replace(EDU_MAP)
    X["MARRIAGE"] = X["MARRIAGE"].replace(MAR_MAP)

    pay = X[PAY_STATUS]
    X["n_late"] = (pay > 0).sum(axis=1)
    X["max_late"] = pay.max(axis=1).clip(lower=0)
    X["late_recent"] = (X["PAY_0"] > 0).astype(int)

    X["pay_amt_mean"] = X[PAY_AMT].mean(axis=1)
    X["n_zero_pay"] = (X[PAY_AMT] == 0).sum(axis=1)
    # PAY_AMTk 是第 k 期的还款，对应上一期账单 BILL_AMT(k+1)
    X["pay_ratio1"] = _safe_ratio(X["PAY_AMT1"], X["BILL_AMT2"], 1.0, 0, 2)
    paid = X[PAY_AMT[:5]].sum(axis=1)
    owed = X[BILL_AMT[1:]].clip(lower=0).sum(axis=1)
    X["pay_ratio_avg"] = _safe_ratio(paid, owed, 1.0, 0, 2)

    lim = X["LIMIT_BAL"].astype(float)
    X["util1"] = (X["BILL_AMT1"] / lim).clip(-1, 2)
    X["util_mean"] = (X[BILL_AMT].mean(axis=1) / lim).clip(-1, 2)
    X["bill_le0"] = (X["BILL_AMT1"] <= 0).astype(int)
    X["bill_trend"] = ((X["BILL_AMT1"] - X["BILL_AMT6"]) / lim).clip(-2, 2)
    return X


def signed_log1p(x):
    """金额列的对称对数变换，供逻辑回归使用（无拟合参数）。"""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))
