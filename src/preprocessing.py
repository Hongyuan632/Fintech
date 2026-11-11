"""两种模型的预处理管道。所有带拟合参数的步骤（标准化、独热字典）都在 Pipeline 内，
交叉验证时只在训练折上拟合。"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_loader import BILL_AMT, PAY_AMT, PAY_STATUS
from src.feature_engineering import CATEGORICAL, ENGINEERED, add_features, signed_log1p

AMOUNT_COLS = ["LIMIT_BAL"] + BILL_AMT + PAY_AMT + ["pay_amt_mean"]


def clip_pay(x):
    """PAY_* 截断到 [-2, 3]：≥3 的样本少，合并为一档后再独热。"""
    return np.clip(np.asarray(x, dtype=float), -2, 3)


def feature_lists(use_eng: bool) -> tuple[list[str], list[str]]:
    """返回 (数值列, 还款状态列)。use_eng=False 时只用原始字段（不含 SEX）。"""
    numeric = ["LIMIT_BAL", "AGE"] + BILL_AMT + PAY_AMT
    if use_eng:
        numeric = numeric + ENGINEERED
    return numeric, PAY_STATUS


def build_preprocessor(kind: str, use_eng: bool) -> ColumnTransformer:
    numeric, pay = feature_lists(use_eng)
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "lr":
        amount = [c for c in numeric if c in AMOUNT_COLS]
        other = [c for c in numeric if c not in AMOUNT_COLS]
        log = FunctionTransformer(signed_log1p, feature_names_out="one-to-one")
        clip = FunctionTransformer(clip_pay, feature_names_out="one-to-one")
        return ColumnTransformer([
            ("amount", Pipeline([("log", log), ("scale", StandardScaler())]), amount),
            ("num", StandardScaler(), other),
            ("pay", Pipeline([("clip", clip), ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                       sparse_output=False))]), pay),
            ("cat", onehot, CATEGORICAL),
        ])
    if kind == "rf":  # 树模型不需要缩放，还款状态保留数值形式
        return ColumnTransformer([
            ("num", "passthrough", numeric + pay),
            ("cat", onehot, CATEGORICAL),
        ])
    raise ValueError(kind)


def make_pipeline(kind: str, use_eng: bool, clf) -> Pipeline:
    return Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(kind, use_eng)),
        ("clf", clf),
    ])
