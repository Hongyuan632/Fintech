"""获取 UCI 信用卡违约数据、校验字段并生成固定的训练 / 验证 / 测试划分。

运行：.\\.venv\\Scripts\\python.exe -m src.data_loader

产出：
  data/raw/credit_default.csv        业务字段名的原始数据（含 ID 与 target）
  data/raw/SOURCE.md                 来源、获取时间、许可与 SHA-256
  data/processed/split.csv           ID → train / valid / test
  data/processed/data_checks.json    字段校验与划分检查结果
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_CSV = RAW_DIR / "credit_default.csv"
SPLIT_CSV = PROC_DIR / "split.csv"
CHECKS_JSON = PROC_DIR / "data_checks.json"

UCI_ID = 350
SEED = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # 9 月 → 4 月
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]                     # 9 月 → 4 月
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]                       # 9 月 → 4 月
COLUMNS = (["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
           + PAY_STATUS + BILL_AMT + PAY_AMT + ["target"])

# 原说明定义的类别编码；其他值保留原值并单独统计
DOC_CODES = {
    "SEX": {1: "男", 2: "女"},
    "EDUCATION": {1: "研究生", 2: "大学", 3: "高中", 4: "其他"},
    "MARRIAGE": {1: "已婚", 2: "单身", 3: "其他"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw() -> pd.DataFrame:
    """从 UCI 接口下载，按变量说明把 X1–X23 / Y 映射为业务字段名。"""
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=UCI_ID)
    var = d.variables.set_index("name")["description"]
    feats = d.data.features.rename(columns=var.to_dict())
    target = d.data.targets.rename(columns={"Y": "target"})
    df = pd.concat([d.data.ids, feats, target], axis=1)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"字段映射与预期不一致：{list(df.columns)}")
    return df, d.metadata.get("data_url")


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """优先读本地副本；首次运行或 refresh=True 时从 UCI 下载并写来源记录。"""
    if RAW_CSV.exists() and not refresh:
        return pd.read_csv(RAW_CSV)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df, url = fetch_raw()
    df.to_csv(RAW_CSV, index=False)
    (RAW_DIR / "SOURCE.md").write_text(
        "# 原始数据来源记录\n\n"
        f"- 数据集：Default of Credit Card Clients（UCI 编号 {UCI_ID}）\n"
        f"- 获取方式：`ucimlrepo.fetch_ucirepo(id={UCI_ID})`，接口文件 {url}\n"
        f"- 获取时间（UTC）：{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}\n"
        f"- 本地文件：`credit_default.csv`，{len(df)} 行 × {df.shape[1]} 列\n"
        f"- SHA-256：`{_sha256(RAW_CSV)}`\n"
        "- 改动：仅将接口字段名 X1–X23、Y 按 UCI 变量说明改为业务字段名，Y 改名为 target；数值未改动。\n"
        "- 许可：CC BY 4.0。引用 Yeh, I. (2009). Default of Credit Card Clients [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H\n",
        encoding="utf-8",
    )
    return df


def validate(df: pd.DataFrame) -> dict:
    """字段校验：形状、ID、标签、缺失、编码、取值范围。只做检查，不改数据。"""
    c: dict = {}
    c["shape"] = list(df.shape)
    c["n_features_candidate"] = df.shape[1] - 2  # 去掉 ID 与 target
    c["id_unique"] = bool(df["ID"].is_unique)
    c["id_range"] = [int(df["ID"].min()), int(df["ID"].max())]
    c["target_values"] = {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()}
    c["default_rate"] = round(float(df["target"].mean()), 4)
    c["null_cells"] = int(df.isna().sum().sum())
    c["duplicate_rows_excl_id"] = int(df.drop(columns="ID").duplicated().sum())

    cats = {}
    for col, doc in DOC_CODES.items():
        vc = df[col].value_counts().sort_index()
        cats[col] = {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "undocumented": {str(k): int(v) for k, v in vc.items() if k not in doc},
        }
    c["categorical"] = cats

    c["AGE_range"] = [int(df["AGE"].min()), int(df["AGE"].max())]
    c["LIMIT_BAL_range"] = [int(df["LIMIT_BAL"].min()), int(df["LIMIT_BAL"].max())]

    pay = {}
    for col in PAY_STATUS:
        vc = df[col].value_counts().sort_index()
        pay[col] = {str(k): int(v) for k, v in vc.items()}
    c["pay_status_counts"] = pay
    c["pay_status_undocumented_share"] = {  # -2 与 0 在原说明中未解释
        col: round(float(df[col].isin([-2, 0]).mean()), 4) for col in PAY_STATUS
    }

    c["negative_bill_rows"] = {col: int((df[col] < 0).sum()) for col in BILL_AMT}
    c["negative_pay_amt_rows"] = {col: int((df[col] < 0).sum()) for col in PAY_AMT}
    return c


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """按 target 分层，种子 42，70 / 15 / 15。返回 ID → split。"""
    ids, y = df["ID"].to_numpy(), df["target"].to_numpy()
    id_rest, id_test, y_rest, _ = train_test_split(
        ids, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    id_train, id_valid = train_test_split(
        id_rest, test_size=VALID_SIZE / (1 - TEST_SIZE), stratify=y_rest, random_state=SEED)
    split = pd.concat([
        pd.DataFrame({"ID": id_train, "split": "train"}),
        pd.DataFrame({"ID": id_valid, "split": "valid"}),
        pd.DataFrame({"ID": id_test, "split": "test"}),
    ]).sort_values("ID").reset_index(drop=True)
    return split


def check_split(df: pd.DataFrame, split: pd.DataFrame) -> dict:
    sets = {s: set(split.loc[split["split"] == s, "ID"]) for s in ("train", "valid", "test")}
    m = df[["ID", "target"]].merge(split, on="ID", validate="one_to_one")
    by = m.groupby("split")["target"].agg(["size", "sum", "mean"])
    return {
        "seed": SEED,
        "ratio": [1 - TEST_SIZE - VALID_SIZE, VALID_SIZE, TEST_SIZE],
        "id_overlap": {
            "train&valid": len(sets["train"] & sets["valid"]),
            "train&test": len(sets["train"] & sets["test"]),
            "valid&test": len(sets["valid"] & sets["test"]),
        },
        "covers_all_ids": len(split) == len(df) and set(split["ID"]) == set(df["ID"]),
        "by_split": {
            s: {"n": int(r["size"]), "defaults": int(r["sum"]), "default_rate": round(float(r["mean"]), 4)}
            for s, r in by.loc[["train", "valid", "test"]].iterrows()
        },
    }


def load_split(name: str) -> pd.DataFrame:
    """读取某个划分的数据（'train' / 'valid' / 'test'）。测试集在模型与阈值锁定前不要读。"""
    if name not in {"train", "valid", "test"}:
        raise ValueError(name)
    df = pd.read_csv(RAW_CSV)
    split = pd.read_csv(SPLIT_CSV)
    ids = split.loc[split["split"] == name, "ID"]
    return df[df["ID"].isin(ids)].reset_index(drop=True)


def main(refresh: bool = False) -> None:
    df = load_raw(refresh=refresh)
    checks = validate(df)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if SPLIT_CSV.exists():
        split = pd.read_csv(SPLIT_CSV)
        if not split.equals(make_split(df)):  # 固定种子下应完全复现
            raise RuntimeError("已存在的 split.csv 与按种子 42 重算的结果不一致，请检查数据版本。")
    else:
        split = make_split(df)
        split.to_csv(SPLIT_CSV, index=False)
    checks["split"] = check_split(df, split)
    checks["raw_sha256"] = _sha256(RAW_CSV)
    CHECKS_JSON.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    s = checks["split"]
    print(f"数据 {checks['shape']}，ID 唯一：{checks['id_unique']}，缺失单元格：{checks['null_cells']}，"
          f"违约率：{checks['default_rate']:.2%}")
    print("未定义编码：", {k: v["undocumented"] for k, v in checks["categorical"].items()})
    for name, r in s["by_split"].items():
        print(f"  {name:5s} n={r['n']:5d}  违约 {r['defaults']:4d}  违约率 {r['default_rate']:.2%}")
    print("ID 交集：", s["id_overlap"], "覆盖全部 ID：", s["covers_all_ids"])
    assert checks["id_unique"] and not any(s["id_overlap"].values()) and s["covers_all_ids"]


if __name__ == "__main__":
    import sys
    main(refresh="--refresh" in sys.argv)
