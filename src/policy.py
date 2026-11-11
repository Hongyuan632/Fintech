"""阶段 ③（上）：在验证集比较三档阈值方案，按预先声明的约束选定并锁定策略。

运行：.\\.venv\\Scripts\\python.exe -m src.policy
只读验证集。产出 models/policy.json 与 reports/policy_results.json。

三档规则（边界归属写死在 assign_tier 中）：
  p <  t1        低风险 → 模拟通过
  t1 ≤ p < t2    中风险 → 人工复核（最终结果不可观察）
  p ≥ t2         高风险 → 模拟暂缓授信
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import BILL_AMT, PAY_AMT, PAY_STATUS, ROOT, load_split

MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
POLICY_PATH = ROOT / "models" / "policy.json"
RESULTS_PATH = ROOT / "reports" / "policy_results.json"

TIERS = ["pass", "review", "suspend"]
CANDIDATES = [(0.10, 0.30), (0.15, 0.40), (0.20, 0.50)]  # README 给出的教学情景

# ---- 情景约束：在计算任何阈值指标之前写定（此前只看过阶段 ② 的预测分布图） ----
REVIEW_CAPACITY = 0.30        # 复核团队最多能复核 30% 的客户
MAX_PASS_DEFAULT_RATE = 0.10  # 直接通过的客户中，违约率不得超过 10%
SELECTION_RULE = (
    "在同时满足“复核率 ≤ 复核容量”和“通过组违约率 ≤ 上限”的候选中，选通过人数最多者。"
    "若没有候选同时满足，复核容量视为硬约束（人力无法临时增加）："
    "在满足容量的候选中选通过组违约率最低者，并记录约束冲突。"
)

INPUT_FIELDS = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS + BILL_AMT + PAY_AMT
AGE_BANDS = ["21–29", "30–39", "40–49", "50–59", "60+"]


def assign_tier(p, t1: float, t2: float) -> np.ndarray:
    """返回每个客户的档位。p 恰好等于 t1 归入复核，恰好等于 t2 归入暂缓。"""
    if not 0 < t1 < t2 < 1:
        raise ValueError(f"需要 0 < t1 < t2 < 1，收到 t1={t1}, t2={t2}")
    p = np.asarray(p, dtype=float)
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("概率必须在 [0, 1] 内且不能缺失")
    return np.where(p < t1, "pass", np.where(p < t2, "review", "suspend"))


def _ratio(num, den):
    """分母为零返回 None（报告中显示 N/A）。"""
    return None if den == 0 else float(num / den)


def tier_metrics(y, p, t1: float, t2: float) -> dict:
    """三档策略指标。“高风险”= 暂缓档（p ≥ t2）。"""
    y = np.asarray(y)
    tier = assign_tier(p, t1, t2)
    n, n_def, n_good = len(y), int(y.sum()), int((y == 0).sum())
    m = {"t1": t1, "t2": t2, "n": n, "defaults": n_def}
    for t in TIERS:
        k = tier == t
        m[f"n_{t}"] = int(k.sum())
        m[f"{t}_rate"] = _ratio(k.sum(), n)
        m[f"{t}_default_rate"] = _ratio(y[k].sum(), k.sum())
    susp, pas, rev = tier == "suspend", tier == "pass", tier == "review"
    m["high_recall"] = _ratio(y[susp].sum(), n_def)                 # 高风险召回率
    m["false_block_rate"] = _ratio((susp & (y == 0)).sum(), n_good)  # 误拦率
    m["miss_rate"] = _ratio(y[~susp].sum(), n_def)                  # 漏报率：违约者未进入高风险档
    m["leak_rate"] = _ratio(y[pas].sum(), n_def)                    # 漏放率：违约者被直接通过
    m["review_share_of_defaults"] = _ratio(y[rev].sum(), n_def)     # 复核队列中的违约者占比（结果不可观察）
    assert m["n_pass"] + m["n_review"] + m["n_suspend"] == n
    return m


def age_band(age) -> pd.Series:
    return pd.cut(pd.Series(age), [20, 29, 39, 49, 59, 200], labels=AGE_BANDS).astype(str)


def group_audit(df: pd.DataFrame, p, t1: float, t2: float) -> list[dict]:
    """按性别与年龄段分组审计。df 需含 SEX、AGE、target。"""
    p = np.asarray(p)
    groups = {"性别": df["SEX"].map({1: "男", 2: "女"}).to_numpy(),
              "年龄段": age_band(df["AGE"].to_numpy()).to_numpy()}
    order = {"性别": ["男", "女"], "年龄段": AGE_BANDS}
    rows = []
    for dim, key in groups.items():
        for g in order[dim]:
            k = key == g
            if not k.any():
                continue
            m = tier_metrics(df["target"].to_numpy()[k], p[k], t1, t2)
            rows.append({"dimension": dim, "group": g, "small_sample": bool(m["n"] < 300),
                         **{c: m[c] for c in ["n", "defaults", "pass_rate", "review_rate", "suspend_rate",
                                              "pass_default_rate", "false_block_rate", "miss_rate", "leak_rate"]},
                         "mean_pd": float(p[k].mean()), "default_rate": _ratio(m["defaults"], m["n"])})
    return rows


def choose(rows: list[dict]) -> tuple[dict, dict]:
    """按 SELECTION_RULE 选方案。返回 (选中的行, 选择说明)。"""
    for r in rows:
        r["ok_capacity"] = r["review_rate"] <= REVIEW_CAPACITY
        r["ok_default"] = r["pass_default_rate"] <= MAX_PASS_DEFAULT_RATE
    feasible = [r for r in rows if r["ok_capacity"] and r["ok_default"]]
    if feasible:
        best = max(feasible, key=lambda r: r["n_pass"])
        return best, {"conflict": False, "n_feasible": len(feasible)}
    cap_ok = [r for r in rows if r["ok_capacity"]]
    if not cap_ok:
        raise RuntimeError("没有候选满足复核容量，需重新设定候选阈值或容量。")
    best = min(cap_ok, key=lambda r: r["pass_default_rate"])
    return best, {"conflict": True, "n_feasible": 0,
                  "note": "没有候选同时满足两项约束，按复核容量优先的回退规则选择。"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        raise FileNotFoundError("缺少 models/policy.json，请先运行 python -m src.policy")
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def main() -> None:
    va = load_split("valid")
    y = va["target"].to_numpy()
    model = joblib.load(MODEL_PATH)
    p = model.predict_proba(va.drop(columns="target"))[:, 1]
    saved = pd.read_csv(ROOT / "data" / "processed" / "valid_predictions.csv")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    assert np.allclose(p, saved[f"p_{meta['chosen']}"]), "模型文件与阶段 ② 保存的验证集预测不一致"

    rows = [tier_metrics(y, p, t1, t2) for t1, t2 in CANDIDATES]
    best, how = choose(rows)

    # 连续扫描，用于画权衡曲线（只作展示，不参与选择）
    sweep1 = [{"t1": float(t), "pass_rate": float((p < t).mean()),
               "pass_default_rate": _ratio(y[p < t].sum(), (p < t).sum()),
               "leak_rate": _ratio(y[p < t].sum(), y.sum())} for t in np.arange(0.05, 0.3501, 0.01)]
    sweep2 = [{"t2": float(t), "suspend_rate": float((p >= t).mean()),
               "precision": _ratio(y[p >= t].sum(), (p >= t).sum()),
               "high_recall": _ratio(y[p >= t].sum(), y.sum()),
               "false_block_rate": _ratio(((p >= t) & (y == 0)).sum(), (y == 0).sum())}
              for t in np.arange(0.25, 0.7001, 0.01)]

    policy = {
        "version": "v1",
        "locked_at_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
        "model_file": MODEL_PATH.name, "model_sha256": sha256(MODEL_PATH), "model": meta["chosen"],
        "t1": best["t1"], "t2": best["t2"],
        "tiers": {"pass": "p < t1：模拟通过", "review": "t1 ≤ p < t2：人工复核", "suspend": "p ≥ t2：模拟暂缓授信"},
        "positive_rule": "阈值类指标以 PD ≥ t2 为预测正类",
        "constraints": {"review_capacity": REVIEW_CAPACITY, "max_pass_default_rate": MAX_PASS_DEFAULT_RATE},
        "selection_rule": SELECTION_RULE, "selection": how, "selected_on": "valid split (4,500)",
        "input_fields": INPUT_FIELDS, "excluded_fields": ["ID", "SEX"],
    }
    POLICY_PATH.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps({
        "constraints": policy["constraints"], "selection_rule": SELECTION_RULE, "selection": how,
        "chosen": [best["t1"], best["t2"]], "candidates": rows, "sweep_t1": sweep1, "sweep_t2": sweep2,
        "valid_audit": group_audit(va, p, best["t1"], best["t2"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    pct = lambda v: "N/A" if v is None else f"{v:.1%}"
    print(f"约束：复核率 ≤ {REVIEW_CAPACITY:.0%}，通过组违约率 ≤ {MAX_PASS_DEFAULT_RATE:.0%}")
    for r in rows:
        print(f"  ({r['t1']:.2f}, {r['t2']:.2f}) 通过 {pct(r['pass_rate'])} 复核 {pct(r['review_rate'])} "
              f"暂缓 {pct(r['suspend_rate'])} | 通过组违约率 {pct(r['pass_default_rate'])} "
              f"高风险召回 {pct(r['high_recall'])} 误拦 {pct(r['false_block_rate'])} 漏放 {pct(r['leak_rate'])} "
              f"| 容量{'✓' if r['ok_capacity'] else '✗'} 违约率{'✓' if r['ok_default'] else '✗'}")
    print(f"选定 ({best['t1']:.2f}, {best['t2']:.2f})；约束冲突：{how['conflict']}")


if __name__ == "__main__":
    main()
