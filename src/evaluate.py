"""阶段 ③（下）：按锁定的模型与策略，在测试集上做一次性评估与分组审计。

运行：.\\.venv\\Scripts\\python.exe -m src.evaluate
前提：models/policy.json 已存在，且其中记录的模型哈希与当前模型文件一致。
每次运行都会在 reports/test_eval_log.json 追加一条记录，用来核对测试集被读取了几次。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from src.data_loader import PROC_DIR, ROOT, load_split
from src.metrics import calibration_table, summarize
from src.policy import CANDIDATES, MODEL_PATH, group_audit, load_policy, sha256, tier_metrics

RESULTS_PATH = ROOT / "reports" / "test_results.json"
LOG_PATH = ROOT / "reports" / "test_eval_log.json"


def bootstrap_ci(y, p, n_boot: int = 2000, seed: int = 42) -> dict:
    """测试集指标的 95% 百分位 bootstrap 区间。"""
    rng = np.random.default_rng(seed)
    n, out = len(y), {"roc_auc": [], "ap": [], "brier": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        out["roc_auc"].append(roc_auc_score(y[i], p[i]))
        out["ap"].append(average_precision_score(y[i], p[i]))
        out["brier"].append(brier_score_loss(y[i], p[i]))
    return {k: [float(v) for v in np.percentile(vals, [2.5, 97.5])] for k, vals in out.items()}


def main() -> None:
    policy = load_policy()
    if sha256(MODEL_PATH) != policy["model_sha256"]:
        raise RuntimeError("模型文件与 policy.json 记录的哈希不一致：模型在锁定后被改动，请重新运行 src.policy。")
    t1, t2 = policy["t1"], policy["t2"]

    log = json.loads(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.exists() else []
    if log:
        print(f"注意：测试集此前已评估 {len(log)} 次。重复评估后不得再据此调整模型或阈值。")

    te = load_split("test")
    y = te["target"].to_numpy()
    model = joblib.load(MODEL_PATH)
    p = model.predict_proba(te.drop(columns="target"))[:, 1]
    base_rate = float(load_split("train")["target"].mean())  # 基准：训练集违约率

    pred = (p >= t2).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    results = {
        "policy_version": policy["version"], "model": policy["model"], "t1": t1, "t2": t2,
        "n_test": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
        "metrics": summarize(y, p),
        "ci95": bootstrap_ci(y, p),
        "baseline": {"p": base_rate, "brier": float(brier_score_loss(y, np.full(len(y), base_rate))),
                     "ap": float(y.mean()), "roc_auc": 0.5},
        "calibration": calibration_table(y, p).round(5).to_dict("records"),
        "threshold_at_t2": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "accuracy": float(accuracy_score(y, pred)),
            "accuracy_all_negative": float(1 - y.mean()),
        },
        "policy": tier_metrics(y, p, t1, t2),
        "other_candidates": [tier_metrics(y, p, a, b) for a, b in CANDIDATES if (a, b) != (t1, t2)],
        "audit": group_audit(te, p, t1, t2),
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    te[["ID", "target", "SEX", "AGE"]].assign(p=p).to_csv(PROC_DIR / "test_predictions.csv", index=False)
    log.append({"time_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
                "policy_version": policy["version"], "model_sha256": policy["model_sha256"], "t1": t1, "t2": t2})
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    m, ci, pol, th = results["metrics"], results["ci95"], results["policy"], results["threshold_at_t2"]
    print(f"测试集 n={len(y)}，违约 {int(y.sum())}（{y.mean():.2%}）")
    print(f"ROC-AUC {m['roc_auc']:.4f} {ci['roc_auc']}  KS {m['ks']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
          f"AP {m['ap']:.4f} {ci['ap']}")
    print(f"Brier {m['brier']:.4f} {ci['brier']}（基准 {results['baseline']['brier']:.4f}）  ECE {m['ece']:.4f}")
    print(f"PD ≥ {t2}: TP {tp} FP {fp} FN {fn} TN {tn}  P {th['precision']:.3f} R {th['recall']:.3f} "
          f"F1 {th['f1']:.3f} Acc {th['accuracy']:.3f}")
    print(f"通过 {pol['pass_rate']:.1%} 复核 {pol['review_rate']:.1%} 暂缓 {pol['suspend_rate']:.1%}；"
          f"通过组违约率 {pol['pass_default_rate']:.1%}，漏放率 {pol['leak_rate']:.1%}，误拦率 {pol['false_block_rate']:.1%}")


if __name__ == "__main__":
    main()
