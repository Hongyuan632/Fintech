"""阶段 ②：训练集内交叉验证调参与特征消融 → 折外预测拟合校准器 → 验证集比较候选模型。

运行：.\\.venv\\Scripts\\python.exe -m src.train
测试集在本阶段不读取。

产出：
  models/credit_risk_model.joblib       按预先声明的规则选出的最终管道（含预处理与校准）
  models/model_meta.json                选择结果、超参数、数据版本与依赖版本
  data/processed/valid_predictions.csv  验证集上各候选的预测概率（阶段 ③ 选阈值用）
  reports/model_results.json            交叉验证、验证集指标、校准表、bootstrap 结果
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_loader import CHECKS_JSON, PROC_DIR, ROOT, SEED, load_split
from src.metrics import bootstrap_auc_diff, calibration_table, summarize
from src.preprocessing import make_pipeline

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

GRIDS = {
    "lr": (LogisticRegression(max_iter=5000),
           {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "rf": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
           {"clf__max_depth": [8, 14, None],
            "clf__min_samples_leaf": [10, 40],
            "clf__max_features": ["sqrt", 0.33]}),
}
NAMES = {"lr": "逻辑回归", "rf": "随机森林"}
CALIB = ["none", "sigmoid", "isotonic"]

# 预先声明的选择规则（在看验证集结果之前写定）
SELECTION_RULE = (
    "1) 模型族：比较两族未校准版本的验证集 ROC-AUC，用配对 bootstrap 求差值的 95% 区间；"
    "区间不含 0 时选 AUC 高者，否则选逻辑回归（更易解释）。"
    "2) 校准方式：在所选模型族内，选验证集 Brier Score 最低的一种（none / sigmoid / isotonic）。"
)
# 2026-11-09 查看验证集结果后补充：保序回归仅以 0.00003 的 Brier 优势胜出，
# 但带来离散台阶与 PD = 0，因此增加平局规则。论文中须注明此规则为事后补充。
TIE_RULE = (
    "3) 平局规则（事后补充）：Brier 与最低值相差不足 0.001 的方案视为相同，"
    "按 未校准 → Platt → 保序回归 的顺序选最简单者。"
)
BRIER_TIE = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune(kind: str, use_eng: bool, X, y) -> GridSearchCV:
    clf, grid = GRIDS[kind]
    gs = GridSearchCV(make_pipeline(kind, use_eng, clf), grid, cv=CV, n_jobs=-1,
                      scoring={"roc_auc": "roc_auc", "brier": "neg_brier_score"},
                      refit="roc_auc")
    return gs.fit(X, y)


def main() -> None:
    tr, va = load_split("train"), load_split("valid")
    Xtr, ytr = tr.drop(columns="target"), tr["target"].to_numpy()
    Xva, yva = va.drop(columns="target"), va["target"].to_numpy()
    log(f"训练集 {len(tr)}，验证集 {len(va)}（测试集未读取）")

    # ---- A. 训练集 5 折 CV：调参 + 特征消融（原始字段 vs 原始 + 派生特征） ----
    cv_rows, best = [], {}
    for kind in ("lr", "rf"):
        for use_eng in (False, True):
            gs = tune(kind, use_eng, Xtr, ytr)
            i = gs.best_index_
            row = {
                "model": kind, "features": "raw+eng" if use_eng else "raw",
                "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                "cv_auc": float(gs.cv_results_["mean_test_roc_auc"][i]),
                "cv_auc_sd": float(gs.cv_results_["std_test_roc_auc"][i]),
                "cv_brier": float(-gs.cv_results_["mean_test_brier"][i]),
            }
            cv_rows.append(row)
            log(f"{kind} {row['features']:7s} CV AUC {row['cv_auc']:.4f}±{row['cv_auc_sd']:.4f} "
                f"Brier {row['cv_brier']:.4f} {row['best_params']}")
            if kind not in best or row["cv_auc"] > best[kind]["row"]["cv_auc"]:
                best[kind] = {"row": row, "pipe": gs.best_estimator_}

    # ---- B. 校准：用训练集 5 折折外预测拟合校准器（ensemble=False），基模型在全训练集重拟合 ----
    candidates = {"baseline": DummyClassifier(strategy="prior").fit(Xtr, ytr)}
    for kind in ("lr", "rf"):
        base = best[kind]["pipe"]  # GridSearchCV 已在全训练集重拟合
        candidates[f"{kind}_none"] = base
        for method in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(base, method=method, cv=CV, ensemble=False)
            candidates[f"{kind}_{method}"] = cal.fit(Xtr, ytr)
            log(f"{kind} {method} 校准完成")

    # ---- C. 验证集比较 ----
    preds = {name: m.predict_proba(Xva)[:, 1] for name, m in candidates.items()}
    valid_metrics = {name: summarize(yva, p) for name, p in preds.items()}
    calib = {name: calibration_table(yva, p).round(5).to_dict("records")
             for name, p in preds.items() if name != "baseline"}
    boot = bootstrap_auc_diff(yva, preds["rf_none"], preds["lr_none"])
    for name, m in valid_metrics.items():
        log(f"{name:13s} AUC {m['roc_auc']:.4f} KS {m['ks']:.4f} PR-AUC {m['pr_auc']:.4f} "
            f"AP {m['ap']:.4f} Brier {m['brier']:.4f} ECE {m['ece']:.4f}")
    log(f"AUC(RF) − AUC(LR) = {boot['diff']:.4f}，95% CI {boot['ci95']}")

    # ---- D. 按预先声明的规则选择 ----
    lo, hi = boot["ci95"]
    family = ("rf" if boot["diff"] > 0 else "lr") if (lo > 0 or hi < 0) else "lr"
    briers = {c: valid_metrics[f"{family}_{c}"]["brier"] for c in CALIB}
    method_by_rule2 = min(briers, key=briers.get)
    method = next(c for c in CALIB if briers[c] - briers[method_by_rule2] < BRIER_TIE)  # CALIB 按简单程度排序
    chosen = f"{family}_{method}"
    log(f"规则 2 选 {method_by_rule2}；加平局规则后选 {method}")
    log(f"选定：{chosen}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(candidates[chosen], MODELS_DIR / "credit_risk_model.joblib")
    out = va[["ID", "target", "SEX", "AGE"]].copy()
    for name, p in preds.items():
        out[f"p_{name}"] = p
    out.to_csv(PROC_DIR / "valid_predictions.csv", index=False)

    # 解释信息：LR 系数 / RF 不纯度重要性（均基于全训练集拟合的未校准管道）
    explain = {}
    for kind in ("lr", "rf"):
        pipe = best[kind]["pipe"]
        names = pipe["prep"].get_feature_names_out()
        vals = pipe["clf"].coef_[0] if kind == "lr" else pipe["clf"].feature_importances_
        explain[kind] = dict(sorted(zip(names, map(float, vals)), key=lambda kv: -abs(kv[1])))

    checks = json.loads(Path(CHECKS_JSON).read_text(encoding="utf-8"))
    meta = {
        "chosen": chosen, "family": NAMES[family], "calibration": method,
        "features": best[family]["row"]["features"], "best_params": best[family]["row"]["best_params"],
        "selection_rule": SELECTION_RULE + TIE_RULE,
        "rule2_choice": f"{family}_{method_by_rule2}", "trained_on": "train split (21,000)", "seed": SEED,
        "raw_sha256": checks["raw_sha256"],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "excluded_inputs": ["ID", "SEX"],
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {"cv": cv_rows, "valid": valid_metrics, "calibration": calib, "bootstrap_rf_minus_lr": boot,
               "chosen": chosen, "rule2_choice": f"{family}_{method_by_rule2}",
               "selection_rule": SELECTION_RULE + TIE_RULE, "explain": explain,
               "n_valid": int(len(va)), "valid_default_rate": float(yva.mean())}
    (REPORTS_DIR / "model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    log("已保存模型、验证集预测与结果")


if __name__ == "__main__":
    main()
