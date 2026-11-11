"""模型保存 / 加载一致性与输入约束测试。只用验证集样本，不读测试集。"""

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data_loader import ROOT, load_split
from src.feature_engineering import ENGINEERED, add_features

MODEL = ROOT / "models" / "credit_risk_model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="先运行 python -m src.train")


@pytest.fixture(scope="module")
def sample():
    return load_split("valid").drop(columns="target").head(500)


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL)


def test_features_rowwise_and_complete(sample):
    before = sample.copy()
    X = add_features(sample)
    assert sample.equals(before), "add_features 不应修改输入"
    assert not X[ENGINEERED].isna().any().any()
    # 逐行计算：单独算一行与整批算结果相同
    one = add_features(sample.iloc[[7]])
    assert np.allclose(one[ENGINEERED].to_numpy(float), X.iloc[[7]][ENGINEERED].to_numpy(float))


def test_probabilities_valid(model, sample):
    p = model.predict_proba(sample)[:, 1]
    assert p.shape == (len(sample),) and np.all((p >= 0) & (p <= 1))


def test_save_reload_consistent(model, sample, tmp_path: Path):
    p1 = model.predict_proba(sample)[:, 1]
    joblib.dump(model, tmp_path / "m.joblib")
    p2 = joblib.load(tmp_path / "m.joblib").predict_proba(sample)[:, 1]
    assert np.allclose(p1, p2, atol=1e-12)


def test_sex_and_id_not_used(model, sample):
    flipped = sample.assign(SEX=3 - sample["SEX"], ID=sample["ID"] + 10**6)
    assert np.allclose(model.predict_proba(sample)[:, 1], model.predict_proba(flipped)[:, 1])
