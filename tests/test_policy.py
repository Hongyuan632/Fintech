"""三档策略的边界与计数测试（不依赖模型文件，不读任何数据划分）。"""

import numpy as np
import pytest

from src.policy import assign_tier, tier_metrics


def test_boundaries_belong_to_upper_tier():
    t1, t2 = 0.15, 0.40
    p = np.array([0.0, 0.1499999, 0.15, 0.3999999, 0.40, 1.0])
    assert assign_tier(p, t1, t2).tolist() == ["pass", "pass", "review", "review", "suspend", "suspend"]


@pytest.mark.parametrize("t1,t2", [(0.4, 0.15), (0.2, 0.2), (0.0, 0.5), (0.2, 1.0)])
def test_invalid_thresholds_rejected(t1, t2):
    with pytest.raises(ValueError):
        assign_tier([0.3], t1, t2)


@pytest.mark.parametrize("bad", [[np.nan], [-0.01], [1.01]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        assign_tier(bad, 0.15, 0.40)


def test_counts_sum_and_rates():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    m = tier_metrics(y, p, 0.15, 0.40)
    assert m["n_pass"] + m["n_review"] + m["n_suspend"] == 1000
    assert abs(m["pass_rate"] + m["review_rate"] + m["suspend_rate"] - 1) < 1e-12
    # 违约者要么被暂缓（召回），要么漏报；漏报 = 漏放 + 进入复核
    assert abs(m["high_recall"] + m["miss_rate"] - 1) < 1e-12
    assert abs(m["leak_rate"] + m["review_share_of_defaults"] - m["miss_rate"]) < 1e-12


def test_zero_denominator_is_none():
    m = tier_metrics(np.array([0, 0, 0]), np.array([0.5, 0.6, 0.7]), 0.15, 0.40)
    assert m["pass_default_rate"] is None and m["high_recall"] is None
    assert m["false_block_rate"] == 1.0
