"""岩石系数标定算法测试：增量/全量一致、衰减、反推、撤回重放。"""

from __future__ import annotations

import math
import random

import pytest

from app.calibration import (
    DEFAULT_BASELINE_WEIGHT,
    Fold,
    fold_from_observations,
    fold_to_factor,
    full_refit,
    incremental_update,
)


def incremental_factor(values, rho, baseline=10.0, bw=1.0):
    fold = Fold()
    for a in values:
        incremental_update(fold, a, rho, baseline, bw)
    return fold_to_factor(fold, baseline, bw), fold


def test_no_decay_incremental_equals_full_refit_tight():
    """rho=1 时增量与全量重拟合一致，相对差 < 1e-9（题目要求）。"""
    random.seed(42)
    for trial in range(20):
        k = random.randint(1, 60)
        values = [random.uniform(4.0, 16.0) for _ in range(k)]
        baseline = random.uniform(5.0, 12.0)
        bw = random.choice([0.0, 0.5, 1.0, 3.0])
        inc, _ = incremental_factor(values, 1.0, baseline, bw)
        full = full_refit(values, baseline, bw, forgetting=1.0).factor
        rel = abs(inc - full) / full
        assert rel < 1e-9, (trial, k, inc, full, rel)


def test_no_decay_matches_geometric_mean_formula():
    """rho=1 时结果就是基线+观测的加权几何均值。"""
    values = [6.0, 8.4, 7.2, 9.1]
    baseline, bw = 10.0, 1.0
    inc, fold = incremental_factor(values, 1.0, baseline, bw)
    log_mean = (bw * math.log(baseline) + sum(math.log(a) for a in values)) / (bw + len(values))
    expected = math.exp(log_mean)
    assert inc == pytest.approx(expected, rel=1e-12)
    assert fold.w == pytest.approx(len(values), abs=1e-12)


def test_fold_from_observations_equivalence():
    random.seed(7)
    values = [random.uniform(3, 20) for _ in range(30)]
    f1 = fold_from_observations([math.log(a) for a in values], 1.0)
    factor_a = fold_to_factor(f1, 10.0, 1.0)
    factor_b, _ = incremental_factor(values, 1.0)
    assert factor_a == pytest.approx(factor_b, rel=1e-12)


def test_forgetting_weights_recent_more():
    """rho<1：同样的序列，最新一条的影响更大；遗忘使旧观测权重下降。"""
    baseline = 10.0
    low_then_high, _ = incremental_factor([5.0, 5.0, 5.0, 15.0], 0.9, baseline)
    high_then_low, _ = incremental_factor([15.0, 5.0, 5.0, 5.0], 0.9, baseline)
    # 最新是 15 的序列结果应明显高于最新是 5 的
    assert low_then_high > high_then_low


def test_forgetting_old_high_value_fades():
    """旧的高观测随新观测入库，影响逐版衰减。"""
    fold = Fold()
    factors = []
    for a in [20.0, 8.0, 8.0, 8.0, 8.0]:
        incremental_update(fold, a, 0.8, 10.0, 1.0)
        factors.append(fold_to_factor(fold, 10.0, 1.0))
    # 从 20 被纳入后，系数随时间回落到 8 附近
    assert factors[0] > factors[1] > factors[-1]
    assert abs(factors[-1] - 8.0) < abs(factors[0] - 8.0)


def test_forgetting_matches_closed_form_weights():
    """折叠权重定义与全量重拟合的 rho**(k-1-i) 完全一致。"""
    rho = 0.83
    values = [6.0, 11.0, 7.5, 9.3, 12.0, 5.4]
    inc, fold = incremental_factor(values, rho, 9.0, 1.0)
    full = full_refit(values, 9.0, 1.0, rho).factor
    # 衰减下二者也是同一估计量（相同顺序求和顺序也一致）
    assert inc == pytest.approx(full, rel=1e-12)
    expected_w = sum(rho ** (len(values) - 1 - i) for i in range(len(values)))
    assert fold.w == pytest.approx(expected_w, rel=1e-12)


def test_empty_is_baseline():
    full = full_refit([], 7.5).factor
    assert full == pytest.approx(7.5, rel=1e-12)
    fold = Fold()
    assert fold_to_factor(
        fold, 7.5, DEFAULT_BASELINE_WEIGHT
    ) == pytest.approx(7.5, rel=1e-12)


def test_baseline_weight_shrinks_with_data():
    """观测越多，基线先验影响越小。"""
    values = [12.0] * 20
    weak_prior = full_refit(values, 8.0, baseline_weight=0.1).factor
    strong_prior = full_refit(values, 8.0, baseline_weight=5.0).factor
    assert abs(weak_prior - 12.0) < abs(strong_prior - 12.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        incremental_update(Fold(), 0, 0.9, 10.0)
    with pytest.raises(ValueError):
        incremental_update(Fold(), -3, 0.9, 10.0)
    with pytest.raises(ValueError):
        Fold().add(math.log(5), 1.2)
    with pytest.raises(ValueError):
        full_refit([5], -1.0)
