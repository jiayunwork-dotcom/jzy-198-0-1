"""块度模型测试：参考值、单调关系、正比与单次反推、拒收。"""

from __future__ import annotations

import math

import pytest

from app.fragmentation import (
    DesignParams,
    implied_rock_factor,
    passing_fraction,
    predict,
    predict_uniformity,
    predict_x50,
    ValidationError,
)
from tests.conftest import TYPICAL_PARAMS


def make(**over) -> DesignParams:
    data = dict(TYPICAL_PARAMS)
    data.update(over)
    p = DesignParams(**data)
    p.validate()
    return p


# ---------------------------------------------------------- 参考值

def test_reference_passing_fraction():
    """X50=0.30 m、n=1.5 时，0.60 m 筛孔通过率约 85.9%。"""
    p = passing_fraction(0.30, 1.5, 0.60)
    assert p == pytest.approx(0.859, abs=5e-4)
    assert p == pytest.approx(0.8592143, rel=1e-7)


def test_passing_is_monotonic_in_sieve_and_midpoint():
    ds = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    vals = [passing_fraction(0.30, 1.5, d) for d in ds]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    # 中值块度处通过率恒为 50%（与 n 无关）
    for n in (0.8, 1.2, 1.5, 2.2):
        assert passing_fraction(0.4, n, 0.4) == pytest.approx(0.5, abs=1e-12)


def test_higher_n_narrows_distribution():
    """n 越大越均匀：在大于 X50 的筛孔通过率更高，小筛孔更低。"""
    coarse_n1 = passing_fraction(0.3, 1.0, 0.6)
    coarse_n2 = passing_fraction(0.3, 2.0, 0.6)
    fine_n1 = passing_fraction(0.3, 1.0, 0.1)
    fine_n2 = passing_fraction(0.3, 2.0, 0.1)
    assert coarse_n2 > coarse_n1
    assert fine_n2 < fine_n1


# ---------------------------------------------------------- 单调关系

def test_x50_decreases_with_powder_factor():
    """单耗越高，中值块度越小（通过增大药量、缩小孔网等多条途径验证）。"""
    base = make()
    # 加大药量
    assert predict_x50(make(explosive_mass_kg=500), 7.0) < predict_x50(base, 7.0)
    assert predict_x50(make(explosive_mass_kg=300), 7.0) > predict_x50(base, 7.0)
    # 缩小孔距/排距/台阶高（单耗增大）
    assert predict_x50(make(spacing_m=7.0), 7.0) < predict_x50(base, 7.0)
    assert predict_x50(make(burden_m=6.0), 7.0) < predict_x50(base, 7.0)
    assert predict_x50(make(bench_height_m=10.0), 7.0) < predict_x50(base, 7.0)
    # 任意单耗 q 都满足 X50 随 q 严格递减
    qs = [0.3, 0.45, 0.6, 0.8, 1.1, 1.5]
    xs = [predict_x50(make(explosive_mass_kg=q * 7 * 8 * 12), 7.0) for q in qs]
    assert all(a > b for a, b in zip(xs, xs[1:]))


def test_x50_proportional_to_rock_factor():
    """中值块度与岩石系数成正比（精确线性）。"""
    p = make()
    x7 = predict_x50(p, 7.0)
    x14 = predict_x50(p, 14.0)
    assert x14 / x7 == pytest.approx(2.0, abs=1e-15)
    x1 = predict_x50(p, 1.0)
    assert x7 / x1 == pytest.approx(7.0, abs=1e-12)


def test_single_event_back_calculation_84():
    """A=7 预测 0.40、实测 0.48，则单次反推 A=8.4。"""
    a_new = implied_rock_factor(make(), measured_x50_m=0.48,
                                predicted_x50_m=0.40, used_factor=7.0)
    assert a_new == pytest.approx(8.4, abs=1e-12)


def test_x50_decreases_with_explosive_strength():
    """炸药威力越高（RWS 越大），中值块度越小。"""
    weak = predict_x50(make(explosive_rws=70.0), 7.0)
    strong = predict_x50(make(explosive_rws=130.0), 7.0)
    assert strong < weak


def test_uniformity_index_basic():
    n = predict_uniformity(make())
    assert 0.05 < n < 5.0
    # 钻孔偏差越大，n 越低（越不均匀）；理想几何 n 更高
    n_good = predict_uniformity(make(hole_deviation_m=0.0))
    n_bad = predict_uniformity(make(hole_deviation_m=1.5))
    assert n_good > n_bad
    # 典型孔网下 n 应落在工程经验常见区间（约 0.8~2.0）
    assert 0.8 < n < 2.0


def test_predict_oversize_judgement():
    p = make(oversize_sieve_m=0.6, oversize_limit_pct=5.0)
    r = predict(p, 7.0)
    assert 0.0 < r.oversize_pct < 100.0
    assert r.oversize_ok == (r.oversize_pct <= 5.0)
    # 系数增大 -> X50 增大 -> 大块率升高
    r_big = predict(p, 14.0)
    assert r_big.oversize_pct > r.oversize_pct


# ---------------------------------------------------------- 拒收

@pytest.mark.parametrize("field,value", [
    ("hole_diameter_mm", 0),
    ("hole_diameter_mm", -3),
    ("burden_m", 0),
    ("spacing_m", -1),
    ("bench_height_m", 0),
    ("explosive_mass_kg", -10),
])
def test_reject_non_positive(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_reject_stemming_longer_than_hole():
    with pytest.raises(ValidationError):
        make(stemming_m=14.0, hole_depth_m=13.5)


def test_reject_non_positive_rws():
    with pytest.raises(ValidationError):
        make(explosive_rws=0)
    with pytest.raises(ValidationError):
        make(explosive_rws=-1)


def test_reject_non_positive_rock_factor():
    with pytest.raises(ValidationError):
        predict_x50(make(), 0)
    with pytest.raises(ValidationError):
        predict_x50(make(), -2)
