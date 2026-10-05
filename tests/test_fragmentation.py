"""块度模型：参考值、单调关系、岩石系数正比与单次反推。"""

import pytest

from app import fragmentation as frag


def test_reference_passing_85_9():
    """参考值：X50=0.30m、n=1.5 时 0.60m 通过率约 85.9%。"""
    p = frag.passing_fraction(0.60, 0.30, 1.5)
    assert p == pytest.approx(85.9, abs=0.05)


def test_median_is_50_percent():
    for n in (0.8, 1.5, 2.2):
        assert frag.passing_fraction(0.30, 0.30, n) == pytest.approx(50.0, abs=1e-9)


def test_passing_monotone_in_sieve():
    sizes = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    vals = [frag.passing_fraction(s, 0.30, 1.5) for s in sizes]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_higher_powder_factor_smaller_x50_via_charge():
    """单耗升高（增大装药量）=> 中值块度变小。"""
    base = dict(burden=6.0, spacing=7.5, bench_height=12.0, explosive_rws=1.0)
    xs = [frag.median_size(7.0, charge_kg=q, **base) for q in (280, 350, 430)]
    assert xs[0] > xs[1] > xs[2]


def test_higher_powder_factor_smaller_x50_via_pattern():
    """单耗升高（缩小排距）=> 中值块度变小。"""
    xs = [
        frag.median_size(7.0, burden=b, spacing=7.5, bench_height=12.0,
                         charge_kg=350.0, explosive_rws=1.0)
        for b in (7.0, 6.0, 5.0)
    ]
    assert xs[0] > xs[1] > xs[2]


def test_x50_proportional_to_factor():
    kw = dict(burden=6.0, spacing=7.5, bench_height=12.0, charge_kg=350.0,
              explosive_rws=1.0)
    x7 = frag.median_size(7.0, **kw)
    x14 = frag.median_size(14.0, **kw)
    assert x14 / x7 == pytest.approx(2.0, abs=1e-12)


def test_single_back_calculation_8_4():
    """题给：A=7 预测 0.40m，实测 0.48m => 单次反推 8.4。"""
    a_new = frag.back_calculate_factor(
        measured_x50=0.48, used_factor=7.0, predicted_x50=0.40
    )
    assert a_new == pytest.approx(8.4, abs=1e-12)


def test_rws_stronger_explosive_finer():
    base = dict(burden=6.0, spacing=7.5, bench_height=12.0, charge_kg=350.0)
    x_anfo = frag.median_size(7.0, explosive_rws=1.0, **base)
    x_heavy = frag.median_size(7.0, explosive_rws=1.2, **base)
    assert x_heavy < x_anfo


def test_uniformity_positive_and_reasonable():
    n = frag.uniformity_index(
        burden=6.0, diameter_mm=250.0, hole_depth=13.5, stemming=4.0,
        bench_height=12.0,
    )
    assert 0.5 < n < 2.5


def test_uniformity_rejects_stemming_too_long():
    with pytest.raises(ValueError):
        frag.uniformity_index(
            burden=6.0, diameter_mm=250.0, hole_depth=10.0, stemming=10.0,
            bench_height=12.0,
        )


def test_oversize_is_complement():
    over = frag.oversize_fraction(0.8, 0.30, 1.5)
    p = frag.passing_fraction(0.8, 0.30, 1.5)
    assert over + p == pytest.approx(100.0, abs=1e-9)
