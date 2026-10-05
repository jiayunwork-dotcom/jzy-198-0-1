"""筛分拟合：已知参数合成曲线还原、开区间处理、坏曲线标记。"""

import numpy as np
import pytest

from app.fitting import fit_curve, quality_weight
from app.fragmentation import passing_fraction
from tests.conftest import OPEN_SIEVE_SIZES, SIEVE_SIZES


def test_recover_known_params():
    """对已知 X50、n 合成的曲线，拟合应还原到很高精度。"""
    x50_true, n_true = 0.32, 1.37
    sizes = np.array(SIEVE_SIZES)
    p = [passing_fraction(s, x50_true, n_true) for s in sizes]
    r = fit_curve(sizes, p)
    assert r.usable, r.reasons
    assert r.x50 == pytest.approx(x50_true, rel=1e-9)
    assert r.uniformity == pytest.approx(n_true, rel=1e-9)
    assert r.r2 == pytest.approx(1.0, abs=1e-9)
    assert r.max_p_residual_pct < 1e-7


def test_recover_across_parameter_range():
    for x50_true in (0.15, 0.25, 0.45):
        for n_true in (0.9, 1.5, 2.0):
            p = [passing_fraction(s, x50_true, n_true) for s in SIEVE_SIZES]
            r = fit_curve(SIEVE_SIZES, p)
            assert r.usable, (x50_true, n_true, r.reasons)
            assert r.x50 == pytest.approx(x50_true, rel=1e-8)
            assert r.uniformity == pytest.approx(n_true, rel=1e-8)


def test_noisy_curve_still_good():
    """轻微测量噪声的单峰曲线应判可用，参数偏差很小。"""
    sizes = np.array(SIEVE_SIZES)
    p_exact = np.array([passing_fraction(s, 0.30, 1.4) for s in sizes])
    rng = np.random.default_rng(11)
    bins = np.diff(p_exact, prepend=0.0) * (1 + rng.normal(0, 0.05, sizes.size))
    bins = np.clip(bins, 0.01, None)
    p = np.minimum(np.cumsum(bins), 99.5)
    r = fit_curve(sizes, p)
    assert r.usable, r.reasons
    assert abs(r.x50 - 0.30) < 0.03
    assert abs(r.uniformity - 1.4) < 0.15


def test_open_bins_excluded_but_curve_recovered():
    """最粗/最细开区间（0%、100%）不进入线性化回归，但内部档仍还原参数。"""
    x50_true, n_true = 0.30, 1.5
    sizes = OPEN_SIEVE_SIZES
    p = [0.0, passing_fraction(0.1, x50_true, n_true),
         passing_fraction(0.2, x50_true, n_true),
         passing_fraction(0.4, x50_true, n_true),
         passing_fraction(0.8, x50_true, n_true), 100.0]
    opens = [True, False, False, False, False, True]
    r = fit_curve(sizes, p, opens)
    assert r.usable, r.reasons
    assert r.included == [False, True, True, True, True, False]
    assert r.x50 == pytest.approx(x50_true, rel=1e-8)
    assert r.uniformity == pytest.approx(n_true, rel=1e-8)


def test_too_few_interior_points_flagged():
    r = fit_curve([0.1, 0.2, 0.3], [0.0, 100.0, 100.0],
                  [True, False, True])
    assert not r.usable
    assert any("interior points" in why for why in r.reasons)


def test_bimodal_curve_flagged_unusable():
    """两个峰（0.12m 与 0.75m 各一半）的混合曲线不能被单一 RR 描述。"""
    LN2 = np.log(2.0)
    sizes = np.array([0.02, 0.04, 0.08, 0.16, 0.25, 0.4, 0.63, 1.0, 1.5])

    def rr(x, x50, n):
        return 1 - np.exp(-LN2 * (x / x50) ** n)

    p = 100 * (0.5 * rr(sizes, 0.12, 1.6) + 0.5 * rr(sizes, 0.75, 1.6))
    r = fit_curve(sizes, p)
    assert not r.usable
    assert r.bimodal
    assert any("bimodal" in why for why in r.reasons)


def test_unimodal_not_falsely_flagged_bimodal():
    for seed in range(10):
        sizes = np.array(SIEVE_SIZES)
        p_exact = np.array([passing_fraction(s, 0.28, 1.3) for s in sizes])
        rng = np.random.default_rng(seed)
        bins = np.diff(p_exact, prepend=0.0) * (1 + rng.normal(0, 0.06, sizes.size))
        bins = np.clip(bins, 0.01, None)
        p = np.minimum(np.cumsum(bins), 99.5)
        r = fit_curve(sizes, p)
        assert not r.bimodal, seed


def test_residuals_align_with_input_rows():
    sizes = np.array(SIEVE_SIZES)
    p = np.array([passing_fraction(s, 0.3, 1.5) for s in sizes])
    r = fit_curve(sizes, p)
    assert len(r.residuals_pct) == len(sizes)
    assert all(abs(v) < 1e-7 for v in r.residuals_pct)


def test_quality_weight_zero_when_unusable():
    r = fit_curve([0.1, 0.2, 0.3], [0.0, 100.0, 100.0], [True, False, True])
    assert quality_weight(r) == 0.0
