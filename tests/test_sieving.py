"""筛分拟合测试：已知参数还原、开区间处理、坏数据与坏曲线。"""

from __future__ import annotations

import math

import pytest

from app.sieving import (
    SieveValidationError,
    bins_to_points,
    fit_curve,
    parse_bins,
)
from tests.conftest import (
    _renorm,
    rr_passing,
    synthetic_bins,
    synthetic_bins_custom_passing,
)


# ---------------------------------------------------------- 已知参数还原

@pytest.mark.parametrize("x50,n", [
    (0.15, 1.0),
    (0.30, 1.5),
    (0.42, 0.9),
    (0.25, 2.0),
])
def test_recover_known_curve(x50, n):
    bins = synthetic_bins(x50, n)
    r = fit_curve(bins)
    assert r.usable, r.reject_reason
    assert r.x50_m == pytest.approx(x50, rel=1e-6)
    assert r.uniformity_n == pytest.approx(n, rel=1e-6)
    assert r.weighted_r2 == pytest.approx(1.0, abs=1e-6)
    assert r.max_abs_residual_pp < 0.01


def test_recover_with_noise_still_usable():
    """小扰动（图像筛分正常误差）下仍应可用，且参数接近真值。"""
    bins = synthetic_bins(0.30, 1.5)
    # 给每个内部档 ±3% 的相对扰动并重新归一（避免产生负档）
    perturbed = []
    for i, b in enumerate(bins):
        factor = 1.03 if i % 2 == 0 else 0.97
        perturbed.append({**b, "retained_pct": b["retained_pct"] * factor})
    total = sum(b["retained_pct"] for b in perturbed)
    for b in perturbed:
        b["retained_pct"] *= 100.0 / total
    r = fit_curve(perturbed)
    assert r.usable, r.reject_reason
    assert r.x50_m == pytest.approx(0.30, rel=0.05)
    assert r.uniformity_n == pytest.approx(1.5, abs=0.2)


# ---------------------------------------------------------- 开区间

def test_open_bins_are_accepted_and_weighted_less():
    bins_open = synthetic_bins(0.30, 1.5, open_fine=True, open_coarse=True)
    parsed = parse_bins(bins_open)
    points = bins_to_points(parsed)
    # 最细边界点权重含开区间折扣
    fine_pt = points[0]
    assert fine_pt.open_boundary is True
    coarse_frac = bins_open[-1]["retained_pct"]
    fine_frac = bins_open[0]["retained_pct"]
    # 开细档：权重=占比*0.5；开粗档并入上一边界：+粗占比*0.5
    assert fine_pt.weight == pytest.approx(fine_frac * 0.5)
    top_pt = points[-1]
    assert top_pt.weight == pytest.approx(
        bins_open[-2]["retained_pct"] + coarse_frac * 0.5
    )
    r = fit_curve(bins_open)
    assert r.usable, r.reject_reason
    assert r.x50_m == pytest.approx(0.30, rel=1e-6)


def test_closed_ends_also_fit():
    bins_closed = synthetic_bins(0.30, 1.5, open_fine=False, open_coarse=False)
    r = fit_curve(bins_closed)
    assert r.usable, r.reject_reason
    assert r.x50_m == pytest.approx(0.30, rel=1e-6)


def test_single_open_fine_only():
    bins = synthetic_bins(0.25, 1.3, open_fine=True, open_coarse=False)
    r = fit_curve(bins)
    assert r.usable, r.reject_reason


def test_open_coarse_only():
    bins = synthetic_bins(0.25, 1.3, open_fine=False, open_coarse=True)
    r = fit_curve(bins)
    assert r.usable, r.reject_reason


# ---------------------------------------------------------- 校验拒收

def test_reject_non_monotonic_cumulative():
    # 构造累计通过率不单调（某档负占比在 parse 阶段被拦，
    # 这里用"占比都非负但加总不为100"之外，真正测不单调需要
    # 让累计回退——分档按尺寸排序后只能靠负档，因此直接构造点验证）：
    # 负占比必须被拒收
    bad = [
        {"lower": None, "upper": 0.1, "retained_pct": 20},
        {"lower": 0.1, "upper": 0.3, "retained_pct": -5},
        {"lower": 0.3, "upper": None, "retained_pct": 85},
    ]
    with pytest.raises(SieveValidationError):
        parse_bins(bad)


def test_reject_passing_out_of_range_via_gap():
    # 档与档不衔接，会造成累计定义错误（等价于曲线越界/不可信）
    bad = [
        {"lower": None, "upper": 0.1, "retained_pct": 30},
        {"lower": 0.2, "upper": 0.5, "retained_pct": 40},
        {"lower": 0.5, "upper": None, "retained_pct": 30},
    ]
    with pytest.raises(SieveValidationError):
        parse_bins(bad)


def test_reject_mass_not_100():
    bad = [
        {"lower": None, "upper": 0.1, "retained_pct": 30},
        {"lower": 0.1, "upper": None, "retained_pct": 50},
    ]
    with pytest.raises(SieveValidationError):
        parse_bins(bad)


def test_reject_too_few_bins():
    with pytest.raises(SieveValidationError):
        parse_bins([{"lower": 0, "upper": 1.0, "retained_pct": 100}])


def test_reject_two_open_same_end():
    bad = [
        {"lower": None, "upper": 0.05, "retained_pct": 5},
        {"lower": None, "upper": 0.1, "retained_pct": 10},
        {"lower": 0.1, "upper": None, "retained_pct": 85},
    ]
    with pytest.raises(SieveValidationError):
        parse_bins(bad)


def test_reject_zero_or_negative_sieve():
    for bad in (
        [{"lower": None, "upper": 0.0, "retained_pct": 10},
         {"lower": 0.0, "upper": None, "retained_pct": 90}],
        [{"lower": None, "upper": -0.1, "retained_pct": 10},
         {"lower": -0.1, "upper": None, "retained_pct": 90}],
        [{"lower": 0.0, "upper": -0.5, "retained_pct": 10},
         {"lower": -0.5, "upper": None, "retained_pct": 90}],
    ):
        with pytest.raises(SieveValidationError):
            parse_bins(bad)


# ---------------------------------------------------------- 坏曲线标记

def test_bimodal_curve_marked_unusable():
    """明显双峰（两条 Rosin-Rammler 的混合）必须被标记为不可用。"""
    def bimodal(d):
        return 0.45 * rr_passing(0.08, 1.6, d) + 0.55 * rr_passing(0.55, 1.9, d)

    bins = synthetic_bins_custom_passing(bimodal)
    r = fit_curve(bins)
    assert r.usable is False
    assert r.reject_reason
    # 残差/质量指标都有值
    assert math.isfinite(r.weighted_r2)
    structured = (r.sign_runs >= 4 and r.sign_blocks >= 2) or r.max_abs_residual_pp > 6.0
    assert structured


def test_s_shape_deviation_marked_unusable():
    """构造 S 形（中间平、两头陡）偏离 RR 的曲线，也要被拒。"""
    def s_curve(d):
        # 分段的逻辑式形状
        import numpy as np
        x = (math.log(d) - math.log(0.25)) / 0.25
        return 1.0 / (1.0 + math.exp(-x))

    bins = synthetic_bins_custom_passing(s_curve)
    r = fit_curve(bins)
    assert r.usable is False
    assert r.reject_reason


def test_unusable_does_not_need_x50():
    def bimodal(d):
        return 0.5 * rr_passing(0.07, 1.6, d) + 0.5 * rr_passing(0.6, 2.0, d)

    r = fit_curve(synthetic_bins_custom_passing(bimodal))
    assert r.usable is False
    assert r.residuals  # 仍然报告每个点的残差


def test_report_residuals_in_passing_space():
    r = fit_curve(synthetic_bins(0.3, 1.5))
    assert len(r.residuals) >= 3
    for pr in r.residuals:
        assert 0 <= pr.observed_pct <= 100
        assert pr.used_in_fit in (True, False)
        assert abs(pr.residual_pp) < 0.01


def test_extreme_endpoints_excluded_from_fit():
    """P=100 的零占比尾端点不进线性拟合但仍报告残差。"""
    edges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]
    vals = [rr_passing(0.3, 1.5, e) for e in edges]
    bins = [
        {"lower": None, "upper": edges[0], "retained_pct": vals[0] * 100}
    ]
    for i in range(1, len(edges)):
        bins.append({
            "lower": edges[i - 1], "upper": edges[i],
            "retained_pct": (vals[i] - vals[i - 1]) * 100,
        })
    # 1.2 处累计约 99.5%；加一个零占比闭尾档使 P=100 显式出现
    bins.append({"lower": edges[-1], "upper": 2.0,
                 "retained_pct": (1 - vals[-1]) * 100})
    bins.append({"lower": 2.0, "upper": 3.0, "retained_pct": 0.0})
    bins = _renorm(bins)
    r = fit_curve(bins)
    assert r.usable, r.reject_reason
    endpoint = r.residuals[-1]
    assert endpoint.observed_pct == pytest.approx(100.0)
    assert endpoint.used_in_fit is False
    # 其余内部点都参与了拟合
    assert r.n_interior_points >= 7
