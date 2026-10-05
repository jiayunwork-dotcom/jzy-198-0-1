"""系数标定数值：加权折叠、增量与全量一致（不衰减时）、撤回剔除。"""

import pytest

from app import calibration as cal


def _obs(pairs, evidence_start=1):
    return [
        cal.Observation(observation=a, qweight=w, evidence_id=evidence_start + i)
        for i, (a, w) in enumerate(pairs)
    ]


def test_weighted_average_basic():
    obs = _obs([(8.0, 1.0), (9.0, 0.5), (7.0, 0.25)])
    assert cal.fit_all(obs, decay=1.0) == pytest.approx(
        (8 * 1 + 9 * 0.5 + 7 * 0.25) / 1.75
    )


def test_empty_is_none():
    assert cal.fit_all([], 1.0) is None


def test_incremental_matches_full_no_decay_rel_1e_9():
    """不衰减时，增量追加与全量重拟合相对差 < 1e-9（多条、随机权重）。"""
    rng = __import__("numpy").random.default_rng(0)
    pairs = [(float(6 + rng.normal(0, 1.2)), float(rng.uniform(0.5, 1.0)))
             for _ in range(40)]

    # 全量
    full = cal.fit_all(_obs(pairs), decay=1.0)

    # 增量
    s0 = s1 = 0.0
    for a, w in pairs:
        s0, s1 = cal.fold_state(s0, s1, a, w, decay=1.0)
    incr = cal.fold_value(s0, s1)

    assert cal.relative_difference(incr, full) < 1e-9
    assert incr == pytest.approx(full, abs=1e-12)


def test_incremental_matches_full_prefix_every_step():
    """每一步增量结果都与当时全量结果一致（不衰减）。"""
    rng = __import__("numpy").random.default_rng(3)
    pairs = [(float(7 + rng.normal(0, 0.8)), float(rng.uniform(0.3, 1.0)))
             for _ in range(20)]
    s0 = s1 = 0.0
    for k, (a, w) in enumerate(pairs, start=1):
        s0, s1 = cal.fold_state(s0, s1, a, w, 1.0)
        full_k = cal.fit_all(_obs(pairs[:k]), 1.0)
        assert cal.relative_difference(cal.fold_value(s0, s1), full_k) < 1e-12


def test_decay_downweights_old_data():
    """衰减存在时，较新数据权重更大：结果向最新观测靠拢。"""
    obs = _obs([(5.0, 1.0), (9.0, 1.0)])
    no_decay = cal.fit_all(obs, 1.0)
    decayed = cal.fit_all(obs, 0.5)
    assert no_decay == pytest.approx(7.0)
    # 旧观测只拿一半权重：(0.5*5 + 1*9)/1.5
    assert decayed == pytest.approx((0.5 * 5 + 9) / 1.5)
    assert decayed > no_decay  # 偏向新值 9


def test_excluding_observation():
    pairs = [(8.0, 1.0), (9.0, 1.0), (7.0, 1.0)]
    obs = _obs(pairs)
    kept = cal.fit_all_excluding(obs, evidence_id=2, decay=1.0)
    assert kept == pytest.approx(7.5)  # (8+7)/2


def test_back_calc_single_factor():
    # g 与实测 x50：A=7 给 0.40，实测 0.48
    g = 0.40 / 7.0
    assert cal.single_factor(g, 0.48) == pytest.approx(8.4)
