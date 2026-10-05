"""pytest 夹具：每个用例独立的临时数据库与 TestClient，以及合成筛分数据。"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import main  # noqa: E402
from app.storage import Storage  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def store(tmp_path):
    db = Storage(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
def client(tmp_path):
    main.init_with(str(tmp_path / "api.db"))
    with TestClient(main.app) as c:
        yield c


# ---------------------------------------------------------------- 合成数据

def rr_passing(x50: float, n: float, d: float) -> float:
    """给定 X50、n 的 Rosin-Rammler 通过率（0~1）。"""
    xc = x50 / math.log(2) ** (1 / n)
    return 1 - math.exp(-(d / xc) ** n)


def synthetic_bins(
    x50: float,
    n: float,
    edges=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2),
    open_fine: bool = True,
    open_coarse: bool = True,
):
    """从已知 X50、n 生成质量分档（百分数）。

    两端开口用 null 表示；两端都闭时，最细档从 0 起，最粗档补一个
    2 倍边界尺寸的闭档（真实 RR 在有限筛系上尾部的截断误差约
    1e-7，归一后可忽略）。
    """
    vals = [rr_passing(x50, n, e) for e in edges]
    bins = []
    if open_fine:
        bins.append({"lower": None, "upper": edges[0], "retained_pct": vals[0] * 100})
    for i, e in enumerate(edges):
        if i == 0:
            if open_fine:
                continue
            lo = 0.0
            frac = vals[0] * 100
        else:
            lo = edges[i - 1]
            frac = (vals[i] - vals[i - 1]) * 100
        bins.append({"lower": lo, "upper": e, "retained_pct": frac})
    if open_coarse:
        bins.append(
            {"lower": edges[-1], "upper": None,
             "retained_pct": (1 - vals[-1]) * 100}
        )
    else:
        e_top = 2.0 * edges[-1]
        bins.append({
            "lower": edges[-1], "upper": e_top,
            "retained_pct": (1 - vals[-1]) * 100,
        })
    return _renorm(bins)


def synthetic_bins_custom_passing(
    passing_fn,
    edges=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2),
):
    """按任意通过率函数 P(d) 生成带两端开区间的分档。"""
    vals = [max(0.0, min(1.0, passing_fn(e))) for e in edges]
    bins = [{"lower": None, "upper": edges[0], "retained_pct": vals[0] * 100}]
    for i, e in enumerate(edges):
        if i == 0:
            continue
        bins.append({
            "lower": edges[i - 1], "upper": e,
            "retained_pct": (vals[i] - vals[i - 1]) * 100,
        })
    bins.append({"lower": edges[-1], "upper": None,
                 "retained_pct": (1 - vals[-1]) * 100})
    return _renorm(bins)


def _renorm(bins):
    total = sum(b["retained_pct"] for b in bins)
    if total > 0:
        for b in bins:
            b["retained_pct"] = b["retained_pct"] * 100.0 / total
    return bins


# 一套典型中硬岩台阶爆破设计参数（米/毫米/千克）
TYPICAL_PARAMS = {
    "hole_diameter_mm": 250.0,
    "burden_m": 7.0,
    "spacing_m": 8.0,
    "bench_height_m": 12.0,
    "hole_depth_m": 13.5,
    "stemming_m": 4.5,
    "explosive_mass_kg": 420.0,
    "explosive_rws": 100.0,
    "hole_deviation_m": 0.3,
    "oversize_sieve_m": 0.6,
    "oversize_limit_pct": 5.0,
}
