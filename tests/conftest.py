"""共享 pytest fixtures：每个测试独立临时 SQLite + FastAPI TestClient。"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="blast-test-")
    os.close(fd)
    os.environ["BLAST_DB"] = path
    # storage 在 import 时把 DB_PATH 固化为模块属性，需导入后覆盖
    from app import storage
    storage.DB_PATH = path
    storage.init_db(storage.connect(path))
    yield path
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


@pytest.fixture()
def conn(db_path):
    from app import storage
    c = storage.connect(db_path)
    yield c
    c.close()


@pytest.fixture()
def client(db_path):
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def standard_design_payload():
    """一组合理的生产爆破参数（250mm 孔径，6x7.5m 孔网）。"""
    return {
        "name": "1080平台N3爆区",
        "diameter_mm": 250.0,
        "burden": 6.0,
        "spacing": 7.5,
        "bench_height": 12.0,
        "hole_depth": 13.5,
        "stemming": 4.0,
        "charge_kg": 350.0,
        "explosive_rws": 1.0,
        "deviation": 0.0,
        "oversize_m": 0.8,
        "oversize_limit_pct": 10.0,
    }


SIEVE_SIZES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
OPEN_SIEVE_SIZES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2]  # 首末为开区间


def rr_curve(sizes, x50, n, noise=0.0, seed=0, open_ends=False):
    """按 RR(X50,n) 合成一条累计通过率曲线；可选轻微单调噪声。"""
    import numpy as np
    from app.fragmentation import passing_fraction

    p = np.array([passing_fraction(s, x50, n) for s in sizes])
    if noise:
        rng = np.random.default_rng(seed)
        bins = np.diff(p, prepend=0.0) * (1.0 + rng.normal(0, noise, p.size))
        bins = np.clip(bins, 0.01, None)
        p = np.minimum(np.cumsum(bins), 99.9)
    p = np.clip(p, 0.0, 100.0)
    points = []
    for i, (s, v) in enumerate(zip(sizes, p)):
        is_open = open_ends and i in (0, len(sizes) - 1)
        points.append({
            "size_m": float(s),
            "passing_pct": 0.0 if (is_open and i == 0) else
                           (100.0 if is_open else float(round(v, 6))),
            "open": bool(is_open),
        })
    return points
