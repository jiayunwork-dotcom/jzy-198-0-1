"""块度模型（Kuz-Ram）与 Rosin-Rammler 通过率。

预测采用业界通行的 Kuz-Ram 方法（Kuznetsov 中值块度 + Rosin-Rammler 分布 +
Cunningham 均匀性指数）：

  中值块度（米，Kuznetsov 1973 / Cunningham 1983 单耗形式）：
      X50 = 0.01 * A * q^(-0.8) * Q^(1/6) * (1.15/RWS)^(19/30)
  其中
      q   = Q / (B * S * H)   单耗（炸药质量/爆破岩石体积），kg/m^3
      Q   单孔装药量，kg
      RWS 炸药相对威力（ANFO=1.0）
      A   岩石系数（分区标定值）
  该式满足两条硬关系：q 增大时 X50 单调减小；X50 与 A 严格成正比。

  Rosin-Rammler 累计通过率（用中值块度 X50 与均匀性指数 n 参数化）：
      P(x) = 100 * (1 - exp(-ln2 * (x/X50)^n))
  参考：X50=0.30 m、n=1.5 时，0.60 m 筛孔通过率约 85.9%。

  均匀性指数（Cunningham 1987）：
      n = (2.2 - 14 B/D) * sqrt((H-W)/B) * (1 - W/B) * (Lc/H)^0.3
  单位约定（该经验式的原始标定单位）：B、S、H、W 用米；孔径 D 用毫米；
  Lc = 孔深 - 堵塞长度 为装药长度。
"""

from __future__ import annotations

import math
from typing import Mapping

LN2 = math.log(2.0)


def powder_factor(
    burden: float, spacing: float, bench_height: float, charge_kg: float
) -> float:
    """单耗 q（kg/m^3）= 单孔药量 / 单孔负担体积 B*S*H。"""
    return charge_kg / (burden * spacing * bench_height)


def median_size(
    rock_factor: float,
    burden: float,
    spacing: float,
    bench_height: float,
    charge_kg: float,
    explosive_rws: float,
) -> float:
    """Kuznetsov 中值块度 X50（米）。与 rock_factor 严格成正比，随单耗增大而减小。"""
    q = powder_factor(burden, spacing, bench_height, charge_kg)
    return (
        0.01
        * rock_factor
        * q ** (-0.8)
        * charge_kg ** (1.0 / 6.0)
        * (1.15 / explosive_rws) ** (19.0 / 30.0)
    )


def uniformity_index(
    burden: float,
    diameter_mm: float,
    hole_depth: float,
    stemming: float,
    bench_height: float,
    deviation: float = 0.0,
) -> float:
    """Cunningham 均匀性指数 n。deviation 为钻孔偏差（亚钻），米。"""
    charge_length = hole_depth - stemming
    if charge_length <= 0:
        raise ValueError("charge length must be positive (stemming >= hole depth)")
    n = (
        (2.2 - 14.0 * burden / diameter_mm)
        * math.sqrt((bench_height - deviation) / burden)
        * (1.0 - deviation / burden)
        * (charge_length / bench_height) ** 0.3
    )
    if not math.isfinite(n) or n <= 0:
        raise ValueError("uniformity index is not positive for these parameters")
    return n


def passing_fraction(sieve_m: float, x50: float, n: float) -> float:
    """Rosin-Rammler：筛孔 sieve_m（米）下的累计通过率，百分数（0~100）。"""
    if sieve_m <= 0:
        raise ValueError("sieve size must be positive")
    return 100.0 * (1.0 - math.exp(-LN2 * (sieve_m / x50) ** n))


def oversize_fraction(oversize_m: float, x50: float, n: float) -> float:
    """大块率（%）= 100 - 大块界定筛孔的通过率。"""
    return 100.0 - passing_fraction(oversize_m, x50, n)


def back_calculate_factor(
    measured_x50: float,
    used_factor: float,
    predicted_x50: float,
) -> float:
    """单次反推岩石系数。由于 X50 与 A 成正比：A_new = A_used * X_meas / X_pred。"""
    return used_factor * measured_x50 / predicted_x50


def predict(params: Mapping[str, float], rock_factor: float) -> dict:
    """对一版爆破设计参数做完整预测。"""
    x50 = median_size(
        rock_factor,
        params["burden"],
        params["spacing"],
        params["bench_height"],
        params["charge_kg"],
        params["explosive_rws"],
    )
    n = uniformity_index(
        params["burden"],
        params["diameter_mm"],
        params["hole_depth"],
        params["stemming"],
        params["bench_height"],
        params.get("deviation", 0.0),
    )
    over_m = params.get("oversize_m", 0.8)
    return {
        "x50": x50,
        "uniformity": n,
        "powder_factor": powder_factor(
            params["burden"], params["spacing"], params["bench_height"],
            params["charge_kg"],
        ),
        "oversize_pct": oversize_fraction(over_m, x50, n),
    }
