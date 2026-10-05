"""块度预测模型（Kuz-Ram）。

采用业界通行的 Kuz-Ram 方法：

1. Kuznetsov（库兹涅佐夫）方程给出中值块度 X50；
2. Cunningham 均匀性方程给出 Rosin-Rammler（罗辛-拉姆勒）分布的
   均匀性指数 n；
3. 由 X50、n 按 Rosin-Rammler 分布计算任意筛孔的通过率：

       P(d) = 1 - exp( -(d / Xc) ** n )
       Xc   = X50 / (ln 2) ** (1/n)

约定：长度全部用米，装药量用千克，单耗用 kg/m^3。
岩石系数 A（岩体系数）为无量纲经验值，是标定的对象。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Kuznetsov 方程的单位经验常数（长度 m、单耗 kg/m^3）。
#: 按中硬岩典型参数（q≈0.6 kg/m^3、A=7）反标到 X50≈0.3 m 的量级；
#: X50 对岩石系数仍为精确线性关系，不影响标定逻辑。
KUZNETSOV_CONSTANT = 0.04

#: 均匀性指数允许的数值范围（防止极端孔网参数把指数推出物理区间）。
N_MIN = 0.05
N_MAX = 5.0

#: 默认大块判定筛孔（米）。
DEFAULT_OVERSIZE_SIEVE = 0.60
#: 默认大块率合格上限（百分数）。
DEFAULT_OVERSIZE_LIMIT_PCT = 5.0


class ValidationError(ValueError):
    """设计参数不合法。"""


@dataclass(frozen=True)
class DesignParams:
    """单次爆破设计所需的几何与装药参数。"""

    hole_diameter_mm: float          # 孔径 D（毫米）
    burden_m: float                  # 底盘抵抗线 B（米）
    spacing_m: float                 # 孔距 S（米）
    bench_height_m: float            # 台阶高度 H（米）
    hole_depth_m: float              # 孔深 L（米）
    stemming_m: float                # 堵塞长度 T（米）
    explosive_mass_kg: float         # 单孔装药量 Q（千克）
    explosive_rws: float = 100.0     # 炸药相对威力（ANFO=100）
    hole_deviation_m: float = 0.0    # 钻孔偏差
    oversize_sieve_m: float = DEFAULT_OVERSIZE_SIEVE
    oversize_limit_pct: float = DEFAULT_OVERSIZE_LIMIT_PCT

    def validate(self) -> None:
        """按需求做输入拒收检查。"""
        positives = {
            "孔径": self.hole_diameter_mm,
            "孔距": self.spacing_m,
            "排距(抵抗线)": self.burden_m,
            "台阶高度": self.bench_height_m,
            "装药量": self.explosive_mass_kg,
        }
        for name, value in positives.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValidationError(f"{name}必须是有限数值")
            if value <= 0:
                raise ValidationError(f"{name}必须为正数，收到 {value}")
        if not math.isfinite(self.hole_depth_m) or self.hole_depth_m <= 0:
            raise ValidationError("孔深必须为正数")
        if not math.isfinite(self.stemming_m) or self.stemming_m < 0:
            raise ValidationError("堵塞长度不能为负")
        if self.stemming_m > self.hole_depth_m:
            raise ValidationError(
                f"堵塞长度({self.stemming_m} m)大于孔深({self.hole_depth_m} m)"
            )
        if not math.isfinite(self.explosive_rws) or self.explosive_rws <= 0:
            raise ValidationError("炸药相对威力必须为正数")
        if not math.isfinite(self.hole_deviation_m) or self.hole_deviation_m < 0:
            raise ValidationError("钻孔偏差不能为负")
        charge_length = self.hole_depth_m - self.stemming_m
        if charge_length <= 0:
            raise ValidationError("装药长度（孔深-堵塞）必须为正数")
        if self.oversize_sieve_m <= 0:
            raise ValidationError("大块判定筛孔必须为正数")
        if not 0.0 < self.oversize_limit_pct < 100.0:
            raise ValidationError("大块率合格上限必须在 0 到 100% 之间")


@dataclass(frozen=True)
class Prediction:
    """块度预测结果。"""

    x50_m: float                # 中值块度
    uniformity_n: float         # 均匀性指数
    characteristic_m: float     # Rosin-Rammler 特征块度 Xc（P=63.2%）
    powder_factor_kg_m3: float  # 单耗
    oversize_sieve_m: float
    oversize_pct: float         # 大块率（筛上累计百分数）
    oversize_limit_pct: float
    oversize_ok: bool           # 大块率是否合格
    rock_factor: float

    def passing_fraction(self, sieve_m: float) -> float:
        """给定筛孔尺寸（米）下的通过率，返回 0~1 之间的值。"""
        return passing_fraction(self.x50_m, self.uniformity_n, sieve_m)


def powder_factor(p: DesignParams) -> float:
    """单耗 q = 单孔装药量 / 每孔破碎岩体体积（B*S*H）。"""
    p.validate()
    volume = p.burden_m * p.spacing_m * p.bench_height_m
    return p.explosive_mass_kg / volume


def predict_x50(p: DesignParams, rock_factor: float) -> float:
    """Kuznetsov 方程（单耗形式）：中值块度（米）。

        X50 = C * A * q**(-1/6) * (115/RWS)**0.633
        q   = Q / (B*S*H)          # 单耗 kg/m^3

    A 为岩石系数，RWS 为炸药相对威力（ANFO≈100），C 为单位经验常数。
    性质：X50 对 A 是精确正比；对单耗 q 单调递减（q 越大，破碎越细）；
    RWS 越大块度越小。经典 Kuznetsov 形式
    C*A*(V0/Q)**(1/6)*Q**(1/6) 退化为只与体积有关，无法体现"提高装药
    使块度变细"，故这里采用显式含单耗的形式。
    """
    p.validate()
    if not math.isfinite(rock_factor) or rock_factor <= 0:
        raise ValidationError("岩石系数必须为正数")
    volume = p.burden_m * p.spacing_m * p.bench_height_m
    q = p.explosive_mass_kg / volume
    x50 = (
        KUZNETSOV_CONSTANT
        * rock_factor
        * q ** (-1.0 / 6.0)
        * (115.0 / p.explosive_rws) ** 0.633
    )
    return x50


def predict_uniformity(p: DesignParams) -> float:
    """Cunningham 均匀性方程，返回 Rosin-Rammler 指数 n。

        n = (2.2 - 14*B/D_mm) * (1 - |H-L|/B) *
            (|B-S|/(2B))**0.1 * (L/H)**0.1 * 1.1/(|dev|/B + 1)

    五个因子依次反映：
      1) 孔径与抵抗线的匹配（B 抵抗线 m，D_mm 孔径 mm，典型 B/D_mm≈
         28，首项约 1.8）；
      2) 超钻合理性（|H-L| 超深过大/欠深都会降低均匀性）；
      3) 孔网几何（孔距 S 与抵抗线 B 的匹配程度）；
      4) 孔深/台阶高（装药相对高度）；
      5) 钻孔偏差（偏差越大越不均匀）。

    注意堵塞长度 W 不出现在 Cunningham 的 n 公式中，它通过
    DesignParams 的合法性（不能大于孔深）与块度上限间接约束。
    结果夹紧在 [N_MIN, N_MAX]。
    """
    p.validate()
    b = p.burden_m
    f1 = 2.2 - 14.0 * b / p.hole_diameter_mm
    f2 = 1.0 - abs(p.bench_height_m - p.hole_depth_m) / b
    f3 = (abs(b - p.spacing_m) / (2.0 * b)) ** 0.1
    f4 = (p.hole_depth_m / p.bench_height_m) ** 0.1
    f5 = 1.1 / (abs(p.hole_deviation_m) / b + 1.0)
    n = f1 * f2 * f3 * f4 * f5
    if not math.isfinite(n) or n < N_MIN:
        n = N_MIN
    if n > N_MAX:
        n = N_MAX
    return n


def characteristic_size(x50_m: float, n: float) -> float:
    """由中值块度求 Rosin-Rammler 特征块度 Xc。"""
    return x50_m / (math.log(2.0) ** (1.0 / n))


def passing_fraction(x50_m: float, n: float, sieve_m: float) -> float:
    """Rosin-Rammler 累计通过率（0~1）：P(d)=1-exp(-(d/Xc)^n)。"""
    if sieve_m < 0:
        raise ValidationError("筛孔尺寸不能为负")
    if sieve_m == 0.0:
        return 0.0
    xc = characteristic_size(x50_m, n)
    return 1.0 - math.exp(-((sieve_m / xc) ** n))


def predict(p: DesignParams, rock_factor: float) -> Prediction:
    """完整预测：中值块度、均匀性指数、大块率与合格判定。"""
    p.validate()
    x50 = predict_x50(p, rock_factor)
    n = predict_uniformity(p)
    xc = characteristic_size(x50, n)
    pf = powder_factor(p)
    passing = passing_fraction(x50, n, p.oversize_sieve_m)
    oversize_pct = (1.0 - passing) * 100.0
    return Prediction(
        x50_m=x50,
        uniformity_n=n,
        characteristic_m=xc,
        powder_factor_kg_m3=pf,
        oversize_sieve_m=p.oversize_sieve_m,
        oversize_pct=oversize_pct,
        oversize_limit_pct=p.oversize_limit_pct,
        oversize_ok=oversize_pct <= p.oversize_limit_pct,
        rock_factor=rock_factor,
    )


def implied_rock_factor(
    p: DesignParams, measured_x50_m: float, predicted_x50_m: float, used_factor: float
) -> float:
    """由单次实测中值块度反推岩石系数。

    X50 与 A 成正比：A_new = A_used * X50_measured / X50_predicted。
    例：A=7 预测 0.40 m，实测 0.48 m，则 A_new = 7*0.48/0.40 = 8.4。
    """
    if predicted_x50_m <= 0:
        raise ValidationError("预测中值块度必须为正数")
    if measured_x50_m <= 0:
        raise ValidationError("实测中值块度必须为正数")
    return used_factor * measured_x50_m / predicted_x50_m
