"""岩石系数标定：单次反推、带遗忘因子的增量更新、全量重拟合。

估计量
------
一次合格实测反推出一个"本次隐含岩石系数"

    a_i = A_used * X50_measured / X50_predicted

（X50 与 A 为精确线性关系，见 fragmentation.implied_rock_factor）。

分区系数取全部观测的**加权几何均值**（在对数域做算术平均）：

    A_new = exp( (b*ln A0 + Σ w_i ln a_i) / (b + Σ w_i) )

  * A0 为分区基线系数，b 为基线权重——在任何实测入库前系数就是 A0，
    基线权重固定（不随时间衰减），保证始终有弱先验收束；
  * 对数域平均保证系数恒正，且"预测系统性偏小/偏大"按比例修正而不
    是按绝对量修正，与 X50∝A 的物理关系一致。

遗忘策略（增量）
----------------
取遗忘因子 rho∈(0,1]（默认 0.90）。按入库先后编号，越旧的观测权
重越低：当前最新观测权重为 1，倒数第 k 条为 rho**(k-1)。维护折叠
状态

    W  = Σ w_i          WL = Σ w_i ln a_i

新观测入库时只需

    W  ← rho*W + 1
    WL ← rho*WL + ln a_new

代价：每次更新 O(1) 时间、O(1) 存储，与历史条数无关；代价是对
"旧观测被撤回"无法原地逆操作——撤回时需要按存活观测重放折叠
（O(k)，仅在撤回时发生一次）。

rho=1（不衰减）时折叠退化为普通求和，与"用全部历史重新拟合"
数学上完全相同；测试要求两者相对差 < 1e-9。

并发
----
两次实测同时入库同一分区时，调用方（storage 层写入串行锁）保证
按确定的入库顺序依次调用 fold_add；本模块的折叠结果只取决于序列
顺序，与并发到达先后无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: 默认基线权重：查表系数与实际岩体往往偏差较大，默认不向其收缩，
#: 基线只作为"尚无实测"时的初值；需要弱先验时可按分区配置正值。
DEFAULT_BASELINE_WEIGHT = 0.0
#: 默认遗忘因子（开采推进，岩体逐渐变化；约每 7 条观测旧权减半）
DEFAULT_FORGETTING = 0.90


@dataclass
class Fold:
    """某分区当前的加权折叠状态（只含实测，不含固定基线）。"""

    w: float = 0.0
    wl: float = 0.0
    count: int = 0

    def add(self, log_a: float, forgetting: float) -> None:
        """并入一条新观测（其权重归一为 1，旧观测统一乘遗忘因子）。"""
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("遗忘因子必须在 (0, 1] 内")
        self.w = forgetting * self.w + 1.0
        self.wl = forgetting * self.wl + log_a
        self.count += 1


@dataclass
class Calibration:
    """一次标定计算的输入快照与结果。"""

    factor: float
    n_observations: int
    effective_weight: float
    baseline_factor: float
    baseline_weight: float
    forgetting: float
    # 参与计算的观测（按权重降序的入库顺序），便于审计
    contributions: list[tuple[float, float]] = field(default_factory=list)


def fold_from_observations(
    log_observations: list[float], forgetting: float
) -> Fold:
    """按入库顺序从空折叠重放全部观测（全量重拟合的折叠形式）。"""
    fold = Fold()
    for log_a in log_observations:
        fold.add(log_a, forgetting)
    return fold


def full_refit(
    implied_factors: list[float],
    baseline_factor: float,
    baseline_weight: float = DEFAULT_BASELINE_WEIGHT,
    forgetting: float = DEFAULT_FORGETTING,
) -> Calibration:
    """用全部历史观测重新拟合分区系数。

    implied_factors 必须按**入库顺序**排列；权重为
    rho**(距最新观测的条数)。与 fold 增量更新在同一权重定义下结果
    一致（rho=1 时逐位相同）。
    """
    _validate_baseline(baseline_factor, baseline_weight)
    if not implied_factors:
        return Calibration(
            factor=baseline_factor, n_observations=0,
            effective_weight=baseline_weight,
            baseline_factor=baseline_factor,
            baseline_weight=baseline_weight, forgetting=forgetting,
        )
    arr = np.asarray(implied_factors, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("隐含岩石系数必须全部为有限正数")
    k = int(arr.size)
    weights = forgetting ** (k - 1 - np.arange(k, dtype=np.float64))
    total_w = float(weights.sum())
    total_wl = float(np.dot(weights, np.log(arr)))
    log_base = math.log(baseline_factor)
    denom = baseline_weight + total_w
    factor = math.exp((baseline_weight * log_base + total_wl) / denom)
    contributions = list(zip(
        [float(v) for v in arr], [float(v) for v in weights]
    ))
    return Calibration(
        factor=factor, n_observations=k, effective_weight=denom,
        baseline_factor=baseline_factor,
        baseline_weight=baseline_weight, forgetting=forgetting,
        contributions=contributions,
    )


def fold_to_factor(
    fold: Fold,
    baseline_factor: float,
    baseline_weight: float = DEFAULT_BASELINE_WEIGHT,
) -> float:
    """由折叠状态 + 固定基线求当前系数。"""
    _validate_baseline(baseline_factor, baseline_weight)
    denom = baseline_weight + fold.w
    if denom <= 0:
        return baseline_factor
    return math.exp(
        (baseline_weight * math.log(baseline_factor) + fold.wl) / denom
    )


def incremental_update(
    fold: Fold,
    implied_factor: float,
    forgetting: float,
    baseline_factor: float,
    baseline_weight: float = DEFAULT_BASELINE_WEIGHT,
) -> float:
    """并入一条实测，返回更新后的系数（fold 原地更新）。"""
    if not math.isfinite(implied_factor) or implied_factor <= 0:
        raise ValueError(f"隐含岩石系数必须为正数，收到 {implied_factor}")
    fold.add(math.log(implied_factor), forgetting)
    return fold_to_factor(fold, baseline_factor, baseline_weight)


def _validate_baseline(baseline_factor: float, baseline_weight: float) -> None:
    if not math.isfinite(baseline_factor) or baseline_factor <= 0:
        raise ValueError("基线岩石系数必须为正数")
    if not math.isfinite(baseline_weight) or baseline_weight < 0:
        raise ValueError("基线权重不能为负")
