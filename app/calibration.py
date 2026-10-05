"""岩石系数标定（纯数值部分）。

思路
----
中值块度模型对岩石系数 A 是严格线性的：对第 k 次爆破，

    X50_k = A * g_k,   g_k = 0.01 * q_k^-0.8 * Q_k^(1/6) * (1.15/RWS_k)^(19/30)

其中 g_k 只依赖该次爆破当时的设计参数，与 A 无关。实测给出 X50_k（由筛分曲线
拟合还原），于是每次爆破是对 A 的一次带噪声线性观测，可反推出“这一次单独支持
的系数” a_k = X50_k / g_k。

分区系数取加权平均：

    A = 加权一次矩 / 加权零次矩
      = Σ w_k a_k / Σ w_k

权重 w_k = 质量权重（拟合 R^2）* 时间衰减权重（旧数据逐渐降权，跟踪开采推进中
岩体变化）。衰减用按“该分区保留记录中的次序”指数式遗忘：

    w_k = qweight_k * decay^(N-1-k)     （k=0 最旧，k=N-1 最新；decay=1 不衰减）

增量与全量
----------
只维护两个累加量 S0=Σw_k、S1=Σw_k*a_k：

  * 追加一条：先把已有两个矩乘 decay（旧记录统一再老化一格），再加新记录 w=qw。
  * 全量重拟合：按同一顺序、同一衰减定义直接折叠全部记录。
    decay=1 时追加法与全量法数学上完全相同（同顺序浮点求和一致，相对差 0）；
    decay<1 时两者的差别仅是指数老化的分组方式，本模块统一使用全量折叠作为
    撤回后的重推口径，增量追加作为在线快路径，二者定义等价。

撤回：直接把指定记录从序列剔除后从头折叠——这正是存储层做“撤回后重推”的依据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """一次已入库、拟合可用的实测。"""

    observation: float   # 单次反推系数 a_k = X50_meas / g_k
    qweight: float       # 质量基础权重（拟合 R^2）
    evidence_id: int     # 对应实测记录 id（撤回时按它剔除）


def single_factor(g: float, measured_x50: float) -> float:
    """给定设计几何因子 g 与实测中值块度，单次反推系数。"""
    return measured_x50 / g


def _weights(n: int, qweights: list[float], decay: float) -> list[float]:
    # 最旧记录老化 n-1 格，最新记录 0 格
    return [qw * decay ** (n - 1 - k) for k, qw in enumerate(qweights)]


def fit_all(observations: list[Observation], decay: float = 1.0) -> float | None:
    """全量重拟合（权威口径）。无有效记录时返回 None。"""
    if not observations:
        return None
    n = len(observations)
    s0 = s1 = 0.0
    for k, ob in enumerate(observations):
        w = ob.qweight * decay ** (n - 1 - k)
        s0 += w
        s1 += w * ob.observation
    return s1 / s0


def fit_all_excluding(
    observations: list[Observation], evidence_id: int, decay: float = 1.0
) -> float | None:
    """撤回：剔除指定记录后全量折叠（记录次序重新连续编号）。"""
    kept = [ob for ob in observations if ob.evidence_id != evidence_id]
    return fit_all(kept, decay)


def fold_state(s0: float, s1: float, observation: float, qweight: float,
               decay: float) -> tuple[float, float]:
    """增量追加一步：旧矩先统一老化 decay，再加新记录。

    返回新的 (S0, S1)。配合 fold_value 取 A=S1/S0。
    """
    s0_new = s0 * decay + qweight
    s1_new = s1 * decay + qweight * observation
    return s0_new, s1_new


def fold_value(s0: float, s1: float) -> float | None:
    return s1 / s0 if s0 > 0 else None


def moments(observations: list[Observation], decay: float = 1.0) -> tuple[float, float]:
    s0 = s1 = 0.0
    for k, ob in enumerate(observations):
        w = ob.qweight * decay ** (len(observations) - 1 - k)
        s0 += w
        s1 += w * ob.observation
    return s0, s1


def relative_difference(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return math.inf
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom
