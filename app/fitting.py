"""爆堆图像筛分曲线拟合：从实测累计通过率还原 Rosin-Rammler 的 X50 与 n。

设计决定（及原因）
------------------
1. 拟合空间：先把 Rosin-Rammler 累计分布线性化，再做**加权一元线性回归**（闭式，
   不调用任何曲线拟合/优化库）：

       R(x) = 1 - P/100 = exp(-(x/xc)^n)
       y = ln(-ln(1-P)) ,  z = ln(x)
       y = n*z + b ,  xc = exp(-b/n)
       X50 = xc * (ln2)^(1/n)

   线性空间里只需一次闭式解，残差有明确统计含义，也避免了非线性迭代（题目禁止使用
   现成优化/曲线拟合库）。

2. 各档权重：P 接近 0 或 1 时，y 对 P 的测量误差极其敏感（dy/dP = 1/((1-P) ln(1-P))
   在两端发散），图像筛分的百分率噪声若直接等权拟合会让两端档主导结果。采用
   delta 方法权重

       w = ((1-P) * (-ln(1-P)))^2 = Var(y) 的倒数（相对因子）

   使两端档权重自然趋零、中段（约 20%~80%）权重最高。

3. 开区间处理（“大于1.2m”“小于0.05m”）：这两档只给出边界上的开约束，其累计值
   分别固定为 100% 与 0%，在线性化空间里对应 y=+/-无穷，**无法作为观测点**。
   因此：开档不进入回归，仅用于校验数据范围；若开档恰好被录成非 0/100 的
   有限值（内部档），则按普通闭区间观测点参与拟合。open 标志的边界档权重再折半。

4. 拟合质量与双峰：在线性化空间计算残差与 R^2，在原始通过率空间给出最大绝对偏差
   （更直观、便于现场判读）；另外按对数间距的分档质量密度检测“两个峰夹一个明显
   谷”的双峰形态——双峰曲线无法用单一 RR 分布描述，标记 usable=False，不参与标定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LN2 = math.log(2.0)

# 质量门限（常量集中在此，便于现场调参）
MIN_INTERIOR_POINTS = 3
MIN_UNIFORMITY = 0.2
MAX_P_RESIDUAL_PCT = 5.0      # 通过率空间最大绝对偏差，百分点
MIN_R2 = 0.985                # 线性化空间加权 R^2
DENSITY_MIN_BINS = 5
HUMP_HEIGHT_RATIO = 0.35      # 次峰相对全局峰的最低高度
VALLEY_RATIO = 0.75           # 谷底相对全局峰需低至此值


@dataclass
class FitResult:
    x50: float
    uniformity: float
    usable: bool
    r2: float
    rmse_y: float
    max_p_residual_pct: float
    residuals_pct: list[float]          # 与输入逐档对齐；未参与档为 0.0
    included: list[bool]
    interior: list[bool]
    bimodal: bool
    reasons: list[str] = field(default_factory=list)


def _two_hump(density: np.ndarray) -> bool:
    """对数间距分档质量密度上的双峰检测：两侧各存在一个明显峰，中间有深谷。"""
    k = density.size
    if k < DENSITY_MIN_BINS:
        return False
    peak = float(density.max())
    # 扫描“局部最小”的候选谷；谷底左、右都要有不低的峰
    for i in range(1, k - 1):
        is_valley = density[i] < density[i - 1] and density[i] < density[i + 1]
        if not is_valley:
            continue
        left = float(density[:i].max())
        right = float(density[i + 1 :].max())
        if (
            density[i] < VALLEY_RATIO * peak
            and left > HUMP_HEIGHT_RATIO * peak
            and right > HUMP_HEIGHT_RATIO * peak
            and left > density[i] * 1.25
            and right > density[i] * 1.25
        ):
            return True
    return False


def fit_curve(
    sizes,
    passing_pct,
    open_flags=None,
) -> FitResult:
    """拟合 RR 曲线。

    sizes:        筛孔尺寸（米），严格递增、互不相同
    passing_pct:  各档累计通过率（0~100），非递减
    open_flags:   逐档是否开区间（True=“大于/小于该尺寸”的边界档）
    """
    sizes = np.asarray(sizes, dtype=float)
    pct = np.asarray(passing_pct, dtype=float)
    m = sizes.size
    open_flags = (
        np.zeros(m, dtype=bool)
        if open_flags is None
        else np.asarray(open_flags, dtype=bool)
    )

    P = pct / 100.0
    interior = (P > 1e-12) & (P < 1.0 - 1e-12)
    included = np.zeros(m, dtype=bool)

    reasons: list[str] = []
    if int(interior.sum()) < MIN_INTERIOR_POINTS:
        reasons.append(
            f"only {int(interior.sum())} finite interior points "
            f"(need >= {MIN_INTERIOR_POINTS})"
        )
        return FitResult(
            x50=float("nan"), uniformity=float("nan"), usable=False,
            r2=float("nan"), rmse_y=float("nan"),
            max_p_residual_pct=float("nan"),
            residuals_pct=[0.0] * m,
            included=included.tolist(), interior=interior.tolist(),
            bimodal=False, reasons=reasons,
        )

    z = np.log(sizes[interior])
    Pi = P[interior]
    y = np.log(-np.log(1.0 - Pi))

    # delta 方法权重：y 的相对方差倒数；开区间边界档（若因录入成有限值而进入）折半
    wq = ((1.0 - Pi) * (-np.log(1.0 - Pi))) ** 2
    wq = np.where(open_flags[interior], 0.5 * wq, wq)

    sw = float(wq.sum())
    mz = float(np.sum(wq * z) / sw)
    my = float(np.sum(wq * y) / sw)
    ssz = float(np.sum(wq * (z - mz) ** 2))
    slope = float(np.sum(wq * (z - mz) * (y - my)) / ssz)
    intercept = my - slope * mz

    n_fit = slope
    xc = math.exp(-intercept / n_fit)
    x50 = xc * LN2 ** (1.0 / n_fit)

    # ---- 残差与质量 ----
    included[interior] = True
    y_pred = slope * z + intercept
    res_y = y - y_pred
    rmse_y = float(math.sqrt(float(np.sum(wq * res_y**2)) / sw))
    ss_tot = float(np.sum(wq * (y - my) ** 2))
    r2 = 1.0 - float(np.sum(wq * res_y**2)) / ss_tot if ss_tot > 0 else float("nan")

    p_pred = 100.0 * (1.0 - np.exp(-(sizes / xc) ** n_fit))
    res_p = np.where(included, pct - p_pred, 0.0)
    max_p = float(np.max(np.abs(res_p[included])))

    # 双峰检测：按筛孔对数间距归一化的分档质量密度
    dP = np.diff(pct)
    dlog = np.diff(np.log(sizes))
    density = dP / dlog
    bimodal = _two_hump(density)

    if not math.isfinite(n_fit) or n_fit <= MIN_UNIFORMITY:
        reasons.append(f"fitted uniformity n={n_fit:.3g} not positive/physical")
    if math.isfinite(r2) and r2 < MIN_R2:
        reasons.append(f"weighted R2={r2:.4f} below {MIN_R2}")
    if max_p > MAX_P_RESIDUAL_PCT:
        reasons.append(
            f"max passing residual {max_p:.2f} pts exceeds {MAX_P_RESIDUAL_PCT}"
        )
    if bimodal:
        reasons.append("bimodal/non-single-RR pattern detected in bin density")

    return FitResult(
        x50=float(x50),
        uniformity=float(n_fit),
        usable=len(reasons) == 0,
        r2=float(r2),
        rmse_y=rmse_y,
        max_p_residual_pct=max_p,
        residuals_pct=[float(v) for v in res_p],
        included=included.tolist(),
        interior=interior.tolist(),
        bimodal=bimodal,
        reasons=reasons,
    )


def quality_weight(result: FitResult) -> float:
    """参与标定时该条实测的基础权重：拟合 R^2 截到 [0,1]；不可用曲线权重为 0。"""
    if not result.usable or not math.isfinite(result.r2):
        return 0.0
    return min(1.0, max(0.0, result.r2))
