# 露天矿爆破块度闭环管理服务

把**爆破设计 → 块度预测 → 爆后图像筛分实测 → 岩石系数标定**串成闭环的后端服务。

* 语言/框架：Python 3.12 + FastAPI，数据库 SQLite（文件放在挂载卷 `/data`）
* 数值：NumPy；**块度模型、筛分拟合、系数标定全部自己实现**，不调用
  scipy / curve_fit / 任何优化库
* 测试：pytest

## 模块划分

| 模块 | 职责 |
| --- | --- |
| `app/fragmentation.py` | Kuz-Ram 块度模型：Kuznetsov 中值块度 X50、Cunningham 均匀性指数 n、Rosin-Rammler 通过率、大块率判定、单次反推 |
| `app/sieving.py` | 图像筛分数据校验、开区间处理、双对数线性化加权拟合、残差与拟合质量判定 |
| `app/calibration.py` | 岩石系数：加权几何均值、带遗忘因子的增量折叠、全量重拟合 |
| `app/services.py` | 领域编排：分区/设计版本/预测/爆破实施/实测入库/撤回重推/批量重预测 |
| `app/storage.py` | SQLite 表结构、写锁串行化、版本与谱系查询 |
| `app/main.py` / `app/schemas.py` | HTTP 接口与请求模型 |

## 块度模型（Kuz-Ram）

```
X50 = C·A·q^(−1/6)·(115/RWS)^0.633           q = Q/(B·S·H)   单耗
n   = (2.2−14B/D_mm)·(1−|H−L|/B)·(|B−S|/(2B))^0.1·(L/H)^0.1·1.1/(|dev|/B+1)
P(d) = 1 − exp(−(d/Xc)^n),    Xc = X50/(ln2)^(1/n)
大块率 = 1 − P(大块判定筛孔)
```

* **单耗 q 越高，X50 越小**（严格单调）；RWS 越大，X50 越小。
* **X50 与岩石系数 A 精确成正比**，因此单次反推
  `A_new = A_used · X50_measured / X50_predicted`。
  例：A=7 预测 0.40 m、实测 0.48 m ⇒ A=8.4。
* 参考点：X50=0.30 m、n=1.5 时，0.60 m 筛孔通过率 = **85.9%**。

## 筛分拟合

Rosin-Rammler 双次对数后在 `(ln d, ln(−ln(1−P)))` 空间是严格直线：

```
y = n·x − n·ln Xc      （斜率即均匀性指数 n）
```

* **拟合空间**：双对数线性空间——有闭式解，无须迭代/优化库。
* **权重**：各档质量占比（信息量越大权重越大）；两端开区间档权重打 5 折。
* **开区间**："小于 a"在边界 a 处计入累计但降权；"大于 a"不产生新点，
  其质量按折扣并入前一边界点；P≈0、P≈100 的端点不进拟合但仍报残差。
* **质量判定**（任一不过即 `usable=false`，**不参与标定**）：
  内部点 < 3；变换空间加权 R² < 0.97；通过率空间最大残差 > 6 个百分点；
  残差出现"多游程且存在同号连续段"（系统性弯曲，典型如**双峰**；
  纯交替的小残差视为随机噪声放行）。

## 岩石系数标定

每个岩石分区独立维护。一次合格实测反推一个隐含系数 `a_i`，分区系数取
**加权几何均值**（对数域算术平均，保证恒正、按比例修正）：

```
A = exp( (b·ln A0 + Σ w_i·ln a_i) / (b + Σ w_i) )
```

* `A0` 为查表基线系数，默认 `b=0`——基线只作为"无实测时的初值"，
  一旦有实测即完全由实测决定（可按分区配置弱先验权重）。
* **增量更新（默认方案）**：遗忘因子 ρ（默认 0.90，可按分区设）。维护
  折叠 `W = Σw_i, WL = Σw_i·ln a_i`，新观测入库
  `W ← ρW + 1, WL ← ρWL + ln a_new`。每次 O(1)、与历史长度无关，
  旧观测权重随开采推进逐版衰减。
* **全量重拟合**：用分区全部存活观测按入库顺序重算；`ρ=1` 时与增量
  结果逐位一致（测试要求相对差 **< 1e-9**）。撤回时用它做一次重放。
* **代价**：增量 O(1) 更新但无法对"撤回旧观测"原地逆操作；撤回是
  罕见操作，发生时按存活观测 O(k) 重放一次。

## 版本、撤回与并发

* 每次系数变化追加一个**不可变版本**（baseline / measurement / withdrawal），
  可查分区任意时刻（`/factor/as-of?at=...`）的系数，也可查某次爆破当时
  冻结的是哪一版（爆破行固定 `factor_version_id`）。
* **撤回实测**：该实测与它之后的活动版本标 `superseded`（历史保留、
  `include_superseded=true` 可查），在剔除该条的基础上追加 withdrawal
  版本与 rebuilt 版本，存活实测的版本外键重挂到新版本。
* **并发入库**：所有写事务在进程内一把锁 + SQLite WAL 下串行；每次实测
  的分区序号 `seq_in_zone` 在锁内分配，最终系数恒等于"按实际持锁顺序
  逐条入库"的结果。
* 预测**同时绑定设计版本与系数版本**；设计修改产生新版本，已实施设计
  冻结不可改。系数更新后 `POST /designs/repredict` 对未实施设计批量
  重预测，并给出大块率合格状态翻转清单（`pass_to_fail` / `fail_to_pass`）。

## 运行

```bash
docker build -t blast-fragmentation .
docker run -p 8000:8000 -v $(pwd)/data:/data blast-fragmentation
# 或
docker compose up --build
```

数据库路径由 `BLAST_DB_PATH` 控制，默认 `/data/blast.db`。

## 主要接口

| 方法/路径 | 说明 |
| --- | --- |
| `POST /zones` `GET /zones` `GET /zones/{id}` | 岩石分区 |
| `GET /zones/{id}/factor` | 当前系数版本 |
| `GET /zones/{id}/factor/history` | 系数历史（`include_superseded=true` 含已取代行） |
| `GET /zones/{id}/factor/as-of?at=ISO` | 任意时刻系数 |
| `POST /zones/{id}/factor/rebuild` | 全量重拟合与增量结果对比（审计） |
| `POST /designs` `PATCH /designs/{id}` `GET /designs` | 具名设计（引用分区）/改版本 |
| `POST /designs/{id}/predict` `GET /designs/{id}/predictions` | 预测（绑定双版本） |
| `POST /designs/repredict` | 批量重预测 + 翻转清单 |
| `GET /passing?sieve_m=&x50_m=&uniformity_n=` | 任意筛孔通过率 |
| `POST /blasts` `GET /blasts/{id}` | 登记爆破实施（冻结设计/系数版本） |
| `POST /measurements` | 上传图像筛分（拟合+反推+标定） |
| `GET /measurements/{id}` | 拟合结果（X50、n、残差、质量、是否可用） |
| `POST /measurements/{id}/withdraw` | 撤回实测并级联重推系数版本 |

开区间筛档："小于 0.05 m" 用 `{"lower": null, "upper": 0.05, "retained_pct": 3}`，
"大于 1.2 m" 用 `{"lower": 1.2, "upper": null, "retained_pct": 2}`。

## 拒收（HTTP 400/404/409）

* 孔径/孔距/排距/台阶高/装药量非正；堵塞长度 > 孔深；炸药相对威力非正 → 400
* 筛分累计通过率不单调（出现负档）、加总不为 100%（容差 1 个百分点）、
  筛孔非正、档与档不衔接 → 400
* 引用不存在的分区/设计/爆破/预测 → 404
* 重复实施、对已实施设计改参数、同一爆破重复上有效实测 → 409

## 测试

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

覆盖：通过率参考值（85.9%）、单耗/威力单调、X50∝A 与 8.4 单次反推、
已知参数合成曲线还原、开区间、坏曲线（双峰）标记、ρ=1 增量与全量
< 1e-9、撤回后重推、并发入库与按序处理一致、批量重预测翻转清单，
以及全部 HTTP 拒收规则。
