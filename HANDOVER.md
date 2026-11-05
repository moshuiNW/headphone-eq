# 交接文档 · headphone-eq skill

> 更新时间：2026-11-03　状态：**可用，性能回归已解决（139 s → 3.3 s），全部闸门 PASS**
> 位置：`hifi-eq-g2pro/skills/headphone-eq/`

---

## 1. 项目目的

给任意耳机生成**两份 Equalizer APO 参数 EQ 配置**：

| 配置 | 目标曲线 | 用途 |
|---|---|---|
| **HARMAN** | Harman over-ear 2018 | 中性参考调音 |
| **GAME** | 游戏头戴参考曲线（评测站蓝虚线，已数字化） | 游戏调音：低频更多、3 kHz 更高 |

输入三种都支持：耳机型号（自动去 huihifi.com 搜频响曲线）、指定评测 uuid + 曲线名、或用户自备的频响 CSV。

### 用户已确认的设计决定

| 决定 | 内容 |
|---|---|
| 蓝虚线来源 | 复用仓库已数字化的 `data/blue-dashed.csv`（对哈曼真值 <1.5 dB @15 kHz 以下） |
| 交付形态 | SKILL.md 判据文档 + 可运行 Python 工具集 |
| 滤波器约束 | **不限段数，能多准就多准**（用户明确表示多几个滤波器无所谓） |
| 多曲线策略 | 多版本并列给候选并推荐其一；**找不到则明确拒绝出配置** |
| 多曲线交互 | 评测含多条频响曲线时**必须先问用户**（如 `2.4G模式频响:SPL(dB)` vs `一致性R频响:SPL(dB)`），选定后用 `--mode` 指定 |
| **速度** | **充分利用 CPU，不允许长时间等待**（本轮的主要目标） |

### 质量判据（`metrics.GATES`，单一权威来源）

- 加权 RMS 20 Hz–16 kHz ≤ 1.0 dB；核心段（20 Hz–5 kHz）≤ 0.8 dB
- 最差点：核心段 ≤ 2.5 dB；5–10 kHz ≤ 5.0 dB
- 单滤波器：boost ≤ +6 dB（>5 kHz 只允许 +3 dB）；cut ≥ −20 dB；Q ≤ 4.5（>5 kHz ≤ 3.0）
- >5 kHz 滤波器 ≤ 3 个；chain peak + preamp ≤ −0.5 dBFS
- **能力闸门与拟合闸门分离**：残差闸门超标时报告「修复需多少 dB、被哪条上限挡住」，绝不静默放宽

---

## 2. 目录结构与运行方式

```
skills/headphone-eq/
├── SKILL.md              判据文档：闸门阈值、拒绝策略、速度契约、拟合四阶段
├── README.md             依赖、数据来源、性能表、已修缺陷表
├── HANDOVER.md           本文档
├── data/
│   ├── harman_over_ear_2018.csv   AutoEq 官方目标（已校验一致）
│   ├── blue-dashed.csv            数字化的游戏参考曲线
│   └── headphones/                缓存的测量 + meta.json
├── scripts/
│   ├── generate.py       主命令：一次生成两份配置
│   ├── verify.py         从交付文件独立重算残差（能抓手工编辑）
│   ├── compare.py        两份配置并排对比 + 差异图
│   ├── selftest.py       不变量自检（无网络，约 3 s）
│   ├── fixtures.py       回归测试（10 个 fixture，需网络，约 8 s）
│   └── eqkit/            grid / curves / apo / metrics / analyse / fit / judge / emit / fetch / repo
├── fixtures/expected.json   冻结的回归期望值
└── out/                     生成产物（gitignore）
```

```bash
cd hifi-eq-g2pro/skills/headphone-eq
$env:PYTHONIOENCODING='utf-8'                                     # Windows 控制台必设

python scripts/generate.py --headphone "HIFIMAN Sundara"          # 按型号搜
python scripts/generate.py --uuid <uuid> --mode "2.4G模式"        # 指定曲线
python scripts/generate.py --uuid <uuid> --list-curves            # 只列曲线
python scripts/verify.py "out/<slug>/<name>-GAME.txt" --target game --uuid <uuid>
python scripts/compare.py out/<slug>
python scripts/selftest.py                                        # 自检 ~3 s
python scripts/fixtures.py                                        # 回归 ~9 s
python scripts/fixtures.py --record                               # 重冻结期望值
```

依赖：Python 3.9+、numpy、matplotlib。需可达 huihifi.com；代理用 `--proxy http://127.0.0.1:7897`。

---

## 3. 核心设计决策（为什么这么做）

1. **APO Q 形式搁架**：`LS/HS` 在 APO 里是 Q 形式而非 S 形式，同参数差最多 6 dB。`eqkit/apo.py` 按 `BiQuad.cpp` 精确转写并已向量化，所有报告数字都出自这个模型。
2. **可变宽度平滑**：5 kHz 以下 1/6 倍频程（数据可信），9 kHz 以上 3/4 倍频程（台架梳状伪影）。
3. **频带边缘淡出**：修正曲线在 16 kHz 以上淡出到 0 dB——数据之外「不变」才是诚实的说法。
4. **衰减优先**：修正曲线整体下移使峰值在 0 dB，多数配置无需 preamp。
5. **判据分离**：`FIT_GATES`（拟合器负责）与能力闸门分开；达不到时用 `unreachable_notes()` 说明差多少 dB、被哪条上限挡住。
6. **消融验证替代类别黑名单**：判断滤波器是否多余，唯一可靠的测试是「删掉它、重新优化其余、误差没变差才删」。类别规则曾删掉过承重滤波器（0.12 dB → 1.69 dB）。
7. **拟合四阶段**：`greedy`（定结构）→ `refine`（联合 LM）→ `prune`（唯一一次消融）→ `quantise`（舍入）。四者在 `fit.py` 内职责不重叠。

---

## 4. 本轮做完的事

### 4.1 【已解决】性能回归：139 s → **3.3 s**（原 §6.1）

先做测量再改代码。profile 结论：`refine` 在单个候选上被调用 18–58 次、累计占总时间约 99.9%；`drop_misplaced_shelves` 与 `compact` 各自内嵌一套完整消融循环，对同一份滤波器列表重复做了 4 次「删了再优化」的判定。

| 热点 | 原来 | 现在 |
|---|---|---|
| **`--jobs 0`（默认值）** | 被当成「0 个 worker」，整个扫描悄悄跑在单核上，文档却写着并行 | `_job_count()` 把 0/None/负数解释成「用满核心」，只有显式正数才固定；selftest 断言自动档不是串行 |
| 清理管线 | drop / interaction / treble / compact **四段**，各有独立消融循环 | 合并成**一次** `prune`，一个循环、一个预算（`COMPARE_ITERS=90`） |
| 每次删除的验证 | 对**每一个**候选都跑完整 `refine` | 闭式筛选用一次向量化调用给全部滤波器排队，每轮只对最便宜的 **3** 个做完整复核 |
| 雅可比 | 每列 2 次单滤波器级联求值 | **一次批量调用**算出全部列（数值完全相同，selftest 断言 <1e-12） |
| LM 迭代 | 固定跑到 140 步 | 连续 18 步无改进即停（`STALL_STEPS`），尾部纯属重复推导 |
| 角色并行 | 两个角色串行，各自开池 | 两角色 × 全部 count 合成**一条扁平队列**打满所有逻辑核心 |
| 工作进程数 | 固定上限 12 | `os.cpu_count()`（本机 24），`EQ_JOBS` 可覆盖 |

实测（同一台 i9-13980HX / 24 核）：

| 任务 | 旧 | 新 |
|---|---|---|
| `generate.py` 单型号双目标 | 139.1 s | **3.3 s** |
| `fixtures.py`（10 fixture / 20 拟合） | 挂死（嵌套池） | **9.2 s** |
| `selftest.py` | ~5 s | **2.9 s** |
| `verify.py` 单个交付配置 | — | 0.7 s |

现在的时间去向：取数 0.1 s、准备 0.5 s、拟合 1.9 s、出图出文件 0.8 s。拟合阶段已到临界路径（最慢的单个候选约 1.2 s），24 核上并行加速比饱和在 5 倍左右——再想快只能让单个候变更便宜，而不是加核。

### 4.2 【本轮发现并修复】两个会让结果静默出错的 bug

| 缺陷 | 症状 | 现在的防线 |
|---|---|---|
| `_run_tasks` 按 count 倒序派发，却按**原始顺序**装配结果 | 某个候选的滤波器被安到另一个 count 名下；hyperx-cloud2-stock/harman 上一条 1.5 dB 的链被选成「最佳」 | 派发时记下顺序，收回时按原槽位还原；selftest `parallel results keep their candidate order` |
| `fixtures.py` 在 worker 里再开进程池 | Windows 上嵌套池永远等不到结果，跑 12 分钟一行输出都没有 | `fit.in_worker()` 让 worker 走串行路径；fixtures 改成**一个进程负责一个 fixture**（两个 role 在里面用一次 `build_many`）；selftest `in_worker() is False in the parent process` |

> 这两个都是「看起来在跑、其实结果不对 / 不动」的类型，任何冒烟测试都抓不到。

### 4.3 回归数值：全部变好或持平（`fixtures/expected.json` 已重冻结）

| fixture | role | 滤波器 | RMS 旧 → 新 | 判定 |
|---|---|---|---|---|
| edifier-g2pro | harman / game | 12 / 11 | 0.544 → 0.530 / 0.627 → 0.583 | 变好 |
| hifiman-sundara | harman / game | 10 / 9 | 0.447 → 0.383 / 0.658 → 0.561 | 变好 |
| hyperx-cloud2-71 | harman / game | 11 / 12 | 0.791 → 0.794 / 0.751 → 0.754 | 持平（±0.003） |
| hyperx-cloud2-stock | harman / game | 10 / 10 | **0.779 → 0.306** / 0.601 → 0.600 | 显著变好 |
| moondrop-blessing3l | harman / game | 7 / 7 | 0.149 → 0.150 / 0.324 → 0.316 | 持平 |

hyperx-cloud2-stock/harman 的 −0.47 dB 是**真实改善而非过拟合**，已用「删掉每个滤波器后同预算重优化」逐条验证：10 个滤波器全部承重（最便宜的一个也要 +0.087 dB）。旧管线的 n=10 分支丢掉了第 10 个滤波器，新 `prune` 在容差内保留了它。判定方法已写进 SKILL.md 的 Regression 一节。

### 4.4 游戏目标曲线改为「从原图重新数字化」

`data/blue-dashed.csv` 从 1184 点、约 1/48 倍频程间距，改为 **1451 点、每条像素列一点**（中位 0.0066 倍频程）。而且现在它是**可复现的派生产物**：`scripts/digitise_target.py` 用新截图重跑，输出与已提交文件**逐字节一致**。

数字化方法的三个关键点，都是踩坑后写下的：

| 问题 | 后果 | 现在 |
|---|---|---|
| 用「把红线拟合到已有 CSV」来定坐标轴 | 红线按构造完美吻合，坐标轴误差被标定系数吸收，无从发现 | 坐标轴**只**来自图上自己印的刻度标签（11 个 dB 标签，残差 0.19 dB）；红线只用来**独立验证**（0.55 dB 均值） |
| 裁剪后的**局部列号**当成**全局列号** | 整条频率轴平移 79 px，红线自检从 0.55 dB 恶化到 3.3 dB | 列号一律相对绘图区宽度；`col_to_hz(c, width)` 显式传宽度 |
| 把轨迹降采样到 1/48 倍频程的整齐网格 | 5.5 / 8.7 kHz 的窄尖峰只有几像素宽，采样点落在峰两侧，尖峰从 +8.9 dB 塌到 +8.4 dB，且**看不出异常** | 保留**每一条**追踪到的像素列；selftest 断言间距 < 1/24 倍频程且 8.6 kHz 尖峰仍 > 8 dB |

结论：原来那份 `blue-dashed.csv` **本来就是对的**（逐点差 ≤ 0.53 dB），本轮的真实收益是**把目标变成可复现、可自证的派生产物**，并给尖峰密度加了防线。GAME 配置因此重拟合为 12 段（原 11 段），RMS 0.60 dB 不变。

### 4.5 试过但**不该**做的优化（记录以免重复踩）

| 想法 | 实测结果 | 结论 |
|---|---|---|
| 把 `prune` 的比较预算从 `COMPARE_ITERS=90` 降到 60 / 45 / 30 / 20 | 20 个候选里分别有 7 / 9 / 10 / 13 个给出了**不同**的滤波器集合 | 违反「同预算比较」铁律，只省 ~3 s，不做 |
| 把 `_shape_bank` 改成批量 `responses_db` | 0.027 s → 0.020 s，且数值有 4.7e-10 的差异 | 收益远小于风险，只建一次的表，不做 |
| 加更多 worker（16 / 24） | 12 worker 4.07 s，24 worker 4.12 s，不再变快 | 已到临界路径，加核无效 |

### 4.6 其它清理

- 删除交接用的调试探针 `probe.py / probe2.py / probe3.py / probe4.py / prof.py`，以及 `e2e.log / gen.log`（内容已被本轮表格取代）。
- `fit.py` 里 `drop_misplaced_shelves / limit_interaction / compact / limit_treble_filters` 保留为 `prune` 的薄封装（selftest 断言它们的承诺，旧读者也能找到熟悉的入口），但**只有一份实现、一个预算**。
- `fixtures.py` 的 count 固定为 `COUNTS = (10, 12)` 常量，与冻结的期望值绑定，不再暴露成会被误改的参数。

---

## 5. 当前回归快照（全部 PASS，2026-11-03 重冻结）

| fixture | role | 滤波器数 | RMS |
|---|---|---|---|
| edifier-g2pro | harman / game | 12 / 11 | 0.53 / 0.58 |
| hifiman-sundara | harman / game | 10 / 9 | 0.38 / 0.55 |
| hyperx-cloud2-71 | harman / game | 11 / 12 | 0.79 / 0.75 |
| hyperx-cloud2-stock | harman / game | 10 / 11 | 0.31 / 0.55 |
| moondrop-blessing3l | harman / game | 7 / 7 | 0.15 / 0.30 |

`selftest` 全绿（含 9 项新增不变量），`fixtures` 0 regression，`verify.py` 对交付文件独立复算 PASS。GAME 侧 RMS 因目标曲线改为按像素列全分辨率数字化而普遍略有变化（`blue-dashed.csv` 从 1184 点增至 1451 点）。

---

## 6. 【铁律】未来任何「删除 / 简化」类优化必须遵守

1. **同预算比较**：任何比较两个滤波器集合的决策，两边必须用相同迭代预算（`COMPARE_ITERS` 是常量，不允许调用方各传各的）。曾经的教训：试算用 `iters/3`、基线用全预算，每次删除都显得免费，工作滤波器被逐个吃掉（moondrop 从 0.15 dB 崩到 1.01 dB，滤波器数塌到 1）。
2. **便宜地预测，昂贵地决定**：闭式估计只能用来给候选**排序**，永远不能用来**接受**一次删除。接受必须来自完整重优化。
3. **禁止嵌套进程池**：worker 里只走串行路径。`fit.in_worker()` 是唯一判据。
4. **参数化调度必须还原顺序**：任何「为了缩短关键路径」而重排的工作队列，收回结果时必须回到原槽位。
5. **「自动」必须真的自动**：任何 `0` / `None` 之类表示「自己决定」的默认值，都要在入口处解析成具体数值，并有一条测试断言它不等于最保守的那个分支。`--jobs 0` 这条隐性串行就是这么藏了一整轮交接的。
6. **并行加速比有上限**：本机 24 核但加速比饱和在 5 倍左右，因为最慢的单个候选就是临界路径。继续加 worker 无效，只能削峰。

---

## 7. 仍未处理（都不阻塞交付）

| 优先级 | 事项 | 说明 |
|---|---|---|
| 中 | AutoEq 回退链 | `fetch.autoeq_*` 已实现（需代理），但主流程「huihifi 找不到 → 问是否用 AutoEq」尚未接上，目前直接拒绝 |
| 低 | `compare.py` 覆盖 | 两份配置并排对比可用，但未纳入 selftest |
| 低 | 多曲线示例话术 | SKILL.md 已补示例问法（引全站标签、不替用户选），脚本行为本就完整 |
| 低 | 缓存复用 | `data/headphones/` 里的 `meta.json` 已写，但 `--uuid` 重跑仍会重新拉网络；本机 0.3 s，未优化 |

---

## 8. 快速上手（5 分钟版）

1. `cd hifi-eq-g2pro/skills/headphone-eq`
2. `$env:PYTHONIOENCODING='utf-8'`（Windows 控制台）
3. `python scripts/selftest.py` → 应全绿（约 3 s）
4. `python scripts/generate.py --uuid f7e7a082-1244-4894-829d-b7f7fb8b9720 --label G2Pro` → 生成两份配置（约 9 s）
5. `python scripts/fixtures.py` → 对比 `fixtures/expected.json`（约 8 s，应 0 regression）
6. 改了拟合器就先跑 3 和 5；改了滤波器形状逻辑再跑 `verify.py` 复算一遍交付文件

## 9. 延伸阅读

闸门数值依据、为什么用消融不用类别规则、速度契约——见 `SKILL.md`（含「Speed」「Structure」两节）；缺陷表与性能分解见 `README.md`。两者与代码同步更新过，读它们比读 git 历史快。
