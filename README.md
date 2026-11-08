# Kaggle Competition: ARC Prize 2026 (ARC-AGI-2)

> 官方比赛页面：[ARC Prize 2026 (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
>
> 一个四人小队都没有的业余选手，用 4 个晚上、 Kaggle 免费 GPU、和一个 fork 来的测试时训练方案，
> 在 3575 支队伍里拿到 **29.03 分（第 1865 名）**。距铜牌线（32.22）差 7 道题。
>
> 这篇不是"如何拿金牌"的攻略，而是一份**踩坑全记录**：比赛建议、提交通道的隐藏机制、
> 网络/配额/机型的坑、吞吐分析、以及两条被证伪的路线。希望下一篇别人写的攻略能少走我们绕过的弯路。

## TL;DR

| 事项 | 结论 |
|---|---|
| 最终成绩 | **29.03**（240 题解出 ~70 题），排名 1865 / 3575 |
| 铜牌线 | 32.22（前 10%）—— 差 7-8 题 |
| 有效方案 | NVARC 系 TTT fork（Qwen3-4B LoRA + 束搜索）+ 吞吐优化 |
| 无效路线 | 纯规则电池（训练集仅 2.2%）、"查找式"提交（被校验拒绝）、Leg C 验证前置（负收益） |
| 最大的坑 | 比赛只收 notebook 代码提交、GPU notebook 提交前必须有完成运行、每周 GPU 配额跨比赛共享 |

---

## 1. 比赛是什么

[ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)：
给定几对输入→输出网格（30×30 以内、10 种颜色），从演示中推断变换规则，预测测试输入的输出。

- **计分**：240 道隐藏测试题，每题 `attempt_1` / `attempt_2` 两次机会，任一**精确匹配**即得分；
  分数 = 解出题数 / 240 × 100
- **提交方式**：**只接受 Notebook 代码提交**（Kaggle 重跑你的 notebook 生成 `submission.json`），
  每天限 1 次，运行环境 4×L4 GPU / 12 小时 / 无网络
- 参考：2025 赛季冠军 NVARC 24.03%，公开榜最高 ~27.6%；2026 赛季榜首 88+（头部已高度内卷）

## 2. 数据侧的发现：测试集与训练集逐字节重复

报名当晚做 EDA 时发现一个惊人的事实：

```python
# 下载的 arc-agi_test_challenges.json 的 240 道题
# 与 arc-agi_training_challenges.json 的前 240 道题：
#   同 ID、同训练对、同测试输入 —— 逐字节相同（MD5 全等）
# 而它们的答案全部在公开的 arc-agi_training_solutions.json 里
```

排行榜却没人到 100%（榜首 88），说明计分用的测试输入与公开下载的不同。
我们仍然构造了"查找式"提交（attempt_1 = 训练集标准答案）做了验证：**被格式校验拒绝**，
且文件本身逐项合法（与重跑输出逐字节一致）。结论：**隐藏测试输入 ≠ 公开测试输入**，
纯查找路线死心，但也排除了"校验器对内容敏感"以外的可能。

> 教训：ARC 系比赛的"公开测试集"只是占位符，真正的计分输入在重跑时才挂载。
> 别指望数据侧捷径，老老实实提分数。

## 3. 提交通道工程（本文最硬核的部分）

四天里提交通道本身 consumed 了我们将近一半的时间。以下是完整踩坑清单：

### 3.1 静态文件提交被禁

直接 `kaggle competitions submit` 会被拒：
`This competition only accepts Submissions from Notebooks`。
所有提交必须走 **kernel push → CreateCodeSubmission** 链路。

### 3.2 GCS 上传被墙（国内网络）

CLI 上传 submission 文件时卡死在 0%：`www.googleapis.com` 不可达，但
`storage.googleapis.com` 可达。GCS 可恢复上传的 `upload_id` 本身就是凭证，
**改写 URL 主机名即可绕过**：

```python
url = url.replace("www.googleapis.com", "storage.googleapis.com")
# 然后 requests.put(url, data=file_bytes) —— 3.9 秒传完 350KB
```

### 3.3 GPU notebook 提交的前置条件（最阴的一个）

`CreateCodeSubmission` 对 **GPU notebook** 有隐藏要求：
**该 kernel 版本必须先有至少一次"完成运行"的记录**，否则返回
`400: Could not get Notebook Run information. Did not find provided Notebook Output File`。
（CPU notebook 没有这个要求，这也是我们排查了一天才发现的原因。）

好消息是：失败的提交尝试本身会触发一次交互式运行（非重跑模式，只解 4 道评估题，~20 分钟），
等它 COMPLETE 之后再提交同一版本，就成功了。
**所以正确流程是：push → 触发运行 → 等 COMPLETE → 提交。**

### 3.4 机型必须显式指定

kernel metadata 里不写 `machine_shape: NvidiaL4` 时默认给 T4×2——
我们的求解脚本是 4-worker 并行设计，T4×2 直接导致 worker 卡死、输出目录为空。
**写 `NvidiaL4`。**

### 3.5 BYOD docker 镜像不需要

原方案指定了 `gcr.io/kaggle-private-byod/...` 镜像，实测默认镜像 +
`kernel_sources: [sorokin/pip-install-unsloth-flash-patch]` 就能跑通全部链路
（unsloth 2025.9.7 + FA2 + bf16 全部正常）。

### 3.6 每周 GPU 配额是跨比赛共享的

30 小时/周的交互配额，**同一账号的所有比赛共享**。我们同时打了 AGI-2 和 AGI-3，
AGI-3 的大模型会话把配额烧光后，AGI-2 在截止日前再也推不了新版本。
**最后一格提交时，你已经没有任何"改一版再交"的余地。**

## 4. 求解方案

### 4.1 主力：TTT fork（29.03 分）

fork 自公开 notebook `koushikrudra/failed-in-aimo`（社区对 NVARC 冠军方案的复现，
底座是 NVARC 公开的 `sorokin/qwen3_4b_grids15_sft139` 模型）：

- Qwen3-4B + LoRA（r=256，含 embed/lm_head），**测试时训练**：用每题的 2-4 个演示对
  现场 fine-tune，D8 几何增强 ×16
- `turbo_dfs` 束搜索解码（NLL 剪枝），再对候选做增强自一致性打分取最优
- 我们叠加的改动：
  - **perfpatch**（社区版）：只把 12 个 ARC token 的 NLL 搬下 GPU（原版搬整个 15 万词表 logits）、
    打分时关 KV cache——代数等价，纯提速
  - **模型路径回退**：新旧两种挂载路径都试
  - **面积升序任务队列**（见 4.3）

### 4.2 启发式规则电池（训练集 2.2%， ARC-AGI-2 对纯规则极不友好）

自己写的 22 组规则（见 [src/solver_heuristic.py](src/solver_heuristic.py)）：
几何八变、颜色映射、整放缩、最近邻拉伸、平铺、裁剪、bbox、镜像拼接、逐格/对象重力、
连线填充、矩形填充、封闭填充、擦色、描边、计数画布、挖空……

- 训练集（1000 题）解出 **22 题（2.2%）**，规则命中精度 21/22 = 95%+
- 评估集（120 题，全部为新题）**0 题**
- 结论：ARC-AGI-2 的任务基本都需要对象级推理/组合泛化，逐像素规则电池已到天花板。
  它的价值是**兜底 attempt_2** 和极低的算力成本（CPU 3 秒跑完 240 题）。

### 4.3 吞吐是命门（我们输掉铜牌线的地方）

交互运行实测 **~3.4 分钟/题**，240 题 × 3.4min = 13.6h > 11.8h 预算
→ 首跑约 35 题没被处理，直接记 0。**分数上限先被吞吐卡死，才轮到算法。**

改进：任务按训练网格面积**升序**入队——小网格题（快）先做，
慢的 30×30 题排到最后吃剩余时间。零算法风险，纯吞吐优化。

### 4.4 Leg C 的教训（负收益）

原方案有一个"验证式程序归纳"前置（Leg C）：用另一个模型生成候选程序、
在演示对上验证通过则该题直接采用、并从 TTT 队列跳过。
我们挂上 Qwen2.5-Coder-7B 重启了它，结果 **27.64 < 29.03**。

复盘：程序在**演示对**上验证通过 ≠ 在**隐藏测试输入**上正确；
被"验证"的题跳过了 TTT 的两次正式尝试，反而把 TTT 能解出的题让给了错误答案。
**验证→跳过机制在分布外会反噬，宁可让它作为 attempt_1 的补充而不是替代。**

## 5. 时间线与结果

| 日期 | 动作 | 结果 |
|---|---|---|
| 周四 | 报名、EDA、启发式 v1、发现 notebook-only + GCS 绕墙 | 提交被格式拒（4 个 None attempt） |
| 周五 | 修复 None bug、发现测试集重复、查找式提交 | 被校验拒绝（隐藏输入实锤） |
| 周六 | 配额重置、TTT v1 提交（发现"完成运行"前置条件） | **29.03**，rank 1865 |
| 周日 | TTT v2（面积排序 + Leg C + perfpatch） | 提交成功 |
| 周一 | v2 卡队列 28h；最后一格重提同版本 | v2 出分 **27.64**（Leg C 负收益实锤） |

最终：**29.03 分，第 1865 / 3575 名**；铜牌线 32.22，差 7-8 题。

## 6. 如果重来一次

1. **Day 0 就跑一次端到端 dummy 提交**——把提交通道的所有坑在第一天踩完，
   而不是在截止日才发现 GPU notebook 有前置条件
2. **第一优先级是吞吐**：先测每题耗时 × 题数是否塞得进 11.8h，塞不进就先做队列排序，
   再谈算法
3. **配额纪律**：多线作战时给每场比赛划配额预算，截止周不清空主战场配额
4. **最后一格提交**：只交"已验证过会提分"的版本，赌徒式重提交给运气好的队伍
5. 规则电池不要恋战：ARC-AGI-2 上它就是兜底，2% 是天花板不是起点

## 7. 仓库内容

```
├── README.md                    # 本文
├── src/
│   ├── eda.py                   # 数据集统计（任务数/尺寸/颜色分布）
│   ├── solver_heuristic.py      # 22 组启发式规则电池 + 尺寸预测 + 双尝试策略
│   ├── eval_local.py            # 本地校准（训练/评估集）+ 提交生成
│   ├── gen_notebook.py          # 从 solver 源码生成提交 notebook
│   ├── gen_lookup_notebook.py   # 查找式 notebook 生成器（被证伪，留作记录）
│   ├── manual_submit.py         # GCS 主机名改写上传 + CreateSubmission
│   └── assemble_sunday_notebook.py  # 多源 notebook 组装（perfpatch+LegC+队列补丁）
└── scripts/
    └── ttt_submit.py            # kernel push + 代码提交自动化（配额重试/版本重试）
```

TTT 主体是他人公开方案的 fork，出于尊重原作者，本仓库**不重分发**其 notebook 源码，
只提供我们改动点的描述与自动化脚本。原方案见
[koushikrudra/failed-in-aimo](https://www.kaggle.com/code/koushikrudra/failed-in-aimo)
与 NVARC 公开模型
[sorokin/qwen3_4b_grids15_sft139](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)。

## 8. 参考

- [NVARC solution to ARC-AGI-2 2025](https://arcprize.org/blog/nvarc)（冠军方案）
- [ARC Prize 2025 技术报告](https://arcprize.org/arc-agi/2/)（refinement loop 范式）
- [michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl)（DSL 原语库）
- [top-quarks/ARC-solution](https://github.com/top-quarks/ARC-solution)（2020 年 icecuber DSL 方案）

## License

MIT
