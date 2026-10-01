# Planner / Oracle 专项评测

## 为什么需要这项评测

自然集 `bank_eval_natural_v1` 的 Minimal Pipeline 已经达到
`Complete@5 = 96%`。继续在自然集上调 Planner，既难以暴露多证据问题，
也容易把开发过程变成针对旧测试集调参。

因此当前实验只回答一个问题：

> 对真正需要多证据覆盖的复杂 Query，Query Decomposition 是否存在上限收益；
> 如果存在，当前损失发生在 Planner、Retriever/Reranker 还是 Fusion？

## 数据集边界

- `bank_eval_natural_v1.json`：61 条冻结自然集，只用于回归。
- `planner_dev_v1.json`：20 条复杂问题，允许分析和改进。
- `planner_test_locked_v1.json`：10 条锁定问题，不参与调参。

开发集与锁定集没有 Query 重复，也没有必需证据 Chunk 重叠。每个复杂样本都保留：

- `required_evidence`
- `expected_points`
- `gold_subqueries`
- `source_case_ids`

`source_case_ids` 用于追溯组合问题中的事实和标签，不代表直接复制原 Query。

## 三组严格对照

三组使用同一份 Chunk、Embedding、Hybrid Retriever、Cross Encoder、
`candidate_k=20` 和 `final_k=5`：

1. `original`：只使用原始复杂问题。
2. `llm_planner`：原始问题加当前 Qwen Planner 生成的子查询。
3. `oracle`：原始问题加人工 Gold Subqueries。

评测记录每个 Query 的 Hybrid Top20、Reranker Top20、Fusion 后排名、
Complete@K、MRR 和延迟。

## 运行命令

在项目根目录执行：

```powershell
$p2Python = "D:\Miniconda3\envs\project1\python.exe"
& $p2Python src\eval\evaluate_planner_oracle.py --dataset dev
```

只根据已有逐案结果重建 Summary 和 Markdown：

```powershell
& $p2Python src\eval\evaluate_planner_oracle.py --dataset dev --report-only
```

锁定集有显式保护。只有开发方案、Fusion 和参数全部冻结后，才允许执行：

```powershell
& $p2Python src\eval\evaluate_planner_oracle.py `
    --dataset locked `
    --allow-locked-evaluation `
    --fusion-strategy distribution_based_original_anchor
```

不能使用锁定集结果继续修改 Prompt、Fusion 或参数。

## Round-robin 开发集基线

| Method | Complete@5 | Recall@5 | MRR@5 | Avg Queries | Avg Latency |
|---|---:|---:|---:|---:|---:|
| Original | 70% | 85.8% | 0.950 | 1.00 | 1.721s |
| LLM Planner | 75% | 89.2% | 0.925 | 3.35 | 16.645s |
| Oracle | 85% | 92.5% | 0.917 | 3.00 | 4.887s |

Oracle 相对 Original 净增加 3 条 Complete@5；LLM Planner 净增加 1 条。
这说明 Query Decomposition 在复杂问题上存在上限收益，但当前 LLM 尚未达到
Oracle 上限。

同时，三组方法的必需证据都曾进入各自的 Top20。LLM Planner 有 5 条、Oracle
有 3 条样本在 Fusion 后把必需证据挤出 Top5。因此当前优先级是：

1. 对照 Fusion / Coverage 策略。
2. 再判断剩余差距是否来自 Planner 拆分质量。
3. 开发方案冻结后才运行 locked test。
4. locked test 验证存在净收益后，才进入 Function Calling 路由。

## Fusion 开发集消融

在同一批已保存的 Query Reranker Top20 上离线比较了 round-robin、RRF、
加权 RRF、distribution-based 和 coverage quota。RRF 类方法明显下降；最佳
LLM Planner 候选为 `distribution_based_original_anchor`：先按每个 Query 的
分数分布归一化，重复 Chunk 取最高归一化分数，再保留 Original Query Top1
作为安全锚点。

| Planner Fusion | Complete@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Current round-robin | 75% | 89.2% | 0.925 |
| Distribution-based + Original Top1 | 85% | 92.5% | 0.950 |

开发集相对 current 净拉回 2 条 Complete@5，因此进入自然集回归。

## Natural v1 回归

纯 distribution-based 的 50 条真实运行全部成功。安全锚点结果是对该运行的
已保存排名做确定性重排，不重新调用模型；相关逻辑另有单元测试覆盖。

| Low Fusion | Complete@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Current round-robin | 96% | 98% | 0.799 |
| Distribution-based | 98% | 99% | 0.779 |
| Distribution-based + Original Top1 | 98% | 99% | 0.824 |

安全锚点消除了纯分布归一化带来的首条证据排名下降，因此候选在 locked test
前被冻结；生产默认值仍保持 `round_robin`。

## Locked test 最终验收

锁定集只运行一次，不再用于调参。10 条样本全部执行成功。

| Method | Complete@5 | Recall@5 | MRR@5 | Avg Latency |
|---|---:|---:|---:|---:|
| Original | 80% | 86.7% | 0.850 | 2.048s |
| LLM Planner | 80% | 91.7% | 0.883 | 19.491s |
| Oracle | 80% | 91.7% | 0.883 | 5.808s |

`planner_test_006` 和 `planner_test_009` 的必需证据均曾进入各 Query Top20，
但在 Fusion 后掉出 Top5。LLM Planner 和 Oracle 的 Complete@5 都没有超过
Original，且 LLM Planner 平均延迟约为 Original 的 9.5 倍。

最终结论是 `no_go`：保留实验代码和可复现报告，但不把候选设为生产默认，
不进入 Function Calling，也不继续针对 locked test 调 Fusion 或 Prompt。
本轮结果同样不构成引入 LangGraph、MCP、LoRA 或 vLLM 的依据。

## 工程依据

- [Pyserini](https://github.com/castorini/pyserini) 将固定 topics、qrels、索引和评测脚本作为可复现实验的核心。
- [LangSmith Evaluation](https://docs.langchain.com/langsmith/evaluation-types) 将固定数据集上的离线评测用于版本比较和回归检查。
