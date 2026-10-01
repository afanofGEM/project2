# Fusion Ablation - Planner Dev

本报告离线复用 Planner / Oracle 已保存的每个 Query Reranker Top20，
不重新调用模型，也不运行 locked test。

## llm_planner

| Strategy | Complete@5 | Recall@5 | MRR@5 | Net Complete vs Current |
|---|---:|---:|---:|---:|
| current_round_robin | 75.0% | 89.2% | 0.925 | +0 |
| rrf | 60.0% | 78.3% | 0.646 | -3 |
| weighted_rrf | 55.0% | 75.8% | 0.643 | -4 |
| distribution_based | 85.0% | 92.5% | 0.917 | +2 |
| distribution_based_original_anchor | 85.0% | 92.5% | 0.950 | +2 |
| coverage_quota | 75.0% | 89.2% | 0.900 | +0 |

候选：`distribution_based_original_anchor`。

## oracle

| Strategy | Complete@5 | Recall@5 | MRR@5 | Net Complete vs Current |
|---|---:|---:|---:|---:|
| current_round_robin | 85.0% | 92.5% | 0.917 | +0 |
| rrf | 35.0% | 55.8% | 0.525 | -10 |
| weighted_rrf | 30.0% | 45.0% | 0.479 | -11 |
| distribution_based | 75.0% | 87.5% | 0.892 | -2 |
| distribution_based_original_anchor | 80.0% | 90.0% | 0.950 | -1 |
| coverage_quota | 85.0% | 92.5% | 0.904 | +0 |

候选：`current_round_robin`。

## Natural v1 回归

锚点策略由已保存的 distribution-based 排名确定性重排得到，
没有重新调用模型。

| Strategy | Complete@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| current_round_robin | 96.0% | 98.0% | 0.799 |
| distribution_based | 98.0% | 99.0% | 0.779 |
| distribution_based_original_anchor | 98.0% | 99.0% | 0.824 |

## 决策边界

只将开发集最优策略作为候选；先做代码级集成和 natural_v1 回归，再冻结方案并运行 locked test。

本结果来自 dev，不得用 locked test 继续选策略。
