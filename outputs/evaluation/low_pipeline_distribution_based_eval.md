# Minimal vs Low Pipeline Evaluation

## 评测设置

- 数据集：`data\evaluation\bank_eval_natural_v1.json`
- 成功配对样本：50 / 50
- 多证据样本：7
- 每个 Query Task 的候选数：Top20
- Query Planner：`Qwen/Qwen2.5-1.5B-Instruct`
- Reranker：`BAAI/bge-reranker-base`
- Fusion：`distribution_based`
- Minimal：只使用 Original Query 检索并重排。
- Low：使用 Original Query 与子问题分别检索重排，再合并去重。
- K 只用于离线评价，不会截断或修改生产 Pipeline。

## 排名指标

| K | 指标 | Minimal | Low | Low - Minimal |
|---:|---|---:|---:|---:|
| 1 | Hit@K | 70.0% | 62.0% | -8.0% |
| 1 | Mean Recall@K | 65.0% | 58.0% | -7.0% |
| 1 | Complete@K | 60.0% | 54.0% | -6.0% |
| 1 | MRR@K | 70.0% | 62.0% | -8.0% |
| 1 | Multi-Complete@K | 0.0% | 0.0% | 0.0% |
| 3 | Hit@K | 94.0% | 92.0% | -2.0% |
| 3 | Mean Recall@K | 90.0% | 88.0% | -2.0% |
| 3 | Complete@K | 86.0% | 84.0% | -2.0% |
| 3 | MRR@K | 81.0% | 76.0% | -5.0% |
| 3 | Multi-Complete@K | 42.9% | 42.9% | 0.0% |
| 5 | Hit@K | 100.0% | 100.0% | 0.0% |
| 5 | Mean Recall@K | 98.0% | 99.0% | 1.0% |
| 5 | Complete@K | 96.0% | 98.0% | 2.0% |
| 5 | MRR@K | 82.4% | 77.9% | -4.5% |
| 5 | Multi-Complete@K | 71.4% | 85.7% | 14.3% |
| 10 | Hit@K | 100.0% | 100.0% | 0.0% |
| 10 | Mean Recall@K | 100.0% | 100.0% | 0.0% |
| 10 | Complete@K | 100.0% | 100.0% | 0.0% |
| 10 | MRR@K | 82.4% | 77.9% | -4.5% |
| 10 | Multi-Complete@K | 100.0% | 100.0% | 0.0% |
| 20 | Hit@K | 100.0% | 100.0% | 0.0% |
| 20 | Mean Recall@K | 100.0% | 100.0% | 0.0% |
| 20 | Complete@K | 100.0% | 100.0% | 0.0% |
| 20 | MRR@K | 82.4% | 77.9% | -4.5% |
| 20 | Multi-Complete@K | 100.0% | 100.0% | 0.0% |

## 运行开销

下表为每条样本的平均值。延迟不包含模型初始化时间。

| 指标 | Minimal | Low |
|---|---:|---:|
| Query 数量 | 1.00 | 2.12 |
| 检索候选总数 | 20.00 | 42.40 |
| 最终去重候选数 | 20.00 | 25.96 |
| 合并去重数 | 0.00 | 16.44 |
| Reranker Pair 数量 | 20.00 | 42.40 |
| Planner 延迟（秒） | avg 0.000, P95 0.000 | avg 7.652, P95 10.501 |
| Retrieval 延迟（秒） | avg 0.008, P95 0.008 | avg 0.014, P95 0.022 |
| Reranker 延迟（秒） | avg 1.669, P95 1.985 | avg 3.472, P95 5.125 |
| 总延迟（秒） | avg 1.677, P95 1.992 | avg 11.138, P95 15.515 |

## 拉回与回退

| K | Hit 拉回 | Hit 回退 | Complete 拉回 | Complete 回退 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 5 | 1 | 4 |
| 3 | 0 | 1 | 0 | 1 |
| 5 | 0 | 0 | 1 | 0 |
| 10 | 0 | 0 | 0 | 0 |
| 20 | 0 | 0 | 0 | 0 |

## Low Top5 不完整案例

| Case | Evidence Mode | Minimal 正确证据排名 | Low 正确证据排名 |
|---|---|---|---|
| bank_eval_028 | all_required | `01_个人账户综合服务协议.pdf_13`: 1<br>`01_个人账户综合服务协议.pdf_14`: 6 | `01_个人账户综合服务协议.pdf_13`: 1<br>`01_个人账户综合服务协议.pdf_14`: 8 |
