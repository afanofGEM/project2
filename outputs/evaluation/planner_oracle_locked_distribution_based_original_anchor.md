# Planner / Oracle Evaluation

- 数据集：`bank_planner_test_locked_v1`
- Split：`test_locked`
- 成功样本：10
- 失败样本：0
- Candidate K：20
- Final K：5
- Fusion：`distribution_based_original_anchor`

## 三组对照

| Method | Complete@5 | Recall@5 | MRR@5 | Avg Queries | Avg Latency |
|---|---:|---:|---:|---:|---:|
| original | 80.0% | 86.7% | 0.850 | 1.00 | 2.048s |
| llm_planner | 80.0% | 91.7% | 0.883 | 3.60 | 19.491s |
| oracle | 80.0% | 91.7% | 0.883 | 3.00 | 5.808s |

## 诊断

- `original` Fusion 前缺证据：[]
- `original` Fusion 前已找到、Top5 后丢失：['planner_test_006', 'planner_test_009']
- `llm_planner` Fusion 前缺证据：[]
- `llm_planner` Fusion 前已找到、Top5 后丢失：['planner_test_006', 'planner_test_009']
- `oracle` Fusion 前缺证据：[]
- `oracle` Fusion 前已找到、Top5 后丢失：['planner_test_006', 'planner_test_009']

## 当前决策

- Outcome：`no_go_oracle_no_complete_gain`
- 建议：锁定集上 Oracle Complete@5 未超过 Original；停止扩展 Planner 链路，不进入 Function Calling，也不得针对 locked case 继续调参。


> 这是锁定集最终验收结果，只能用于 go/no-go；
> 不得根据失败案例继续调整 Prompt、Fusion 或参数。
