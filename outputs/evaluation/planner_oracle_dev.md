# Planner / Oracle Evaluation

- 数据集：`bank_planner_dev_v1`
- Split：`dev`
- 成功样本：20
- 失败样本：0
- Candidate K：20
- Final K：5

## 三组对照

| Method | Complete@5 | Recall@5 | MRR@5 | Avg Queries | Avg Latency |
|---|---:|---:|---:|---:|---:|
| original | 70.0% | 85.8% | 0.950 | 1.00 | 1.721s |
| llm_planner | 75.0% | 89.2% | 0.925 | 3.35 | 16.645s |
| oracle | 85.0% | 92.5% | 0.917 | 3.00 | 4.887s |

## 诊断

- `original` Fusion 前缺证据：[]
- `original` Fusion 前已找到、Top5 后丢失：['planner_dev_002', 'planner_dev_007', 'planner_dev_008', 'planner_dev_013', 'planner_dev_014', 'planner_dev_015']
- `llm_planner` Fusion 前缺证据：[]
- `llm_planner` Fusion 前已找到、Top5 后丢失：['planner_dev_007', 'planner_dev_008', 'planner_dev_009', 'planner_dev_013', 'planner_dev_015']
- `oracle` Fusion 前缺证据：[]
- `oracle` Fusion 前已找到、Top5 后丢失：['planner_dev_002', 'planner_dev_004', 'planner_dev_015']

## 当前决策

- Outcome：`fusion_bottleneck_before_planner_tuning`
- 建议：Oracle 已证明分解有上限收益，但 LLM/Oracle 均有证据在 Fusion 后掉出 Top5；先做 Fusion/Coverage 对照，再判断剩余差距是否来自 Planner。

> 这是开发集结果，可以用于定位和修改；锁定测试集尚未运行，不能作为最终验收结论。
