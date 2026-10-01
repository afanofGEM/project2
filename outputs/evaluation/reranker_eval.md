# Reranker Evaluation Record

## 评测设置

- Hybrid 固定候选数：Top20
- Reranker 最终输出：Top5
- Reranker 模型：`BAAI/bge-reranker-base`
- 候选来源：现有 Retriever 评测保存的 Hybrid Top20

## Top5 前后对比

| 指标 | Hybrid Top5 | Reranker Top5 | 变化 |
|---|---:|---:|---:|
| Hit@5 | 94.0% | 100.0% | +6.0% |
| Mean Recall@5 | 93.0% | 98.0% | +5.0% |
| Complete@5 | 92.0% | 96.0% | +4.0% |
| MRR@5 | 0.753 | 0.824 | +0.071 |
| Multi-Complete@5 | 71.4% | 71.4% | +0.0% |

## 拉回与破坏情况

- 可拉回案例：4 个
- 成功完整拉回：4 个，案例拉回率 100.0%
- 位于 Hybrid 第6～20名的正确 chunk：5 个
- 成功进入 Reranker Top5：5 个，chunk 拉回率 100.0%
- 原本 Complete@5 的案例：46 个
- 被 Reranker 破坏完整性的案例：2 个，破坏率 4.3%

## 为什么多依据问题可能在重排后回退

Cross Encoder 会把同一个 query 分别与每个候选 chunk 组成输入，
并为每一对输入独立计算一个相关性分数：

```text
query + chunk1 -> score1
query + chunk2 -> score2
query + chunk3 -> score3
```

最后按照单个 chunk 的分数从高到低排序。计算 `chunk2` 分数时，
模型不知道 `chunk1` 是否已经进入 Top5，也不知道 `chunk1` 已经
覆盖了问题中的哪一部分。因此，它优化的是单个 query-chunk 对的
相关性，而不是 Top5 证据集合对整个问题的覆盖完整性。

例如一个问题同时包含 A、B 两个子问题，`chunk_A` 只回答 A，
`chunk_B` 只回答 B。即使 `chunk_A` 已经被排进 Top5，
Cross Encoder 也不会因此给 `chunk_B` 增加分数。其他几个与 A
高度相似但内容重复的 chunk，仍可能排在 `chunk_B` 前面，最终形成
“Top5 中有多份证据都回答 A，但缺少回答 B 的证据”的结果。

本轮两个回退案例正是这种情况：

- `bank_eval_028` 的一个正确 chunk 保持第1名，另一个从第2名
  降到第6名，导致“不执行指令的其他合同条件”没有进入 Top5。
- `bank_eval_050` 的一个正确 chunk 从第4名升到第1名，另一个从
  第3名降到第7名，导致“非本人交易的查询、投诉和报案流程”没有
  进入 Top5。

这也解释了为什么 `Hit@5` 可以达到100%，但 `Complete@5` 只有
96%：`Hit@5` 只要求至少命中一份正确证据，而 `Complete@5` 要求
多依据问题所需的全部正确证据都进入 Top5。对于最终需要综合多份
资料回答的问题，`Complete@5` 和 `Multi-Complete@5` 比单独观察
`Hit@5` 更重要。

因此，这两个回退案例不属于 Retriever 没有找回证据，因为正确
证据已经存在于 Hybrid Top20；它们反映的是独立相关性排序与证据
集合覆盖目标之间的差异。后续可通过多意图问题扩充、子问题拆分、
覆盖感知的证据选择，或 Top5 与 Top7 的生成效果对比继续验证。

## 可拉回案例明细

| Case | Hybrid正确证据排名 | Reranker正确证据排名 | 完整拉回 |
|---|---|---|---|
| bank_eval_005 | 07_手机号转账服务协议.pdf_3: 3, 07_手机号转账服务协议.pdf_5: 7 | 07_手机号转账服务协议.pdf_3: 5, 07_手机号转账服务协议.pdf_5: 2 | 是 |
| bank_eval_012 | 07_手机号转账服务协议.pdf_5: 8, 07_手机号转账服务协议.pdf_6: 10 | 07_手机号转账服务协议.pdf_5: 2, 07_手机号转账服务协议.pdf_6: 5 | 是 |
| bank_eval_027 | 01_个人账户综合服务协议.pdf_8: 6 | 01_个人账户综合服务协议.pdf_8: 3 | 是 |
| bank_eval_047 | 08_个人自动转账业务协议.pdf_6: 11 | 08_个人自动转账业务协议.pdf_6: 3 | 是 |

## 回退案例

以下案例原本在 Hybrid Top5 中证据完整，但经过 Reranker 后不再完整：

- bank_eval_028
- bank_eval_050
