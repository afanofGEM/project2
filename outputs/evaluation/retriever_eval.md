# Retrieval Evaluation Record

## 当前测试结果

本轮使用 50 条可回答问题，对 Vector、BM25 和 Hybrid Retriever 进行了检索评测。

| Retriever | Hit@5 | Complete@5 | Complete@10 | Complete@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Vector | 84% | 80% | 98% | 100% | 0.748 |
| BM25 | 90% | 88% | 92% | 98% | 0.734 |
| Hybrid | **94%** | **92%** | **98%** | **100%** | **0.761** |

## 当前结论

Hybrid Retriever 有必要保留。它通过 RRF 融合 Vector 的语义检索能力和 BM25 的关键词匹配能力，整体表现比单独使用其中一种检索方法更加均衡。

Hybrid 和 Vector 的 `Complete@20` 都达到了 100%，说明两者都能在 Top20 候选中完整找回当前测试集所需的全部证据。但是，Hybrid 的优势体现在正确证据的前排质量更好：

- **Hybrid 的 `Hit@5` 为 94%，高于 Vector 的 84%**。
- **Hybrid 的 `Complete@5` 为 92%，高于 Vector 的 80%**。
- Hybrid 的 `MRR@20` 为 0.761，高于 Vector 的 0.748，说明第一个正确证据通常排名更靠前。
- Hybrid 同时利用语义匹配和关键词匹配，对不同表达方式的问题更稳健。

因此，当前项目采用以下流程是合理的：

```text
Query
  ↓
Vector Retrieval + BM25 Retrieval
  ↓
RRF Hybrid Retrieval Top20
  ↓
Cross Encoder Reranker
  ↓
Top5 Evidence
```

选择 Top20 交给 Reranker，是因为当前 Hybrid 的 `Complete@20 = 100%`，所有必要证据都已经进入候选集合。Reranker 不负责重新搜索缺失证据，而是从候选集合中进行更精细的相关性判断和重新排序。

后续评测需要验证：Reranker 能否将排名第 6～20 位的正确证据提升到 Top5，同时不破坏原本已经排在前面的正确结果。

## 后续补充

- Reranker 评测结果
- Rerank 前后 Top5 指标对比
- 多依据问题表现
- 失败案例分析
- Generation 和端到端评测结果
