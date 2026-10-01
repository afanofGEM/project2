# 银行业务 RAG：基于 Microsoft Agentic Retrieval 的升级记录

> 开始日期：2026-09-24  
> 架构路线确认日期：2026-09-25  
> 当前状态：Microsoft Minimal 基线已验证，准备进入 Low 模式的 Query Planner 注入阶段  
> 改造原则：一次只改一个阶段，先独立评价，再接入后续阶段；不根据测试集个例硬编码固定 Top-K 或专用规则。

## 1. 项目当前能力

项目已经具备：

- PDF Document Loading
- Chunking 与 Metadata
- Embedding
- FAISS Dense Retrieval
- BM25 Sparse Retrieval
- RRF Hybrid Retrieval
- Cross Encoder Reranker
- Citation-based Generation
- Retriever / Reranker / Query Planner Evaluation

当前基础流程：

```text
Query
  ↓
Hybrid Retrieval
  ↓
Cross Encoder Reranker
  ↓
Evidence Guard
  ↓
Prompt Construction
  ↓
Qwen Generation + Citation
```

当前主要问题不是“完全检索不到正确证据”，而是复合问题需要多份证据时，独立的 Query-Chunk 排序不能保证最终证据集合覆盖所有询问目标。

## 2. 已完成评测与指标

### 2.1 Retriever 评测

评测数据：50 条可回答问题。  
详细报告：[`outputs/evaluation/retriever_eval.md`](outputs/evaluation/retriever_eval.md)

| Retriever | Hit@5 | Complete@5 | Complete@10 | Complete@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Vector | 84% | 80% | 98% | 100% | 0.748 |
| BM25 | 90% | 88% | 92% | 98% | 0.734 |
| Hybrid | **94%** | **92%** | **98%** | **100%** | **0.761** |

结论：

- Hybrid Retriever 有必要保留。
- Hybrid `Complete@20 = 100%`，说明当前测试集中需要的证据都能进入 Top20 候选。
- Top5 不完整的主要问题发生在候选排序和多证据覆盖阶段。

### 2.2 Reranker 评测

评测设置：Hybrid Top20 → `BAAI/bge-reranker-base` → Top5。  
详细报告：[`outputs/evaluation/reranker_eval.md`](outputs/evaluation/reranker_eval.md)

| 指标 | Hybrid Top5 | Reranker Top5 | 变化 |
|---|---:|---:|---:|
| Hit@5 | 94.0% | 100.0% | +6.0% |
| Mean Recall@5 | 93.0% | 98.0% | +5.0% |
| Complete@5 | 92.0% | 96.0% | +4.0% |
| MRR@5 | 0.753 | 0.824 | +0.071 |
| Multi-Complete@5 | 71.4% | 71.4% | +0.0% |

补充结果：

- 4 个可拉回 Case 全部成功拉回，Case Rescue Rate 为 100%。
- 5 个位于 Hybrid 第 6～20 名的正确 Chunk 全部进入 Reranker Top5。
- **46 个原本 Complete@5 的 Case 中有 2 个发生回退，破坏率为 4.3%。**
- 回退 Case：`bank_eval_028`、`bank_eval_050`。

原因：Cross Encoder 分别计算每一个 Query-Chunk Pair：

```text
query + chunk1 → score1
query + chunk2 → score2
query + chunk3 → score3
```

**它不知道已经选中的 Chunk 覆盖了哪个询问目标，也不知道当前证据集合还缺什么。因此，单个 Chunk 的相关性排序提高，不等于多证据集合的覆盖率提高。**

### 2.3 Query Planner 最新评测

评测数据：`data/evaluation/bank_rag_component_cases.json` 中 61 条 Query。  
最新结果：[`outputs/evaluation/query_planner_eval_results.json`](outputs/evaluation/query_planner_eval_results.json)

| 指标 | 最新结果 |
|---|---:|
| Total Cases | 61 |
| Execution Success | 61 |
| Valid Plan | 61 |
| Runtime Error | 0 |
| Fallback | 0 |
| Simple | 41 |
| Complex | 20 |
| Average Subquery Count | 1.393 |
| Minimum Subquery Count | 1 |
| Maximum Subquery Count | 4 |
| Average Planner Latency | 6.325 秒 |

子问题数量分布：

| 子问题数量 | Case 数量 |
|---:|---:|
| 1 | 41 |
| 2 | 18 |
| 3 | 0 |
| 4 | 2 |

与上一轮固定最多 3 个子问题的历史结果相比：

| 指标 | 固定最多 3 个 | 不限制固定数量 |
|---|---:|---:|
| Average Subquery Count | 1.344 | 1.393 |
| Maximum Subquery Count | 3 | 4 |
| Average Planner Latency | 5.623 秒 | 6.325 秒 |

当前观察：

- `bank_eval_026` 正确生成 4 个子问题，修复了固定最多 3 个时的信息遗漏。
- `bank_eval_019` 产生了错误的 2×2 组合，说明取消数量上限后仍可能过度拆分。
- `bank_eval_033` 仍生成“我应该做什么？”，不满足独立可检索要求。
- `bank_eval_044`、`bank_eval_050`、`bank_eval_054` 的部分子问题缺少完整业务对象。
- `bank_eval_053` 的比较型子问题缺少明确比较对象。

重要解释：`Valid Plan = 61` 只代表 JSON 与字段结构校验通过，不代表 61 条 Query Plan 在语义上全部正确。

## 3. 统一架构决策

### 3.1 选择 Microsoft，不混用 NVIDIA

本项目从 2026-09-25 起统一参考 Microsoft Azure Agentic Retrieval：

- Query Planner 生成 focused subqueries。
- 多个查询分别执行 Hybrid Retrieval。
- 每个查询的候选分别进行 Semantic Reranking。
- 合并、去重并保留检索来源。
- 检查现有证据是否充分。
- 证据不足时，只使用修订后的查询补充检索一次。
- 合并最终证据并生成带引用答案。

不采用 NVIDIA 的递归多跳流程。原因是当前银行测试集以并列、多方面问题为主，很少需要“先回答 A，才能生成 B”的连续推理。递归中间答案和多轮递归状态会增加当前项目不需要的复杂度。

工程参考：

- [Azure Agentic Retrieval Overview](https://learn.microsoft.com/azure/search/agentic-retrieval-overview)
- [Azure Retrieval Reasoning Effort](https://learn.microsoft.com/azure/search/agentic-retrieval-how-to-set-retrieval-reasoning-effort)
- [Azure RAG Information Retrieval](https://learn.microsoft.com/azure/architecture/ai-ml/guide/rag/rag-information-retrieval)

### 3.2 不再要求 Query Planner 单独保证最终正确性

Query Planner 的目标是提高证据召回率，不是生成语言上绝对完美的子问题。

后续不再围绕每一条评价 Bad Case 不断增加专用 Prompt 规则。Prompt 只保留通用要求：

- 子问题完整、独立、可检索。
- 子问题合计覆盖原问题。
- 不增加原问题没有的意图。
- 使用完成检索所需的最少数量。
- 保留业务对象、条件和询问目标之间的对应关系。

Planner 多生成一个无效查询通常只增加成本；Planner 漏掉一个必要查询则可能导致证据无法召回。因此，Planner 阶段优先保证覆盖率，最终质量由完整检索链路评价。

## 4. Microsoft 三档模式在本项目中的映射

### 4.1 Minimal：单查询基线

```text
Original Query
  ↓
Hybrid Retrieval
  ↓
Cross Encoder Reranker
  ↓
Answer
```

状态：已完成 Retriever 与 Reranker 独立评价，是后续实验的基线组。

### 4.2 Low：单轮多查询检索

```text
Original Query
  ↓
Query Planner
  ↓
Original Query + Focused Subqueries
  ↓
每个 Query 分别执行 Hybrid Retrieval
  ↓
每个 Query 的候选分别执行 Cross Encoder Reranking
  ↓
按 Chunk ID 合并、去重并保留命中来源
  ↓
构造统一 Grounding Context
  ↓
Answer
```

状态：Query Planner 已完成；Multi-query Retrieval、Per-query Reranking 和 Result Merge 尚未实现。

说明：保留 Original Query 是本项目的安全保底。即使某个子问题不完整，原问题仍有机会召回正确证据。

### 4.3 Medium：证据不足时补检索一次

```text
Low 第一轮结果
  ↓
Evidence Sufficiency Check
  ├─ sufficient = true  → Answer
  └─ sufficient = false
          ↓
      Revised / Follow-up Queries
          ↓
      第二轮 Retrieval + Reranking
          ↓
      Merge + Deduplication
          ↓
      Answer
```

状态：尚未实现。

限制：

- 最多补检索一次。
- 不设计无限 Agent 循环。
- 第二轮只处理缺失的信息目标。
- 重试次数、额外查询数和延迟必须进入 Trace。

## 5. 最终目标架构

```text
User Query
  ↓
Query Planner
  ↓
Query Execution Plan
  ↓
Parallel Hybrid Retrieval
  ↓
Per-query Cross Encoder Reranking
  ↓
Result Merge + Chunk Deduplication
  ↓
Evidence Sufficiency Check
  ↓
Optional One Follow-up Retrieval
  ↓
Grounding Context Construction
  ↓
Qwen Generation
  ↓
Citation Extraction
```

模块职责：

| 模块 | 职责 | 不负责 |
|---|---|---|
| Query Planner | 生成有助于检索的 focused subqueries | 保证最终答案正确 |
| Hybrid Retriever | 扩大 Dense + Sparse 候选召回 | 最终证据选择 |
| Cross Encoder Reranker | 计算单个 Query-Chunk 相关性 | 多证据集合覆盖 |
| Result Merger | 合并、去重、保留 Query 来源与分数 | 生成业务答案 |
| Sufficiency Checker | 判断证据是否覆盖原问题 | 无限次重试 |
| Generator | 仅根据最终证据生成带引用答案 | 补充资料之外的知识 |

## 6. 已完成代码改造

### 6.1 Reranker 职责拆分

文件：`src/reranker/reranker.py`

- `score_candidates()` 批量构造 Query-Chunk Pair。
- 为候选副本写入 `reranker_score`。
- 按相关性分数降序返回结果。
- Reranker 不负责多证据覆盖选择。

### 6.2 Pipeline 保存中间结果

文件：`src/pipeline/rag_pipeline.py`

当前函数：

```text
retriever_search()
rerank_candidates()
retriever_reranker()
run()
```

`run()` 当前保留：

```python
{
    "retrieved_chunks": retrieved_chunks,
    "reranked_chunks": reranked_chunks,
    "evidence_chunks": evidence_chunks
}
```

### 6.3 Query Planner

文件：

- `src/prompt/query_planner_prompt.py`
- `src/query_processing/query_planner.py`

已实现：

- Qwen 只输出 `subqueries`。
- 提取模型输出中的 JSON。
- 校验列表、字典、字符串和空值。
- 对完全相同的子问题去重。
- 由程序生成 `sq_1`、`sq_2` 等 ID。
- 根据最终子问题数量生成 `simple` / `complex`。
- 解析或结构失败时回退到原始 Query。
- `max_subqueries=None` 时不设置固定子问题数量限制。
- 评价时保存 `raw_model_output` 和 `fallback_reason`。

### 6.4 Query Planner Evaluation

文件：`src/eval/evaluate_query_planner.py`

已记录：

- 执行成功数与运行错误数。
- Valid Plan 与 Fallback。
- Simple / Complex 数量。
- 每个 Case 的子问题数量和延迟。
- 平均、最小、最大子问题数量。
- 子问题数量分布。
- 平均 Planner 延迟。

## 7. 当前已知问题

### 7.1 Reranker 接口不一致

当前静态检查发现：

```python
# rag_pipeline.py
self.reranker.rerank(
    query=query,
    candidates=retrieved_chunks,
    top_k=None
)
```

但是当前 `src/reranker/reranker.py` 中的方法签名是：

```python
def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
```

因此真实运行会发生 unexpected keyword argument `top_k`。在开始 Low 模式前，必须先统一这个接口。

### 7.2 Evidence Guard 仍是临时实现

当前仍使用：

```python
threshold=0.9
max_chunks=5
```

Cross Encoder 分数尚未进行绝对阈值校准，而且固定 5 条不能保证多证据覆盖。该逻辑暂时保留用于旧流程，后续由 Microsoft 风格的 Result Merge、Grounding Context 和 Sufficiency Check 逐步替代。

真实 Minimal 单查询验证再次证明固定阈值不可靠：

- Query：`信用卡年费是多少？`
- Top4 Chunk 与年费关联较弱，但 `reranker_score=0.9109`，会被 `threshold=0.9` 选中。
- Top5 Chunk 明确包含信用卡年费标准，但 `reranker_score=0.7847`，会被 `threshold=0.9` 删除。

因此，Cross Encoder 分数可以用于相对排序，但当前不能直接解释为统一的绝对相关概率。

### 7.3 Planner 评价主要是结构评价

当前没有 `query_plan_gold`，也没有自动化语义等价判断。暂时不把“子问题文本完全正确”设为上线门槛，最终通过证据召回与端到端指标判断 Planner 是否有用。

## 8. 分阶段实施计划

### 阶段 0：Minimal 基线

- [x] 完成 Vector、BM25、Hybrid Retriever。
- [x] 完成 RRF Hybrid Retrieval。
- [x] 完成 Cross Encoder Reranker。
- [x] 完成 Retriever 独立评价。
- [x] 完成 Reranker 独立评价。
- [x] 保存 Minimal 模式基线指标。

### 阶段 1：稳定现有接口

- [x] 统一 `RAGPipeline.rerank_candidates()` 与 `CrossEncoderReranker.rerank()` 的参数。
- [x] 使用 Fake Retriever / Fake Reranker 验证 `retriever_reranker()`。
- [x] 确保旧 Minimal 流程在加入多查询前仍能运行。

当前验证记录：

- `src/pipeline/rag_pipeline.py` 与 `src/reranker/reranker.py` 语法检查通过。
- Fake Retriever 正确接收到 Original Query 和 `per_retriever_k=20`。
- Fake Reranker 正确接收到 Retriever 返回的全部候选。
- `retriever_reranker()` 正确同时返回检索结果与重排结果。
- Fake Reranker 写入的 `reranker_score` 没有污染原始 `retrieved_chunks`。
- 已使用真实 Hybrid Retriever 和 Cross Encoder 运行 Query `信用卡年费是多少？`。
- 真实链路返回 20 条完整重排结果，每条结果均包含 `reranker_score`。
- Top1 为 `05_信用卡收费标准.pdf_0`，内容直接包含信用卡年费标准。
- 真实链路没有出现 `top_k` 参数错误。

验收标准：

- 不加载真实模型也能完成接口测试。
- Reranker 返回全部已排序候选。
- `retrieved_chunks` 和 `reranked_chunks` 都能正常返回。

### 阶段 2：Low - Multi-query Retrieval

- [ ] 将 Original Query 与 Query Planner 生成的 Subqueries 统一包装为 Query Tasks。
- [ ] 每个 Query Task 独立调用 Hybrid Retriever。
- [ ] 为每次检索保存 `query_id`、`query_text`、原始排名和检索耗时。
- [ ] 第一版先顺序执行，结果正确后再考虑并行优化。

建议 Query Task：

```python
{
    "query_id": "original",
    "query_text": "原始问题",
    "query_source": "original"
}
```

以及：

```python
{
    "query_id": "sq_1",
    "query_text": "完整子问题",
    "query_source": "planner"
}
```

验收标准：

- Simple Query 至少执行 Original Query，不重复执行完全相同的子问题。
- Complex Query 能分别返回每个 Query Task 的候选。
- 单个查询失败时能够定位到具体 `query_id`。

### 阶段 3：Low - Per-query Reranking 与 Result Merge

- [ ] 每个 Query Task 的候选分别经过 Cross Encoder。
- [ ] 按 Chunk ID 合并重复候选。
- [ ] 保存每个 Chunk 命中的 Query、Hybrid Rank、RRF Score 和 Reranker Score。
- [ ] 不在 Merge 阶段静默丢弃某个子问题的全部结果。

建议合并结果：

```python
{
    "id": "chunk_id",
    "text": "chunk text",
    "matched_query_ids": ["original", "sq_1"],
    "query_matches": {
        "original": {
            "hybrid_rank": 3,
            "reranker_score": 0.82
        },
        "sq_1": {
            "hybrid_rank": 1,
            "reranker_score": 0.91
        }
    }
}
```

验收标准：

- 同一个 Chunk 只保留一份正文。
- Query-Chunk 来源信息不丢失。
- 可以回答“某个正确 Chunk 是由哪个 Query 找到的”。

### 阶段 4：Low 模式评价

- [ ] 新增 Minimal 与 Low 的同数据集对比。
- [ ] 评价最终合并候选的 Hit@K、Recall@K、Complete@K 和 Multi-Complete@K。
- [ ] 记录 Query 数量、候选数量、Reranker Pair 数量和总延迟。
- [ ] 输出 Low 模式 Bad Cases。

核心问题：

```text
Original Query + Subqueries
是否比只使用 Original Query 找到了更完整的证据？
```

只有 Low 模式通过评价后，才进入 Medium。

### 阶段 5：Medium - Evidence Sufficiency 与一次补检索

- [ ] 新增 `EvidenceSufficiencyChecker`。
- [ ] 输入 Original Query 和第一轮 Grounding Evidence。
- [ ] 输出 `sufficient`、`missing_aspects` 和 `follow_up_queries`。
- [ ] 证据不足时只补检索一次。
- [ ] 第二轮结果与第一轮结果合并去重。
- [ ] 仍然不足时进入统一拒答流程。

验收指标：

- Missing Aspect Detection Accuracy。
- Retry Trigger Rate。
- Retry Success Rate。
- Medium 相对 Low 的 Complete@K 提升。
- 第二轮增加的查询数与延迟。

### 阶段 6：Generation 与端到端集成

- [ ] 使用最终合并证据构造 Citation Prompt。
- [ ] 保留 Query Plan、检索活动、重排活动和重试活动。
- [ ] 评价答案要点覆盖、引用正确性、拒答和端到端延迟。

## 9. 实验对照设计

| 模式 | Query Planning | 多查询检索 | 证据充分性检查 | 补检索 |
|---|---|---|---|---|
| Minimal | 否 | 否 | 否 | 否 |
| Low | 是 | 是 | 否 | 否 |
| Medium | 是 | 是 | 是 | 最多一次 |

共同记录：

- Hit@K
- Recall@K
- Complete@K
- Multi-Complete@K
- MRR@K
- Planner Latency
- Retrieval Latency
- Reranker Latency
- Total Latency
- Query Count
- Reranker Pair Count
- Retry Count

## 10. Trace 与可观测性

参考 Azure activity log，最终 Pipeline 应返回：

```python
{
    "mode": "low",
    "original_query": "...",
    "query_plan": {},
    "query_tasks": [],
    "retrieval_activities": [],
    "reranker_activities": [],
    "merged_chunks": [],
    "sufficiency_check": None,
    "retry_count": 0,
    "total_latency_seconds": 0.0
}
```

Trace 的目标是定位失败发生在哪一层：

```text
Planner 漏掉信息
Retrieval 没召回
Reranker 排序错误
Merge 丢失 Query 来源
Sufficiency 判断错误
Generator 没使用已有证据
```

## 11. 不采用的方案

### 11.1 不固定只拆 3 个子问题

固定数量可能直接遗漏必要组合。子问题数量由语义需要决定，资源限制放在执行预算层。

### 11.2 不把 Prompt 调整当作最终保障

Prompt 只提供通用行为约束。最终是否改进以证据召回、完整性、成本和延迟为准，不追求评价集上每条子问题文本完全符合人工写法。

### 11.3 不采用固定 Top7 解决多证据问题

未来正确证据可能出现在其他排名。固定扩大 Top-K 不能解决证据集合覆盖问题，还会增加无关上下文。

### 11.4 不直接使用固定 `reranker_score >= 0.9`

Cross Encoder 分数没有经过统一阈值校准，正确证据也可能低于 0.9。

### 11.5 不实现 NVIDIA 式无限递归

本项目采用 Microsoft Medium 风格的最多一次补检索，控制复杂度、延迟和成本。

## 12. 关键原则

1. 统一采用 Microsoft Minimal → Low → Medium 路线。
2. 每次只实现并验证一个阶段。
3. Query Planner 服务于证据召回，不单独承担最终正确性。
4. Retriever 负责候选召回，Reranker 负责单个 Query-Chunk 相关性。
5. Result Merge 必须保留每个 Chunk 的 Query 来源。
6. Runtime 不能读取 Gold Chunk ID、Expected Points 或人工 Gold Subquery。
7. 评价以最终证据完整性为主，以 Planner 文本检查为辅。
8. 不根据当前测试集个例硬编码 Top-K、阈值或专用规则。
9. 新模块先使用 Fake Component 测试，再加载真实模型。
10. Medium 模式最多补检索一次。

## 13. 下一步唯一任务

进入阶段 2，但本轮只完成第一项：

```text
向 RAGPipeline 注入可选的 QueryPlanner：
query_planner=None
```

这一步只建立依赖关系，不调用 Query Planner、不构造 Query Tasks，也不修改旧 `run()` 流程。完成并确认 Minimal 行为不变后，再实现 `build_query_tasks()`。

## 14. 更新日志

### 2026-09-24

- 完成 Retriever 与 Reranker 独立评价。
- 完成 Reranker 打分排序与证据选择职责拆分。
- Pipeline 开始返回检索、重排和证据三个阶段结果。
- 创建 Query Planner Prompt 与 QueryPlanner。
- 完成 JSON 提取、结构校验、去重、编号和 fallback。

### 2026-09-25

- 使用 61 条统一评价 Query 完成 Query Planner 真实模型评测。
- Query Planner 输出协议简化为只生成 `subqueries`。
- 由程序根据子问题数量生成 `query_type`。
- 取消固定最多 3 个子问题的语义限制。
- 评价代码新增最小、最大、平均子问题数量和数量分布。
- 最新结果为 61/61 Valid、0 Fallback、最大 4 个子问题、平均延迟 6.325 秒。
- 确认项目统一采用 Microsoft Azure Agentic Retrieval 路线。
- 明确后续按 Minimal → Low → Medium 分阶段实现。
- 记录当前 Reranker 调用接口不一致问题，作为下一步唯一修复任务。
- 删除 Pipeline 调用 Reranker 时已经失效的 `top_k=None` 参数。
- `rag_pipeline.py` 与 `reranker.py` 语法检查通过。
- Fake Retriever / Fake Reranker 接口测试通过，阶段 1 还剩真实 Minimal 单查询验证。
- 使用真实 Hybrid Retriever 和 Cross Encoder 完成 Minimal 单查询验证，阶段 1 全部完成。
- 真实结果进一步证明固定 `reranker_score >= 0.9` 不能可靠区分证据相关性。
