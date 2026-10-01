import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generator.llm_generator import LLMGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.query_processing.query_planner import QueryPlanner
from src.reranker.reranker import CrossEncoderReranker
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.vector_retriever import VectorRetriever


CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_info.json"
EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "bank_eval_natural_v1.json"
)
OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "low_pipeline_eval_results.json"
)
OUTPUT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "low_pipeline_eval.md"
)
FUSION_STRATEGIES = (
    "round_robin",
    "distribution_based",
    "distribution_based_original_anchor",
)

K_VALUES = (1, 3, 5, 10, 20)
PER_RETRIEVER_K = 20
RANK_CONSTANT = 60
QUERY_PLANNER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_cases() -> list[dict]:
    evaluation_data = load_json(EVALUATION_PATH)

    cases = [
        case
        for case in evaluation_data["cases"]
        if "retriever" in case["use_for"]
        and "reranker" in case["use_for"]
        and not case["should_refuse"]
    ]

    if not cases:
        raise ValueError("没有可用于 Minimal / Low 对比的评价样本")

    return cases


def validate_evaluation_data(
    cases: list[dict],
    chunks_info: list[dict],
) -> None:
    chunk_ids = [chunk["id"] for chunk in chunks_info]
    duplicate_chunk_ids = sorted(
        {
            chunk_id
            for chunk_id in chunk_ids
            if chunk_ids.count(chunk_id) > 1
        }
    )

    if duplicate_chunk_ids:
        raise ValueError(
            "chunks_info.json 中存在重复 Chunk ID：\n"
            + "\n".join(duplicate_chunk_ids)
        )

    existing_chunk_ids = set(chunk_ids)
    missing_chunk_ids = sorted(
        {
            chunk_id
            for case in cases
            for chunk_id in case["evidence_chunk_ids"]
            if chunk_id not in existing_chunk_ids
        }
    )

    if missing_chunk_ids:
        raise ValueError(
            "评价集中的以下证据不在当前 chunks_info.json 中：\n"
            + "\n".join(missing_chunk_ids)
        )

    empty_evidence_case_ids = [
        case["id"]
        for case in cases
        if not case["evidence_chunk_ids"]
    ]

    if empty_evidence_case_ids:
        raise ValueError(
            "以下评价样本没有 evidence_chunk_ids：\n"
            + "\n".join(empty_evidence_case_ids)
        )


def build_evaluation_pipeline(
    chunks_info: list[dict],
) -> RAGPipeline:
    vector_retriever = VectorRetriever()
    vector_retriever.fit(chunks_info)

    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(chunks_info)

    hybrid_retriever = HybridRetriever(
        denseretriever=vector_retriever,
        sparseretriever=bm25_retriever,
        rank_constant=RANK_CONSTANT,
        per_retriever_k=PER_RETRIEVER_K,
    )

    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL_NAME
    )
    generator = LLMGenerator(
        model_name=QUERY_PLANNER_MODEL_NAME
    )
    query_planner = QueryPlanner(
        generator=generator,
        max_subqueries=None,
    )

    return RAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        query_planner=query_planner,
        generator=None,
    )


def calculate_case_metrics(
    ranked_ids: list[str],
    relevant_ids: list[str],
) -> dict:
    relevant_set = set(relevant_ids)
    metrics_by_k = {}

    for k in K_VALUES:
        top_k_ids = ranked_ids[:k]
        found_ids = relevant_set.intersection(top_k_ids)
        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    top_k_ids,
                    start=1,
                )
                if chunk_id in relevant_set
            ),
            None,
        )

        metrics_by_k[str(k)] = {
            "hit": bool(found_ids),
            "recall": len(found_ids) / len(relevant_set),
            "complete": relevant_set.issubset(top_k_ids),
            "reciprocal_rank": (
                1.0 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            ),
        }

    return metrics_by_k


def get_relevant_ranks(
    ranked_ids: list[str],
    relevant_ids: list[str],
) -> dict[str, int | None]:
    rank_lookup = {
        chunk_id: rank
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    }

    return {
        chunk_id: rank_lookup.get(chunk_id)
        for chunk_id in relevant_ids
    }


def run_minimal(
    pipeline: RAGPipeline,
    query: str,
) -> dict:
    total_start_time = time.perf_counter()

    retrieval_start_time = time.perf_counter()
    retrieved_chunks = pipeline.retriever_search(
        query=query,
        per_retriever_k=PER_RETRIEVER_K,
    )
    retrieval_latency_seconds = (
        time.perf_counter() - retrieval_start_time
    )

    reranker_start_time = time.perf_counter()
    reranked_chunks = pipeline.rerank_candidates(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )
    reranker_latency_seconds = (
        time.perf_counter() - reranker_start_time
    )

    total_latency_seconds = (
        time.perf_counter() - total_start_time
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": reranked_chunks,
        "query_count": 1,
        "total_retrieved_chunk_count": len(retrieved_chunks),
        "merged_chunk_count": len(reranked_chunks),
        "duplicate_chunk_count": 0,
        "reranker_pair_count": len(retrieved_chunks),
        "planner_latency_seconds": 0.0,
        "retrieval_latency_seconds": retrieval_latency_seconds,
        "reranker_latency_seconds": reranker_latency_seconds,
        "total_latency_seconds": total_latency_seconds,
    }


def compare_boolean(
    minimal_value: bool,
    low_value: bool,
) -> str:
    if not minimal_value and low_value:
        return "rescued"

    if minimal_value and not low_value:
        return "regressed"

    return "unchanged"


def build_low_relevant_trace(
    merged_chunks: list[dict],
    relevant_ids: list[str],
) -> dict:
    chunk_lookup = {
        chunk["id"]: (rank, chunk)
        for rank, chunk in enumerate(merged_chunks, start=1)
    }
    trace = {}

    for chunk_id in relevant_ids:
        rank_and_chunk = chunk_lookup.get(chunk_id)

        if rank_and_chunk is None:
            trace[chunk_id] = {
                "merged_rank": None,
                "matched_query_ids": [],
                "diff_query_performance": {},
            }
            continue

        merged_rank, chunk = rank_and_chunk
        trace[chunk_id] = {
            "merged_rank": merged_rank,
            "matched_query_ids": chunk["matched_query_ids"],
            "diff_query_performance": chunk[
                "diff_query_performance"
            ],
        }

    return trace


def evaluate_case(
    pipeline: RAGPipeline,
    case: dict,
    fusion_strategy: str = "round_robin",
) -> dict:
    query = case["query"]
    relevant_ids = case["evidence_chunk_ids"]

    minimal_result = run_minimal(
        pipeline=pipeline,
        query=query,
    )
    low_result = pipeline.run_low(
        query=query,
        per_retriever_k=PER_RETRIEVER_K,
        fusion_strategy=fusion_strategy,
    )

    minimal_retrieved_ids = [
        chunk["id"]
        for chunk in minimal_result["retrieved_chunks"]
    ]
    minimal_ranked_ids = [
        chunk["id"]
        for chunk in minimal_result["reranked_chunks"]
    ]
    low_ranked_ids = [
        chunk["id"]
        for chunk in low_result["accumulated_chunks"]
    ]

    minimal_metrics = calculate_case_metrics(
        ranked_ids=minimal_ranked_ids,
        relevant_ids=relevant_ids,
    )
    low_metrics = calculate_case_metrics(
        ranked_ids=low_ranked_ids,
        relevant_ids=relevant_ids,
    )

    comparison_by_k = {}

    for k in K_VALUES:
        k_text = str(k)
        minimal_at_k = minimal_metrics[k_text]
        low_at_k = low_metrics[k_text]
        comparison_by_k[k_text] = {
            "hit_outcome": compare_boolean(
                minimal_value=minimal_at_k["hit"],
                low_value=low_at_k["hit"],
            ),
            "complete_outcome": compare_boolean(
                minimal_value=minimal_at_k["complete"],
                low_value=low_at_k["complete"],
            ),
            "recall_delta": (
                low_at_k["recall"]
                - minimal_at_k["recall"]
            ),
            "reciprocal_rank_delta": (
                low_at_k["reciprocal_rank"]
                - minimal_at_k["reciprocal_rank"]
            ),
        }

    return {
        "id": case["id"],
        "query": query,
        "evidence_mode": case["evidence_mode"],
        "task_type": case.get("task_type"),
        "query_style": case.get("query_style"),
        "relevant_chunk_ids": relevant_ids,
        "minimal": {
            "retrieval_relevant_ranks": get_relevant_ranks(
                ranked_ids=minimal_retrieved_ids,
                relevant_ids=relevant_ids,
            ),
            "relevant_ranks": get_relevant_ranks(
                ranked_ids=minimal_ranked_ids,
                relevant_ids=relevant_ids,
            ),
            "top20_chunk_ids": minimal_ranked_ids[
                : max(K_VALUES)
            ],
            "metrics": minimal_metrics,
            "query_count": minimal_result["query_count"],
            "total_retrieved_chunk_count": minimal_result[
                "total_retrieved_chunk_count"
            ],
            "merged_chunk_count": minimal_result[
                "merged_chunk_count"
            ],
            "duplicate_chunk_count": minimal_result[
                "duplicate_chunk_count"
            ],
            "reranker_pair_count": minimal_result[
                "reranker_pair_count"
            ],
            "planner_latency_seconds": minimal_result[
                "planner_latency_seconds"
            ],
            "retrieval_latency_seconds": minimal_result[
                "retrieval_latency_seconds"
            ],
            "reranker_latency_seconds": minimal_result[
                "reranker_latency_seconds"
            ],
            "total_latency_seconds": minimal_result[
                "total_latency_seconds"
            ],
        },
        "low": {
            "fusion_strategy": low_result["fusion_strategy"],
            "query_plan": low_result["query_plan"],
            "query_tasks": low_result["query_tasks"],
            "relevant_ranks": get_relevant_ranks(
                ranked_ids=low_ranked_ids,
                relevant_ids=relevant_ids,
            ),
            "relevant_chunk_trace": build_low_relevant_trace(
                merged_chunks=low_result[
                    "accumulated_chunks"
                ],
                relevant_ids=relevant_ids,
            ),
            "top20_chunk_ids": low_ranked_ids[: max(K_VALUES)],
            "metrics": low_metrics,
            "query_count": low_result["query_count"],
            "total_retrieved_chunk_count": low_result[
                "total_retrieved_chunk_count"
            ],
            "merged_chunk_count": low_result[
                "merged_chunk_count"
            ],
            "duplicate_chunk_count": low_result[
                "duplicate_chunk_count"
            ],
            "reranker_pair_count": low_result[
                "reranker_pair_count"
            ],
            "planner_latency_seconds": low_result[
                "planner_latency_seconds"
            ],
            "retrieval_latency_seconds": low_result[
                "retrieval_latency_seconds"
            ],
            "reranker_latency_seconds": low_result[
                "reranker_latency_seconds"
            ],
            "total_latency_seconds": low_result[
                "total_latency_seconds"
            ],
        },
        "comparison": comparison_by_k,
        "error": None,
    }


def summarize_ranking_metrics(
    case_results: list[dict],
    mode: str,
) -> dict:
    summary = {}
    multi_evidence_results = [
        result
        for result in case_results
        if result["evidence_mode"] == "all_required"
    ]

    for k in K_VALUES:
        k_text = str(k)
        metrics = [
            result[mode]["metrics"][k_text]
            for result in case_results
        ]
        multi_metrics = [
            result[mode]["metrics"][k_text]
            for result in multi_evidence_results
        ]

        summary[k_text] = {
            "hit_rate": (
                sum(item["hit"] for item in metrics)
                / len(metrics)
            ),
            "mean_recall": (
                sum(item["recall"] for item in metrics)
                / len(metrics)
            ),
            "complete_rate": (
                sum(item["complete"] for item in metrics)
                / len(metrics)
            ),
            "mrr": (
                sum(
                    item["reciprocal_rank"]
                    for item in metrics
                )
                / len(metrics)
            ),
            "multi_evidence_complete_rate": (
                sum(item["complete"] for item in multi_metrics)
                / len(multi_metrics)
                if multi_metrics
                else None
            ),
        }

    return summary


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    sorted_values = sorted(values)
    index = max(
        0,
        math.ceil(percentile_value * len(sorted_values)) - 1,
    )
    return sorted_values[index]


def summarize_numeric_values(values: list[float]) -> dict:
    return {
        "average": sum(values) / len(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_efficiency(
    case_results: list[dict],
    mode: str,
) -> dict:
    count_fields = (
        "query_count",
        "total_retrieved_chunk_count",
        "merged_chunk_count",
        "duplicate_chunk_count",
        "reranker_pair_count",
    )
    latency_fields = (
        "planner_latency_seconds",
        "retrieval_latency_seconds",
        "reranker_latency_seconds",
        "total_latency_seconds",
    )

    summary = {}

    for field in count_fields:
        values = [
            float(result[mode][field])
            for result in case_results
        ]
        summary[field] = summarize_numeric_values(values)

    for field in latency_fields:
        values = [
            result[mode][field]
            for result in case_results
        ]
        summary[field] = summarize_numeric_values(values)

    return summary


def build_summary(case_results: list[dict]) -> dict:
    successful_results = [
        result
        for result in case_results
        if result["error"] is None
    ]
    failed_results = [
        result
        for result in case_results
        if result["error"] is not None
    ]

    if not successful_results:
        raise RuntimeError("所有评价样本均执行失败，无法计算指标")

    minimal_ranking = summarize_ranking_metrics(
        case_results=successful_results,
        mode="minimal",
    )
    low_ranking = summarize_ranking_metrics(
        case_results=successful_results,
        mode="low",
    )

    ranking_deltas = {}
    comparison_by_k = {}

    for k in K_VALUES:
        k_text = str(k)
        ranking_deltas[k_text] = {
            metric_name: (
                low_ranking[k_text][metric_name]
                - minimal_ranking[k_text][metric_name]
                if low_ranking[k_text][metric_name] is not None
                and minimal_ranking[k_text][metric_name]
                is not None
                else None
            )
            for metric_name in minimal_ranking[k_text]
        }

        hit_rescued_ids = [
            result["id"]
            for result in successful_results
            if result["comparison"][k_text]["hit_outcome"]
            == "rescued"
        ]
        hit_regressed_ids = [
            result["id"]
            for result in successful_results
            if result["comparison"][k_text]["hit_outcome"]
            == "regressed"
        ]
        complete_rescued_ids = [
            result["id"]
            for result in successful_results
            if result["comparison"][k_text][
                "complete_outcome"
            ]
            == "rescued"
        ]
        complete_regressed_ids = [
            result["id"]
            for result in successful_results
            if result["comparison"][k_text][
                "complete_outcome"
            ]
            == "regressed"
        ]

        comparison_by_k[k_text] = {
            "hit_rescued_case_count": len(hit_rescued_ids),
            "hit_rescued_case_ids": hit_rescued_ids,
            "hit_regressed_case_count": len(hit_regressed_ids),
            "hit_regressed_case_ids": hit_regressed_ids,
            "complete_rescued_case_count": len(
                complete_rescued_ids
            ),
            "complete_rescued_case_ids": complete_rescued_ids,
            "complete_regressed_case_count": len(
                complete_regressed_ids
            ),
            "complete_regressed_case_ids": (
                complete_regressed_ids
            ),
        }

    low_incomplete_at_5_ids = [
        result["id"]
        for result in successful_results
        if not result["low"]["metrics"]["5"]["complete"]
    ]

    return {
        "total_case_count": len(case_results),
        "successful_paired_case_count": len(successful_results),
        "failed_case_count": len(failed_results),
        "failed_case_ids": [
            result["id"] for result in failed_results
        ],
        "multi_evidence_case_count": sum(
            result["evidence_mode"] == "all_required"
            for result in successful_results
        ),
        "minimal": {
            "ranking": minimal_ranking,
            "efficiency": summarize_efficiency(
                case_results=successful_results,
                mode="minimal",
            ),
        },
        "low": {
            "ranking": low_ranking,
            "efficiency": summarize_efficiency(
                case_results=successful_results,
                mode="low",
            ),
        },
        "ranking_delta_low_minus_minimal": ranking_deltas,
        "comparison_by_k": comparison_by_k,
        "low_incomplete_at_5_case_count": len(
            low_incomplete_at_5_ids
        ),
        "low_incomplete_at_5_case_ids": (
            low_incomplete_at_5_ids
        ),
    }


def format_rate(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def format_rank_map(rank_map: dict[str, int | None]) -> str:
    return "<br>".join(
        f"`{chunk_id}`: {rank if rank is not None else '未召回'}"
        for chunk_id, rank in rank_map.items()
    )


def build_markdown_report(
    summary: dict,
    case_results: list[dict],
    fusion_strategy: str = "round_robin",
) -> str:
    lines = [
        "# Minimal vs Low Pipeline Evaluation",
        "",
        "## 评测设置",
        "",
        f"- 数据集：`{EVALUATION_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 成功配对样本：{summary['successful_paired_case_count']} / "
        f"{summary['total_case_count']}",
        f"- 多证据样本：{summary['multi_evidence_case_count']}",
        f"- 每个 Query Task 的候选数：Top{PER_RETRIEVER_K}",
        f"- Query Planner：`{QUERY_PLANNER_MODEL_NAME}`",
        f"- Reranker：`{RERANKER_MODEL_NAME}`",
        f"- Fusion：`{fusion_strategy}`",
        "- Minimal：只使用 Original Query 检索并重排。",
        "- Low：使用 Original Query 与子问题分别检索重排，再合并去重。",
        "- K 只用于离线评价，不会截断或修改生产 Pipeline。",
        "",
        "## 排名指标",
        "",
        "| K | 指标 | Minimal | Low | Low - Minimal |",
        "|---:|---|---:|---:|---:|",
    ]

    metric_labels = {
        "hit_rate": "Hit@K",
        "mean_recall": "Mean Recall@K",
        "complete_rate": "Complete@K",
        "mrr": "MRR@K",
        "multi_evidence_complete_rate": "Multi-Complete@K",
    }

    for k in K_VALUES:
        k_text = str(k)
        minimal_metrics = summary["minimal"]["ranking"][k_text]
        low_metrics = summary["low"]["ranking"][k_text]
        delta_metrics = summary[
            "ranking_delta_low_minus_minimal"
        ][k_text]

        for metric_name, metric_label in metric_labels.items():
            minimal_value = minimal_metrics[metric_name]
            low_value = low_metrics[metric_name]
            delta_value = delta_metrics[metric_name]
            lines.append(
                f"| {k} | {metric_label} | "
                f"{format_rate(minimal_value)} | "
                f"{format_rate(low_value)} | "
                f"{format_rate(delta_value)} |"
            )

    lines.extend(
        [
            "",
            "## 运行开销",
            "",
            "下表为每条样本的平均值。延迟不包含模型初始化时间。",
            "",
            "| 指标 | Minimal | Low |",
            "|---|---:|---:|",
        ]
    )

    efficiency_labels = {
        "query_count": "Query 数量",
        "total_retrieved_chunk_count": "检索候选总数",
        "merged_chunk_count": "最终去重候选数",
        "duplicate_chunk_count": "合并去重数",
        "reranker_pair_count": "Reranker Pair 数量",
    }

    for field, label in efficiency_labels.items():
        minimal_average = summary["minimal"]["efficiency"][
            field
        ]["average"]
        low_average = summary["low"]["efficiency"][field][
            "average"
        ]
        lines.append(
            f"| {label} | {minimal_average:.2f} | "
            f"{low_average:.2f} |"
        )

    latency_labels = {
        "planner_latency_seconds": "Planner 延迟（秒）",
        "retrieval_latency_seconds": "Retrieval 延迟（秒）",
        "reranker_latency_seconds": "Reranker 延迟（秒）",
        "total_latency_seconds": "总延迟（秒）",
    }

    for field, label in latency_labels.items():
        minimal_latency = summary["minimal"]["efficiency"][field]
        low_latency = summary["low"]["efficiency"][field]
        lines.append(
            f"| {label} | "
            f"avg {minimal_latency['average']:.3f}, "
            f"P95 {minimal_latency['p95']:.3f} | "
            f"avg {low_latency['average']:.3f}, "
            f"P95 {low_latency['p95']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 拉回与回退",
            "",
            "| K | Hit 拉回 | Hit 回退 | Complete 拉回 | Complete 回退 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )

    for k in K_VALUES:
        comparison = summary["comparison_by_k"][str(k)]
        lines.append(
            f"| {k} | "
            f"{comparison['hit_rescued_case_count']} | "
            f"{comparison['hit_regressed_case_count']} | "
            f"{comparison['complete_rescued_case_count']} | "
            f"{comparison['complete_regressed_case_count']} |"
        )

    lines.extend(
        [
            "",
            "## Low Top5 不完整案例",
            "",
            "| Case | Evidence Mode | Minimal 正确证据排名 | Low 正确证据排名 |",
            "|---|---|---|---|",
        ]
    )

    successful_results = [
        result
        for result in case_results
        if result["error"] is None
    ]

    for result in successful_results:
        if result["low"]["metrics"]["5"]["complete"]:
            continue

        lines.append(
            f"| {result['id']} | {result['evidence_mode']} | "
            f"{format_rank_map(result['minimal']['relevant_ranks'])} | "
            f"{format_rank_map(result['low']['relevant_ranks'])} |"
        )

    if summary["failed_case_count"]:
        lines.extend(
            [
                "",
                "## 执行失败案例",
                "",
            ]
        )
        lines.extend(
            f"- `{result['id']}`：{result['error']}"
            for result in case_results
            if result["error"] is not None
        )

    return "\n".join(lines) + "\n"


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 96)
    print(
        "Minimal vs Low | "
        f"Paired cases: {summary['successful_paired_case_count']} | "
        f"Failed: {summary['failed_case_count']}"
    )
    print("=" * 96)
    print(
        f"{'K':>3} "
        f"{'Minimal Complete':>18} "
        f"{'Low Complete':>14} "
        f"{'Delta':>10} "
        f"{'Minimal Multi':>15} "
        f"{'Low Multi':>11}"
    )

    for k in K_VALUES:
        k_text = str(k)
        minimal = summary["minimal"]["ranking"][k_text]
        low = summary["low"]["ranking"][k_text]
        delta = summary["ranking_delta_low_minus_minimal"][
            k_text
        ]
        print(
            f"{k:>3} "
            f"{minimal['complete_rate']:>18.3f} "
            f"{low['complete_rate']:>14.3f} "
            f"{delta['complete_rate']:>+10.3f} "
            f"{minimal['multi_evidence_complete_rate']:>15.3f} "
            f"{low['multi_evidence_complete_rate']:>11.3f}"
        )

    minimal_total = summary["minimal"]["efficiency"][
        "total_latency_seconds"
    ]
    low_total = summary["low"]["efficiency"][
        "total_latency_seconds"
    ]
    print("\nAverage / P95 total latency:")
    print(
        f"Minimal: {minimal_total['average']:.3f}s / "
        f"{minimal_total['p95']:.3f}s"
    )
    print(
        f"Low:     {low_total['average']:.3f}s / "
        f"{low_total['p95']:.3f}s"
    )


def save_results(
    summary: dict,
    case_results: list[dict],
    fusion_strategy: str = "round_robin",
    output_json_path: Path = OUTPUT_JSON_PATH,
    output_markdown_path: Path = OUTPUT_MARKDOWN_PATH,
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "evaluation_file": str(EVALUATION_PATH),
        "chunks_file": str(CHUNKS_PATH),
        "config": {
            "k_values": list(K_VALUES),
            "per_retriever_k": PER_RETRIEVER_K,
            "rank_constant": RANK_CONSTANT,
            "query_planner_model": QUERY_PLANNER_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "fusion_strategy": fusion_strategy,
        },
        "summary": summary,
        "cases": case_results,
    }

    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    markdown_report = build_markdown_report(
        summary=summary,
        case_results=case_results,
        fusion_strategy=fusion_strategy,
    )
    with output_markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评价 Minimal 与 Low Pipeline。",
    )
    parser.add_argument(
        "--fusion-strategy",
        choices=FUSION_STRATEGIES,
        default="round_robin",
        help="Low 模式的跨 Query Fusion 策略。",
    )
    return parser.parse_args()


def get_output_paths(fusion_strategy: str) -> tuple[Path, Path]:
    if fusion_strategy == "round_robin":
        return OUTPUT_JSON_PATH, OUTPUT_MARKDOWN_PATH

    stem = f"low_pipeline_{fusion_strategy}_eval"
    output_directory = OUTPUT_JSON_PATH.parent
    return (
        output_directory / f"{stem}_results.json",
        output_directory / f"{stem}.md",
    )


def main() -> None:
    args = parse_args()
    output_json_path, output_markdown_path = get_output_paths(
        fusion_strategy=args.fusion_strategy,
    )
    chunks_info = load_json(CHUNKS_PATH)
    cases = load_evaluation_cases()

    validate_evaluation_data(
        cases=cases,
        chunks_info=chunks_info,
    )

    print(f"Loaded {len(cases)} evaluation cases")
    print("Building shared Minimal / Low evaluation pipeline...")
    pipeline = build_evaluation_pipeline(chunks_info=chunks_info)

    case_results = []

    for index, case in enumerate(cases, start=1):
        print("\n" + "-" * 80)
        print(f"[{index}/{len(cases)}] {case['id']}")
        print(case["query"])

        try:
            result = evaluate_case(
                pipeline=pipeline,
                case=case,
                fusion_strategy=args.fusion_strategy,
            )
            minimal_rank = result["minimal"]["relevant_ranks"]
            low_rank = result["low"]["relevant_ranks"]
            print("Minimal ranks:", minimal_rank)
            print("Low ranks:", low_rank)
            print("Query count:", result["low"]["query_count"])

        except Exception as error:
            result = {
                "id": case["id"],
                "query": case["query"],
                "evidence_mode": case["evidence_mode"],
                "task_type": case.get("task_type"),
                "query_style": case.get("query_style"),
                "relevant_chunk_ids": case["evidence_chunk_ids"],
                "minimal": None,
                "low": None,
                "comparison": None,
                "error": f"{type(error).__name__}: {error}",
            }
            print("ERROR:", result["error"])

        case_results.append(result)

    summary = build_summary(case_results=case_results)
    save_results(
        summary=summary,
        case_results=case_results,
        fusion_strategy=args.fusion_strategy,
        output_json_path=output_json_path,
        output_markdown_path=output_markdown_path,
    )
    print_summary(summary=summary)

    print("\nSaved JSON to:")
    print(output_json_path)
    print("\nSaved Markdown to:")
    print(output_markdown_path)


if __name__ == "__main__":
    main()
