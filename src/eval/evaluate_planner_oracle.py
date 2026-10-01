import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime
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
from src.retriever.vector_retriever import (
    MODEL_NAME as VECTOR_MODEL_NAME,
    VectorRetriever,
)


BASELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "baseline_v1.json"
K_VALUES = (1, 3, 5, 10, 20)
SCORE_FIELDS = (
    "vector_retriever_score",
    "bm25_retriever_score",
    "rrf_score",
    "reranker_score",
)


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def validate_baseline_config(config: dict) -> None:
    for section in ("dataset", "chunks"):
        item = config[section]
        path = resolve_project_path(item["path"])

        if not path.is_file():
            raise FileNotFoundError(f"Baseline 文件不存在：{path}")

        actual_hash = sha256_file(path)
        expected_hash = item["sha256"].upper()

        if actual_hash != expected_hash:
            raise ValueError(
                f"{section} SHA256 不一致："
                f"expected={expected_hash}, actual={actual_hash}"
            )

    for split, item in config["planner_evaluation"].items():
        path = resolve_project_path(item["path"])

        if not path.is_file():
            raise FileNotFoundError(
                f"Planner {split} 数据集不存在：{path}"
            )

        actual_hash = sha256_file(path)
        expected_hash = item["sha256"].upper()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Planner {split} 数据集 SHA256 不一致："
                f"expected={expected_hash}, actual={actual_hash}"
            )

    configured_embedding = config["models"]["embedding"]

    if configured_embedding != VECTOR_MODEL_NAME:
        raise ValueError(
            "baseline_v1.json 的 Embedding 模型与 "
            "VectorRetriever 实际模型不一致："
            f"{configured_embedding} != {VECTOR_MODEL_NAME}"
        )


def validate_planner_dataset(
    dataset: dict,
    chunks_info: list[dict],
    natural_queries: set[str],
) -> None:
    cases = dataset.get("cases")

    if not isinstance(cases, list) or not cases:
        raise ValueError("Planner 数据集必须包含非空 cases 列表")

    case_ids = [case.get("id") for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Planner 数据集存在重复 case id")

    existing_chunk_ids = {chunk["id"] for chunk in chunks_info}

    for case in cases:
        required_fields = (
            "id",
            "query",
            "task_type",
            "evidence_mode",
            "required_evidence",
            "expected_points",
            "gold_subqueries",
            "source_case_ids",
        )
        missing_fields = [
            field
            for field in required_fields
            if field not in case
        ]

        if missing_fields:
            raise ValueError(
                f"{case.get('id')} 缺少字段：{missing_fields}"
            )

        if case["query"] in natural_queries:
            raise ValueError(
                f"{case['id']} 与 natural_v1 Query 重复"
            )

        required_evidence = case["required_evidence"]

        if len(required_evidence) < 2:
            raise ValueError(
                f"{case['id']} 不是多证据样本"
            )

        if len(required_evidence) != len(set(required_evidence)):
            raise ValueError(
                f"{case['id']} required_evidence 存在重复"
            )

        missing_chunks = sorted(
            set(required_evidence) - existing_chunk_ids
        )

        if missing_chunks:
            raise ValueError(
                f"{case['id']} 引用了不存在的 Chunk："
                f"{missing_chunks}"
            )

        if not case["expected_points"]:
            raise ValueError(
                f"{case['id']} expected_points 不能为空"
            )

        gold_subqueries = case["gold_subqueries"]

        if len(gold_subqueries) < 2:
            raise ValueError(
                f"{case['id']} 至少需要两个 gold_subqueries"
            )

        gold_evidence = {
            chunk_id
            for subquery in gold_subqueries
            for chunk_id in subquery["required_evidence"]
        }

        if gold_evidence != set(required_evidence):
            raise ValueError(
                f"{case['id']} 的 Gold Subquery 证据集合"
                "必须与 required_evidence 完全一致"
            )


def validate_split_isolation(
    dev_dataset: dict,
    locked_dataset: dict,
) -> None:
    dev_evidence = {
        chunk_id
        for case in dev_dataset["cases"]
        for chunk_id in case["required_evidence"]
    }
    locked_evidence = {
        chunk_id
        for case in locked_dataset["cases"]
        for chunk_id in case["required_evidence"]
    }
    overlap = sorted(dev_evidence.intersection(locked_evidence))

    if overlap:
        raise ValueError(
            "planner_dev 与 planner_test_locked 存在证据重叠："
            f"{overlap}"
        )


def build_evaluation_pipeline(
    chunks_info: list[dict],
    config: dict,
) -> RAGPipeline:
    vector_retriever = VectorRetriever()
    vector_retriever.fit(chunks_info)

    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(chunks_info)

    hybrid_retriever = HybridRetriever(
        denseretriever=vector_retriever,
        sparseretriever=bm25_retriever,
        rank_constant=config["retrieval"]["rank_constant"],
        per_retriever_k=(
            config["retrieval"]["per_retriever_k"]
        ),
    )

    reranker = CrossEncoderReranker(
        model_name=config["models"]["reranker"]
    )
    planner_generator = LLMGenerator(
        model_name=config["models"]["planner"]
    )
    query_planner = QueryPlanner(
        generator=planner_generator,
        max_subqueries=None,
    )

    return RAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        query_planner=query_planner,
        generator=None,
    )


def calculate_metrics(
    ranked_ids: list[str],
    required_evidence: list[str],
) -> dict:
    required_set = set(required_evidence)
    metrics = {}

    for k in K_VALUES:
        top_k_ids = ranked_ids[:k]
        found_ids = required_set.intersection(top_k_ids)
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    top_k_ids,
                    start=1,
                )
                if chunk_id in required_set
            ),
            None,
        )
        metrics[str(k)] = {
            "hit": bool(found_ids),
            "recall": len(found_ids) / len(required_set),
            "complete": required_set.issubset(top_k_ids),
            "reciprocal_rank": (
                1.0 / first_rank
                if first_rank is not None
                else 0.0
            ),
        }

    return metrics


def get_ranks(
    ranked_ids: list[str],
    required_evidence: list[str],
) -> dict[str, int | None]:
    rank_lookup = {
        chunk_id: rank
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    }
    return {
        chunk_id: rank_lookup.get(chunk_id)
        for chunk_id in required_evidence
    }


def compact_ranked_chunks(chunks: list[dict]) -> list[dict]:
    compact = []

    for rank, chunk in enumerate(chunks, start=1):
        item = {
            "rank": rank,
            "chunk_id": chunk["id"],
            "source": chunk.get("source"),
        }

        for score_field in SCORE_FIELDS:
            if score_field in chunk:
                item[score_field] = chunk[score_field]

        compact.append(item)

    return compact


def build_query_task_trace(
    query_task_results: list[dict],
    required_evidence: list[str],
) -> list[dict]:
    traces = []

    for query_result in query_task_results:
        retrieved_ids = [
            chunk["id"]
            for chunk in query_result["retrieved_chunks"]
        ]
        reranked_ids = [
            chunk["id"]
            for chunk in query_result["reranked_chunks"]
        ]
        traces.append(
            {
                "query_id": query_result["query_id"],
                "query_text": query_result["query_text"],
                "query_source": query_result["query_source"],
                "retrieval_latency_seconds": query_result[
                    "retrieval_latency_seconds"
                ],
                "reranker_latency_seconds": query_result[
                    "reranker_latency_seconds"
                ],
                "required_evidence_ranks": {
                    "hybrid": get_ranks(
                        retrieved_ids,
                        required_evidence,
                    ),
                    "reranker": get_ranks(
                        reranked_ids,
                        required_evidence,
                    ),
                },
                "hybrid_top20": compact_ranked_chunks(
                    query_result["retrieved_chunks"]
                ),
                "reranker_top20": compact_ranked_chunks(
                    query_result["reranked_chunks"]
                ),
            }
        )

    return traces


def build_fusion_diagnostics(
    query_task_results: list[dict],
    accumulated_chunks: list[dict],
    required_evidence: list[str],
    final_k: int,
) -> dict:
    pre_fusion_ids = {
        chunk["id"]
        for query_result in query_task_results
        for chunk in query_result["reranked_chunks"]
    }
    post_fusion_ids = [
        chunk["id"]
        for chunk in accumulated_chunks
    ]
    pre_fusion_found = {
        chunk_id: chunk_id in pre_fusion_ids
        for chunk_id in required_evidence
    }
    post_fusion_ranks = get_ranks(
        post_fusion_ids,
        required_evidence,
    )
    dropped_at_final_k = [
        chunk_id
        for chunk_id in required_evidence
        if pre_fusion_found[chunk_id]
        and (
            post_fusion_ranks[chunk_id] is None
            or post_fusion_ranks[chunk_id] > final_k
        )
    ]

    return {
        "all_required_found_before_fusion": all(
            pre_fusion_found.values()
        ),
        "required_evidence_found_before_fusion": (
            pre_fusion_found
        ),
        "required_evidence_ranks_after_fusion": (
            post_fusion_ranks
        ),
        "dropped_required_evidence_at_final_k": (
            dropped_at_final_k
        ),
    }


def build_original_tasks(query: str) -> list[dict]:
    return [
        {
            "query_id": "original",
            "query_text": query,
            "query_source": "original",
        }
    ]


def build_oracle_tasks(case: dict) -> list[dict]:
    tasks = build_original_tasks(case["query"])

    for subquery in case["gold_subqueries"]:
        tasks.append(
            {
                "query_id": subquery["id"],
                "query_text": subquery["query"],
                "query_source": "oracle",
            }
        )

    return tasks


def run_query_tasks(
    pipeline: RAGPipeline,
    method: str,
    query_tasks: list[dict],
    required_evidence: list[str],
    per_retriever_k: int,
    final_k: int,
    method_start_time: float,
    planner_latency_seconds: float = 0.0,
    query_plan: dict | None = None,
    fusion_strategy: str = "round_robin",
) -> dict:
    query_task_results = (
        pipeline.query_tasks_retriever_reranker(
            query_tasks=query_tasks,
            per_retriever_k=per_retriever_k,
        )
    )
    merged_chunks_dict = pipeline.merge_query_task_results(
        query_task_results=query_task_results
    )
    accumulated_chunks = pipeline.build_accumulated_context(
        query_task_results=query_task_results,
        merged_chunks_dict=merged_chunks_dict,
        fusion_strategy=fusion_strategy,
    )
    ranked_ids = [
        chunk["id"]
        for chunk in accumulated_chunks
    ]
    retrieval_latency_seconds = sum(
        result["retrieval_latency_seconds"]
        for result in query_task_results
    )
    reranker_latency_seconds = sum(
        result["reranker_latency_seconds"]
        for result in query_task_results
    )

    return {
        "method": method,
        "fusion_strategy": fusion_strategy,
        "query_plan": query_plan,
        "query_count": len(query_tasks),
        "metrics": calculate_metrics(
            ranked_ids=ranked_ids,
            required_evidence=required_evidence,
        ),
        "required_evidence_ranks": get_ranks(
            ranked_ids=ranked_ids,
            required_evidence=required_evidence,
        ),
        "fusion_diagnostics": build_fusion_diagnostics(
            query_task_results=query_task_results,
            accumulated_chunks=accumulated_chunks,
            required_evidence=required_evidence,
            final_k=final_k,
        ),
        "query_task_traces": build_query_task_trace(
            query_task_results=query_task_results,
            required_evidence=required_evidence,
        ),
        "fusion_top20": compact_ranked_chunks(
            accumulated_chunks[:per_retriever_k]
        ),
        "latency": {
            "planner_seconds": planner_latency_seconds,
            "retrieval_seconds": retrieval_latency_seconds,
            "reranker_seconds": reranker_latency_seconds,
            "total_seconds": (
                time.perf_counter() - method_start_time
            ),
        },
    }


def evaluate_case(
    pipeline: RAGPipeline,
    case: dict,
    per_retriever_k: int,
    final_k: int,
    fusion_strategy: str = "round_robin",
) -> dict:
    query = case["query"]
    required_evidence = case["required_evidence"]

    original_start = time.perf_counter()
    original = run_query_tasks(
        pipeline=pipeline,
        method="original",
        query_tasks=build_original_tasks(query),
        required_evidence=required_evidence,
        per_retriever_k=per_retriever_k,
        final_k=final_k,
        method_start_time=original_start,
        fusion_strategy=fusion_strategy,
    )

    llm_start = time.perf_counter()
    planner_start = time.perf_counter()
    query_plan = pipeline.query_planner_split(query=query)
    planner_latency = time.perf_counter() - planner_start
    llm_tasks = pipeline.build_query_task(
        query=query,
        query_planner_output=query_plan,
    )
    llm_planner = run_query_tasks(
        pipeline=pipeline,
        method="llm_planner",
        query_tasks=llm_tasks,
        required_evidence=required_evidence,
        per_retriever_k=per_retriever_k,
        final_k=final_k,
        method_start_time=llm_start,
        planner_latency_seconds=planner_latency,
        query_plan=query_plan,
        fusion_strategy=fusion_strategy,
    )

    oracle_start = time.perf_counter()
    oracle = run_query_tasks(
        pipeline=pipeline,
        method="oracle",
        query_tasks=build_oracle_tasks(case),
        required_evidence=required_evidence,
        per_retriever_k=per_retriever_k,
        final_k=final_k,
        method_start_time=oracle_start,
        query_plan={
            "source": "human_gold",
            "subqueries": case["gold_subqueries"],
        },
        fusion_strategy=fusion_strategy,
    )

    return {
        "id": case["id"],
        "query": query,
        "task_type": case["task_type"],
        "required_evidence": required_evidence,
        "expected_points": case["expected_points"],
        "source_case_ids": case["source_case_ids"],
        "original": original,
        "llm_planner": llm_planner,
        "oracle": oracle,
    }


def summarize_values(values: list[float]) -> dict:
    sorted_values = sorted(values)
    p95_index = max(
        0,
        min(
            len(sorted_values) - 1,
            int(len(sorted_values) * 0.95 + 0.999999) - 1,
        ),
    )
    return {
        "average": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": sorted_values[p95_index],
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_method(
    case_results: list[dict],
    method: str,
) -> dict:
    method_results = [case[method] for case in case_results]
    ranking = {}

    for k in K_VALUES:
        k_text = str(k)
        metrics = [
            result["metrics"][k_text]
            for result in method_results
        ]
        ranking[k_text] = {
            "hit_rate": statistics.fmean(
                float(metric["hit"])
                for metric in metrics
            ),
            "mean_recall": statistics.fmean(
                metric["recall"]
                for metric in metrics
            ),
            "complete_rate": statistics.fmean(
                float(metric["complete"])
                for metric in metrics
            ),
            "mrr": statistics.fmean(
                metric["reciprocal_rank"]
                for metric in metrics
            ),
        }

    fusion_loss_case_ids = [
        case["id"]
        for case in case_results
        if case[method]["fusion_diagnostics"][
            "all_required_found_before_fusion"
        ]
        and not case[method]["metrics"]["5"]["complete"]
    ]
    pre_fusion_missing_case_ids = [
        case["id"]
        for case in case_results
        if not case[method]["fusion_diagnostics"][
            "all_required_found_before_fusion"
        ]
    ]

    return {
        "ranking": ranking,
        "average_query_count": statistics.fmean(
            result["query_count"]
            for result in method_results
        ),
        "total_latency_seconds": summarize_values(
            [
                result["latency"]["total_seconds"]
                for result in method_results
            ]
        ),
        "fusion_loss_case_ids_at_5": fusion_loss_case_ids,
        "pre_fusion_missing_case_ids": pre_fusion_missing_case_ids,
    }


def compare_methods(
    case_results: list[dict],
    baseline_method: str,
    candidate_method: str,
    k: int = 5,
) -> dict:
    k_text = str(k)
    rescued = []
    regressed = []

    for case in case_results:
        baseline_complete = case[baseline_method]["metrics"][
            k_text
        ]["complete"]
        candidate_complete = case[candidate_method]["metrics"][
            k_text
        ]["complete"]

        if not baseline_complete and candidate_complete:
            rescued.append(case["id"])
        elif baseline_complete and not candidate_complete:
            regressed.append(case["id"])

    return {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "k": k,
        "rescued_case_ids": rescued,
        "regressed_case_ids": regressed,
        "net_complete_case_delta": len(rescued) - len(regressed),
    }


def build_decision(summary: dict) -> dict:
    original_complete = summary["methods"]["original"][
        "ranking"
    ]["5"]["complete_rate"]
    llm_complete = summary["methods"]["llm_planner"][
        "ranking"
    ]["5"]["complete_rate"]
    oracle_complete = summary["methods"]["oracle"][
        "ranking"
    ]["5"]["complete_rate"]
    llm_fusion_losses = summary["methods"]["llm_planner"][
        "fusion_loss_case_ids_at_5"
    ]
    oracle_fusion_losses = summary["methods"]["oracle"][
        "fusion_loss_case_ids_at_5"
    ]

    if oracle_complete <= original_complete:
        if oracle_fusion_losses:
            outcome = "fusion_bottleneck"
            recommendation = (
                "Oracle 子查询已找到证据，但 Fusion 后仍有"
                "必需证据掉出 Top5；先比较 Fusion/Coverage，"
                "暂不调整 Planner Prompt。"
            )
        else:
            outcome = "oracle_no_net_gain"
            recommendation = (
                "停止把 Planner 当作当前主线；先检查任务设计、"
                "基础检索与证据集合。"
            )
    elif llm_fusion_losses or oracle_fusion_losses:
        outcome = "fusion_bottleneck_before_planner_tuning"
        recommendation = (
            "Oracle 已证明分解有上限收益，但 LLM/Oracle 均有"
            "证据在 Fusion 后掉出 Top5；先做 Fusion/Coverage "
            "对照，再判断剩余差距是否来自 Planner。"
        )
    elif llm_complete < oracle_complete:
        outcome = "planner_quality_gap"
        recommendation = (
            "Oracle 有收益但 LLM Planner 未达到上限；"
            "先分析拆分差距，再决定 Prompt 或 Few-shot。"
        )
    else:
        outcome = "planner_promising"
        recommendation = (
            "Planner 在开发集有净收益；冻结方案后再运行"
            " locked test，未验收前不进入 Function Calling。"
        )

    return {
        "outcome": outcome,
        "recommendation": recommendation,
        "complete_at_5": {
            "original": original_complete,
            "llm_planner": llm_complete,
            "oracle": oracle_complete,
        },
        "llm_fusion_loss_case_ids_at_5": llm_fusion_losses,
        "oracle_fusion_loss_case_ids_at_5": (
            oracle_fusion_losses
        ),
    }


def build_summary(
    case_results: list[dict],
    failed_cases: list[dict],
) -> dict:
    methods = {
        method: summarize_method(case_results, method)
        for method in ("original", "llm_planner", "oracle")
    }
    summary = {
        "successful_case_count": len(case_results),
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases,
        "methods": methods,
        "comparisons_at_5": {
            "llm_vs_original": compare_methods(
                case_results,
                "original",
                "llm_planner",
            ),
            "oracle_vs_original": compare_methods(
                case_results,
                "original",
                "oracle",
            ),
            "llm_vs_oracle": compare_methods(
                case_results,
                "oracle",
                "llm_planner",
            ),
        },
    }
    summary["decision"] = build_decision(summary)
    return summary


def build_locked_decision(summary: dict) -> dict:
    original_complete = summary["methods"]["original"]["ranking"][
        "5"
    ]["complete_rate"]
    llm_complete = summary["methods"]["llm_planner"]["ranking"][
        "5"
    ]["complete_rate"]
    oracle_complete = summary["methods"]["oracle"]["ranking"][
        "5"
    ]["complete_rate"]

    if oracle_complete <= original_complete:
        return {
            "outcome": "no_go_oracle_no_complete_gain",
            "recommendation": (
                "锁定集上 Oracle Complete@5 未超过 Original；"
                "停止扩展 Planner 链路，不进入 Function Calling，"
                "也不得针对 locked case 继续调参。"
            ),
            "complete_at_5": {
                "original": original_complete,
                "llm_planner": llm_complete,
                "oracle": oracle_complete,
            },
        }

    return {
        **summary["decision"],
        "locked_test_passed": llm_complete > original_complete,
    }


def build_environment_metadata() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    }


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown_report(
    dataset: dict,
    config: dict,
    summary: dict,
    fusion_strategy: str = "round_robin",
) -> str:
    lines = [
        "# Planner / Oracle Evaluation",
        "",
        f"- 数据集：`{dataset['dataset_id']}`",
        f"- Split：`{dataset['split']}`",
        f"- 成功样本：{summary['successful_case_count']}",
        f"- 失败样本：{summary['failed_case_count']}",
        f"- Candidate K：{config['retrieval']['per_retriever_k']}",
        f"- Final K：{config['retrieval']['final_k']}",
        f"- Fusion：`{fusion_strategy}`",
        "",
        "## 三组对照",
        "",
        "| Method | Complete@5 | Recall@5 | MRR@5 | Avg Queries | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in ("original", "llm_planner", "oracle"):
        method_summary = summary["methods"][method]
        rank_5 = method_summary["ranking"]["5"]
        latency = method_summary["total_latency_seconds"]["average"]
        lines.append(
            f"| {method} "
            f"| {format_rate(rank_5['complete_rate'])} "
            f"| {format_rate(rank_5['mean_recall'])} "
            f"| {rank_5['mrr']:.3f} "
            f"| {method_summary['average_query_count']:.2f} "
            f"| {latency:.3f}s |"
        )

    lines.extend(
        [
            "",
            "## 诊断",
            "",
        ]
    )

    for method in ("original", "llm_planner", "oracle"):
        method_summary = summary["methods"][method]
        lines.append(
            f"- `{method}` Fusion 前缺证据："
            f"{method_summary['pre_fusion_missing_case_ids']}"
        )
        lines.append(
            f"- `{method}` Fusion 前已找到、Top5 后丢失："
            f"{method_summary['fusion_loss_case_ids_at_5']}"
        )

    lines.extend(
        [
            "",
            "## 当前决策",
            "",
            f"- Outcome：`{summary['decision']['outcome']}`",
            f"- 建议：{summary['decision']['recommendation']}",
            "",
        ]
    )

    if dataset["split"] == "dev":
        lines.append(
            "> 这是开发集结果，可以用于定位和修改；"
            "锁定测试集尚未运行，不能作为最终验收结论。"
        )

    if dataset["split"] == "test_locked":
        lines.extend(
            [
                "",
                "> 这是锁定集最终验收结果，只能用于 go/no-go；",
                "> 不得根据失败案例继续调整 Prompt、Fusion 或参数。",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对比 Original、LLM Planner 与 Oracle Subqueries"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("dev", "locked"),
        default="dev",
    )
    parser.add_argument(
        "--allow-locked-evaluation",
        action="store_true",
        help="显式允许运行锁定测试集",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="使用已有逐案结果重建 Summary 和 Markdown，不重新推理",
    )
    parser.add_argument(
        "--fusion-strategy",
        choices=(
            "round_robin",
            "distribution_based",
            "distribution_based_original_anchor",
        ),
        default="round_robin",
        help="Original、LLM Planner、Oracle 共用的 Fusion 策略",
    )
    args = parser.parse_args()

    if args.dataset == "locked" and not args.allow_locked_evaluation:
        parser.error(
            "锁定集不得用于调参；确认方案已冻结后，"
            "请显式增加 --allow-locked-evaluation"
        )

    return args


def main() -> None:
    args = parse_args()
    config = load_json(BASELINE_CONFIG_PATH)
    validate_baseline_config(config)

    chunks_path = resolve_project_path(config["chunks"]["path"])
    natural_path = resolve_project_path(config["dataset"]["path"])
    dataset_paths = {
        split: resolve_project_path(item["path"])
        for split, item in config["planner_evaluation"].items()
    }
    chunks_info = load_json(chunks_path)
    natural_dataset = load_json(natural_path)
    dev_dataset = load_json(dataset_paths["dev"])
    locked_dataset = load_json(dataset_paths["locked"])
    natural_queries = {
        case["query"]
        for case in natural_dataset["cases"]
    }

    validate_planner_dataset(
        dev_dataset,
        chunks_info,
        natural_queries,
    )
    validate_planner_dataset(
        locked_dataset,
        chunks_info,
        natural_queries,
    )
    validate_split_isolation(dev_dataset, locked_dataset)

    dataset = (
        dev_dataset
        if args.dataset == "dev"
        else locked_dataset
    )

    output_stem = (
        "planner_oracle_dev"
        if args.dataset == "dev"
        else "planner_oracle_locked"
    )
    if args.fusion_strategy != "round_robin":
        output_stem = (
            f"{output_stem}_{args.fusion_strategy}"
        )
    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    output_json_path = output_dir / f"{output_stem}_results.json"
    output_markdown_path = output_dir / f"{output_stem}.md"

    if dataset["corpus_sha256"] != config["chunks"]["sha256"]:
        raise ValueError(
            "Planner 数据集的 corpus_sha256 与 Baseline 不一致"
        )

    if args.report_only:
        if not output_json_path.is_file():
            raise FileNotFoundError(
                f"没有可重建的逐案结果：{output_json_path}"
            )

        output_data = load_json(output_json_path)
        failed_cases = output_data.get("summary", {}).get(
            "failed_cases",
            [],
        )
        output_data["summary"] = build_summary(
            output_data["cases"],
            failed_cases,
        )
        if dataset["split"] == "test_locked":
            output_data["summary"]["decision"] = (
                build_locked_decision(output_data["summary"])
            )
        output_data["dataset_file"] = str(
            dataset_paths[args.dataset]
        )
        output_data["dataset_sha256"] = sha256_file(
            dataset_paths[args.dataset]
        )
        output_data["baseline_config_sha256"] = sha256_file(
            BASELINE_CONFIG_PATH
        )
        output_data["config"] = config
        output_data["report_rebuilt_at"] = (
            datetime.now().astimezone().isoformat()
        )
        output_json_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_markdown_path.write_text(
            build_markdown_report(
                dataset,
                config,
                output_data["summary"],
                fusion_strategy=args.fusion_strategy,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                output_data["summary"]["decision"],
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Rebuilt JSON: {output_json_path}")
        print(f"Rebuilt report: {output_markdown_path}")
        return

    print(
        f"Loaded {len(dataset['cases'])} cases from "
        f"{dataset['dataset_id']}",
        flush=True,
    )
    print("Building shared evaluation pipeline...", flush=True)
    pipeline = build_evaluation_pipeline(chunks_info, config)

    case_results = []
    failed_cases = []
    per_retriever_k = config["retrieval"]["per_retriever_k"]
    final_k = config["retrieval"]["final_k"]

    for index, case in enumerate(dataset["cases"], start=1):
        print(
            f"[{index}/{len(dataset['cases'])}] "
            f"{case['id']} {case['query']}",
            flush=True,
        )

        try:
            result = evaluate_case(
                pipeline=pipeline,
                case=case,
                per_retriever_k=per_retriever_k,
                final_k=final_k,
                fusion_strategy=args.fusion_strategy,
            )
            case_results.append(result)
            print(
                "  Complete@5 "
                f"O={result['original']['metrics']['5']['complete']} "
                f"L={result['llm_planner']['metrics']['5']['complete']} "
                f"Oracle={result['oracle']['metrics']['5']['complete']}",
                flush=True,
            )
        except Exception as error:
            failed_cases.append(
                {
                    "id": case["id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"  ERROR: {type(error).__name__}: {error}",
                flush=True,
            )

    if not case_results:
        raise RuntimeError("所有 Planner 专项评测样本均执行失败")

    summary = build_summary(case_results, failed_cases)
    if dataset["split"] == "test_locked":
        summary["decision"] = build_locked_decision(summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset_file": str(dataset_paths[args.dataset]),
        "dataset_sha256": sha256_file(dataset_paths[args.dataset]),
        "baseline_config_file": str(BASELINE_CONFIG_PATH),
        "baseline_config_sha256": sha256_file(
            BASELINE_CONFIG_PATH
        ),
        "chunks_file": str(chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "environment": build_environment_metadata(),
        "fusion_strategy": args.fusion_strategy,
        "config": config,
        "summary": summary,
        "cases": case_results,
    }
    output_json_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_markdown_path.write_text(
        build_markdown_report(
            dataset,
            config,
            summary,
            fusion_strategy=args.fusion_strategy,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))
    print(f"Saved JSON: {output_json_path}")
    print(f"Saved report: {output_markdown_path}")


if __name__ == "__main__":
    main()
