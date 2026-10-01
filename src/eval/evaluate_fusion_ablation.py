import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.eval.evaluate_planner_oracle import (
    BASELINE_CONFIG_PATH,
    K_VALUES,
    calculate_metrics,
    get_ranks,
    load_json,
    sha256_file,
    validate_baseline_config,
)


PLANNER_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "planner_oracle_dev_results.json"
)
NATURAL_CURRENT_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "low_pipeline_eval_results.json"
)
NATURAL_DISTRIBUTION_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "low_pipeline_distribution_based_eval_results.json"
)
OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "fusion_ablation_dev_results.json"
)
OUTPUT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "fusion_ablation_dev.md"
)
STRATEGIES = (
    "current_round_robin",
    "rrf",
    "weighted_rrf",
    "distribution_based",
    "distribution_based_original_anchor",
    "coverage_quota",
)


def deduplicate_in_order(chunk_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_ids))


def get_query_lists(method_result: dict) -> list[dict]:
    return [
        {
            "query_id": trace["query_id"],
            "query_source": trace["query_source"],
            "chunks": trace["reranker_top20"],
        }
        for trace in method_result["query_task_traces"]
    ]


def fuse_rrf(
    query_lists: list[dict],
    rank_constant: int = 60,
    original_weight: float = 1.0,
    planned_weight: float = 1.0,
) -> list[str]:
    scores = defaultdict(float)
    first_seen = {}
    next_seen = 0

    for query_list in query_lists:
        weight = (
            original_weight
            if query_list["query_source"] == "original"
            else planned_weight
        )

        for rank, chunk in enumerate(
            query_list["chunks"],
            start=1,
        ):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] += weight / (rank_constant + rank)

            if chunk_id not in first_seen:
                first_seen[chunk_id] = next_seen
                next_seen += 1

    return sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            first_seen[chunk_id],
        ),
    )


def fuse_distribution_based(query_lists: list[dict]) -> list[str]:
    best_scores = {}
    first_seen = {}
    next_seen = 0

    for query_list in query_lists:
        chunks = query_list["chunks"]

        if not chunks:
            continue

        scores = [
            float(chunk.get("reranker_score", 0.0))
            for chunk in chunks
        ]
        mean_score = statistics.fmean(scores)
        std_dev = math.sqrt(
            statistics.fmean(
                (score - mean_score) ** 2
                for score in scores
            )
        )
        minimum = mean_score - 3 * std_dev
        maximum = mean_score + 3 * std_dev
        delta = maximum - minimum
        scores_vary = min(scores) != max(scores)

        for chunk, score in zip(chunks, scores):
            chunk_id = chunk["chunk_id"]
            normalized = (
                (score - minimum) / delta
                if scores_vary and delta != 0.0
                else score
            )

            if chunk_id not in first_seen:
                first_seen[chunk_id] = next_seen
                next_seen += 1

            if (
                chunk_id not in best_scores
                or normalized > best_scores[chunk_id]
            ):
                best_scores[chunk_id] = normalized

    return sorted(
        best_scores,
        key=lambda chunk_id: (
            -best_scores[chunk_id],
            first_seen[chunk_id],
        ),
    )


def anchor_original_top1(
    ranked_ids: list[str],
    query_lists: list[dict],
) -> list[str]:
    original_top1_id = next(
        (
            query_list["chunks"][0]["chunk_id"]
            for query_list in query_lists
            if query_list["query_source"] == "original"
            and query_list["chunks"]
        ),
        None,
    )

    if original_top1_id is None:
        return ranked_ids

    return [original_top1_id] + [
        chunk_id
        for chunk_id in ranked_ids
        if chunk_id != original_top1_id
    ]


def fuse_coverage_quota(
    query_lists: list[dict],
    final_k: int,
) -> list[str]:
    planned_lists = [
        query_list
        for query_list in query_lists
        if query_list["query_source"] != "original"
    ]
    original_lists = [
        query_list
        for query_list in query_lists
        if query_list["query_source"] == "original"
    ]

    if not planned_lists:
        return deduplicate_in_order(
            [
                chunk["chunk_id"]
                for query_list in original_lists
                for chunk in query_list["chunks"]
            ]
        )

    quota = max(1, final_k // len(planned_lists))
    selected = []
    seen = set()

    for query_list in planned_lists:
        accepted = 0

        for chunk in query_list["chunks"]:
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen:
                continue

            selected.append(chunk_id)
            seen.add(chunk_id)
            accepted += 1

            if accepted >= quota:
                break

    ordered_lists = planned_lists + original_lists
    max_length = max(
        len(query_list["chunks"])
        for query_list in ordered_lists
    )

    for rank_index in range(max_length):
        for query_list in ordered_lists:
            if rank_index >= len(query_list["chunks"]):
                continue

            chunk_id = query_list["chunks"][rank_index][
                "chunk_id"
            ]

            if chunk_id in seen:
                continue

            selected.append(chunk_id)
            seen.add(chunk_id)

    return selected


def fuse_method_result(
    method_result: dict,
    strategy: str,
    rank_constant: int,
    final_k: int,
) -> list[str]:
    if strategy == "current_round_robin":
        return [
            chunk["chunk_id"]
            for chunk in method_result["fusion_top20"]
        ]

    query_lists = get_query_lists(method_result)

    if strategy == "rrf":
        return fuse_rrf(
            query_lists=query_lists,
            rank_constant=rank_constant,
        )

    if strategy == "weighted_rrf":
        return fuse_rrf(
            query_lists=query_lists,
            rank_constant=rank_constant,
            original_weight=0.5,
            planned_weight=1.0,
        )

    if strategy == "distribution_based":
        return fuse_distribution_based(query_lists)

    if strategy == "distribution_based_original_anchor":
        return anchor_original_top1(
            ranked_ids=fuse_distribution_based(query_lists),
            query_lists=query_lists,
        )

    if strategy == "coverage_quota":
        return fuse_coverage_quota(
            query_lists=query_lists,
            final_k=final_k,
        )

    raise ValueError(f"未知 Fusion 策略：{strategy}")


def evaluate_cases(
    planner_results: dict,
    rank_constant: int,
    final_k: int,
) -> list[dict]:
    case_results = []

    for case in planner_results["cases"]:
        required_evidence = case["required_evidence"]
        case_result = {
            "id": case["id"],
            "query": case["query"],
            "required_evidence": required_evidence,
            "original": case["original"]["metrics"],
        }

        for method in ("llm_planner", "oracle"):
            method_results = {}

            for strategy in STRATEGIES:
                ranked_ids = fuse_method_result(
                    method_result=case[method],
                    strategy=strategy,
                    rank_constant=rank_constant,
                    final_k=final_k,
                )
                method_results[strategy] = {
                    "ranked_ids": ranked_ids,
                    "required_evidence_ranks": get_ranks(
                        ranked_ids,
                        required_evidence,
                    ),
                    "metrics": calculate_metrics(
                        ranked_ids,
                        required_evidence,
                    ),
                }

            case_result[method] = method_results

        case_results.append(case_result)

    return case_results


def summarize_strategy(
    case_results: list[dict],
    method: str,
    strategy: str,
) -> dict:
    ranking = {}

    for k in K_VALUES:
        metrics = [
            case[method][strategy]["metrics"][str(k)]
            for case in case_results
        ]
        ranking[str(k)] = {
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

    current_complete = {
        case["id"]: case[method]["current_round_robin"][
            "metrics"
        ]["5"]["complete"]
        for case in case_results
    }
    candidate_complete = {
        case["id"]: case[method][strategy]["metrics"]["5"][
            "complete"
        ]
        for case in case_results
    }
    rescued = [
        case_id
        for case_id in current_complete
        if not current_complete[case_id]
        and candidate_complete[case_id]
    ]
    regressed = [
        case_id
        for case_id in current_complete
        if current_complete[case_id]
        and not candidate_complete[case_id]
    ]

    return {
        "ranking": ranking,
        "rescued_vs_current_at_5": rescued,
        "regressed_vs_current_at_5": regressed,
        "net_complete_case_delta_vs_current": (
            len(rescued) - len(regressed)
        ),
    }


def choose_candidate(method_summary: dict) -> dict:
    ranked = sorted(
        STRATEGIES,
        key=lambda strategy: (
            -method_summary[strategy]["ranking"]["5"][
                "complete_rate"
            ],
            len(
                method_summary[strategy][
                    "regressed_vs_current_at_5"
                ]
            ),
            -method_summary[strategy]["ranking"]["5"]["mrr"],
            STRATEGIES.index(strategy),
        ),
    )
    best = ranked[0]
    return {
        "strategy": best,
        "summary": method_summary[best],
        "beats_current_complete_at_5": (
            method_summary[best]["ranking"]["5"]["complete_rate"]
            > method_summary["current_round_robin"]["ranking"]["5"]
            ["complete_rate"]
        ),
    }


def build_summary(case_results: list[dict]) -> dict:
    methods = {}

    for method in ("llm_planner", "oracle"):
        strategy_summary = {
            strategy: summarize_strategy(
                case_results,
                method,
                strategy,
            )
            for strategy in STRATEGIES
        }
        methods[method] = {
            "strategies": strategy_summary,
            "candidate": choose_candidate(strategy_summary),
        }

    return {
        "case_count": len(case_results),
        "methods": methods,
        "decision": {
            "llm_candidate": methods["llm_planner"]["candidate"],
            "oracle_candidate": methods["oracle"]["candidate"],
            "next_step": (
                "只将开发集最优策略作为候选；先做代码级集成和"
                " natural_v1 回归，再冻结方案并运行 locked test。"
            ),
        },
    }


def summarize_natural_regression() -> dict:
    current_results = load_json(NATURAL_CURRENT_RESULTS_PATH)
    distribution_results = load_json(
        NATURAL_DISTRIBUTION_RESULTS_PATH
    )
    current_by_id = {
        case["id"]: case
        for case in current_results["cases"]
        if case["error"] is None
    }
    distribution_by_id = {
        case["id"]: case
        for case in distribution_results["cases"]
        if case["error"] is None
    }

    if current_by_id.keys() != distribution_by_id.keys():
        raise ValueError(
            "自然集 current 与 distribution-based 的成功样本不一致"
        )

    strategy_metrics = {
        "current_round_robin": [],
        "distribution_based": [],
        "distribution_based_original_anchor": [],
    }
    case_results = []

    for case_id, distribution_case in distribution_by_id.items():
        current_case = current_by_id[case_id]
        relevant_ids = distribution_case["relevant_chunk_ids"]
        original_top1 = distribution_case["minimal"][
            "top20_chunk_ids"
        ][0]
        distribution_ids = distribution_case["low"][
            "top20_chunk_ids"
        ]
        anchored_ids = [original_top1] + [
            chunk_id
            for chunk_id in distribution_ids
            if chunk_id != original_top1
        ]
        ranked_by_strategy = {
            "current_round_robin": current_case["low"][
                "top20_chunk_ids"
            ],
            "distribution_based": distribution_ids,
            "distribution_based_original_anchor": anchored_ids,
        }
        case_result = {"id": case_id}

        for strategy, ranked_ids in ranked_by_strategy.items():
            metrics = calculate_metrics(ranked_ids, relevant_ids)
            strategy_metrics[strategy].append(metrics)
            case_result[strategy] = {
                "required_evidence_ranks": get_ranks(
                    ranked_ids,
                    relevant_ids,
                ),
                "metrics": metrics,
            }

        case_results.append(case_result)

    summary = {}

    for strategy, metrics_list in strategy_metrics.items():
        ranking = {}

        for k in K_VALUES:
            metrics_at_k = [
                metrics[str(k)] for metrics in metrics_list
            ]
            ranking[str(k)] = {
                "complete_rate": statistics.fmean(
                    float(metric["complete"])
                    for metric in metrics_at_k
                ),
                "mean_recall": statistics.fmean(
                    metric["recall"] for metric in metrics_at_k
                ),
                "mrr": statistics.fmean(
                    metric["reciprocal_rank"]
                    for metric in metrics_at_k
                ),
            }

        summary[strategy] = {"ranking": ranking}

    return {
        "case_count": len(case_results),
        "source_files": {
            "current_round_robin": str(
                NATURAL_CURRENT_RESULTS_PATH
            ),
            "distribution_based": str(
                NATURAL_DISTRIBUTION_RESULTS_PATH
            ),
        },
        "source_sha256": {
            "current_round_robin": sha256_file(
                NATURAL_CURRENT_RESULTS_PATH
            ),
            "distribution_based": sha256_file(
                NATURAL_DISTRIBUTION_RESULTS_PATH
            ),
        },
        "strategies": summary,
        "cases": case_results,
    }


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown(
    summary: dict,
    natural_regression: dict,
) -> str:
    lines = [
        "# Fusion Ablation - Planner Dev",
        "",
        "本报告离线复用 Planner / Oracle 已保存的每个 Query Reranker Top20，",
        "不重新调用模型，也不运行 locked test。",
        "",
    ]

    for method in ("llm_planner", "oracle"):
        lines.extend(
            [
                f"## {method}",
                "",
                "| Strategy | Complete@5 | Recall@5 | MRR@5 | Net Complete vs Current |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for strategy in STRATEGIES:
            item = summary["methods"][method]["strategies"][
                strategy
            ]
            rank_5 = item["ranking"]["5"]
            lines.append(
                f"| {strategy} "
                f"| {format_rate(rank_5['complete_rate'])} "
                f"| {format_rate(rank_5['mean_recall'])} "
                f"| {rank_5['mrr']:.3f} "
                f"| {item['net_complete_case_delta_vs_current']:+d} |"
            )

        candidate = summary["methods"][method]["candidate"]
        lines.extend(
            [
                "",
                f"候选：`{candidate['strategy']}`。",
                "",
            ]
        )

    lines.extend(
        [
            "## Natural v1 回归",
            "",
            "锚点策略由已保存的 distribution-based 排名确定性重排得到，",
            "没有重新调用模型。",
            "",
            "| Strategy | Complete@5 | Recall@5 | MRR@5 |",
            "|---|---:|---:|---:|",
        ]
    )

    for strategy, item in natural_regression["strategies"].items():
        rank_5 = item["ranking"]["5"]
        lines.append(
            f"| {strategy} "
            f"| {format_rate(rank_5['complete_rate'])} "
            f"| {format_rate(rank_5['mean_recall'])} "
            f"| {rank_5['mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 决策边界",
            "",
            summary["decision"]["next_step"],
            "",
            "本结果来自 dev，不得用 locked test 继续选策略。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = load_json(BASELINE_CONFIG_PATH)
    validate_baseline_config(config)

    if not PLANNER_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"请先运行 Planner / Oracle dev 评测：{PLANNER_RESULTS_PATH}"
        )

    planner_results = load_json(PLANNER_RESULTS_PATH)
    expected_dataset_hash = config["planner_evaluation"]["dev"][
        "sha256"
    ]

    if planner_results["dataset_sha256"] != expected_dataset_hash:
        raise ValueError(
            "Planner dev 结果与当前 dev 数据集 SHA256 不一致"
        )

    case_results = evaluate_cases(
        planner_results=planner_results,
        rank_constant=config["retrieval"]["rank_constant"],
        final_k=config["retrieval"]["final_k"],
    )
    summary = build_summary(case_results)
    natural_regression = summarize_natural_regression()
    output = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_results_file": str(PLANNER_RESULTS_PATH),
        "source_results_sha256": sha256_file(
            PLANNER_RESULTS_PATH
        ),
        "dataset_sha256": expected_dataset_hash,
        "strategies": {
            "current_round_robin": "当前按 Query 与 rank 逐层交错的顺序",
            "rrf": "等权 Reciprocal Rank Fusion，k=60",
            "weighted_rrf": "原问题权重0.5，计划子查询权重1.0",
            "distribution_based": (
                "按每个 Query 的 Reranker 分数均值和标准差归一化，"
                "重复 Chunk 取最高归一化分数"
            ),
            "distribution_based_original_anchor": (
                "Distribution-based Fusion 后保留 Original Query Top1，"
                "兼顾多证据覆盖与首条结果稳定性"
            ),
            "coverage_quota": (
                "先为每个计划子查询分配 final_k // query_count 个唯一候选，"
                "再按 round-robin 补齐"
            ),
        },
        "summary": summary,
        "natural_regression": natural_regression,
        "cases": case_results,
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MARKDOWN_PATH.write_text(
        build_markdown(summary, natural_regression),
        encoding="utf-8",
    )

    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))
    print(f"Saved JSON: {OUTPUT_JSON_PATH}")
    print(f"Saved report: {OUTPUT_MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
