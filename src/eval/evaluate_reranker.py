import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reranker.reranker import CrossEncoderReranker


CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_info.json"
RETRIEVER_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "retriever_eval_results.json"
)
OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "reranker_eval_results.json"
)
OUTPUT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "reranker_eval.md"
)

CANDIDATE_K = 20
FINAL_K = 5
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_hybrid_result(retriever_results: dict) -> dict:
    for result in retriever_results["results"]:
        if result["retriever"] == "hybrid":
            return result

    raise ValueError("Retriever 评测结果中没有找到 hybrid 结果。")


def build_chunk_lookup(chunks_info: list[dict]) -> dict[str, dict]:
    chunk_lookup = {}

    for chunk in chunks_info:
        chunk_id = chunk["id"]

        if chunk_id in chunk_lookup:
            raise ValueError(f"chunks_info.json 中存在重复 ID：{chunk_id}")

        chunk_lookup[chunk_id] = chunk

    return chunk_lookup


def validate_hybrid_cases(
    hybrid_cases: list[dict],
    chunk_lookup: dict[str, dict],
) -> None:
    missing_chunk_ids = set()

    for case in hybrid_cases:
        candidate_ids = case["retrieved_chunk_ids"][:CANDIDATE_K]

        if len(candidate_ids) < CANDIDATE_K:
            raise ValueError(
                f"{case['id']} 的 Hybrid 候选不足 {CANDIDATE_K} 条，"
                f"实际只有 {len(candidate_ids)} 条。"
            )

        missing_chunk_ids.update(
            chunk_id
            for chunk_id in candidate_ids + case["relevant_chunk_ids"]
            if chunk_id not in chunk_lookup
        )

    if missing_chunk_ids:
        missing_text = "\n".join(sorted(missing_chunk_ids))
        raise ValueError(
            "以下候选或正确证据 ID 不在当前 chunks_info.json 中：\n"
            + missing_text
        )


def calculate_top_k_metrics(
    ranked_ids: list[str],
    relevant_ids: list[str],
    top_k: int,
) -> dict:
    top_k_ids = ranked_ids[:top_k]
    relevant_set = set(relevant_ids)
    found_ids = relevant_set.intersection(top_k_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(top_k_ids, start=1)
            if chunk_id in relevant_set
        ),
        None,
    )

    return {
        "hit": bool(found_ids),
        "recall": len(found_ids) / len(relevant_set),
        "complete": relevant_set.issubset(top_k_ids),
        "reciprocal_rank": (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
    }


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


def summarize_top5(
    case_results: list[dict],
    metrics_key: str,
) -> dict:
    metrics = [result[metrics_key] for result in case_results]
    multi_evidence_metrics = [
        result[metrics_key]
        for result in case_results
        if result["evidence_mode"] == "all_required"
    ]

    return {
        "hit_rate": sum(item["hit"] for item in metrics) / len(metrics),
        "mean_recall": (
            sum(item["recall"] for item in metrics) / len(metrics)
        ),
        "complete_rate": (
            sum(item["complete"] for item in metrics) / len(metrics)
        ),
        "mrr": (
            sum(item["reciprocal_rank"] for item in metrics)
            / len(metrics)
        ),
        "multi_evidence_complete_rate": (
            sum(item["complete"] for item in multi_evidence_metrics)
            / len(multi_evidence_metrics)
            if multi_evidence_metrics
            else None
        ),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def evaluate_case(
    case: dict,
    chunk_lookup: dict[str, dict],
    reranker: CrossEncoderReranker,
) -> dict:
    hybrid_ids = case["retrieved_chunk_ids"][:CANDIDATE_K]
    hybrid_candidates = []

    for hybrid_rank, chunk_id in enumerate(hybrid_ids, start=1):
        hybrid_candidates.append(
            {
                **chunk_lookup[chunk_id],
                "hybrid_rank": hybrid_rank,
            }
        )

    reranked_chunks = reranker.rerank(
        query=case["query"],
        candidates=hybrid_candidates,
    )
    reranked_ids = [chunk["id"] for chunk in reranked_chunks]
    relevant_ids = case["relevant_chunk_ids"]

    hybrid_ranks = get_relevant_ranks(hybrid_ids, relevant_ids)
    reranker_ranks = get_relevant_ranks(reranked_ids, relevant_ids)
    hidden_relevant_ids = [
        chunk_id
        for chunk_id, rank in hybrid_ranks.items()
        if rank is not None and FINAL_K < rank <= CANDIDATE_K
    ]
    rescued_relevant_ids = [
        chunk_id
        for chunk_id in hidden_relevant_ids
        if reranker_ranks[chunk_id] is not None
        and reranker_ranks[chunk_id] <= FINAL_K
    ]
    dropped_relevant_ids = [
        chunk_id
        for chunk_id, rank in hybrid_ranks.items()
        if rank is not None
        and rank <= FINAL_K
        and (
            reranker_ranks[chunk_id] is None
            or reranker_ranks[chunk_id] > FINAL_K
        )
    ]

    hybrid_metrics = calculate_top_k_metrics(
        ranked_ids=hybrid_ids,
        relevant_ids=relevant_ids,
        top_k=FINAL_K,
    )
    reranker_metrics = calculate_top_k_metrics(
        ranked_ids=reranked_ids,
        relevant_ids=relevant_ids,
        top_k=FINAL_K,
    )
    eligible_for_complete_rescue = (
        not hybrid_metrics["complete"]
        and set(relevant_ids).issubset(hybrid_ids)
    )

    rank_changes = {
        chunk_id: {
            "hybrid_rank": hybrid_ranks[chunk_id],
            "reranker_rank": reranker_ranks[chunk_id],
            "rank_improvement": (
                hybrid_ranks[chunk_id] - reranker_ranks[chunk_id]
                if hybrid_ranks[chunk_id] is not None
                and reranker_ranks[chunk_id] is not None
                else None
            ),
        }
        for chunk_id in relevant_ids
    }

    return {
        "id": case["id"],
        "query": case["query"],
        "evidence_mode": case["evidence_mode"],
        "relevant_chunk_ids": relevant_ids,
        "hybrid_top5_chunk_ids": hybrid_ids[:FINAL_K],
        "reranker_top5_chunk_ids": reranked_ids[:FINAL_K],
        "rank_changes": rank_changes,
        "hidden_relevant_chunk_ids": hidden_relevant_ids,
        "rescued_relevant_chunk_ids": rescued_relevant_ids,
        "dropped_relevant_chunk_ids": dropped_relevant_ids,
        "eligible_for_complete_rescue": eligible_for_complete_rescue,
        "complete_rescued": (
            eligible_for_complete_rescue
            and reranker_metrics["complete"]
        ),
        "hybrid_metrics_at_5": hybrid_metrics,
        "reranker_metrics_at_5": reranker_metrics,
        "reranker_top20": [
            {
                "rank": rank,
                "chunk_id": chunk["id"],
                "reranker_score": chunk["reranker_score"],
            }
            for rank, chunk in enumerate(reranked_chunks, start=1)
        ],
    }


def build_summary(case_results: list[dict]) -> dict:
    hybrid_top5 = summarize_top5(
        case_results=case_results,
        metrics_key="hybrid_metrics_at_5",
    )
    reranker_top5 = summarize_top5(
        case_results=case_results,
        metrics_key="reranker_metrics_at_5",
    )
    eligible_cases = [
        result
        for result in case_results
        if result["eligible_for_complete_rescue"]
    ]
    rescued_cases = [
        result for result in eligible_cases if result["complete_rescued"]
    ]
    hidden_relevant_chunk_count = sum(
        len(result["hidden_relevant_chunk_ids"])
        for result in case_results
    )
    rescued_relevant_chunk_count = sum(
        len(result["rescued_relevant_chunk_ids"])
        for result in case_results
    )
    baseline_complete_cases = [
        result
        for result in case_results
        if result["hybrid_metrics_at_5"]["complete"]
    ]
    regressed_cases = [
        result
        for result in baseline_complete_cases
        if not result["reranker_metrics_at_5"]["complete"]
    ]

    metric_deltas = {
        key: reranker_top5[key] - hybrid_top5[key]
        if reranker_top5[key] is not None
        and hybrid_top5[key] is not None
        else None
        for key in hybrid_top5
    }

    return {
        "case_count": len(case_results),
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "hybrid_top5": hybrid_top5,
        "reranker_top5": reranker_top5,
        "top5_metric_deltas": metric_deltas,
        "eligible_rescue_case_count": len(eligible_cases),
        "rescued_case_count": len(rescued_cases),
        "case_rescue_rate": safe_rate(
            len(rescued_cases),
            len(eligible_cases),
        ),
        "hidden_relevant_chunk_count": hidden_relevant_chunk_count,
        "rescued_relevant_chunk_count": rescued_relevant_chunk_count,
        "hidden_chunk_rescue_rate": safe_rate(
            rescued_relevant_chunk_count,
            hidden_relevant_chunk_count,
        ),
        "baseline_complete_case_count": len(baseline_complete_cases),
        "regressed_complete_case_count": len(regressed_cases),
        "complete_regression_rate": safe_rate(
            len(regressed_cases),
            len(baseline_complete_cases),
        ),
        "rescued_case_ids": [result["id"] for result in rescued_cases],
        "regressed_case_ids": [result["id"] for result in regressed_cases],
    }


def format_rate(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def build_markdown_report(
    summary: dict,
    case_results: list[dict],
) -> str:
    hybrid = summary["hybrid_top5"]
    reranker = summary["reranker_top5"]
    deltas = summary["top5_metric_deltas"]
    lines = [
        "# Reranker Evaluation Record",
        "",
        "## 评测设置",
        "",
        f"- Hybrid 固定候选数：Top{CANDIDATE_K}",
        f"- Reranker 最终输出：Top{FINAL_K}",
        f"- Reranker 模型：`{RERANKER_MODEL_NAME}`",
        "- 候选来源：现有 Retriever 评测保存的 Hybrid Top20",
        "",
        "## Top5 前后对比",
        "",
        "| 指标 | Hybrid Top5 | Reranker Top5 | 变化 |",
        "|---|---:|---:|---:|",
        (
            f"| Hit@5 | {hybrid['hit_rate']:.1%} | "
            f"{reranker['hit_rate']:.1%} | {deltas['hit_rate']:+.1%} |"
        ),
        (
            f"| Mean Recall@5 | {hybrid['mean_recall']:.1%} | "
            f"{reranker['mean_recall']:.1%} | "
            f"{deltas['mean_recall']:+.1%} |"
        ),
        (
            f"| Complete@5 | {hybrid['complete_rate']:.1%} | "
            f"{reranker['complete_rate']:.1%} | "
            f"{deltas['complete_rate']:+.1%} |"
        ),
        (
            f"| MRR@5 | {hybrid['mrr']:.3f} | "
            f"{reranker['mrr']:.3f} | {deltas['mrr']:+.3f} |"
        ),
        (
            "| Multi-Complete@5 | "
            f"{format_rate(hybrid['multi_evidence_complete_rate'])} | "
            f"{format_rate(reranker['multi_evidence_complete_rate'])} | "
            f"{deltas['multi_evidence_complete_rate']:+.1%} |"
        ),
        "",
        "## 拉回与破坏情况",
        "",
        (
            f"- 可拉回案例：{summary['eligible_rescue_case_count']} 个"
        ),
        (
            f"- 成功完整拉回：{summary['rescued_case_count']} 个，"
            f"案例拉回率 {format_rate(summary['case_rescue_rate'])}"
        ),
        (
            f"- 位于 Hybrid 第6～20名的正确 chunk："
            f"{summary['hidden_relevant_chunk_count']} 个"
        ),
        (
            f"- 成功进入 Reranker Top5："
            f"{summary['rescued_relevant_chunk_count']} 个，"
            f"chunk 拉回率 "
            f"{format_rate(summary['hidden_chunk_rescue_rate'])}"
        ),
        (
            f"- 原本 Complete@5 的案例："
            f"{summary['baseline_complete_case_count']} 个"
        ),
        (
            f"- 被 Reranker 破坏完整性的案例："
            f"{summary['regressed_complete_case_count']} 个，"
            f"破坏率 {format_rate(summary['complete_regression_rate'])}"
        ),
        "",
        "## 为什么多依据问题可能在重排后回退",
        "",
        "Cross Encoder 会把同一个 query 分别与每个候选 chunk 组成输入，",
        "并为每一对输入独立计算一个相关性分数：",
        "",
        "```text",
        "query + chunk1 -> score1",
        "query + chunk2 -> score2",
        "query + chunk3 -> score3",
        "```",
        "",
        "最后按照单个 chunk 的分数从高到低排序。计算 `chunk2` 分数时，",
        "模型不知道 `chunk1` 是否已经进入 Top5，也不知道 `chunk1` 已经",
        "覆盖了问题中的哪一部分。因此，它优化的是单个 query-chunk 对的",
        "相关性，而不是 Top5 证据集合对整个问题的覆盖完整性。",
        "",
        "例如一个问题同时包含 A、B 两个子问题，`chunk_A` 只回答 A，",
        "`chunk_B` 只回答 B。即使 `chunk_A` 已经被排进 Top5，",
        "Cross Encoder 也不会因此给 `chunk_B` 增加分数。其他几个与 A",
        "高度相似但内容重复的 chunk，仍可能排在 `chunk_B` 前面，最终形成",
        "“Top5 中有多份证据都回答 A，但缺少回答 B 的证据”的结果。",
        "",
        "本轮两个回退案例正是这种情况：",
        "",
        "- `bank_eval_028` 的一个正确 chunk 保持第1名，另一个从第2名",
        "  降到第6名，导致“不执行指令的其他合同条件”没有进入 Top5。",
        "- `bank_eval_050` 的一个正确 chunk 从第4名升到第1名，另一个从",
        "  第3名降到第7名，导致“非本人交易的查询、投诉和报案流程”没有",
        "  进入 Top5。",
        "",
        "这也解释了为什么 `Hit@5` 可以达到100%，但 `Complete@5` 只有",
        "96%：`Hit@5` 只要求至少命中一份正确证据，而 `Complete@5` 要求",
        "多依据问题所需的全部正确证据都进入 Top5。对于最终需要综合多份",
        "资料回答的问题，`Complete@5` 和 `Multi-Complete@5` 比单独观察",
        "`Hit@5` 更重要。",
        "",
        "因此，这两个回退案例不属于 Retriever 没有找回证据，因为正确",
        "证据已经存在于 Hybrid Top20；它们反映的是独立相关性排序与证据",
        "集合覆盖目标之间的差异。后续可通过多意图问题扩充、子问题拆分、",
        "覆盖感知的证据选择，或 Top5 与 Top7 的生成效果对比继续验证。",
        "",
        "## 可拉回案例明细",
        "",
        "| Case | Hybrid正确证据排名 | Reranker正确证据排名 | 完整拉回 |",
        "|---|---|---|---|",
    ]

    for result in case_results:
        if not result["eligible_for_complete_rescue"]:
            continue

        hybrid_ranks = ", ".join(
            f"{chunk_id}: {change['hybrid_rank']}"
            for chunk_id, change in result["rank_changes"].items()
        )
        reranker_ranks = ", ".join(
            f"{chunk_id}: {change['reranker_rank']}"
            for chunk_id, change in result["rank_changes"].items()
        )
        rescued_text = "是" if result["complete_rescued"] else "否"
        lines.append(
            f"| {result['id']} | {hybrid_ranks} | "
            f"{reranker_ranks} | {rescued_text} |"
        )

    if summary["regressed_case_ids"]:
        lines.extend(
            [
                "",
                "## 回退案例",
                "",
                "以下案例原本在 Hybrid Top5 中证据完整，"
                "但经过 Reranker 后不再完整：",
                "",
            ]
        )
        lines.extend(
            f"- {case_id}" for case_id in summary["regressed_case_ids"]
        )

    return "\n".join(lines) + "\n"


def print_summary(summary: dict) -> None:
    hybrid = summary["hybrid_top5"]
    reranker = summary["reranker_top5"]

    print("\n" + "=" * 88)
    print(
        f"Reranker: {RERANKER_MODEL_NAME} | "
        f"Hybrid Top{CANDIDATE_K} -> Reranker Top{FINAL_K}"
    )
    print("=" * 88)
    print(
        f"{'Metric':<24} "
        f"{'Hybrid Top5':>14} "
        f"{'Reranker Top5':>16}"
    )
    print(
        f"{'Hit@5':<24} "
        f"{hybrid['hit_rate']:>14.3f} "
        f"{reranker['hit_rate']:>16.3f}"
    )
    print(
        f"{'Mean Recall@5':<24} "
        f"{hybrid['mean_recall']:>14.3f} "
        f"{reranker['mean_recall']:>16.3f}"
    )
    print(
        f"{'Complete@5':<24} "
        f"{hybrid['complete_rate']:>14.3f} "
        f"{reranker['complete_rate']:>16.3f}"
    )
    print(
        f"{'MRR@5':<24} "
        f"{hybrid['mrr']:>14.3f} "
        f"{reranker['mrr']:>16.3f}"
    )
    print("-" * 88)
    print(
        "Eligible rescue cases: "
        f"{summary['eligible_rescue_case_count']}"
    )
    print(
        "Rescued complete cases: "
        f"{summary['rescued_case_count']} "
        f"({format_rate(summary['case_rescue_rate'])})"
    )
    print(
        "Hidden relevant chunks rescued: "
        f"{summary['rescued_relevant_chunk_count']}/"
        f"{summary['hidden_relevant_chunk_count']} "
        f"({format_rate(summary['hidden_chunk_rescue_rate'])})"
    )
    print(
        "Complete@5 regressions: "
        f"{summary['regressed_complete_case_count']}/"
        f"{summary['baseline_complete_case_count']} "
        f"({format_rate(summary['complete_regression_rate'])})"
    )


def main() -> None:
    chunks_info = load_json(CHUNKS_PATH)
    retriever_results = load_json(RETRIEVER_RESULTS_PATH)
    hybrid_result = get_hybrid_result(retriever_results)
    hybrid_cases = hybrid_result["cases"]
    chunk_lookup = build_chunk_lookup(chunks_info)

    validate_hybrid_cases(
        hybrid_cases=hybrid_cases,
        chunk_lookup=chunk_lookup,
    )

    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL_NAME)
    case_results = [
        evaluate_case(
            case=case,
            chunk_lookup=chunk_lookup,
            reranker=reranker,
        )
        for case in hybrid_cases
    ]
    summary = build_summary(case_results)

    output_data = {
        "chunks_file": str(CHUNKS_PATH),
        "retriever_results_file": str(RETRIEVER_RESULTS_PATH),
        "reranker_model": RERANKER_MODEL_NAME,
        "summary": summary,
        "cases": case_results,
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    markdown_report = build_markdown_report(
        summary=summary,
        case_results=case_results,
    )
    OUTPUT_MARKDOWN_PATH.write_text(markdown_report, encoding="utf-8")

    print_summary(summary)
    print(f"\n详细结果已保存到：{OUTPUT_JSON_PATH}")
    print(f"评测报告已保存到：{OUTPUT_MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
