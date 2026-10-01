import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "retriever_eval_results.json"
)

K_VALUES = (1, 3, 5, 10, 20)
RETRIEVERS_TO_TEST = ("vector", "bm25", "hybrid")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_retrieval_cases() -> list[dict]:
    evaluation_data = load_json(EVALUATION_PATH)

    return [
        case
        for case in evaluation_data["cases"]
        if "retriever" in case["use_for"] and not case["should_refuse"]
    ]


def validate_evidence_chunk_ids(
    cases: list[dict],
    chunks_info: list[dict],
) -> None:
    existing_chunk_ids = {chunk["id"] for chunk in chunks_info}
    missing_chunk_ids = sorted(
        {
            chunk_id
            for case in cases
            for chunk_id in case["evidence_chunk_ids"]
            if chunk_id not in existing_chunk_ids
        }
    )

    if missing_chunk_ids:
        missing_text = "\n".join(missing_chunk_ids)
        raise ValueError(
            "测试集中的以下证据 chunk 不在当前 chunks_info.json 中：\n"
            + missing_text
        )


def build_retrievers(chunks_info: list[dict]) -> dict:
    vector_retriever = VectorRetriever()
    vector_retriever.fit(chunks_info)

    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(chunks_info)

    hybrid_retriever = HybridRetriever(
        denseretriever=vector_retriever,
        sparseretriever=bm25_retriever,
        rank_constant=60,
        per_retriever_k=max(K_VALUES),
    )

    return {
        "vector": vector_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever,
    }


def calculate_case_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> dict:
    relevant_set = set(relevant_ids)
    metrics_by_k = {}

    for k in K_VALUES:
        top_k_ids = retrieved_ids[:k]
        found_ids = relevant_set.intersection(top_k_ids)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(top_k_ids, start=1)
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


def evaluate_retriever(
    retriever_name: str,
    retriever,
    cases: list[dict],
) -> dict:
    case_results = []

    for case in cases:
        retrieved_chunks = retriever.search(
            query=case["query"],
            per_retriever_k=max(K_VALUES),
        )
        retrieved_ids = [chunk["id"] for chunk in retrieved_chunks]
        relevant_ids = case["evidence_chunk_ids"]

        relevant_ranks = {
            chunk_id: (
                retrieved_ids.index(chunk_id) + 1
                if chunk_id in retrieved_ids
                else None
            )
            for chunk_id in relevant_ids
        }

        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "evidence_mode": case["evidence_mode"],
                "relevant_chunk_ids": relevant_ids,
                "relevant_ranks": relevant_ranks,
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": calculate_case_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                ),
            }
        )

    summary = summarize_results(case_results)

    return {
        "retriever": retriever_name,
        "case_count": len(case_results),
        "summary": summary,
        "cases": case_results,
    }


def summarize_results(case_results: list[dict]) -> dict:
    summary = {}
    case_count = len(case_results)
    multi_evidence_cases = [
        result
        for result in case_results
        if result["evidence_mode"] == "all_required"
    ]

    for k in K_VALUES:
        k_text = str(k)
        all_metrics = [result["metrics"][k_text] for result in case_results]
        multi_metrics = [
            result["metrics"][k_text]
            for result in multi_evidence_cases
        ]

        summary[k_text] = {
            "hit_rate": sum(metric["hit"] for metric in all_metrics)
            / case_count,
            "mean_recall": sum(metric["recall"] for metric in all_metrics)
            / case_count,
            "complete_rate": sum(
                metric["complete"] for metric in all_metrics
            )
            / case_count,
            "mrr": sum(
                metric["reciprocal_rank"] for metric in all_metrics
            )
            / case_count,
            "multi_evidence_complete_rate": (
                sum(metric["complete"] for metric in multi_metrics)
                / len(multi_metrics)
                if multi_metrics
                else None
            ),
        }

    return summary


def print_summary(result: dict) -> None:
    print("\n" + "=" * 88)
    print(
        f"Retriever: {result['retriever']} | "
        f"Cases: {result['case_count']}"
    )
    print("=" * 88)
    print(
        f"{'K':>3} "
        f"{'Hit@K':>10} "
        f"{'Recall@K':>10} "
        f"{'Complete@K':>12} "
        f"{'MRR@K':>10} "
        f"{'Multi-Complete@K':>18}"
    )

    for k in K_VALUES:
        metrics = result["summary"][str(k)]
        multi_complete = metrics["multi_evidence_complete_rate"]
        multi_text = (
            f"{multi_complete:.3f}"
            if multi_complete is not None
            else "N/A"
        )

        print(
            f"{k:>3} "
            f"{metrics['hit_rate']:>10.3f} "
            f"{metrics['mean_recall']:>10.3f} "
            f"{metrics['complete_rate']:>12.3f} "
            f"{metrics['mrr']:>10.3f} "
            f"{multi_text:>18}"
        )


def main() -> None:
    chunks_info = load_json(CHUNKS_PATH)
    cases = load_retrieval_cases()

    validate_evidence_chunk_ids(
        cases=cases,
        chunks_info=chunks_info,
    )

    retrievers = build_retrievers(chunks_info)
    evaluation_results = []

    for retriever_name in RETRIEVERS_TO_TEST:
        result = evaluate_retriever(
            retriever_name=retriever_name,
            retriever=retrievers[retriever_name],
            cases=cases,
        )
        evaluation_results.append(result)
        print_summary(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "evaluation_file": str(EVALUATION_PATH),
                "chunks_file": str(CHUNKS_PATH),
                "k_values": list(K_VALUES),
                "results": evaluation_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n详细结果已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
