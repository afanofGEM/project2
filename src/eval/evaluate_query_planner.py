from pathlib import Path
import json
import sys
import time
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generator.llm_generator import LLMGenerator
from src.query_processing.query_planner import QueryPlanner


EVALUATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "bank_eval_natural_v1.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "query_planner_eval_results.json"
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_evaluation_cases() -> list[dict]:

    with open(
        EVALUATION_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        evaluation_data = json.load(file)

    return evaluation_data["cases"]


def evaluate_query_planner(
    query_planner: QueryPlanner,
    evaluation_cases: list[dict]
) -> dict:

    case_results = []

    for index, case in enumerate(
        evaluation_cases,
        start=1
    ):

        case_id = case["id"]
        query = case["query"]

        print("\n" + "=" * 80)
        print(
            f"[{index}/{len(evaluation_cases)}] "
            f"{case_id}"
        )
        print("Query:", query)

        start_time = time.perf_counter()

        try:
            query_plan = query_planner.run(
                query=query,
                include_debug=True
            )

            latency = time.perf_counter() - start_time

            result = {
                "id": case_id,
                "query": query,
                "evidence_mode": case.get(
                    "evidence_mode"
                ),
                "task_type": case.get(
                    "task_type"
                ),
                "query_style": case.get(
                    "query_style"
                ),
                "query_plan": query_plan,
                "query_type": query_plan[
                    "query_type"
                ],
                "subquery_count": len(
                    query_plan["subqueries"]
                ),
                "used_fallback": query_plan[
                    "used_fallback"
                ],
                "fallback_reason": query_plan.get(
                    "fallback_reason"
                ),
                "raw_model_output": query_plan.get(
                    "raw_model_output"
                ),
                "latency_seconds": latency,
                "error": None
            }

            print(
                json.dumps(
                    query_plan,
                    ensure_ascii=False,
                    indent=2
                )
            )

            if query_plan["used_fallback"]:

                print(
                    "Fallback reason:",
                    query_plan["fallback_reason"]
                )

                print(
                    "Raw model output:",
                    query_plan["raw_model_output"]
                )

            print(
                "Latency:",
                f"{latency:.3f}s"
            )

        except Exception as error:

            latency = time.perf_counter() - start_time

            result = {
                "id": case_id,
                "query": query,
                "evidence_mode": case.get(
                    "evidence_mode"
                ),
                "task_type": case.get(
                    "task_type"
                ),
                "query_style": case.get(
                    "query_style"
                ),
                "query_plan": None,
                "query_type": None,
                "subquery_count": 0,
                "used_fallback": None,
                "fallback_reason": None,
                "raw_model_output": None,
                "latency_seconds": latency,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            }

            print("ERROR:", result["error"])

        case_results.append(result)

    # 程序正常执行完成，包括正常Plan和Fallback
    successful_results = [
        result
        for result in case_results
        if result["error"] is None
    ]

    # Qwen输出通过JSON解析和结构校验
    valid_plan_results = [
        result
        for result in successful_results
        if not result["used_fallback"]
    ]

    # Qwen输出没有通过校验，使用原问题回退
    fallback_results = [
        result
        for result in successful_results
        if result["used_fallback"]
    ]

    # 只统计真正通过校验的Simple
    simple_results = [
        result
        for result in valid_plan_results
        if result["query_type"] == "simple"
    ]

    # 只统计真正通过校验的Complex
    complex_results = [
        result
        for result in valid_plan_results
        if result["query_type"] == "complex"
    ]

    total_cases = len(case_results)
    execution_success_count = len(successful_results)
    valid_plan_count = len(valid_plan_results)
    fallback_count = len(fallback_results)

    if execution_success_count > 0:

        fallback_rate = (
            fallback_count
            / execution_success_count
        )

        average_latency_seconds = sum(
            result["latency_seconds"]
            for result in successful_results
        ) / execution_success_count

    else:
        fallback_rate = 0.0
        average_latency_seconds = 0.0

    if valid_plan_count > 0:

        subquery_counts = [
            result["subquery_count"]
            for result in valid_plan_results
        ]

        average_subquery_count = sum(
            subquery_counts
        ) / valid_plan_count

        minimum_subquery_count = min(
            subquery_counts
        )

        maximum_subquery_count = max(
            subquery_counts
        )

        subquery_count_distribution = {
            str(subquery_count): count
            for subquery_count, count in sorted(
                Counter(subquery_counts).items()
            )
        }

    else:
        average_subquery_count = 0.0
        minimum_subquery_count = 0
        maximum_subquery_count = 0
        subquery_count_distribution = {}

    summary = {
        "total_cases": total_cases,
        "execution_success_count": (
            execution_success_count
        ),
        "valid_plan_count": valid_plan_count,
        "runtime_error_count": (
            total_cases - execution_success_count
        ),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "simple_count": len(simple_results),
        "complex_count": len(complex_results),
        "average_subquery_count": (
            average_subquery_count
        ),
        "minimum_subquery_count": (
            minimum_subquery_count
        ),
        "maximum_subquery_count": (
            maximum_subquery_count
        ),
        "subquery_count_distribution": (
            subquery_count_distribution
        ),
        "average_latency_seconds": (
            average_latency_seconds
        )
    }

    return {
        "summary": summary,
        "cases": case_results
    }


def save_results(results: dict):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():

    evaluation_cases = load_evaluation_cases()

    print(
        f"Loaded {len(evaluation_cases)} "
        f"evaluation cases"
    )

    generator = LLMGenerator(
        model_name=MODEL_NAME
    )

    query_planner = QueryPlanner(
        generator=generator,
        max_subqueries=None
    )

    results = evaluate_query_planner(
        query_planner=query_planner,
        evaluation_cases=evaluation_cases
    )

    save_results(results=results)

    print("\n" + "=" * 80)
    print("Summary:")

    print(
        json.dumps(
            results["summary"],
            ensure_ascii=False,
            indent=2
        )
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
