import unittest


from src.eval.evaluate_planner_oracle import (
    build_decision,
    calculate_metrics,
    validate_planner_dataset,
    validate_split_isolation,
)
from src.pipeline.rag_pipeline import RAGPipeline


class PlannerOracleEvaluationTest(unittest.TestCase):

    def test_decision_prioritizes_fusion_loss(self):
        summary = {
            "methods": {
                "original": {
                    "ranking": {"5": {"complete_rate": 0.7}},
                },
                "llm_planner": {
                    "ranking": {"5": {"complete_rate": 0.75}},
                    "fusion_loss_case_ids_at_5": ["case_a"],
                },
                "oracle": {
                    "ranking": {"5": {"complete_rate": 0.85}},
                    "fusion_loss_case_ids_at_5": ["case_b"],
                },
            }
        }

        decision = build_decision(summary)

        self.assertEqual(
            decision["outcome"],
            "fusion_bottleneck_before_planner_tuning",
        )

    def test_complete_requires_all_evidence(self):
        metrics = calculate_metrics(
            ranked_ids=["chunk_a", "noise", "chunk_b"],
            required_evidence=["chunk_a", "chunk_b"],
        )

        self.assertFalse(metrics["1"]["complete"])
        self.assertEqual(metrics["1"]["recall"], 0.5)
        self.assertTrue(metrics["3"]["complete"])
        self.assertEqual(metrics["3"]["reciprocal_rank"], 1.0)

    def test_dataset_validation_accepts_complete_gold_coverage(self):
        dataset = {
            "cases": [
                {
                    "id": "planner_dev_001",
                    "query": "组合问题",
                    "task_type": "multi_intent",
                    "evidence_mode": "all_required",
                    "required_evidence": ["chunk_a", "chunk_b"],
                    "expected_points": ["要点A", "要点B"],
                    "gold_subqueries": [
                        {
                            "id": "gold_1",
                            "query": "问题A",
                            "required_evidence": ["chunk_a"],
                        },
                        {
                            "id": "gold_2",
                            "query": "问题B",
                            "required_evidence": ["chunk_b"],
                        },
                    ],
                    "source_case_ids": ["source_a", "source_b"],
                }
            ]
        }
        chunks = [{"id": "chunk_a"}, {"id": "chunk_b"}]

        validate_planner_dataset(
            dataset=dataset,
            chunks_info=chunks,
            natural_queries=set(),
        )

    def test_split_validation_rejects_evidence_leakage(self):
        dev = {
            "cases": [
                {"required_evidence": ["shared_chunk"]}
            ]
        }
        locked = {
            "cases": [
                {"required_evidence": ["shared_chunk"]}
            ]
        }

        with self.assertRaisesRegex(ValueError, "证据重叠"):
            validate_split_isolation(dev, locked)

    def test_oracle_queries_are_prioritized_by_existing_fusion(self):
        pipeline = RAGPipeline(retriever=None)
        query_task_results = [
            self._query_result(
                query_id="original",
                query_source="original",
                chunk_ids=["generic", "chunk_a", "chunk_b"],
            ),
            self._query_result(
                query_id="gold_1",
                query_source="oracle",
                chunk_ids=["chunk_a", "generic", "chunk_b"],
            ),
            self._query_result(
                query_id="gold_2",
                query_source="oracle",
                chunk_ids=["chunk_b", "generic", "chunk_a"],
            ),
        ]
        merged = pipeline.merge_query_task_results(
            query_task_results=query_task_results
        )
        accumulated = pipeline.build_accumulated_context(
            query_task_results=query_task_results,
            merged_chunks_dict=merged,
        )

        self.assertEqual(
            [chunk["id"] for chunk in accumulated[:3]],
            ["chunk_a", "chunk_b", "generic"],
        )

    def test_distribution_based_fusion_matches_offline_ablation(self):
        pipeline = RAGPipeline(retriever=None)
        query_task_results = [
            self._query_result_with_scores(
                query_id="sq_1",
                query_source="query_planner",
                chunks=[("evidence_a", 10.0), ("noise", 0.0)],
            ),
            self._query_result_with_scores(
                query_id="sq_2",
                query_source="query_planner",
                chunks=[("evidence_b", 1.0), ("other", 0.9)],
            ),
        ]
        merged = pipeline.merge_query_task_results(
            query_task_results=query_task_results
        )
        accumulated = pipeline.build_accumulated_context(
            query_task_results=query_task_results,
            merged_chunks_dict=merged,
            fusion_strategy="distribution_based",
        )

        self.assertEqual(
            {chunk["id"] for chunk in accumulated[:2]},
            {"evidence_a", "evidence_b"},
        )
        self.assertTrue(
            all("fusion_score" in chunk for chunk in accumulated)
        )

    def test_unknown_fusion_strategy_is_rejected(self):
        pipeline = RAGPipeline(retriever=None)

        with self.assertRaisesRegex(ValueError, "未知 Fusion 策略"):
            pipeline.build_accumulated_context(
                query_task_results=[],
                merged_chunks_dict={},
                fusion_strategy="unknown",
            )

    def test_distribution_based_can_anchor_original_top1(self):
        pipeline = RAGPipeline(retriever=None)
        query_task_results = [
            self._query_result_with_scores(
                query_id="original",
                query_source="original",
                chunks=[("original_best", 0.9), ("other", 0.8)],
            ),
            self._query_result_with_scores(
                query_id="sq_1",
                query_source="query_planner",
                chunks=[("planned_best", 1.0), ("original_best", 0.1)],
            ),
        ]
        merged = pipeline.merge_query_task_results(
            query_task_results=query_task_results
        )
        accumulated = pipeline.build_accumulated_context(
            query_task_results=query_task_results,
            merged_chunks_dict=merged,
            fusion_strategy=(
                "distribution_based_original_anchor"
            ),
        )

        self.assertEqual(accumulated[0]["id"], "original_best")

    @staticmethod
    def _query_result(
        query_id: str,
        query_source: str,
        chunk_ids: list[str],
    ) -> dict:
        chunks = [
            {
                "id": chunk_id,
                "text": chunk_id,
                "source": "test.pdf",
                "reranker_score": float(len(chunk_ids) - index),
            }
            for index, chunk_id in enumerate(chunk_ids)
        ]
        return {
            "query_id": query_id,
            "query_text": query_id,
            "query_source": query_source,
            "retrieved_chunks": chunks,
            "reranked_chunks": chunks,
            "retrieval_latency_seconds": 0.0,
            "reranker_latency_seconds": 0.0,
        }

    @staticmethod
    def _query_result_with_scores(
        query_id: str,
        query_source: str,
        chunks: list[tuple[str, float]],
    ) -> dict:
        chunk_records = [
            {
                "id": chunk_id,
                "text": chunk_id,
                "source": "test.pdf",
                "reranker_score": score,
            }
            for chunk_id, score in chunks
        ]
        return {
            "query_id": query_id,
            "query_text": query_id,
            "query_source": query_source,
            "retrieved_chunks": chunk_records,
            "reranked_chunks": chunk_records,
            "retrieval_latency_seconds": 0.0,
            "reranker_latency_seconds": 0.0,
        }


if __name__ == "__main__":
    unittest.main()
