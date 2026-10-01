import unittest


from src.eval.evaluate_fusion_ablation import (
    anchor_original_top1,
    fuse_coverage_quota,
    fuse_distribution_based,
    fuse_rrf,
)


def query_list(
    query_id: str,
    query_source: str,
    chunks: list[tuple[str, float]],
) -> dict:
    return {
        "query_id": query_id,
        "query_source": query_source,
        "chunks": [
            {
                "chunk_id": chunk_id,
                "reranker_score": score,
            }
            for chunk_id, score in chunks
        ],
    }


class FusionAblationTest(unittest.TestCase):

    def test_rrf_rewards_cross_query_agreement(self):
        lists = [
            query_list(
                "gold_1",
                "oracle",
                [("shared", 0.8), ("only_a", 0.7)],
            ),
            query_list(
                "gold_2",
                "oracle",
                [("only_b", 0.9), ("shared", 0.6)],
            ),
        ]

        ranked = fuse_rrf(lists, rank_constant=60)

        self.assertEqual(ranked[0], "shared")

    def test_weighted_rrf_can_reduce_original_query_influence(self):
        lists = [
            query_list(
                "original",
                "original",
                [("generic", 0.9), ("focused", 0.8)],
            ),
            query_list(
                "gold_1",
                "oracle",
                [("focused", 0.9), ("generic", 0.8)],
            ),
        ]

        ranked = fuse_rrf(
            lists,
            rank_constant=60,
            original_weight=0.5,
            planned_weight=1.0,
        )

        self.assertEqual(ranked[0], "focused")

    def test_coverage_quota_keeps_each_planned_query(self):
        lists = [
            query_list(
                "original",
                "original",
                [("generic", 0.9), ("shared", 0.8)],
            ),
            query_list(
                "gold_1",
                "oracle",
                [("evidence_a", 0.9), ("noise_a", 0.8)],
            ),
            query_list(
                "gold_2",
                "oracle",
                [("evidence_b", 0.9), ("noise_b", 0.8)],
            ),
        ]

        ranked = fuse_coverage_quota(lists, final_k=5)

        self.assertEqual(
            ranked[:4],
            ["evidence_a", "noise_a", "evidence_b", "noise_b"],
        )

    def test_distribution_based_uses_best_normalized_score(self):
        lists = [
            query_list(
                "gold_1",
                "oracle",
                [("evidence_a", 10.0), ("noise", 0.0)],
            ),
            query_list(
                "gold_2",
                "oracle",
                [("evidence_b", 1.0), ("other", 0.9)],
            ),
        ]

        ranked = fuse_distribution_based(lists)

        self.assertEqual(set(ranked[:2]), {"evidence_a", "evidence_b"})

    def test_original_top1_anchor_preserves_safety_result(self):
        lists = [
            query_list(
                "original",
                "original",
                [("original_best", 0.9), ("other", 0.8)],
            ),
            query_list(
                "sq_1",
                "query_planner",
                [("planned_best", 1.0), ("original_best", 0.1)],
            ),
        ]

        ranked = anchor_original_top1(
            ranked_ids=["planned_best", "original_best", "other"],
            query_lists=lists,
        )

        self.assertEqual(
            ranked,
            ["original_best", "planned_best", "other"],
        )


if __name__ == "__main__":
    unittest.main()
