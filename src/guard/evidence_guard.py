# threshold临界点
def select_evidence_chunks(retrieved_chunks: list[dict],
                           threshold: float = 0.9,
                           max_chunks: int = 5) -> list[dict]:

    evidence_chunks = []

    for chunk in retrieved_chunks:

        reranker_score = chunk.get("reranker_score")

        if reranker_score is None:
            continue

        if reranker_score >= threshold:
            evidence_chunks.append(chunk)

    return evidence_chunks[:max_chunks]