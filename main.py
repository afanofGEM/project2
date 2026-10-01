from src.utils.load_data import (pdf_to_list,paragraph_chunks_with_source)
from src.retriever.vector_retriever import VectorRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.reranker import CrossEncoderReranker
from src.pipeline.rag_pipeline import RAGPipeline
from src.generator.llm_generator import LLMGenerator
from src.query_processing.query_planner import QueryPlanner
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'real'


def load_chunks()->list[dict]:

    # 1. 读取PDF文件
    texts_list = pdf_to_list(DATA_ROOT)

    # 2. 按照PDF自然段切分
    chunks_info = paragraph_chunks_with_source(
        texts_list=texts_list,
        chunk_size=500
    )

    return chunks_info


def build_pipeline(chunks_info:list[dict],
                   use_reranker:bool=True,
                   use_query_planner: bool = False,
                   use_generator:bool=True)->RAGPipeline:

    # 3. 初始化并建立向量索引
    vectorretriever = VectorRetriever()
    vectorretriever.fit(chunks_info)

    # 4. 初始化并建立BM25索引
    bm25retriever = BM25Retriever()
    bm25retriever.fit(chunks_info)

    # 5. 使用RRF融合两个Retriever
    hybridretriever = HybridRetriever(
        denseretriever=vectorretriever,
        sparseretriever=bm25retriever,
        rank_constant=60,
        per_retriever_k=20
    )

    # 6. 根据参数决定是否使用Reranker
    if use_reranker:
        reranker = CrossEncoderReranker()
    else:
        reranker = None

    # 7. 大语言模型生成器
    # Query Planner和Answer Generator复用同一个Qwen
    if use_query_planner or use_generator:

        llm_generator = LLMGenerator(
            model_name="Qwen/Qwen2.5-1.5B-Instruct"
        )

    else:
        llm_generator = None

    # 8.查询分割器
    if use_query_planner:

        query_planner = QueryPlanner(
            generator=llm_generator,
            max_subqueries=None
        )

    else:
        query_planner = None


    if use_generator:
        generator = llm_generator
    else:
        generator = None

    # 8. 创建完整Pipeline
    pipeline = RAGPipeline(retriever=hybridretriever,
                           reranker=reranker,
                           query_planner=query_planner,
                           generator=generator)
    

    return pipeline


def print_results(result: dict) -> None:

    print("\n" + "=" * 80)
    print("Query:")
    print(result["query"])

    # 大语言模型最终回答
    if "answer" in result:
        print("\nAnswer:")
        print(result["answer"])

    # 大语言模型的回答依据chunk信息
    if "used_citation" in result:
        print("\nUsed citation")

        for item in result["used_citation"]:

            print(
                item["citation_id"],
                "->",
                item["chunk_id"],
                "->",
                item["source"]
            )

    print("\nReranked Chunks:")
    
    for rank, chunk in enumerate(result["reranked_chunks"],start=1):
        print("\n" + "-" * 80)
        print(f"Top {rank}")
        print("Chunk ID:",chunk["id"])
        print("Source:",chunk["source"])

        if "vector_retriever_score" in chunk:
            print("Vector score:",f"{chunk['vector_retriever_score']:.4f}")

        if "bm25_retriever_score" in chunk:
            print("BM25 score:",f"{chunk['bm25_retriever_score']:.4f}")

        if "rrf_score" in chunk:
            print("RRF score:",f"{chunk['rrf_score']:.6f}")

        if "reranker_score" in chunk:
            print("Reranker score:",f"{chunk['reranker_score']:.4f}")

        print("Text:",chunk["text"])


def print_low_results(result: dict) -> None:

    print("\n" + "=" * 80)
    print("Mode:", result["mode"])
    print("Query:", result["query"])

    print("\nQuery Plan:")

    print(
        json.dumps(
            result["query_plan"],
            ensure_ascii=False,
            indent=2
        )
    )

    print("\nQuery Tasks:")

    for query_task in result["query_tasks"]:

        print(
            query_task["query_id"],
            "->",
            query_task["query_text"]
        )

    print("\nExecution Summary:")

    print(
        "Query count:",
        result["query_count"]
    )

    print(
        "Retrieved chunk count:",
        result["total_retrieved_chunk_count"]
    )

    print(
        "Merged chunk count:",
        result["merged_chunk_count"]
    )

    print(
        "Duplicate chunk count:",
        result["duplicate_chunk_count"]
    )

    print(
        "Reranker pair count:",
        result["reranker_pair_count"]
    )

    print(
        "Planner latency:",
        f"{result['planner_latency_seconds']:.3f}s"
    )

    print(
        "Retrieval latency:",
        f"{result['retrieval_latency_seconds']:.3f}s"
    )

    print(
        "Reranker latency:",
        f"{result['reranker_latency_seconds']:.3f}s"
    )

    print(
        "Total latency:",
        f"{result['total_latency_seconds']:.3f}s"
    )

    print("\nAccumulated Chunks:")

    for rank, chunk in enumerate(
        result["accumulated_chunks"],
        start=1
    ):

        print("\n" + "-" * 80)
        print("Accumulated Rank:", rank)
        print("Chunk ID:", chunk["id"])
        print("Source:", chunk["source"])

        print(
            "Matched query IDs:",
            chunk["matched_query_ids"]
        )

        print(
            "First bring query ID:",
            chunk["first_bring_query_id"]
        )

        print(
            "First bring reranker rank:",
            chunk[
                "first_bring_query_reranker_rank"
            ]
        )

        print(
            "Best reranker score:",
            chunk["best_reranker_score"]
        )

        print("Query performance:")

        print(
            json.dumps(
                chunk["diff_query_performance"],
                ensure_ascii=False,
                indent=2
            )
        )

        print("Text:", chunk["text"])


def main():

    chunks_info = load_chunks()

    pipeline = build_pipeline(chunks_info=chunks_info,
                              use_reranker=True,
                              use_query_planner=True,
                              use_generator=False)

    while True:
        query = input("\n请输入问题，输入1退出：").strip()

        if query.lower() == "1": break

        if not query: 
            print("问题不能为空") 
            continue

        result = pipeline.run_low(
            query=query,
            per_retriever_k=20
        )

        print_low_results(result)


if __name__ == "__main__":
    main()
