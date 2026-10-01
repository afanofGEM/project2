'''
vectorretriever:先定义，初始化fit,再RAGPipeline(retriever=vectorretriever)得结果
bm25retriever：先定义，初始化fit,RAGPipeline(retriever=bm25retriever)
hybridretriever：用初始化后的vectorretriever，bm25retriever定义，
RAGPipeline(retriever=hybridretriever,reranker)
'''
from src.prompt.rag_prompt import build_rag_prompt
import math
import re
import statistics
from src.guard.evidence_guard import select_evidence_chunks
import time


class RAGPipeline:

    def __init__(
            self,
            retriever,
            reranker=None,
            query_planner=None,
            generator=None):
        
        self.retriever = retriever
        self.reranker = reranker 
        self.query_planner = query_planner
        self.generator = generator
        '''因为hybrid retriever中的组分包含已经
        初始化好的dense/sparse retriever'''


    # 只负责让Query_Planner切问题
    def query_planner_split(self,query:str)->dict:
        if self.query_planner is None:
            raise ValueError(
                "Low模式必须提供query_planner"
            )

        query_planner_output = self.query_planner.run(
            query = query
        )
        '''
            {
                "query_type": "simple", # 最终认定模型是否可划分
                "subqueries": [
                    {
                        "id": "sq_1",
                        "query": query
                    }，
                    {
                        "id": "sq_1",
                        "query": query
                    }
                ],
                "used_fallback": True/False,
                "fallback_reason":None | fallback_reason # 如果失败，记录一下为什么划分失败
        }
        '''

        return query_planner_output

    # 得到了query_planner的输出JSON之后，只提取有用的id和query
    def build_query_task(
            self,
            query:str,
            query_planner_output:dict
    ) -> list[dict]:

        # 先把初始的query装进去
        query_tasks = [
            {
                "query_id": "original",
                "query_text": query,
                "query_source": "original"
            }]

        # 用于避免原问题和子问题完全重复
        # 先加入原始问题
        seen_queries = {
            " ".join(query.split()).casefold()
        }

        for subquery_dict in query_planner_output['subqueries']:

            subquery_text = subquery_dict['query']
            
            normalized_subquery_text = (
                " ".join(subquery_text.split())
                .casefold()
            )

            # 这个查询先前已有记载，跳过
            if normalized_subquery_text in seen_queries:
                continue

            # 其实seen_queries对结果没什么用，只是当筛子
            seen_queries.add(normalized_subquery_text)

            # 确保子问题与原问题，子问题之前没有重复：
            query_tasks.append(
                {
                    "query_id": subquery_dict["id"],
                    "query_text": subquery_text,
                    "query_source": "query_planner"
                }
            )

        return query_tasks
        '''
        [
            {
                "query_id": "original",
                "query_text": "信用卡取现有免息期吗？透支利率是多少？",
                "query_source": "original"
            },
            {
                "query_id": "sq_1",
                "query_text": "信用卡取现是否享受免息期？",
                "query_source": "planner"
            },
            {
                "query_id": "sq_2",
                "query_text": "信用卡透支利率是多少？",
                "query_source": "planner"
            }
        ]'''


    # 它只负责调用dense sparse hybrid单一的检索器
    def retriever_search(self,query:str,per_retriever_k:int=20) ->list[dict]:

        retrieved_chunks = self.retriever.search(query=query,per_retriever_k=per_retriever_k)
        '''list[dict]:
                {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'vector_retriever_score':vector_retriever匹配分数
                'bm25_retriever_score':bm25_retriever匹配分数
                "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
        }'''

        return retrieved_chunks


    def rerank_candidates(self,query:str,retrieved_chunks:list[dict]) -> list[dict]:

        if self.reranker is None:
            return retrieved_chunks

        # 新修改：reranker只负责打分，不负责截断
        reranked_chunks = self.reranker.rerank(query=query,candidates=retrieved_chunks)

        return reranked_chunks


    def retriever_reranker(self,query: str,per_retriever_k: int = 20) -> tuple[list[dict], list[dict]]:

        # 1. Hybrid Retriever返回的原始候选
        retrieved_chunks = self.retriever_search(
            query=query,
            per_retriever_k=per_retriever_k
        )

        # 2. 对全部候选进行重排，不在 Reranker 内截断
        reranked_chunks = self.rerank_candidates(
            query=query,
            retrieved_chunks=retrieved_chunks
        )

        return retrieved_chunks, reranked_chunks

    # 核心变量：query_task_results:
    # 包括每一个子查询的信息(id,查询内容,原查询或子查询)以及
    # 与它相关的chunks:retrieverd chunks, reranked chunks
    # 还有查询它时的消耗时间和reranker重排它时的消耗时间
    def query_tasks_retriever_reranker(
        self,
        query_tasks: list[dict], # query_planner切好的，整理好的子问题集合
        per_retriever_k: int = 20
    ) -> list[dict]:

        '''query_tasks:
        [
            {
                "query_id": "original",
                "query_text": "信用卡取现有免息期吗？透支利率是多少？",
                "query_source": "original"
            },
            {
                "query_id": "sq_1",
                "query_text": "信用卡取现是否享受免息期？",
                "query_source": "planner"
            },
            {
                "query_id": "sq_2",
                "query_text": "信用卡透支利率是多少？",
                "query_source": "planner"
            }
        ]'''

        query_task_results = []

        '''针对每个查询都retriever+reranker一遍，包括原始问题'''
        for query_task in query_tasks:

            query_id = query_task["query_id"]
            query_text = query_task["query_text"]
            query_source = query_task["query_source"]

            # 当前 Query Task 的 Hybrid Retrieval
            retrieval_start_time = time.perf_counter()

            # 调用了3个retriever，dense sparse hybrid
            retrieved_chunks = self.retriever_search(
                query=query_text,
                per_retriever_k=per_retriever_k
            )
            '''list[dict]:
                {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'vector_retriever_score':vector_retriever匹配分数
                'bm25_retriever_score':bm25_retriever匹配分数
                "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
            }'''

            retrieval_latency_seconds = (
                time.perf_counter()
                - retrieval_start_time
            )

            # 当前 Query Task 的 Cross Encoder Reranking
            reranker_start_time = time.perf_counter()

            reranked_chunks = self.rerank_candidates(
                query=query_text,
                retrieved_chunks=retrieved_chunks
            )
            '''        
            {
            [   'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'vector_retriever_score':vector_retriever匹配分数
                'bm25_retriever_score':bm25_retriever匹配分数
                "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
                "reranker_score":根据hybrid retriever提供的top-k，再经过cross encoder transformer后的分数
            ]
            }'''

            reranker_latency_seconds = (
                time.perf_counter()
                - reranker_start_time
            )

            query_task_results.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_source": query_source, # query_tasks老三样

                    "retrieved_chunks": retrieved_chunks, # list[dict]
                    "reranked_chunks": reranked_chunks, # list[dict]
                    "retrieval_latency_seconds": (
                        retrieval_latency_seconds
                    ),
                    "reranker_latency_seconds": (
                        reranker_latency_seconds
                    )
                }
            )

        return query_task_results

    # 修改各查询结果chunks中的合并规则，
    '''不用第二轮RRF了，因为那样会奖励出现次数多的chunks
    而真正回答某一子查询的chunk反而得不到好名次'''
    def merge_query_task_results(
        self,
        query_task_results: list[dict],
    ) -> list[dict]:

        '''query_task_results:   
        这是一个子查询为主体的list[dict]，集合每一个子查询的结果集             
        {
            "query_id": query_id,
            "query_text": query_text,
            "query_source": query_source, # query_tasks老三样
            "retrieved_chunks": retrieved_chunks, # list[dict]
            "reranked_chunks": reranked_chunks, # list[dict]
            "retrieval_latency_seconds": (
                retrieval_latency_seconds
            ),
            "reranker_latency_seconds": (
                reranker_latency_seconds
            )
        }'''

        # 这些分数都与具体Query有关，
        # 不能直接作为合并后Chunk的公共属性
        score_metrics = {
            "vector_retriever_score",
            "bm25_retriever_score",
            "rrf_score",
            "reranker_score"
        }
                        
        '''最终的结果集：所有子查询找到的相关的**去重**chunks集合
        因为字典本身就可以去重
        这是一个以chunk id为主体的list[dict]'''
        merged_chunks_dict = {}
        
        # 核心变量：query_task_results:
        # 包括每一个子查询的信息(id,查询内容,原查询或子查询)以及
        # 与它相关的chunks:retrieverd chunks, reranked chunks
        # 还有查询它时的消耗时间和reranker重排它时的消耗时间
        for query_task_result in query_task_results:

            # 每一个子查询的结果
            query_id = query_task_result["query_id"]
            query_text = query_task_result["query_text"]
            query_source = query_task_result["query_source"]

            # 子查询qi的retriever检索出的20条相关chunks
            '''它里面其实就retriever顺序重要，其他的reranked_chunks都有'''
            retrieved_chunks = query_task_result[
                "retrieved_chunks"
            ]

            # 子查询qi的reranker重排出的20条相关chunks
            reranked_chunks = query_task_result[
                "reranked_chunks"
            ]

            # 在某一个子查询中，每个Chunk原来的Hybrid排名
            '''这是一个向chunk_id为主体的转变'''
            hybrid_rank = {
                retrieved_chunk["id"]: retrieved_rank
                for retrieved_rank, retrieved_chunk in enumerate(
                    retrieved_chunks,start=1)
            }
            '''retrieved_chunk:
            {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'vector_retriever_score':vector_retriever匹配分数
                'bm25_retriever_score':bm25_retriever匹配分数
                "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
            }'''

            
            for reranker_rank, reranked_chunk in enumerate(
                reranked_chunks,
                start=1
            ):
                ''' reranked_chunk                          
                {   'id':pdf名称+pdf内chunk id
                    'source':来源的pdf
                    'document type':所属pdf文件的类型
                    'source org':pdf所属的银行
                    'update at':pdf官方文档的更新时间
                    'text':chunk内容
                    'source_url':pdf官方文档的链接
                    'vector_retriever_score':vector_retriever匹配分数
                    'bm25_retriever_score':bm25_retriever匹配分数
                    "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
                    "reranker_score":根据hybrid retriever提供的top-k，再经过cross encoder transformer后的分数
                }'''

                # reranked_chunk:是一个子查询找到的某一个chunk
                # merged_reranked_chunk:是所有查询找到的chunks集合，
                #   通过将每一个子查询找到的chunk去除当前查询的分数，合并构成
                '''遍历每一个子查询查询到的chunk,
                开始以chunk做主体'''
                chunk_id = reranked_chunk["id"]

                # 遍历每一个子查询找到的20个reranked_chunks，当前Chunk第一次出现在
                # 统计一共有哪些相关的chunks的集合中时
                # 先进行一些指标初始化
                if chunk_id not in merged_chunks_dict:

                    # 复制Chunk正文和Metadata，
                    # 暂时排除与具体Query相关的分数
                    '''结构并不变，只不过删了几个关于分数的键值对'''
                    merged_reranked_chunk = {
                        key: value
                        for key, value in reranked_chunk.items()
                        if key not in score_metrics
                    }
                    ''' merged_reranked_chunk                          
                        {   'id':pdf名称+pdf内chunk id
                            'source':来源的pdf
                            'document type':所属pdf文件的类型
                            'source org':pdf所属的银行
                            'update at':pdf官方文档的更新时间
                            'text':chunk内容
                            'source_url':pdf官方文档的链接
                        }'''

                    # 哪些 Query 找到了这个 Chunk，
                    # 当判断reranked_id已经出现在merged_chunks_dict时，会qppend
                    '''开始补一些以chunk为主体的指标了'''
                    merged_reranked_chunk["matched_query_ids"] = [] 

                    merged_reranked_chunk["diff_query_performance"] = {}
                    '''这个 Chunk 针对不同 Query，分别是什么表现。

                    "diff_query_performance": {
                        "q1": {
                            "query_text": "银行卡怎么挂失？",
                            "query_source": "planner",

                            "hybrid_rank": 2,
                            "reranker_rank": 1,

                            "vector_retriever_score": 0.79,
                            "bm25_retriever_score": 9.8,
                            "rrf_score": 0.029,
                            "reranker_score": 0.97
                        },

                        "q2": {
                            "query_text": "挂失后还能转账吗？",
                            "query_source": "planner",

                            "hybrid_rank": 7,
                            "reranker_rank": 4,

                            "vector_retriever_score": 0.68,
                            "bm25_retriever_score": 5.1,
                            "rrf_score": 0.021,
                            "reranker_score": 0.81
                        }
                    }'''


                    merged_reranked_chunk["best_reranker_score"] = None

                    '''统计完成，在merged_chunks_dict存档'''
                    merged_chunks_dict[chunk_id] = merged_reranked_chunk


                '''
                1.如果当前chunk并不是第一次出现在merged_chunks_dict
                被不同的子查询找到
                先读取存档
                2.如果当前chunk 是 第一次出现在merged_chunks_dict
                上方的初始化工作结束，这里填内容'''
                merged_reranked_chunk = merged_chunks_dict[chunk_id]
                ''' 
                merged_reranked_chunk                          
                {   
                    'id':pdf名称+pdf内chunk id
                    'source':来源的pdf
                    'document type':所属pdf文件的类型
                    'source org':pdf所属的银行
                    'update at':pdf官方文档的更新时间
                    'text':chunk内容
                    'source_url':pdf官方文档的链接
                    "matched_query_ids": 这一个chunk被哪些query标记为相关
                    “diff_query_performance”：这一个chunk在不同的query中的分数
                    "best_reranker_score"
                }'''

                # 防止同一个Query结果中意外出现重复Chunk
                # query_id : 顶层遍历查询时的老三样
                if query_id in merged_reranked_chunk["diff_query_performance"]:
                    continue

                merged_reranked_chunk["matched_query_ids"].append(query_id)
                
                # query_text,query_source : 顶层遍历查询时的老三样
                certain_query_performance = {
                    "query_text": query_text,
                    "query_source": query_source,
                    "hybrid_rank": (
                        hybrid_rank.get(chunk_id)
                    ),
                    "reranker_rank": reranker_rank
                }

                # 保存当前Query对应的检索和重排分数
                for score_metric in score_metrics:

                    ''' reranked_chunk                          
                    {   'id':pdf名称+pdf内chunk id
                        'source':来源的pdf
                        'document type':所属pdf文件的类型
                        'source org':pdf所属的银行
                        'update at':pdf官方文档的更新时间
                        'text':chunk内容
                        'source_url':pdf官方文档的链接
                        'vector_retriever_score':vector_retriever匹配分数
                        'bm25_retriever_score':bm25_retriever匹配分数
                        "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
                        "reranker_score":根据hybrid retriever提供的top-k，再经过cross encoder transformer后的分数
                    }'''
                    if score_metric in reranked_chunk:
                        # 补充指标
                        certain_query_performance[score_metric] = reranked_chunk[score_metric]

                '''补充完回填'''
                merged_reranked_chunk["diff_query_performance"][query_id] = certain_query_performance

                reranker_score = reranked_chunk.get("reranker_score")

                if reranker_score is not None:

                    best_reranker_score = merged_reranked_chunk["best_reranker_score"]

                    # 更新子查询中最高的reranker分数
                    if best_reranker_score is None or reranker_score > best_reranker_score:
                        merged_reranked_chunk["best_reranker_score"] = reranker_score

        '''merged_chunks:只取merged_chunks_dict中的值，不要键
        所以需要merged_reranked_chunk中的id
        [
        
            {   
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                "matched_query_ids": 这一个chunk被哪些query标记为相关
                “diff_query_performance”：这一个chunk在不同的query中的分数
                "best_reranker_score"
            },
                
            {   
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                "matched_query_ids": 这一个chunk被哪些query标记为相关
                “diff_query_performance”：这一个chunk在不同的query中的分数
                "best_reranker_score"
            },
        ]'''

        return merged_chunks_dict


    def build_accumulated_context(
        self,
        query_task_results: list[dict],
        merged_chunks_dict: dict,
        fusion_strategy: str = "round_robin",
    ) -> list[dict]:

        '''
        [{
            "query_id": query_id,
            "query_text": query_text,
            "query_source": query_source, # query_tasks老三样
            "retrieved_chunks": retrieved_chunks, # list[dict]
            "reranked_chunks": reranked_chunks, # list[dict]
            "retrieval_latency_seconds": (
                retrieval_latency_seconds
            ),
            "reranker_latency_seconds": (
                reranker_latency_seconds
            )
        }]'''
        if fusion_strategy in {
            "distribution_based",
            "distribution_based_original_anchor",
        }:
            return self.build_distribution_based_context(
                query_task_results=query_task_results,
                merged_chunks_dict=merged_chunks_dict,
                anchor_original_top1=(
                    fusion_strategy
                    == "distribution_based_original_anchor"
                ),
            )

        if fusion_strategy != "round_robin":
            raise ValueError(
                f"未知 Fusion 策略：{fusion_strategy}"
            )

        original_query_task_results = [
            query_task_result
            for query_task_result in query_task_results
            if query_task_result["query_source"] == "original"
        ]

        planner_query_task_results = [
            query_task_result
            for query_task_result in query_task_results
            if query_task_result["query_source"] in {
                "query_planner",
                "oracle",
            }
        ]

        '''注意ordered_query_results在格式上和query_task_results没什么区别
        只不过调整了内容的顺序
        依旧以查询为主体'''
        # Complex Query：
        # 优先按照Focused Subqueries积累证据，
        # Original Query作为安全补充。
        if len(planner_query_task_results) > 1:
            ordered_query_results = planner_query_task_results + original_query_task_results

        # Simple Query：
        # 优先保留Original Query的排序，
        # 避免简单改写破坏原来的正确排名。
        else:
            ordered_query_results = original_query_task_results + planner_query_task_results

        if not ordered_query_results:
            return list(
                merged_chunks_dict.values()
            )

        # 统计reranked_chunks最多的子查询
        max_reranked_chunks_count = max(len(query_task_result["reranked_chunks"])
            for query_task_result in ordered_query_results)

        results = []
        seen_chunk_ids = set()

        '''
        按照Reranker Rank逐层积累：
    
        第一轮：
        sq_1 rank1
        sq_2 rank1
        original rank1
        
        第二轮：
        sq_1 rank2
        sq_2 rank2
        original rank2
        
        而不是继续累加跨Query RRF。'''

        # 遍历名次 range(max_reranked_chunks_count) 可以理解成名次总数
        for reranker_rank in range(max_reranked_chunks_count):

            # 遍历所有查询
            for single_query_result in ordered_query_results:

                reranked_chunks = single_query_result["reranked_chunks"]
                '''list[dict]
                reranked_chunk                          
                {   'id':pdf名称+pdf内chunk id
                    'source':来源的pdf
                    'document type':所属pdf文件的类型
                    'source org':pdf所属的银行
                    'update at':pdf官方文档的更新时间
                    'text':chunk内容
                    'source_url':pdf官方文档的链接
                    'vector_retriever_score':vector_retriever匹配分数
                    'bm25_retriever_score':bm25_retriever匹配分数
                    "rrf_score": hybrid retriever给出的参考2个retriever排名给出的综合分数
                    "reranker_score":根据hybrid retriever提供的top-k，再经过cross encoder transformer后的分数
                }'''

                # 当前查询没有那么多名次的chunks，跳过这个查询
                if reranker_rank >= len(reranked_chunks):
                    continue

                # 按索引取chunk，某查询中，排名为reranked_rank的相关chunk
                reranked_chunk = reranked_chunks[reranker_rank]

                chunk_id = reranked_chunk["id"]

                if chunk_id in seen_chunk_ids:
                    continue

                '''以下情况是这个chunk_id第一次进入结果集时的操作'''
                # 从merged_chunks_dict中取此chunk的跨query信息
                # 相当于调它的档案
                chunk_file = merged_chunks_dict[chunk_id]
                '''{   
                    'id':pdf名称+pdf内chunk id
                    'source':来源的pdf
                    'document type':所属pdf文件的类型
                    'source org':pdf所属的银行
                    'update at':pdf官方文档的更新时间
                    'text':chunk内容
                    'source_url':pdf官方文档的链接
                    "matched_query_ids": 这一个chunk被哪些query标记为相关
                    “diff_query_performance”：这一个chunk在不同的query中的分数
                    "best_reranker_score"
                }'''

                # 是哪个 Query 第一次把它带进最终 Context 的
                chunk_file["first_bring_query_id"] = single_query_result["query_id"]

                chunk_file["first_bring_query_reranker_rank"] = reranker_rank + 1

                results.append(chunk_file)

                seen_chunk_ids.add(chunk_id)

        # 安全兜底：
        # 正常情况下所有Merged Chunk都会来自某个Reranked List。
        # 如果将来数据结构变化，仍然确保Chunk不会丢失。
        for merged_chunk in merged_chunks_dict.values():

            chunk_id = merged_chunk["id"]

            if chunk_id in seen_chunk_ids: continue

            merged_chunk["first_bring_query_id"] = None

            merged_chunk["first_bring_query_reranker_rank"] = None

            results.append(merged_chunk)

            seen_chunk_ids.add(chunk_id)

        for accumulated_rank, chunk in enumerate(results,start=1):

            chunk["accumulated_rank"] = accumulated_rank

        return results

    def build_distribution_based_context(
        self,
        query_task_results: list[dict],
        merged_chunks_dict: dict,
        anchor_original_top1: bool = False,
    ) -> list[dict]:
        """归一化各 Query 的 Reranker 分数后再合并。

        不同 Query 的原始 Reranker 分数分布可能不同，因此先在各自
        结果列表内按 mean +/- 3 * std 做归一化。Chunk 被多个 Query
        命中时保留最高归一化分数，避免用出现次数重复加分。
        """
        best_scores = {}
        best_sources = {}
        first_seen = {}
        next_seen = 0

        for query_result in query_task_results:
            reranked_chunks = query_result["reranked_chunks"]

            if not reranked_chunks:
                continue

            scores = [
                float(chunk.get("reranker_score", 0.0))
                for chunk in reranked_chunks
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
            score_range = maximum - minimum
            scores_vary = min(scores) != max(scores)

            for rank, (chunk, score) in enumerate(
                zip(reranked_chunks, scores),
                start=1,
            ):
                chunk_id = chunk["id"]
                normalized_score = (
                    (score - minimum) / score_range
                    if scores_vary and score_range != 0.0
                    else score
                )

                if chunk_id not in first_seen:
                    first_seen[chunk_id] = next_seen
                    next_seen += 1

                if (
                    chunk_id not in best_scores
                    or normalized_score > best_scores[chunk_id]
                ):
                    best_scores[chunk_id] = normalized_score
                    best_sources[chunk_id] = {
                        "query_id": query_result["query_id"],
                        "reranker_rank": rank,
                    }

        ordered_chunk_ids = sorted(
            best_scores,
            key=lambda chunk_id: (
                -best_scores[chunk_id],
                first_seen[chunk_id],
            ),
        )

        if anchor_original_top1:
            original_top1_id = next(
                (
                    query_result["reranked_chunks"][0]["id"]
                    for query_result in query_task_results
                    if query_result["query_source"] == "original"
                    and query_result["reranked_chunks"]
                ),
                None,
            )

            if original_top1_id in best_scores:
                ordered_chunk_ids = [original_top1_id] + [
                    chunk_id
                    for chunk_id in ordered_chunk_ids
                    if chunk_id != original_top1_id
                ]
        results = []

        for accumulated_rank, chunk_id in enumerate(
            ordered_chunk_ids,
            start=1,
        ):
            chunk = merged_chunks_dict[chunk_id]
            best_source = best_sources[chunk_id]
            chunk["fusion_score"] = best_scores[chunk_id]
            chunk["first_bring_query_id"] = best_source[
                "query_id"
            ]
            chunk["first_bring_query_reranker_rank"] = (
                best_source["reranker_rank"]
            )
            chunk["accumulated_rank"] = accumulated_rank
            results.append(chunk)

        return results


    def run_low(
        self,
        query: str,
        per_retriever_k: int = 20,
        fusion_strategy: str = "round_robin",
    ) -> dict:

        query = query.strip()

        if not query:
            raise ValueError("query不能为空")

        low_start_time = time.perf_counter()

        # 1. Query Planner生成子问题
        planner_start_time = time.perf_counter()

        query_planner_output = self.query_planner_split(
            query=query
        )

        planner_latency_seconds = (
            time.perf_counter()
            - planner_start_time
        )

        # 2. 构造Original Query和Subquery Tasks
        query_tasks = self.build_query_task(
            query=query,
            query_planner_output=query_planner_output
        )

        # 3. 得到以query为主体的查询信息
        query_task_results = (
            self.query_tasks_retriever_reranker(
                query_tasks=query_tasks,
                per_retriever_k=per_retriever_k
            )
        )

        # 4.得到以chunk为主体的档案
        merged_chunks_dict = self.merge_query_task_results(
            query_task_results=query_task_results)

        merged_chunks = list(
            merged_chunks_dict.values()
        )

        '''子查询高 rank>整体查询高 rank>子查询低 rank>整体查询低 rank'''
        accumulated_chunks = (
            self.build_accumulated_context(
                query_task_results=query_task_results,
                merged_chunks_dict=merged_chunks_dict,
                fusion_strategy=fusion_strategy,
            )
        )

        # 5. 汇总运行指标
        '''检索器一共找到多少条相关的'''
        total_retrieved_chunk_count = sum(
            len(query_task_result["retrieved_chunks"])
            for query_task_result in query_task_results
        )

        retrieval_latency_seconds = sum(
            query_task_result[
                "retrieval_latency_seconds"
            ]
            for query_task_result in query_task_results
        )

        reranker_latency_seconds = sum(
            query_task_result[
                "reranker_latency_seconds"
            ]
            for query_task_result in query_task_results
        )

        total_latency_seconds = (
            time.perf_counter()
            - low_start_time
        )

        if self.reranker is None:
            reranker_pair_count = 0
        else:
            reranker_pair_count = (
                total_retrieved_chunk_count
            )

        return {
            "mode": "low",
            "query": query,
            "fusion_strategy": fusion_strategy,
            "query_plan": query_planner_output,
            "query_tasks": query_tasks,
            "query_task_results": query_task_results,
            "merged_chunks": merged_chunks,
            "accumulated_chunks": accumulated_chunks,

            "query_count": len(query_tasks),
            "total_retrieved_chunk_count": (
                total_retrieved_chunk_count
            ),
            "merged_chunk_count": len(merged_chunks), # 所有的查询一共有多少条相关的
            "accumulated_chunk_count": len(
                accumulated_chunks
            ),

            # 有多少条重复的chunks
            "duplicate_chunk_count": (
                total_retrieved_chunk_count
                - len(merged_chunks)
            ),

            # reranker 一共计算了多少组cross encoded
            "reranker_pair_count":reranker_pair_count,
            
            # 分割查询时长
            "planner_latency_seconds": (
                planner_latency_seconds
            ),

            # 检索时长
            "retrieval_latency_seconds": (
                retrieval_latency_seconds
            ),
            
            "reranker_latency_seconds": (
                reranker_latency_seconds
            ),

            "total_latency_seconds": (
                total_latency_seconds
            )
        }




    def extract_used_citation(self,answer: str,citation_sources: list[dict]) -> list[dict]:

        cited_labels = re.findall(r"\[资料\d+\]",answer)

        # 去重，同时保持原出现顺序
        cited_labels = list(dict.fromkeys(cited_labels))

        used_citation = [
            item for item in citation_sources
            if item["citation_id"] in cited_labels
        ]

        '''list[dict]
        {
            "citation_id": num_citation,
            "chunk_id": chunk["id"],
            "source": chunk["source"]
        }'''
        return used_citation


    def run(self,query:str,per_retriever_k:int=20)->dict:

        retrieved_chunks, reranked_chunks = self.retriever_reranker(
            query=query,
            per_retriever_k=per_retriever_k
        )

        # 2. 从重排结果里筛真正用于回答的证据
        evidence_chunks = select_evidence_chunks(retrieved_chunks=reranked_chunks,
                                                 threshold=0.9,
                                                 max_chunks=5)

        results = {
            "query":query,
            "retrieved_chunks":retrieved_chunks,
            "reranked_chunks":reranked_chunks,
            "evidence_chunks":evidence_chunks
        }

        # 3. 只做检索评测，不生成答案
        if self.generator == None: 
            # 不需要让decoder-only回答，在评测retriever reranker时，不需要回答
            return results

        else:
            # 4. 没有足够证据，直接拒答
            if not evidence_chunks:

                results["answer"] = "根据当前提供的资料，无法确定。"
                results["used_citation"] = []
                results["refused"] = True

                return results

            # 5. 只把 evidence_chunks 给大语言模型
            prompt,citation_source = build_rag_prompt(query=query,
                                                      evidence_chunks=evidence_chunks)

            '''切片后的答案：...银行不再补扣。[资料1]'''
            answer = self.generator.generate(prompt=prompt,
                                             max_new_tokens=256)
            
            # 6. 提取模型真正引用的资料
            used_citation = self.extract_used_citation(
                answer=answer,
                citation_sources=citation_source
            )

            results['answer'] = answer
            results['used_citation'] = used_citation
            results["refused"] = False
            '''{
                "query":query,
                "retrieved_chunks":retrieved_chunks,
                "answer":answer,
                "'citation_source'":'citation_source'各个回答依据chunk的来源
            }'''

            return results
