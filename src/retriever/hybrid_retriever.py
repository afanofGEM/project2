'''Query
 │
 ├── denseretriever → Top-N
 │
 └── sparseretriever   → Top-N
             │
             ↓
          RRF融合
             ↓
        Hybrid Top-k

RRF融合的原理：
求1/(C+rank(chunk_i))之和，比如说chunk_2被dense retriever放在rank2，被sparse retriever放在rank3，
那么chunk_2的RRF就是 1/(C+2) + 1/(C+3),
越大说明它的综合排名更靠前，是同时被多个 Retriever 排得比较靠前的结果'''

class HybridRetriever:

    def __init__(self,denseretriever,sparseretriever,rank_constant:int=60,per_retriever_k:int=20):

        self.denseretriever = denseretriever
        self.sparseretriever = sparseretriever # 来融合的dense retriever和sparse retriever

        self.rank_constant = rank_constant # RRF评分的C
        self.per_retriever_k = per_retriever_k 
        '''Vector 和 BM25 各自检索多少条参加 RRF'''


    def fuse_retrievers_results(self,retriever_results:list[dict],fused_results:dict):

        for rank, result in enumerate(retriever_results, start=1):

            chunk_id = result["id"]

            # 1. 如果这个 chunk 第一次出现，就先放进 fused_results
            if chunk_id not in fused_results:

                fused_results[chunk_id] = result.copy()
                '''{
                    "chunk_1": {
                        "id": "chunk_1",
                        "text": "信用卡挂失...",
                        "vector_retriever_score": 0.91
                    }
                }'''

                # 初始化它的 RRF 分数
                fused_results[chunk_id]["rrf_score"] = 0.0


            # 2. 如果这个 chunk 之前已经被另一个 Retriever 找到了
            else:
                # 把当前 Retriever 中新增的字段补进去
                for key, value in result.items():

                    if key not in fused_results[chunk_id]:

                        fused_results[chunk_id][key] = value

            # 3. 计算当前 Retriever 给这个 chunk 的 RRF 分数
            # 无论它之前在不在fused_results里面，只要它出现了就加一次
            current_rrf_score = 1 / (self.rank_constant + rank)

            # 4. 累加到这个 chunk 的总 RRF 分数
            fused_results[chunk_id]["rrf_score"] += current_rrf_score

        return fused_results
    

    # hybrid retriever里面的per_retriever_k就是交给reranker多少chunks
    def search(self,query:str,per_retriever_k:int=20)->list[dict]:

        if per_retriever_k <= 0: return []

        vector_results = self.denseretriever.search(query=query,per_retriever_k=self.per_retriever_k)

        bm25_results = self.sparseretriever.search(query=query,per_retriever_k=self.per_retriever_k)
        '''格式都是
        [
            {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'score':匹配分数
            },
            {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'score':匹配分数
            },
        ]'''

        hybrid_results = {}

        hybrid_results = self.fuse_retrievers_results(retriever_results=vector_results,
                                                      fused_results=hybrid_results)

        hybrid_results = self.fuse_retrievers_results(retriever_results=bm25_results,
                                                      fused_results=hybrid_results)
        
        results = list(hybrid_results.values())

        results.sort(key=lambda result: result["rrf_score"],reverse=True)
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

        return results[:per_retriever_k]