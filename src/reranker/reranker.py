from sentence_transformers import CrossEncoder 
# 从Hugging Face上加载已经训练好的 Cross Encoder Reranker

class CrossEncoderReranker:

    def __init__(self,model_name="BAAI/bge-reranker-base"):

        self.model = CrossEncoder(model_name)


    def score_candidates(self,query:str,candidates:list[dict])-> list[dict]:

        if not candidates:
            return []
        
        pairs = []

        '''candidates:hybrid retriever排序得到的top-k个chunk dict
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
        for item in candidates:

            pairs.append([query ,item['text']])
            '''pairs:同一个query与它的所有的hybrid retriever筛选出的chunks
            [
                (
                "信用卡怎么免年费？",
                "信用卡年费收费标准如下..."
                ),

                (
                "信用卡怎么免年费？",
                "信用卡遗失后应该立即挂失..."
                )
            ]'''

        #(candidates_num)
        scores_list = self.model.predict(pairs)

        '''hybrid retriever合并dense sparse retriever时新开了一个list[dict]存它的结果
        所以reranker整合重排hybrid retriever结果时，也新开一个list[dict]'''
        reranker_results = []

        for item,reranker_score in zip(candidates,scores_list):

            result = {
                **item,
                'reranker_score':float(reranker_score)
            }

            reranker_results.append(result)
             
            # 新增一个属性，与之前hybrid retriever给的score对比

        reranker_results.sort(key=lambda x:x["reranker_score"], reverse=True)
        '''
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
            "reranker_score":根据hybrid retriever提供的top-k，再经过cross encoder transformer后的分数
        }
        '''
        return reranker_results

    # 现在reranker中不负责截断，只打分
    def rerank(self,query:str,candidates:list[dict])-> list[dict]:
        
        reranked_results = self.score_candidates(query=query,candidates=candidates)
        return reranked_results
        

