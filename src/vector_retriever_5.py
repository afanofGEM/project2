
'''query_embedded，每条chunk的信息(用于最终结果的展示)，
存储chunks的仓库，规定搜索的范围'''
def search(query:list[str],chunks_info:list[dict],storehouse,top_k:int=3):
    scores_list,indices_list = storehouse.search(query,top_k)
    '''storehouse拿着query与仓库中所有的chunks进行内积运算
    内积结果就是向量夹角的余弦值，越大说明两个文本包含的信息越相似'''

    '''scores_list:[query_num,top-k]
    indices_list:[query_num,top-k]与每一条query最匹配的top-k个chunk'''

    results = []
    for score,index in zip(scores_list[0],indices_list[0]):
        '''同时遍历分数集合和索引集合'''

        chunk_info = chunks_info[index]
        result = {
            "score": float(score),
            "text": chunk_info["text"],
            "source": chunk_info["source"],
            "id": chunk_info["id"],
        }
        results.append(result)

    return results