import jieba
from collections import Counter
import math

# 先把一个chunk的内容按照中文词语分开
def tokenizer(text:str)->list[str]:
    return [token.strip() for token in jieba.lcut(text) 
            if token.strip()]


class BM25Retriever:

    def __init__(self, k1: float = 1.5, b: float = 0.75):# 接受chunks_info以及初始化search需要的TF IDF列表

        self.k1 = k1
        self.b = b

        self.chunks_info: list[dict] = []
        '''
        {
            'id':pdf名称+pdf内chunk id
            'source':来源的pdf
            'document type':所属pdf文件的类型
            'source org':pdf所属的银行
            'update at':pdf官方文档的更新时间
            'text':chunk内容
            'source_url':pdf官方文档的链接
        }'''

        self.tokenized_chunks: list[list[str]] = []
        '''每一个chunk被划分成的token列表list[str]'''

        self.token_in_chunk_frequencies: list[Counter] = []
        '''用来算TF，每一个token在当前chunk中出现的次数'''

        self.token_in_document_frequencies: Counter = Counter()
        '''用来算IDF,所有token出现在多少个chunk中'''

        self.chunk_lengths: list[int] = []
        '''每一个chunk的长度'''

        self.average_chunk_length: float = 0.0

        self.chunks_count: int = 0

    # 开始初始化，从外界传入所有的chunks集合
    def fit(self,chunks_info: list[dict]) -> None:

        self.chunks_info = chunks_info

        # list[list[str]]
        '''[[我 喜欢 吃 苹果],[其实 梨 也 不错]]'''
        self.tokenized_chunks = [tokenizer(chunk["text"]) 
                                    for chunk in chunks_info]

        # list[Counter]
        '''[[我:1 喜欢:1 吃:1 苹果:1],[其实:1 梨:1 也:1 不错:1]]'''
        self.token_in_chunk_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_chunks
        ]

        # list[int]
        '''[4,4]'''
        self.chunk_lengths = [
            len(tokens)
            for tokens in self.tokenized_chunks
        ]

        # int
        '''2'''
        self.chunks_count = len(chunks_info)

        if self.chunks_count == 0:
            raise ValueError(
                "Cannot build BM25 index from empty chunks."
            )

        # int
        '''4'''
        self.average_chunk_length = sum(self.chunk_lengths) / self.chunks_count

        self.token_in_document_frequencies = Counter()
        '''[我:1 喜欢:1 吃:1 苹果:1 其实:1 梨:1 也:1 不错:1]'''

        for tokens in self.tokenized_chunks:

            # 每一个chunk中的token先去个重
            '''(我 喜欢 吃 苹果)
               (其实 梨 也 不错)'''
            unique_tokens_list = set(tokens) # 统计每一个Token在多少个chunks中出现

            # 挨个遍历，累计次数
            for token in unique_tokens_list:
                self.token_in_document_frequencies[token] += 1


    '''一个 token 出现在越少的 chunk 里，它的 IDF 越高，也越有区分度'''
    def idf(self, token: str) -> float:

        # token出现在多少个chunks中
        # token_in_document_frequencies：Counter()
        document_frequency = (self.token_in_document_frequencies.get(token, 0))

        numerator = (self.chunks_count - document_frequency + 0.5)

        denominator = (document_frequency + 0.5)

        return math.log(1 + numerator / denominator)


    # 挨个chunk看与query的匹配程度
    '''query_tokens=['我','要','翘课']'''
    def single_chunk_score(self,query_tokens: list[str],chunk_index: int) -> float:
        '''这个词在当前chunk出现得多+
            这个词在整个语料库又比较稀有+
            这个chunk不是又臭又长只碰巧提了一嘴
                        ↓
        高分'''

        token_frequencies = self.token_in_chunk_frequencies[chunk_index]
        '''token_frequencies = [我:1 喜欢:1 吃:1 苹果:1]'''

        chunk_length = self.chunk_lengths[chunk_index]

        score = 0.0 # 遍历所有的Token后，整个chunk的得分
        for token in query_tokens:

            tf = token_frequencies.get(token, 0) # 看当前查询的token在这一chunk中的出现频次
            # 当前查询token出现在此chunk中频次越高，分越高

            if tf == 0:
                continue

            idf = self.idf(token) # 当前token的稀有程度
            # 当前查询token在所有chunk中越稀有，分越高

            length_normalization = 1 - self.b + self.b * chunk_length / self.average_chunk_length

            denominator = tf + self.k1 * length_normalization
            
            token_score = idf * tf * (self.k1 + 1) / denominator

            score += token_score

        return score


    def search(self, query: str, per_retriever_k: int = 30) -> list[dict]:

        if not self.chunks_info:
            raise RuntimeError("你的chunks_info list[dict]还没有建立，要先fit")

        query_tokens = tokenizer(query) # 用jieba切词

        scored_chunks = []

        for chunk_index in range(self.chunks_count): # 要对每个chunk挨个算匹配分数

            score = self.single_chunk_score(query_tokens=query_tokens,chunk_index=chunk_index)
            
            scored_chunks.append((chunk_index,score))
            '''每一个chunk针对query的得分'''

        scored_chunks.sort(key=lambda item: item[1], reverse=True)

        results = []

        '''依旧Top-k选择方法'''
        for chunk_index, score in scored_chunks[:per_retriever_k]:

            chunk_info = self.chunks_info[chunk_index] 
            # chunk_index是列表的索引，chunk_info:dict，它不是chunk的Id

            results.append(
                {
                    "bm25_retriever_score": float(score),
                    **chunk_info
                }
            )
            '''
            {
                'id':pdf名称+pdf内chunk id
                'source':来源的pdf
                'document type':所属pdf文件的类型
                'source org':pdf所属的银行
                'update at':pdf官方文档的更新时间
                'text':chunk内容
                'source_url':pdf官方文档的链接
                'bm25_retriever_score':bm25_retriever匹配分数
            }'''

        return results # list[dict]
