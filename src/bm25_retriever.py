import jieba
from collections import Counter
import math

def tokenizer(text:str)->list[str]:
    return [token.strip() for token in jieba.lcut(text) 
            if token.strip()]


class BM25Retriever:

    def __init__(self, k1: float = 1.5, b: float = 0.75):

        self.k1 = k1
        self.b = b

        self.chunks_info: list[dict] = []
        '''
        {
            'text':
            'metadata':
        }最初的chunks集合，文本及其信息'''

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


    def fit(self,chunks_info: list[dict]) -> None:

        self.chunks_info = chunks_info

        self.tokenized_chunks = [tokenizer(chunk["text"]) 
                                    for chunk in chunks_info]

        self.token_in_chunk_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_chunks
        ]

        self.chunk_lengths = [
            len(tokens)
            for tokens in self.tokenized_chunks
        ]

        self.chunks_count = len(chunks_info)

        if self.chunks_count == 0:
            raise ValueError(
                "Cannot build BM25 index from empty chunks."
            )

        self.average_chunk_length = sum(self.chunk_lengths) / self.chunks_count

        self.token_in_document_frequencies = Counter()

        for tokens in self.tokenized_chunks:

            unique_tokens_list = set(tokens) # 统计每一个Token在多少个chunks中出现

            for token in unique_tokens_list:
                self.token_in_document_frequencies[token] += 1


    '''一个 token 出现在越少的 chunk 里，它的 IDF 越高，也越有区分度'''
    def idf(self, token: str) -> float:

        # token出现在多少个chunks中
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
        '''list[Counter]用来算TF，每一个token在当前chunk中出现的次数'''

        chunk_length = self.chunk_lengths[chunk_index]

        score = 0.0 # 遍历所有的Token后，整个chunk的得分
        for token in query_tokens:

            tf = token_frequencies.get(token, 0)

            if tf == 0:
                continue

            idf = self.idf(token) # 这个token的稀有程度

            length_normalization = 1 - self.b + self.b * chunk_length / self.average_chunk_length

            denominator = tf + self.k1 * length_normalization
            
            token_score = idf * tf * (self.k1 + 1) / denominator

            score += token_score

        return score


    def search(self, query: str, top_k: int = 3) -> list[dict]:

        if not self.chunks_info:
            raise RuntimeError("你的chunks_info list[dict]还没有建立")

        query_tokens = tokenizer(query) # 用jieba切词

        scored_chunks = []

        for chunk_index in range(self.chunks_count):

            score = self.single_chunk_score(query_tokens=query_tokens,chunk_index=chunk_index)
            
            scored_chunks.append((chunk_index,score))
            '''每一个chunk针对query的得分'''

        scored_chunks.sort(key=lambda item: item[1], reverse=True)

        results = []

        '''依旧Top-k选择方法'''
        for chunk_index, score in scored_chunks[:top_k]:

            chunk = self.chunks_info[chunk_index]

            results.append(
                {
                    "score": float(score),
                    **chunk
                }
            )

        return results
