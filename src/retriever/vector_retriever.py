import faiss 
'''对中文的chunks embedding''' 
from sentence_transformers import SentenceTransformer 
MODEL_NAME = "BAAI/bge-small-zh-v1.5" 

'''与bm25retriever类似，只需要输入chunks_info 
bm25retriever会先行fit得到每一个token的TF IDF 
随后针对用户Query遍历搜索每个chunk:Token在chunk中的出现次数，以及其稀有程度，整个chunk的长度，计算分数 
 
而vectorretriever会提取chunks_list，对其整体embedding 
随后逐个与query内积算向量夹角''' 
class VectorRetriever: 
 
    def __init__(self): #接收chunks_info和初始化search需要的存储仓库
        self.chunks_info: list[dict] = [] 
        self.storehouse = None 
        self.model = SentenceTransformer(MODEL_NAME) # 这个模型默认embedding_dim = 512 
 
    '''一次性对一个chunks集合embedding,把文本映射到一个高维语义空间''' 
    def chunks_embedding(self): 
 
        chunks_list = [] 
        for chunk_info in self.chunks_info: 
            '''{ 
                    'id':chunk_id, 
                    'chunk':chunk, 
                    'source':text_dict['source'] 
                }''' 
            chunks_list.append(chunk_info['text'])  #(num_chunks,different_chunk_len)
 
        # 对每个编码后的chunk向量正则化，向量长度为1 
        # Inner Product = Cosine Similarity内积结果就是向量之间的夹角 
        chunks_embedd = self.model.encode(chunks_list, 
                                    convert_to_numpy=True, 
                                    normalize_embeddings=True) 
        '''chunks_embedd:(chunks_num,embedding_dim) 
        检索实际上变成：用户Query的向量，和哪个Chunk的向量最相似 
        使用向量之间的夹角余弦值判断，越接近1说明两个向量的方向越接近， 
        同理Query与某个文本向量也最相似''' 
 
        return chunks_embedd.astype('float32') #(num_chunk,embedding_dim)
 
 
    def single_query_embedding(self,query:str): 
 
        query_embedded = self.model.encode([query], 
                                    convert_to_numpy=True, 
                                    normalize_embeddings=True) 
        '''query_embedded:(1,embedding_dim)''' 
 
        return query_embedded.astype('float32') 
 
 
    def multi_query_embedding(self,queries:list[str]): 
 
        query_embedded = self.model.encode(queries, 
                                    convert_to_numpy=True, 
                                    normalize_embeddings=True) 
        '''query_embedded:(num_query,embedding_dim)''' 
 
        return query_embedded.astype('float32') 
 
 
    def build_index(self): 
        chunks_embedded = self.chunks_embedding() 
 
        '''chunks_embedded:(num_chunks,embedding_dim)''' 
        embedding_dim = chunks_embedded.shape[1] 
 
        '''像一个空仓库，先规定“以后只能往这里塞384维向量”''' 
        storehouse = faiss.IndexFlatIP(embedding_dim) 
        storehouse.add(chunks_embedded) 
        '''dimension = embedding_dim 
        ntotal = num_chunks 
        ID      Vector 
        0       [0.12, -0.34, ..., 0.21]   # embedding_dim个数字 
        1       [0.43,  0.18, ..., -0.07]  # embedding_dim个数字 
        2       [...] 
        ... 
        19      [...]''' 
 
        return storehouse 
 
 
    def fit(self,chunks_info:list[dict]) -> None:  # 都是构建完成search所需的，这样search时不用每次都构建仓库
 
        self.chunks_info = chunks_info 
 
        self.storehouse = self.build_index() 
 
 
    '''query_embedded，每条chunk的信息(用于最终结果的展示)， 
    存储chunks的仓库，规定搜索的范围''' 
    def search(self,query:str,per_retriever_k:int=30): 
 
        if self.storehouse is None: 
            raise RuntimeError( 
                "VectorRetriever must be fitted before search." 
            ) 

        if per_retriever_k <= 0: return []

        per_retriever_k = min(per_retriever_k,len(self.chunks_info))

 
        scores_list,indices_list = self.storehouse.search( 
            self.single_query_embedding(query),per_retriever_k 
        ) 
        '''storehouse拿着query与仓库中所有的chunks进行内积运算 
        内积结果就是向量夹角的余弦值，越大说明两个文本包含的信息越相似''' 
 
        '''scores_list:[query_num,top-k] 
        indices_list:[query_num,top-k]与每一条query最匹配的top-k个chunk''' 
 
        results = [] 
        for score,index in zip(scores_list[0],indices_list[0]): 
            '''同时遍历分数集合和索引集合''' 
 
            chunk_info = self.chunks_info[index] # 这里是按索引查找，与id是怎么命名的无关 
            result = { 
                "vector_retriever_score": float(score), 
                **chunk_info
            } 
            results.append(result) 

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
            }'''
        return results