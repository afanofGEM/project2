'''对中文的chunks embedding'''
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(MODEL_NAME) # 这个模型默认embedding_dim = 512

'''一次性对一个chunks集合embedding,
把文本映射到一个高维语义空间'''
def chunks_embedding(chunks_info:list[dict]):
    chunks_list = []
    for chunk_info in chunks_info:
        '''{
                'id':chunk_id,
                'chunk':chunk,
                'source':text_dict['source']
            }'''
        chunks_list.append(chunk_info['text'])

    # 对每个编码后的chunk向量正则化，向量长度为1
    # Inner Product = Cosine Similarity内积结果就是向量之间的夹角
    chunks_embedd = model.encode(chunks_list,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)
    '''chunks_embedd:(chunks_num,embedding_dim)
    检索实际上变成：用户Query的向量，和哪个Chunk的向量最相似
    使用向量之间的夹角余弦值判断，越接近1说明两个向量的方向越接近，
    同理Query与某个文本向量也最相似'''

    return chunks_embedd.astype('float32')


def single_query_embedding(query:str):

    query_embedded = model.encode([query],
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)
    '''query_embedded:(1,embedding_dim)'''

    return query_embedded.astype('float32')


def multi_query_embedding(queries:list[str]):

    query_embedded = model.encode(queries,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)
    '''query_embedded:(num_query,embedding_dim)'''

    return query_embedded.astype('float32')
