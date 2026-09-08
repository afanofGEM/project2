'''保存chunks_embedded'''
import faiss

def build_index(chunks_embedded):
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