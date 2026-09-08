from pathlib import Path
from src.document_loader_1 import pdf_to_list
from src.chunker_2 import chunks_with_source
from src.embedding_3 import chunks_embedding,multi_query_embedding
from src.embedded_store_4 import build_index
from src.retriever_5 import search

texts_list = pdf_to_list("data/real")
'''
[
    {
        "text": "PDF全文……",
        "source": "01_个人账户综合服务协议.pdf",
        "document_type": "个人账户服务",
        "source_org": "中国银行",
        "updated_at": "2025",
        "source_url": "https://..."
    },
    ...
]
'''

chunks_info = chunks_with_source(texts_list,chunk_size=100,overlap=20)
'''
[
    {
        "id": 0,
        "text": "某一个chunk正文……",
        "source": "01_个人账户综合服务协议.pdf",
        "document_type": "个人账户服务",
        "source_org": "中国银行",
        "updated_at": "2025",
        "source_url": "https://..."
    },
    ...
]
'''

chunks_embedded = chunks_embedding(chunks_info)
storehouse = build_index(chunks_embedded)

print("texts_list count:", len(texts_list))
print("chunks_info count:", len(chunks_info))
print("chunks_embedded shape:", chunks_embedded.shape)
print("FAISS vector count:", storehouse.ntotal)

queries = [
    "信用卡怎么免年费？",
    "银行卡丢了怎么办？",
    "转账限额是多少？",
    "定期存款能不能提前取？",
    "手机银行密码忘了怎么办？",
]
queries_embedded = multi_query_embedding(queries) #（num_query,embedding_dim)二维numpy向量

for id,query in enumerate(queries_embedded): #(embedding_dim)一维numpy数组
    results = search(query=query.reshape(1,-1),chunks_info=chunks_info,
                     storehouse=storehouse,top_k=3)
    '''尺寸都是(1,top_k)因为是一条一条查询的'''

    print("\n" + "=" * 80)
    print("Query:", queries[id])

    '''从1开始计数'''
    for rank, result in enumerate(results, start=1):
        print(f"\nTop {rank}")
        print(f"score: {result['score']:.4f}")
        print(f"source: {result['source']}")
        print(f"chunk_id: {result['id']}")
        print(f"text: {result['text']}")