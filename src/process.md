数据的变化过程：
1. 分散在各个md文件中的说明手册：

# 转账业务说明
## 行内转账
同一银行账户之间进行人民币转账通常不收取手续费，资金一般可以实时到账。

2. 一个以字典为元素的列表**texts_list**：
每一条text是一个md文件的**所有文字**
[
    {
        "text": "信用卡年费政策……",
        "source": "credit_card.md"
    },
    {
        "text": "转账业务说明……",
        "source": "transfer.md"
    },
    ...
]

3. 利用texts_list划分chunk一个以字典为元素的列表**chunks_info**：
每一个字典代表一条由**start_index,chunk_size,overlap**决定的chunk
'''[{
    'id':chunk_id, int
    'chunk':chunk, str
    'source':text_dict['source'] str
},...]'''

4. 利用chunks_info提取chunks列表(chunks_num,1)

再利用模型
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(MODEL_NAME) # 这个模型默认embedding_dim = 512
对**chunks列表**embedding,chunks_embedded(chunks_num,embedding_dim)
在对用户的**查询向量Query**进行embedding,query_embedded
**embedding的作用是把一条chunk变成高维空间中的一个向量**
**同时将向量的长度正则化为1**