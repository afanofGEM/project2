import json
import re
from pathlib import Path

OUTPUT_ROOT = Path(__file__).parent.parent / 'outputs'
CHUNKS_INFO_PATH = OUTPUT_ROOT / "chunks_info.json"
'''把一篇文档分割成不同chunk的意义:
比如account.md:
1. 账户的支付密码怎么改。答：。。。。。。
2. 账户的挂失流程是怎么样的。答：。。。。。。
3. 账户的开户流程是怎么样的。答：。。。。。
4. 账户的开户材料需要什么。答：。。。。。。
比如说当用户说：我曾经向诈骗网站泄露过银行卡号和支付密码，我
想挂失我的银行卡。
这个时候AI就不会给他一份account.md，而是通过相似度给他说挂失流程是怎么样的'''

# 把一个文件中的一长串text转化成一个chunks列表
def text_to_chunks(text:str,chunk_size:int=100,overlap:int=20):
    chunks_list = []
    start_index = 0

    while start_index <= len(text):
        chunk = text[start_index:start_index+chunk_size]
        start_index = start_index + chunk_size - overlap

        chunks_list.append(chunk)

    return chunks_list


'''overlap的作用:降低切分边界导致语义断裂的问题
假设原文是：
    如果客户连续输错三次银行卡密码，
    系统会自动锁定账户24小时。
    锁定期间无法进行转账、取现或支付。
如果没有overlap，刚好切成：
Chunk 1：
    如果客户连续输错三次银行卡密码，
    系统会自动锁定账户24小时。
Chunk 2：
    锁定期间无法进行转账、取现或支付。
现在用户问：
    银行卡密码输错三次后，还能转账吗？
问题就很明显了，一条完整的因果关系被拆成两半了

但是不切，不能保证这几个chunk都进入最后筛选的top-k的k'''

# 转化所有文件中的text为不同的chunks集合，再为每一个chunk提供id和source
def chunks_with_source(texts_list,chunk_size:int=100,
                         overlap:int=20):
    '''texts_list=[
        {
            "text": "存款证明……",
            "source": "11_个人存款证明申请须知.pdf"
            "document_type": "存款证明",
            "source_org": "中国农业银行",
            "updated_at": "2023",
            "source_url": "https://www.abchina.com/cn/PersonaLServices/zcxy221101/zhkh221101/202301/P020230111446363779555.pdf"
        },
        {
            "text": "转账业务说明……",
            "source": "transfer.md",
            "document_type": "存款证明",
            "source_org": "中国农业银行",
            "updated_at": "2023",
            "source_url": "https://www.abchina.com/cn/PersonaLServices/zcxy221101/zhkh221101/202301/P020230111446363779555.pdf"
        },
        ...
    ]'''
    chunks_info = []

    for text_dict in texts_list:

        meta_data = {
            key:value for key,value in text_dict.items()
            if key != 'text'
        }

        chunks_list = text_to_chunks(text=text_dict['text'],
                                     chunk_size=chunk_size,
                                     overlap=overlap)
        '''list[str],存每个chunk内容'''

        for chunk_id,chunk in enumerate(chunks_list):
            chunk_info = {
                'id':f"{text_dict['source']}_{chunk_id}",
                'text':chunk,
                **meta_data
            }

            chunks_info.append(chunk_info)


    with open(CHUNKS_INFO_PATH,"w",encoding="utf-8") as f:
        json.dump(
            chunks_info,
            f,
            ensure_ascii=False,
            indent=2
        )
    return chunks_info

