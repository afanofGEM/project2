from pathlib import Path
from pypdf import PdfReader
import re

'''目标，把分散在不同md/pdf文档中的知识库集中成
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
]'''


DOCUMENT_METADATA = {
    "01_个人账户综合服务协议.pdf": {
        "document_type": "个人账户服务",
        "source_org": "中国银行",
        "updated_at": "2025",
        "source_url": "https://pic.bankofchina.com/bocappd/pbservice/202601/P020260109615510106614.pdf"
    },

    "02_银行卡挂失客户须知.pdf": {
        "document_type": "银行卡挂失",
        "source_org": "中国银行",
        "updated_at": None,
        "source_url": "https://www.boc.cn/ebanking/online/201310/P020210723336996476793.pdf"
    },

    "03_借记卡章程.pdf": {
        "document_type": "借记卡",
        "source_org": "华夏银行",
        "updated_at": "2023",
        "source_url": "https://www.hxb.com.cn/images/jrhx/khfw/zxgg/2023/01/06/06163050445E100116D165A51AA94C229E8D9684.pdf"
    },

    "04_信用卡综合服务合约.pdf": {
        "document_type": "信用卡",
        "source_org": "中国民生银行",
        "updated_at": "2025-08",
        "source_url": "https://creditcard.cmbc.com.cn/tyglweb/statics/tyglweb/home/cn/active/wap/wonderful/pdfs/1GRKHY8Y.pdf"
    },

    "05_信用卡收费标准.pdf": {
        "document_type": "信用卡收费",
        "source_org": "中国建设银行",
        "updated_at": "2024",
        "source_url": "https://us.ccb.com/cn/creditcard/news/upload/20240719165838759713.pdf"
    },

    "06_电子银行个人客户服务协议.pdf": {
        "document_type": "电子银行",
        "source_org": "中国农业银行",
        "updated_at": None,
        "source_url": "https://www.abchina.com/cn/PersonalServices/grzsc/zhkh/201912/P020220211569048870335.pdf"
    },

    "07_手机号转账服务协议.pdf": {
        "document_type": "转账服务",
        "source_org": "中国银行",
        "updated_at": "2020",
        "source_url": "https://www.boc.cn/pbservice/pb4/202012/P020201204853990084008.pdf"
    },

    "08_个人自动转账业务协议.pdf": {
        "document_type": "自动转账",
        "source_org": "中国农业银行",
        "updated_at": "2024",
        "source_url": "https://www.abchina.com/cn/PersonaLServices/zcxy221101/zfjs221101/202301/P020250107539334245114.pdf"
    },

    "09_快捷支付授权扣款协议.pdf": {
        "document_type": "快捷支付",
        "source_org": "中国农业银行",
        "updated_at": "2023",
        "source_url": "https://www.abchina.com/cn/PersonalServices/zcxy221101/zfjs221101/202311/P020231110537016241541.pdf"
    },

    "10_个人贷款合同.pdf": {
        "document_type": "个人贷款",
        "source_org": "中国农业银行",
        "updated_at": "2023",
        "source_url": "https://www.abchina.com/cn/PersonalServices/Loans/jkhtgs/202303/W020230320333834190555.pdf"
    },

    "11_个人存款证明申请须知.pdf": {
        "document_type": "存款证明",
        "source_org": "中国农业银行",
        "updated_at": "2023",
        "source_url": "https://www.abchina.com/cn/PersonaLServices/zcxy221101/zhkh221101/202301/P020230111446363779555.pdf"
    }
}


def clean_text(text: str) -> str:

    # 制表符变成空格
    text = text.replace("\t", " ")

    # 连续多个空格变成一个
    text = re.sub(r" +", " ", text)

    # 连续多个换行最多保留一个
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def md_to_list(data_path: str):

    data_path = Path(data_path)

    texts_list = []

    # 在文件路径中找所有以md结尾的文件
    for file in data_path.glob('*.md'):

        file_text = file.read_text(encoding='utf-8') #得到的是一长串字符串，后续需要切割
        text_dict = {
            'text':file_text,
            'source':file.name
        }
        texts_list.append(text_dict)

    return texts_list


def pdf_to_list(data_path:str):
    data_path = Path(data_path)

    texts_list = []

    '''file:Path对象，内容：路径名'''
    for file in data_path.glob('*.pdf'):
        pdf_reader = PdfReader(file)
        file_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text() #str

            if page_text:
                file_text += page_text + '\n'  # 在终端输出时美观，分页自然分行

        file_text = clean_text(file_text)

        meta_data = DOCUMENT_METADATA[file.name]

        text_dict = {
            'text':file_text,
            'source':file.name,
            **meta_data #自动写入键值对
        }

        texts_list.append(text_dict)

    return texts_list