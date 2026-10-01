from pathlib import Path
from pypdf import PdfReader
import re
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "real"
DOCUMENT_METADATA_JSON_PATH = DATA_ROOT / "DOCUMENT_METADATA.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHUNKS_INFO_PATH = OUTPUT_ROOT / "chunks_info.json"


# 判断某一行是否明显是一个新段落/新条款的开始
PARAGRAPH_START_PATTERN = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百千万零〇两0-9]+(?:章|节|条|款|项)"
    r"|[一二三四五六七八九十百千万零〇两]+[、.]"
    r"|[（(][一二三四五六七八九十百千万零〇两0-9]+[）)]"
    r"|\d+(?:\.\d+)*[、.．)]"
    r"|[①②③④⑤⑥⑦⑧⑨⑩]"
    r")"
)

# 用于长自然段内部按句子切分
SENTENCE_PATTERN = re.compile(r".*?(?:[。！？；.!?;]+|$)")


def load_json(file_path: str):
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


DOCUMENT_METADATA = load_json(DOCUMENT_METADATA_JSON_PATH)

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


'''固定长度切分，适合Markdown等结构简单的文本'''
def text_to_chunks(text:str,chunk_size:int=100,overlap:int=20):
    chunks_list = []
    start_index = 0

    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap必须大于等于0并且小于chunk_size")

    while start_index < len(text):
        chunk = text[start_index:start_index+chunk_size]
        start_index = start_index + chunk_size - overlap
        chunks_list.append(chunk)

    return chunks_list


# 给固定长度切分得到的chunks添加id、source和metadata
def chunks_with_source(texts_list,chunk_size:int=100,overlap:int=20):
    chunks_info = []

    for text_dict in texts_list:
        meta_data = {
            key:value for key,value in text_dict.items()
            if key != 'text'
        }

        chunks_list = text_to_chunks(
            text=text_dict['text'],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_id,chunk in enumerate(chunks_list):
            chunk_info = {
                'id':f"{text_dict['source']}_{chunk_id}",
                'text':chunk,
                **meta_data
            }
            chunks_info.append(chunk_info)

    save_chunks(chunks_info)
    return chunks_info


def normalize_pdf_text(text: str) -> str:
    '''统一PDF文本的换行符并去掉首尾空白'''
    if not text:
        return ""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def clean_pdf_line(line: str) -> str:
    '''把连续空格和Tab压缩成一个普通空格'''
    return re.sub(r"[ \t]+", " ", line).strip()


def merge_pdf_lines(current: str,line: str) -> str:
    '''合并PDF排版造成的单行折行'''
    if not current:
        return line

    separator = " " if current[-1].isascii() and line[0].isascii() else ""
    return current + separator + line


def split_long_paragraph(paragraph:str,chunk_size:int)->list[str]:
    '''超长自然段优先按完整句子切分，超长句子才按字符硬切'''
    sentences = [
        part.strip()
        for part in SENTENCE_PATTERN.findall(paragraph)
        if part.strip()
    ]

    pieces = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(
                sentence[i:i+chunk_size]
                for i in range(0,len(sentence),chunk_size)
            )
        elif current and len(current) + len(sentence) > chunk_size:
            pieces.append(current)
            current = sentence
        else:
            current += sentence

    if current:
        pieces.append(current)

    return pieces


def add_paragraph_to_chunks(
    paragraph:str,
    chunks:list[str],
    current_chunk:str,
    chunk_size:int
)->str:
    '''把一个完整自然段加入最终chunk'''
    if not paragraph:
        return current_chunk

    if len(paragraph) <= chunk_size:
        pieces = [paragraph]
    else:
        pieces = split_long_paragraph(paragraph,chunk_size)

    for piece in pieces:
        if not current_chunk:
            current_chunk = piece
            continue

        candidate_length = len(current_chunk) + 1 + len(piece)

        if candidate_length <= chunk_size:
            current_chunk += "\n" + piece
        else:
            chunks.append(current_chunk)
            current_chunk = piece

    return current_chunk


def pdf_text_to_chunks(text:str,chunk_size:int=500)->list[str]:
    '''识别PDF自然段，在尽量保留段落结构的情况下切分文本'''
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    text = normalize_pdf_text(text)
    if not text:
        return []

    chunks = []
    current_paragraph = ""
    current_chunk = ""
    previous_was_blank = False

    for raw_line in text.split("\n"):
        line = clean_pdf_line(raw_line)

        if not line:
            if current_paragraph:
                current_chunk = add_paragraph_to_chunks(
                    paragraph=current_paragraph,
                    chunks=chunks,
                    current_chunk=current_chunk,
                    chunk_size=chunk_size
                )
                current_paragraph = ""

            previous_was_blank = True
            continue

        starts_new_paragraph = bool(PARAGRAPH_START_PATTERN.match(line))

        if current_paragraph and (previous_was_blank or starts_new_paragraph):
            current_chunk = add_paragraph_to_chunks(
                paragraph=current_paragraph,
                chunks=chunks,
                current_chunk=current_chunk,
                chunk_size=chunk_size
            )
            current_paragraph = line
        else:
            current_paragraph = merge_pdf_lines(current_paragraph,line)

        previous_was_blank = False

    if current_paragraph:
        current_chunk = add_paragraph_to_chunks(
            paragraph=current_paragraph,
            chunks=chunks,
            current_chunk=current_chunk,
            chunk_size=chunk_size
        )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# 给PDF段落切分得到的chunks添加id、source和metadata
def paragraph_chunks_with_source(texts_list:list[dict],chunk_size:int=500)->list[dict]:
    chunks_info = []

    for text_dict in texts_list:
        text = text_dict.get('text','')
        meta_data = {
            key:value for key,value in text_dict.items()
            if key != 'text'
        }

        chunks_list = pdf_text_to_chunks(text=text,chunk_size=chunk_size)

        for chunk_id,chunk in enumerate(chunks_list):
            chunk_info = {
                'id':f"{text_dict['source']}_{chunk_id}",
                'text':chunk,
                **meta_data
            }
            chunks_info.append(chunk_info)

    save_chunks(chunks_info)
    return chunks_info


def save_chunks(chunks_info:list[dict])->None:
    '''把切分结果保存到outputs/chunks_info.json'''
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)

    with open(CHUNKS_INFO_PATH,"w",encoding="utf-8") as file:
        json.dump(
            chunks_info,
            file,
            ensure_ascii=False,
            indent=2
        )


def main() -> None:
    """按照当前 PDF 自然段切分规则重新生成 chunks_info.json。"""
    texts_list = pdf_to_list(DATA_ROOT)

    if not texts_list:
        raise RuntimeError(f"未在 {DATA_ROOT} 中找到 PDF 文件")

    chunks_info = paragraph_chunks_with_source(
        texts_list=texts_list,
        chunk_size=500
    )

if __name__ == "__main__":
    main()
