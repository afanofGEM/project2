import json
import re
from pathlib import Path


OUTPUT_ROOT = Path(__file__).parent.parent / "outputs"
CHUNKS_INFO_PATH = OUTPUT_ROOT / "chunks_info.json"


# ============================================================
# 1. PDF 段落结构规则
# ============================================================

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
SENTENCE_PATTERN = re.compile(
    r".*?(?:[。！？；.!?;]+|$)"
)


# ============================================================
# 2. 基础文本处理
# ============================================================
def normalize_pdf_text(text: str) -> str:
    """
    对 PDF 提取文本做基础清洗。
    这里只处理最基础、最稳定的事情：1. 统一换行符 2. 去掉全文首尾空白
    """
    if not text: return ""
    return (text.replace("\r\n", "\n").replace("\r", "\n").strip())


def clean_pdf_line(line: str) -> str:
    """连续空格 / Tab 压缩成一个普通空格。"""
    return re.sub(r"[ \t]+", " ", line).strip()


def merge_pdf_lines(current: str, line: str) -> str:
    """
    合并 PDF 排版造成的单行折行。

    中文：
        银行卡遗失后应立即
        联系银行

        ->
        银行卡遗失后应立即联系银行

    英文：
        please contact
        the bank

        ->
        please contact the bank
    """
    if not current: return line

    separator = (" " if current[-1].isascii() and line[0].isascii() else "")

    return current + separator + line


# ============================================================
# 3. 超长自然段切分
# ============================================================
def split_long_paragraph(
    paragraph: str,
    chunk_size: int
) -> list[str]:
    """
    当一个自然段本身已经超过 chunk_size 时，
    优先按照完整句子切分。

    如果某一句自己都超过 chunk_size，
    才按固定字符长度硬切。
    """

    sentences = [
        part.strip()
        for part in SENTENCE_PATTERN.findall(paragraph)
        if part.strip()
    ]

    pieces = []
    current = ""

    for sentence in sentences:

        # 情况1：
        # 单句话自己就超过 chunk_size
        if len(sentence) > chunk_size:

            # 先保存前面已经积累的内容
            if current:
                pieces.append(current)
                current = ""

            # 超长句只能硬切
            pieces.extend(
                sentence[i:i + chunk_size]
                for i in range(0, len(sentence), chunk_size)
            )

        # 情况2：
        # 当前内容 + 新句子会超过 chunk_size
        elif current and len(current) + len(sentence) > chunk_size:

            pieces.append(current)
            current = sentence

        # 情况3：
        # 还能继续放
        else:
            current += sentence

    if current:
        pieces.append(current)

    return pieces


# ============================================================
# 4. 将一个完整自然段加入最终 chunk
# ============================================================

def add_paragraph_to_chunks(
    paragraph: str,
    chunks: list[str],
    current_chunk: str,
    chunk_size: int
) -> str:
    """
    把一个已经恢复完成的自然段加入 chunk。

    返回新的 current_chunk。

    逻辑：
        自然段 <= chunk_size
            -> 直接尝试加入当前 chunk

        自然段 > chunk_size
            -> 先按句子切成多个 piece
            -> 再逐个加入 chunk
    """

    if not paragraph:
        return current_chunk

    if len(paragraph) <= chunk_size:
        pieces = [paragraph]
    else:
        pieces = split_long_paragraph(
            paragraph,
            chunk_size
        )

    for piece in pieces:

        # 当前 chunk 还是空的
        if not current_chunk:
            current_chunk = piece
            continue

        # 两个自然段之间保留一个 \n
        candidate_length = (
            len(current_chunk)
            + 1
            + len(piece)
        )

        # 还能放下
        if candidate_length <= chunk_size:
            current_chunk += "\n" + piece

        # 放不下
        else:
            chunks.append(current_chunk)
            current_chunk = piece

    return current_chunk


# ============================================================
# 5. 主函数：PDF全文 -> chunks
# ============================================================

def pdf_text_to_chunks(text: str,chunk_size: int = 500) -> list[str]:
    """
    将一个 PDF 提取出来的完整文本转换成 chunks。

    主流程：

    PDF全文
        ↓
    逐行扫描
        ↓
    空行 / 条款编号
        -> 判断自然段边界

    普通单换行
        -> 认为可能只是 PDF 排版折行
        -> 拼接到 current_paragraph

    自然段结束
        ↓
    直接尝试加入 current_chunk

    current_chunk 超过 chunk_size
        ↓
    封存到 chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    text = normalize_pdf_text(text)
    if not text: return []

    chunks = []

    # 当前正在恢复的自然段
    current_paragraph = ""

    # 当前正在构建的最终 chunk
    current_chunk = ""

    # 标记上一行是否是空行
    previous_was_blank = False

    for raw_line in text.split("\n"):

        line = clean_pdf_line(raw_line) # 处理格式错误的多个空格\t等

        # ----------------------------------------------------
        # 情况1：遇到空行
        #
        # 空行比普通 \n 更像真正的自然段边界
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 情况2：这一行明显是新的条款 / 段落
        #
        # 第一条
        # 二、
        # （三）
        # 1.
        # ①
        # ----------------------------------------------------
        starts_new_paragraph = bool(
            PARAGRAPH_START_PATTERN.match(line)
        )

        if current_paragraph and (
            previous_was_blank
            or starts_new_paragraph
        ):

            # 当前自然段结束
            current_chunk = add_paragraph_to_chunks(
                paragraph=current_paragraph,
                chunks=chunks,
                current_chunk=current_chunk,
                chunk_size=chunk_size
            )

            # 当前行开启新自然段
            current_paragraph = line

        else:

            # ------------------------------------------------
            # 情况3：普通单换行
            #
            # 默认认为可能只是 PDF 页面排版折行
            # 所以继续拼接
            # ------------------------------------------------
            current_paragraph = merge_pdf_lines(
                current_paragraph,
                line
            )

        previous_was_blank = False

    # ========================================================
    # PDF最后可能还有一个自然段没有处理
    # ========================================================

    if current_paragraph:
        current_chunk = add_paragraph_to_chunks(
            paragraph=current_paragraph,
            chunks=chunks,
            current_chunk=current_chunk,
            chunk_size=chunk_size
        )

    # 最后一个 current_chunk 也需要封存
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ============================================================
# 6. 给 chunks 添加 source / metadata
# ============================================================

def paragraph_chunks_with_source(texts_list: list[dict],chunk_size: int = 500) -> list[dict]:
    """
    将多个 PDF 文档转换成带 metadata 的 chunks。
    texts_list 示例：
    [
        {
            "text": "PDF全文……",
            "source": "个人存款证明.pdf",
            "document_type": "存款证明",
            "source_org": "中国农业银行",
            "updated_at": "2023",
            "source_url": "https://..."
        }
    ]
    返回：
    [
        {
            "id": "个人存款证明.pdf_0",
            "text": "某个chunk内容",
            "source": "个人存款证明.pdf",
            "document_type": "存款证明",
            ...
        }
    ]
    """

    chunks_info = []

    for text_dict in texts_list:

        text = text_dict.get("text", "")

        meta_data = {
            key: value
            for key, value in text_dict.items()
            if key != "text"
        }

        chunks_list = pdf_text_to_chunks(text=text,chunk_size=chunk_size)

        for chunk_id, chunk in enumerate(chunks_list):

            chunk_info = {
                "id": chunk_id,
                "text": chunk,
                **meta_data
            }

            chunks_info.append(chunk_info)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CHUNKS_INFO_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks_info,
            f,
            ensure_ascii=False,
            indent=2
        )

    return chunks_info

'''先识别自然段
↓
一个自然段能完整放进当前 chunk
→ 整段放进去

放不下
→ 当前 chunk 先封存
→ 这个自然段放进下一个 chunk

如果某个自然段自己就 > chunk_size
→ 再按句子切
→ 还不行才硬切字符'''