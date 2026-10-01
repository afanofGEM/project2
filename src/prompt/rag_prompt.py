'''retrieved_chunks就是reranker返回的包含top-k个元素的chunk list[dict]'''
def build_rag_prompt(query: str,evidence_chunks: list[dict]) -> tuple[str, list[dict]]:

    context_parts = []
    citation_sources = [] # 这个不是给模型的，这个是给系统的，有一个资料i->z真正的chunk的映射
    # 资料id + chunk['source']就是chunk id

    for index, chunk in enumerate(evidence_chunks,start=1):

        num_citation = f"[资料{index}]"

        context_parts.append(
            f"{num_citation}\n" # 回答依据的编号，比如资料1，资料2...
            f"来源：{chunk['source']}\n" # 回答依据的来源文件
            f"内容：{chunk['text']}"
        )

        citation_sources.append(
            {
                "citation_id": num_citation,
                "chunk_id": chunk["id"],
                "source": chunk["source"]
            }
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
            你是一名银行业务问答助手。

            请严格根据下面提供的参考资料回答用户问题。

            要求：

            1. 只能回答参考资料中明确写出的内容。
            2. 禁止根据常识、经验或推测补充资料中没有写明的信息。
            3. 禁止将不同业务场景中的规则互相套用。
            4. 每个关键结论后必须使用下面这种格式标注来源：

            [资料1]

            例如：

            自动转账操作不会成功，银行不再补扣。[资料1]

            5. 不允许使用“根据资料1”“参考资料1”“资料1显示”等其他引用格式，
            必须严格使用方括号格式：[资料1]。

            6. 如果一句话包含多个结论，每个结论都必须有资料直接支持。

            7. 如果资料只说明“转账失败”，
            不得自行推断用户之后应该采取什么措施。

            8. 如果参考资料不足以回答问题，只回答：
            “根据当前提供的资料，无法确定。”

            9. 回答简洁，不要解释资料没有说明的后续影响。

            参考资料：

            {context}

            用户问题：
            {query}

            回答：
            """.strip()

    return prompt,citation_sources
