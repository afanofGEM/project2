import json
from src.prompt.query_planner_prompt import build_query_planner_prompt

'''0925版的改动并不影响
single retriever - hybrid retriever - reranker- LLM生成的链路
而是在single retriever前将query划分成1个或多个子问题，
实现这一功能的就是QueryPrompt - QueryPlanner
QueryPrompt就告诉AI帮我划分子问题，并且返回固定的json格式
{
    "subqueries": [
        {
            "query": "完整的检索问题"
        }
    ]
}
QueryPlanner就是重新提取子问题ID的，并且做一些安全性检查'''
class QueryPlanner:

    def __init__(self,generator,max_subqueries: int | None = None):

        if (
            max_subqueries is not None
            and max_subqueries <= 0
        ):
            raise ValueError("max_subqueries必须大于0")

        self.generator = generator
        self.max_subqueries = max_subqueries


    # Qwen说它划分不了问题的情况
    def build_fallback_plan(self, query: str, fallback_reason:str) -> dict:
        
        """
        当模型输出无法解析时，退回原始问题。
        返回的结构是把所有原问题全部当成一个子问题

        这里不是判断原问题一定是 simple，
        而是保证检索流程还能继续运行。
        """

        return {
            "query_type": "simple", # 最终认定模型是否可划分
            "subqueries": [
                {
                    "id": "sq_1",
                    "query": query
                }
            ],
            "used_fallback": True,
            "fallback_reason":fallback_reason # 记录一下为什么划分不了
        }


    # 用于去除模型在json两端增加的一些说明
    def extract_json(self, model_output: str) -> dict:
        """
        从模型输出中提取最外层 JSON 对象。

        即使模型意外输出：
        ```json
        {...}
        ```
        或者在 JSON 前后增加说明，也可以尝试提取。
        """

        start_index = model_output.find("{")
        end_index = model_output.rfind("}")

        if start_index == -1 or end_index == -1:
            raise ValueError("模型输出中没有找到JSON对象")

        if start_index >= end_index:
            raise ValueError("模型输出中的JSON结构不完整")

        json_text = model_output[start_index:end_index + 1]

        return json.loads(json_text)


    def validate_json(self, query_planner_json: dict) -> dict:
        """
        query_planner_json: 这是针对一个query的
        {
            "subqueries": 
            [
                {
                    "query": "完整的检索问题"
                },
                {
                    "query":",......"
                }
            ]
        }
        """

        if not isinstance(query_planner_json, dict):
            raise ValueError("Query Plan Json必须是字典")

        subqueries = query_planner_json.get("subqueries") #list[dict]
        '''            
        [
            {
                "query": "完整的检索问题"
            },
            {
                "query":",......"
            }
        ]'''

        if not isinstance(subqueries, list):
            raise ValueError("subqueries必须是列表")

        if not subqueries:
            raise ValueError("subqueries不能为空")

        # 只有在创建query_planner时规定了max_subqueries时，才会进行检查
        if (
            self.max_subqueries is not None
            and len(subqueries) > self.max_subqueries
        ):
            raise ValueError(
                f"subqueries不能超过{self.max_subqueries}个"
            )

        # list[dict] 包含每一个子问题的text和重新编排的id
        normalized_query_planner_subqueries = []
        
        # 用于收集所有划分好的子问题的集合
        seen_queries = set()

        for item in subqueries: #dict

            if not isinstance(item, dict):
                raise ValueError("每个subquery必须是字典")

            subquery = item.get("query")

            if not isinstance(subquery, str):
                raise ValueError("subquery中的query必须是字符串")

            subquery = subquery.strip()

            if not subquery:
                raise ValueError("subquery中的query不能为空")

            # 避免模型生成重复子问题
            if subquery in seen_queries:
                continue

            # 只有没出现过的全新的子问题，才能进入结果集
            seen_queries.add(subquery)

            # 由程序重新编号
            normalized_query_planner_subqueries.append(
                {
                    "id": f"sq_{len(normalized_query_planner_subqueries) + 1}",
                    "query": subquery
                }
            )

        if not normalized_query_planner_subqueries:
            raise ValueError("去重后没有可用的subquery")

        # query_type 问题实际上是否可划分，
        if len(normalized_query_planner_subqueries) == 1:
            query_type = "simple"
        else:
            query_type = "complex"

        return {
            "query_type": query_type,
            "subqueries": normalized_query_planner_subqueries,
            "used_fallback": False,
            "fallback_reason":None
        }


    def run(self, query: str, include_debug: bool = False) -> dict:
        """
        Query Planner的统一调用入口。
        """

        query = query.strip()

        if not query:
            raise ValueError("query不能为空")

        # 这个只是告诉Qwen，你要怎么样生成我的输出json
        prompt = build_query_planner_prompt(query=query)

        # 模型调用放在try外面。
        # 模型加载或推理失败时不应该被误认为JSON解析失败。
        # 这里才是真正的生成json
        model_output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=256
        )

        try:
            query_planner_json = self.extract_json(model_output=model_output)

            query_planner = self.validate_json(query_planner_json=query_planner_json)
            '''{
                “query_type”:由程序给出的官方问题类型认证
                "subqueries": [
                    {
                        "query": "完整的检索问题"
                    },
                    {
                        "query":"......"    
                    }
                ]
                “use_fallback”:False
                "fallback_reason":None
            }'''

            '''用于解决空query	 没有JSON	超过数量限制'''
        except ValueError as error:

            query_planner = self.build_fallback_plan(
                query=query,
                fallback_reason=str(error)
            )
            # 和正常分隔query的返回类型一样
            '''
            {
                "query_type": "simple", # 最终认定模型是否可划分
                "subqueries": [
                    {
                        "id": "sq_1",
                        "query": query
                    }
                ],
                "used_fallback": True,
                "fallback_reason":fallback_reason # 记录一下为什么划分失败
            }'''
            
        if include_debug:
            # 如果是用测试集跑的话，把Qwen生成的原始json也传回去
            query_planner["raw_model_output"] = model_output 

        return query_planner
        
