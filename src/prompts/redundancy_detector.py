import json
import re
from typing import Dict, List

from jinja2 import Template

from src.logger import logger
from src.prompts.base_prompt import BasePrompt


class RedundancyDetectorPrompt(BasePrompt):
    """
    冗余检测器 Prompt。
    输入多个 planner 回答，输出需要删除的下标数组。
    """

    TEMPLATE_STR = """<role>
Facet Redundancy Detector
</role>
<task>
你会收到一个 query 以及多个回答项 planners。
每个回答项结构为 {planner, answer}。
请判断哪些回答与其他回答在核心结论或论证路径上冗余，并输出需要删除的下标数组。
</task>
<format>
- 只输出 JSON 数组。
- 数组元素为整数下标，表示应删除的 planners 索引。
- 如果没有冗余，输出 []。
- 不要输出解释或额外文字。
</format>
<rule>
- 下标必须基于输入数组的 0-based 索引。
- 只能输出存在的下标。
</rule>
## 输入
query: {{ query }}
planners:
{{ planners_json }}
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, query: str, planners: List[Dict[str, str]]) -> str:
        planners_json = json.dumps(planners, ensure_ascii=False, indent=2)
        return self.template.render(query=query, planners_json=planners_json)

    def parse_response(self, response: str, planners: List[Dict[str, str]]) -> List[int]:
        """
        解析模型输出并做严格校验:
        1) 必须是 JSON 数组
        2) 元素必须是整数
        3) 下标必须在有效范围内
        """
        text = response.strip()

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("解析失败：未能在模型输出中检测到有效的边界符 '[' 和 ']'。")

        json_str = match.group(0)

        try:
            index_list = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析失败：不是合法的 JSON 数组结构。内部错误: {str(e)}")

        if not isinstance(index_list, list):
            raise ValueError(f"解析失败：预期输出为 JSON 数组(list)，但收到了 {type(index_list).__name__}")

        max_index = len(planners) - 1
        validated: List[int] = []
        seen = set()
        for i, item in enumerate(index_list):
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(f"校验失败：数组第 {i} 个元素不是整数下标。")

            if item < 0 or item > max_index:
                raise ValueError(
                    f"校验失败：下标 {item} 超出有效范围 [0, {max_index}]。"
                )

            if item in seen:
                continue

            seen.add(item)
            validated.append(item)

        logger.debug(f"RedundancyDetector 解析通过，输出 {len(validated)} 个删除下标")
        return validated
