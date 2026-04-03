import json
import re
from typing import List

from jinja2 import Template

from src.logger import logger
from src.prompts.base_prompt import BasePrompt


class SynthesisAgentPrompt(BasePrompt):
    """
    多答案综合总结器。
    输入多个候选回答，输出单一整合文本。
    """

    REQUIRED_SECTIONS = [
        "结论概览",
        "核心依据整合",
        "完整展开说明",
        "风险与边界条件",
        "实务/操作建议",
        "不确定性说明",
    ]

    TEMPLATE_STR = """<role>
Multi-Answer Synthesis Agent
</role>
<task>
你会收到 query 与多个 answers。请整合为单一高质量中文文本。
</task>
<output_rule>
- 输出必须是单一整合文本，不得输出 JSON 数组。
- 输出必须包含以下 6 个标准部分（建议使用小标题）：
  1. 结论概览
  2. 核心依据整合
  3. 完整展开说明
  4. 风险与边界条件
  5. 实务/操作建议
  6. 不确定性说明
- 不得出现英文字符。
</output_rule>
<input>
query: {{ query }}
answers:
{{ answers_json }}
</input>
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, query: str, answers: List[str]) -> str:
        answers_json = json.dumps(answers, ensure_ascii=False, indent=2)
        return self.template.render(query=query, answers_json=answers_json)

    def parse_response(self, response: str) -> str:
        text = response.strip()
        if not text:
            raise ValueError("校验失败：输出为空，期望为单一整合文本。")

        # 规则1：输出不能是 JSON 数组
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raise ValueError("校验失败：输出不能是 JSON 数组，必须为纯文本。")
        except json.JSONDecodeError:
            # 不是合法 JSON，符合“纯文本”方向，继续校验
            pass

        # 规则2：必须包含 6 个标准部分
        missing = [sec for sec in self.REQUIRED_SECTIONS if sec not in text]
        if missing:
            raise ValueError("校验失败：缺少标准部分: " + "、".join(missing))

        # 规则3：无英文字符
        if re.search(r"[a-zA-Z]", text):
            raise ValueError("校验失败：检测到英文字符，输出必须为纯中文文本。")

        logger.debug("SynthesisAgent 解析通过，输出结构完整")
        return text
