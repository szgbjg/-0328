import json
import re
from typing import List
from jinja2 import Template

from src.prompts.base_prompt import BasePrompt
from src.logger import logger


class QuestionCreatorPrompt(BasePrompt):
    """
    问题创造者 (Question Creator) Prompt 管理类
    负责渲染上下文生成 Prompt，并严密解析模型输出的 JSON 数组结构。
    """
    
    TEMPLATE_STR = """<role>
问题创造者（Question Creator）
</role>
<task>
你会收到一个上下文（context，可能由多段资料/片段组成）。你的任务是：严格基于这些上下文创造若干“可被上下文回答”的高质量问题。任何问题的答案都必须能在这些上下文中直接找到或通过合理归纳得到。每次生成三个左右的不同问题。
</task>
<rule>
- 只根据 context 造问题：不得依赖常识补全、不得引入 context 之外的新事实/新实体/新结论。
- 每个问题都必须可被 context 支撑：能在 context 中定位到明确依据（原句或可归纳的信息）。
- 问题必须独立表述：不要出现“根据以上/上述/这段话/文中”等指代性措辞。
- 不要生成需要外部资料才能回答的问题（如“最新进展”“现实世界数据”“超出材料的推测”）。
- 避免重复：同义改写视为重复；同一答案的不同问法只保留一个。
- 避免泄露过程：不要在问题中暗示“上下文提到/资料显示/检索到”等措辞。
- **多问题低交集**：不同问题尽量对应 context 的不同信息点/不同段落/不同概念；尽量避免共享同一核心答案或大量重叠依据。
- 若 context 信息不足以生成合格且低交集的问题集合，可以只生成一个问题，并确保以单元素数组 ["问题1"] 返回。如果完全无信息则返回空数组 []。
</rule>
<format>
- 只输出一个 JSON 数组（不要额外文字、不要解释、不要代码块标注）。
- 数组元素为字符串，每个字符串是一条问题。
- 示例：["问题1","问题2","问题3"]
</format>
<strategy>
- 先通读 context，将信息点按“主题簇”划分（定义/条件/流程/对比/机制/风险/限制/建议/例外/指标等）。
- 优先选择不同主题簇各出 1 题，确保问题之间的答案依据尽量不重叠。
- 优先覆盖高信息增益的问题：可验证、可定位、信息密度高、能区分概念边界。
- 问题重复自检：采用聚类自检法：若两道题答案高度一致或提取的依据重叠度高，则自动舍弃或合并其中一题，直到交集最小。
</strategy>
## 输入
<context>
{% for item in context_list %}
[{{ loop.index }}] {{ item }}
{% endfor %}
</context>
## 输出
- 只输出 JSON 数组
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, context: List[str]) -> str:
        """
        渲染 Prompt
        
        Args:
            context (List[str]): 包含多段医药资料/片段的字符串列表
            
        Returns:
            str: 组装后的完整 Prompt 文本
        """
        if not context:
            logger.warning("传入的 context 列表为空")
            
        return self.template.render(context_list=context)

    def parse_response(self, response: str) -> List[str]:
        """
        验证和解析模型的 JSON 数组输出
        
        Args:
            response (str): 大模型返回的纯文本结果
            
        Returns:
            List[str]: 解析并验证通过的问题列表
            
        Raises:
            ValueError: 若 JSON 解析失败或数组内容不符合规则
        """
        # 1. 提取JSON部分 (使用正则防止大模型不听话包裹了 ```json ... ```)
        text = response.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError(
                "解析失败：未能在模型输出中检测到有效的边界符 '[' 和 ']'。请检查大模型是否违规输出了冗余对话文字。"
            )
            
        json_str = match.group(0)

        # 2. 尝试反序列化
        try:
            questions = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析失败：不是合法的 JSON 数组结构。内部错误: {str(e)}")

        # 3. 验证格式类型
        if not isinstance(questions, list):
            raise ValueError(f"解析失败：预期输出为 JSON 数组(list)，但收到了 {type(questions).__name__}")
            
        # 4. 允许为空的降级情况
        if len(questions) == 0:
            logger.info("模型判定上下文不足，返回了空数组")
            return []

        # 5. 验证数量 (1-5个)
        if len(questions) > 5:
            raise ValueError(f"校验失败：生成的问题数量({len(questions)})超出了设定的上限 5 个。")

        # 6. 验证字符串长度和类型
        valid_questions = []
        for i, q in enumerate(questions):
            if not isinstance(q, str):
                raise ValueError(f"校验失败：数组第 {i} 个元素不是字符串格式。")
            
            clean_q = q.strip()
            if len(clean_q) <= 10:
                raise ValueError(f"校验失败：生成的问题 '{clean_q}' 长度过短（<=10个字符），缺乏足够的信息量。")
                
            valid_questions.append(clean_q)

        logger.debug(f"QuestionCreator 解析通过，成功提取 {len(valid_questions)} 个问题")
        return valid_questions
