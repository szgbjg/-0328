import json
import re
from typing import List
from jinja2 import Template

from src.prompts.base_prompt import BasePrompt
from src.logger import logger


class FacetPlannerPrompt(BasePrompt):
    """
    问题角度规划者 (Facet Planner) Prompt 管理类
    负责将用户问题拆解成多个回答维度的短词组，并校验拆解的视角结构。
    """
    
    TEMPLATE_STR = """<role>
问题角度规划者（Facet Planner）
</role>
<task>
用户会给出一个 query 。你需要为这个 query 规划“回答的侧重点（facet）”，用于分发给下游多个 agent 进行角色扮演式的完整回答。
</task>
<rule>
- 你只输出“角度（facet）”，不输出真实答案、不展开解释。
- facet 不是把问题拆成多个子问题，而是“同一个问题的不同叙事重心/回答框架”。
- 每一个 facet 都必须能引导出一篇针对该 query 的完整回答（而不是片面补充）。
- 允许：名词或短词组（2-10个字符最佳）。
- 禁止：长句、反问句、包含疑问词（如“如何”、“怎么”、“为什么”）。
- 示例：✅"临床安全", ✅"作用机制" ❌"从临床安全角度怎么看", ❌"如何起作用"
- facet 之间要尽量互相区分，避免同义重复。
- 数量由你根据 query 复杂度决定：一般 1-8 个。
</rule>
<format>
- 只输出一个 JSON 数组（不要额外文字、不要代码块标注、不要解释）。
- 数组元素为字符串，每个字符串就是一个 facet。
- 示例格式：["临床安全", "用药禁忌", "应用落地"]
</format>
<strategy>
- 必须要能形成完整叙述的“回答框架”，例如：技术实现、应用落地、决策建议、对比评估、风险治理、方法论抽象、产品化视角、成本收益等。
- 如果 query 很简单：给 1-2 个 facet。
- 如果 query 很复杂：给 3-8 个 facet。
- 不要提出反问或澄清问题，直接给 facets。
</strategy>
## 输入
query：{{ query }}
## 输出
- 只输出 JSON 数组
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, query: str) -> str:
        """
        渲染 Prompt
        
        Args:
            query (str): 用户原始问题
            
        Returns:
            str: 组装后的完整 Prompt 文本
        """
        if not query or not query.strip():
            logger.warning("传入的 query 为空")
            
        return self.template.render(query=query)

    def parse_response(self, response: str) -> List[str]:
        """
        验证和解析模型的 JSON 数组输出
        
        Args:
            response (str): 大模型返回的纯文本结果
            
        Returns:
            List[str]: 解析并验证通过的 facet 列表
            
        Raises:
            ValueError: 验证未通过或解析失败
        """
        text = response.strip()
        
        # 1. 使用正则贪婪提取JSON数组结构
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError(
                "解析失败：未能在模型输出中检测到有效的边界符 '[' 和 ']'。请检查大模型是否违规输出文字。"
            )
            
        json_str = match.group(0)

        # 2. 尝试 JSON 反序列化
        try:
            facets = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析失败：不是合法的 JSON 数组结构。内部错误: {str(e)}")

        # 3. 数据类型验证
        if not isinstance(facets, list):
            raise ValueError(f"解析失败：预期输出为 JSON 数组(list)，但收到了 {type(facets).__name__}")

        # 4. 数量验证 (1-8个)
        if len(facets) < 1:
            raise ValueError("校验失败：生成的 facet 数量不能为 0，至少需要 1 个提取角度。")
        if len(facets) > 8:
            raise ValueError(f"校验失败：生成的 facet 数量({len(facets)})超出了设定的上限 8 个。")

        valid_facets = []
        seen_facets = set()

        for i, f in enumerate(facets):
            if not isinstance(f, str):
                raise ValueError(f"校验失败：数组第 {i} 个元素不是字符串格式。")
            
            clean_f = f.strip()
            
            # 5. 字符长度限制 (2-10字)
            if len(clean_f) < 2 or len(clean_f) > 10:
                raise ValueError(
                    f"校验失败：角度 '{clean_f}' 长度为 {len(clean_f)}。必须是短词组(2-10个字符)。"
                )
            
            # 6. 不允许出现疑问句/长句特征
            forbidden_words = ["如何", "怎么", "为何", "什么", "吗", "呢", "？", "?"]
            if any(word in clean_f for word in forbidden_words):
                raise ValueError(f"校验失败：角度 '{clean_f}' 包含了疑问词或类似子问题的句式，必须是结构框架名词。")

            # 7. 重复筛选
            if clean_f.lower() in seen_facets:
                raise ValueError(f"校验失败：发现重复的角度 '{clean_f}'。")
                
            seen_facets.add(clean_f.lower())
            valid_facets.append(clean_f)

        logger.debug(f"FacetPlanner 解析通过，成功提取 {len(valid_facets)} 个观察维度")
        return valid_facets
