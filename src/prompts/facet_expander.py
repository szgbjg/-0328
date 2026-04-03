import json
import re
from typing import List
from jinja2 import Template

from src.prompts.base_prompt import BasePrompt
from src.logger import logger


class FacetExpanderPrompt(BasePrompt):
    """
    问题角度扩展者 (Facet Expander) Prompt 管理类
    负责在已有角度的基础上，补充 0-6 个全新的角度。
    """
    
    TEMPLATE_STR = """<role>
问题角度扩展者（Facet Expander）
</role>
<task>
用户给出了一个 query 以及一组目前已经构思好的“已有角度（existing_facets）”。
你需要发散思维，为这个 query 补充 0 到 6 个**全新**的回答维度/框架视角。
</task>
<rule>
- 你只输出“新角度（facet）”，不输出真实答案、不展开解释。
- 补充的新角度**绝对不能**与“已有角度”重复。
- 允许：名词或短词组（2-10个字符最佳）。
- 禁止：长句、反问句、包含疑问词（如“如何”、“怎么”）。
- 数量要求：0 到 6 个。如果你认为已有的角度已经很完善，涵盖了最关键的信息，不需要补充，可以直接输出空数组 []。
</rule>
<format>
- 只输出一个 JSON 数组（不要额外文字、不要代码块标注、不要解释）。
- 数组元素为字符串，每个字符串就是一个 facet。
- 示例格式：["成本差异", "长期预后"]，或者 []
</format>
<strategy>
- 仔细阅读已有角度，找出未被覆盖的长尾重点。
- 保证新角度和已有角度处于同一层次的概括抽象度。
</strategy>
## 输入
query：{{ query }}
已有角度（existing_facets）：
{{ existing_facets_json }}

## 输出
- 只输出 JSON 数组
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, query: str, existing_facets: List[str]) -> str:
        """
        渲染 Prompt
        
        Args:
            query (str): 用户原始问题
            existing_facets (List[str]): 已经存在的 facet 列表
            
        Returns:
            str: 组装后的完整 Prompt 文本
        """
        existing_facets_json = json.dumps(existing_facets, ensure_ascii=False, indent=2)
        return self.template.render(query=query, existing_facets_json=existing_facets_json)

    def parse_response(self, response: str, existing_facets: List[str]) -> List[str]:
        """
        验证和解析模型输出的 JSON 数组，包含防止重复的自动过滤。
        
        Args:
            response (str): 大模型返回的纯文本结果
            existing_facets (List[str]): 已存在的 facet 列表
            
        Returns:
            List[str]: 校验后的**新增** facet 列表 (0-6个)
            
        Raises:
            ValueError: 验证未通过或解析失败（格式严重违规时）
        """
        text = response.strip()
        
        # 1. 贪婪提取边界
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError("解析失败：未能在模型输出中检测到有效的边界符 '[' 和 ']'。")
            
        json_str = match.group(0)

        # 2. 尝试 JSON 反序列化
        try:
            new_facets = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析失败：不是合法的 JSON 数组结构。内部错误: {str(e)}")

        # 3. 数据类型验证
        if not isinstance(new_facets, list):
            raise ValueError(f"解析失败：预期输出为 JSON 数组(list)，但收到了 {type(new_facets).__name__}")

        valid_new_facets = []
        # 初始化哈希集合时，将 existing_facets 加入作为过滤基准
        seen = set(f.strip() for f in existing_facets)
        forbidden_words = ["如何", "怎么", "为何", "什么", "吗", "呢", "？", "?"]

        for i, f in enumerate(new_facets):
            if not isinstance(f, str):
                raise ValueError(f"校验失败：数组第 {i} 个元素不是字符串格式。")
                
            clean_f = f.strip()
            
            # 自动过滤逻辑 1：和 existing_facets 字符串完全匹配，或在本次生成中重复
            if clean_f in seen:
                logger.debug(f"FacetExpander 自动过滤重复的角度: '{clean_f}'")
                continue
                
            # 格式硬校验 1：字数限制
            if len(clean_f) < 2 or len(clean_f) > 10:
                raise ValueError(
                    f"校验失败：新增角度 '{clean_f}' 长度为 {len(clean_f)}。必须是短词组(2-10个字符)。"
                )
                
            # 格式硬校验 2：疑问句限制
            if any(word in clean_f for word in forbidden_words):
                raise ValueError(f"校验失败：新增角度 '{clean_f}' 包含了疑问词，必须是结构名词。")

            seen.add(clean_f)
            valid_new_facets.append(clean_f)
            
            # 自动过滤逻辑 2：仅允许最多截取 6 个
            if len(valid_new_facets) == 6:
                logger.debug("FacetExpander 生成的角度数量已达到 6 个上限，截取并丢弃后续部分。")
                break

        logger.debug(f"FacetExpander 验证通过，成功新增 {len(valid_new_facets)} 个维度")
        return valid_new_facets
