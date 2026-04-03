import json
import re
from typing import List
from jinja2 import Template

from src.prompts.base_prompt import BasePrompt
from src.logger import logger


class FacetReducerPrompt(BasePrompt):
    """
    问题角度精简者 (Facet Reducer) Prompt 管理类
    负责当生成的角度 (facets) 超过 8 个时，强制过滤和精炼到最核心的 8 个角度。
    """
    
    TEMPLATE_STR = """<role>
多角度筛选者（Facet Reducer）
</role>
<task>
用户会给出一个 query 以及针对这个 query 生成的一大批候选“回答角度（facets）”。
受限于系统处理能力，最多只能保留 8 个角度进行并发搜罗。
请基于 query 的核心诉求，从提供的候选视角中，**挑选出最重要、最具有代表性、差异最大的 8 个角度**。
</task>
<rule>
- 原样输出你挑选出来的 8 个角度，**绝对不能**修改原字符串的任何一个字，**不能**合并，只能“选择”。
- **绝对不能**自己凭空编造新的角度，只能从 `候选角度（facets）` 列表中挑选。
- 必须有且仅有 8 个角度。
- 挑选出的角度不能有意义上高度重复的情况（如果你认为两个候选高度重复，只选其中一个）。
</rule>
<format>
- 只输出一个 JSON 数组（不要额外文字、不要代码块标注、不要解释）。
- 数组元素为字符串，每个字符串就是一个 facet。
- 示例格式：["临床安全", "用药禁忌", "应用落地", "作用机制", "副作用", "合成工艺", "市场分析", "替代方案"]
</format>
<strategy>
- 优先保留与用户 query 最契合、能独立成段落解释的重点领域。
- 剔除那些过于长尾、生僻或与其它角度高度重叠的视角。
</strategy>
## 输入
query：{{ query }}
候选角度（facets）：
{{ facets | tojson(indent=2) }}

## 输出
- 只输出包含 8 个字符串的 JSON 数组
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(self, query: str, facets: List[str]) -> str:
        """
        渲染 Prompt
        
        Args:
            query (str): 用户原始问题
            facets (List[str]): 原始超量返回的 facet 列表
            
        Returns:
            str: 组装后的完整 Prompt 文本
        """
        if not facets or len(facets) <= 8:
            logger.warning(f"FacetReducer 接收到的 facets 数量为 {len(facets) if facets else 0}，通常应大于 8")
            
        return self.template.render(query=query, facets=facets)

    def parse_response(self, response: str, original_facets: List[str]) -> List[str]:
        """
        验证和解析模型精简后的 JSON 数组输出，并包含自动修正逻辑
        
        自动修正逻辑：
        1. 发现新增（幻觉）的 facet -> 过滤掉
        2. 发现重复的 facet -> 去重
        3. 总数超过 8 个 -> 截取前 8 个
        4. 总数少于 8 个 -> 从 original_facets 中按顺序补充
        
        Args:
            response (str): 大模型返回的纯文本结果
            original_facets (List[str]): 传入给模型的原始候选 facets 列表
            
        Returns:
            List[str]: 严格保障为 8 个的有效 facet 列表
            
        Raises:
            ValueError: 彻底无法解析（非 JSON，非数组）时抛出
        """
        text = response.strip()
        
        # 1. 使用正则贪婪提取JSON数组结构
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError("解析失败：未能在模型输出中检测到有效的边界符 '[' 和 ']'。")
            
        json_str = match.group(0)

        # 2. 尝试 JSON 反序列化
        try:
            model_facets = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析失败：不是合法的 JSON 数组结构。内部错误: {str(e)}")

        # 3. 数据类型验证
        if not isinstance(model_facets, list):
            raise ValueError(f"解析失败：预期输出为 JSON 数组(list)，但收到了 {type(model_facets).__name__}")

        # 核心验证与自动修正逻辑
        valid_set = set(original_facets)
        processed_facets = []
        seen = set()

        for f in model_facets:
            if not isinstance(f, str):
                continue
            
            clean_f = f.strip()
            
            # 过滤：必须在原文中且不重复
            if clean_f in valid_set and clean_f not in seen:
                processed_facets.append(clean_f)
                seen.add(clean_f)
                
            if len(processed_facets) == 8:
                break
                
        # 自动补全：如果不满 8 个，从 original_facets 中找未使用的补充
        if len(processed_facets) < 8:
            logger.warning(f"模型精简出 {len(processed_facets)} 个有效角度，不足 8 个，正在自动从原始列表补充...")
            for f in original_facets:
                if f not in seen:
                    processed_facets.append(f)
                    seen.add(f)
                if len(processed_facets) == 8:
                    break
                    
        # 极端情况：如果 original_facets 本身就不足 8 个，只能返回实际数量
        # 但按照前置约束，传入的 original_facets 应该是 > 8 的
        
        if len(processed_facets) > 8:
            processed_facets = processed_facets[:8]

        logger.debug(f"FacetReducer 自动修正完毕，成功锁定 {len(processed_facets)} 个观察维度")
        return processed_facets
