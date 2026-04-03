import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jinja2 import Template

from src.logger import logger
from src.prompts.base_prompt import BasePrompt


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    forbidden_hits: List[str] = field(default_factory=list)


class FacetQAAgentPrompt(BasePrompt):
    """
    Facet 维度回答生成器，负责:
    1) 渲染带证据上下文的提示词
    2) 解析 think 与正文
    3) 对输出做格式/安全/证据链完整性校验
    """

    FORBIDDEN_TERMS = [
        "检索",
        "查阅",
        "搜索",
        "工具返回",
        "调用接口",
        "从图谱里查到",
        "我查了一下",
        "系统显示",
        "根据检索结果",
        "retrieved",
        "searched",
        "looked up",
        "tool says",
    ]

    REQUIRED_BLOCKS = [
        "问题拆解",
        "证据清单",
        "推理链",
        "冲突识别",
        "最终结论摘要",
    ]

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    FACET_PATTERN = re.compile(r"<facet=([^>\n]+)>", re.DOTALL)
    EVIDENCE_ITEM_PATTERN = re.compile(r"\[证据([RG]\d+)\s*:[^\]]*\]")
    EVIDENCE_REF_PATTERN = re.compile(r"([RG]\d+)")

    TEMPLATE_STR = """<role>
FacetGraph-QA Agent
</role>
<task>
针对给定 query 与指定 facet，结合 refs 与 graph_context 产出该 facet 下的高质量回答。
</task>
<output_contract>
你必须严格输出如下结构:
<think>
<facet={{ facet }}>
问题拆解: ...
证据清单: ...
推理链: ...
冲突识别: ...
最终结论摘要: ...
</think>

[正文回答]
</output_contract>
<safety>
禁止出现以下词汇或同义表达:
检索、查阅、搜索、工具返回、调用接口、从图谱里查到、我查了一下、系统显示、根据检索结果、retrieved、searched、looked up、tool says
</safety>
<context>
query: {{ query }}
facet: {{ facet }}
refs:
{% for ref in refs %}
- R{{ loop.index }} | source={{ ref.source if ref.source is defined else "" }} | location={{ ref.location if ref.location is defined else "" }}
  content={{ ref.content if ref.content is defined else "" }}
{% endfor %}
{% if graph_context %}
graph_context:
{{ graph_context }}
{% endif %}
</context>
"""

    def __init__(self):
        self.template = Template(self.TEMPLATE_STR)

    def render(
        self,
        query: str,
        facet: str,
        refs: List[Dict],
        graph_context: Optional[str] = None,
    ) -> str:
        return self.template.render(
            query=query,
            facet=facet,
            refs=refs,
            graph_context=graph_context,
        )

    def parse_thinking(self, response: str) -> Dict[str, str]:
        think_match = self.THINK_PATTERN.search(response)
        if not think_match:
            raise ValueError("格式校验失败：缺少<think>...</think>块。")

        think_body = think_match.group(1).strip()
        facet_match = self.FACET_PATTERN.search(think_body)
        if not facet_match:
            raise ValueError("格式校验失败：think 块内缺少<facet=XXX>标签。")

        facet_value = facet_match.group(1).strip()
        body_wo_facet = self.FACET_PATTERN.sub("", think_body, count=1).strip()

        sections: Dict[str, str] = {
            "facet": facet_value,
            "raw": think_body,
        }

        # 按锚点切分必填段落，确保结构化提取
        label_pattern = re.compile(
            r"(问题拆解|证据清单|推理链|冲突识别|最终结论摘要)\s*:\s*",
            re.DOTALL,
        )
        matches = list(label_pattern.finditer(body_wo_facet))
        if not matches:
            raise ValueError("格式校验失败：think 块内未检测到任何必填段落标签。")

        for idx, m in enumerate(matches):
            label = m.group(1)
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body_wo_facet)
            sections[label] = body_wo_facet[start:end].strip()

        missing = [lb for lb in self.REQUIRED_BLOCKS if lb not in sections or not sections[lb]]
        if missing:
            raise ValueError(f"格式校验失败：think 块缺少必填内容: {', '.join(missing)}")

        return sections

    def parse_answer(self, response: str) -> str:
        think_match = self.THINK_PATTERN.search(response)
        if not think_match:
            raise ValueError("格式校验失败：缺少<think>...</think>块，无法提取正文。")

        answer = response[think_match.end() :].strip()
        if not answer:
            raise ValueError("格式校验失败：think 块后缺少正文回答。")
        return answer

    def _detect_forbidden_terms(self, response: str) -> List[str]:
        text = response.lower()
        hits = []
        for term in self.FORBIDDEN_TERMS:
            if term.lower() in text:
                hits.append(term)
        return hits

    def _validate_evidence_chain(
        self,
        sections: Dict[str, str],
        refs: List[Dict],
        graph_context: Optional[str],
    ) -> List[str]:
        errors: List[str] = []

        evidence_text = sections.get("证据清单", "")
        reasoning_text = sections.get("推理链", "")

        evidence_items = self.EVIDENCE_ITEM_PATTERN.findall(evidence_text)
        if not evidence_items:
            errors.append("证据链校验失败：证据清单缺少合法证据格式，需为[证据R1:...]或[证据G1:...]。")
            return errors

        normalized_ids = [eid.upper() for eid in evidence_items]

        # 证据必须在 refs 或 graph_context 中可对应
        for eid in normalized_ids:
            prefix = eid[0]
            num = int(eid[1:])
            if prefix == "R":
                if num < 1 or num > len(refs):
                    errors.append(f"证据链校验失败：{eid} 在 refs 中无对应条目。")
            else:
                if not graph_context or not graph_context.strip():
                    errors.append(f"证据链校验失败：{eid} 需要 graph_context 支撑，但输入为空。")

        # 推理链必须引用证据编号
        referenced_ids = {x.upper() for x in self.EVIDENCE_REF_PATTERN.findall(reasoning_text)}
        if not referenced_ids:
            errors.append("证据链校验失败：推理链未引用任何证据编号(Rn/Gn)。")
            return errors

        evidence_id_set = set(normalized_ids)
        if referenced_ids.isdisjoint(evidence_id_set):
            errors.append("证据链校验失败：推理链引用的证据编号与证据清单不一致。")

        unknown_refs = [rid for rid in referenced_ids if rid not in evidence_id_set]
        if unknown_refs:
            errors.append("证据链校验失败：推理链包含未在证据清单声明的编号: " + ", ".join(sorted(unknown_refs)))

        return errors

    def validate(
        self,
        response: str,
        facet: str,
        refs: List[Dict],
        graph_context: Optional[str] = None,
    ) -> ValidationResult:
        errors: List[str] = []

        # A. 格式级硬校验
        if "<think>" not in response:
            errors.append("格式校验失败：缺少<think>起始标签。")
        if "</think>" not in response:
            errors.append("格式校验失败：缺少</think>结束标签。")

        sections: Dict[str, str] = {}
        if not errors:
            try:
                sections = self.parse_thinking(response)
            except ValueError as e:
                errors.append(str(e))

            if sections:
                if sections.get("facet") != facet:
                    errors.append(
                        f"格式校验失败：think 中 facet 属性不匹配，期望 '{facet}'，实际 '{sections.get('facet')}'。"
                    )

            try:
                _ = self.parse_answer(response)
            except ValueError as e:
                errors.append(str(e))

        # B. 安全词校验
        forbidden_hits = self._detect_forbidden_terms(response)
        if forbidden_hits:
            errors.append("安全校验失败：检测到禁止词汇: " + ", ".join(forbidden_hits))

        # C. 证据链完整性
        if sections:
            errors.extend(self._validate_evidence_chain(sections, refs, graph_context))

        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            forbidden_hits=forbidden_hits,
        )

        if not result.is_valid:
            logger.warning("FacetQAAgent 校验失败: {}", " | ".join(result.errors))
        else:
            logger.debug("FacetQAAgent 校验通过")

        return result

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        为兼容 BasePrompt 接口，返回结构化解析结果。
        """
        return {
            "thinking": self.parse_thinking(response),
            "answer": self.parse_answer(response),
        }
