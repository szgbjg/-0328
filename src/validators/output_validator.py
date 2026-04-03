import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.logger import logger
from src.prompts.facet_qa_agent import FacetQAAgentPrompt


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_output: Optional[Any] = None


def _extract_json_array(raw_output: str) -> tuple[Optional[List[Any]], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    text = (raw_output or "").strip()
    if not text:
        return None, ["输出为空。"], warnings

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None, ["未检测到 JSON 数组边界符 '[' 和 ']'。"], warnings

    json_str = match.group(0)
    if json_str != text:
        warnings.append("检测到额外包裹文本，已自动提取 JSON 数组部分。")

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, [f"JSON 解析失败: {str(e)}"], warnings

    if not isinstance(parsed, list):
        return None, [f"输出类型错误：期望 JSON 数组(list)，实际为 {type(parsed).__name__}。"], warnings

    return parsed, errors, warnings


class QuestionCreatorValidator:
    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        arr, errors, warnings = _extract_json_array(raw_output)
        if errors:
            return ValidationResult(False, errors, warnings)

        cleaned: List[str] = []
        for i, item in enumerate(arr or []):
            if not isinstance(item, str):
                warnings.append(f"第 {i} 项不是字符串，已自动忽略。")
                continue
            q = item.strip()
            if len(q) <= 10:
                errors.append(f"第 {i} 项问题过短(<=10)：{q}")
                continue
            cleaned.append(q)

        if len(cleaned) > 5:
            warnings.append(f"问题数量为 {len(cleaned)}，已自动截断到 5 个。")
            cleaned = cleaned[:5]

        if len(cleaned) == 0 and len(arr or []) > 0:
            errors.append("没有任何通过校验的问题条目。")

        return ValidationResult(len(errors) == 0, errors, warnings, cleaned)


class FacetPlannerValidator:
    _forbidden_words = ["如何", "怎么", "为何", "什么", "吗", "呢", "？", "?"]

    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        arr, errors, warnings = _extract_json_array(raw_output)
        if errors:
            return ValidationResult(False, errors, warnings)

        facets: List[str] = []
        seen = set()
        for i, item in enumerate(arr or []):
            if not isinstance(item, str):
                warnings.append(f"第 {i} 项不是字符串，已自动忽略。")
                continue
            f = item.strip()
            if len(f) < 2 or len(f) > 10:
                errors.append(f"第 {i} 项长度非法(2-10)：{f}")
                continue
            if any(w in f for w in self._forbidden_words):
                errors.append(f"第 {i} 项疑似子问题而非回答框架：{f}")
                continue
            key = f.lower()
            if key in seen:
                warnings.append(f"检测到重复角度 '{f}'，已自动去重。")
                continue
            seen.add(key)
            facets.append(f)

        if len(facets) > 8:
            warnings.append(f"角度数量为 {len(facets)}，已自动截断到 8 个。")
            facets = facets[:8]

        if len(facets) < 1:
            errors.append("有效角度数量不足，至少需要 1 个。")

        return ValidationResult(len(errors) == 0, errors, warnings, facets)


class FacetReducerValidator:
    """最严格：必须精确产出 8 个，且仅从 original_facets 选择。"""

    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        original_facets = context.get("facets") or context.get("original_facets") or []
        if not isinstance(original_facets, list) or len(original_facets) == 0:
            return ValidationResult(False, ["上下文缺失：context['facets'] 或 context['original_facets'] 必须为非空列表。"], [])

        arr, errors, warnings = _extract_json_array(raw_output)
        if errors:
            return ValidationResult(False, errors, warnings)

        source_set = set(original_facets)
        selected: List[str] = []
        seen = set()

        for i, item in enumerate(arr or []):
            if not isinstance(item, str):
                warnings.append(f"第 {i} 项不是字符串，已自动忽略。")
                continue
            f = item.strip()
            if f not in source_set:
                warnings.append(f"检测到新增/改写角度 '{f}'，已自动过滤。")
                continue
            if f in seen:
                warnings.append(f"检测到重复角度 '{f}'，已自动去重。")
                continue
            seen.add(f)
            selected.append(f)
            if len(selected) == 8:
                break

        if len(selected) < 8:
            warnings.append(f"输出有效角度仅 {len(selected)} 个，已从原始列表按顺序自动补齐到 8 个。")
            for f in original_facets:
                if f not in seen:
                    selected.append(f)
                    seen.add(f)
                if len(selected) == 8:
                    break

        if len(selected) > 8:
            warnings.append("输出超过 8 个，已自动截断。")
            selected = selected[:8]

        # 严格保障：原始输入不足 8 时直接 invalid
        if len(selected) != 8:
            errors.append(
                f"无法修正为严格 8 个角度：当前为 {len(selected)}，请检查 original_facets 是否至少提供 8 个可用项。"
            )

        return ValidationResult(len(errors) == 0, errors, warnings, selected)


class FacetExpanderValidator:
    _forbidden_words = ["如何", "怎么", "为何", "什么", "吗", "呢", "？", "?"]

    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        existing_facets = context.get("existing_facets") or []
        if not isinstance(existing_facets, list):
            return ValidationResult(False, ["上下文错误：context['existing_facets'] 必须是列表。"], [])

        arr, errors, warnings = _extract_json_array(raw_output)
        if errors:
            return ValidationResult(False, errors, warnings)

        existing = set(str(x).strip() for x in existing_facets)
        seen = set(existing)
        new_facets: List[str] = []

        for i, item in enumerate(arr or []):
            if not isinstance(item, str):
                warnings.append(f"第 {i} 项不是字符串，已自动忽略。")
                continue
            f = item.strip()
            if f in seen:
                warnings.append(f"第 {i} 项与 existing 或本次输出重复('{f}')，已自动过滤。")
                continue
            if len(f) < 2 or len(f) > 10:
                errors.append(f"第 {i} 项长度非法(2-10)：{f}")
                continue
            if any(w in f for w in self._forbidden_words):
                errors.append(f"第 {i} 项包含疑问词，不是框架角度：{f}")
                continue
            seen.add(f)
            new_facets.append(f)
            if len(new_facets) == 6:
                warnings.append("新增角度超过 6 个，已自动截断。")
                break

        return ValidationResult(len(errors) == 0, errors, warnings, new_facets)


class FacetQAAgentValidator:
    """最复杂验证器，完整继承并增强 FacetQAAgentPrompt 的规则。"""

    def __init__(self):
        self._prompt = FacetQAAgentPrompt()

    def _build_fallback_output(self, raw_output: str, facet: str, refs: List[Dict]) -> str:
        body = (raw_output or "").strip()
        if body.startswith("[正文回答]"):
            body = body[len("[正文回答]") :].strip()
        if not body:
            return ""

        evidence_ids: List[str] = []
        for idx, _ in enumerate(refs, start=1):
            evidence_ids.append(f"R{idx}")
            if len(evidence_ids) == 3:
                break

        if not evidence_ids:
            evidence_lines = "- [证据G1:来源=graph_context]"
            reasoning_lines = "- 由G1可得当前结论需要图谱上下文进一步支撑。"
            summary = "当前结论缺少 refs 支撑，需补充结构化证据。"
        else:
            evidence_lines = "\n".join([f"- [证据{eid}:来源=refs]" for eid in evidence_ids])
            reasoning_lines = "\n".join([f"- 由{eid}可得正文关键结论。" for eid in evidence_ids])
            summary = f"综合 {'、'.join(evidence_ids)}，可支持当前正文回答。"

        return (
            f"<think>\n"
            f"<facet={facet}>\n"
            f"问题拆解: 先识别该 facet 的核心风险点与适用边界。\n"
            f"证据清单:\n{evidence_lines}\n"
            f"推理链:\n{reasoning_lines}\n"
            f"冲突识别: 未发现明显冲突。\n"
            f"最终结论摘要: {summary}\n"
            f"</think>\n\n"
            f"{body}"
        )

    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        facet = context.get("facet")
        refs = context.get("refs", [])
        graph_context = context.get("graph_context")

        if not facet:
            return ValidationResult(False, ["上下文缺失：context['facet'] 不能为空。"], [])
        if not isinstance(refs, list):
            return ValidationResult(False, ["上下文错误：context['refs'] 必须是列表。"], [])

        # 小错误自动修正：去掉 Markdown 围栏
        corrected_raw = (raw_output or "").strip()
        if corrected_raw.startswith("```") and corrected_raw.endswith("```"):
            corrected_raw = re.sub(r"^```[a-zA-Z]*\n?", "", corrected_raw)
            corrected_raw = re.sub(r"\n?```$", "", corrected_raw).strip()

        core_result = self._prompt.validate(
            corrected_raw,
            facet=facet,
            refs=refs,
            graph_context=graph_context,
        )

        fallback_used = False
        if (not core_result.is_valid) and ("<think>" not in corrected_raw):
            fallback_raw = self._build_fallback_output(corrected_raw, facet=facet, refs=refs)
            if fallback_raw:
                retry_result = self._prompt.validate(
                    fallback_raw,
                    facet=facet,
                    refs=refs,
                    graph_context=graph_context,
                )
                if retry_result.is_valid:
                    corrected_raw = fallback_raw
                    core_result = retry_result
                    fallback_used = True

        errors = list(core_result.errors)
        warnings: List[str] = []
        if fallback_used:
            warnings.append("检测到缺少<think>结构，已自动补齐最小推理框架。")

        # 额外调试信息增强
        if not corrected_raw:
            errors.append("原始输出为空。")

        corrected_output: Optional[Dict[str, Any]] = None
        if core_result.is_valid:
            parsed = self._prompt.parse_response(corrected_raw)
            corrected_output = parsed
        else:
            # 若禁止词命中，明确标记为严重错误
            if core_result.forbidden_hits:
                errors.append("严重违规：命中禁止词汇，必须重试生成。")

        return ValidationResult(
            is_valid=core_result.is_valid,
            errors=errors,
            warnings=warnings,
            corrected_output=corrected_output,
        )


class RedundancyDetectorValidator:
    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        planners = context.get("planners") or []
        if not isinstance(planners, list):
            return ValidationResult(False, ["上下文错误：context['planners'] 必须是列表。"], [])

        arr, errors, warnings = _extract_json_array(raw_output)
        if errors:
            return ValidationResult(False, errors, warnings)

        max_index = len(planners) - 1
        idxs: List[int] = []
        seen = set()

        for i, item in enumerate(arr or []):
            if not isinstance(item, int) or isinstance(item, bool):
                errors.append(f"第 {i} 项不是整数下标：{item}")
                continue
            if item < 0 or item > max_index:
                errors.append(f"第 {i} 项下标越界：{item}，有效范围是 [0, {max_index}]。")
                continue
            if item in seen:
                warnings.append(f"第 {i} 项下标重复({item})，已自动去重。")
                continue
            seen.add(item)
            idxs.append(item)

        return ValidationResult(len(errors) == 0, errors, warnings, idxs)


class SynthesisAgentValidator:
    REQUIRED_SECTIONS = [
        "结论概览",
        "核心依据整合",
        "完整展开说明",
        "风险与边界条件",
        "实务/操作建议",
        "不确定性说明",
    ]

    def validate(self, raw_output: str, context: Dict) -> ValidationResult:
        text = (raw_output or "").strip()
        errors: List[str] = []
        warnings: List[str] = []

        if not text:
            return ValidationResult(False, ["输出为空。"], warnings)

        # 自动修正：删除 Markdown 围栏
        if text.startswith("```") and text.endswith("```"):
            warnings.append("检测到 Markdown 围栏，已自动去除。")
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        # 规则1：不能是 JSON 数组
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                errors.append("输出类型错误：不应为 JSON 数组，必须为单一整合文本。")
        except json.JSONDecodeError:
            pass

        # 规则2：6个标准部分必须存在
        missing = [s for s in self.REQUIRED_SECTIONS if s not in text]
        if missing:
            errors.append("缺少标准部分: " + "、".join(missing))

        # 规则3：不含英文
        if re.search(r"[a-zA-Z]", text):
            errors.append("检测到英文字符，输出必须为纯中文。")

        return ValidationResult(len(errors) == 0, errors, warnings, text if len(errors) == 0 else None)


__all__ = [
    "ValidationResult",
    "QuestionCreatorValidator",
    "FacetPlannerValidator",
    "FacetReducerValidator",
    "FacetExpanderValidator",
    "FacetQAAgentValidator",
    "RedundancyDetectorValidator",
    "SynthesisAgentValidator",
]
