import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.logger import logger
from src.utils.evidence_formatter import format_paragraph_with_source, format_final_answer
from src.validators.content_guard import check_banned_words, sanitize_output
from src.validators.evidence_minimal import EvidenceValidator


def build_prompt(query: str, facet: str, refs: List[Dict[str, Any]]) -> str:
    ref_lines = []
    for idx, ref in enumerate(refs, start=1):
        source = ref.get("source", "")
        content = ref.get("content", "")
        ref_lines.append(f"R{idx} | {source} | {content}")
    refs_text = "\n".join(ref_lines)
    return (
        f"请基于以下信息回答。\n"
        f"query: {query}\n"
        f"facet: {facet}\n"
        f"refs:\n{refs_text}\n"
        f"输出格式：<think>...<facet={facet}>...证据清单...</think> + 正文"
    )


def extract_thinking(raw_output: str) -> str:
    m = re.search(r"<think>.*?</think>", raw_output or "", re.DOTALL)
    return m.group(0).strip() if m else ""


def extract_body(raw_output: str) -> str:
    thinking = extract_thinking(raw_output)
    if not thinking:
        return (raw_output or "").strip()
    return (raw_output or "").replace(thinking, "", 1).strip()


class FacetQAAgent:
    def __init__(self):
        self.evidence_validator = EvidenceValidator()

    async def generate_facet_answer(
        self,
        query: str,
        facet: str,
        refs: List[Dict[str, Any]],
        model_call: Callable[..., Awaitable[str]],
    ) -> Optional[Dict[str, Any]]:
        prompt = build_prompt(query, facet, refs)
        raw_output = await model_call(prompt=prompt, temperature=0.7)

        banned = check_banned_words(raw_output)
        if banned:
            logger.warning(f"检测到禁用词: {banned}")
            fixed = sanitize_output(raw_output, banned)
            if check_banned_words(fixed):
                raw_output = await model_call(prompt=prompt, temperature=0.3)
                banned = check_banned_words(raw_output)
                if banned:
                    logger.error(f"重试后仍有禁用词，丢弃facet: {facet}")
                    return None
            else:
                raw_output = fixed

        result = self.evidence_validator.validate(raw_output, refs)
        if not result["is_valid"]:
            if result.get("fixed_text"):
                raw_output = result["fixed_text"]
                result = self.evidence_validator.validate(raw_output, refs)
            if not result["is_valid"]:
                logger.error(f"证据链无法修复，丢弃facet: {facet}")
                return None

        thinking = extract_thinking(raw_output)
        body = extract_body(raw_output)
        for ref in refs:
            rid = str(ref.get("id", "")).upper()
            if rid and rid in result.get("found", []):
                body = format_paragraph_with_source(body, ref.get("source", "未知来源"))

        formatted = format_final_answer(thinking, body)
        return {
            "thinking": extract_thinking(formatted),
            "body": extract_body(formatted),
            "evidence_count": len(result.get("found", [])),
            "raw_output": formatted,
        }
