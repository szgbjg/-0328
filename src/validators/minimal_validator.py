import json
import re


def validate_json_array(output: str) -> list | None:
    text = (output or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def check_think_tags(output: str, expected_facet: str) -> bool:
    text = output or ""
    if "<think>" not in text or "</think>" not in text:
        return False
    m = re.search(r"<facet=([^>\n]+)>", text)
    if not m:
        return False
    return m.group(1).strip() == (expected_facet or "").strip()


def check_banned_words(output: str, banned_words: list) -> list:
    text = (output or "").lower()
    hits = []
    for w in banned_words or []:
        if str(w).lower() in text:
            hits.append(w)
    return hits


def check_evidence_chain(output: str) -> bool:
    text = output or ""
    return bool(re.search(r"\[证据[RG]\d*", text))


class SimpleValidator:
    def validate(
        self,
        output: str,
        expected_facet: str = "",
        banned_words: list | None = None,
        require_json_array: bool = False,
        require_think_tags: bool = False,
        require_evidence_chain: bool = False,
    ) -> dict:
        errors = []

        if require_json_array and validate_json_array(output) is None:
            errors.append("JSON数组校验失败")

        if require_think_tags and not check_think_tags(output, expected_facet):
            errors.append("think/facet标签校验失败")

        hits = check_banned_words(output, banned_words or [])
        if hits:
            errors.append("检测到禁用词: " + ", ".join(hits))

        if require_evidence_chain and not check_evidence_chain(output):
            errors.append("证据链校验失败")

        return {"is_valid": len(errors) == 0, "errors": errors}
