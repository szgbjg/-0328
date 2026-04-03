import re
from typing import Dict, List


def extract_evidence_refs(text: str) -> List[str]:
    raw = text or ""
    refs = re.findall(r"\[证据([RG]\d*)", raw)
    seen = set()
    out: List[str] = []
    for r in refs:
        ref = r.upper()
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def validate_evidence_chain(text: str, available_refs: List[str]) -> Dict:
    refs = extract_evidence_refs(text)
    available = {str(x).upper() for x in (available_refs or [])}
    missing = [r for r in refs if r not in available]
    return {
        "valid": len(refs) > 0 and len(missing) == 0,
        "missing": missing,
        "count": len(refs),
    }


def auto_fix_evidence(text: str, refs: List[str]) -> str:
    raw = text or ""
    result = raw
    existing = set(extract_evidence_refs(raw))

    if "说明书" in raw and "R1" not in existing:
        result += " [证据R1]"
    if "指南" in raw and "R2" not in existing:
        result += " [证据R2]"

    return result
