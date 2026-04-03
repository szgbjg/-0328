import re


def extract_evidence_tags(text: str) -> list[dict]:
    try:
        out = []
        for m in re.finditer(r"(\[证据([RG])(\d+)[^\]]*\])", text or ""):
            out.append({"type": m.group(2), "num": m.group(3), "full": m.group(1)})
        return out
    except Exception:
        return []


def validate_evidence_sources(extracted: list, available_refs: list) -> dict:
    try:
        ids = {str(x.get("id", "")).upper() for x in (available_refs or [])}
        found, missing = [], []
        for e in extracted or []:
            eid = f"{e['type']}{e['num']}".upper()
            (found if eid in ids else missing).append(eid)
        return {"valid": len(extracted or []) > 0 and len(missing) == 0, "found": found, "missing": missing}
    except Exception:
        return {"valid": False, "found": [], "missing": []}


def auto_append_evidence(text: str, ref: dict) -> str:
    try:
        raw = text or ""
        if not ref:
            return raw

        source = ref.get("source", "未知来源")
        rid = str(ref.get("id", "R1")).upper()
        tag = f"[证据{rid}:来源=refs:{source}]"

        fixed = raw
        if "[证据" not in fixed:
            if "<think>" in fixed and "</think>" in fixed:
                fixed = fixed.replace("</think>", f"\n证据清单: {tag}\n</think>", 1)
            else:
                fixed = f"{tag}\n{fixed}"

        if "（来源：" not in fixed:
            if "</think>" in fixed:
                head, body = fixed.split("</think>", 1)
                body = body.strip()
                if body and "（来源：" not in body:
                    body = f"{body}（来源：{source}）"
                fixed = f"{head}</think>\n{body}"
            else:
                fixed = f"{fixed}（来源：{source}）"
        return fixed
    except Exception:
        return text or ""


class EvidenceValidator:
    def validate(self, text, available_refs):
        try:
            tags = extract_evidence_tags(text)
            if tags:
                v = validate_evidence_sources(tags, available_refs)
                print(f"[EvidenceValidator] 提取到证据: {len(tags)}")
                return {
                    "is_valid": v["valid"],
                    "errors": [] if v["valid"] else [f"缺失证据: {v['missing']}"],
                    "fixed_text": None,
                    "found": v["found"],
                }

            fixed = auto_append_evidence(text, (available_refs or [{}])[0] if available_refs else {})
            tags2 = extract_evidence_tags(fixed)
            if not tags2:
                return {"is_valid": False, "errors": ["无证据引用且无法自动补充"], "fixed_text": fixed, "found": []}

            v2 = validate_evidence_sources(tags2, available_refs)
            print(f"[EvidenceValidator] 自动补充后证据: {len(tags2)}")
            return {
                "is_valid": v2["valid"],
                "errors": [] if v2["valid"] else [f"缺失证据: {v2['missing']}"],
                "fixed_text": fixed,
                "found": v2["found"],
            }
        except Exception:
            return {"is_valid": False, "errors": ["证据验证异常"], "fixed_text": text, "found": []}
