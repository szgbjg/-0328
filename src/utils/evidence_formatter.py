import re


def format_paragraph_with_source(paragraph: str, source: str) -> str:
    p = (paragraph or "").strip()
    if not p:
        return p
    if "（来源：" in p:
        return p
    return f"{p}（来源：{source}）"


def format_final_answer(thinking: str, body: str) -> str:
    t = (thinking or "").strip()
    b = (body or "").strip()
    if "<facet=" not in t:
        t = "<facet=未指定>\n" + t
    if not t.startswith("<think>"):
        t = "<think>\n" + t
    if not t.endswith("</think>"):
        t = t + "\n</think>"
    t = re.sub(r"<think>\s*<think>", "<think>", t)
    return f"{t}\n\n{b}".strip()
