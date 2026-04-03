from pathlib import Path
from typing import List


_BANNED_WORDS_CACHE: List[str] = []


def _load_banned_words() -> List[str]:
    global _BANNED_WORDS_CACHE
    if _BANNED_WORDS_CACHE:
        return _BANNED_WORDS_CACHE

    file_path = Path(__file__).with_name("banned_words.txt")
    words: List[str] = []
    if file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)

    _BANNED_WORDS_CACHE = words
    return _BANNED_WORDS_CACHE


def check_banned_words(text: str) -> List[str]:
    output = text or ""
    low = output.lower()
    hits: List[str] = []
    for w in _load_banned_words():
        if w.lower() in low:
            hits.append(w)
    return hits


def sanitize_output(raw_output: str, banned_words: List[str]) -> str:
    text = raw_output or ""
    if not banned_words:
        return text

    replacements = {
        "我查了一下": "资料显示",
        "根据检索结果": "根据相关资料",
        "系统显示": "相关记录表明",
    }

    fixed = text
    for src, dst in replacements.items():
        fixed = fixed.replace(src, dst)

    if check_banned_words(fixed):
        return f"{text}\n[内容安全告警: 未完全修正禁用词]"
    return fixed
