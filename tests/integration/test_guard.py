from pathlib import Path

from src.validators.content_guard import check_banned_words, sanitize_output
from src.validators.evidence_tracker import extract_evidence_refs, auto_fix_evidence


def _read_bad_example() -> str:
    p = Path("tests/fixtures/bad_example.txt")
    return p.read_text(encoding="utf-8")


def test_detect_three_banned_words_from_fixture():
    text = _read_bad_example()
    banned = check_banned_words(text)

    assert "我查了一下" in banned
    assert "根据检索结果" in banned
    assert "我调研了" in banned
    assert len(banned) >= 3


def test_sanitize_replaces_wo_cha_le_yi_xia():
    text = "我查了一下说明书，结论较明确。"
    fixed = sanitize_output(text, check_banned_words(text))

    assert "我查了一下" not in fixed
    assert "资料显示" in fixed


def test_evidence_autofix_when_missing_refs():
    text = "说明书提示该药禁忌症较多。"

    assert extract_evidence_refs(text) == []

    fixed = auto_fix_evidence(text, ["R1", "R2"])
    refs = extract_evidence_refs(fixed)

    assert "R1" in refs
