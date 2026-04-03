from src.validators.content_guard import check_banned_words, sanitize_output
from src.validators.evidence_tracker import (
    auto_fix_evidence,
    extract_evidence_refs,
    validate_evidence_chain,
)


def test_check_banned_words_case_insensitive():
    text = "我查了一下，并根据检索结果给出结论"
    hits = check_banned_words(text)
    assert "我查了一下" in hits
    assert "根据检索结果" in hits


def test_sanitize_output_replace_known_phrases():
    text = "我查了一下，系统显示需要注意风险，并根据检索结果调整。"
    fixed = sanitize_output(text, check_banned_words(text))
    assert "资料显示" in fixed
    assert "相关记录表明" in fixed
    assert "根据相关资料" in fixed


def test_extract_evidence_refs():
    text = "证据: [证据R1:来源=refs] 和 [证据G2:来源=图谱]"
    refs = extract_evidence_refs(text)
    assert refs == ["R1", "G2"]


def test_validate_evidence_chain_missing_refs():
    text = "证据: [证据R1] [证据R3]"
    result = validate_evidence_chain(text, ["R1", "R2"])
    assert result["count"] == 2
    assert result["missing"] == ["R3"]
    assert result["valid"] is False


def test_auto_fix_evidence():
    text = "说明书提示该药存在禁忌，指南建议加强随访。"
    fixed = auto_fix_evidence(text, ["R1", "R2"])
    assert "[证据R1]" in fixed
    assert "[证据R2]" in fixed
