from src.validators.evidence_minimal import (
    extract_evidence_tags,
    validate_evidence_sources,
    auto_append_evidence,
    EvidenceValidator,
)
from src.utils.evidence_formatter import format_paragraph_with_source, format_final_answer


def test_extract_evidence_tags():
    text = "证据 [证据R1:来源=refs] 与 [证据G2:来源=图谱]"
    tags = extract_evidence_tags(text)
    assert tags[0]["type"] == "R"
    assert tags[0]["num"] == "1"
    assert tags[1]["type"] == "G"


def test_validate_evidence_sources():
    ext = [{"type": "R", "num": "1", "full": "[证据R1:来源=refs]"}]
    refs = [{"id": "R1", "source": "《说明书》", "content": "..."}]
    v = validate_evidence_sources(ext, refs)
    assert v["valid"] is True
    assert v["found"] == ["R1"]


def test_auto_append_evidence_and_validator():
    text = "<think>\n<facet=临床安全>\n问题拆解: ...\n</think>\n说明书提示需谨慎使用。"
    refs = [{"id": "R1", "source": "《说明书》", "content": "..."}]
    fixed = auto_append_evidence(text, refs[0])
    assert "[证据R1" in fixed
    ev = EvidenceValidator().validate(text, refs)
    assert ev["is_valid"] is True


def test_evidence_formatter():
    p = format_paragraph_with_source("正文内容", "《说明书》")
    assert "（来源：《说明书》）" in p
    out = format_final_answer("<facet=临床安全>\n证据清单: [证据R1]", "正文")
    assert out.startswith("<think>")
    assert "</think>" in out
