from src.validators.evidence_minimal import EvidenceValidator


def test_normal_text_with_r1_pass():
    validator = EvidenceValidator()
    text = """
<think>
<facet=测试>
问题拆解: S1
证据清单: [证据R1:来源=refs:《说明书》]
推理链: P1 -> C1
最终结论摘要: 测试通过
</think>
正文：结论内容。
"""
    refs = [{"id": "R1", "source": "《说明书》", "content": "..."}]
    result = validator.validate(text, refs)
    assert result["is_valid"] is True
    assert "R1" in result["found"]


def test_no_evidence_auto_append_then_pass():
    validator = EvidenceValidator()
    text = "正文：结论内容。"
    refs = [{"id": "R1", "source": "《说明书》", "content": "..."}]
    result = validator.validate(text, refs)
    assert result["is_valid"] is True
    assert result["fixed_text"] is not None
    assert "R1" in result["found"]


def test_missing_r9_marked_missing():
    validator = EvidenceValidator()
    text = "证据清单: [证据R9:来源=refs:《未知》]"
    refs = [{"id": "R1", "source": "《说明书》", "content": "..."}]
    result = validator.validate(text, refs)
    assert result["is_valid"] is False
    assert any("R9" in e for e in result["errors"])
