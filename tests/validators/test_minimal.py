from src.validators.minimal_validator import (
    SimpleValidator,
    check_banned_words,
    check_evidence_chain,
    check_think_tags,
    validate_json_array,
)


def test_validate_json_array_ok():
    assert validate_json_array('["a", "b"]') == ["a", "b"]
    assert validate_json_array('```json\n[1,2]\n```') == [1, 2]


def test_validate_json_array_fail():
    assert validate_json_array('{"a":1}') is None
    assert validate_json_array('abc') is None


def test_check_think_tags():
    ok = "<think>\n<facet=临床安全>\n内容\n</think>"
    bad = "<think>\n<facet=其他>\n内容\n</think>"
    assert check_think_tags(ok, "临床安全") is True
    assert check_think_tags(bad, "临床安全") is False


def test_check_banned_words():
    text = "我查阅后根据检索结果给出结论"
    hits = check_banned_words(text, ["查阅", "根据检索结果", "tool says"])
    assert hits == ["查阅", "根据检索结果"]


def test_check_evidence_chain():
    assert check_evidence_chain("证据清单: [证据R1:来源=refs]") is True
    assert check_evidence_chain("证据清单: [证据G2:来源=图谱]") is True
    assert check_evidence_chain("无证据") is False


def test_simple_validator():
    validator = SimpleValidator()
    output = "<think>\n<facet=临床安全>\n证据:[证据R1]\n</think>"
    result = validator.validate(
        output,
        expected_facet="临床安全",
        banned_words=["检索"],
        require_think_tags=True,
        require_evidence_chain=True,
    )
    assert result["is_valid"] is True
    assert result["errors"] == []


def test_simple_validator_failures():
    validator = SimpleValidator()
    output = "<think>\n<facet=其他>\n没有证据\n</think> 我查阅了一下"
    result = validator.validate(
        output,
        expected_facet="临床安全",
        banned_words=["查阅"],
        require_think_tags=True,
        require_evidence_chain=True,
    )
    assert result["is_valid"] is False
    assert any("标签" in e for e in result["errors"])
    assert any("禁用词" in e for e in result["errors"])
    assert any("证据链" in e for e in result["errors"])
