import pytest

from src.prompts.redundancy_detector import RedundancyDetectorPrompt


@pytest.fixture
def detector_prompt():
    return RedundancyDetectorPrompt()


@pytest.fixture
def sample_planners():
    return [
        {"planner": "临床安全", "answer": "强调禁忌与风险人群"},
        {"planner": "风险治理", "answer": "核心结论与临床安全高度重复"},
        {"planner": "应用落地", "answer": "强调执行流程与随访"},
        {"planner": "机制解释", "answer": "强调作用机制"},
    ]


def test_render_contains_query_and_planners(detector_prompt, sample_planners):
    query = "阿司匹林使用注意事项"
    prompt = detector_prompt.render(query, sample_planners)

    assert "阿司匹林使用注意事项" in prompt
    assert '"planner": "临床安全"' in prompt
    assert '"planner": "机制解释"' in prompt


def test_parse_response_valid_indices(detector_prompt, sample_planners):
    response = "[1, 3]"

    result = detector_prompt.parse_response(response, sample_planners)

    assert result == [1, 3]


def test_parse_response_with_markdown_wrapped(detector_prompt, sample_planners):
    response = """
    建议删除如下:
    ```json
    [1, 2]
    ```
    """

    result = detector_prompt.parse_response(response, sample_planners)

    assert result == [1, 2]


def test_parse_response_empty_array(detector_prompt, sample_planners):
    response = "[]"

    result = detector_prompt.parse_response(response, sample_planners)

    assert result == []


def test_parse_response_non_list(detector_prompt, sample_planners):
    response = '{"drop": "1,2"}'

    with pytest.raises(ValueError, match="未能在模型输出中检测到有效的边界符"):
        detector_prompt.parse_response(response, sample_planners)


def test_parse_response_non_integer_item(detector_prompt, sample_planners):
    response = '[1, "2"]'

    with pytest.raises(ValueError, match="不是整数下标"):
        detector_prompt.parse_response(response, sample_planners)


def test_parse_response_out_of_range(detector_prompt, sample_planners):
    response = "[4]"

    with pytest.raises(ValueError, match="超出有效范围"):
        detector_prompt.parse_response(response, sample_planners)


def test_parse_response_negative_index(detector_prompt, sample_planners):
    response = "[-1]"

    with pytest.raises(ValueError, match="超出有效范围"):
        detector_prompt.parse_response(response, sample_planners)


def test_parse_response_deduplicate_indices(detector_prompt, sample_planners):
    response = "[1, 1, 2]"

    result = detector_prompt.parse_response(response, sample_planners)

    assert result == [1, 2]


def test_parse_response_reject_bool_index(detector_prompt, sample_planners):
    response = "[true]"

    with pytest.raises(ValueError, match="不是整数下标"):
        detector_prompt.parse_response(response, sample_planners)
