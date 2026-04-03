import pytest
from src.prompts.facet_planner import FacetPlannerPrompt


@pytest.fixture
def planner_prompt():
    return FacetPlannerPrompt()


def test_render_with_valid_query(planner_prompt):
    query = "头孢与酒精的禁忌分析"
    prompt = planner_prompt.render(query)
    assert "头孢与酒精的禁忌分析" in prompt
    assert "<role>" in prompt
    assert "规划“回答的侧重点（facet）”" in prompt


def test_parse_response_valid_json(planner_prompt):
    response = '["药理作用", "临床副作用", "禁忌人群"]'
    result = planner_prompt.parse_response(response)
    assert len(result) == 3
    assert result == ["药理作用", "临床副作用", "禁忌人群"]


def test_parse_response_with_markdown_wrap(planner_prompt):
    response = '''
    这里是您需要的角度规划：
    ```json
    ["作用机制", "用药禁忌"]
    ```
    '''
    result = planner_prompt.parse_response(response)
    assert len(result) == 2
    assert result == ["作用机制", "用药禁忌"]


def test_parse_response_not_a_list(planner_prompt):
    response = '{"message": "这里没有数组"}'
    with pytest.raises(ValueError, match="未能在模型输出中检测到有效的边界符"):
        planner_prompt.parse_response(response)


def test_parse_response_too_many_facets(planner_prompt):
    response = '["1", "22", "333", "44", "55", "66", "77", "88", "99"]'
    with pytest.raises(ValueError, match="超出了设定的上限 8 个"):
        planner_prompt.parse_response(response)


def test_parse_response_empty_list(planner_prompt):
    response = '[]'
    with pytest.raises(ValueError, match="数量不能为 0"):
        planner_prompt.parse_response(response)


def test_parse_response_invalid_length(planner_prompt):
    # Less than 2 chars
    with pytest.raises(ValueError, match="长度为 1。必须是短词组"):
        planner_prompt.parse_response('["短", "正好"]')
        
    # More than 10 chars
    with pytest.raises(ValueError, match="长度为 12。必须是短词组"):
        planner_prompt.parse_response('["这个角度名字太长了受不了", "恰好十个字啦啦啦"]')


def test_parse_response_with_question_words(planner_prompt):
    response = '["如何用药", "副作用是什么"]'
    with pytest.raises(ValueError, match="包含了疑问词或类似子问题的句式"):
        planner_prompt.parse_response(response)


def test_parse_response_duplicate_facets(planner_prompt):
    # Case insensitive duplicate logic (if enforced) or pure match
    response = '["药理作用", "临床副作用", "药理作用"]'
    with pytest.raises(ValueError, match="发现重复的角度"):
        planner_prompt.parse_response(response)
