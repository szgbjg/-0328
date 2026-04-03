import pytest

from src.prompts.synthesis_agent import SynthesisAgentPrompt


@pytest.fixture
def synthesis_prompt():
    return SynthesisAgentPrompt()


def _valid_response() -> str:
    return """结论概览：该药物应在明确适应证下使用，并优先考虑高风险人群筛查。
核心依据整合：多份资料均提示禁忌、相互作用与剂量调整是决策重点。
完整展开说明：临床决策应按病情分层、并发症风险、既往用药史逐步展开，兼顾获益与风险。
风险与边界条件：对活动性出血、严重器官功能异常及过敏史患者需严格限制使用。
实务/操作建议：建议建立处方前核查清单，执行用药教育，并在治疗期间进行动态随访。
不确定性说明：个体差异与资料时效性可能影响结论外推，必要时应结合最新规范再评估。"""


def test_render_contains_query_and_answers(synthesis_prompt):
    query = "阿司匹林的综合用药建议"
    answers = ["回答一", "回答二"]

    prompt = synthesis_prompt.render(query, answers)

    assert "阿司匹林的综合用药建议" in prompt
    assert '"回答一"' in prompt
    assert '"回答二"' in prompt


def test_parse_response_valid_text(synthesis_prompt):
    result = synthesis_prompt.parse_response(_valid_response())
    assert "结论概览" in result
    assert "不确定性说明" in result


def test_parse_response_reject_json_array(synthesis_prompt):
    response = '["结论概览", "核心依据整合"]'

    with pytest.raises(ValueError, match="输出不能是 JSON 数组"):
        synthesis_prompt.parse_response(response)


def test_parse_response_missing_required_sections(synthesis_prompt):
    response = """结论概览：这是结论。
核心依据整合：这是依据。
完整展开说明：这是展开。
风险与边界条件：这是风险。
实务/操作建议：这是建议。"""

    with pytest.raises(ValueError, match="缺少标准部分"):
        synthesis_prompt.parse_response(response)


def test_parse_response_reject_english_characters(synthesis_prompt):
    response = """结论概览：这是结论。
核心依据整合：这是依据。
完整展开说明：这是展开。
风险与边界条件：这是风险。
实务/操作建议：这是建议。
不确定性说明：包含英文ABC。"""

    with pytest.raises(ValueError, match="检测到英文字符"):
        synthesis_prompt.parse_response(response)


def test_parse_response_empty_text(synthesis_prompt):
    with pytest.raises(ValueError, match="输出为空"):
        synthesis_prompt.parse_response("   ")
