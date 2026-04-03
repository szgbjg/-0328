import pytest

from src.validators.output_validator import (
    FacetExpanderValidator,
    FacetPlannerValidator,
    FacetQAAgentValidator,
    FacetReducerValidator,
    QuestionCreatorValidator,
    RedundancyDetectorValidator,
    SynthesisAgentValidator,
)


def test_question_creator_validator_auto_trim_and_limit():
    validator = QuestionCreatorValidator()
    raw = '["这个问题长度明显超过十个字符吗", "短", 123, "第二个问题同样足够详细并可回答"]'

    result = validator.validate(raw, {})

    assert result.is_valid is False
    assert any("过短" in e for e in result.errors)
    assert any("不是字符串" in w for w in result.warnings)


def test_facet_planner_validator_dedup_and_invalid_question_style():
    validator = FacetPlannerValidator()
    raw = '["临床安全", "临床安全", "如何用药"]'

    result = validator.validate(raw, {})

    assert result.is_valid is False
    assert any("疑似子问题" in e for e in result.errors)
    assert any("自动去重" in w for w in result.warnings)


def test_facet_reducer_validator_strict_autocorrect():
    validator = FacetReducerValidator()
    context = {
        "original_facets": [
            "临床安全",
            "用药禁忌",
            "作用机制",
            "适应人群",
            "不良反应",
            "联合用药",
            "成本收益",
            "长期随访",
            "替代方案",
        ]
    }
    raw = '["临床安全", "新增角度", "用药禁忌", "临床安全"]'

    result = validator.validate(raw, context)

    assert result.is_valid is True
    assert len(result.corrected_output) == 8
    assert "新增角度" not in result.corrected_output
    assert any("自动补齐" in w for w in result.warnings)


def test_facet_expander_validator_filter_existing_and_cap_to_six():
    validator = FacetExpanderValidator()
    context = {"existing_facets": ["临床安全", "作用机制"]}
    raw = '["临床安全", "风险管理", "依从管理", "禁忌筛查", "人群分层", "成本分析", "长期随访", "联合用药"]'

    result = validator.validate(raw, context)

    assert result.is_valid is True
    assert len(result.corrected_output) == 6
    assert "临床安全" not in result.corrected_output


def test_facet_qa_agent_validator_forbidden_terms_must_fail():
    validator = FacetQAAgentValidator()
    context = {
        "facet": "临床安全",
        "refs": [
            {"source": "说明书", "content": "过敏者禁用", "location": "禁忌"},
        ],
        "graph_context": None,
    }
    raw = """<think>
<facet=临床安全>
问题拆解: 先识别禁忌。
证据清单:
- [证据R1:来源=refs:说明书]
推理链:
- 由R1可得过敏者禁用。
冲突识别: 无。
最终结论摘要: 高风险人群需禁用。
</think>

我查了一下，说明书明确过敏者禁用。
"""

    result = validator.validate(raw, context)

    assert result.is_valid is False
    assert any("禁止词汇" in e for e in result.errors)
    assert any("严重违规" in e for e in result.errors)


def test_facet_qa_agent_validator_valid_path():
    validator = FacetQAAgentValidator()
    context = {
        "facet": "临床安全",
        "refs": [
            {"source": "说明书", "content": "过敏者禁用", "location": "禁忌"},
            {"source": "指南", "content": "活动性出血避免使用", "location": "第3章"},
        ],
        "graph_context": "节点A-关系B-节点C",
    }
    raw = """<think>
<facet=临床安全>
问题拆解: 拆分禁忌与风险边界。
证据清单:
- [证据R1:来源=refs:说明书]
- [证据R2:来源=refs:指南]
推理链:
- 由R1可得过敏者禁用。
- 由R2可得活动性出血人群应避免。
冲突识别: 无。
最终结论摘要: 过敏和活动性出血是核心限制条件。
</think>

说明书与指南均提示，过敏体质及活动性出血人群应避免使用。
"""

    result = validator.validate(raw, context)

    assert result.is_valid is True
    assert isinstance(result.corrected_output, dict)
    assert "thinking" in result.corrected_output
    assert "answer" in result.corrected_output


def test_facet_qa_agent_validator_autofix_when_missing_think():
    validator = FacetQAAgentValidator()
    context = {
        "facet": "临床安全",
        "refs": [
            {"source": "说明书", "content": "过敏者禁用", "location": "禁忌"},
        ],
        "graph_context": None,
    }
    raw = "[正文回答]\n说明书提示过敏者禁用，活动性出血人群应避免。"

    result = validator.validate(raw, context)

    assert result.is_valid is True
    assert any("自动补齐" in w for w in result.warnings)
    assert isinstance(result.corrected_output, dict)
    assert result.corrected_output["thinking"]["facet"] == "临床安全"


def test_redundancy_detector_validator_range_and_type_errors():
    validator = RedundancyDetectorValidator()
    context = {"planners": [{"planner": "A", "answer": "a"}, {"planner": "B", "answer": "b"}]}
    raw = '[0, 2, "1", true]'

    result = validator.validate(raw, context)

    assert result.is_valid is False
    assert any("越界" in e for e in result.errors)
    assert any("不是整数下标" in e for e in result.errors)


def test_synthesis_agent_validator_reject_array_and_english():
    validator = SynthesisAgentValidator()
    raw = '["结论概览", "核心依据整合"]'

    result = validator.validate(raw, {})

    assert result.is_valid is False
    assert any("不应为 JSON 数组" in e for e in result.errors)


def test_synthesis_agent_validator_valid_text():
    validator = SynthesisAgentValidator()
    raw = """结论概览：该方案总体可行。
核心依据整合：多份证据在关键结论上相互支持。
完整展开说明：实施路径应按人群分层与风险分级推进。
风险与边界条件：需关注出血风险与肝肾功能边界。
实务/操作建议：建议处方前核查、用药教育与随访。
不确定性说明：个体差异与资料时效性可能影响外推。"""

    result = validator.validate(raw, {})

    assert result.is_valid is True
    assert isinstance(result.corrected_output, str)
