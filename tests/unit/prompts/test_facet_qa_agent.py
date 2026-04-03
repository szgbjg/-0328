import pytest

from src.prompts.facet_qa_agent import FacetQAAgentPrompt


@pytest.fixture
def qa_prompt():
    return FacetQAAgentPrompt()


@pytest.fixture
def sample_refs():
    return [
        {
            "source": "《阿司匹林肠溶片说明书》",
            "content": "【禁忌】过敏者禁用；活动性消化性溃疡禁用。",
            "location": "禁忌章节",
        },
        {
            "source": "《2023年临床心血管用药指南》",
            "content": "严重肝肾功能不全和活动性出血患者应避免使用。",
            "location": "第3章",
        },
    ]


def _valid_response(facet: str) -> str:
    return f"""<think>
<facet={facet}>
问题拆解: 围绕禁忌人群与临床风险进行分层。
证据清单:
- [证据R1:来源=refs:《阿司匹林肠溶片说明书》]
- [证据R2:来源=refs:《2023年临床心血管用药指南》]
推理链:
- 由R1可得过敏与溃疡属于明确禁忌。
- 由R2可得出血风险与严重肝肾功能不全属于高风险禁用条件。
冲突识别: 无明显冲突。
最终结论摘要: 主要禁忌集中在过敏、活动性出血/溃疡和重度器官功能异常。
</think>

《阿司匹林肠溶片说明书》与《2023年临床心血管用药指南》都提示，过敏体质、活动性出血或溃疡、重度肝肾功能异常患者应避免使用。
"""


def test_validate_happy_path(qa_prompt, sample_refs):
    facet = "临床安全"
    response = _valid_response(facet)

    result = qa_prompt.validate(response, facet=facet, refs=sample_refs)

    assert result.is_valid is True
    assert result.errors == []
    assert result.forbidden_hits == []


def test_parse_thinking_and_answer(qa_prompt):
    facet = "临床安全"
    response = _valid_response(facet)

    thinking = qa_prompt.parse_thinking(response)
    answer = qa_prompt.parse_answer(response)

    assert thinking["facet"] == facet
    assert "证据R1" in thinking["证据清单"]
    assert "由R1可得" in thinking["推理链"]
    assert answer.startswith("《阿司匹林肠溶片说明书》")


@pytest.mark.parametrize(
    "bad_term",
    [
        "检索",
        "查阅",
        "搜索",
        "工具返回",
        "调用接口",
        "从图谱里查到",
        "我查了一下",
        "系统显示",
        "根据检索结果",
        "retrieved",
        "searched",
        "looked up",
        "tool says",
    ],
)
def test_validate_forbidden_terms_all_covered(qa_prompt, sample_refs, bad_term):
    facet = "临床安全"
    response = _valid_response(facet) + f"\n补充说明：{bad_term}"

    result = qa_prompt.validate(response, facet=facet, refs=sample_refs)

    assert result.is_valid is False
    assert any("安全校验失败" in err for err in result.errors)
    assert bad_term in result.forbidden_hits


def test_validate_facet_mismatch(qa_prompt, sample_refs):
    response = _valid_response("治疗策略")

    result = qa_prompt.validate(response, facet="临床安全", refs=sample_refs)

    assert result.is_valid is False
    assert any("facet 属性不匹配" in err for err in result.errors)


def test_validate_evidence_format_error(qa_prompt, sample_refs):
    facet = "临床安全"
    response = f"""<think>
<facet={facet}>
问题拆解: 先识别禁忌人群。
证据清单:
- R1 来自说明书
- R2 来自指南
推理链:
- 由R1可得过敏者禁用。
- 由R2可得出血患者应避免。
冲突识别: 无。
最终结论摘要: 两类来源均支持风险人群禁用。
</think>

正文回答。
"""

    result = qa_prompt.validate(response, facet=facet, refs=sample_refs)

    assert result.is_valid is False
    assert any("证据清单缺少合法证据格式" in err for err in result.errors)


def test_validate_evidence_not_in_refs(qa_prompt, sample_refs):
    facet = "临床安全"
    response = f"""<think>
<facet={facet}>
问题拆解: 检查证据有效性。
证据清单:
- [证据R3:来源=refs:《不存在的来源》]
推理链:
- 由R3可得关键结论。
冲突识别: 无。
最终结论摘要: 结论成立。
</think>

正文回答。
"""

    result = qa_prompt.validate(response, facet=facet, refs=sample_refs)

    assert result.is_valid is False
    assert any("在 refs 中无对应条目" in err for err in result.errors)


def test_validate_reasoning_must_reference_evidence(qa_prompt, sample_refs):
    facet = "临床安全"
    response = f"""<think>
<facet={facet}>
问题拆解: 提取禁忌证据。
证据清单:
- [证据R1:来源=refs:《阿司匹林肠溶片说明书》]
推理链:
- 仅陈述风险，不给编号。
冲突识别: 无。
最终结论摘要: 应规避高风险人群。
</think>

正文回答。
"""

    result = qa_prompt.validate(response, facet=facet, refs=sample_refs)

    assert result.is_valid is False
    assert any("推理链未引用任何证据编号" in err for err in result.errors)
