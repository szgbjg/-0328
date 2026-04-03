import asyncio

from src.agents.facet_qa_agent import FacetQAAgent


async def _fake_model_ok(prompt: str, temperature: float = 0.7) -> str:
    _ = prompt, temperature
    return """<think>
<facet=临床安全>
证据清单: [证据R1:来源=refs:《说明书》]
推理链: P1 -> C1
</think>
正文：说明书提示过敏者禁用。
"""


async def _fake_model_banned_then_ok(prompt: str, temperature: float = 0.7) -> str:
    _ = prompt
    if temperature >= 0.7:
        return """<think>
<facet=临床安全>
证据清单: [证据R1:来源=refs:《说明书》]
</think>
正文：我查了一下说明书，过敏者禁用。
"""
    return """<think>
<facet=临床安全>
证据清单: [证据R1:来源=refs:《说明书》]
</think>
正文：资料显示过敏者禁用。
"""


async def _fake_model_no_evidence(prompt: str, temperature: float = 0.7) -> str:
    _ = prompt, temperature
    return "正文：说明书提示过敏者禁用。"


def test_generate_facet_answer_ok():
    agent = FacetQAAgent()
    refs = [{"id": "R1", "source": "《说明书》", "content": "过敏者禁用"}]
    result = asyncio.run(agent.generate_facet_answer("q", "临床安全", refs, _fake_model_ok))
    assert result is not None
    assert result["evidence_count"] == 1
    assert "<facet=临床安全>" in result["thinking"]


def test_generate_facet_answer_banned_retry_ok():
    agent = FacetQAAgent()
    refs = [{"id": "R1", "source": "《说明书》", "content": "过敏者禁用"}]
    result = asyncio.run(agent.generate_facet_answer("q", "临床安全", refs, _fake_model_banned_then_ok))
    assert result is not None
    assert "我查了一下" not in result["body"]


def test_generate_facet_answer_no_evidence_autofix():
    agent = FacetQAAgent()
    refs = [{"id": "R1", "source": "《说明书》", "content": "过敏者禁用"}]
    result = asyncio.run(agent.generate_facet_answer("q", "临床安全", refs, _fake_model_no_evidence))
    assert result is not None
    assert result["evidence_count"] >= 1
