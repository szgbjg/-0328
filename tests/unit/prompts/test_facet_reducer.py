import pytest
from src.prompts.facet_reducer import FacetReducerPrompt


@pytest.fixture
def reducer_prompt():
    return FacetReducerPrompt()


def test_render_with_valid_inputs(reducer_prompt):
    query = "胰岛素抵抗"
    facets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    prompt = reducer_prompt.render(query, facets)
    assert "胰岛素抵抗" in prompt
    assert "挑选出最重要、最具有代表性、差异最大的 8 个角度" in prompt
    assert '"A"' in prompt
    assert '"J"' in prompt


def test_parse_response_perfect_8_facets(reducer_prompt):
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    response = '["A", "B", "C", "D", "E", "F", "G", "H"]'
    
    result = reducer_prompt.parse_response(response, original)
    
    assert len(result) == 8
    assert result == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_parse_response_with_hallucinations(reducer_prompt):
    """测试模型输出了不在 original 列表中的新角度（幻觉）"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # 模拟模型输出了 X, Y 两个自己编造的角度
    response = '["A", "B", "C", "D", "X", "Y", "G", "H"]'
    
    result = reducer_prompt.parse_response(response, original)
    
    # 应该过滤掉 X, Y，剩余 6 个(A,B,C,D,G,H)，然后从 original 中自动补充(E, F)凑满 8 个
    assert len(result) == 8
    assert "X" not in result
    assert "Y" not in result
    assert result == ["A", "B", "C", "D", "G", "H", "E", "F"]


def test_parse_response_with_duplicates(reducer_prompt):
    """测试模型输出了重复项"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    # 模型输出中，"A" 重复出现3次，导致有效只有 6 个不同的项
    response = '["A", "A", "A", "D", "E", "F", "G", "H"]'
    
    result = reducer_prompt.parse_response(response, original)
    
    # 去重后剩 A,D,E,F,G,H (6个)，自动补充 B,C 凑满 8 个
    assert len(result) == 8
    assert result.count("A") == 1
    assert result == ["A", "D", "E", "F", "G", "H", "B", "C"]


def test_parse_response_too_many_facets(reducer_prompt):
    """测试模型过度生成，返回超过 8 个角度"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    # 模型未听从指令，返回了 10 个
    response = '["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]'
    
    result = reducer_prompt.parse_response(response, original)
    
    # 按照自动修复逻辑，应该直接截取前 8 个
    assert len(result) == 8
    assert result == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_parse_response_too_few_facets(reducer_prompt):
    """测试模型未能返回充足的 8 个角度"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    # 模型仅仅返回了 3 个
    response = '["A", "E", "J"]'
    
    result = reducer_prompt.parse_response(response, original)
    
    # 应该自动从未使用的 original 中顺位补充到 8 个
    assert len(result) == 8
    assert result == ["A", "E", "J", "B", "C", "D", "F", "G"]


def test_parse_response_not_a_list(reducer_prompt):
    """测试完全拉爆：不返回有效 JSON"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    response = '这是分析结论，不用谢'
    
    with pytest.raises(ValueError, match="未能在模型输出中检测到有效的边界符"):
        reducer_prompt.parse_response(response, original)
