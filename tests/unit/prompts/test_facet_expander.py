import pytest
from src.prompts.facet_expander import FacetExpanderPrompt


@pytest.fixture
def expander_prompt():
    return FacetExpanderPrompt()


def test_render_attributes(expander_prompt):
    query = "阿尔茨海默症"
    existing = ["临床表现", "致病机制"]
    prompt = expander_prompt.render(query, existing)
    assert "阿尔茨海默症" in prompt
    assert '"临床表现"' in prompt
    assert '"致病机制"' in prompt
    assert "0 到 6 个" in prompt


def test_parse_response_perfect_expansion(expander_prompt):
    """测试常规的新增：无重复，长度合规"""
    existing = ["方向A", "方向B", "方向C"]
    response = '["疗效评估", "依从管理"]'
    result = expander_prompt.parse_response(response, existing)
    assert len(result) == 2
    assert result == ["疗效评估", "依从管理"]


def test_parse_response_auto_filter_existing(expander_prompt):
    """测试模型幻觉：包含了已经存在的 facet 应当被自动过滤"""
    existing = ["历史沿革", "适应症", "禁忌症"]
    # 模型不仅输出了新的 ["并发症", "不良反应"]，还把 "适应症" 和 "禁忌症" 重复输出了
    response = '["并发症", "适应症", "不良反应", "禁忌症"]'
    
    result = expander_prompt.parse_response(response, existing)
    
    assert len(result) == 2
    assert result == ["并发症", "不良反应"]
    assert "适应症" not in result
    assert "禁忌症" not in result


def test_parse_response_auto_filter_self_duplicates(expander_prompt):
    """测试模型自身输出的数组中包含重复数据"""
    existing = ["基础A", "基础B"]
    response = '["风险管理", "用药教育", "风险管理"]'
    
    result = expander_prompt.parse_response(response, existing)
    assert len(result) == 2
    assert result == ["风险管理", "用药教育"]


def test_parse_response_truncate_to_six(expander_prompt):
    """测试当模型过度发散超过6个时，自动截断前6个"""
    existing = ["基础A", "基础B"]
    response = '["疗效评估", "风险管理", "依从管理", "禁忌筛查", "人群分层", "成本分析", "长期随访", "联合用药"]'
    
    result = expander_prompt.parse_response(response, existing)
    
    assert len(result) == 6
    assert result == ["疗效评估", "风险管理", "依从管理", "禁忌筛查", "人群分层", "成本分析"]


def test_parse_response_empty_is_valid(expander_prompt):
    """测试模型认为不需要扩展（输出空数组）应当合法通过"""
    existing = ["非常完美的角度1", "非常完美的角度2"]
    response = '[]'
    
    result = expander_prompt.parse_response(response, existing)
    assert len(result) == 0
    assert result == []


def test_parse_response_invalid_format(expander_prompt):
    """测试基础格式不合法：过长或含有疑问句应直接抛错熔断"""
    existing = ["起因"]
    
    with pytest.raises(ValueError, match="必须是短词组"):
        expander_prompt.parse_response('["这个新的角度真的太长了受不了呀受不了"]', existing)
        
    with pytest.raises(ValueError, match="包含了疑问词"):
        expander_prompt.parse_response('["怎么治疗"]', existing)
        
    with pytest.raises(ValueError, match="不是字符串格式"):
        expander_prompt.parse_response('[123, "正常"]', existing)
