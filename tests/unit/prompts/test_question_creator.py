import pytest
from src.prompts.question_creator import QuestionCreatorPrompt

@pytest.fixture
def prompt_handler():
    return QuestionCreatorPrompt()

def test_render_with_valid_context(prompt_handler):
    contexts = [
        "阿司匹林的主要不良反应是消化道出血。",
        "在妊娠最后三个月禁用该药物。"
    ]
    rendered = prompt_handler.render(contexts)
    assert "<role>" in rendered
    assert "问题创造者（Question Creator）" in rendered
    assert "[1] 阿司匹林的主要不良反应是消化道出血。" in rendered
    assert "[2] 在妊娠最后三个月禁用该药物。" in rendered

def test_render_with_empty_context(prompt_handler):
    rendered = prompt_handler.render([])
    assert "<context>" in rendered
    assert "</context>" in rendered
    # 无内容被渲染
    assert "[1]" not in rendered

def test_parse_response_valid_json(prompt_handler):
    response = '["阿司匹林的主要不良反应表现为什么？", "妊娠期妇女使用该药物有哪些明确禁忌？"]'
    questions = prompt_handler.parse_response(response)
    assert len(questions) == 2
    assert "阿司匹林的主要不良反应表现为什么？" in questions

def test_parse_response_with_markdown_blocks(prompt_handler):
    # 测试能否容错解析包裹了 ```json ``` 的输出
    response = '```json\n["这个药物在临床治疗上的用药禁忌有哪些？", "患者用药期间需要注意什么日常习惯？"]\n```'
    questions = prompt_handler.parse_response(response)
    assert len(questions) == 2

def test_parse_response_empty_array(prompt_handler):
    response = '[]'
    questions = prompt_handler.parse_response(response)
    assert questions == []

def test_parse_response_exceed_max_quantity(prompt_handler):
    response = '["问题1需要大于拾个字符", "问题2也需要大于拾个字", "问题3也满足了十个字哦", "问题4凑字数凑字数凑足", "问题5凑字数凑字数凑足", "问题6超出了最大限制上限"]'
    with pytest.raises(ValueError, match="超出了设定的上限 5 个"):
        prompt_handler.parse_response(response)

def test_parse_response_question_too_short(prompt_handler):
    response = '["什么副作用？", "这是个足够长足够常的好问题符合条件"]'
    with pytest.raises(ValueError, match="长度过短（<=10个字符）"):
        prompt_handler.parse_response(response)

def test_parse_response_not_a_list(prompt_handler):
    response = '{"q1": "这对吗对吗对吗？似乎不对呢", "q2": "这是什么格式的报错"}'
    with pytest.raises(ValueError, match="未能在模型输出中检测到有效的边界符"):
        prompt_handler.parse_response(response)

def test_parse_response_invalid_json_format(prompt_handler):
    # 模拟大模型强行附带文字导致的异常被正则防线或者Load防线拦住
    response = '下面是我给您造的问题：\n["这是个足够长的问题", "这当然也是个问题啦啦啦"'
    with pytest.raises(ValueError) as exc:
        prompt_handler.parse_response(response)
    # 不应该匹配到合法的 JSON 结束符或进入loads崩溃环节
    assert "解析失败" in str(exc.value)
