import argparse
import asyncio
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.api_client import ModelAPIClient
from src.prompts.facet_expander import FacetExpanderPrompt
from src.prompts.facet_planner import FacetPlannerPrompt
from src.prompts.facet_qa_agent import FacetQAAgentPrompt
from src.prompts.facet_reducer import FacetReducerPrompt
from src.prompts.question_creator import QuestionCreatorPrompt
from src.prompts.redundancy_detector import RedundancyDetectorPrompt
from src.prompts.synthesis_agent import SynthesisAgentPrompt
from src.validators.output_validator import (
    FacetExpanderValidator,
    FacetPlannerValidator,
    FacetQAAgentValidator,
    FacetReducerValidator,
    QuestionCreatorValidator,
    RedundancyDetectorValidator,
    SynthesisAgentValidator,
)


SUPPORTED_MODELS = {
    "gpt5.2",
    "gemini3pro",
    "kimi2.5",
    "qwen",
    "qwen3.5-plus",
    "gpt-5.2",
    "gemini-3-pro-preview",
    "kimi-k2.5",
    "z-ai/glm-4.5-air:free",
    "stepfun-ai/Step-3.5-Flash",
}

ROLES = {
    "question_creator",
    "facet_planner",
    "facet_reducer",
    "facet_expander",
    "facet_qa_agent",
    "redundancy_detector",
    "synthesis_agent",
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_fixture() -> Dict[str, Any]:
    fixture_path = workspace_root() / "tests" / "fixtures" / "sample_context.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_fixture(data: Dict[str, Any]) -> Dict[str, Any]:
    # 兼容两种结构:
    # 1) 扁平结构(旧版)
    # 2) test_cases[0] 结构(新版)
    if "test_cases" in data and isinstance(data["test_cases"], list) and data["test_cases"]:
        case = data["test_cases"][0]
        refs = case.get("context", [])
        if refs and isinstance(refs[0], dict):
            context_list = [str(x.get("content", "")).strip() for x in refs]
        else:
            context_list = [str(x).strip() for x in refs]

        query = case.get("query") or data.get("query") or "请基于给定证据进行医药问答分析"
        expected_types = case.get("expected_question_types", [])

        return {
            "query": query,
            "context": context_list,
            "refs": refs if refs and isinstance(refs[0], dict) else [],
            "facet": "临床安全",
            "graph_context": data.get("graph_context", "图谱节点: 阿司匹林 -> 禁忌 -> 活动性消化性溃疡"),
            "existing_facets": data.get("existing_facets", ["临床安全", "适应症", "不良反应"]),
            "facets": data.get(
                "facets",
                [
                    "临床安全",
                    "适应症",
                    "不良反应",
                    "禁忌症",
                    "人群分层",
                    "相互作用",
                    "剂量建议",
                    "长期随访",
                    "替代方案",
                ],
            ),
            "planners": data.get(
                "planners",
                [
                    {"planner": "临床安全", "answer": "聚焦禁忌症和高风险人群。"},
                    {"planner": "风险治理", "answer": "强调筛查与监测路径，结论与临床安全接近。"},
                    {"planner": "应用落地", "answer": "强调处方前核查与随访执行。"},
                ],
            ),
            "answers": data.get(
                "answers",
                [
                    "说明书与临床指南均提示应优先识别禁忌症与高风险人群。",
                    "建议通过处方前核查、患者教育和持续随访降低用药风险。",
                ],
            ),
            "expected_question_types": expected_types,
        }

    return data


def build_prompt_and_context(role: str, fixture: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Any]:
    query = fixture.get("query", "")

    if role == "question_creator":
        context_list = fixture.get("context", [])
        prompt_obj = QuestionCreatorPrompt()
        prompt_text = prompt_obj.render(context=context_list)
        return prompt_text, {"context": context_list, "query": query}, prompt_obj

    if role == "facet_planner":
        prompt_obj = FacetPlannerPrompt()
        prompt_text = prompt_obj.render(query=query)
        return prompt_text, {"query": query}, prompt_obj

    if role == "facet_reducer":
        facets = fixture.get("facets", [])
        prompt_obj = FacetReducerPrompt()
        prompt_text = prompt_obj.render(query=query, facets=facets)
        return prompt_text, {"query": query, "original_facets": facets}, prompt_obj

    if role == "facet_expander":
        existing_facets = fixture.get("existing_facets", [])
        prompt_obj = FacetExpanderPrompt()
        prompt_text = prompt_obj.render(query=query, existing_facets=existing_facets)
        return prompt_text, {"query": query, "existing_facets": existing_facets}, prompt_obj

    if role == "facet_qa_agent":
        facet = fixture.get("facet", "临床安全")
        refs = fixture.get("refs", [])
        graph_context = fixture.get("graph_context")
        prompt_obj = FacetQAAgentPrompt()
        prompt_text = prompt_obj.render(query=query, facet=facet, refs=refs, graph_context=graph_context)
        return prompt_text, {
            "query": query,
            "facet": facet,
            "refs": refs,
            "graph_context": graph_context,
        }, prompt_obj

    if role == "redundancy_detector":
        planners = fixture.get("planners", [])
        prompt_obj = RedundancyDetectorPrompt()
        prompt_text = prompt_obj.render(query=query, planners=planners)
        return prompt_text, {"query": query, "planners": planners}, prompt_obj

    if role == "synthesis_agent":
        answers = fixture.get("answers", [])
        prompt_obj = SynthesisAgentPrompt()
        prompt_text = prompt_obj.render(query=query, answers=answers)
        return prompt_text, {"query": query, "answers": answers}, prompt_obj

    raise ValueError(f"不支持的角色名: {role}")


def build_validator(role: str):
    if role == "question_creator":
        return QuestionCreatorValidator()
    if role == "facet_planner":
        return FacetPlannerValidator()
    if role == "facet_reducer":
        return FacetReducerValidator()
    if role == "facet_expander":
        return FacetExpanderValidator()
    if role == "facet_qa_agent":
        return FacetQAAgentValidator()
    if role == "redundancy_detector":
        return RedundancyDetectorValidator()
    if role == "synthesis_agent":
        return SynthesisAgentValidator()
    raise ValueError(f"不支持的角色名: {role}")


def extract_model_text(api_data: Dict[str, Any]) -> str:
    choices = api_data.get("choices") if isinstance(api_data, dict) else None
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            if parts:
                return "\n".join(parts).strip()

    for key in ("output_text", "text", "response"):
        val = api_data.get(key) if isinstance(api_data, dict) else None
        if isinstance(val, str):
            return val

    return json.dumps(api_data, ensure_ascii=False, indent=2)


def parse_with_prompt(role: str, prompt_obj: Any, raw_text: str, fixture: Dict[str, Any]) -> Any:
    if role == "facet_reducer":
        return prompt_obj.parse_response(raw_text, fixture.get("facets", []))
    if role == "facet_expander":
        return prompt_obj.parse_response(raw_text, fixture.get("existing_facets", []))
    if role == "redundancy_detector":
        return prompt_obj.parse_response(raw_text, fixture.get("planners", []))
    return prompt_obj.parse_response(raw_text)


def save_outputs(role: str, model: str, raw_text: str, parsed: Any, validation_report: Dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "_", model)
    save_dir = workspace_root() / "tests" / "manual" / "results" / f"{role}_{safe_model}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    (save_dir / "parsed_result.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (save_dir / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return save_dir


async def run_manual_test(role: str, model: str, save: bool, verbose: bool) -> int:
    if role not in ROLES:
        raise ValueError(f"不支持的角色名: {role}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"--model 必须是 {sorted(SUPPORTED_MODELS)} 之一")

    print("=" * 72)
    print("[步骤1] 读取并规范化测试输入")
    fixture_raw = load_fixture()
    fixture = normalize_fixture(fixture_raw)

    print(f"- 角色: {role}")
    print(f"- 模型: {model}")
    print(f"- 保存结果: {save}")
    print(f"- 详细输出: {verbose}")

    print("=" * 72)
    print("[步骤2] 组装 Prompt 与验证上下文")
    prompt_text, validation_context, prompt_obj = build_prompt_and_context(role, fixture)
    print("- 输入参数:")
    print(json.dumps(validation_context, ensure_ascii=False, indent=2))

    if verbose:
        print("- 渲染后的 Prompt:")
        print(prompt_text)

    print("=" * 72)
    print("[步骤3] 调用真实模型 API")
    async with ModelAPIClient() as client:
        api_resp = await client.generate(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,
        )

    raw_text = extract_model_text(api_resp.data or {})
    print("- 模型输出(raw):")
    print(raw_text)

    print("=" * 72)
    print("[步骤4] 解析模型输出")
    parsed_result: Any = None
    parse_error = ""
    try:
        parsed_result = parse_with_prompt(role, prompt_obj, raw_text, fixture)
        print("- 解析成功")
        if verbose:
            print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
    except Exception as e:
        parse_error = str(e)
        print(f"- 解析失败: {parse_error}")

    print("=" * 72)
    print("[步骤5] 执行统一验证")
    validator = build_validator(role)
    validation_result = validator.validate(raw_text, validation_context)

    if parsed_result is None and validation_result.corrected_output is not None:
        parsed_result = validation_result.corrected_output
        parse_error = ""
        print("- 已使用验证器自动修正结果完成结构化解析")

    print(f"- 验证状态: {'通过' if validation_result.is_valid else '失败'}")
    if validation_result.errors:
        print("- 错误详情:")
        for err in validation_result.errors:
            print(f"  * {err}")
    if validation_result.warnings:
        print("- 警告详情:")
        for warn in validation_result.warnings:
            print(f"  * {warn}")

    validation_report = asdict(validation_result)
    validation_report["role"] = role
    validation_report["model"] = model
    validation_report["parse_error"] = parse_error

    if save:
        save_dir = save_outputs(
            role=role,
            model=model,
            raw_text=raw_text,
            parsed=parsed_result if parsed_result is not None else {},
            validation_report=validation_report,
        )
        print("=" * 72)
        print("[步骤6] 保存测试产物")
        print(f"- 已保存到: {save_dir}")
        print(f"  * {(save_dir / 'raw_response.txt').name}")
        print(f"  * {(save_dir / 'parsed_result.json').name}")
        print(f"  * {(save_dir / 'validation_report.json').name}")

    print("=" * 72)
    return 0 if validation_result.is_valid else 2


def parse_args(default_role: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"手动测试脚本({default_role})：真实 API 调用 + 解析 + 验证")
    parser.add_argument(
        "--model",
        default="gpt5.2",
        help="选择模型（例如 gpt5.2 / qwen / qwen3.5-plus / gemini3pro / kimi2.5）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存 raw_response.txt / parsed_result.json / validation_report.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )
    return parser.parse_args()
