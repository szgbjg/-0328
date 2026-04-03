import argparse
import asyncio
import json
import time
from pathlib import Path

from .logger import setup_logger, logger
from .workflow_engine import MedicalQAWorkflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical QA CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("generate_single", help="Generate one round result")
    single.add_argument("--context", required=True, help="Input context")
    single.add_argument("--output", required=True, help="Output JSON file path")
    return parser


async def _generate_single(context: str, output: str) -> int:
    setup_logger("INFO")

    session_id = f"single_{int(time.time())}"
    workflow = MedicalQAWorkflow(session_id=session_id)
    round_state = await workflow.generate_round(context=context, round_num=1)

    result = {
        "question": [round_state.question] if round_state.question else [],
        "facets": round_state.processed_facets or [],
        "facet_answers": round_state.facet_answers or [],
        "final_answer": round_state.final_response or "",
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Single-round result written to: {out_path}")
    return 0


async def _main_async() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate_single":
        return await _generate_single(args.context, args.output)

    return 1


def main() -> None:
    code = asyncio.run(_main_async())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
