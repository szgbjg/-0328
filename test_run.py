import asyncio
from pathlib import Path

from src.logger import setup_logger, logger
from src.workflow_engine import MedicalQAWorkflow
from src.config_loader import get_config

async def main():
    setup_logger(log_level="DEBUG")
    logger.info("Medical KG QA demo (multi-round) starting")

    config = get_config()
    logger.info(
        f"Workflow config: max_backtracks={config.workflow.max_backtracks}, "
        f"max_concurrent_requests={config.workflow.max_concurrent_requests}"
    )

    session_id = "test_session_001"
    seed_context = "患者主诉：最近总是头晕，伴随恶心想吐，服用感冒药后无明显改善。"
    
    workflow = MedicalQAWorkflow(session_id=session_id)
    
    logger.info("===============================================")
    logger.info(f"Start workflow demo, Session: {session_id}")
    logger.info(f"Seed context: {seed_context}")
    logger.info("===============================================")

    result = await workflow.generate_multi_round(seed_context, min_round=2, max_round=3)

    out_path = Path("output/test_round.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    logger.info("===============================================")
    logger.info("Workflow demo finished. Final states:")
    for rnd in result.rounds_data:
        logger.info(f"[Round {rnd.round_num}] {rnd.final_response}")
    logger.info(f"Result JSON written to: {out_path}")
    logger.info("===============================================")


if __name__ == "__main__":
    asyncio.run(main())
