import asyncio
from src.logger import setup_logger, logger
from src.workflow_engine import MedicalQAWorkflow
from src.config_loader import get_config

async def main():
    # 1. 初始化日志
    setup_logger(log_level="DEBUG")
    logger.info("医疗图谱问答系统 - 测试模式启动")

    # 2. 读取配置测试
    config = get_config()
    logger.info(f"读取到工作流配置：最大回溯 {config.workflow.max_backtracks}，并发控制 {config.workflow.max_concurrent_requests}")

    # 3. 初始化并运行核心工作流
    session_id = "test_session_001"
    seed_context = "患者主诉：最近总是头晕，伴随恶心想吐，服用了一些感冒药但没有效果。"
    
    workflow = MedicalQAWorkflow(session_id=session_id)
    
    logger.info("===============================================")
    logger.info(f"开始演示工作流运行, Session: {session_id}")
    logger.info(f"初始输入Context: {seed_context}")
    logger.info("===============================================")

    # 执行模拟的多轮问答，指定最小2轮，最大3轮以快速查看结果
    result = await workflow.generate_multi_round(seed_context, min_round=2, max_round=3)

    logger.info("===============================================")
    logger.info("工作流执行完毕。最终生成状态：")
    for rnd in result.rounds_data:
        logger.info(f"【第 {rnd.round_num} 轮】最终结论: {rnd.final_response}")
    logger.info("===============================================")


if __name__ == "__main__":
    asyncio.run(main())
