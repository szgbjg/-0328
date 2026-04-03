import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    配置全局日志记录器
    
    Args:
        log_level (str): 日志输出级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): 日志文件路径
    """
    # 移除默认的 handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level=log_level)
    
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加文件输出，按大小轮转，保留10天
    logger.add(log_file, rotation="10 MB", retention="10 days", level=log_level)

__all__ = ["logger", "setup_logger"]
