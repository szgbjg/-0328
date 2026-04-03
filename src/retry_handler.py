import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .logger import logger

def get_default_retry_policy():
    """
    获取默认的重试策略：
    - 最大重试次数：3次
    - 间隔时间：指数退避，最小2秒，最大10秒
    - 仅在此类异常发生时重试：aiohttp.ClientError, asyncio.TimeoutError
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
        before_sleep=log_retry_attempt
    )

def log_retry_attempt(retry_state):
    """
    重试失败时的日志回调函数
    """
    exception = retry_state.outcome.exception()
    logger.warning(f"操作失败，准备进行第 {retry_state.attempt_number} 次重试. 异常: {exception}")

def async_retry(max_attempts: int = 3, min_wait: float = 2.0, max_wait: float = 10.0, exceptions: tuple = (Exception,)):
    """
    自定义异步重试装饰器
    
    Args:
        max_attempts (int): 最大重试次数
        min_wait (float): 最小等待时间(秒)
        max_wait (float): 最大等待时间(秒)
        exceptions (tuple): 触发重试的异常类型元组
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=log_retry_attempt
    )
