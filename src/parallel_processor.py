import asyncio
from typing import List, Callable, Any, Coroutine, TypeVar
from .logger import logger
from .config_manager import config

T = TypeVar('T')

class ParallelProcessor:
    """
    异步并发处理工具类
    控制并发量，避免打满外部 API 及提高处理效率。
    """
    def __init__(self, max_concurrent: int = config.max_concurrent_requests):
        """
        Args:
            max_concurrent (int): 最大并发数，默认为配置中的参数
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _task_wrapper(self, task: Coroutine[Any, Any, T], task_id: int) -> T:
        """使用信号量包装异步任务"""
        async with self.semaphore:
            logger.debug(f"任务 {task_id} 开始执行...")
            try:
                result = await task
                logger.debug(f"任务 {task_id} 执行完成.")
                return result
            except Exception as e:
                logger.error(f"任务 {task_id} 执行中发生错误: {str(e)}")
                raise

    async def process_batch(self, tasks: List[Coroutine[Any, Any, T]]) -> List[T | Exception]:
        """
        并发执行一组异步任务，并控制并发数
        
        Args:
            tasks: 协程任务列表
            
        Returns:
            List: 返回每个任务的执行结果。如果出错，则返回对应的 Exception 对象。
        """
        logger.info(f"开始批量处理 {len(tasks)} 个任务, 最大并发数: {self.semaphore._value}")
        
        # 将协程包装进限流器中
        wrapped_tasks = [
            self._task_wrapper(task, i) for i, task in enumerate(tasks)
        ]
        
        # return_exceptions=True 确保个别任务失败不会中断整体批量流程并直接抛出异常
        results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
        
        logger.info(f"批量任务处理完毕，共完成 {len(results)} 个任务")
        return results
