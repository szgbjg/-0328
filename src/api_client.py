import asyncio
import json
import time
from typing import Dict, Any, Optional, AsyncGenerator, List, Union
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from pydantic import BaseModel

from .logger import logger
from .config_manager import config

# ================= 自定义异常 =================
class APIException(Exception):
    """API 基类异常"""
    pass

class TimeoutException(APIException):
    """请求超时异常"""
    pass

class RetryExhaustedException(APIException):
    """重试耗尽异常"""
    pass


# ================= 单例元类 =================
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ================= 响应模型 =================
class APIResponse(BaseModel):
    status: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ================= 重试日志记录 =================
def log_retry_attempt(retry_state):
    exception = retry_state.outcome.exception()
    logger.warning(
        f"[{retry_state.fn.__name__}] 操作失败，尝试第 {retry_state.attempt_number} 次重试. 异常: {exception}"
    )


# ================= 模型API客户端 =================
class ModelAPIClient(metaclass=SingletonMeta):
    """
    大语言模型 API 客户端 (单例)
    """
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
    MODEL_ALIASES = {
        "gpt5.2": "gpt-5.2",
        "gemini3pro": "gemini-3-pro-preview",
        "kimi2.5": "kimi-k2.5",
        "qwen": "qwen3.5-plus",
    }
    def __init__(self):
        # 连接超时10s，读取超时60s
        self.timeout = aiohttp.ClientTimeout(sock_connect=10, sock_read=60)
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.model_api_url

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("ModelAPIClient 会话已开启")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("ModelAPIClient 会话已关闭")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("ModelAPIClient 会话已隐式开启")
        return self.session

    def _get_retry_decorator(self):
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, TimeoutException)),
            before_sleep=log_retry_attempt,
            reraise=True
        )

    async def generate(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> Union[APIResponse, AsyncGenerator[str, None]]:
         # 使用装饰器包装具体实现以捕获重试耗尽
        retry_decorator = self._get_retry_decorator()
        
        @retry_decorator
        async def _do_request():
            model_name = self.MODEL_ALIASES.get(model, model)

            if model_name not in self.SUPPORTED_MODELS:
                raise ValueError(f"不支持的模型类型: {model}。可选模型: {self.SUPPORTED_MODELS}")

            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {config.model_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": stream
            }

            try:
                # 若需要流式，请留意在aiohttp中必须手动处理流对象
                response = await session.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()

                if stream:
                    return self._stream_response(response)
                else:
                    data = await response.json()
                    status = response.status
                    # 在非流式情况下即可关闭 response，因为 data 已收取完
                    response.close()
                    return APIResponse(status=status, data=data)

            except asyncio.TimeoutError:
                logger.error(f"模型 [{model_name}] 请求超时")
                raise TimeoutException("请求模型API超时")
            except aiohttp.ClientResponseError as e:
                logger.error(f"模型 [{model_name}] HTTP错误: 状态码 {e.status}")
                # 抛出异常供 tenacity 重试或向外抛出
                raise APIException(f"API请求失败，状态码: {e.status}")
            except aiohttp.ClientError as e:
                logger.error(f"模型 [{model_name}] 客户端请求异常: {str(e)}")
                raise APIException(f"客户端错误: {str(e)}")
                
        try:
            return await _do_request()
        except RetryError as e:
            logger.error("ModelAPIClient 请求重试次数已耗尽")
            raise RetryExhaustedException("模型API请求重试耗尽") from e

    async def _stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """处理流式响应数据"""
        try:
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            # 尝试解析 JSON chunk
                            chunk = json.loads(data_str)
                            # yield chunk
                            yield json.dumps(chunk, ensure_ascii=False)
                        except json.JSONDecodeError:
                            yield data_str
        finally:
            response.close()

# ================= 知识图谱客户端 =================
class KnowledgeGraphClient(metaclass=SingletonMeta):
    """
    医药知识图谱 API 客户端 (单例)
    """
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        # 简单缓存过期时间设置 (例如: 300秒)
        self._cache_ttl = 300

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("KnowledgeGraphClient 会话已开启")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("KnowledgeGraphClient 会话已关闭")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("KnowledgeGraphClient 会话已隐式开启")
        return self.session

    def _get_cache_key(self, query: str, entity_ids: str, count: int, kb_id: int, hop_count: int) -> str:
        return f"{query}_{entity_ids}_{count}_{kb_id}_{hop_count}"
        
    def _deduplicate_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        实体去重逻辑：提取结果中实体列表并根据实体ID进行去重
        """
        if "entities" in data and isinstance(data["entities"], list):
            seen_ids = set()
            unique_entities = []
            for entity in data["entities"]:
                eid = entity.get("id")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    unique_entities.append(entity)
            data["entities"] = unique_entities
            logger.debug(f"知识图谱实体去重完毕，现余 {len(unique_entities)} 个实体")
        return data

    @retry(
        stop=stop_after_attempt(3), # 包含首次共尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, TimeoutException)),
        before_sleep=log_retry_attempt,
        reraise=True
    )
    async def _do_query(self, url: str, headers: dict, payload: dict) -> Dict[str, Any]:
        session = await self._get_session()
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except asyncio.TimeoutError:
            logger.error("知识图谱请求超时")
            raise TimeoutException("知识图谱API请求超时")
        except aiohttp.ClientError as e:
            logger.error(f"知识图谱请求发生客户端异常: {str(e)}")
            raise APIException(f"知识图谱客户端请求失败: {str(e)}")

    async def query(self, 
                    query: str, 
                    count: int = 2, 
                    knowledge_base_id: int = 201, 
                    hop_count: int = 2,
                    entity_ids: str = "") -> APIResponse:
        """
        查询知识图谱，包含数据缓存机制与实体去重
        """
        cache_key = self._get_cache_key(query, entity_ids, count, knowledge_base_id, hop_count)
        
        # 检查缓存是否存在且未过期
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            # 这里先假设为未过期，可以实现逻辑
            if time.time() - cache_entry["timestamp"] < self._cache_ttl:
                logger.info("命中知识图谱数据缓存")
                return APIResponse(status=200, data=cache_entry["data"])
            else:
                # 缓存失效则删除
                del self._cache[cache_key]

        headers = {
            "Authorization": f"Bearer {config.kg_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "entityIds": entity_ids,
            "count": count,
            "knowledgeBaseId": knowledge_base_id,
            "hopCount": hop_count
        }

        try:
            # 执行带有重试机制的查询
            raw_data = await self._do_query(config.kg_api_url, headers, payload)
        except RetryError as e:
            logger.error("KnowledgeGraphClient 请求重试次数已耗尽")
            raise RetryExhaustedException("知识图谱API请求重试耗尽") from e
            
        # 实体去重
        processed_data = self._deduplicate_entities(raw_data)
        
        # 写入缓存
        self._cache[cache_key] = {
            "timestamp": time.time(),
            "data": processed_data
        }
        logger.debug(f"已缓存知识图谱查询结果，Key: {cache_key}")
        
        return APIResponse(status=200, data=processed_data)
