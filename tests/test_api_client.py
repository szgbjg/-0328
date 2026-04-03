import pytest
import asyncio
import itertools
from aioresponses import aioresponses
from src.api_client import (
    ModelAPIClient, 
    KnowledgeGraphClient, 
    APIException, 
    TimeoutException, 
    RetryExhaustedException
)
from src.config_manager import config

# mock 环境变量
config.model_api_key = "test_model_key"
config.kg_api_key = "test_kg_key"

@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m

@pytest.fixture(autouse=True)
def reset_singletons():
    """每次测试前清理单例状态，确保测试隔离"""
    ModelAPIClient._instances = {}
    KnowledgeGraphClient._instances = {}

@pytest.mark.asyncio
async def test_model_client_singleton():
    client1 = ModelAPIClient()
    client2 = ModelAPIClient()
    assert client1 is client2

@pytest.mark.asyncio
async def test_kg_client_singleton():
    client1 = KnowledgeGraphClient()
    client2 = KnowledgeGraphClient()
    assert client1 is client2

@pytest.mark.asyncio
async def test_model_client_generate_success(mock_aioresponse):
    url = ModelAPIClient.BASE_URL
    mock_aioresponse.post(url, status=200, payload={"choices": [{"message": {"content": "Hello"}}]})
    
    async with ModelAPIClient() as client:
        response = await client.generate("gpt5.2", [{"role": "user", "content": "Hi"}])
        assert response.status == 200
        assert response.data["choices"][0]["message"]["content"] == "Hello"

@pytest.mark.asyncio
async def test_model_client_invalid_model():
    async with ModelAPIClient() as client:
        with pytest.raises(ValueError, match="不支持的模型类型"):
            await client.generate("invalid_model", [{"role": "user", "content": "Hi"}])

@pytest.mark.asyncio
async def test_model_client_timeout_and_retry_exhausted(mock_aioresponse):
    url = ModelAPIClient.BASE_URL
    # 模拟连续超时3次
    mock_aioresponse.post(url, timeout=True, repeat=True)
    
    async with ModelAPIClient() as client:
        with pytest.raises(RetryExhaustedException):
            await client.generate("gemini3pro", [{"role": "user", "content": "Hi"}])

@pytest.mark.asyncio
async def test_kg_client_success_and_deduplication(mock_aioresponse):
    url = config.kg_api_url
    mock_response_data = {
        "entities": [
            {"id": "e1", "name": "阿司匹林"},
            {"id": "e2", "name": "感冒"},
            {"id": "e1", "name": "阿司匹林"}  # 重复实体
        ]
    }
    mock_aioresponse.post(url, status=200, payload=mock_response_data)
    
    async with KnowledgeGraphClient() as client:
        res = await client.query(query="感冒吃什么药")
        assert res.status == 200
        # 验证去重逻辑，e1和e2共两个
        assert len(res.data["entities"]) == 2
        assert res.data["entities"][0]["id"] == "e1"
        assert res.data["entities"][1]["id"] == "e2"

@pytest.mark.asyncio
async def test_kg_client_caching(mock_aioresponse):
    url = config.kg_api_url
    mock_response_data = {"entities": []}
    
    # 仅允许1次请求响应，第二次如果有请求则会因为没有mock而报错
    mock_aioresponse.post(url, status=200, payload=mock_response_data)
    
    async with KnowledgeGraphClient() as client:
        # 第一次请求：调用 API
        res1 = await client.query(query="头痛", count=2, knowledge_base_id=201, hop_count=2, entity_ids="")
        assert res1.status == 200
        
        # 第二次请求：应从缓存读取，因而不需要 mock 网络请求
        res2 = await client.query(query="头痛", count=2, knowledge_base_id=201, hop_count=2, entity_ids="")
        assert res2.status == 200
