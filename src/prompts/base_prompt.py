from abc import ABC, abstractmethod
from typing import Any

class BasePrompt(ABC):
    """
    Prompt 模板基类
    每个具体的 Prompt 类必须实现 render 与 parse_response 方法
    """
    
    @abstractmethod
    def render(self, **kwargs) -> str:
        """渲染生成 Prompt 文本"""
        pass
        
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """解析模型返回的结果并进行验证"""
        pass
