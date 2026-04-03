import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppConfig(BaseSettings):
    """
    应用程序全局配置管理类
    自动从环境变量和 .env 文件中加载配置
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        protected_namespaces=('settings_',),
    )

    kg_api_url: str = Field(default="https://api.example.com/kg", description="知识图谱 API 地址")
    kg_api_key: Optional[str] = Field(default=None, description="知识图谱 API 密钥")
    
    model_api_url: str = Field(
        default="https://volley.inner.yzint.cn/v1/chat/completions",
        description="模型 API 地址",
    )
    model_api_key: Optional[str] = Field(default=None, description="模型 API 密钥")
    
    log_level: str = Field(default="INFO", description="日志级别")
    max_concurrent_requests: int = Field(default=10, description="最大并发请求数")

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        file_path (str): YAML 文件路径
        
    Returns:
        Dict[str, Any]: 解析后的字典数据
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    加载 JSON 配置文件
    
    Args:
        file_path (str): JSON 文件路径
        
    Returns:
        Dict[str, Any]: 解析后的字典数据
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 全局配置实例
config = AppConfig()
