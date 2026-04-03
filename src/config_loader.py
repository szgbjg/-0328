import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Type, Tuple

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict, 
    PydanticBaseSettingsSource
)
from .logger import logger


# ================= 配置数据模型 =================
class APIConfig(BaseModel):
    model_endpoint: str = "https://volley.inner.yzint.cn/v1/chat/completions"
    model_api_key: str = ""
    kg_endpoint: str = "https://api.example.com/kg"
    kg_api_key: str = ""
    timeout_connect: int = 10
    timeout_read: int = 60

class KnowledgeGraphConfig(BaseModel):
    base_id: int = 201
    hop_count: int = 2
    count: int = 2

class WorkflowConfig(BaseModel):
    max_retries: int = 3
    max_backtracks: int = 3
    max_concurrent_requests: int = 10
    min_rounds: int = 3
    max_rounds: int = 8
    checkpoint_dir: str = "data/checkpoints"

class ModelRoutingConfig(BaseModel):
    question_creator: str = "gpt5.2"
    facet_planner: str = "gpt5.2"
    facet_expander: str = "gemini3pro"
    facet_reducer: str = "gemini3pro"
    facet_qa_agent: str = "kimi2.5"
    redundancy_detector: str = "gpt5.2"
    synthesis_agent: str = "kimi2.5"


# ================= 自定义 YAML 数据源 =================
class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """自定义数据源：读取 YAML，使其优先级位于默认值之上，环境变量之下"""
    def __init__(self, settings_cls: Type[BaseSettings], yaml_path: str = "config/settings.yaml"):
        super().__init__(settings_cls)
        self.yaml_path = Path(yaml_path)

    def get_field_value(self, field, field_name: str) -> Tuple[Any, str, bool]:
        # 不在该方法中处理单独字段，直接整块注入
        raise NotImplementedError

    def __call__(self) -> Dict[str, Any]:
        if self.yaml_path.exists():
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}


# ================= 全局根配置 =================
class AppConfig(BaseSettings):
    api: APIConfig = Field(default_factory=APIConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    
    model_config = SettingsConfigDict(
        # 环境变量前缀 APP_，子嵌套使用双下划线如 APP_API__MODEL_API_KEY=xxx
        env_prefix="APP_",
        env_nested_delimiter="__", 
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # 配置优先级排序: 参数硬注入 > 环境变量 > .env文件 > YAML文件 > 默认字典
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


# ================= 配置热重载管理器 =================
class ConfigManager:
    """包装 AppConfig 实现 YAML 配置文件的热重载机制"""
    def __init__(self, yaml_path: str = "config/settings.yaml"):
        self.yaml_path = Path(yaml_path)
        self._config = AppConfig()
        self._last_mtime = self._get_mtime()

    def _get_mtime(self) -> float:
        return self.yaml_path.stat().st_mtime if self.yaml_path.exists() else 0.0

    @property
    def config(self) -> AppConfig:
        """获取配置 (属性访问时检测是否需要热重载)"""
        current_mtime = self._get_mtime()
        
        # 精确监测文件修改时间变动
        if current_mtime > self._last_mtime:
            try:
                logger.info(f"检测到配置文件 {self.yaml_path} 已修改，正在重新加载配置...")
                # 重新实例化触发解析链
                AppConfig.model_config["yaml_path"] = str(self.yaml_path) # placeholder if needed
                self._config = AppConfig()
                self._last_mtime = current_mtime
                logger.info("✅ 配置文件重新加载成功")
            except Exception as e:
                logger.error(f"热重载配置失败: {e}，将维持上一次有效配置。")
                
        return self._config

# 单例配置管理对象
global_config_manager = ConfigManager()

# 提供便捷方法，替换旧代码中 import config
def get_config() -> AppConfig:
    return global_config_manager.config
