import os
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .logger import logger


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


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Load nested settings from YAML and merge with defaults/env settings."""

    def __init__(self, settings_cls: Type[BaseSettings], yaml_path: str):
        super().__init__(settings_cls)
        self.yaml_path = Path(yaml_path)

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        raise NotImplementedError

    def __call__(self) -> Dict[str, Any]:
        if not self.yaml_path.exists():
            return {}

        try:
            with self.yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                logger.warning(
                    f"YAML config should be a mapping, got {type(data)}. Ignore file: {self.yaml_path}"
                )
                return {}
            return data
        except Exception as exc:
            logger.warning(f"Failed to parse YAML config {self.yaml_path}: {exc}. Falling back to defaults/env.")
            return {}


class AppConfig(BaseSettings):
    api: APIConfig = Field(default_factory=APIConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_path="config/settings.yaml",
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
        yaml_path = str(settings_cls.model_config.get("yaml_path", "config/settings.yaml"))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_path=yaml_path),
            file_secret_settings,
        )


class ConfigManager:
    """Manage app config and support YAML hot reload by file mtime."""

    def __init__(self, yaml_path: str = "config/settings.yaml"):
        self.yaml_path = Path(yaml_path)
        self._config = self._load_config()
        self._last_mtime = self._get_mtime()

    def _build_runtime_config_cls(self) -> Type[AppConfig]:
        base_model_config = dict(AppConfig.model_config)
        base_model_config["yaml_path"] = str(self.yaml_path)

        class RuntimeAppConfig(AppConfig):
            model_config = SettingsConfigDict(**base_model_config)

        return RuntimeAppConfig

    def _load_config(self) -> AppConfig:
        runtime_cls = self._build_runtime_config_cls()
        return runtime_cls()

    def _get_mtime(self) -> float:
        return self.yaml_path.stat().st_mtime if self.yaml_path.exists() else 0.0

    @property
    def config(self) -> AppConfig:
        current_mtime = self._get_mtime()
        if current_mtime > self._last_mtime:
            try:
                logger.info(f"Detected config file change, reloading: {self.yaml_path}")
                self._config = self._load_config()
                self._last_mtime = current_mtime
                logger.info("Config reloaded successfully")
            except Exception as exc:
                logger.error(f"Config reload failed: {exc}. Keeping previous valid config.")
        return self._config


global_config_manager = ConfigManager(yaml_path=os.getenv("APP_CONFIG_YAML", "config/settings.yaml"))


def get_config() -> AppConfig:
    return global_config_manager.config
