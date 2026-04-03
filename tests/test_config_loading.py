from pathlib import Path

from src.config_loader import ConfigManager


def test_config_manager_uses_defaults_when_yaml_missing(tmp_path: Path):
    manager = ConfigManager(yaml_path=str(tmp_path / "missing.yaml"))
    cfg = manager.config

    assert cfg.api.model_endpoint
    assert cfg.workflow.max_concurrent_requests == 10
    assert cfg.workflow.checkpoint_dir == "data/checkpoints"


def test_config_manager_loads_yaml_override(tmp_path: Path):
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        """
workflow:
  max_backtracks: 7
  max_concurrent_requests: 4
""".strip(),
        encoding="utf-8",
    )

    manager = ConfigManager(yaml_path=str(yaml_path))
    cfg = manager.config

    assert cfg.workflow.max_backtracks == 7
    assert cfg.workflow.max_concurrent_requests == 4