# 医药知识图谱多轮问答系统 (Medical KG QA System)

本项目提供可直接运行的 demo 入口，支持本地 Python 与 Docker 两种方式。

## 快速开始（本地）

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备配置

```bash
cp .env.example .env
cp config/settings.example.yaml config/settings.yaml
```

3. 单轮生成（CLI）

```bash
python -m src.cli generate_single --context "患者主诉：头晕恶心，服药后无改善" --output output/cli_single.json
```

输出文件：`output/cli_single.json`

4. 多轮 demo（脚本）

```bash
python test_run.py
```

输出文件：`output/test_round.json`

## Docker 运行

1. 启动容器

```bash
docker compose up -d
```

2. 在容器里运行单轮 CLI

```bash
docker compose exec medical_qa_app python -m src.cli generate_single --context "患者主诉：头晕恶心，服药后无改善" --output output/cli_single.json
```

3. 在容器里运行多轮 demo

```bash
docker compose exec medical_qa_app python test_run.py
```

容器内输出同样位于 `output/`，并映射到宿主机项目目录。

## 配置说明

- 环境变量使用前缀 `APP_`，嵌套字段用双下划线：
  - `APP_API__MODEL_API_KEY`
  - `APP_API__KG_API_KEY`
- YAML 默认路径是 `config/settings.yaml`。
- 可用 `APP_CONFIG_YAML` 指定自定义 YAML 路径。
- 配置优先级：初始化参数 > 环境变量 > `.env` > YAML > 默认值。

## 最小验证

```bash
pytest tests/test_config_loading.py -q
```
