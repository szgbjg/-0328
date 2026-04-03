# 医药知识图谱多轮问答系统 (Medical KG QA System)

基于医药知识图谱的多轮问答系统基础框架。

## 项目结构
- `src/`: 核心源代码模块
- `config/`: 配置文件目录 (YAML, JSON等)
- `tests/`: 单元测试与集成测试
- `docs/`: 项目文档
- `data/`: 数据存储目录

## 核心模块
- `api_client.py`: 异步HTTP API客户端封装，包含对KG和LLM的调用支持。
- `config_manager.py`: 基于 Pydantic 的配置管理，支持环境变量。
- `logger.py`: 基于 loguru 的日志记录。
- `retry_handler.py`: 基于 tenacity 的通用重试与回溯机制。
- `parallel_processor.py`: 基于 asyncio 的并发任务处理工具。

## 安装与运行
1. 克隆代码后安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 复制环境变量文件并配置：
   ```bash
   cp .env.example .env
   ```
3. 运行主程序（待实现）。
