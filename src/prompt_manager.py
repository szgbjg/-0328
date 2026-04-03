import jinja2
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from .logger import logger

class PromptVersionInfo(BaseModel):
    """Prompt 模板的注册与版本信息"""
    description: str
    version: str
    required_vars: List[str]


class PromptManager:
    """
    Prompt 模板管理系统
    负责 Jinja2 模板加载、变量注入、变量验证及基本版本记录
    """
    
    # 全局 Prompt 注册表
    REGISTRY: Dict[str, PromptVersionInfo] = {
        "question_creator": PromptVersionInfo(
            description="基于上下文生成跟进问题", version="1.0", required_vars=["context"]
        ),
        "facet_planner": PromptVersionInfo(
            description="对用户问题进行多角度分析与拆解", version="1.0", required_vars=["query"]
        ),
        "facet_expander": PromptVersionInfo(
            description="当拆解角度不足时提供角度补充", version="1.0", required_vars=["query", "facets"]
        ),
        "facet_reducer": PromptVersionInfo(
            description="当拆解角度过多时进行角度筛选合并", version="1.0", required_vars=["query", "facets"]
        ),
        "facet_qa_agent": PromptVersionInfo(
            description="结合单一角度及图谱知识生成带 think 的回答", version="1.0", required_vars=["query", "facet", "kg_context"]
        ),
        "redundancy_detector": PromptVersionInfo(
            description="对多个视角的回答进行信息去重清洗", version="1.0", required_vars=["answers"]
        ),
        "synthesis_agent": PromptVersionInfo(
            description="综合所有清洗过的多视角回答生成最终总结", version="1.0", required_vars=["query", "answers"]
        ),
    }

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Jinja2 环境
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            autoescape=False, # LLM Prompt通常不需要 HTML escape
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _validate_vars(self, template_name: str, kwargs: Dict[str, Any]) -> None:
        """验证是否所有必须的变量都被注入"""
        if template_name not in self.REGISTRY:
            logger.warning(f"模板 '{template_name}' 未在注册表中找到，将跳过必须变量检查。")
            return
            
        required = self.REGISTRY[template_name].required_vars
        missing = [var for var in required if var not in kwargs]
        if missing:
            error_msg = f"渲染模板 '{template_name}' 失败，缺少必需变量: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_template_version(self, template_name: str) -> str:
        """获取指定模板的版本号"""
        return self.REGISTRY[template_name].version if template_name in self.REGISTRY else "unknown"

    def render(self, template_name: str, **kwargs) -> str:
        """
        从文件中渲染基于 Jinja2 的 Prompt
        
        Args:
            template_name (str): 模板名称(不含扩展名)
            **kwargs: 注入到模板中的变量及对应的值
            
        Returns:
            str: 渲染后的 Prompt 字符串
        """
        self._validate_vars(template_name, kwargs)
        
        try:
            template = self.env.get_template(f"{template_name}.jinja")
            rendered = template.render(**kwargs)
            logger.debug(f"成功渲染模板 '{template_name}' (v{self.get_template_version(template_name)})")
            return rendered
        except jinja2.TemplateNotFound:
            error_msg = f"找不到 Prompt 模板文件: {self.prompts_dir}/{template_name}.jinja"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"渲染模板 '{template_name}' 发生其他错误: {str(e)}")
            raise

    def render_inline(self, inline_template: str, required_vars: List[str], **kwargs) -> str:
        """
        从代码字符串内联渲染 Prompt
        
        Args:
            inline_template (str): Jinja2 模板内容字符串
            required_vars (List[str]): 必须变量列表
            **kwargs: 注入变量
            
        Returns:
            str: 渲染后的 Prompt 字符串
        """
        missing = [var for var in required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"内联渲染失败，缺少必需变量: {missing}")
            
        template = self.env.from_string(inline_template)
        return template.render(**kwargs)

# 默认的单例管理器对象
prompt_manager = PromptManager()