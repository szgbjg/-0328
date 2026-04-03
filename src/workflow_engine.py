import asyncio
import json
import random
import time
import functools
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .logger import logger
from .retry_handler import async_retry
from .validators.minimal_validator import SimpleValidator
from .validators.content_guard import check_banned_words, sanitize_output
from .validators.evidence_minimal import EvidenceValidator
from .utils.evidence_formatter import format_paragraph_with_source, format_final_answer


# ================ 数据模型 ================
class RoundState(BaseModel):
    """单轮问答状态模型"""
    round_num: int
    context: str
    question: Optional[str] = None
    raw_facets: Optional[List[str]] = None
    processed_facets: Optional[List[str]] = None
    facet_answers: Optional[List[str]] = None
    dedup_answers: Optional[List[str]] = None
    final_response: Optional[str] = None
    is_completed: bool = False

class SessionState(BaseModel):
    """整个会话的运行状态"""
    session_id: str
    seed_context: str
    current_round: int = 1
    total_rounds: int = 0
    rounds_data: List[RoundState] = Field(default_factory=list)


# ================ 性能监控装饰器 ================
def monitor_performance(func):
    """监控异步函数的执行时间"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"开始执行步骤: {func.__name__} ...")
        
        try:
            result = await func(*args, **kwargs)
            cost_time = time.time() - start_time
            logger.info(f"✅ 步骤 {func.__name__} 执行完毕, 耗时 {cost_time:.2f} 秒")
            return result
        except Exception as e:
            cost_time = time.time() - start_time
            logger.error(f"❌ 步骤 {func.__name__} 执行失败, 耗时 {cost_time:.2f} 秒. 错误: {str(e)}")
            raise
    return wrapper


# ================ 核心工作流引擎 ================
class MedicalQAWorkflow:
    """
    医药多轮问答生成工作流引擎
    支持：状态机驱动、独立重试、节点回溯、断点续传、并发调用
    """
    STEPS = ["QUESTION", "FACET", "QA", "DEDUP", "SYNTHESIS"]

    def __init__(self, session_id: str, max_backtracks: int = 3):
        self.session_id = session_id
        self.max_backtracks = max_backtracks
        self.checkpoints_dir = Path("data/checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.state: Optional[SessionState] = None
        self.simple_validator = SimpleValidator()
        self.evidence_validator = EvidenceValidator()

    # ---------- 断点续传与状态管理 ----------
    def save_checkpoint(self):
        """保存当前会话状态到 JSON"""
        if not self.state:
            return
        filepath = self.checkpoints_dir / f"{self.session_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.state.model_dump_json(indent=2))
        logger.debug(f"已保存进度断点至 {filepath}")

    def load_checkpoint(self) -> bool:
        """尝试加载现有的会话状态"""
        filepath = self.checkpoints_dir / f"{self.session_id}.json"
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state = SessionState.model_validate(data)
                logger.info(f"成功加载会话断点: {self.session_id}，恢复至第 {self.state.current_round} 轮")
                return True
            except Exception as e:
                logger.error(f"加载断点失败: {str(e)}")
        return False

    # ---------- 各处理节点 (模拟智能体行为) ----------
    @async_retry(max_attempts=3)
    @monitor_performance
    async def _question_creator(self, context: str) -> str:
        """生成跟进问题"""
        await asyncio.sleep(0.5)  # 模拟网络调用延迟
        return f"基于[{context[:10]}...]的新问题: 药物相互作用是什么？"

    @async_retry(max_attempts=3)
    @monitor_performance
    async def _facet_planner(self, question: str) -> List[str]:
        """规划回答维度的角度"""
        await asyncio.sleep(0.5)
        # 随机产生 1 到 10 个维度的视角，用于验证后续的校验逻辑
        num_facets = random.randint(1, 10)
        return [f"角度_{i+1}" for i in range(num_facets)]

    def _preprocess_facets(self, facets: List[str]) -> List[str]:
        """角度预处理逻辑：>8筛选，<2补充"""
        logger.debug(f"预处理前的角度数量: {len(facets)}")
        if len(facets) > 8:
            logger.info("角度大于8个，执行筛选截断只保留前8个有效角度")
            facets = facets[:8]
        elif len(facets) < 2:
            logger.info("角度少于2个，执行补充策略凑齐2个")
            missing = 2 - len(facets)
            facets.extend([f"补充角度_补{i+1}" for i in range(missing)])
        logger.debug(f"预处理后的角度数量: {len(facets)}")
        return facets

    @async_retry(max_attempts=2) # 内部单个QA调用也支持重试
    async def _single_facet_qa(self, question: str, facet: str, temperature: float = 0.7) -> str:
        """处理单角度问答"""
        await asyncio.sleep(max(0.2, random.random()))
        # 随机模拟少概率失败
        if random.random() < 0.05:
            raise ValueError(f"针对角度 '{facet}' 的问答生成意外中断")
        return (
            f"<think>\n<facet={facet}>\n证据清单: [证据R1:来源=refs]\n</think>\n"
            f"【{facet}】的回答内容"
        )

    @monitor_performance
    async def _facet_qa_agent(self, question: str, facets: List[str]) -> List[str]:
        """并发并行生成多角度对应的回答"""
        available_refs = [
            {"id": "R1", "source": "《药品说明书》", "content": "禁忌与不良反应"},
            {"id": "R2", "source": "《临床指南》", "content": "风险与建议"},
            {"id": "G1", "source": "图谱关系", "content": "实体关系"},
        ]
        answers: List[str] = []

        for facet in facets:
            raw_output = await self._single_facet_qa(question, facet, temperature=0.7)

            banned = check_banned_words(raw_output)
            if banned:
                logger.warning(f"Facet '{facet}' 检测到禁用词: {banned}")
                fixed_output = sanitize_output(raw_output, banned)
                if check_banned_words(fixed_output):
                    logger.warning(f"Facet '{facet}' 自动修正后仍违规，准备低温重试一次(temperature=0.3)")
                    retry_output = await self._single_facet_qa(question, facet, temperature=0.3)
                    banned = check_banned_words(retry_output)
                    if banned:
                        logger.error(f"Facet '{facet}' 重试后仍有禁用词: {banned}，丢弃该facet")
                        continue
                    logger.info(f"Facet '{facet}' 低温重试通过内容安全检查")
                    raw_output = retry_output
                else:
                    logger.info(f"Facet '{facet}' 已通过自动修正清理禁用词")
                    raw_output = fixed_output

            ev = self.evidence_validator.validate(raw_output, available_refs)
            if not ev["is_valid"] and ev.get("fixed_text"):
                logger.warning(f"Facet '{facet}' 证据链尝试自动修正: {ev['errors']}")
                raw_output = ev["fixed_text"]
                ev = self.evidence_validator.validate(raw_output, available_refs)
            if not ev["is_valid"]:
                logger.error(f"Facet '{facet}' 证据链无法修复，丢弃该facet: {ev['errors']}")
                continue

            vr = self.simple_validator.validate(
                raw_output,
                expected_facet=facet,
                banned_words=[],
                require_think_tags=True,
                require_evidence_chain=True,
            )
            if not vr["is_valid"]:
                logger.error(f"Facet '{facet}' 格式校验失败，丢弃该facet: {vr['errors']}")
                continue

            thinking, body = raw_output, ""
            if "</think>" in raw_output:
                thinking, body = raw_output.split("</think>", 1)
                thinking = thinking + "</think>"
                body = body.strip()
            for ref in available_refs:
                if ref["id"] in ev.get("found", []):
                    body = format_paragraph_with_source(body, ref["source"])
            answers.append(format_final_answer(thinking, body))

        return answers

    @async_retry(max_attempts=3)
    @monitor_performance
    async def _redundancy_detector(self, answers: List[str]) -> List[str]:
        """结果去重清洗"""
        await asyncio.sleep(0.3)
        # 模拟去重功能
        return list(set(answers))

    @async_retry(max_attempts=3)
    @monitor_performance
    async def _synthesis_agent(self, question: str, dedup_answers: List[str]) -> str:
        """合成最终答案"""
        await asyncio.sleep(0.5)
        combined = " | ".join(dedup_answers)
        return f"针对【{question}】的综合总结结果包含：{combined}"

    # ---------- 回溯机制流水线管理 ----------
    async def generate_round(self, context: str, round_num: int) -> RoundState:
        """
        生成单轮多角度问答
        包含按步骤流转及失败时回溯机制 (Fallback to previous step)
        """
        logger.info(f"========= 开始生成第 {round_num} 轮 =========")
        round_state = RoundState(round_num=round_num, context=context)
        
        step_idx = 0
        backtrack_count = 0

        while step_idx < len(self.STEPS):
            step_name = self.STEPS[step_idx]
            try:
                if step_name == "QUESTION":
                    if not round_state.question:
                        round_state.question = await self._question_creator(context)
                
                elif step_name == "FACET":
                    if not round_state.processed_facets:
                        raw_facets = await self._facet_planner(round_state.question)
                        round_state.raw_facets = raw_facets
                        round_state.processed_facets = self._preprocess_facets(raw_facets)
                
                elif step_name == "QA":
                    if not round_state.facet_answers:
                        round_state.facet_answers = await self._facet_qa_agent(
                            round_state.question, round_state.processed_facets
                        )
                
                elif step_name == "DEDUP":
                    if not round_state.dedup_answers:
                        round_state.dedup_answers = await self._redundancy_detector(round_state.facet_answers)
                
                elif step_name == "SYNTHESIS":
                    if not round_state.final_response:
                        round_state.final_response = await self._synthesis_agent(
                            round_state.question, round_state.dedup_answers
                        )

                # 将进度落地并步入下一步
                self.save_checkpoint()
                step_idx += 1
                
            except Exception as e:
                backtrack_count += 1
                logger.warning(f"节点 {step_name} 执行失败: {str(e)}。当前累计回溯次数: {backtrack_count}")
                
                if backtrack_count > self.max_backtracks:
                    logger.error(f"第 {round_num} 轮重试回溯次数耗尽，流程中止。")
                    raise RuntimeError(f"Workflow failed after {self.max_backtracks} backtracks.") from e
                
                # 清除当前步骤和上一步的数据缓存
                if step_name == "QUESTION":
                    round_state.question = None
                    step_idx = max(0, step_idx - 1)
                elif step_name == "FACET":
                    round_state.raw_facets = None
                    round_state.processed_facets = None
                    round_state.question = None  # 回退到上一个节点，甚至清理上一个节点结果以重新生成
                    step_idx = max(0, step_idx - 1)
                elif step_name == "QA":
                    round_state.facet_answers = None
                    round_state.raw_facets = None
                    round_state.processed_facets = None
                    step_idx = max(0, step_idx - 1)
                elif step_name == "DEDUP":
                    round_state.dedup_answers = None
                    round_state.facet_answers = None
                    step_idx = max(0, step_idx - 1)
                elif step_name == "SYNTHESIS":
                    round_state.final_response = None
                    round_state.dedup_answers = None
                    step_idx = max(0, step_idx - 1)

                logger.info(f"已回溯。准备重试节点: {self.STEPS[step_idx]}")
                # 等待片刻再重试
                await asyncio.sleep(1)
        
        round_state.is_completed = True
        self.save_checkpoint()
        logger.info(f"========= 第 {round_num} 轮生成成功 =========")
        return round_state

    # ---------- 多轮生成主入口 ----------
    async def generate_multi_round(self, seed_context: str, min_round: int = 3, max_round: int = 8) -> SessionState:
        """
        基于初始上下文生成多轮对话
        支持中间失败后断点续传
        """
        # 尝恢复断点
        if not self.load_checkpoint():
            total_rounds = random.randint(min_round, max_round)
            self.state = SessionState(
                session_id=self.session_id,
                seed_context=seed_context,
                total_rounds=total_rounds,
                current_round=1
            )
            self.save_checkpoint()
            logger.info(f"新建会话 {self.session_id}，计划生成总轮数: {total_rounds}")

        current_context = self.state.seed_context

        # 如果已有历史轮次，需要将 context 推推演到最新的那一侧，确保上下文平滑
        for completed_round in self.state.rounds_data:
            if completed_round.is_completed:
                # 把上一轮最终回答拼接为下一轮上下文 (简化逻辑)
                current_context = f"{current_context} \n {completed_round.final_response}"

        while self.state.current_round <= self.state.total_rounds:
            rr = self.state.current_round
            # 生成该轮问答结果
            round_result = await self.generate_round(current_context, rr)
            
            # 保存到会话列表
            self.state.rounds_data.append(round_result)
            
            # 更新上下文，模拟多轮连续关联
            current_context = f"{current_context} \n 机器人: {round_result.final_response}"
            
            self.state.current_round += 1
            self.save_checkpoint()
            
        logger.info(f"🎉 会话 {self.session_id} 所有多轮 ({self.state.total_rounds} 轮) 生成完毕 ! 🎉")
        return self.state
