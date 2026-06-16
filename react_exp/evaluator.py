"""
Reflexion 多策略评估器

判断 ReAct 输出是否满足任务需求，支持三种模式:
- heuristic: 启发式规则检测（零成本，毫秒级）
- llm: LLM 评估（高质量，消耗 token）
- hybrid: 混合模式（默认，先启发式后 LLM 确认）
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# 加载 .env
try:
    from dotenv import load_dotenv

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from react_exp.logger_config import get_logger
from react_exp.config import get_model_for_role

logger = get_logger("evaluator")


# ==========================================================================
# 数据结构
# ==========================================================================

@dataclass
class EvalResult:
    """评估结果

    Attributes:
        success: 是否成功
        confidence: 置信度 [0, 1]
        reason: 判定理由
        feedback_signal: 反馈信号（传给反思模型）
        failure_mode: 失败模式分类（仅失败时有效）
        heuristic_result: 启发式评估的原始结果
        llm_result: LLM 评估的原始结果
    """
    success: bool
    confidence: float
    reason: str
    feedback_signal: str = ""
    failure_mode: Optional[str] = None
    heuristic_result: Optional[dict] = None
    llm_result: Optional[dict] = None


# ==========================================================================
# 启发式失败规则
# ==========================================================================

# 不确定性标志短语
UNCERTAINTY_PHRASES = [
    "我不确定", "我不太确定", "我无法确定", "可能", "也许",
    "I'm not sure", "I am not sure", "uncertain",
    "无法回答", "没有足够信息", "信息不足",
]


def _detect_context_overflow(trajectory: str) -> Optional[dict]:
    """检测是否达到最大步数但未给出有效答案"""
    if "达到最大步数" in trajectory:
        return {
            "failure_mode": "context_overflow",
            "reason": "达到最大步数限制，未能完成任务",
            "severity": "high",
        }
    return None


def _detect_repeated_actions(trajectory: str) -> Optional[dict]:
    """检测轨迹中是否出现重复的工具调用（死循环）

    两层检测:
    1. 相同工具+相同参数（规范化后）≥3 次 → 死循环
    2. 相同工具名 ≥15 次（不同参数也计数）→ 搜索策略失效
    """
    # 提取所有 (tool_name, args) 对，args 做规范化（去引号、去首尾空白）
    action_pattern = r"Action\s*[:：]\s*(\w+)\s*\(\s*(.*?)\s*\)"
    raw_matches = re.findall(action_pattern, trajectory, re.IGNORECASE)
    if not raw_matches:
        return None

    # 规范化 args：去掉首尾引号、归一化空白
    def _normalize_args(a: str) -> str:
        a = a.strip()
        if len(a) >= 2 and a[0] == a[-1] and a[0] in ('"', "'"):
            a = a[1:-1]
        return a.strip()

    matches = [(name, _normalize_args(args)) for name, args in raw_matches]

    # 检测 1: 相同工具+相同参数（规范化后）重复 ≥3 次
    pair_counter = Counter(matches)
    for (tool_name, args), count in pair_counter.items():
        if count >= 3:
            args_short = args[:60] if len(args) > 60 else args
            return {
                "failure_mode": "loop",
                "reason": f"工具 '{tool_name}({args_short})' 被重复调用 {count} 次，陷入循环",
                "severity": "high",
            }

    # 检测 2: 同工具名过多（不同参数，疑似搜索无果持续换词）
    name_counter = Counter(name for name, _ in matches)
    for name, count in name_counter.items():
        if count >= 15:
            return {
                "failure_mode": "loop",
                "reason": f"工具 '{name}' 被调用 {count} 次（不同参数），搜索策略可能失效",
                "severity": "high",
            }

    return None


def _detect_tool_errors(trajectory: str) -> Optional[dict]:
    """检测连续工具调用错误"""
    error_count = len(re.findall(r"\[错误\]", trajectory))
    if error_count >= 3:
        return {
            "failure_mode": "tool_misuse",
            "reason": f"轨迹中出现 {error_count} 次工具错误",
            "severity": "high",
        }
    return None


def _detect_uncertainty(answer: str) -> Optional[dict]:
    """检测答案中是否有不确定性标志"""
    for phrase in UNCERTAINTY_PHRASES:
        if phrase.lower() in answer.lower():
            return {
                "failure_mode": "wrong_reasoning",
                "reason": f"答案包含不确定性标志: '{phrase}'",
                "severity": "low",
            }
    return None


# 启发式规则列表（按优先级排序）
HEURISTIC_RULES = [
    ("context_overflow", _detect_context_overflow),
    ("loop", _detect_repeated_actions),
    ("tool_misuse", _detect_tool_errors),
    ("wrong_reasoning", _detect_uncertainty),
]


# ==========================================================================
# 评估器
# ==========================================================================

class Evaluator:
    """多策略评估器

    Usage::

        evaluator = Evaluator(mode="hybrid")
        result = evaluator.evaluate(
            task="2024年全球GDP排名",
            answer="美国第一，中国第二...",
            trajectory="Thought: ... Action: web_search(...)...",
        )
        if not result.success:
            print(f"失败模式: {result.failure_mode}")
            print(f"反馈: {result.feedback_signal}")
    """

    def __init__(
        self,
        mode: str = "hybrid",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """初始化评估器

        Args:
            mode: 评估模式 — "heuristic" | "llm" | "hybrid"
            model_name: LLM 评估使用的模型（默认与 ReAct 相同）
            base_url: API 地址
            api_key: API Key
        """
        if mode not in ("heuristic", "llm", "hybrid"):
            raise ValueError(f"不支持的评估模式: {mode}，可选: heuristic, llm, hybrid")

        self.mode = mode
        self.model_name = model_name or get_model_for_role("evaluator_llm")
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL",
            os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com"),
        )
        self.api_key = api_key or os.environ.get(
            "DEEPSEEK_API_KEY",
            os.environ.get("KIMI_API_KEY", ""),
        )
        logger.info("Evaluator 初始化 mode=%s model=%s", mode, self.model_name)

    # ── 主入口 ──

    def evaluate(
        self,
        task: str,
        answer: str,
        trajectory: str,
        terminated_reason: str = "",
    ) -> EvalResult:
        """评估 ReAct 输出质量

        Args:
            task: 原始任务描述
            answer: ReAct 最终答案
            trajectory: 完整推理轨迹
            terminated_reason: 终止原因 (final_answer | max_steps | llm_error | parse_error)

        Returns:
            EvalResult
        """
        # 如果是 LLM 错误或解析错误，直接判定失败
        if terminated_reason in ("llm_error", "parse_error"):
            return EvalResult(
                success=False,
                confidence=0.95,
                reason=f"执行异常终止: {terminated_reason}",
                feedback_signal=f"Agent 执行过程中发生错误: {terminated_reason}。"
                               f"错误信息: {answer[:200]}",
                failure_mode=terminated_reason,
            )

        if self.mode == "heuristic":
            return self._heuristic_evaluate(trajectory, answer, terminated_reason)
        elif self.mode == "llm":
            return self._llm_evaluate(task, answer, trajectory)
        elif self.mode == "hybrid":
            return self._hybrid_evaluate(task, answer, trajectory, terminated_reason)

    # ── 启发式评估 ──

    def _heuristic_evaluate(
        self, trajectory: str, answer: str, terminated_reason: str,
    ) -> EvalResult:
        """运行启发式规则，按优先级检测第一个失败信号"""
        for rule_name, rule_func in HEURISTIC_RULES:
            if rule_name == "context_overflow":
                result = rule_func(trajectory)
            else:
                result = rule_func(trajectory if rule_name in ("loop", "tool_misuse") else answer)

            if result:
                return EvalResult(
                    success=False,
                    confidence=0.85,
                    reason=result["reason"],
                    feedback_signal=f"启发式检测 [{result['failure_mode']}]: {result['reason']}",
                    failure_mode=result["failure_mode"],
                    heuristic_result=result,
                )

        # 特殊检查：达到 max_steps 但终止原因标记为 max_steps
        if terminated_reason == "max_steps":
            return EvalResult(
                success=False,
                confidence=0.7,
                reason="达到最大步数限制",
                feedback_signal="Agent 在最大步数内未能完成任务，可能需要更多步骤或不同的搜索策略。",
                failure_mode="context_overflow",
            )

        return EvalResult(
            success=True,
            confidence=0.6,
            reason="启发式规则未检测到明显失败信号",
            feedback_signal="",
        )

    # ── LLM 评估 ──

    def _llm_evaluate(
        self, task: str, answer: str, trajectory: str,
    ) -> EvalResult:
        """使用 LLM 评估答案质量"""
        prompt = self._build_llm_eval_prompt(task, answer, trajectory)

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_llm_eval_response(response)

            return EvalResult(
                success=parsed.get("success", False),
                confidence=parsed.get("confidence", 0.7),
                reason=parsed.get("reason", ""),
                feedback_signal=parsed.get("feedback", ""),
                failure_mode=parsed.get("failure_mode") if not parsed.get("success") else None,
                llm_result=parsed,
            )
        except Exception as exc:
            logger.error("LLM 评估失败: %s，回退到启发式评估", exc)
            return self._heuristic_evaluate(trajectory, answer, "")

    def _build_llm_eval_prompt(self, task: str, answer: str, trajectory: str) -> str:
        """构建 LLM 评估 prompt"""
        # 截断轨迹（取首尾关键部分）
        traj_summary = trajectory[:1500] if len(trajectory) <= 2000 else (
            trajectory[:800] + "\n...(中间省略)...\n" + trajectory[-700:]
        )

        return f"""你是一个严格的任务评估专家。请判断以下智能体的回答是否充分解决了用户的问题。

【用户任务】: {task}

【智能体最终答案】: {answer}

【推理轨迹摘要】:
{traj_summary}

评估标准：
1. 答案是否直接回应了用户的问题核心？
2. 答案是否有实际内容支撑（而非空泛陈述）？
3. 答案中是否存在明显的事实性错误或自相矛盾？
4. 答案长度是否足够（一般应超过 20 个字符）？

请输出 JSON 格式（不要输出其他内容）：
{{"success": true/false, "confidence": 0.0~1.0, "reason": "简要判定理由", "feedback": "如果失败，给出具体反馈；如果成功，留空", "failure_mode": "如果失败，选择: context_overflow/premature_answer/loop/tool_misuse/wrong_reasoning/null"}}"""

    def _parse_llm_eval_response(self, response: str) -> dict:
        """解析 LLM 评估的 JSON 响应"""
        # 尝试提取 JSON 块
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 兜底：文本判断
        response_lower = response.lower()
        if any(w in response_lower for w in ["成功", "success", "正确", "充分"]):
            if not any(w in response_lower for w in ["但是", "然而", "however", "失败", "fail"]):
                return {"success": True, "confidence": 0.7, "reason": response[:200], "feedback": ""}

        return {"success": False, "confidence": 0.5, "reason": response[:200],
                "feedback": response[:300], "failure_mode": "wrong_reasoning"}

    # ── 混合评估 ──

    def _hybrid_evaluate(
        self, task: str, answer: str, trajectory: str, terminated_reason: str,
    ) -> EvalResult:
        """先跑启发式 → 高严重度失败直接返回 → 低严重度/通过走 LLM 确认"""
        heuristic = self._heuristic_evaluate(trajectory, answer, terminated_reason)

        if not heuristic.success:
            # 高严重度失败（loop / tool_misuse / context_overflow）→ 直接返回，不浪费 LLM 调用
            high_severity_modes = {"loop", "tool_misuse", "context_overflow"}
            if heuristic.failure_mode in high_severity_modes:
                logger.info("Hybrid: 启发式判定失败 (%s, high)，跳过 LLM", heuristic.failure_mode)
                return heuristic

            # 低严重度失败（premature_answer / wrong_reasoning）→ LLM 复审
            logger.info("Hybrid: 启发式判定失败 (%s, low)，启动 LLM 复审...", heuristic.failure_mode)
            llm = self._llm_evaluate(task, answer, trajectory)
            if llm.success:
                return EvalResult(
                    success=True,
                    confidence=llm.confidence,
                    reason=f"启发式存疑但 LLM 复审通过: {llm.reason}",
                    heuristic_result={"heuristic_failed": heuristic.failure_mode},
                    llm_result=llm.llm_result,
                )
            return llm

        # 启发式通过，用 LLM 二次确认
        logger.info("Hybrid: 启发式通过，启动 LLM 二次确认...")
        llm = self._llm_evaluate(task, answer, trajectory)

        # 如果 LLM 也通过，成功
        if llm.success:
            return EvalResult(
                success=True,
                confidence=max(heuristic.confidence, llm.confidence),
                reason=f"启发式 + LLM 双重确认通过: {llm.reason}",
                heuristic_result=heuristic.__dict__ if hasattr(heuristic, '__dict__') else None,
                llm_result=llm.llm_result,
            )

        # LLM 认为失败，以 LLM 为准
        return EvalResult(
            success=False,
            confidence=llm.confidence,
            reason=f"LLM 评估判定失败: {llm.reason}",
            feedback_signal=llm.feedback_signal,
            failure_mode=llm.failure_mode or heuristic.failure_mode,
            heuristic_result=heuristic.__dict__ if hasattr(heuristic, '__dict__') else None,
            llm_result=llm.llm_result,
        )

    # ── LLM 调用 ──

    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM 进行评估"""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
        )

        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个严格的任务评估专家。请只输出 JSON，不要输出其他内容。"},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        }

        # DeepSeek thinking 关闭
        if "deepseek" in self.model_name.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ==========================================================================
# 便捷函数
# ==========================================================================

def create_evaluator(mode: str = "hybrid") -> Evaluator:
    """创建评估器的便捷函数"""
    return Evaluator(mode=mode)
