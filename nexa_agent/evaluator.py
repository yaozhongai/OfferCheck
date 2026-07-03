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

from nexa_agent.logger import get_logger
from nexa_agent.config import get_model_for_role, MODEL_CONFIG, SUPPORTS_THINKING_PARAM

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


def _detect_tool_gap(trajectory: str) -> Optional[dict]:
    """检测工具能力缺口：Agent 反复尝试读取同一资源但无法获取有效内容

    典型场景：PDF 链接用 web_fetch/tavily_extract 反复读取但拿到乱码。
    判定条件：同一 URL 被 ≥2 种不同工具访问过，且都未产生有效结果。
    """
    # 提取所有 URL 参数（从 Action 和 tool call 中）
    url_pattern = r"https?://[^\s\)\]\},\"'<>]+"
    urls_in_actions = re.findall(url_pattern, trajectory)
    if not urls_in_actions:
        return None

    # 统计每个 URL 被访问的次数
    url_counter = Counter()
    for url in urls_in_actions:
        # 归一化：去掉 query string 和 fragment
        base_url = url.split("?")[0].split("#")[0].rstrip("/")
        url_counter[base_url] += 1

    # 找出被访问 ≥3 次的 URL
    repeated_urls = {url: cnt for url, cnt in url_counter.items() if cnt >= 3}
    if not repeated_urls:
        return None

    # 检查这些 URL 是否是 PDF（.pdf 后缀）
    pdf_urls = [u for u in repeated_urls if u.lower().endswith(".pdf")]

    # 检查答案中是否包含"无法"相关表述
    inability_markers = ["无法", "不支持", "cannot", "unable", "乱码", "无效"]
    traj_lower = trajectory.lower()
    has_inability = any(m in traj_lower for m in inability_markers)

    if pdf_urls:
        top_url = max(pdf_urls, key=lambda u: repeated_urls[u])
        return {
            "failure_mode": "tool_gap",
            "reason": f"PDF '{top_url[-60:]}' 被访问 {repeated_urls[top_url]} 次但未获取有效内容"
                      f"（可能需要 read_pdf 工具）",
            "severity": "high",
        }

    if has_inability and repeated_urls:
        top_url = max(repeated_urls, key=lambda u: repeated_urls[u])
        return {
            "failure_mode": "tool_gap",
            "reason": f"URL '{top_url[-60:]}' 被访问 {repeated_urls[top_url]} 次，"
                      f"Agent 可能缺少处理该资源类型的工具",
            "severity": "medium",
        }

    return None


# 启发式规则列表（按优先级排序）
HEURISTIC_RULES = [
    ("context_overflow", _detect_context_overflow),
    ("tool_gap", _detect_tool_gap),
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
        # base_url/api_key 必须与引擎 provider 一致（MODEL_CONFIG，唯一真源）。
        # 旧默认写死 DEEPSEEK_BASE_URL（官方端点），而模型名是 GMI 风格 →
        # 官方 API 不认 → 每次 400 静默回退启发式，LLM Judge 名存实亡。
        self.base_url = base_url or MODEL_CONFIG["base_url"]
        self.api_key = api_key or MODEL_CONFIG["api_key"]
        logger.info("Evaluator 初始化 mode=%s model=%s base_url=%s",
                    mode, self.model_name, self.base_url)

    # ── 主入口 ──

    def evaluate(
        self,
        task: str,
        answer: str,
        trajectory: str,
        terminated_reason: str = "",
        stage: Optional[str] = None,
    ) -> EvalResult:
        """评估 ReAct 输出质量

        Args:
            task: 原始任务描述
            answer: ReAct 最终答案
            trajectory: 完整推理轨迹
            terminated_reason: 终止原因 (final_answer | max_steps | llm_error | parse_error)
            stage: 可选的场景阶段标识（如 "stage4"），用于 stage-aware 评估

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
            return self._llm_evaluate(task, answer, trajectory, stage=stage)
        elif self.mode == "hybrid":
            return self._hybrid_evaluate(task, answer, trajectory, terminated_reason, stage=stage)

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
        self, task: str, answer: str, trajectory: str, stage: Optional[str] = None,
    ) -> EvalResult:
        """使用 LLM 评估答案质量"""
        prompt = self._build_llm_eval_prompt(task, answer, trajectory, stage=stage)

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_llm_eval_response(response)

            success = parsed.get("success", False)
            reason = parsed.get("reason", "")
            logger.info("LLM 评估结果: success=%s reason=%s failure_mode=%s",
                        success, reason[:200], parsed.get("failure_mode"))

            return EvalResult(
                success=success,
                confidence=parsed.get("confidence", 0.7),
                reason=reason,
                feedback_signal=parsed.get("feedback", ""),
                failure_mode=parsed.get("failure_mode") if not success else None,
                llm_result=parsed,
            )
        except Exception as exc:
            logger.error("LLM 评估失败: %s，回退到启发式评估", exc)
            return self._heuristic_evaluate(trajectory, answer, "")

    def _build_llm_eval_prompt(self, task: str, answer: str, trajectory: str,
                               stage: Optional[str] = None) -> str:
        """构建 LLM 评估 prompt"""
        # 截断轨迹（取首尾关键部分）
        traj_summary = trajectory[:1500] if len(trajectory) <= 2000 else (
            trajectory[:800] + "\n...(中间省略)...\n" + trajectory[-700:]
        )

        if stage and "stage4" in stage:
            criteria = """评估标准（OfferCheck Stage 4 — offer 证伪专用）：
1. 裁定是否给出三态之一：「靠谱」「存疑」「大概率有坑」？（必须有，否则失败）
2. 裁定是否绑定了真实检索到的证据？（而非凭空判断）
3. 是否识别并列出了 red_flags，或明确说明「无」？
4. 是否包含 need_user_confirm（用户需自行确认的事项）？
5. 答案是否合理——「靠谱」「存疑」「大概率有坑」都是合法的正确裁定，不能因裁定结论本身判为失败。
   注意：「大概率有坑」在证据支持下是最严重但完全合法的裁定。

以下情况才算失败：
- 完全没有给出裁定（空答案 / 只有过程无结论）
- 编造了从未检索过的来源（轨迹中没有对应工具调用）
- 裁定与证据明显矛盾（如证据全部支持靠谱却裁定有坑，反之亦然）"""
        else:
            criteria = """评估标准：
1. 答案是否直接回应了用户的问题核心？
2. 答案是否有实际内容支撑（而非空泛陈述）？
3. 答案中是否存在明显的事实性错误或自相矛盾？
4. 答案长度是否足够（一般应超过 20 个字符）？"""

        return f"""你是一个严格的任务评估专家。请判断以下智能体的回答是否充分解决了用户的问题。

【用户任务】: {task}

【智能体最终答案】: {answer}

【推理轨迹摘要】:
{traj_summary}

{criteria}

请输出 JSON 格式（不要输出其他内容）：
{{"success": true/false, "confidence": 0.0~1.0, "reason": "简要判定理由", "feedback": "如果失败，给出具体反馈；如果成功，留空", "failure_mode": "如果失败，选择: context_overflow/premature_answer/loop/tool_misuse/wrong_reasoning/null"}}\
"""

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

    def _has_substantive_answer(self, answer: str, terminated_reason: str) -> bool:
        """判断 Agent 是否给出了实质性答案（而非错误消息或空答案）"""
        if terminated_reason in ("llm_error", "parse_error"):
            return False
        if not answer or len(answer.strip()) < 15:
            return False
        if answer.startswith("[错误]"):
            return False
        return True

    def _hybrid_evaluate(
        self, task: str, answer: str, trajectory: str, terminated_reason: str,
        stage: Optional[str] = None,
    ) -> EvalResult:
        """结果优先的混合评估

        核心原则：过程问题是 warning，不是 veto。只有结果问题才能否决。
        - 有实质性答案 → 先 LLM 评估答案质量，过程问题只降置信度
        - 无答案/错误终止 → 走启发式流程诊断失败原因
        """
        heuristic = self._heuristic_evaluate(trajectory, answer, terminated_reason)

        # ── 结果优先：有实质性答案时，先评估答案质量 ──
        if self._has_substantive_answer(answer, terminated_reason):
            logger.info("Hybrid: 检测到实质性答案 (len=%d)，启动 LLM 答案质量评估 stage=%s...",
                        len(answer), stage or "generic")
            llm = self._llm_evaluate(task, answer, trajectory, stage=stage)

            if llm.success:
                # 答案通过：过程问题记为 warning 但不否决
                confidence = llm.confidence
                reason = llm.reason
                if not heuristic.success:
                    confidence *= 0.85  # 过程有问题则降低置信度
                    reason = f"答案质量通过（过程存在 {heuristic.failure_mode} 问题）: {llm.reason}"
                    logger.info("Hybrid: 答案通过，过程有 %s，置信度降为 %.2f",
                                heuristic.failure_mode, confidence)
                else:
                    reason = f"启发式 + LLM 双重确认通过: {llm.reason}"

                return EvalResult(
                    success=True,
                    confidence=confidence,
                    reason=reason,
                    heuristic_result=heuristic.__dict__ if hasattr(heuristic, '__dict__') else None,
                    llm_result=llm.llm_result,
                )
            else:
                # LLM 认为答案质量不行 → 结合过程诊断返回失败
                failure_mode = llm.failure_mode or heuristic.failure_mode or "wrong_reasoning"
                return EvalResult(
                    success=False,
                    confidence=llm.confidence,
                    reason=f"LLM 判定答案质量不足: {llm.reason}",
                    feedback_signal=llm.feedback_signal,
                    failure_mode=failure_mode,
                    heuristic_result=heuristic.__dict__ if hasattr(heuristic, '__dict__') else None,
                    llm_result=llm.llm_result,
                )

        # ── 无实质性答案：走启发式诊断 ──
        if not heuristic.success:
            # 高严重度 → 直接返回（没有答案可评估，无需 LLM）
            high_severity_modes = {"loop", "tool_misuse", "context_overflow", "tool_gap"}
            if heuristic.failure_mode in high_severity_modes:
                logger.info("Hybrid: 无实质答案 + 启发式失败 (%s)，直接返回", heuristic.failure_mode)
                return heuristic

            # 低严重度 → LLM 复审
            logger.info("Hybrid: 无实质答案 + 启发式低严重度 (%s)，LLM 复审...", heuristic.failure_mode)
            llm = self._llm_evaluate(task, answer, trajectory, stage=stage)
            if llm.success:
                return EvalResult(
                    success=True,
                    confidence=llm.confidence * 0.8,
                    reason=f"启发式存疑但 LLM 复审通过: {llm.reason}",
                    heuristic_result={"heuristic_failed": heuristic.failure_mode},
                    llm_result=llm.llm_result,
                )
            return llm

        # 启发式通过但无实质答案 → LLM 二次确认
        logger.info("Hybrid: 启发式通过，启动 LLM 二次确认...")
        llm = self._llm_evaluate(task, answer, trajectory, stage=stage)
        if llm.success:
            return EvalResult(
                success=True,
                confidence=max(heuristic.confidence, llm.confidence),
                reason=f"启发式 + LLM 双重确认通过: {llm.reason}",
                heuristic_result=heuristic.__dict__ if hasattr(heuristic, '__dict__') else None,
                llm_result=llm.llm_result,
            )
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

        # DeepSeek thinking 关闭（仅官方 API 支持该私有参数；GMI 收到会 422）
        if SUPPORTS_THINKING_PARAM and "deepseek" in self.model_name.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ==========================================================================
# 便捷函数
# ==========================================================================

def create_evaluator(mode: str = "hybrid") -> Evaluator:
    """创建评估器的便捷函数"""
    return Evaluator(mode=mode)
