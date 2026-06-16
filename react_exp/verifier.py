"""
Verifier Agent — 事实网关（方案 B）

从 Agent 的结构化输出中提取 [Fact]/[Source]/[Confidence] 键值对，
调用 LLM 评估来源可信度。不读完整搜索轨迹，只审关键事实。

设计原则:
- 只读格式化输出，Token 消耗极低
- 输出 pass/fail + 具象操作指引（不是抽象错误）
- 使用 fast 模型，成本敏感
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

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

logger = get_logger("verifier")


# ==========================================================================
# 数据结构
# ==========================================================================

@dataclass
class VerdictResult:
    """Verifier 裁决结果"""
    passed: bool
    reason: str
    unreliable_facts: list[dict] = field(default_factory=list)
    feedback: str = ""  # 驳回时的具体操作指引


# ==========================================================================
# 事实提取
# ==========================================================================

def _parse_facts_from_output(answer: str) -> list[dict]:
    """从 Agent 输出中提取 [Fact]/[Source]/[Confidence] 三元组

    支持两种格式:
    1. 标准格式: [Fact] ... [Source] ... [Confidence] ...
    2. 降级格式: 直接从文本中提取关键数字和来源引用

    Returns:
        [{"fact": "...", "source": "...", "confidence": "..."}, ...]
    """
    facts = []

    # 尝试解析标准 [Fact]/[Source]/[Confidence] 格式
    fact_blocks = re.split(r"\n(?=\[Fact\])", answer)
    for block in fact_blocks:
        fact_match = re.search(r"\[Fact\]\s*(.*?)(?:\n|$)", block, re.IGNORECASE)
        source_match = re.search(r"\[Source\]\s*(.*?)(?:\n|$)", block, re.IGNORECASE)
        conf_match = re.search(r"\[Confidence\]\s*(.*?)(?:\n|$)", block, re.IGNORECASE)

        if fact_match:
            facts.append({
                "fact": fact_match.group(1).strip(),
                "source": source_match.group(1).strip() if source_match else "未标注",
                "confidence": conf_match.group(1).strip() if conf_match else "Unstated",
            })

    # 如果标准格式没有匹配到，尝试从 URL 引用中提取
    if not facts:
        urls = re.findall(r'https?://[^\s\)\]>,"\']+', answer)
        if urls:
            last_url = urls[-1] if urls else ""
            # 尝试在 URL 附近找数字/数据
            for url in urls:
                idx = answer.find(url)
                context = answer[max(0, idx - 150):idx + len(url) + 50]
                facts.append({
                    "fact": f"数据来自 {url}",
                    "source": url,
                    "confidence": "Unstated",
                })

    return facts


# ==========================================================================
# VerifierAgent
# ==========================================================================

class VerifierAgent:
    """事实核查网关

    只在行为评估器通过后才被调用。从 Agent 的结构化输出中提取事实，
    判断来源可信度。驳回时给出具象操作指引。

    Usage::

        verifier = VerifierAgent()
        verdict = verifier.verify(
            answer=agent_final_answer,
            task=original_task,
        )
        if not verdict.passed:
            print(verdict.feedback)  # 传给反思生成器
    """

    def __init__(self):
        self.model = get_model_for_role("evaluator_llm")  # fast 模型
        self.base_url = os.environ.get(
            "DEEPSEEK_BASE_URL",
            os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com"),
        )
        self.api_key = os.environ.get(
            "DEEPSEEK_API_KEY",
            os.environ.get("KIMI_API_KEY", ""),
        )
        # 来源可信度规则（不需要 LLM 就能判断的快捷规则）
        self._quick_reject_patterns = [
            (r"quora\.com", "Quora（问答网站，UGC内容）"),
            (r"zhihu\.com", "知乎（问答网站，UGC内容）"),
            (r"reddit\.com", "Reddit（论坛，UGC内容）"),
            (r"fiqueligadonews\.com\.br", "疑似AI生成的聚合站"),
            (r"voltologo\.net", "疑似AI生成的聚合站"),
        ]
        logger.info("VerifierAgent 初始化 model=%s", self.model)

    # ── 主入口 ──

    def verify(self, answer: str, task: str) -> VerdictResult:
        """核查 Agent 输出的事实可靠性

        Args:
            answer: Agent 的完整 Final Answer（含数据溯源段落）
            task: 原始用户问题

        Returns:
            VerdictResult
        """
        # Step 1: 提取事实
        facts = _parse_facts_from_output(answer)

        if not facts:
            # 没有可核查的事实（纯推理/代码问题），直接放行
            return VerdictResult(
                passed=True,
                reason="无外部事实数据需要核查",
            )

        # Step 2: 快捷规则检查（零成本）
        quick_reject = self._quick_source_check(facts)
        if quick_reject:
            return quick_reject

        # Step 3: LLM 深度评估（仅对需要的事实）
        return self._llm_verify(facts, task)

    # ── 快捷规则 ──

    def _quick_source_check(self, facts: list[dict]) -> Optional[VerdictResult]:
        """用正则快速识别明显不可靠的来源，零 Token 开销"""
        unreliable = []
        for f in facts:
            source = f.get("source", "")
            for pattern, label in self._quick_reject_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    unreliable.append({**f, "reject_reason": label})

        if unreliable:
            return self._build_reject(unreliable, facts)
        return None

    # ── LLM 深度评估 ──

    def _llm_verify(self, facts: list[dict], task: str) -> VerdictResult:
        """用 LLM 评估来源可信度"""
        # 构建精简的核查 prompt（只传事实，不传轨迹）
        facts_text = ""
        for i, f in enumerate(facts, 1):
            facts_text += (
                f"[{i}] 事实: {f['fact'][:200]}\n"
                f"    来源: {f['source'][:300]}\n"
                f"    自评: {f['confidence']}\n\n"
            )

        prompt = f"""你是严格的事实核查员。请判断以下 AI 在回答问题时使用的数据来源是否可靠。

【用户问题】: {task[:200]}

【AI 依赖的事实和来源】:
{facts_text}

请逐条判断每个来源的可信度:
- 官方网站/学术期刊/政府数据 → 可靠
- Wikipedia/权威媒体 → 基本可靠
- 论坛/问答网站/个人博客/AI聚合站 → 不可靠

如果发现任何关键事实来自不可靠来源，请驳回并给出具体的重新搜索指引。
如果存在多个可靠来源交叉验证同一数据，即使其中有一个不可靠来源也应通过。

输出 JSON:
{{"passed": true/false, "reason": "...", "feedback": "如果驳回，给 Agent 的具体搜索指引（必须包含: 去哪里搜、搜什么关键词、禁止用什么来源）"}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            return result
        except Exception as exc:
            logger.error("Verifier LLM 调用失败: %s，默认放行", exc)
            return VerdictResult(passed=True, reason=f"Verifier 异常，默认放行: {exc}")

    def _call_llm(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
        )

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是事实核查员。只输出 JSON，不要输出其他内容。"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        }
        if "deepseek" in self.model.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _parse_response(self, text: str) -> VerdictResult:
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if not data.get("passed", True):
                    return VerdictResult(
                        passed=False,
                        reason=data.get("reason", "来源不可靠"),
                        feedback=data.get("feedback", "请从更权威的来源重新搜索数据。"),
                    )
                return VerdictResult(passed=True, reason=data.get("reason", ""))
            except json.JSONDecodeError:
                pass
        return VerdictResult(passed=True, reason="Verifier 无法解析，默认放行")

    # ── 构建驳回 ──

    def _build_reject(
        self, unreliable: list[dict], all_facts: list[dict],
    ) -> VerdictResult:
        """构建驳回结果，含具体的重新搜索指引"""
        # 列出被驳回的事实
        rejected_items = []
        bad_sources = set()
        for f in unreliable:
            rejected_items.append(f"- '{f['fact'][:80]}' —— 来自 {f.get('reject_reason', '不可靠来源')}")
            bad_sources.add(f.get("reject_reason", "不可靠来源"))

        # 构建反馈
        bad_source_list = "、".join(bad_sources)
        feedback = (
            f"你在回答中使用了来自 {bad_source_list} 的数据。"
            f"这类来源属于用户生成内容或低质量聚合站，数据未经验证，置信度极低。\n\n"
            f"请重新组织搜索策略：\n"
            f"1. 直接前往数据原始出处（官方网站、学术数据库、政府统计局）\n"
            f"2. 搜索关键词应包含 \"official\"、\"annual report\"、\"statistics\" 等\n"
            f"3. 严禁使用 Quora/知乎/Reddit/论坛/第三方聚合站作为数据支撑\n"
        )

        return VerdictResult(
            passed=False,
            reason=f"不可靠来源: {', '.join(bad_sources)}",
            unreliable_facts=unreliable,
            feedback=feedback,
        )


# ==========================================================================
# 动态触发判断
# ==========================================================================

def should_trigger_verifier(
    task: str,
    trial_details: list[dict],
    current_trial: int,
) -> bool:
    """判断是否需要在当前 Trial 触发 Verifier

    触发条件（满足任意一条即触发）:
    1. 意图路由: 问题包含统计数据、年份、财报等关键词
    2. 行为触发: 上轮 Trial 步数 ≥ 5 或触发过 loop/tool_error
    3. 记忆触发: 上轮 Trial 被 Verifier 因 unreliable_source 驳回

    Args:
        task: 原始问题
        trial_details: 历史 Trial 详情
        current_trial: 当前 Trial 编号

    Returns:
        True 表示应触发 Verifier
    """
    # 条件 1: 意图路由 — 包含需要事实核查的关键词
    fact_keywords = [
        "统计", "数据", "数字", "数量", "多少", "几年", "哪年", "年份",
        "统计", "报告", "财报", "GDP", "收入", "价格", "比例", "百分比",
        "number of", "how many", "how much", "what year", "statistics",
        "published", "annual", "per year", "per month",
        "p-value", "significance", "articles", "papers",
    ]
    task_lower = task.lower()
    intent_triggered = any(kw.lower() in task_lower for kw in fact_keywords)

    # 条件 2: 行为触发 — 历史 Trial 有异常
    behavior_triggered = False
    if trial_details:
        last_trial = trial_details[-1]
        steps = last_trial.get("steps_used", 0)
        failure_mode = last_trial.get("failure_mode", "")
        if steps >= 5:
            behavior_triggered = True
        if failure_mode in ("loop", "tool_misuse"):
            behavior_triggered = True

    # 条件 3: 记忆触发 — 上次被 Verifier 驳回
    memory_triggered = False
    if trial_details:
        last_trial = trial_details[-1]
        if last_trial.get("failure_mode") == "unreliable_source":
            memory_triggered = True

    triggered = intent_triggered or behavior_triggered or memory_triggered

    logger.debug(
        "Verifier trigger check: intent=%s behavior=%s memory=%s → %s",
        intent_triggered, behavior_triggered, memory_triggered,
        "FIRE" if triggered else "SKIP",
    )
    return triggered
