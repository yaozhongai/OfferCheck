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

from nexa_agent.logger import get_logger
from nexa_agent.config import MODEL_CONFIG, SUPPORTS_THINKING_PARAM, thinking_extra_body, get_model_for_role

logger = get_logger("verifier")


# ==========================================================================
# JSON 提取 / 截断修复（Verifier CoVe 响应容错）
# ==========================================================================

def _extract_json_block(text: str) -> Optional[str]:
    """从 LLM 响应里抠出最外层 JSON 对象。

    容错 markdown 代码围栏（```json ... ```）与前后夹带的说明文字，
    从第一个 `{` 起做括号配对（忽略字符串内的括号）取到匹配的 `}`。
    截断（无匹配闭合）时返回从 `{` 到结尾的整段，交给修复函数补齐。
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    # 未闭合 → 截断，返回整段供修复
    return t[start:]


def _repair_truncated_json(s: str) -> str:
    """补齐被 max_tokens 截断的 JSON：闭合未结束的字符串与括号。

    尽力而为——去掉尾部残缺 token 后，按栈补上缺失的 `]` / `}`。
    修复后仍可能非法（如遗留尾逗号），由调用方 try/except 兜底。
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    repaired = s
    if in_str:
        repaired += '"'
    # 去掉尾部残缺片段（截断在逗号/冒号/未完成键值处）
    repaired = re.sub(r"[,:]\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


# ==========================================================================
# 数据结构
# ==========================================================================

@dataclass
class VerdictResult:
    """Verifier 裁决结果

    status 三态:
      "verified"   — 事实逐条核查通过，来源可靠
      "unverified" — 无可核查的外部事实（纯推理/代码任务），放行但标记
      "failed"     — 发现不可靠来源或来源与事实矛盾，驳回
    """
    status: str  # "verified" | "unverified" | "failed"
    reason: str
    unreliable_facts: list[dict] = field(default_factory=list)
    feedback: str = ""  # 驳回时的具体操作指引

    @property
    def passed(self) -> bool:
        """向后兼容：status != 'failed' 均视为通过"""
        return self.status != "failed"


@dataclass
class OfferVerdict:
    """OfferCheck 阶段裁定的结构化解析结果（供 server/前端裁定卡片渲染）

    对应 stage prompt 要求 Agent 输出的标签体系：
        [Verdict] 靠谱 / 存疑 / 大概率有坑 —— 理由
        [Fact] / [Source] / [Confidence]  （可多条，复用事实核查解析）
        [RedFlag] 红旗（可多条）
        [NeedUserConfirm] 需用户自行确认事项（可多条）
    """
    verdict: str = ""                 # 裁定结论原文
    verdict_level: str = "unknown"    # 归一化: reliable | suspicious | likely_scam | unknown
    facts: list[dict] = field(default_factory=list)   # [{fact, source, confidence}]
    red_flags: list[str] = field(default_factory=list)
    need_user_confirm: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "verdict_level": self.verdict_level,
            "facts": self.facts,
            "red_flags": self.red_flags,
            "need_user_confirm": self.need_user_confirm,
        }


# ==========================================================================
# 事实提取
# ==========================================================================


def _classify_verdict_level(verdict_text: str) -> str:
    """把裁定原文归一化为可枚举的等级，供前端着色/图标

    只信任裁定 label（首个分隔符之前那截）：理由部分常含否定语境关键词
    （如「未发现…诈骗案例」），整行子串匹配会把「靠谱」误判成 likely_scam。
    label 判不出等级时才回退整行匹配兜底。
    """
    label = re.split(r"——|—|：|:|\s-\s", verdict_text.strip(), maxsplit=1)[0]
    level = _match_verdict_keywords(label)
    return level if level != "unknown" else _match_verdict_keywords(verdict_text)


def _match_verdict_keywords(text: str) -> str:
    """有序关键词匹配（scam → suspicious → reliable）"""
    t = text.lower()
    # 大概率有坑 / scam
    if any(k in text for k in ("大概率有坑", "有坑", "诈骗", "骗局", "不推荐")) or \
       any(k in t for k in ("scam", "fraud", "likely scam", "high risk")):
        return "likely_scam"
    # 存疑 / suspicious / 谨慎
    if any(k in text for k in ("存疑", "谨慎", "可疑")) or \
       any(k in t for k in ("suspicious", "caution", "uncertain")):
        return "suspicious"
    # 靠谱 / reliable / 推荐
    if any(k in text for k in ("靠谱", "可靠", "推荐")) or \
       any(k in t for k in ("reliable", "legit", "trustworthy", "safe")):
        return "reliable"
    return "unknown"


def _parse_tag_list(answer: str, tag: str) -> list[str]:
    """提取某个标签的所有出现（每行一条），过滤「无」等空占位"""
    items = []
    for m in re.finditer(rf"\[{tag}\]\s*(.*?)(?:\n|$)", answer, re.IGNORECASE):
        val = m.group(1).strip()
        if val and val not in ("无", "None", "N/A", "-", "（无）", "(无)"):
            items.append(val)
    return items


def parse_offer_verdict(answer: str) -> OfferVerdict:
    """从 Agent 的 Final Answer 中解析 OfferCheck 结构化裁定

    复用 `_parse_facts_from_output` 抽取 [Fact]/[Source]/[Confidence]，
    再补充解析 [Verdict]/[RedFlag]/[NeedUserConfirm]。这样裁定输出与
    既有的事实核查共用同一套结构化协议。

    Args:
        answer: Agent 的 Final Answer 文本

    Returns:
        OfferVerdict（无对应标签时字段为空，不抛错）
    """
    verdict_match = re.search(r"\[Verdict\]\s*(.*?)(?:\n|$)", answer, re.IGNORECASE)
    verdict_text = verdict_match.group(1).strip() if verdict_match else ""

    return OfferVerdict(
        verdict=verdict_text,
        verdict_level=_classify_verdict_level(verdict_text) if verdict_text else "unknown",
        facts=_parse_facts_from_output(answer),
        red_flags=_parse_tag_list(answer, "RedFlag"),
        need_user_confirm=_parse_tag_list(answer, "NeedUserConfirm"),
    )

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
        self.base_url = MODEL_CONFIG["base_url"]
        self.api_key = MODEL_CONFIG["api_key"]
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

    def verify(self, answer: str, task: str, stage: Optional[str] = None) -> VerdictResult:
        """核查 Agent 输出的事实可靠性

        Args:
            answer: Agent 的完整 Final Answer（含数据溯源段落）
            task: 原始用户问题
            stage: 可选的场景阶段标识（如 "stage4"），用于 stage-aware 核查标准

        Returns:
            VerdictResult
        """
        # Step 1: 提取事实
        facts = _parse_facts_from_output(answer)

        if not facts:
            # 没有可核查的事实（纯推理/代码问题），直接放行
            return VerdictResult(
                status="unverified",
                reason="无外部事实数据需要核查",
            )

        # Step 2: 快捷规则检查（零成本）
        quick_reject = self._quick_source_check(facts)
        if quick_reject:
            return quick_reject

        # Step 3: LLM 深度评估（仅对需要的事实）
        return self._llm_verify(facts, task, stage=stage)

    # ── 快捷规则 ──

    def _quick_source_check(self, facts: list[dict]) -> Optional[VerdictResult]:
        """用正则快速识别明显不可靠的来源，零 Token 开销

        只有当 UGC/聚合站是**全部**事实的来源时才直接驳回；混合场景（如 whois/官方
        一手数据为主、Reddit 同类案例仅作佐证——诈骗核查中很常见）交给 LLM 深度评估
        权衡，避免一条补充引用枪毙整个 trial。
        """
        unreliable = []
        for f in facts:
            source = f.get("source", "")
            for pattern, label in self._quick_reject_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    unreliable.append({**f, "reject_reason": label})
                    break

        if unreliable and len(unreliable) >= len(facts):
            return self._build_reject(unreliable, facts)
        if unreliable:
            logger.info("Verifier 快检: %d/%d 条事实引用 UGC 来源，交由 LLM 深度评估权衡",
                        len(unreliable), len(facts))
        return None

    # ── LLM 深度评估 ──

    def _llm_verify(self, facts: list[dict], task: str,
                    stage: Optional[str] = None) -> VerdictResult:
        """用 LLM 评估来源可信度"""
        # 构建精简的核查 prompt（只传事实，不传轨迹）
        facts_text = ""
        for i, f in enumerate(facts, 1):
            facts_text += (
                f"[{i}] 事实: {f['fact'][:200]}\n"
                f"    来源: {f['source'][:300]}\n"
                f"    自评: {f['confidence']}\n\n"
            )

        if stage and ("stage4" in stage or "stage3" in stage):
            source_criteria = """来源可信度判断标准（OfferCheck offer/沟通 证伪专用）:
- 官方网站/官方社交媒体（LinkedIn 官方账号/Twitter 官方/微博官方）→ 可靠
- 主流新闻媒体（搜狐、网易、36Kr、InfoQ 等转载报道）→ 基本可靠（媒体报道，非 UGC）
- Wikipedia/权威媒体 → 基本可靠
- LinkedIn 团队负责人/官方人员发帖 → 高可信度（一手官方信息）
- 企业招聘平台（Boss直聘、拉勾、maimai、mokahr 等）→ 基本可靠
- 论坛/问答网站（知乎、Reddit、Quora）/个人博客/AI聚合站 → 不可靠

放行规则（满足任意一条则通过）:
1. 核心裁定事实（公司是否存在、招聘是否真实）有 ≥2 个基本可靠以上的来源
2. 有 ≥1 个官方一手来源支撑裁定
3. 所有来源交叉一致且无矛盾，且核心来源可信度 Medium 及以上
4. **证伪/负面裁定（"大概率有坑"/Likely a Scam）的特殊规则**：对不存在或冒名的实体，
   **不可能**找到官方来源证明它是假的——"多渠道检索查无官方存在"（无官网/无注册记录/
   无招聘平台记录）**本身就是核心证据**。当裁定为负面且满足以下两点即通过：
   a) 有工具一手数据支撑（如 domain_whois_lookup 返回的注册时间/注册商、多次 web_search
      的"查无结果"）——工具直接返回的数据视为**可靠一手来源**；
   b) 与已知诈骗模式匹配（加密货币付薪、预付费用、仅 Telegram 联系、无面试直录等）。

驳回规则（以下情况才驳回）:
- 全部核心事实的来源都是不可靠来源（UGC/AI聚合站）——注意：工具一手数据（whois/检索结果）
  与"查无"类负面证据**不属于**不可靠来源，不要因此驳回
- 发现来源与事实矛盾（如声称来自官网但实际是论坛帖子）"""
        else:
            source_criteria = """来源可信度判断标准:
- 官方网站/学术期刊/政府数据 → 可靠
- Wikipedia/权威媒体 → 基本可靠
- 论坛/问答网站/个人博客/AI聚合站 → 不可靠

如果存在多个可靠来源交叉验证同一数据，即使其中有一个不可靠来源也应通过。"""

        prompt = f"""你是严格的事实核查员。请对每条事实**独立**判断其来源可靠性（CoVe factored 核查），然后给出总体结论。

【用户问题】: {task[:200]}

【AI 依赖的事实和来源】:
{facts_text}

{source_criteria}

请逐条评判每个事实，然后给出总体结论。如果总体驳回，给出具体的重新搜索指引。

输出 JSON（严格遵守格式，不要输出其他内容）:
{{
  "fact_verdicts": [
    {{"index": 1, "reliable": true/false, "reason": "该条事实来源的评判理由"}},
    ...
  ],
  "overall": "verified|unverified|failed",
  "reason": "总体评判理由（1-2句）",
  "feedback": "如果 overall=failed，给 Agent 的具体搜索指引（去哪里搜、用什么关键词、禁止用什么来源）；否则留空"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_cove_response(response, facts)
            return result
        except Exception as exc:
            logger.error("Verifier LLM 调用失败: %s，默认放行", exc)
            return VerdictResult(status="unverified", reason=f"Verifier 异常，默认放行: {exc}")

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
            # CoVe 逐条核查每条事实产出一个 JSON 对象，6-8 条事实 + 理由
            # 极易超过旧上限 768 → JSON 被截断 → "无法解析，默认放行"。放大到 2048。
            "max_tokens": 2048,
            "temperature": 0.0,
            "stream": False,
        }
        _eb = thinking_extra_body(self.model, enable_thinking=False)
        if _eb:
            kwargs["extra_body"] = _eb

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _parse_response(self, text: str) -> VerdictResult:
        """兼容旧格式 {"passed": ..., "reason": ..., "feedback": ...}"""
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if not data.get("passed", True):
                    return VerdictResult(
                        status="failed",
                        reason=data.get("reason", "来源不可靠"),
                        feedback=data.get("feedback", "请从更权威的来源重新搜索数据。"),
                    )
                return VerdictResult(status="verified", reason=data.get("reason", ""))
            except json.JSONDecodeError:
                pass
        return VerdictResult(status="unverified", reason="Verifier 无法解析，默认放行")

    def _parse_cove_response(self, text: str, facts: list[dict]) -> VerdictResult:
        """解析 CoVe factored 核查响应（逐条事实 + 总体结论）"""
        block = _extract_json_block(text)
        if not block:
            logger.warning("CoVe 响应无法提取 JSON，默认放行: %s", text[:200])
            return VerdictResult(status="unverified", reason="Verifier 无法解析，默认放行")

        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            # 常见于 max_tokens 截断：尝试补齐未闭合的括号后再解析一次
            try:
                data = json.loads(_repair_truncated_json(block))
                logger.info("CoVe JSON 疑似被截断，已修复后解析成功")
            except json.JSONDecodeError:
                logger.warning("CoVe JSON 解析失败（修复后仍失败），默认放行: %s", text[:200])
                return VerdictResult(status="unverified", reason="Verifier 无法解析，默认放行")

        overall = data.get("overall", "verified")
        reason = data.get("reason", "")
        feedback = data.get("feedback", "")

        # 收集被判为不可靠的逐条事实
        fact_verdicts = data.get("fact_verdicts", [])
        unreliable_facts = []
        for fv in fact_verdicts:
            idx = fv.get("index", 0)
            if not fv.get("reliable", True) and 1 <= idx <= len(facts):
                fact = facts[idx - 1]
                unreliable_facts.append({
                    **fact,
                    "reject_reason": fv.get("reason", "来源不可靠"),
                })
                logger.debug("CoVe 逐条驳回 [%d]: %s — %s", idx, fact.get("source", ""), fv.get("reason", ""))

        if overall == "failed":
            logger.info("CoVe 总体驳回: %s（不可靠事实数=%d）", reason[:100], len(unreliable_facts))
            return VerdictResult(
                status="failed",
                reason=reason,
                unreliable_facts=unreliable_facts,
                feedback=feedback or "请从更权威的来源重新搜索数据。",
            )

        status = "verified" if overall == "verified" else "unverified"
        if unreliable_facts:
            logger.info("CoVe 总体放行（%s），但有 %d 条次要来源不可靠", status, len(unreliable_facts))
        return VerdictResult(status=status, reason=reason, unreliable_facts=unreliable_facts)

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
            status="failed",
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
