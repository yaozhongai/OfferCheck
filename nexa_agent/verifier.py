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
from nexa_agent.config import get_model_for_role
from nexa_agent.util.json_extract import extract_json_block, repair_truncated_json
from nexa_agent.llm_gateway import complete as llm_complete

logger = get_logger("verifier")


# ==========================================================================
# JSON 提取 / 截断修复：已抽到共享工具 nexa_agent.util.json_extract（评审 1.7）。
# 保留模块内别名（下划线私有名）供本文件与既有引用继续使用，行为完全一致。
# ==========================================================================
_extract_json_block = extract_json_block
_repair_truncated_json = repair_truncated_json


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


# ==========================================================================
# 事实提取
# ==========================================================================


# 裁定 label 与理由之间的分隔符（label-first 分类的唯一真源，eval 评分器/AIS 降级共用）。
# 必须覆盖各种破折号变体：此前漏了 en dash（–）——模型输出混合标签
# "Suspicious – Likely a Scam" 时切不开 label，回退全文有序匹配被 scam 词升档成
# likely_scam；同一标签换 em dash 却判 suspicious（分类结果取决于模型碰巧用哪种
# 破折号）。补齐 – / ―，混合标签一律按 label-first 取首段（模型领句的那档）。
VERDICT_SEP_PATTERN = r"——|—|–|―|：|:|\s-\s"


def _classify_verdict_level(verdict_text: str) -> str:
    """把裁定原文归一化为可枚举的等级，供前端着色/图标

    只信任裁定 label（首个分隔符之前那截）：理由部分常含否定语境关键词
    （如「未发现…诈骗案例」），整行子串匹配会把「靠谱」误判成 likely_scam。
    label 判不出等级时才回退整行匹配兜底。
    """
    label = re.split(VERDICT_SEP_PATTERN, verdict_text.strip(), maxsplit=1)[0]
    level = _match_verdict_keywords(label)
    return level if level != "unknown" else _match_verdict_keywords(verdict_text)


def _match_verdict_keywords(text: str) -> str:
    """有序关键词匹配（scam → suspicious → reliable）"""
    t = text.lower()
    # 大概率有坑 / scam（"not recommended"/"recommend skipping" 必须先于 reliable 组
    # 的 "recommended" 测——子串包含）
    if any(k in text for k in ("大概率有坑", "有坑", "诈骗", "骗局", "不推荐", "建议放弃")) or \
       any(k in t for k in ("scam", "fraud", "likely scam", "high risk", "not recommended", "recommend skipping")):
        return "likely_scam"
    # 存疑 / suspicious / 谨慎
    if any(k in text for k in ("存疑", "谨慎", "可疑")) or \
       any(k in t for k in ("suspicious", "caution", "uncertain")):
        return "suspicious"
    # 靠谱 / reliable / 推荐（英文 stage1 档位 "Recommended"/"Worth Applying" 同组）
    if any(k in text for k in ("靠谱", "可靠", "推荐", "值得投递")) or \
       any(k in t for k in ("reliable", "legit", "trustworthy", "safe", "recommended", "worth applying")):
        return "reliable"
    return "unknown"


# ── entailment 内容核查辅助（评审 2.2）──

_URL_IN_TEXT = re.compile(r'https?://[^\s\)\]>,"\']+')
_BARE_DOMAIN = re.compile(r'\b(?:[a-z0-9][a-z0-9-]*\.)+[a-z]{2,}\b', re.IGNORECASE)


def _source_domain(source: str) -> str:
    """从一条 [Source] 文本里提取域名（先找 URL，再找裸域名，都无则空）。"""
    if not source:
        return ""
    m = _URL_IN_TEXT.search(source)
    if m:
        from urllib.parse import urlparse
        return urlparse(m.group(0)).netloc.lower().removeprefix("www.")
    m2 = _BARE_DOMAIN.search(source)
    return m2.group(0).lower().removeprefix("www.") if m2 else ""


def _match_evidence(source_domain: str, evidence: dict) -> str:
    """按域名（含父/子域宽松匹配）取该来源检索到的正文摘录；无则空串。"""
    if not source_domain or not evidence:
        return ""
    if source_domain in evidence:
        return evidence[source_domain]
    for dom, text in evidence.items():
        if dom and (dom.endswith("." + source_domain) or source_domain.endswith("." + dom)):
            return text
    return ""


def _relevant_window(text: str, fact: str, size: int = 300) -> str:
    """从正文摘录里截取与事实断言重叠度最高的 size 字窗口（评审 P0 D2）。

    此前对长正文只取前 size 字——搜索列表头/页面导航区几乎不含断言，judge 必然「查无」
    而误判 misattribution。改为按事实关键词在正文里滑窗打分，取命中最多的一段，让这
    有限的 size 字真正围绕断言，把「内容核实」做到点子上。
    """
    if len(text) <= size:
        return text
    kws = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", fact.lower()))
    for seg in re.findall(r"[一-鿿]{2,}", fact):  # CJK 双字 gram
        for i in range(len(seg) - 1):
            kws.add(seg[i:i + 2])
    if not kws:
        return text[:size]
    tl = text.lower()
    step = max(50, size // 4)
    best_pos, best_score = 0, -1
    for pos in range(0, len(text) - size + 1, step):
        score = sum(tl.count(k, pos, pos + size) for k in kws)
        if score > best_score:
            best_score, best_pos = score, pos
    if best_score <= 0:
        return text[:size]
    start = max(0, best_pos - size // 6)  # 给命中处一点前文
    return text[start:start + size]


# OfferCheck 调查证伪阶段：选岗(1)/沟通(3)/offer(4) 都是「证伪/查真」任务，媒体报道、
# UGC 佐证、以及「多渠道查无」这类负面证据都是合法证据链（评审 P0 D4：此前只认 3/4，
# stage1 选岗调研误落进通用严格分支 → reddit/反诈博客被判不可靠而误杀）。
_OFFERCHECK_INVESTIGATION_STAGES = ("stage1", "stage3", "stage4")


def _source_criteria(stage: Optional[str]) -> str:
    """按 stage 返回来源可信度判断标准（评审 3.8 stage-aware + P0 D4）。"""
    if stage and any(s in stage for s in _OFFERCHECK_INVESTIGATION_STAGES):
        return """来源可信度判断标准（OfferCheck 调查证伪专用——选岗/沟通/offer）:
- 官方网站/官方社交媒体（LinkedIn 官方账号/Twitter 官方/微博官方）→ 可靠
- 主流新闻媒体（搜狐、网易、36Kr、InfoQ 等转载报道）→ 基本可靠（媒体报道，非 UGC）
- Wikipedia/权威媒体 → 基本可靠
- LinkedIn 团队负责人/官方人员发帖 → 高可信度（一手官方信息）
- 企业招聘平台（Boss直聘、拉勾、maimai、mokahr、greenhouse 等）→ 基本可靠
- 论坛/问答/反诈预警帖（知乎、Reddit、Quora、反诈博客）→ **本身不可靠，但作为「已有人反映此类诈骗」
  的旁证是合理的**：在证伪/预警语境下引用它们佐证「存在同类诈骗模式」不应因此驳回整体裁定

放行规则（满足任意一条则通过）:
1. 核心裁定事实（公司是否存在、招聘/岗位是否真实、健康度）有 ≥2 个基本可靠以上的来源
2. 有 ≥1 个官方一手来源支撑裁定
3. 所有来源交叉一致且无矛盾，且核心来源可信度 Medium 及以上
4. **证伪/负面裁定（"大概率有坑"/"谨慎"/Likely a Scam）的特殊规则**：对不存在或冒名的实体，
   **不可能**找到官方来源证明它是假的——"多渠道检索查无官方存在"（无官网/无注册记录/
   无招聘平台记录）**本身就是核心证据**。当裁定为负面且满足以下两点即通过：
   a) 有工具一手数据支撑（如 domain_whois_lookup 返回的注册时间/注册商、多次 web_search
      的"查无结果"）——工具直接返回的数据视为**可靠一手来源**；
   b) 与已知诈骗模式匹配（加密货币付薪、预付费用、仅 Telegram 联系、无面试直录等）。

驳回规则（以下情况才驳回）:
- 全部核心事实的来源都是不可靠来源（UGC/AI聚合站）——注意：工具一手数据（whois/检索结果）
  与"查无"类负面证据**不属于**不可靠来源，不要因此驳回
- 发现来源与事实**矛盾**（如声称来自官网但实际是论坛帖子；数值/名称对不上）"""
    return """来源可信度判断标准:
- 官方网站/学术期刊/政府数据 → 可靠
- Wikipedia/权威媒体 → 基本可靠
- 论坛/问答网站/个人博客/AI聚合站 → 不可靠

如果存在多个可靠来源交叉验证同一数据，即使其中有一个不可靠来源也应通过。"""


def _norm_supported(v) -> Optional[str]:
    """把 judge 的 supported 值归一到三态（评审 P0 D3）。

    只有 `contradicted`（摘录与断言直接矛盾）才是真 misattribution、才硬毙；`no_evidence`
    （摘录截断没提到）是常态噪声、不驳回。对旧格式布尔容错：True→yes、False→no_evidence
    （**保守**，避免把「没提到」误当「造假」——这正是本次误杀的根因）。
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no_evidence"
    s = str(v).strip().lower()
    if s.startswith("contradict"):
        return "contradicted"
    if s in ("yes", "true", "supported", "support"):
        return "yes"
    return "no_evidence"  # no_evidence / unknown / no / 其它一律作「未提及」


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
            source = source_match.group(1).strip() if source_match else ""
            if not source:
                # 无 [Source] 标签但事实文本里内嵌了 URL → 取该 URL 作来源（B′ 配套：
                # 别把「来源写在句子里」的事实误判成「未标注」，后续 AIS/entailment 照常核验）
                m_url = _URL_IN_TEXT.search(fact_match.group(1))
                source = m_url.group(0) if m_url else "未标注"
            facts.append({
                "fact": fact_match.group(1).strip(),
                "source": source,
                "confidence": conf_match.group(1).strip() if conf_match else "Unstated",
            })

    # 如果标准格式没有匹配到，尝试从 URL 引用中提取
    if not facts:
        urls = re.findall(r'https?://[^\s\)\]>,"\']+', answer)
        for url in urls:
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

    def verify(self, answer: str, task: str, stage: Optional[str] = None,
               evidence: Optional[dict] = None) -> VerdictResult:
        """核查 Agent 输出的事实可靠性

        Args:
            answer: Agent 的完整 Final Answer（含数据溯源段落）
            task: 原始用户问题
            stage: 可选的场景阶段标识（如 "stage4"），用于 stage-aware 核查标准
            evidence: 可选的「域名 → 该来源检索到的正文摘录」映射（评审 2.2）。
                      有它时对「来源真实但内容不支持断言」（misattribution）做 entailment 核查——
                      这是 AIS 存在性对账抓不到的幻觉形态。

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

        # Step 3: LLM 深度评估（来源可信度 + 可选的 entailment 内容核实）
        return self._llm_verify(facts, task, stage=stage, evidence=evidence)

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
                    stage: Optional[str] = None,
                    evidence: Optional[dict] = None) -> VerdictResult:
        """用 LLM 评估来源可信度 + （有 evidence 时）内容 entailment 核查"""
        # 构建精简的核查 prompt。有检索正文摘录的事实附上摘录，供判断「内容是否支持断言」。
        # 控体量（评审 2.2 修）：GMI 上大 prompt 偶发返回空 content——摘录 ≤300 字、
        # 最多给 4 条事实附摘录，把总 prompt 压在安全区（实测 4k 字符可用、9k 字符会空返回）。
        _MAX_ENTAIL = 4
        _EXCERPT_CHARS = 300
        facts_text = ""
        entailment_idxs: list[int] = []  # 附了正文摘录、需判 supported 的事实序号
        for i, f in enumerate(facts, 1):
            facts_text += (
                f"[{i}] 事实: {f['fact'][:200]}\n"
                f"    来源: {f['source'][:150]}\n"
                f"    自评: {f['confidence']}\n"
            )
            if len(entailment_idxs) < _MAX_ENTAIL:
                excerpt = _match_evidence(_source_domain(f.get("source", "")), evidence or {})
                if excerpt:
                    entailment_idxs.append(i)
                    # D2：取与断言重叠度最高的窗口，而非机械取前 300 字（否则命中率极低）
                    window = _relevant_window(excerpt, f["fact"], _EXCERPT_CHARS)
                    facts_text += f"    已检索到该来源的正文摘录（核对断言是否属实）: {window}\n"
            facts_text += "\n"

        source_criteria = _source_criteria(stage)  # D4：stage1/3/4 走宽松、其余严格

        entailment_note = ""
        if entailment_idxs:
            # D3：三态判定——只有「摘录直接矛盾」才是真 misattribution 硬毙；「300 字里没提到」
            # （no_evidence）是摘录截断的常态噪声，不作驳回理由（至多降低置信）。
            entailment_note = (
                "\n**内容核实（entailment，三态判定，重点）**：部分事实附了『已检索到该来源的正文摘录』"
                "（这只是原文的一小段截断窗口）。对这些事实判断 `supported`：\n"
                '  - `"yes"`：摘录明确支持该断言；\n'
                '  - `"contradicted"`：摘录**直接与断言矛盾**（如声称来自官网、摘录却是论坛帖；'
                "数值/名称对不上）——这才是真 misattribution；\n"
                '  - `"no_evidence"`：摘录里既不支持也不矛盾（只是这一小段没提到）——**这是常态**，'
                "**不等于造假**；\n"
                "  - 未附摘录的事实 `supported=null`。\n"
                "**只有 `contradicted` 才判该事实不可信并影响总体**；`no_evidence` 绝不作为驳回理由。\n"
            )

        prompt = f"""你是严格的事实核查员。请对每条事实**独立**判断（CoVe factored 核查），然后给出总体结论。

【用户问题】: {task[:200]}

【AI 依赖的事实和来源】:
{facts_text}

{source_criteria}
{entailment_note}
请逐条评判每个事实，然后给出总体结论。如果总体驳回，给出具体的重新搜索指引。

输出 JSON（严格遵守格式，不要输出其他内容）:
{{
  "fact_verdicts": [
    {{"index": 1, "reliable": true/false, "supported": "yes"|"no_evidence"|"contradicted"|null, "reason": "该条事实来源可信度 + 内容与摘录的关系（支持/矛盾/未提及）"}},
    ...
  ],
  "overall": "verified|unverified|failed",
  "reason": "总体评判理由（1-2句）",
  "feedback": "如果 overall=failed，给 Agent 的具体搜索指引（去哪里搜、用什么关键词、禁止用什么来源，或指出哪条断言与来源不符）；否则留空"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_cove_response(response, facts, entailment_idxs)
            return result
        except Exception as exc:
            # fail-safe（评审 1.9）：重试仍失败 → 不硬拦（避免核查服务宕机时全线阻塞），
            # 但如实标注「核查服务不可用」而非「默认放行」，避免让上层误以为已核实通过。
            logger.error("Verifier LLM 调用失败（重试后）: %s", exc)
            return VerdictResult(
                status="unverified",
                reason=f"核查服务暂不可用（非「已核实」结论）: {str(exc)[:150]}",
                feedback="事实核查未能执行，本次裁定未经独立核验，请谨慎采信。",
            )

    def _call_llm(self, prompt: str) -> str:
        # 统一走 LLM Gateway（评审 3.1）：client/thinking/retry/空响应重试/记账集中管理。
        # max_tokens=2048：CoVe 逐条核查 6-8 条事实易超小上限被截断。
        # retry_on_empty：GMI 大 prompt（带 entailment 摘录）偶发空返回（评审 2.2）。
        result = llm_complete(
            [
                {"role": "system", "content": "你是事实核查员。只输出 JSON，不要输出其他内容。"},
                {"role": "user", "content": prompt},
            ],
            model=self.model, max_tokens=2048, temperature=0.0,
            timeout=30.0, retry_on_empty=True,
        )
        return result.content

    def _parse_cove_response(self, text: str, facts: list[dict],
                             entailment_idxs: Optional[list] = None) -> VerdictResult:
        """解析 CoVe factored 核查响应（逐条事实 + 总体结论）

        entailment_idxs：附了正文摘录、要求判 supported 的事实序号。这些事实里任何一条
        被判 supported=false（misattribution）→ 强制 failed，即使 LLM 的 overall 放行。
        """
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
        entail_set = set(entailment_idxs or [])

        # 收集被判为不可靠 / 内容与来源矛盾（misattribution）的逐条事实
        fact_verdicts = data.get("fact_verdicts", [])
        unreliable_facts = []
        misattributed = []  # supported=contradicted 的 entailment 事实（真 misattribution）
        for fv in fact_verdicts:
            idx = fv.get("index", 0)
            if not (1 <= idx <= len(facts)):
                continue
            fact = facts[idx - 1]
            bad_source = not fv.get("reliable", True)
            # D3 三态：只有 contradicted 才是真 misattribution；no_evidence（摘录截断没提到）
            # 是常态噪声，绝不硬毙。只对做过 entailment 的事实认定。
            contradicted = idx in entail_set and _norm_supported(fv.get("supported")) == "contradicted"
            if bad_source or contradicted:
                why = fv.get("reason", "来源不可靠")
                if contradicted:
                    why = f"[内容矛盾/misattribution] {why}"
                    misattributed.append(idx)
                unreliable_facts.append({**fact, "reject_reason": why})
                logger.debug("CoVe 逐条驳回 [%d] bad_source=%s contradicted=%s: %s",
                             idx, bad_source, contradicted, fact.get("source", ""))

        # entailment 硬规则（评审 2.2 + P0 D3 校准）：任何核心事实内容与来源**直接矛盾** →
        # 强制 failed（misattribution 是最危险的幻觉，不容放过）；no_evidence 不触发。
        if misattributed and overall != "failed":
            logger.warning("CoVe: %d 条事实内容与来源矛盾（misattribution），强制驳回", len(misattributed))
            overall = "failed"
            reason = (reason + " " if reason else "") + \
                f"（检出 {len(misattributed)} 条断言与其检索来源正文矛盾）"
            feedback = feedback or "有关键断言与其引用来源的正文相矛盾，请据实修正或更换来源。"

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
