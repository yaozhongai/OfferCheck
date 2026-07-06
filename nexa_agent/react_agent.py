"""
ReAct Agent 实验入口

基于 ReAct（Reasoning + Acting）框架的多工具智能体，使用 OpenAI 原生 SDK
调用 DeepSeek V4 Pro 作为推理核心，支持网页搜索、百科查询、图片分析（端侧/云端）、
数学计算和时间查询。

用法::

    # 纯文字问答
    python -m nexa_agent.react_agent "2025年诺贝尔物理学奖得主是谁？"

    # 携带图片
    python -m nexa_agent.react_agent "分析这张图里的设备状态" --image data/device.jpg

    # 限制步数
    python -m nexa_agent.react_agent "北京今天天气怎么样" --max-steps 5

工作原理::

    [用户问题]
        ↓
    Thought: 分析当前情况，决定调用哪个工具
    Action:  tool_name(arguments)
        ↓ 系统执行工具
    Observation: 工具返回结果
        ↓
    （重复，直到输出 Final Answer 或达到 max_steps）
        ↓
    Final Answer: 最终回答
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Callable, List, Optional, Tuple

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
from nexa_agent.tools import (
    execute_tool, TOOLS,
    get_session_extracts, clear_session_extracts, write_extract_to_disk,
    get_openai_tool_definitions,
)
from nexa_agent.config import (
    MODEL_CONFIG, MODEL_TIER, SUPPORTS_THINKING_PARAM, thinking_extra_body,
    get_model_for_role, DYNAMIC_UPGRADE_THRESHOLD,
)

logger = get_logger("react_agent")

# ==========================================================================
# 配置
# ==========================================================================

# LLM API 配置（provider 由 config.LLM_PROVIDER 决定：GMI Cloud / DeepSeek 官方）
LLM_API_KEY = MODEL_CONFIG["api_key"]
LLM_BASE_URL = MODEL_CONFIG["base_url"]
LLM_MODEL = MODEL_CONFIG["model"]

# 如果 LLM_MODEL 是 flash，ReAct 实验默认升级为 strong 层模型（更好的推理能力）
if "flash" in LLM_MODEL.lower():
    LLM_MODEL = MODEL_TIER["strong"]["model"]
    logger.info("检测到 LLM_MODEL 为 flash 版本，ReAct 实验自动切换为 %s", LLM_MODEL)

DEFAULT_MAX_STEPS = 10

# 强制取证 gate：裁定型输出在零检索时被拦截的最大次数（避免死锁）
MAX_EVIDENCE_GATE_NAGS = 2

# LLM 瞬时错误重试（GMI 网关 Connection error / 超时 / 5xx / 429 常见，重试可自愈；
# 4xx 请求错误不重试——协议/参数问题，见 CLAUDE.md GMI 约束）
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # 秒；指数退避 base（1s → 2s → 4s）
try:
    from openai import (
        APIConnectionError as _APIConnectionError,
        APITimeoutError as _APITimeoutError,
        InternalServerError as _InternalServerError,
        RateLimitError as _RateLimitError,
    )
    _TRANSIENT_LLM_ERRORS: Tuple[type, ...] = (
        _APIConnectionError, _APITimeoutError, _InternalServerError, _RateLimitError,
    )
except ImportError:  # openai 版本差异兜底
    _TRANSIENT_LLM_ERRORS = ()


def _is_transient_llm_error(exc: Exception) -> bool:
    """是否为可重试的瞬时错误（连接/超时/5xx/429）。4xx（400/422）返回 False。"""
    if _TRANSIENT_LLM_ERRORS and isinstance(exc, _TRANSIENT_LLM_ERRORS):
        return True
    # 兜底：按错误文本判断（未装到具体异常类型时）
    msg = str(exc).lower()
    if any(k in msg for k in ("connection", "timeout", "timed out", "temporarily", "econnreset")):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return isinstance(status, int) and status in (429, 500, 502, 503, 504)

# System Prompt 路径
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "react_system.txt")
_CURATION_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "curation.txt")


# ==========================================================================
# System Prompt 加载
# ==========================================================================

def load_system_prompt(stage: Optional[str] = None) -> str:
    """加载 System Prompt，并按 stage 追加该阶段的任务定义层

    通用 ReAct 循环（react_system.txt）是场景无关的引擎底座；
    stage 只在其后追加「本阶段调查目标 + 输出 schema」这层任务定义，
    而非另起一套循环。这样同一个引擎按不同输入切换角色。

    Args:
        stage: 阶段标识（如 "offercheck_stage1"）。None = 纯通用引擎。

    Returns:
        合成后的 System Prompt
    """
    if os.path.isfile(_SYSTEM_PROMPT_PATH):
        with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            base_prompt = f.read()
    else:
        logger.warning("System Prompt 文件不存在: %s，使用内置 Prompt", _SYSTEM_PROMPT_PATH)
        base_prompt = _builtin_system_prompt()

    # 注入实时日期，作为检索时效性基准（{{CURRENT_DATE}} 占位符）
    from datetime import datetime
    today = datetime.now().strftime("%Y年%m月%d日")
    base_prompt = base_prompt.replace("{{CURRENT_DATE}}", today)

    stage_prompt = _load_stage_prompt(stage)
    if stage_prompt:
        return f"{base_prompt}\n\n{stage_prompt}"
    return base_prompt


def _load_stage_prompt(stage: Optional[str]) -> str:
    """加载阶段任务定义 prompt（prompts/<stage>.txt）

    找不到对应文件时返回空串（降级为纯通用引擎），不抛错。
    """
    if not stage:
        return ""
    # 允许传 "stage1" 简写或完整文件名
    candidates = [stage, f"offercheck_{stage}"] if not stage.startswith("offercheck_") else [stage]
    for name in candidates:
        path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                logger.info("加载阶段任务定义: %s", name)
                return f.read()
    logger.warning("阶段 prompt 不存在: stage=%s（降级为通用引擎）", stage)
    return ""


def _builtin_system_prompt() -> str:
    """内置的简化 System Prompt（兜底）"""
    from nexa_agent.tools import get_tools_description

    tools_desc = get_tools_description()
    return f"""你是一个 ReAct（推理 + 行动）智能体。通过交替执行 Thought → Action → Observation 解决问题。

## 可用工具

{tools_desc}

## 输出格式

每一步必须严格遵守：

Thought: <分析当前情况>
Action: <tool_name>(<arguments>)
Observation: <工具返回结果>

...可重复...

Thought: 我有足够信息了。
Final Answer: <最终答案>

## 规则
1. 每次行动前先写 Thought
2. 每步只能调用一个工具
3. 根据 Observation 指导下一步
4. 信息足够立刻输出 Final Answer
5. 不得捏造 Observation
"""


# ==========================================================================
# LLM 调用
# ==========================================================================

def _get_llm_client():
    """获取 OpenAI 客户端实例"""
    from openai import OpenAI

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=120.0,
    )


def call_llm(
    messages: List[dict],
    enable_thinking: bool = True,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    tools: Optional[List[dict]] = None,
) -> Tuple[str, int, int]:
    """调用 DeepSeek LLM（兜底汇总等非 tool calling 场景）

    Returns:
        (response_text, prompt_tokens, completion_tokens)
    """
    client = _get_llm_client()
    actual_model = model or LLM_MODEL

    kwargs = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if tools:
        kwargs["tools"] = tools
    else:
        kwargs["stop"] = ["Observation:"]

    _eb = thinking_extra_body(actual_model, enable_thinking)
    if _eb:
        kwargs["extra_body"] = _eb

    t0 = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed_ms = (time.time() - t0) * 1000

    choice = response.choices[0]
    content = choice.message.content or ""

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else 0

    logger.info(
        "LLM 调用完成 model=%s elapsed=%.0fms tokens(in=%d out=%d total=%d) thinking=%s",
        actual_model, elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
        "on" if enable_thinking else "off",
    )

    return content, prompt_tokens, completion_tokens


def call_llm_with_tools(
    messages: List[dict],
    tools: List[dict],
    enable_thinking: bool = False,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """调用 DeepSeek LLM 并返回完整 choice（支持 tool_calls）

    对瞬时错误（连接/超时/5xx/429）指数退避重试 LLM_MAX_RETRIES 次；4xx 不重试直接抛。
    on_retry(attempt, exc)：每次重试前回调（供上层发 trace 事件）。

    Returns:
        (choice, prompt_tokens, completion_tokens)
        choice.message 可能含 .tool_calls 或 .content
    """
    client = _get_llm_client()
    actual_model = model or LLM_MODEL

    kwargs = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "tools": tools,
    }

    _eb = thinking_extra_body(actual_model, enable_thinking)
    if _eb:
        kwargs["extra_body"] = _eb

    t0 = time.time()
    response = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= LLM_MAX_RETRIES or not _is_transient_llm_error(exc):
                raise
            delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("LLM 瞬时错误，%.1fs 后重试 (%d/%d): %s",
                           delay, attempt + 1, LLM_MAX_RETRIES, exc)
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(delay)
    elapsed_ms = (time.time() - t0) * 1000

    choice = response.choices[0]
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else 0

    has_tool_calls = bool(choice.message.tool_calls)
    logger.info(
        "LLM 调用完成 model=%s elapsed=%.0fms tokens(in=%d out=%d total=%d) "
        "tool_calls=%s thinking=%s",
        actual_model, elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
        has_tool_calls,
        "on" if enable_thinking else "off",
    )

    return choice, prompt_tokens, completion_tokens


_FINAL_ANSWER_RE = re.compile(r"(?:Final\s+Answer|最终答案)\s*[:：]\s*", re.IGNORECASE)
_THOUGHT_PREFIX_RE = re.compile(r"^\s*(?:Thought|思考)\s*[:：]")


def _stream_answer_portion(acc: str) -> str:
    """从流式累积内容里抽出"面向用户的答案"部分，剔除 Thought 推理脚手架。

    - 出现 "Final Answer:" → 取其后为答案；
    - 以 "Thought:" 开头且尚无 Final Answer → 仍是推理，返回空（先不流式）；
    - 其它 → 视为纯文本答案，整段返回。
    这样 tool 步骤里泄漏的 "Thought: ..." 不会被逐字吐给用户。"""
    m = _FINAL_ANSWER_RE.search(acc)
    if m:
        return acc[m.end():]
    if _THOUGHT_PREFIX_RE.match(acc):
        return ""
    return acc


def stream_llm_with_tools(
    messages: List[dict],
    tools: List[dict],
    on_delta: Callable[[str], None],
    enable_thinking: bool = False,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """流式 LLM 调用（用于 answer-mode 逐 token 回复）。

    content 增量通过 on_delta 逐段回调；**只有纯文本回答才流式**——一旦检测到 tool_calls
    就停止 on_delta（工具步骤不该逐字吐给用户）。返回与 call_llm_with_tools 相同的
    (choice, prompt_tokens, completion_tokens)，choice 为鸭子类型对象，可无缝喂给现有
    react_loop 的 情况1（tool_calls）/ 情况2（文本）逻辑，无需改动下游。
    """
    import types as _types

    client = _get_llm_client()
    actual_model = model or LLM_MODEL
    kwargs = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": tools,
    }
    _eb = thinking_extra_body(actual_model, enable_thinking)
    if _eb:
        kwargs["extra_body"] = _eb

    stream = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= LLM_MAX_RETRIES or not _is_transient_llm_error(exc):
                raise
            delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(delay)

    acc_content = ""
    tool_slots: dict = {}
    is_tool = False
    emitted = 0  # 已通过 on_delta 吐出的"答案部分"长度
    prompt_tokens = completion_tokens = 0
    finish_reason = "stop"

    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        if not getattr(chunk, "choices", None):
            continue
        ch0 = chunk.choices[0]
        if getattr(ch0, "finish_reason", None):
            finish_reason = ch0.finish_reason
        delta = ch0.delta
        if getattr(delta, "tool_calls", None):
            is_tool = True  # 这是工具步：停止逐字流式（推理不吐给用户）
            for tc in delta.tool_calls:
                slot = tool_slots.setdefault(tc.index, {"id": None, "type": "function", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                if getattr(tc, "type", None):
                    slot["type"] = tc.type
                fn = getattr(tc, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
        piece = getattr(delta, "content", None)
        if piece:
            acc_content += piece
            # 只逐字流式"答案部分"（剔除 Thought 脚手架），且仅在非工具步
            if not is_tool:
                ans = _stream_answer_portion(acc_content)
                if len(ans) > emitted:
                    try:
                        on_delta(ans[emitted:])
                    except Exception:  # noqa: BLE001
                        pass
                    emitted = len(ans)

    tool_calls_objs = None
    if tool_slots:
        tool_calls_objs = [
            _types.SimpleNamespace(
                id=s["id"], type=s["type"],
                function=_types.SimpleNamespace(name=s["name"], arguments=s["args"]),
            )
            for _, s in sorted(tool_slots.items())
        ]
    msg = _types.SimpleNamespace(
        content=(acc_content or None), tool_calls=tool_calls_objs, reasoning_content=None,
    )
    choice = _types.SimpleNamespace(message=msg, finish_reason=finish_reason)
    logger.info("LLM 流式调用完成 model=%s tokens(in=%d out=%d) tool_calls=%s",
                actual_model, prompt_tokens, completion_tokens, bool(tool_calls_objs))
    return choice, prompt_tokens, completion_tokens


# ==========================================================================
# 响应解析
# ==========================================================================

# 检索类工具：产出外部证据的工具（区别于 calculator/time/图片分析等）
_RETRIEVAL_TOOLS = frozenset({
    "web_search", "wikipedia_search", "web_fetch", "tavily_extract",
    "read_pdf", "read_xlsx", "domain_whois_lookup",
})

# 显式终止工具：模型调用它提交结构化裁定并结束（见 get_openai_tool_definitions 特例）
FINALIZE_TOOL = "submit_verdict"

_URL_RE = re.compile(r'https?://[^\s\)\]\}"\'，。、]+')


def _normalize_url(u: str) -> str:
    """URL 归一化：去 query/fragment/尾部标点，小写。"""
    u = u.split("?")[0].split("#")[0].rstrip("/.,;")
    return u.lower()


def _url_domain(u: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(u).netloc.lower().removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


def attribute_sources(
    answer: str,
    seen_urls: set[str],
    called_tools: set[str],
) -> tuple[str, dict]:
    """来源对账（AIS, Attributable to Identified Sources）。

    逐条检查答案里的 [Source] 行：其引用的 URL / 工具是否真的在本次调查中
    出现过（seen_urls = 观察里见过的所有 URL；called_tools = 实际调用过的工具）。
    对不上的来源标记 ⚠️[未验证]，避免把编造的来源当真凭据。

    Returns:
        (标注后的答案, 报告 dict{total, unverified, unverified_lines})
    """
    seen_norm = {_normalize_url(u) for u in seen_urls}
    seen_domains = {_url_domain(u) for u in seen_norm if _url_domain(u)}
    all_tool_names = set(TOOLS.keys())

    out_lines: list[str] = []
    total = 0
    unverified_lines: list[str] = []

    for line in answer.split("\n"):
        if "[source]" not in line.lower():
            out_lines.append(line)
            continue

        total += 1
        flagged = False

        # 1) 引用的 URL 是否见过（精确或同域）
        for u in _URL_RE.findall(line):
            nu = _normalize_url(u)
            if nu not in seen_norm and _url_domain(u) not in seen_domains:
                flagged = True
                break

        # 2) 引用的工具是否真调用过（如 domain_whois_lookup(...) / web_search(...)）
        if not flagged:
            for mt in re.findall(r'([a-z_]{3,})\s*\(', line):
                if mt in all_tool_names and mt not in called_tools:
                    flagged = True
                    break

        if flagged:
            line = line.rstrip() + "  ⚠️[未验证：该来源未在本次检索记录中找到，可能为臆造]"
            unverified_lines.append(line)
        out_lines.append(line)

    return "\n".join(out_lines), {
        "total_sources": total,
        "unverified": len(unverified_lines),
        "unverified_lines": unverified_lines,
    }


def _as_str_list(v) -> list:
    """把 submit_verdict 的列表型字段规整为字符串列表。

    模型（尤其换用 DeepSeek-V4 后）可能把本应是数组的 evidence/red_flags 传成
    单个字符串——若直接 `for x in v` 会按字符遍历，导致每个字符渲染成一条 [Fact]/
    [RedFlag]（曾见 663 条单字符事实）。这里统一：str→单元素、非序列→单元素、
    序列→逐项 str，并丢掉空串。
    """
    if not v:
        return []
    if isinstance(v, str) or not isinstance(v, (list, tuple)):
        v = [v]
    return [s for s in (str(x).strip() for x in v) if s]


def _render_verdict(fields: dict) -> str:
    """把 submit_verdict 的结构化字段渲染为标准裁定文本。"""
    lines = []
    verdict = (fields.get("verdict") or "").strip()
    summary = (fields.get("summary") or "").strip()
    lines.append(f"[Verdict] {verdict} —— {summary}".rstrip(" —"))
    for ev in _as_str_list(fields.get("evidence")):
        # 模型可能已在 evidence 里自带 [Fact]/[Source]/[Confidence] 标签，
        # 此时原样保留，避免出现 "[Fact] [Fact] ..." 双标签
        lines.append(ev if ev.lstrip().startswith("[") else f"[Fact] {ev}")
    for rf in _as_str_list(fields.get("red_flags")):
        lines.append(rf if rf.lstrip().startswith("[") else f"[RedFlag] {rf}")
    for nc in _as_str_list(fields.get("need_user_confirm")):
        lines.append(nc if nc.lstrip().startswith("[") else f"[NeedUserConfirm] {nc}")
    return "\n".join(lines)


def _answer_requires_evidence(final_answer: str, stage: Optional[str]) -> bool:
    """判断该 Final Answer 是否属于"必须取证"的裁定/事实型输出。

    命中条件：处于 OfferCheck 调查阶段任务，或答案里出现裁定/溯源标签。
    纯计算/纯常识类（stage=None 且无 [Source]/[Verdict]）不强制。
    stage2（简历定向）是对用户提供文本的分析、不做联网证伪，故不强制取证——
    仅当它反常地自带裁定/溯源标签时才要求。
    """
    markers = ("[Verdict]", "[Source]", "[Fact]", "[RedFlag]")
    if stage and ("stage2" in stage):
        return any(m in final_answer for m in markers)
    if stage:
        return True
    return any(m in final_answer for m in markers)


_VERDICT_LABELS = ("靠谱", "存疑", "大概率有坑", "推荐", "谨慎", "不推荐", "值得投递", "谨慎投递", "建议放弃")


def _is_verdict_answer(final_answer: str) -> bool:
    """该答案是否是裁定型输出（含 [Verdict] 标签或裁定标签词）。
    answer-mode 下的非裁定对话式回答不走强制取证 gate。"""
    if "[Verdict]" in final_answer:
        return True
    head = final_answer[:120]
    return any(lbl in head for lbl in _VERDICT_LABELS)


def _assistant_msg_to_dict(msg) -> dict:
    """把 SDK 返回的 assistant message 转为可回传的 dict，保留 reasoning_content。

    DeepSeek 推理模型（GMI 上的 DeepSeek-V4-Pro）在多轮 tool-calling 中，
    返回的 assistant 消息带 reasoning_content（存于 pydantic model_extra）。
    直接把 pydantic 对象 append 回 messages 后，SDK 标准序列化会丢掉这个
    非标准字段，导致下一轮请求被后端拒绝：
        "The `reasoning_content` in the thinking mode must be passed back"。
    这里手动构造 dict 并显式带回 reasoning_content，规避该 400。
    """
    d: dict = {"role": "assistant", "content": msg.content}

    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        d["reasoning_content"] = reasoning

    return d


def parse_llm_response(text: str) -> dict:
    """解析 LLM 响应，提取 Thought、Action 或 Final Answer

    ReAct 格式:
        Thought: <思考>
        Action: tool_name(arguments)

    或:
        Thought: <思考>
        Final Answer: <答案>

    Returns:
        {
            "thought": str | None,
            "action": str | None,       # 工具名
            "action_args": str | None,  # 工具参数
            "final_answer": str | None,
        }
    """
    result = {
        "thought": None,
        "action": None,
        "action_args": None,
        "final_answer": None,
    }

    # 提取 Final Answer
    fa_match = re.search(r"Final\s+Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if fa_match:
        result["final_answer"] = fa_match.group(1).strip()
        # 截断到 Final Answer 之前（取前面的 Thought）
        text_before_fa = text[: fa_match.start()]
    else:
        text_before_fa = text

    # 提取 Thought（取最后一个，因为可能有多个 Thought-Action 对）
    thought_matches = re.findall(r"Thought\s*[:：]\s*(.*?)(?=\n(?:Action|Final|Thought)\s*[:：]|\Z)",
                                 text_before_fa, re.DOTALL | re.IGNORECASE)
    if thought_matches:
        result["thought"] = thought_matches[-1].strip()

    # 提取 Action（只取 Final Answer 之前的最后一个 Action）
    if not result["final_answer"]:
        action_match = re.search(r"Action\s*[:：]\s*(.*)", text_before_fa, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            # 清理 LLM 常添加的 markdown 标记: ** `tool(args)` **
            action_text = re.sub(r"[*`]", "", action_text).strip()
            # 解析 tool_name(arguments)
            tool_match = re.match(r"(\w+)\s*\(\s*(.*?)\s*\)\s*$", action_text, re.DOTALL)
            if tool_match:
                result["action"] = tool_match.group(1)
                args = tool_match.group(2).strip()
                # 去掉 LLM 可能添加的首尾引号（URL常被错误包裹）
                if len(args) >= 2 and args[0] == args[-1] and args[0] in ('"', "'"):
                    args = args[1:-1]
                result["action_args"] = args
            else:
                logger.warning("无法解析 Action 格式: %s", action_text[:100])
                result["action"] = action_text

    return result


# ==========================================================================
# 用户消息构建
# ==========================================================================

def build_user_message(user_query: str, image_path: Optional[str] = None) -> dict:
    """构建发给 LLM 的用户消息

    当有图片时，告知 LLM 图片路径，由 LLM 自行决定调用哪个图片分析工具。
    不将图片数据直接传给 LLM（LLM 不承担视觉感知）。

    Args:
        user_query: 用户问题
        image_path: 可选的图片路径

    Returns:
        OpenAI 格式的消息 dict
    """
    if image_path:
        abs_path = os.path.abspath(image_path)
        # 按文件类型指路：PDF（简历/offer letter/合同）走 read_pdf 文本提取；
        # 图片走 VLM OCR。此前对 PDF 也说「上传了图片…用 analyze_image」，
        # 而 analyze_image_cloud 拒绝非图片扩展名，导致 PDF 附件死路。
        if abs_path.lower().endswith(".pdf"):
            content = (
                f"用户问题: {user_query}\n\n"
                f"注意: 用户上传了一个 PDF 文件（可能是简历、offer letter 或合同），"
                f"路径为: {abs_path}\n"
                f"请先使用 read_pdf 工具读取其文本内容，再基于内容继续任务。"
            )
        else:
            content = (
                f"用户问题: {user_query}\n\n"
                f"注意: 用户上传了一张图片，路径为: {abs_path}\n"
                f"如果需要分析这张图片，请使用 analyze_image（端侧快速）或 "
                f"analyze_image_cloud（云端深度理解）工具。"
                f"参数格式为: 图片路径 | 分析提示词"
            )
    else:
        content = user_query

    return {"role": "user", "content": content}


# ==========================================================================
# ReAct 主循环
# ==========================================================================

def react_loop(
    user_query: str,
    image_path: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
    long_term_memory: Optional[list[str]] = None,
    stage: Optional[str] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    answer_mode: bool = False,
) -> dict:
    """ReAct 主循环 — 基于原生 tool calling

    LLM 通过 function calling API 调用工具，不再依赖正则解析。
    当 LLM 返回 tool_calls 时执行工具；返回纯文本时提取 Final Answer。

    Args:
        stage: 可选的阶段任务定义（如 "offercheck_stage1"），在通用
               System Prompt 后追加该阶段的调查目标与输出 schema。
        on_event: 可选的结构化事件回调 on_event(event: dict)，在关键埋点
                  （步骤开始/工具调用/观察/纠偏/最终答案）处发射，供 server
                  层转 SSE 流式推给前端。回调异常被吞掉，绝不影响主循环。
    """
    def _emit(event_type: str, **payload) -> None:
        if on_event is None:
            return
        try:
            on_event({"type": event_type, **payload})
        except Exception:  # noqa: BLE001 — 可观测钩子绝不能影响主流程
            logger.debug("on_event 回调异常，已忽略", exc_info=True)

    system_prompt = load_system_prompt(stage)
    messages = [{"role": "system", "content": system_prompt}]

    if long_term_memory:
        memory_sys = _build_memory_system_message(long_term_memory)
        if memory_sys:
            messages.append({"role": "system", "content": memory_sys})

    # answer-mode（追问回答模式）：允许基于上文已取证结论直接对话式作答，
    # 只在确需新外部事实时才重新调查。注入一条 system 指令引导。
    if answer_mode:
        messages.append({"role": "system", "content": (
            "【追问回答模式】这是一次针对已有调查结论的追问。若该问题能**基于上文已取证的证据/结论**"
            "直接回答，就用自然、对话式的中文文字**直接回答**，不必重新调查、也不必再套裁定标签或调用 "
            "submit_verdict。仅当确实需要新的外部事实（上文没有）时才调用检索工具。"
            "无论如何都不要凭记忆或常识编造未经查证的新结论。"
        )})

    user_msg = build_user_message(user_query, image_path)
    messages.append(user_msg)

    # 生成 OpenAI tool definitions
    tool_defs = get_openai_tool_definitions()

    step_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    trajectory_parts: list[str] = []

    step_utilities: list[dict] = []
    action_history: list[tuple] = []

    last_tool_success = None
    # 连续未发 tool_calls 计数器（不含已给出 Final Answer 的步骤）
    consecutive_no_toolcall = 0
    # 强制取证 gate：累计成功的检索类工具调用数 + gate 已提醒次数
    successful_retrievals = 0
    evidence_gate_nags = 0
    # 来源对账 registry：本次调查真实见过的 URL（观察全文，截断前收集）
    seen_urls: set[str] = set()
    # 步数预警：OfferCheck stage 下在步数耗尽前注入一次 submit_verdict 提示
    _near_limit_warned = False
    # 跨步硬缓存：tool_name + normalized_args → observation（去重，防冗余调用）
    _tool_cache: dict[str, str] = {}

    print(f"\n{'='*60}")
    print(f"🚀 ReAct Agent 启动 (tool calling 模式)")
    if image_path:
        print(f"🖼️  附带图片: {image_path}")
    print(f"🔧 可用工具: {', '.join(TOOLS.keys())}")
    print(f"🤖 推理模型: 首步={get_model_for_role('react_first')}, 后续={get_model_for_role('react_main')}")
    print(f"📏 最大步数: {max_steps}")
    print(f"{'='*60}\n")

    clear_session_extracts()

    def _gate_should_block(final_answer: str) -> bool:
        """强制取证 gate：裁定/事实型输出但尚无成功检索 → 应拦截（额度内）。"""
        # answer-mode 的非裁定对话式回答：基于上文已取证结论作答，不强制新检索
        if answer_mode and not _is_verdict_answer(final_answer):
            return False
        return (
            _answer_requires_evidence(final_answer, stage)
            and successful_retrievals == 0
            and evidence_gate_nags < MAX_EVIDENCE_GATE_NAGS
        )

    def _build_structured_sources(final_answer: str) -> list[dict]:
        """结构化来源列表：模型在答案中引用的 URL 优先，再用 seen_urls 回填并去重，
        每条标注 verified（是否在真实检索记录中命中）。上限 12 条。"""
        seen_norm = {_normalize_url(u) for u in seen_urls}
        out: list[dict] = []
        chosen: set[str] = set()

        # 本地/内网地址不是真实来源（相对 URL、开发环境自引用等），过滤掉避免噪声
        _LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")

        def _add(u: str) -> None:
            nu = _normalize_url(u)
            dom = _url_domain(u)
            if not dom or nu in chosen or len(out) >= 12:
                return
            if any(dom == h or dom.startswith(h + ":") for h in _LOCAL_HOSTS):
                return
            chosen.add(nu)
            out.append({"url": u, "domain": dom, "verified": nu in seen_norm})

        for u in _URL_RE.findall(final_answer):   # 模型引用的（含可能未验证的）
            _add(u)
        for u in sorted(seen_urls):               # 真实检索过的，回填保证非空
            _add(u)
        return out

    def _finalize(final_answer: str, reason: str = "final_answer",
                  summary_for_user: str = "", suggested_followups: Optional[list] = None) -> dict:
        """收尾：来源对账（AIS）→ 标注答案 → 打印/发射/策展 → 组装结果。"""
        called_tools = {name for name, _ in action_history}
        annotated, attribution = attribute_sources(final_answer, seen_urls, called_tools)
        if attribution["unverified"]:
            logger.warning("来源对账: %d/%d 条来源未验证（已标注）",
                           attribution["unverified"], attribution["total_sources"])
            if verbose:
                print(f"🔎 来源对账: {attribution['unverified']}/{attribution['total_sources']} 条来源未在检索记录中找到（已标注 ⚠️）")
        print(f"\n✅ Final Answer:\n{annotated}")
        logger.info("ReAct 完成 step=%d final_answer_len=%d unverified=%d",
                    step_count, len(annotated), attribution["unverified"])
        # 结构化来源直传前端（问题5）：优先模型在答案里引用的 URL，再用真实检索过的
        # seen_urls 回填，保证只要有过检索就一定有稳定来源；verified=是否在检索记录中命中。
        structured_sources = _build_structured_sources(final_answer)
        _emit("final_answer", step=step_count, answer=annotated, sources=structured_sources,
              summary_for_user=summary_for_user or "",
              suggested_followups=suggested_followups or [])
        _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
        _curation_step(verbose=verbose)
        return {
            "answer": annotated,
            "trajectory": "\n".join(trajectory_parts),
            "steps_used": step_count,
            "terminated_reason": reason,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "step_utilities": step_utilities,
            "critical_step": _find_critical_step(step_utilities),
            "source_attribution": attribution,
            # 结构化行动日志：供 Evaluator 构建紧凑摘要，替代截断的 trajectory 文本
            "action_history": list(action_history),
            "seen_urls": list(seen_urls),
            "successful_retrievals": successful_retrievals,
        }

    def _gate_reprompt_msgs(content_or_note: str) -> None:
        """gate 拦截时向对话注入"先取证"提示（就地 append messages）。"""
        messages.append({
            "role": "user",
            "content": (
                "[系统拦截] 你尚未通过任何检索工具查证，就给出了裁定/事实结论——"
                "这违反接地铁律。严禁凭记忆或常识断言公司/域名/招聘方信息。"
                "请立即调用 web_search / web_fetch / domain_whois_lookup 等工具"
                "实际取证，每条结论都要绑定真实工具返回的来源，然后再下裁定。"
            ),
        })

    while step_count < max_steps:
        step_count += 1

        # 步数预警：剩余 ≤2 步时注入一次 submit_verdict 提示（仅 OfferCheck stage）
        if stage and not _near_limit_warned and (max_steps - step_count) <= 1:
            _near_limit_warned = True
            logger.info("Step %d: 步数预警注入（剩余%d步）", step_count, max_steps - step_count)
            messages.append({
                "role": "user",
                "content": (
                    "⚠️ [系统] 步数即将用尽。如已收集足够证据，"
                    "请立即调用 submit_verdict 工具提交最终裁定，不要再调用其他工具。"
                    "只有在关键信息完全缺失时才继续搜索。"
                ),
            })

        # 动态升级：连续 N 步未发 tool_calls → 切备援模型
        if consecutive_no_toolcall >= DYNAMIC_UPGRADE_THRESHOLD:
            step_model = get_model_for_role("tool_call_upgrade")
            if verbose:
                print(f"⚡ 动态升级: 连续 {consecutive_no_toolcall} 步未调用工具 → {step_model}")
            logger.info(
                "Step %d: 动态升级至 upgrade 层 model=%s (consecutive_no_toolcall=%d)",
                step_count, step_model, consecutive_no_toolcall,
            )
        elif step_count == 1:
            step_model = get_model_for_role("react_first")
        else:
            step_model = get_model_for_role("react_main")

        enable_thinking = (step_count == 1) or (last_tool_success is False)

        print(f"--- Step {step_count}/{max_steps} [模型: {step_model}] ---")
        logger.info("Step %d: 调用 LLM (model=%s thinking=%s, history=%d messages)",
                     step_count, step_model, "on" if enable_thinking else "off", len(messages))
        _emit("step_start", step=step_count, max_steps=max_steps, model=step_model)

        try:
            step_max_tokens = 8192 if enable_thinking else 4096
            _on_retry = lambda attempt, exc: _emit(
                "retry", step=step_count, attempt=attempt,
                max_attempts=LLM_MAX_RETRIES, error=str(exc)[:200])
            if answer_mode:
                # answer-mode：流式调用，纯文本回答逐 token 发 answer_delta（供前端打字机渲染）。
                # 任何流式异常都回退到非流式调用，保证不因流式破坏本轮。
                try:
                    choice, prompt_tok, completion_tok = stream_llm_with_tools(
                        messages, tools=tool_defs, enable_thinking=enable_thinking,
                        model=step_model, max_tokens=step_max_tokens, on_retry=_on_retry,
                        on_delta=lambda t: _emit("answer_delta", step=step_count, text=t),
                    )
                except Exception as _sexc:  # noqa: BLE001
                    logger.warning("流式调用失败，回退非流式: %s", _sexc)
                    choice, prompt_tok, completion_tok = call_llm_with_tools(
                        messages, tools=tool_defs, enable_thinking=enable_thinking,
                        model=step_model, max_tokens=step_max_tokens, on_retry=_on_retry)
            else:
                choice, prompt_tok, completion_tok = call_llm_with_tools(
                    messages, tools=tool_defs, enable_thinking=enable_thinking,
                    model=step_model, max_tokens=step_max_tokens, on_retry=_on_retry)
            total_prompt_tokens += prompt_tok
            total_completion_tokens += completion_tok
        except Exception as exc:
            logger.error("LLM 调用失败 step=%d: %s", step_count, exc, exc_info=True)
            print(f"\n❌ LLM 调用失败: {exc}")
            trajectory = "\n".join(trajectory_parts) if trajectory_parts else "(空轨迹)"
            return {
                "answer": f"[错误] 推理模型调用失败 (step {step_count}): {exc}",
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "llm_error",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
                "action_history": list(action_history),
                "seen_urls": list(seen_urls),
                "successful_retrievals": successful_retrievals,
            }

        msg = choice.message
        content = msg.content or ""

        # ── 情况 1: LLM 返回 tool_calls → 逐个执行所有工具 ──
        if msg.tool_calls:
            consecutive_no_toolcall = 0  # 成功发 tool_calls，重置升级计数器
            # 先把 assistant message（含全部 tool_calls）加入历史
            # 转 dict 并保留 reasoning_content，否则 DeepSeek 推理模型下一轮报 400
            messages.append(_assistant_msg_to_dict(msg))

            if content and verbose:
                print(f"💭 Thought: {content[:200]}{'...' if len(content) > 200 else ''}")

            import json as _json

            # ── 显式终止工具 submit_verdict：结构化裁定 + gate + 来源对账 ──
            verdict_tc = next(
                (tc for tc in msg.tool_calls if tc.function.name == FINALIZE_TOOL), None
            )
            if verdict_tc is not None:
                try:
                    fields = _json.loads(verdict_tc.function.arguments or "{}")
                except _json.JSONDecodeError:
                    fields = {}
                final_answer = _render_verdict(fields)
                trajectory_parts.append(f"### Step {step_count}\nAction: {FINALIZE_TOOL}(...)")

                # 强制取证 gate：零检索不得提交裁定
                if _gate_should_block(final_answer):
                    evidence_gate_nags += 1
                    logger.warning("Step %d: submit_verdict 被 gate 拦截 (nag %d/%d)",
                                   step_count, evidence_gate_nags, MAX_EVIDENCE_GATE_NAGS)
                    if verbose:
                        print(f"🚧 强制取证 gate: submit_verdict 前尚无检索，已拦截 [{evidence_gate_nags}/{MAX_EVIDENCE_GATE_NAGS}]")
                    _emit("evidence_gate", step=step_count, reason="verdict_without_retrieval")
                    messages.append({
                        "role": "tool", "tool_call_id": verdict_tc.id,
                        "content": "[系统拒绝] 你尚未调用任何检索工具就提交裁定。请先用 "
                                   "web_search / web_fetch / domain_whois_lookup 等实际取证，再提交。",
                    })
                    last_tool_success = False
                    continue

                logger.info("Step %d: submit_verdict 提交裁定，结束调查", step_count)
                _emit("action", step=step_count, tool=FINALIZE_TOOL, args="", thought=content[:500] if content else "")
                return _finalize(
                    final_answer, reason="submit_verdict",
                    summary_for_user=(fields.get("summary_for_user") or "").strip(),
                    suggested_followups=_as_str_list(fields.get("suggested_followups")),
                )

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments or "{}"

                try:
                    args_dict = _json.loads(raw_args)
                    tool_args = args_dict.get("input", "")
                except _json.JSONDecodeError:
                    tool_args = raw_args

                # 记录轨迹（首个 tool_call 带 thought）
                if tc is msg.tool_calls[0]:
                    thought_str = f"Thought: {content}\n" if content else ""
                    trajectory_parts.append(
                        f"### Step {step_count}\n{thought_str}"
                        f"Action: {tool_name}({tool_args[:200]})"
                    )
                else:
                    trajectory_parts.append(
                        f"Action (parallel): {tool_name}({tool_args[:200]})"
                    )

                print(f"🔧 Action: {tool_name}({tool_args[:100]}{'...' if len(tool_args) > 100 else ''})")
                _emit("action", step=step_count, tool=tool_name, args=tool_args[:300],
                      thought=(content[:500] if tc is msg.tool_calls[0] and content else ""))

                # ── 跨步硬缓存：相同工具+参数直接返回上次结果，节省步数和 tokens ──
                _cache_key = f"{tool_name}:{tool_args.strip().lower()}"
                if _cache_key in _tool_cache:
                    cached_obs = _tool_cache[_cache_key]
                    observation = f"[缓存] 该查询已在本轮调查中执行过，直接返回缓存结果：\n{cached_obs}"
                    logger.info("Step %d: 工具缓存命中 key=%s...", step_count, _cache_key[:60])
                    last_tool_success = True
                else:
                    observation = execute_tool(tool_name, tool_args)
                    last_tool_success = not observation.startswith("[错误]")
                    if last_tool_success:
                        _tool_cache[_cache_key] = observation

                # 强制取证 gate：累计成功的检索类工具调用
                if last_tool_success and tool_name in _RETRIEVAL_TOOLS:
                    successful_retrievals += 1

                # 来源对账 registry：收集观察全文（截断前）+ 参数里出现的 URL
                if last_tool_success:
                    seen_urls.update(_URL_RE.findall(observation))
                seen_urls.update(_URL_RE.findall(tool_args))

                # 信用分配
                step_utility = _compute_step_utility(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    action_history=action_history,
                )
                step_utilities.append({
                    "step": step_count, "action": tool_name,
                    "args": tool_args[:100], "utility": step_utility,
                })
                action_history.append((tool_name, tool_args.strip()))

                # 按工具类型动态截断
                observation = _truncate_observation(tool_name, observation)

                print(f"👁️  Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}")
                trajectory_parts.append(f"Observation: {observation[:500]}")
                _emit("observation", step=step_count, tool=tool_name,
                      ok=last_tool_success, observation=observation[:600])

                # 每个 tool_call 必须有对应的 tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

            # 所有 tool_calls 处理完后，对最后一个结果做中途纠偏
            correction = _mid_trajectory_check(
                step_count=step_count,
                action_history=action_history,
                step_utilities=step_utilities,
                observation=observation,
            )
            if correction:
                logger.info("Step %d: 中途纠偏触发 — %s", step_count, correction)
                if verbose:
                    print(f"⚡ 中途纠偏: {correction}")
                _emit("correction", step=step_count, message=correction)
                # 纠偏提示作为 user message 注入（不能追加到 tool response 里）
                messages.append({
                    "role": "user",
                    "content": f"[系统纠偏提示] {correction}",
                })

            continue

        # ── 情况 2: 无 tool_calls → 原生终止约定：这段文本即最终答案 ──
        messages.append({"role": "assistant", "content": content})

        content_stripped = content.strip()

        # 空内容（罕见，多为思考模型残留）→ 视为"卡住"，提示继续 + 计入动态升级
        if not content_stripped:
            consecutive_no_toolcall += 1
            logger.warning(
                "Step %d: 返回空内容且无 tool_calls（consecutive_no_toolcall=%d/%d）",
                step_count, consecutive_no_toolcall, DYNAMIC_UPGRADE_THRESHOLD,
            )
            if verbose:
                print(f"⚠️  返回空内容 [{consecutive_no_toolcall}/{DYNAMIC_UPGRADE_THRESHOLD}]，提示继续")
            messages.append({
                "role": "user",
                "content": "请调用合适的工具继续调查，或直接给出你的结论。",
            })
            step_utilities.append({"step": step_count, "action": "_no_action", "args": "", "utility": -0.3})
            last_tool_success = False
            continue

        trajectory_parts.append(f"### Step {step_count}\n{content}")

        # 提取最终答案：有 "Final Answer:" 哨兵则取其后（向后兼容）；
        # 否则整段文本即为最终答案（原生终止约定：无 tool_calls = 已完成）。
        parsed = parse_llm_response(content)
        final_answer = parsed["final_answer"] if parsed["final_answer"] else content_stripped

        # ── 强制取证 gate（no evidence, no answer）──
        # 裁定/事实型输出但尚无成功检索 → 拒绝 finalize，逼其先取证（额度内）。
        if _gate_should_block(final_answer):
            evidence_gate_nags += 1
            logger.warning(
                "Step %d: 强制取证 gate 拦截 — 尚无成功检索却给出裁定 (nag %d/%d)",
                step_count, evidence_gate_nags, MAX_EVIDENCE_GATE_NAGS,
            )
            if verbose:
                print(f"🚧 强制取证 gate: 你还没查证就下结论，已拦截 [{evidence_gate_nags}/{MAX_EVIDENCE_GATE_NAGS}]")
            _emit("evidence_gate", step=step_count, reason="no_retrieval_before_verdict")
            _gate_reprompt_msgs(content)
            last_tool_success = False
            continue

        # 原生终止：收尾（来源对账 + 组装结果）
        return _finalize(final_answer, reason="final_answer")

    # 达到 max_steps：触发兜底汇总（不传 tools，强制纯文本回答）
    logger.warning("达到 max_steps=%d，触发兜底汇总", max_steps)
    print(f"\n⚠️  达到最大步数 {max_steps}，触发兜底汇总...")

    fallback_prompt = (
        "你已经达到了最大步数限制。请基于以上所有信息，"
        "直接给出对用户问题的最终答案。不要再调用任何工具。\n"
        "格式要求: Final Answer: <你的答案>"
    )
    messages.append({"role": "user", "content": fallback_prompt})

    try:
        response_text, prompt_tok, completion_tok = call_llm(
            messages,
            enable_thinking=False,
            model=get_model_for_role("react_main"),
            max_tokens=6144,
        )
        total_prompt_tokens += prompt_tok
        total_completion_tokens += completion_tok
        trajectory_parts.append(f"### 兜底汇总\n{response_text}")

        parsed = parse_llm_response(response_text)
        if parsed["final_answer"]:
            final_answer = parsed["final_answer"]
            print(f"\n✅ (兜底) Final Answer:\n{final_answer}")
            _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
            _curation_step(verbose=verbose)
            trajectory = "\n".join(trajectory_parts)
            return {
                "answer": final_answer,
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "max_steps",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
                "action_history": list(action_history),
                "seen_urls": list(seen_urls),
                "successful_retrievals": successful_retrievals,
            }
        else:
            # 尝试提取完整响应作为答案
            print(f"\n⚠️  兜底汇总未找到 Final Answer 标记，使用完整响应")
            _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
            _curation_step(verbose=verbose)
            trajectory = "\n".join(trajectory_parts)
            return {
                "answer": response_text.strip(),
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "max_steps",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
                "action_history": list(action_history),
                "seen_urls": list(seen_urls),
                "successful_retrievals": successful_retrievals,
            }

    except Exception as exc:
        logger.error("兜底汇总 LLM 调用失败: %s", exc, exc_info=True)
        _curation_step(verbose=verbose)
        trajectory = "\n".join(trajectory_parts) if trajectory_parts else "(空轨迹)"
        return {
            "answer": f"[错误] 兜底汇总失败: {exc}",
            "trajectory": trajectory,
            "steps_used": step_count,
            "terminated_reason": "llm_error",
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "step_utilities": step_utilities,
            "critical_step": _find_critical_step(step_utilities),
            "action_history": list(action_history),
            "seen_urls": list(seen_urls),
            "successful_retrievals": successful_retrievals,
        }


def _compute_step_utility(
    tool_name: str,
    tool_args: str,
    observation: str,
    action_history: list[tuple],
) -> float:
    """计算单个步骤的效用值 [-1.0, +1.0]

    效用规则:
        - 搜索/百科返回有效信息: +0.5
        - 搜索/百科无结果: 0.0
        - 重复搜索相同 query: -0.5
        - fetch/extract 成功获取数据 (>500 chars): +1.0
        - fetch/extract 超时/无内容: -0.3
        - 工具报错: -0.5
        - 计算成功: +0.3
        - 其他默认: 0.0
    """
    obs_lower = observation.lower()
    tool_lower = tool_name.lower()
    args_clean = tool_args.strip()
    has_error = observation.startswith("[错误]") or observation.lower().startswith("[error]")
    has_no_result = any(kw in obs_lower for kw in ["未找到", "无结果", "no result", "not found"])

    # 工具错误
    if has_error:
        return -0.5

    # 重复检测：相同工具 + 相同参数
    if (tool_name, args_clean) in action_history:
        return -0.5

    # tavily_extract / content extraction
    if tool_name in ("tavily_extract",):
        if len(observation) > 500:
            return 1.0
        return -0.3

    # web_search / wikipedia_search
    if tool_name in ("web_search", "wikipedia_search"):
        if has_no_result:
            return 0.0
        return 0.5

    # 计算器
    if tool_name == "calculator":
        if has_error:
            return -0.5
        return 0.3

    # 图片分析
    if tool_name in ("analyze_image", "analyze_image_cloud"):
        if has_error:
            return -0.5
        return 0.3

    # 保存内容
    if tool_name == "save_content":
        if has_error:
            return -0.5
        return 0.3

    # 时间查询
    if tool_name == "get_current_time":
        return 0.1

    return 0.0


def _find_critical_step(step_utilities: list[dict]) -> Optional[dict]:
    """找出效用值最低的步骤作为 critical_step

    Args:
        step_utilities: 步骤效用列表

    Returns:
        效用最低的步骤信息，如果列表为空则返回 None
    """
    if not step_utilities:
        return None

    critical = min(step_utilities, key=lambda x: x["utility"])
    # 仅当有负效用步骤时才返回
    if critical["utility"] < 0:
        return critical
    return None


def _truncate_observation(tool_name: str, observation: str) -> str:
    """按工具类型和内容长度分档处理 Observation

    策略（避免信息丢失）：
    - ≤ 15K chars: 原样返回，不压缩（DeepSeek 128K 窗口完全承受得住）
    - 15K-50K chars: 三明治截断（首尾各保留，中间省略）
    - > 50K chars: 仅保留前 8K + 结构提示，引导 Agent 用分页工具精读
    - 短内容工具（web_search 等）: 上限 3000 chars
    """
    LONG_CONTENT_TOOLS = {"read_pdf", "web_fetch", "tavily_extract"}

    if tool_name not in LONG_CONTENT_TOOLS:
        max_len = 3000
        if len(observation) <= max_len:
            return observation
        return observation[:max_len] + f"\n...(已截断至 {max_len} 字符)"

    length = len(observation)

    # 小文档: 原样返回，不丢任何信息
    if length <= 15000:
        return observation

    # 中等文档: 三明治截断
    if length <= 50000:
        head_len = 6000
        tail_len = 6000
        head = observation[:head_len]
        tail = observation[-tail_len:]
        omitted = length - head_len - tail_len
        return (
            f"{head}\n\n"
            f"...（中间省略 {omitted} 字符，共 {length} 字符。"
            f"如需查看省略部分，请用更精确的搜索或分页读取）...\n\n"
            f"{tail}"
        )

    # 超长文档: 只保留开头 + 提示 Agent 分页精读
    head = observation[:8000]
    return (
        f"{head}\n\n"
        f"...（文档共 {length} 字符，仅显示前 8000 字符。"
        f"请根据以上内容确定需要的章节，然后用工具精确查询具体部分）..."
    )


def _mid_trajectory_check(
    step_count: int,
    action_history: list[tuple],
    step_utilities: list[dict],
    observation: str,
) -> Optional[str]:
    """中途纠偏：在 ReAct 循环内部实时检测异常并生成纠偏提示

    零额外 LLM 调用，复用已有的启发式规则。

    Returns:
        纠偏提示字符串（需要纠偏时），或 None（正常继续）
    """
    if step_count < 2:
        return None

    # 检测 1: 同工具+同参数重复 ≥2 次 → 立即干预
    if len(action_history) >= 2:
        last_action = action_history[-1]
        repeat_count = sum(1 for a in action_history if a == last_action)
        if repeat_count >= 2:
            tool_name, tool_args = last_action
            return (
                f"你已经用相同参数调用 {tool_name} {repeat_count} 次，结果相同。"
                f"请立即改变策略：换用不同的搜索关键词、尝试其他工具、"
                f"或基于已有信息直接给出 Final Answer。"
            )

    # 检测 1.5: URL 级跨工具去重 — 同一 URL 被不同工具访问过 ≥2 次
    if len(action_history) >= 2:
        import re as _re
        url_fetch_tools = {"web_fetch", "tavily_extract", "read_pdf"}
        last_tool, last_args = action_history[-1]
        if last_tool in url_fetch_tools:
            url_match = _re.search(r"https?://[^\s]+", last_args)
            if url_match:
                target_url = url_match.group(0).split("?")[0].rstrip("/")
                prev_hits = 0
                for prev_tool, prev_args in action_history[:-1]:
                    if prev_tool in url_fetch_tools:
                        prev_url_match = _re.search(r"https?://[^\s]+", prev_args)
                        if prev_url_match:
                            prev_url = prev_url_match.group(0).split("?")[0].rstrip("/")
                            if prev_url == target_url:
                                prev_hits += 1
                if prev_hits >= 1:
                    return (
                        f"这个 URL 已经被访问过 {prev_hits + 1} 次了（可能用了不同工具）。"
                        f"重复访问同一 URL 不会得到新信息。"
                        f"请换一个信息源，或基于已有信息直接回答。"
                    )

    # 检测 2: 连续工具错误 ≥2 次
    recent_utils = step_utilities[-2:] if len(step_utilities) >= 2 else []
    if len(recent_utils) == 2 and all(u["utility"] <= -0.3 for u in recent_utils):
        return (
            "最近连续 2 步工具调用都失败或无效。"
            "请停下来重新思考：是否在用错误的工具或错误的参数？"
            "考虑换一个工具或换一种方式获取信息。"
        )

    # 检测 3: 同一类工具调用过多（不同参数但同工具 ≥8 次）
    if len(action_history) >= 8:
        from collections import Counter
        tool_counts = Counter(name for name, _ in action_history)
        for tool_name, count in tool_counts.items():
            if count >= 8:
                return (
                    f"你已经调用 {tool_name} {count} 次了。"
                    f"搜索策略可能已经失效，请尝试：1) 用 tavily_extract 直接抓取已知 URL；"
                    f"2) 用 wikipedia_search 查百科；3) 基于现有信息直接回答。"
                )

    # 检测 4: 观察结果太短（可能是空页面或无效响应）
    # 排除 calculator 和 get_current_time — 它们的正常输出本身就很短
    if observation and len(observation.strip()) < 50 and not observation.startswith("[错误]"):
        last_tool = action_history[-1][0] if action_history else ""
        short_output_tools = {"calculator", "get_current_time"}
        if step_count > 3 and last_tool not in short_output_tools:
            return (
                "上一步返回的信息非常少（不到 50 字符），可能是空页面或无效响应。"
                "请尝试不同的 URL 或搜索词。"
            )

    return None


def _build_memory_prefix(memories: list[str]) -> str:
    """构建记忆注入前缀（保留向后兼容，但不再推荐使用）"""
    if not memories:
        return ""
    prefix = "【重要提醒：你之前在类似任务中犯过以下错误，务必避免重蹈覆辙】\n\n"
    for i, mem in enumerate(memories, 1):
        prefix += f"教训 {i}: {mem}\n\n"
    prefix += "---\n\n"
    return prefix


def _build_memory_system_message(memories: list[str]) -> Optional[str]:
    """构建结构化记忆 system message（推荐方式）

    将教训作为 system role 的结构化约束注入，
    比 user message 前缀有更高的 LLM 遵从率。
    """
    if not memories:
        return None

    constraints = []
    for mem in memories:
        mem = mem.strip()
        if mem:
            if not mem.startswith("- "):
                mem = f"- {mem}"
            constraints.append(mem)

    return (
        "MANDATORY CONSTRAINTS from prior task failures "
        "(violating these will cause task failure):\n"
        + "\n".join(constraints)
    )


def _print_summary(steps: int, prompt_tokens: int, completion_tokens: int) -> None:
    """打印执行摘要"""
    print(f"\n{'='*60}")
    print(f"📊 执行摘要: {steps} 步, "
          f"输入 {prompt_tokens} tokens, 输出 {completion_tokens} tokens, "
          f"合计 {prompt_tokens + completion_tokens} tokens")
    print(f"{'='*60}")


# ==========================================================================
# 策展步骤（ReAct 结束后独立执行）
# ==========================================================================

def _curation_step(verbose: bool = True) -> None:
    """ReAct 循环结束后，由独立 LLM 调用判断哪些 extract 值得保存

    使用独立的 curation prompt，不影响主 System Prompt。
    无缓存时直接跳过，不产生额外 API 调用。
    """
    extracts = get_session_extracts()
    if not extracts:
        return

    logger.info("策展步骤启动 extracts=%d", len(extracts))

    # 加载策展 prompt
    curation_prompt = ""
    if os.path.isfile(_CURATION_PROMPT_PATH):
        with open(_CURATION_PROMPT_PATH, "r", encoding="utf-8") as f:
            curation_prompt = f.read()
    else:
        logger.warning("策展 prompt 文件不存在: %s，跳过", _CURATION_PROMPT_PATH)
        return

    # 构建待判断的内容摘要
    items_text = []
    for i, ext in enumerate(extracts):
        title = ext.get("title", "无标题")
        url = ext.get("url", "")
        length = len(ext.get("raw_content", ""))
        # 取前 500 字供判断
        preview = ext.get("raw_content", "")[:500]
        items_text.append(
            f"### [{i}] {title}\n"
            f"来源: {url}\n"
            f"字符数: {length}\n"
            f"内容预览: {preview}...\n"
        )
    items_block = "\n".join(items_text)

    curation_messages = [
        {"role": "system", "content": curation_prompt},
        {"role": "user", "content": f"请判断以下 {len(extracts)} 条提取内容是否值得保存：\n\n{items_block}"},
    ]

    try:
        response_text, _, _ = call_llm(
            curation_messages, enable_thinking=False, max_tokens=1024,
        )
    except Exception as exc:
        logger.error("策展 LLM 调用失败: %s", exc)
        return

    # 解析 SAVE / SKIP 行
    saved = 0
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("SAVE:") or line.startswith("SAVE："):
            # 格式: SAVE: filename | reason
            body = line.split(":", 1)[1].strip() if ":" in line else ""
            if "|" in body:
                fname = body.split("|")[0].strip()
            else:
                fname = body.strip()
            if fname and saved < len(extracts):
                extract = extracts[saved]  # 按顺序对应
                try:
                    filepath = write_extract_to_disk(extract, fname)
                    logger.info("策展: SAVE → %s", filepath)
                    if verbose:
                        print(f"📥 策展保存: {fname}.md ({len(extract.get('raw_content', ''))} 字符)")
                    saved += 1
                except Exception as exc:
                    logger.error("策展写入失败 %s: %s", fname, exc)

    if verbose and saved > 0:
        print(f"📥 策展完成: 共保存 {saved}/{len(extracts)} 条\n")
    elif verbose:
        print(f"📥 策展完成: 无内容需保存 ({len(extracts)} 条均跳过)\n")


# ==========================================================================
# CLI 入口
# ==========================================================================

def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(
        description="ReAct Agent — 基于 ReAct 框架的多工具智能体实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m nexa_agent.react_agent "2025年诺贝尔物理学奖得主是谁？"
  python -m nexa_agent.react_agent "分析这张图里的设备状态" --image data/device.jpg
  python -m nexa_agent.react_agent "北京今天天气怎么样" --max-steps 5
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        help="用户问题",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="图片路径（相对或绝对路径）",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"最大步数（默认: {DEFAULT_MAX_STEPS}）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印中间过程",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"覆盖推理模型（默认: {LLM_MODEL}）",
    )

    args = parser.parse_args()

    # 覆盖模型
    if args.model:
        LLM_MODEL = args.model

    # 验证 API Key
    if not LLM_API_KEY:
        print("❌ 错误: 未配置 DEEPSEEK_API_KEY 或 KIMI_API_KEY")
        print("   请在 .env 文件中设置 API Key")
        sys.exit(1)

    # 验证图片路径
    image_path = None
    if args.image:
        if os.path.isfile(args.image):
            image_path = args.image
        else:
            # 尝试相对项目根
            alt_path = os.path.join(_project_root, args.image)
            if os.path.isfile(alt_path):
                image_path = alt_path
            else:
                print(f"❌ 错误: 图片文件不存在: {args.image}")
                print(f"   也尝试过: {alt_path}")
                sys.exit(1)

    # 运行 ReAct 循环
    result = react_loop(
        user_query=args.query,
        image_path=image_path,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    # 提取答案（兼容新的 dict 返回格式）
    answer = result["answer"] if isinstance(result, dict) else result
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"📊 终止原因: {result.get('terminated_reason', 'N/A') if isinstance(result, dict) else 'N/A'}")
        print(f"{'='*60}")

    return answer


if __name__ == "__main__":
    main()
