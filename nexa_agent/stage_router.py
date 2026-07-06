"""followup 轻量 stage 路由 — 单一对话内的多能力组合

一个 Case 的对话里，用户的追问可能跨出当前阶段的能力范围（在 offer 证伪
对话里问「顺便看下我简历和这 JD 的差距」）。本模块把这类追问路由到对应
stage 的能力（换 stage prompt 重跑），实现「四阶段任意组合、一个对话流」。

设计为两级，成本导向：

1. **关键词门（零成本）**：绝大多数追问是当前话题的延续（「为什么这么判」
   「来源是哪」），不含任何跨阶段信号词 → 直接保持当前 stage，不发 LLM。
2. **LLM 确认（fast 层单次）**：命中其他 stage 的信号词时才调用 fast 层
   做一次分类确认（keep / stage1..4），防关键词误伤（如 stage4 对话里
   「offer」一词到处都是）。LLM 失败/超时 → 安全回落 keep。

server 的 /run_stage/stream 在 `auto_route=true`（前端 followup 默认开）时
调用 `route_stage_for_followup`，路由发生则发 `stage_routed` 事件供前端展示。
"""

from __future__ import annotations

import json
import re
from typing import Optional

from nexa_agent.config import MODEL_CONFIG, SUPPORTS_THINKING_PARAM, get_model_for_role
from nexa_agent.logger import get_logger

logger = get_logger("stage_router")

# followup 输入里的追问正文标记（与 web/app/ui.tsx buildFollowupInput 的契约）
_QUESTION_MARKER = "[追问/补充信息]"

# 每个 stage 的跨阶段信号词（中英双语；只用于「门」，最终由 LLM 确认）
_STAGE_SIGNALS: dict[str, list[str]] = {
    "stage1": [
        "选岗", "值得投", "值不值得投", "公司调研", "优先级", "僵尸岗", "背景调查",
        "worth applying", "company research", "prioritize", "ghost job",
    ],
    "stage2": [
        "简历", "resume", "cv", "jd 差距", "定向修改", "匹配度", "关键词覆盖",
        "tailor", "ats", "改简历", "简历差距", "突出哪", "怎么改",
    ],
    "stage3": [
        "招聘方消息", "hr 消息", "聊天记录", "沟通记录", "recruiter message",
        "whatsapp", "telegram", "聊天截图", "对话截图", "他说", "对方说",
    ],
    "stage4": [
        "offer", "合同", "录用通知", "薪资条款", "offer letter", "入职要求",
    ],
}

_VALID_STAGES = tuple(_STAGE_SIGNALS.keys())


def extract_followup_question(task_input: str) -> str:
    """从 followup 打包输入里取出追问正文；无标记则返回原文"""
    idx = task_input.rfind(_QUESTION_MARKER)
    if idx == -1:
        return task_input
    return task_input[idx + len(_QUESTION_MARKER):].strip()


def _normalize_stage(stage: str) -> str:
    """'offercheck_stage4' → 'stage4'"""
    m = re.search(r"stage[1-4]", stage or "")
    return m.group(0) if m else (stage or "")


def _keyword_gate(question: str, current: str) -> set[str]:
    """返回命中信号词的**其他** stage 集合；空集 = 无跨阶段信号，保持当前"""
    q = question.lower()
    hits = set()
    for stage, signals in _STAGE_SIGNALS.items():
        if stage == current:
            continue
        if any(sig in q for sig in signals):
            hits.add(stage)
    return hits


def _llm_confirm(question: str, current: str, candidates: set[str]) -> Optional[str]:
    """fast 层单次分类确认：keep 或某个 stage；失败回 None（安全 keep）"""
    from openai import OpenAI

    stage_desc = {
        "stage1": "选岗调研：调查某公司/岗位是否真实、健康、值得投递",
        "stage2": "简历定向：对比 JD 与简历差距，给修改清单",
        "stage3": "沟通证伪：核实招聘方身份、检测聊天中的异常/红旗",
        "stage4": "offer 证伪：对 offer/合同做真伪与风险裁定",
    }
    options = ["keep"] + sorted(candidates)
    prompt = (
        "用户正在求职助手的多阶段对话中追问。判断这条追问应该由哪个能力处理。\n\n"
        f"当前阶段：{current}（{stage_desc.get(current, '')}）\n"
        "候选：\n"
        + "\n".join(f"- {o}" + (f"：{stage_desc[o]}" if o in stage_desc else "：继续当前对话（默认，仅当追问明显要求另一能力时才切换）")
                    for o in options)
        + f"\n\n追问原文：\n{question[:600]}\n\n"
        '只输出 JSON：{"route": "<keep|stageN>", "why": "<一句话>"}'
    )
    try:
        client = OpenAI(
            api_key=MODEL_CONFIG["api_key"], base_url=MODEL_CONFIG["base_url"],
            timeout=15.0, max_retries=1,
        )
        kwargs = {
            "model": get_model_for_role("evaluator_llm"),
            "messages": [
                {"role": "system", "content": "你是意图路由器。只输出 JSON。倾向保守：不确定就 keep。"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        if SUPPORTS_THINKING_PARAM and "deepseek" in kwargs["model"].lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        route = str(data.get("route", "keep")).strip()
        if route in candidates:
            logger.info("stage 路由确认 %s→%s：%s", current, route, data.get("why", ""))
            return route
        return None  # keep / 非候选 → 保持
    except Exception as exc:  # noqa: BLE001 — 路由失败绝不阻塞主流程
        logger.warning("stage 路由 LLM 确认失败（保持当前 stage）：%s", exc)
        return None


def route_stage_for_followup(task_input: str, current_stage: str) -> tuple[Optional[str], str]:
    """判断 followup 是否应切换到其他 stage 能力

    Args:
        task_input: followup 的完整打包输入（含上文 JSON 与追问正文）
        current_stage: 当前 stage（接受 'stage4' 或 'offercheck_stage4'）

    Returns:
        (routed_stage | None, reason)——None 表示保持当前 stage
    """
    current = _normalize_stage(current_stage)
    if current not in _VALID_STAGES:
        return None, "非四阶段对话，不路由"

    question = extract_followup_question(task_input)
    if not question:
        return None, "空追问"

    candidates = _keyword_gate(question, current)
    if not candidates:
        return None, "无跨阶段信号词（零成本保持）"

    routed = _llm_confirm(question, current, candidates)
    if routed is None:
        return None, "LLM 确认保持当前阶段"
    return routed, f"追问命中 {routed} 能力"
