"""Nexa 引擎轨迹事件 —— 单一 typed schema（评审 3.3：双轨合一）。

此前有**两条轨**：
  ① 引擎 `react_loop`/`execute` 的 `_emit` 直接发 ad-hoc dict 给 SSE、**不落库**；
  ② `nexa_agent/trace/schema.py` + `server/trace_store` 是 LangGraph 时代的另一套
     typed schema（NODE_STARTED / VISION_PERCEIVE …），与现役 ReAct 引擎事件不符，
     且**从未接主链路**（死轨）。

本模块把**现役真实事件**收敛为唯一 typed 契约：SSE 与持久化**同源**（server 端
`TraceRecorder` 落库的就是发给浏览器的同一批事件），并对齐 **OpenTelemetry GenAI**
语义约定 + OpenInference span kind（Langfuse / Phoenix 可直接摄取）。

冻结红线：SSE wire 格式（事件 `type` 值、现有 payload 字段、前端 `reduceRun` 消费）
**不变**。本模块是 wire 之上的「类型化 + 校验 + OTel 属性映射」层——只读事件、产出
附加属性用于持久化/导出，绝不改任何既有字段语义。
"""

from __future__ import annotations

from enum import Enum

from nexa_agent.logger import get_logger

logger = get_logger("trace_events")


class EngineEventType(str, Enum):
    """现役引擎 + server 边界真实发出的全部事件类型（单一真源）。

    与 `_emit(...)` 的字符串、server `event_q.put({"type": ...})`、前端
    `EngineEvent.type` / `reduceRun` 逐一对应。新增事件请在此登记。
    """

    # ── 生命周期（server 边界）──
    STARTED = "started"
    STAGE_ROUTED = "stage_routed"
    DONE = "done"
    ERROR = "error"

    # ── Trial 外循环（reflexion_agent）──
    TRIAL_START = "trial_start"
    TRIAL_EVALUATED = "trial_evaluated"
    VERIFIER_START = "verifier_start"
    VERIFIER_RESULT = "verifier_result"
    USAGE = "usage"

    # ── ReAct 内循环（react_agent）──
    STEP_START = "step_start"
    ACTION = "action"
    OBSERVATION = "observation"
    CORRECTION = "correction"
    EVIDENCE_GATE = "evidence_gate"
    INJECTION_DETECTED = "injection_detected"
    AIS_DOWNGRADE = "ais_downgrade"
    RETRY = "retry"
    ANSWER_DELTA = "answer_delta"
    FINAL_ANSWER = "final_answer"


KNOWN_EVENT_TYPES = frozenset(e.value for e in EngineEventType)


# ──────────────────────────────────────────────────────────────────────────
# OpenTelemetry GenAI / OpenInference 语义映射
# ──────────────────────────────────────────────────────────────────────────
# span 概念对齐：一次 run = agent span；LLM 步/用量 = llm span；工具动作/观察 = tool
# span。Phoenix 用 `openinference.span.kind`，通用 OTel 用 `gen_ai.*`——两者都给，
# 摄取端各取所需。engine 私有维度落 `nexa.*` 命名空间，不污染标准键。

_SYSTEM = "gmi"          # gen_ai.system：底层 LLM 提供方（唯一真源 = MODEL_CONFIG）
_SERVICE = "nexa-agent"  # resource service.name

# 事件 → OpenInference span kind
_SPAN_KIND = {
    EngineEventType.STARTED.value: "AGENT",
    EngineEventType.STAGE_ROUTED.value: "AGENT",
    EngineEventType.DONE.value: "AGENT",
    EngineEventType.ERROR.value: "AGENT",
    EngineEventType.TRIAL_START.value: "AGENT",
    EngineEventType.TRIAL_EVALUATED.value: "AGENT",
    EngineEventType.VERIFIER_START.value: "CHAIN",
    EngineEventType.VERIFIER_RESULT.value: "CHAIN",
    EngineEventType.USAGE.value: "LLM",
    EngineEventType.STEP_START.value: "LLM",
    EngineEventType.ACTION.value: "TOOL",
    EngineEventType.OBSERVATION.value: "TOOL",
    EngineEventType.RETRY.value: "LLM",
    EngineEventType.ANSWER_DELTA.value: "LLM",
    EngineEventType.FINAL_ANSWER.value: "AGENT",
    EngineEventType.CORRECTION.value: "CHAIN",
    EngineEventType.EVIDENCE_GATE.value: "CHAIN",
    EngineEventType.INJECTION_DETECTED.value: "CHAIN",
    EngineEventType.AIS_DOWNGRADE.value: "CHAIN",
}


def to_otel_attributes(event: dict) -> dict:
    """把一条引擎事件映射为 OTel GenAI + OpenInference 属性（用于持久化/导出）。

    只读输入、产出附加属性字典；未知事件回退最小属性。None 值一律剔除。
    """
    et = event.get("type", "")
    attrs: dict = {
        "nexa.event.type": et,
        "openinference.span.kind": _SPAN_KIND.get(et, "CHAIN"),
    }

    if et == EngineEventType.STEP_START.value:
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.system"] = _SYSTEM
        attrs["gen_ai.request.model"] = event.get("model")
        attrs["nexa.react.step"] = event.get("step")
        attrs["nexa.react.max_steps"] = event.get("max_steps")

    elif et == EngineEventType.USAGE.value:
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.system"] = _SYSTEM
        attrs["gen_ai.usage.input_tokens"] = event.get("prompt_tokens")
        attrs["gen_ai.usage.output_tokens"] = event.get("completion_tokens")
        attrs["nexa.trial"] = event.get("trial")

    elif et in (EngineEventType.ACTION.value, EngineEventType.OBSERVATION.value):
        attrs["gen_ai.operation.name"] = "execute_tool"
        attrs["gen_ai.tool.name"] = event.get("tool")
        attrs["nexa.react.step"] = event.get("step")
        if et == EngineEventType.OBSERVATION.value:
            attrs["nexa.tool.ok"] = event.get("ok")

    elif et in (EngineEventType.TRIAL_START.value, EngineEventType.TRIAL_EVALUATED.value):
        attrs["gen_ai.operation.name"] = "invoke_agent"
        attrs["gen_ai.agent.name"] = "reflexion_react"
        attrs["nexa.trial"] = event.get("trial")

    elif et in (EngineEventType.STARTED.value, EngineEventType.DONE.value,
                EngineEventType.ERROR.value, EngineEventType.STAGE_ROUTED.value,
                EngineEventType.FINAL_ANSWER.value):
        attrs["gen_ai.operation.name"] = "invoke_agent"
        attrs["gen_ai.agent.name"] = "reflexion_react"

    return {k: v for k, v in attrs.items() if v is not None}


def resource_attributes() -> dict:
    """OTel resource 级属性（整条 trace 共享）。"""
    return {"service.name": _SERVICE, "gen_ai.system": _SYSTEM}


def is_known_event(event: dict) -> bool:
    """事件 `type` 是否为已登记类型；未知只记 debug、不拦截（wire 透传照旧）。"""
    et = event.get("type")
    if et not in KNOWN_EVENT_TYPES:
        logger.debug("未登记的引擎事件类型（仍透传）：%s", et)
        return False
    return True
