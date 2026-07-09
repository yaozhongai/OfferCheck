"""Trace 双轨合一测试（评审 3.3）：单一 typed schema + OTel 映射 + 同源持久化。

引擎/ server 真实事件 → `nexa_agent.trace.events` 类型化 + OTel-GenAI 属性 →
`server.trace_store.recorder` 同源落盘（SSE 发什么就存什么）。全部离线。
"""

import json

from nexa_agent.trace.events import (
    EngineEventType, KNOWN_EVENT_TYPES, to_otel_attributes, is_known_event,
    resource_attributes,
)
from server.trace_store.recorder import TraceRecorder, load_trace, new_trace_id


# ── 单一 schema 覆盖现役事件（防漂移）──────────────────────────────────────
def test_enum_covers_real_events():
    # 这些是 grep 现役 _emit / event_q.put 的真实类型，必须都在枚举里
    real = {
        "started", "stage_routed", "done", "error",
        "trial_start", "trial_evaluated", "verifier_start", "verifier_result", "usage",
        "step_start", "action", "observation", "correction", "evidence_gate",
        "injection_detected", "ais_downgrade", "retry", "answer_delta", "final_answer",
    }
    assert real <= KNOWN_EVENT_TYPES
    assert is_known_event({"type": "step_start"})
    assert not is_known_event({"type": "totally_made_up"})


# ── OTel GenAI / OpenInference 属性映射 ───────────────────────────────────
def test_otel_step_start_is_llm_span():
    a = to_otel_attributes({"type": "step_start", "step": 2, "max_steps": 12,
                            "model": "deepseek-ai/DeepSeek-V4-Pro"})
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.request.model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert a["openinference.span.kind"] == "LLM"
    assert a["nexa.react.step"] == 2


def test_otel_usage_carries_tokens():
    a = to_otel_attributes({"type": "usage", "trial": 1,
                            "prompt_tokens": 1200, "completion_tokens": 300})
    assert a["gen_ai.usage.input_tokens"] == 1200
    assert a["gen_ai.usage.output_tokens"] == 300
    assert a["openinference.span.kind"] == "LLM"


def test_otel_action_is_tool_span():
    a = to_otel_attributes({"type": "action", "step": 1, "tool": "web_search"})
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert a["gen_ai.tool.name"] == "web_search"
    assert a["openinference.span.kind"] == "TOOL"


def test_otel_unknown_event_minimal_no_crash():
    a = to_otel_attributes({"type": "made_up"})
    assert a["nexa.event.type"] == "made_up"
    assert a["openinference.span.kind"] == "CHAIN"  # 未知落 CHAIN，不抛
    assert "gen_ai.request.model" not in a  # None 值已剔除


def test_resource_attributes():
    r = resource_attributes()
    assert r["service.name"] == "nexa-agent" and r["gen_ai.system"] == "gmi"


# ── 同源持久化：recorder 缓冲 → finalize 落盘 → load 读回 ────────────────────
def test_recorder_persists_same_events(tmp_path):
    tid = new_trace_id()
    rec = TraceRecorder(tid, stage="offercheck_stage4", trace_dir=str(tmp_path))
    rec.record({"type": "started", "stage": "offercheck_stage4", "trace_id": tid})
    rec.record({"type": "step_start", "step": 1, "model": "m", "max_steps": 12})
    rec.record({"type": "action", "step": 1, "tool": "web_search", "args": "字节跳动"})
    rec.record({"type": "usage", "trial": 1, "prompt_tokens": 1000, "completion_tokens": 250})
    rec.record({"type": "done", "success": True,
                "verdict": {"verdict": "靠谱", "verdict_level": "reliable"}})
    path = rec.finalize(success=True, latency_ms=1234)

    assert path is not None
    doc = load_trace(tid, trace_dir=str(tmp_path))
    assert doc is not None
    assert doc["trace_id"] == tid and doc["stage"] == "offercheck_stage4"
    assert doc["event_count"] == 5
    # 每个 span 都带 OTel 属性 + 原始事件（同源）
    kinds = [s["type"] for s in doc["spans"]]
    assert kinds == ["started", "step_start", "action", "usage", "done"]
    assert all("attributes" in s and "event" in s for s in doc["spans"])
    # usage 滚动累计 + verdict 抓取
    assert doc["usage"] == {"input_tokens": 1000, "output_tokens": 250, "total_tokens": 1250}
    assert doc["verdict"]["verdict_level"] == "reliable"
    assert doc["success"] is True and doc["duration_ms"] == 1234
    assert doc["resource"]["service.name"] == "nexa-agent"


def test_recorder_best_effort_ignores_junk(tmp_path):
    rec = TraceRecorder(new_trace_id(), trace_dir=str(tmp_path))
    rec.record({"type": "__end__"})   # 哨兵不记
    rec.record("not a dict")           # 非 dict 不崩
    rec.record({"type": "step_start", "step": 1})
    assert len(rec._spans) == 1


def test_recorder_truncates_large_fields(tmp_path):
    tid = new_trace_id()
    rec = TraceRecorder(tid, trace_dir=str(tmp_path))
    rec.record({"type": "observation", "tool": "web_fetch", "observation": "x" * 5000})
    rec.finalize()
    doc = load_trace(tid, trace_dir=str(tmp_path))
    obs = doc["spans"][0]["event"]["observation"]
    assert obs.startswith("x" * 2000) and "(+3000)" in obs


def test_load_trace_missing_returns_none(tmp_path):
    assert load_trace("does-not-exist", trace_dir=str(tmp_path)) is None
