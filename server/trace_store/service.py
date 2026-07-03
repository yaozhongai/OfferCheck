"""
Trace Service — 封装 create / emit / complete / fail 业务接口
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from nexa_agent.trace.schema import (
    AgentTraceRun, AgentTraceEvent, AgentTimelineItem,
    TraceStatus, TraceEventType, TraceEventStatus, TraceEventLevel, TraceVisibility,
)
from server.trace_store.store import TraceStore
from nexa_agent.logger import get_logger

logger = get_logger("trace_service")

_store: Optional[TraceStore] = None


def _get_store() -> TraceStore:
    global _store
    if _store is None:
        from server.config import get_config
        _store = TraceStore(db_path=get_config().db_path)
    return _store


# ── 创建 ──

def create_trace_run(
    request_id: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> AgentTraceRun:
    """每个 request_id 默认对应一个 trace_id"""
    run = AgentTraceRun(
        trace_id=request_id,
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
    )
    _get_store().create_run(run)
    logger.debug("TraceRun 创建 trace_id=%s", run.trace_id)
    return run


# ── 事件 ──

def emit_trace_event(
    trace_id: str,
    request_id: str,
    session_id: str,
    event_type: TraceEventType,
    title: str,
    node_name: Optional[str] = None,
    event_status: TraceEventStatus = TraceEventStatus.SUCCESS,
    event_level: TraceEventLevel = TraceEventLevel.INFO,
    visibility: TraceVisibility = TraceVisibility.USER,
    message: Optional[str] = None,
    input_summary: Optional[str] = None,
    output_summary: Optional[str] = None,
    payload: Optional[dict] = None,
    duration_ms: Optional[int] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> AgentTraceEvent:
    store = _get_store()
    # 计算下一个 seq
    existing = store.get_events(trace_id, limit=1)
    seq = (existing[-1].seq + 1) if existing else 1

    event = AgentTraceEvent(
        event_id=str(uuid.uuid4()),
        trace_id=trace_id,
        request_id=request_id,
        session_id=session_id,
        seq=seq,
        event_type=event_type,
        event_status=event_status,
        event_level=event_level,
        visibility=visibility,
        node_name=node_name,
        title=title,
        message=message,
        input_summary=input_summary,
        output_summary=output_summary,
        payload=payload or {},
        duration_ms=duration_ms,
        error_type=error_type,
        error_message=error_message,
    )
    store.emit_event(event)
    return event


def complete_trace_run(trace_id: str, final_answer_summary: Optional[str] = None) -> AgentTraceRun:
    _get_store().complete_run(trace_id, final_answer_summary)
    emit_trace_event(
        trace_id=trace_id, request_id=trace_id, session_id="",
        event_type=TraceEventType.TRACE_COMPLETED,
        title="请求完成", message=final_answer_summary,
    )
    run = _get_store().get_run(trace_id)
    logger.debug("TraceRun 完成 trace_id=%s", trace_id)
    return run


def fail_trace_run(
    trace_id: str, error_type: str, error_message: str,
    node_name: Optional[str] = None, recoverable: bool = True,
) -> AgentTraceRun:
    _get_store().fail_run(trace_id)
    emit_trace_event(
        trace_id=trace_id, request_id=trace_id, session_id="",
        event_type=TraceEventType.TRACE_FAILED,
        title="请求失败", node_name=node_name,
        event_status=TraceEventStatus.FAILED, event_level=TraceEventLevel.ERROR,
        error_type=error_type, error_message=error_message,
        payload={"recoverable": recoverable},
    )
    return _get_store().get_run(trace_id)


def get_trace_events(trace_id: str, after_seq: Optional[int] = None, limit: int = 200) -> list[AgentTraceEvent]:
    return _get_store().get_events(trace_id, after_seq=after_seq, limit=limit)


def build_timeline_items(trace_id: str) -> list[AgentTimelineItem]:
    """将事件聚合为前端 Timeline"""
    events = _get_store().get_events(trace_id)
    if not events:
        return []
    items: dict[str, AgentTimelineItem] = {}
    for evt in events:
        key = evt.node_name or evt.event_type.value
        if key not in items:
            items[key] = AgentTimelineItem(
                item_id=str(uuid.uuid4()), trace_id=trace_id,
                node_name=key, title=evt.title,
                status=evt.event_status, level=evt.event_level,
                duration_ms=evt.duration_ms,
            )
        items[key].events.append(evt)
    return list(items.values())
