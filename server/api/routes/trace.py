"""
Trace 查询 API — 提供前端调试和查询接口
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from nexa_agent.trace.schema import TraceVisibility
from server.trace_store.service import get_trace_events, build_timeline_items
from server.trace_store.sse import subscribe_trace_events
from nexa_agent.logger import get_logger

logger = get_logger("api_trace")
router = APIRouter(prefix="/api/v0/trace", tags=["trace"])


@router.get("/{trace_id}/events", summary="查询 Trace 事件")
async def list_events(
    trace_id: str,
    after_seq: Optional[int] = Query(None, description="断线重连起始 seq"),
    limit: int = Query(200, ge=1, le=500),
):
    """查询某次请求的所有 Trace 事件"""
    events = get_trace_events(trace_id, after_seq=after_seq, limit=limit)
    return {
        "trace_id": trace_id,
        "events": [
            {
                "seq": e.seq,
                "event_type": e.event_type.value,
                "event_status": e.event_status.value,
                "node_name": e.node_name,
                "title": e.title,
                "message": e.message,
                "input_summary": e.input_summary,
                "output_summary": e.output_summary,
                "payload": e.payload,
                "duration_ms": e.duration_ms,
                "event_level": e.event_level.value,
                "error_type": e.error_type,
                "error_message": e.error_message,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


@router.get("/{trace_id}/timeline", summary="前端时间线")
async def timeline(trace_id: str):
    """获取聚合后的前端时间线"""
    items = build_timeline_items(trace_id)
    return {
        "trace_id": trace_id,
        "items": [
            {
                "node_name": i.node_name,
                "title": i.title,
                "status": i.status.value,
                "duration_ms": i.duration_ms,
                "event_count": len(i.events),
                # 取最近一条 event 的 message / output_summary
                "message": i.events[-1].message if i.events else None,
                "output_summary": i.events[-1].output_summary if i.events else None,
            }
            for i in items
        ],
    }


@router.get("/{trace_id}/stream", summary="SSE 事件流")
async def stream_events(
    trace_id: str,
    after_seq: Optional[int] = Query(None),
):
    """SSE 实时推送 Trace 事件"""
    return StreamingResponse(
        subscribe_trace_events(trace_id, after_seq=after_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
