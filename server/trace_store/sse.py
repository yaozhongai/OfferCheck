"""
Trace SSE — 流式推送 Trace 事件

用于 FastAPI SSE 推送，支持 after_seq 断线重连。
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from nexa_agent.trace.schema import TraceVisibility
from server.trace_store.service import get_trace_events
from nexa_agent.logger import get_logger

logger = get_logger("trace_sse")


async def subscribe_trace_events(
    trace_id: str,
    visibility: TraceVisibility = TraceVisibility.USER,
    after_seq: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """SSE 订阅某个 trace_id 的事件流。

    Args:
        trace_id: 轨迹 ID
        visibility: 可见级别过滤
        after_seq: 断线重连的起始 seq
    """
    last_seq = after_seq or 0

    try:
        while True:
            events = get_trace_events(trace_id, after_seq=last_seq, limit=200)
            for evt in events:
                if evt.seq > last_seq:
                    last_seq = evt.seq
                    yield f"data: {json.dumps(_event_to_sse(evt), ensure_ascii=False)}\n\n"

            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        logger.debug("SSE 取消 trace_id=%s", trace_id)


def _event_to_sse(evt) -> dict:
    return {
        "seq": evt.seq,
        "event_type": evt.event_type.value if hasattr(evt.event_type, 'value') else evt.event_type,
        "event_status": evt.event_status.value if hasattr(evt.event_status, 'value') else evt.event_status,
        "node_name": evt.node_name,
        "title": evt.title,
        "message": evt.message,
        "duration_ms": evt.duration_ms,
        "error_type": evt.error_type,
        "error_message": evt.error_message,
    }
