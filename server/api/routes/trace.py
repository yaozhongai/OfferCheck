"""Trace 查询 API（评审 3.3：双轨合一）。

服务 `TraceRecorder` 落盘的 **OTel-GenAI 对齐** trace（`data/traces/<trace_id>.json`）——
与 `run_stage/stream` 发给浏览器的 SSE 事件**同源**、同一 typed schema
（`nexa_agent/trace/events.py`）。

历史注：本路由此前读 LangGraph 时代的 `server/trace_store/{service,store,sse}.py` +
`nexa_agent/trace/schema.py`（那套 typed schema 与现役事件不符、从未接主链路，是
被本次合并取代的「死轨」——已在 recorder 落地后废弃，待清理）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.trace_store.recorder import load_trace
from nexa_agent.logger import get_logger

logger = get_logger("api_trace")
router = APIRouter(prefix="/api/v0/trace", tags=["trace"])


@router.get("/{trace_id}", summary="查询整条持久化 trace（OTel-GenAI JSON）")
async def get_trace(trace_id: str):
    """返回一次 run 的完整轨迹：resource 属性 + spans（每 span 带 gen_ai.* /
    openinference.span.kind 属性）+ usage / verdict 摘要。不存在则 404。"""
    doc = load_trace(trace_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"trace 不存在或已过期: {trace_id}")
    return doc


@router.get("/{trace_id}/events", summary="查询 trace 的事件（spans）列表")
async def list_events(trace_id: str):
    """只取 spans 列表（每条含 seq / type / timestamp / attributes / 原始事件）。"""
    doc = load_trace(trace_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"trace 不存在或已过期: {trace_id}")
    return {
        "trace_id": trace_id,
        "stage": doc.get("stage"),
        "usage": doc.get("usage"),
        "verdict": doc.get("verdict"),
        "events": doc.get("spans", []),
    }
