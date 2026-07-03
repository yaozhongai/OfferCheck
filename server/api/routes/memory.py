"""
记忆管理接口 — V0 (LTM Schema)

会话记忆、长期记忆管理、遗忘请求。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from server.api.deps import MemoryListResponse, get_long_term_memory, get_short_term_memory
from nexa_agent.logger import get_logger

logger = get_logger("api_memory")
router = APIRouter(prefix="/api/v0/memory", tags=["memory"])


# ── 会话记忆 ──

@router.get("/session/{session_id}", summary="获取会话记忆")
async def get_session_memory(session_id: str):
    stm = get_short_term_memory()
    history = []
    for m in stm.get_history(session_id):
        history.append({"role": m.role, "content": m.content[:500]})
    return {"session_id": session_id, "items": history, "count": len(history)}


@router.delete("/session/{session_id}", summary="清除会话记忆")
async def clear_session_memory(session_id: str):
    stm = get_short_term_memory()
    stm.clear_session(session_id)
    logger.info("会话记忆已清除 session=%s", session_id)
    return {"session_id": session_id, "status": "cleared"}


# ── LTM 管理 ──

@router.get("/ltm", summary="查看长期记忆")
async def list_ltm_memories(
    user_id: str = Query(..., description="用户 ID"),
    memory_type: Optional[str] = Query(None, description="preference / fact / experience"),
    project_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    from server.memory.ltm_schema import LTMMemoryType, LTMMemoryStatus
    ltm = get_long_term_memory()
    mt = LTMMemoryType(memory_type) if memory_type else None
    items = ltm.list_user_memories(
        user_id=user_id, memory_type=mt, project_id=project_id or "",
        status=LTMMemoryStatus.ACTIVE, limit=limit, offset=offset,
    )
    return {
        "user_id": user_id,
        "items": [i.model_dump() for i in items],
        "total": len(items),
        "limit": limit,
        "offset": offset,
    }


@router.delete("/ltm/{memory_id}", summary="遗忘指定记忆")
async def forget_memory(
    memory_id: str,
    user_id: str = Query(..., description="用户 ID"),
    reason: Optional[str] = Query(None),
):
    ltm = get_long_term_memory()
    result = ltm.forget_memory(user_id=user_id, memory_id=memory_id, reason=reason or "")
    return {"memory_id": memory_id, "status": result.status.value}


@router.patch("/ltm/{memory_id}", summary="修改长期记忆")
async def update_memory_item(
    memory_id: str,
    user_id: str = Query(..., description="用户 ID"),
    content: Optional[str] = None,
    title: Optional[str] = None,
    importance: Optional[float] = None,
):
    ltm = get_long_term_memory()
    patch = {}
    if content is not None:
        patch["content"] = content
    if title is not None:
        patch["title"] = title
    if importance is not None:
        patch["importance"] = importance
    item = ltm.update_memory_item(
        user_id=user_id, memory_id=memory_id, patch=patch,
        reason="api_update", actor="user",
    )
    return {"memory_id": memory_id, "version": item.version}
