"""
对话接口 — V0 (LangGraph 原生 + Trace)

FastAPI 是唯一业务入口。
每请求创建 AgentTraceRun，完成后写入 trace_completed。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ChatRequest, ChatResponse, ErrorResponse
from app.agent.state import (
    create_initial_state, to_public_response,
    ImageRef,
)
from app.agent.graph import get_graph
from app.trace.service import create_trace_run, complete_trace_run, fail_trace_run
from app.utils.logger_config import get_logger

logger = get_logger("api_chat")
router = APIRouter(prefix="/api/v0", tags=["chat"])


# ── Node → Trace 事件类型映射 ──

_NODE_EVENT_MAP = {
    "load_short_term_context": "memory_read_completed",
    "retrieve":                 "retrieval_completed",
    "route_task":               "route_decided",
    "update_memory":            "memory_write_completed",
    "vision_direct":            "model_call_completed",
    "vision_schema":            "model_call_completed",
    "vision_perceive":          "model_call_completed",
    "reason":                   "model_call_completed",
    "verify":                   "validation_completed",
    "validate_direct":          "validation_completed",
    "validate_schema":          "validation_completed",
}

_MEMORY_SKIP_NODES = {"update_memory"}
_MODEL_CALL_NODES = {"vision_direct", "vision_schema", "vision_perceive", "reason", "verify"}


@router.post("/chat", response_model=ChatResponse,
             responses={500: {"model": ErrorResponse}},
             summary="多轮对话", description="LangGraph 原生多路径 Agent")
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("请求 session=%s msg_len=%d", request.session_id, len(request.message))

    imgs = [ImageRef(image_id=f"req_{request.session_id}", path=request.image_path)] if request.image_path else []
    state = create_initial_state(
        user_input=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        image_refs=imgs,
        input_metadata=request.metadata or {},
    )

    import time as _time
    _t0 = _time.time()

    request_id = state.get("request_id", "")
    create_trace_run(request_id=request_id, session_id=request.session_id)

    try:
        graph = get_graph()
        result = graph.invoke(state, {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": 30,
        })
        latency_ms = int((_time.time() - _t0) * 1000)
        pub = to_public_response(result)

        # ── 从 action_trace 发射节点级 Trace 事件 ──
        from app.trace.service import emit_trace_event as _emit
        from app.trace.schema import TraceEventType as _TET

        for item in result.get("action_trace", []):
            if not item.node:
                continue

            node = item.node
            status_val = item.status.value if hasattr(item.status, 'value') else str(item.status)

            # 判断 skip 节点
            if status_val == "skipped" and node in _MEMORY_SKIP_NODES:
                _emit(
                    trace_id=request_id, request_id=request_id,
                    session_id=request.session_id,
                    event_type=_TET.MEMORY_WRITE_SKIPPED,
                    title=f"{node} (skipped)",
                    node_name=node,
                    event_status="skipped",
                    message=item.reason or "no memory write",
                    duration_ms=item.latency_ms or 0,
                )
                continue

            # 按节点类型发射细粒度事件
            mapped_event = _NODE_EVENT_MAP.get(node)
            if mapped_event:
                _emit(
                    trace_id=request_id, request_id=request_id,
                    session_id=request.session_id,
                    event_type=_TET(mapped_event),
                    title=item.reason or node,
                    node_name=node,
                    event_status=status_val,
                    message=item.reason or None,
                    input_summary=item.input_summary,
                    output_summary=item.output_summary,
                    duration_ms=item.latency_ms or 0,
                    error_message=item.error_message,
                    payload={
                        "action": item.action,
                        "confidence": item.confidence,
                    },
                )

        # ── 从 model_calls 发射模型调用事件 ──
        model_calls = result.get("model_calls", [])
        for call in model_calls:
            node = getattr(call, "node", "")
            if node not in _MODEL_CALL_NODES:
                continue
            _emit(
                trace_id=request_id, request_id=request_id,
                session_id=request.session_id,
                event_type=_TET.MODEL_CALL_COMPLETED,
                title="模型调用",
                node_name=node,
                event_status="success" if getattr(call, "success", True) else "failed",
                message=getattr(call, "output_summary", None),
                input_summary=getattr(call, "input_summary", None),
                output_summary=getattr(call, "output_summary", None),
                payload={
                    "provider": getattr(call, "provider", None),
                    "model_name": getattr(call, "model_name", ""),
                    "node_name": node,
                    "purpose": node,
                    "latency_ms": getattr(call, "latency_ms", None),
                    "prompt_tokens": getattr(call, "prompt_tokens", None),
                    "completion_tokens": getattr(call, "completion_tokens", None),
                    "total_tokens": getattr(call, "total_tokens", None),
                    "success": getattr(call, "success", True),
                },
                duration_ms=getattr(call, "latency_ms", None),
                error_message=getattr(call, "error_message", None),
            )

        vision_nodes = {"vision_direct", "vision_schema", "vision_perceive"}
        llm_nodes = {"reason", "verify", "route_task"}
        vlm_count = sum(1 for c in model_calls if getattr(c, "node", "") in vision_nodes)
        llm_count = sum(1 for c in model_calls if getattr(c, "node", "") in llm_nodes)

        complete_trace_run(request_id, final_answer_summary=pub.get("answer", "")[:200])

        return ChatResponse(
            request_id=request_id,
            session_id=request.session_id,
            response=pub.get("answer", ""),
            status="ok",
            task_type=pub["route"]["route_type"] if pub.get("route") else "",
            confidence=pub.get("confidence"),
            execution_path=[t["node"] for t in pub.get("trace", [])],
            llm_calls=llm_count,
            vlm_calls=vlm_count,
            latency_ms=latency_ms,
            metadata={"trace": pub.get("trace", []), "errors": pub.get("errors", [])},
        )
    except Exception as exc:
        fail_trace_run(request_id, error_type=type(exc).__name__, error_message=str(exc))
        raise
