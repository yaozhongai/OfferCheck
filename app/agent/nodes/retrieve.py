"""retrieve — 长期记忆检索节点

读取 LTM（偏好 / 事实 / 经验）+ STM 上下文（已由 load_context 注入），合成 retrieved_context。
"""

import time
from app.agent.state import AgentState, StepStatus, EvidenceItem, trace_patch
from app.api.deps import get_long_term_memory
from app.utils.logger_config import get_logger

logger = get_logger("node.retrieve")


def retrieve(state: AgentState) -> dict:
    t0 = time.time()
    step = state.get("step_count", 0) + 1
    route = state.get("route_result")

    if route and not route.need_retrieve:
        return {
            **trace_patch(step=step, node="retrieve", action="skip",
                          latency_ms=int((time.time() - t0) * 1000),
                          status=StepStatus.SKIPPED, reason="route says no retrieve"),
            "step_count": step,
        }

    ltm = get_long_term_memory()
    ctx_items = []

    # ── STM 上下文（load_short_term_context 注入）──
    short_term = state.get("short_term_context", [])
    for m in short_term[-6:]:
        ctx_items.append(EvidenceItem(
            source_type="memory",
            content=f"{m.get('role', '')}: {m.get('content', '')[:200]}",
            title="short_term",
        ))

    # ── LTM 检索 ──
    user_id = state.get("user_id") or state.get("session_id", "")
    query = state.get("user_input", "")
    session_id = state.get("session_id", "")
    request_id = state.get("request_id", "")

    try:
        patch = ltm.long_term_memory_patch(
            user_id=user_id, query=query,
            request_id=request_id, session_id=session_id,
        )
        ltm_items = patch.get("memory_candidates", [])
        for item in ltm_items:
            ctx_items.append(EvidenceItem(
                source_type=item.get("source_type", "memory"),
                content=item.get("content", "")[:300],
                title=item.get("title", ""),
                score=item.get("score"),
                metadata=item.get("metadata", {}),
            ))
        logger.info("RETRIEVE lt=%d st=%d", len(ltm_items), len(short_term))
    except Exception as exc:
        logger.warning("LTM 检索失败: %s", exc)

    return {
        "retrieved_context": ctx_items,
        **trace_patch(step=step, node="retrieve", action="retrieve",
                      latency_ms=int((time.time() - t0) * 1000),
                      status=StepStatus.SUCCESS,
                      reason=f"lt={len(ctx_items)} st={len(short_term)}"),
        "step_count": step,
    }
