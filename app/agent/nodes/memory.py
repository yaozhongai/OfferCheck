"""update_memory — 记忆持久化节点

STM：每轮始终写入（turn + entry），失败不阻断。
LTM：门控写入（need_memory_write / 用户显式指令 / Memory Gate），失败不阻断。
"""

import time
from app.agent.state import AgentState, StepStatus, trace_patch, error_patch
from app.api.deps import get_short_term_memory, get_long_term_memory
from app.utils.logger_config import get_logger

logger = get_logger("node.memory")


def update_memory(state: AgentState) -> dict:
    t0 = time.time()
    step = state.get("step_count", 0) + 1
    sid = state.get("session_id", "")
    result = {}

    # ── 短期记忆：始终写入 ──
    try:
        stm = get_short_term_memory()
        stm.write_from_state(dict(state))
        logger.info("STM written session=%s", sid)
    except Exception as exc:
        logger.warning("STM 写入失败 session=%s: %s", sid, exc)
        result.update(error_patch(
            error_type="short_term_memory_write_failed",
            message=str(exc), node="update_memory", recoverable=True,
        ))

    # ── 长期记忆：门控写入 ──
    try:
        ltm = get_long_term_memory()
        candidates = ltm.build_memory_candidates_from_state(dict(state))
        for c in candidates:
            try:
                ltm.upsert_memory(c)
                logger.info("LTM written type=%s key=%s", c.memory_type.value, c.normalized_key)
            except Exception as exc:
                logger.warning("LTM upsert 失败: %s", exc)
                result.update(error_patch(
                    error_type="long_term_memory_write_failed",
                    message=str(exc), node="update_memory", recoverable=True,
                ))
        if not candidates:
            logger.debug("LTM skipped session=%s (no candidates)", sid)
    except Exception as exc:
        logger.warning("LTM 门控失败 session=%s: %s", sid, exc)
        result.update(error_patch(
            error_type="long_term_memory_gate_failed",
            message=str(exc), node="update_memory", recoverable=True,
        ))

    result.update({
        **trace_patch(step=step, node="update_memory", action="persist",
                      latency_ms=int((time.time() - t0) * 1000),
                      status=StepStatus.SUCCESS,
                      reason="stm_written"),
        "step_count": step,
    })
    return result
