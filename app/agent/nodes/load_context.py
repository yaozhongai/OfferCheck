"""load_short_term_context — 加载上下文（STM + LTM）

STM: ShortTermMemory.get_recent_context() → short_term_context
LTM: LongTermMemory.long_term_memory_patch() → memory_candidates
"""

import time
from app.agent.state import AgentState, StepStatus, trace_patch
from app.api.deps import get_short_term_memory, get_long_term_memory
from app.utils.logger_config import get_logger

logger = get_logger("node.load_context")


def load_short_term_context(state: AgentState) -> dict:
    t0 = time.time()
    step = state.get("step_count", 0) + 1
    session_id = state.get("session_id", "")
    request_id = state.get("request_id", "")

    try:
        stm = get_short_term_memory()
        context_items = stm.get_recent_context(
            session_id, limit_turns=6, token_budget=3000,
        )

        # 附加上当前请求的图片引用
        asset_refs = []
        for ref in state.get("image_refs", []):
            asset_refs.append({
                "asset_type": "image",
                "image_id": ref.image_id,
                "path": ref.path,
                "mime_type": ref.mime_type,
            })

        result_items = []
        for item in context_items:
            d = {
                "role": item.role,
                "content": item.content,
                "entry_type": item.entry_type,
                "created_at": item.created_at,
                "asset_refs": item.asset_refs,
            }
            result_items.append(d)

        if asset_refs and result_items:
            for item in reversed(result_items):
                if item["role"] == "user":
                    item["asset_refs"] = asset_refs
                    break

        elapsed = int((time.time() - t0) * 1000)

        # ── 构建 Trace 可见的上下文摘要 ──
        context_summary = _build_context_summary(result_items)

        # ── LTM 检索 ──
        ltm_items = 0
        try:
            ltm = get_long_term_memory()
            uid = state.get("user_id") or session_id
            patch = ltm.long_term_memory_patch(
                user_id=uid, query=state.get("user_input", ""),
                request_id=request_id, session_id=session_id,
            )
            ltm_items = len(patch.get("memory_candidates", []))
        except Exception as exc:
            logger.warning("LTM 检索失败: %s", exc)
            patch = {"memory_candidates": []}

        logger.info("LOAD_CONTEXT session=%s stm=%d ltm=%d %dms | %s",
                     session_id, len(result_items), ltm_items, elapsed, context_summary["preview"])

        return {
            "short_term_context": result_items,
            "memory_candidates": patch.get("memory_candidates", []),
            **trace_patch(
                step=step, node="load_short_term_context", action="load_context",
                status=StepStatus.SUCCESS,
                reason=f"stm={len(result_items)} ltm={ltm_items}",
                output_summary=context_summary["preview"],
                latency_ms=elapsed,
            ),
            "step_count": step,
        }

    except Exception as exc:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("短期记忆读取失败，继续执行: %s", exc)
        return {
            "short_term_context": [],
            **trace_patch(
                step=step, node="load_short_term_context", action="load_context",
                status=StepStatus.SUCCESS,
                reason=f"memory read failed, continuing: {exc}",
                latency_ms=elapsed,
            ),
            "step_count": step,
        }


def _build_context_summary(items: list[dict]) -> dict:
    """从上下文中提取 Trace 可展示的摘要"""
    if not items:
        return {"turn_count": 0, "total_chars": 0, "preview": "(empty)"}

    total_chars = sum(len(it.get("content", "")) for it in items)
    # 按 turn_id 粗估轮数（entry_id 中不直接有 turn_id，用去重 entry_type+content 分区）
    turns = set()
    for it in items:
        tid = it.get("turn_id") or it.get("entry_id") or ""
        turns.add(tid)

    # 每轮取 user/assistant 的首条作为预览
    preview_parts = []
    for it in items:
        etype = it.get("entry_type", "")
        role = it.get("role", "")
        content = it.get("content", "")
        if etype in ("user_message",) or (role == "user" and not preview_parts):
            preview_parts.append(f"[Q] {content[:80]}")
        elif etype in ("final_answer", "assistant_message"):
            preview_parts.append(f"[A] {content[:80]}")

    preview = " | ".join(preview_parts[-6:])  # 最多展示最近 3 对 QA
    if not preview:
        preview = f"{len(items)} entries"

    return {
        "turn_count": len(turns) or (len(items) // 3) or 1,
        "total_chars": total_chars,
        "preview": preview[:300],
    }
