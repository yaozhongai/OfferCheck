"""
短期记忆模块 — V0

基于内存实现，对齐 docs/Short-Term_Memory_Schema.md。
支持会话管理、对话轮次、记忆条目、上下文裁剪。

日志统一使用 nexa_agent.logger。
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from server.memory.stm_schema import (
    STMSessionStatus, STMTurnStatus, STMEntryRole, STMEntryType, STMSourceModule,
    STMAssetRef, STMSession, STMTurn, STMEntry, STMContextItem,
    OBSERVATION_TO_STM_SOURCE,
)
from nexa_agent.logger import get_logger

logger = get_logger("short_term_memory")

# ── 裁剪默认值 ──
DEFAULT_TOKEN_BUDGET = 3000
DEFAULT_LIMIT_TURNS = 6
DEFAULT_LIMIT_ENTRIES = 30
SINGLE_ENTRY_MAX_CHARS = 2000


class ShortTermMemory:
    """短期记忆管理器 — 内存实现

    对齐 STM Schema §5-§7。
    """

    def __init__(self, max_sessions: int = 1000, session_ttl_seconds: int = 3600):
        self._sessions: Dict[str, STMSession] = OrderedDict()
        self._turns: Dict[str, List[STMTurn]] = {}          # session_id → turns
        self._entries: Dict[str, List[STMEntry]] = {}       # session_id → entries
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
        logger.info("短期记忆初始化 max_sessions=%d ttl=%ds", max_sessions, session_ttl_seconds)

    # ──────────────────────────────────────────────────────────
    # Session 管理
    # ──────────────────────────────────────────────────────────

    def get_or_create_session(
        self, session_id: str, user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> STMSession:
        if session_id in self._sessions:
            s = self._sessions[session_id]
            if s.status in (STMSessionStatus.ARCHIVED, STMSessionStatus.EXPIRED):
                logger.warning("session %s 已 %s，创建新 session", session_id, s.status.value)
                return self._create_session(session_id, user_id, metadata)
            s.last_accessed_at = datetime.utcnow()
            return s
        return self._create_session(session_id, user_id, metadata)

    def _create_session(self, session_id: str, user_id: Optional[str] = None,
                        metadata: Optional[dict] = None) -> STMSession:
        self._ensure_capacity()
        s = STMSession(
            session_id=session_id, user_id=user_id,
            metadata=metadata or {},
            expires_at=(
                datetime.utcnow() + timedelta(seconds=self._session_ttl)
                if self._session_ttl > 0 else None
            ),
        )
        self._sessions[session_id] = s
        if session_id not in self._turns:
            self._turns[session_id] = []
        if session_id not in self._entries:
            self._entries[session_id] = []
        logger.debug("创建新 session: %s", session_id)
        return s

    def suspend_session(self, session_id: str, reason: Optional[str] = None) -> STMSession:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"session {session_id} 不存在")
        s.status = STMSessionStatus.SUSPENDED
        s.suspended_at = datetime.utcnow()
        logger.info("session %s 已挂起: %s", session_id, reason or "")
        return s

    def archive_session(self, session_id: str, reason: Optional[str] = None) -> STMSession:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"session {session_id} 不存在")
        s.status = STMSessionStatus.ARCHIVED
        s.archived_at = datetime.utcnow()
        logger.info("session %s 已归档: %s", session_id, reason or "")
        return s

    def cleanup_expired_sessions(self, hard_delete: bool = False) -> int:
        now = datetime.utcnow()
        expired_ids = []
        for sid, s in list(self._sessions.items()):
            if s.status == STMSessionStatus.EXPIRED:
                expired_ids.append(sid)
            elif s.expires_at and s.expires_at < now:
                s.status = STMSessionStatus.EXPIRED
                expired_ids.append(sid)

        if hard_delete:
            for sid in expired_ids:
                self._sessions.pop(sid, None)
                self._turns.pop(sid, None)
                self._entries.pop(sid, None)
            logger.info("清理 %d 个过期 session", len(expired_ids))
        return len(expired_ids)

    def session_exists(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        return s is not None and s.status == STMSessionStatus.ACTIVE

    # ──────────────────────────────────────────────────────────
    # Turn 管理
    # ──────────────────────────────────────────────────────────

    def start_turn(
        self, session_id: str, request_id: str,
        trace_id: Optional[str] = None, input_type: str = "text",
        user_input: Optional[str] = None,
        asset_refs: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> STMTurn:
        self.get_or_create_session(session_id)
        turn_index = len(self._turns[session_id]) + 1
        turn = STMTurn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id, request_id=request_id,
            trace_id=trace_id, turn_index=turn_index,
            input_type=input_type,
            user_input_summary=(user_input or "")[:200],
            asset_refs=[STMAssetRef(**r) for r in (asset_refs or [])],
            metadata=metadata or {},
        )
        self._turns[session_id].append(turn)

        # 自动写入 user_message entry
        if user_input:
            self.append_entry(
                session_id=session_id, turn_id=turn.turn_id,
                request_id=request_id, trace_id=trace_id,
                role=STMEntryRole.USER, entry_type=STMEntryType.USER_MESSAGE,
                content=user_input[:SINGLE_ENTRY_MAX_CHARS],
                asset_refs=asset_refs,
            )

        # 更新 session 计数
        s = self._sessions[session_id]
        s.last_request_id = request_id
        s.last_trace_id = trace_id
        s.last_accessed_at = datetime.utcnow()

        logger.debug("start_turn session=%s turn=%d", session_id, turn_index)
        return turn

    def complete_turn(
        self, session_id: str, turn_id: str, request_id: str,
        final_answer: Optional[str] = None, route_type: Optional[str] = None,
        status: STMTurnStatus = STMTurnStatus.COMPLETED,
        metadata: Optional[dict] = None,
    ) -> STMTurn:
        turns = self._turns.get(session_id, [])
        for turn in turns:
            if turn.turn_id == turn_id:
                turn.status = status
                turn.completed_at = datetime.utcnow()
                if final_answer:
                    turn.final_answer_summary = final_answer[:200]
                if route_type:
                    turn.route_type = route_type
                if metadata:
                    turn.metadata.update(metadata)
                s = self._sessions.get(session_id)
                if s:
                    s.turn_count = len([t for t in turns if t.status == STMTurnStatus.COMPLETED])
                    s.last_accessed_at = datetime.utcnow()
                logger.debug("complete_turn session=%s turn=%d status=%s", session_id, turn.turn_index, status.value)
                return turn
        raise KeyError(f"turn {turn_id} 不存在于 session {session_id}")

    # ──────────────────────────────────────────────────────────
    # Entry 管理
    # ──────────────────────────────────────────────────────────

    def append_entry(
        self, session_id: str, turn_id: str, request_id: str,
        role: STMEntryRole, entry_type: STMEntryType, content: str,
        trace_id: Optional[str] = None,
        source_module: Optional[STMSourceModule] = None,
        node_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        structured_data: Optional[dict] = None,
        asset_refs: Optional[list[dict]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        visible_to_llm: bool = True,
        visible_to_user: bool = False,
        metadata: Optional[dict] = None,
    ) -> STMEntry:
        entries = self._entries.get(session_id, [])
        entry_index = len(entries) + 1

        # 内容裁剪
        safe_content = content[:SINGLE_ENTRY_MAX_CHARS]

        entry = STMEntry(
            entry_id=str(uuid.uuid4()),
            session_id=session_id, turn_id=turn_id,
            request_id=request_id, trace_id=trace_id,
            entry_index=entry_index, role=role, entry_type=entry_type,
            source_module=source_module, node_name=node_name,
            tool_call_id=tool_call_id, tool_name=tool_name,
            content=safe_content, structured_data=structured_data,
            asset_refs=[STMAssetRef(**r) for r in (asset_refs or [])],
            importance=importance, confidence=confidence,
            visible_to_llm=visible_to_llm, visible_to_user=visible_to_user,
            metadata=metadata or {},
        )
        if session_id not in self._entries:
            self._entries[session_id] = []
        self._entries[session_id].append(entry)

        s = self._sessions.get(session_id)
        if s:
            s.entry_count = len(self._entries[session_id])
            s.last_accessed_at = datetime.utcnow()

        logger.debug("append_entry session=%s index=%d type=%s", session_id, entry_index, entry_type.value)
        return entry

    # ──────────────────────────────────────────────────────────
    # 上下文读取
    # ──────────────────────────────────────────────────────────

    def get_recent_context(
        self, session_id: str,
        limit_turns: int = DEFAULT_LIMIT_TURNS,
        limit_entries: int = DEFAULT_LIMIT_ENTRIES,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        include_assets: bool = True,
        strategy: str = "recent_first",
    ) -> list[STMContextItem]:
        entries = self._entries.get(session_id, [])
        if not entries:
            return []

        # 只看 visible_to_llm 的
        visible = [e for e in entries if e.visible_to_llm]
        if not visible:
            return []

        # ── 按 turn_id 分组 ──
        turns: dict[str, list[STMEntry]] = {}
        turn_order: list[str] = []
        for e in visible:
            if e.turn_id not in turns:
                turns[e.turn_id] = []
                turn_order.append(e.turn_id)
            turns[e.turn_id].append(e)

        # 限制轮数
        if len(turn_order) > limit_turns:
            turn_order = turn_order[-limit_turns:]

        # 最后一个是当前轮，必须保留
        current_turn_id = turn_order[-1] if turn_order else None

        char_budget = token_budget * 3  # 粗略: 1 token ≈ 3 chars

        # ── 从最新往旧遍历 Turn，按 budget 裁剪 ──
        kept_turns: list[str] = []
        total_chars = 0

        for turn_id in reversed(turn_order):
            turn_entries = turns[turn_id]
            turn_chars = sum(len(e.content) for e in turn_entries)

            # 当前轮始终不丢弃
            if turn_id == current_turn_id:
                kept_turns.append(turn_id)
                total_chars += turn_chars
                continue

            # 超出预算则丢弃整个旧 Turn
            if total_chars + turn_chars > char_budget and kept_turns:
                break

            kept_turns.append(turn_id)
            total_chars += turn_chars

        # ── 恢复时间顺序 + 组内按 §10.3 优先级裁剪 ──
        kept_turns.reverse()
        items: list[STMContextItem] = []

        for turn_id in kept_turns:
            turn_entries = turns[turn_id]

            # 组内也做条目级裁剪（总量不超过 limit_entries）
            if len(turn_entries) + len(items) > limit_entries:
                # 优先保留 user_message，再保留 final_answer，其余截断
                priority = {"user_message": 0, "final_answer": 1}
                turn_entries.sort(key=lambda e: priority.get(e.entry_type.value, 99))
                turn_entries = turn_entries[:max(0, limit_entries - len(items))]

            for entry in turn_entries:
                item = STMContextItem(
                    role=entry.role.value,
                    content=entry.content,
                    entry_type=entry.entry_type.value,
                    created_at=entry.created_at.timestamp(),
                    entry_id=entry.entry_id,
                    turn_id=entry.turn_id,
                    request_id=entry.request_id,
                    structured_data=entry.structured_data,
                    source_module=entry.source_module.value if entry.source_module else None,
                    node_name=entry.node_name,
                    importance=entry.importance,
                    confidence=entry.confidence,
                )
                if include_assets and entry.asset_refs:
                    item.asset_refs = [r.model_dump() for r in entry.asset_refs]
                items.append(item)

        logger.debug("get_recent_context session=%s turns=%d items=%d chars=%d budget=%d",
                     session_id, len(kept_turns), len(items), total_chars, char_budget)
        return items

    def short_term_context_patch(
        self, session_id: str, request_id: str,
        trace_id: Optional[str] = None,
        limit_turns: int = DEFAULT_LIMIT_TURNS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict:
        items = self.get_recent_context(
            session_id, limit_turns=limit_turns, token_budget=token_budget,
        )
        return {"short_term_context": [i.model_dump() for i in items]}

    def write_from_state(self, state: dict, trace_id: Optional[str] = None) -> None:
        session_id = state.get("session_id", "")
        request_id = state.get("request_id", "")
        if not session_id or not request_id:
            return

        self.get_or_create_session(session_id)
        turn = self.start_turn(
            session_id=session_id, request_id=request_id, trace_id=trace_id,
            input_type=state.get("input_type", "text").value
            if hasattr(state.get("input_type", "text"), "value")
            else str(state.get("input_type", "text")),
            user_input=state.get("user_input", ""),
        )

        # 写入 assistant 回答
        fa = state.get("final_answer") or ""
        if fa:
            self.append_entry(
                session_id=session_id, turn_id=turn.turn_id,
                request_id=request_id, trace_id=trace_id,
                role=STMEntryRole.ASSISTANT,
                entry_type=STMEntryType.FINAL_ANSWER,
                content=fa[:SINGLE_ENTRY_MAX_CHARS],
                source_module=STMSourceModule.AGENT_NODE,
                node_name="respond",
                visible_to_llm=True, visible_to_user=True,
            )

        # 写入路由摘要
        route = state.get("route_result")
        if route:
            self.append_entry(
                session_id=session_id, turn_id=turn.turn_id,
                request_id=request_id, trace_id=trace_id,
                role=STMEntryRole.SYSTEM,
                entry_type=STMEntryType.ROUTE_SUMMARY,
                content=f"路由: {route.route_type.value if hasattr(route.route_type, 'value') else route.route_type} "
                        f"置信度: {route.confidence}",
                source_module=STMSourceModule.AGENT_NODE,
                node_name="route_task",
                confidence=route.confidence,
            )

        # 写入 VLM/LM 观察
        for obs in state.get("observations", []):
            source = obs.source.value if hasattr(obs.source, 'value') else str(obs.source)
            src_module = OBSERVATION_TO_STM_SOURCE.get(source, STMSourceModule.SYSTEM)
            self.append_entry(
                session_id=session_id, turn_id=turn.turn_id,
                request_id=request_id, trace_id=trace_id,
                role=STMEntryRole.OBSERVER,
                entry_type=STMEntryType.OBSERVATION_SUMMARY,
                content=obs.content[:SINGLE_ENTRY_MAX_CHARS],
                source_module=src_module,
                node_name=getattr(obs, "node", None),
                confidence=getattr(obs, "confidence", None),
            )

        # 写入错误摘要
        for err in state.get("errors", []):
            self.append_entry(
                session_id=session_id, turn_id=turn.turn_id,
                request_id=request_id, trace_id=trace_id,
                role=STMEntryRole.SYSTEM,
                entry_type=STMEntryType.ERROR_SUMMARY,
                content=f"{err.error_type}: {err.message}",
                source_module=STMSourceModule.SYSTEM,
                node_name=getattr(err, "node", None),
            )

        self.complete_turn(
            session_id=session_id, turn_id=turn.turn_id,
            request_id=request_id, final_answer=fa,
            route_type=route.route_type.value
            if route and hasattr(route.route_type, 'value')
            else (str(route.route_type) if route else None),
        )

        logger.info("write_from_state session=%s turn=%d entries=%d",
                    session_id, turn.turn_index, len(self._entries.get(session_id, [])))

    # ──────────────────────────────────────────────────────────
    # 兼容旧接口
    # ──────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """兼容旧接口：通过 turn/entry 机制写入"""
        self.get_or_create_session(session_id)
        request_id = str(uuid.uuid4())
        turn = self.start_turn(
            session_id=session_id, request_id=request_id,
            input_type="text", user_input=content if role == "user" else "",
        )
        entry_type = STMEntryType.USER_MESSAGE if role == "user" else STMEntryType.ASSISTANT_MESSAGE
        entry_role = STMEntryRole.USER if role == "user" else STMEntryRole.ASSISTANT
        self.append_entry(
            session_id=session_id, turn_id=turn.turn_id,
            request_id=request_id, role=entry_role, entry_type=entry_type,
            content=content[:SINGLE_ENTRY_MAX_CHARS],
            source_module=STMSourceModule.FASTAPI,
        )
        self.complete_turn(
            session_id=session_id, turn_id=turn.turn_id,
            request_id=request_id, final_answer=content if role == "assistant" else "",
        )

    def get_history(self, session_id: str, last_n: int = 0) -> list:
        """兼容旧接口：返回 Message-like 对象列表"""
        entries = self._entries.get(session_id, [])
        visible = [e for e in entries if e.visible_to_llm]
        if last_n > 0:
            visible = visible[-last_n:]
        # 返回兼容 dict 格式
        result = []
        for e in visible:
            result.append(type('_Msg', (), {
                'role': e.role.value,
                'content': e.content,
                'timestamp': e.created_at.timestamp(),
                'metadata': {'entry_type': e.entry_type.value},
            })())
        return result

    def clear_session(self, session_id: str) -> None:
        self.archive_session(session_id)
        # 物理清理
        self._sessions.pop(session_id, None)
        self._turns.pop(session_id, None)
        self._entries.pop(session_id, None)
        logger.info("session %s 已清除", session_id)

    def get_session_count(self) -> int:
        return len(self._sessions)

    # ──────────────────────────────────────────────────────────
    # 内部
    # ──────────────────────────────────────────────────────────

    def _ensure_capacity(self) -> None:
        while len(self._sessions) >= self._max_sessions:
            oldest_key, _ = self._sessions.popitem(last=False)
            self._turns.pop(oldest_key, None)
            self._entries.pop(oldest_key, None)
            logger.debug("淘汰最旧会话: %s", oldest_key)
