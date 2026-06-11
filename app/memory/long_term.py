"""
长期记忆模块 — V0 (LTM Schema, SQLAlchemy)

对齐 docs/Long-Term_Memory_Schema.md。
Preference / Fact / Experience 统一存储在 ltm_memory_items 表。
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.storage.database import get_session, init_db
from app.storage.models import (
    LTMMemoryItem as LTMItemORM,
    LTMMemoryEvent as LTMEventORM,
    LTMForgetRequest as LTMForgetORM,
)
from app.memory.ltm_schema import (
    LTMMemoryItem, LTMRetrievalResult, LTMMemoryEvent, LTMForgetRequest,
    MemoryWriteCandidate,
    LTMMemoryType, LTMMemoryScope, LTMMemoryStatus,
    LTMSourceType, LTMSensitivity,
    LTMEventType, LTMForgetStatus,
    to_evidence_item,
)
from app.utils.logger_config import get_logger

logger = get_logger("long_term_memory")

DEFAULT_LIMIT = 8
SINGLE_MAX_CHARS = 500


class LongTermMemory:
    """长期记忆管理器 — LTM Schema 实现"""

    def __init__(self, db_path: str = ""):
        init_db()
        logger.info("LTM 初始化完成 (LTMMemoryItem + Events + Forget)")

    # ── 写入 ──

    def upsert_memory(self, candidate: MemoryWriteCandidate) -> LTMMemoryItem:
        s = get_session()
        try:
            existing = None
            if candidate.normalized_key:
                existing = s.query(LTMItemORM).filter_by(
                    user_id=candidate.user_id,
                    memory_type=candidate.memory_type.value,
                    normalized_key=candidate.normalized_key,
                    status="active",
                ).first()
            if existing:
                return self._update_existing(s, existing, candidate)
            return self._create_new(s, candidate)
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def _create_new(self, s, c: MemoryWriteCandidate) -> LTMMemoryItem:
        mid = str(uuid.uuid4())
        now = datetime.utcnow()
        row = LTMItemORM(
            memory_id=mid, user_id=c.user_id,
            scope=c.scope.value, project_id=c.project_id,
            memory_type=c.memory_type.value, status="active",
            title=c.subject or c.content[:80],
            content=c.content[:SINGLE_MAX_CHARS],
            normalized_key=c.normalized_key,
            subject=c.subject, predicate=c.predicate, object_value=c.object_value,
            confidence=c.confidence, importance=c.importance,
            sensitivity=c.sensitivity.value,
            source_type=c.source_type.value,
            source_session_id=c.source_session_id,
            source_request_id=c.source_request_id,
            source_trace_id=c.source_trace_id,
            source_turn_id=c.source_turn_id,
            source_summary=c.source_summary,
            version=1, use_count=0, created_at=now, updated_at=now,
        )
        row.tags = c.tags
        row.source_entry_ids = c.source_entry_ids
        s.add(row)
        s.commit()
        self._write_event(s, mid, c.user_id, LTMEventType.CREATED,
                          new_snapshot={"content": c.content[:200]})
        logger.info("LTM created %s type=%s", mid, c.memory_type.value)
        return self._orm_to_item(row)

    def _update_existing(self, s, existing, c: MemoryWriteCandidate) -> LTMMemoryItem:
        old_v = existing.version
        old_c = existing.content
        existing.content = c.content[:SINGLE_MAX_CHARS]
        existing.confidence = c.confidence
        existing.importance = c.importance
        existing.version = old_v + 1
        existing.updated_at = datetime.utcnow()
        existing.source_session_id = c.source_session_id or existing.source_session_id
        if c.tags:
            existing.tags = list(set((existing.tags or []) + c.tags))
        s.commit()
        self._write_event(s, existing.memory_id, c.user_id, LTMEventType.UPDATED,
                          old_snapshot={"content": old_c[:200]},
                          new_snapshot={"content": c.content[:200]})
        logger.info("LTM updated %s v%d", existing.memory_id, existing.version)
        return self._orm_to_item(existing)

    def update_memory_item(self, user_id: str, memory_id: str, patch: dict,
                           reason: str, request_id: str = "", session_id: str = "",
                           trace_id: str = "", actor: str = "assistant") -> LTMMemoryItem:
        s = get_session()
        try:
            row = s.query(LTMItemORM).filter_by(memory_id=memory_id, user_id=user_id).first()
            if not row:
                raise KeyError(f"memory {memory_id} not found")
            old = {"content": (row.content or "")[:200]}
            for key in ("content", "title", "confidence", "importance", "tags", "status"):
                if key in patch:
                    if key == "tags":
                        row.tags = patch[key]
                    else:
                        setattr(row, key, patch[key])
            row.version += 1
            row.updated_at = datetime.utcnow()
            s.commit()
            self._write_event(s, memory_id, user_id, LTMEventType.UPDATED, reason=reason,
                              old_snapshot=old,
                              new_snapshot={"content": (patch.get("content") or row.content or "")[:200]},
                              request_id=request_id, session_id=session_id, trace_id=trace_id, actor=actor)
            return self._orm_to_item(row)
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── 检索 ──

    def retrieve_memories(
        self, user_id: str, query: str = "",
        memory_types: Optional[list[LTMMemoryType]] = None,
        scope: Optional[list[LTMMemoryScope]] = None,
        project_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = DEFAULT_LIMIT,
        min_score: float = 0.25,
        include_sensitive: bool = False,
    ) -> list[LTMRetrievalResult]:
        s = get_session()
        try:
            q = s.query(LTMItemORM).filter(
                LTMItemORM.user_id == user_id,
                LTMItemORM.status == "active",
            )
            if memory_types:
                q = q.filter(LTMItemORM.memory_type.in_([t.value for t in memory_types]))
            if scope:
                q = q.filter(LTMItemORM.scope.in_([s_.value for s_ in scope]))
            if project_id:
                q = q.filter(LTMItemORM.project_id == project_id)
            if not include_sensitive:
                q = q.filter(LTMItemORM.sensitivity != "restricted")

            rows = q.order_by(LTMItemORM.importance.desc(), LTMItemORM.updated_at.desc()).limit(limit * 2).all()

            results = []
            for row in rows:
                score = 0.5
                if query:
                    ql = query.lower()
                    if ql in (row.content or "").lower():
                        score = 0.8
                    if row.tags and any(ql in t.lower() for t in (row.tags or [])):
                        score = max(score, 0.7)
                if tags and not (set(tags) & set(row.tags or [])):
                    continue
                if score < min_score:
                    continue
                results.append(LTMRetrievalResult(
                    memory_id=row.memory_id, memory_type=LTMMemoryType(row.memory_type),
                    scope=LTMMemoryScope(row.scope), content=row.content or "",
                    title=row.title, tags=row.tags or [],
                    score=score, importance=row.importance, confidence=row.confidence,
                    source_summary=row.source_summary,
                ))
            results.sort(key=lambda r: (r.score, r.importance or 0), reverse=True)
            return results[:limit]
        finally:
            s.close()

    def long_term_memory_patch(
        self, user_id: str, query: str,
        request_id: str, session_id: str,
        trace_id: str = "", project_id: str = "", limit: int = DEFAULT_LIMIT,
    ) -> dict:
        if not user_id:
            return {"memory_candidates": []}
        results = self.retrieve_memories(user_id, query, project_id=project_id or None, limit=limit)
        return {"memory_candidates": [to_evidence_item(r) for r in results]}

    def record_memory_use(self, user_id: str, memory_ids: list[str],
                          request_id: str = "", session_id: str = "", trace_id: str = "") -> None:
        s = get_session()
        try:
            for mid in memory_ids:
                row = s.query(LTMItemORM).filter_by(memory_id=mid, user_id=user_id).first()
                if row:
                    row.use_count = (row.use_count or 0) + 1
                    row.last_used_at = datetime.utcnow()
                    self._write_event(s, mid, user_id, LTMEventType.USED,
                                      request_id=request_id, session_id=session_id, trace_id=trace_id)
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── 遗忘 ──

    def forget_memory(self, user_id: str, memory_id: str, reason: str = "",
                      request_id: str = "", session_id: str = "", trace_id: str = "",
                      strategy: str = "soft_forget") -> LTMForgetRequest:
        s = get_session()
        try:
            row = s.query(LTMItemORM).filter_by(memory_id=memory_id, user_id=user_id).first()
            if not row:
                raise KeyError(f"memory {memory_id} not found")
            row.status = "forgotten"
            row.updated_at = datetime.utcnow()

            fr = LTMForgetORM(
                forget_request_id=str(uuid.uuid4()), user_id=user_id,
                request_id=request_id, session_id=session_id, trace_id=trace_id,
                status="completed", strategy=strategy, reason=reason,
                created_at=datetime.utcnow(), completed_at=datetime.utcnow(),
            )
            fr.target_memory_ids = [memory_id]
            s.add(fr)
            self._write_event(s, memory_id, user_id, LTMEventType.FORGOTTEN, reason=reason)
            s.commit()
            logger.info("LTM forgotten %s", memory_id)
            return LTMForgetRequest(
                forget_request_id=fr.forget_request_id, user_id=user_id,
                target_memory_ids=[memory_id], status=LTMForgetStatus.COMPLETED,
                strategy=strategy, reason=reason,
            )
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def list_user_memories(self, user_id: str, memory_type: Optional[LTMMemoryType] = None,
                           project_id: str = "",
                           status: LTMMemoryStatus = LTMMemoryStatus.ACTIVE,
                           limit: int = 100, offset: int = 0) -> list[LTMMemoryItem]:
        s = get_session()
        try:
            q = s.query(LTMItemORM).filter_by(user_id=user_id, status=status.value)
            if memory_type:
                q = q.filter(LTMItemORM.memory_type == memory_type.value)
            if project_id:
                q = q.filter(LTMItemORM.project_id == project_id)
            rows = q.order_by(LTMItemORM.updated_at.desc()).offset(offset).limit(limit).all()
            return [self._orm_to_item(r) for r in rows]
        finally:
            s.close()

    # ── 门控：从 AgentState 提取候选 ──

    def build_memory_candidates_from_state(self, state: dict) -> list[MemoryWriteCandidate]:
        user_id = state.get("user_id") or state.get("session_id", "")
        if not user_id:
            return []

        candidates = []
        ui = state.get("user_input", "")
        fa = state.get("final_answer") or ""
        route = state.get("route_result")
        is_explicit = any(kw in ui for kw in ["记住", "以后都这样", "别忘了", "保存", "不要再"])
        is_gated = route.need_memory_write if route else False

        if not (is_explicit or is_gated):
            return []

        if fa and len(fa) > 50:
            candidates.append(MemoryWriteCandidate(
                user_id=user_id,
                memory_type=LTMMemoryType.FACT,
                content=f"Q: {ui[:200]}\nA: {fa[:300]}",
                normalized_key=_make_key(ui),
                confidence=0.7, importance=0.4,
                source_type=LTMSourceType.USER_MESSAGE if is_explicit else LTMSourceType.ASSISTANT_SUMMARY,
                source_session_id=state.get("session_id"),
                source_request_id=state.get("request_id"),
                source_summary=ui[:100],
            ))

        if is_explicit:
            candidates.append(MemoryWriteCandidate(
                user_id=user_id,
                memory_type=LTMMemoryType.PREFERENCE,
                content=ui[:500],
                normalized_key=f"user.preference.{_make_key(ui)}",
                confidence=0.9, importance=0.7,
                source_type=LTMSourceType.USER_EXPLICIT,
                source_session_id=state.get("session_id"),
                source_request_id=state.get("request_id"),
            ))

        return candidates

    # ── 内部 ──

    def _write_event(self, s, memory_id: str, user_id: str, event_type: LTMEventType,
                     reason: str = "", old_snapshot: dict | None = None,
                     new_snapshot: dict | None = None,
                     request_id: str = "", session_id: str = "", trace_id: str = "",
                     actor: str = "assistant") -> None:
        s.add(LTMEventORM(
            event_id=str(uuid.uuid4()), memory_id=memory_id, user_id=user_id,
            event_type=event_type.value,
            request_id=request_id, session_id=session_id, trace_id=trace_id,
            actor=actor, reason=reason,
            old_snapshot_json=json.dumps(old_snapshot, ensure_ascii=False) if old_snapshot else None,
            new_snapshot_json=json.dumps(new_snapshot, ensure_ascii=False) if new_snapshot else None,
            created_at=datetime.utcnow(),
        ))

    def _orm_to_item(self, row) -> LTMMemoryItem:
        return LTMMemoryItem(
            memory_id=row.memory_id, user_id=row.user_id,
            scope=LTMMemoryScope(row.scope), project_id=row.project_id,
            memory_type=LTMMemoryType(row.memory_type), status=LTMMemoryStatus(row.status),
            title=row.title, content=row.content or "",
            normalized_key=row.normalized_key,
            subject=row.subject, predicate=row.predicate, object_value=row.object_value,
            tags=row.tags or [],
            confidence=row.confidence or 0.8, importance=row.importance or 0.5,
            sensitivity=LTMSensitivity(row.sensitivity or "low"),
            source_type=LTMSourceType(row.source_type or "assistant_summary"),
            source_session_id=row.source_session_id, source_request_id=row.source_request_id,
            source_trace_id=row.source_trace_id, source_turn_id=row.source_turn_id,
            source_entry_ids=row.source_entry_ids or [], source_summary=row.source_summary,
            version=row.version or 1,
            last_used_at=row.last_used_at, use_count=row.use_count or 0,
            created_at=row.created_at or datetime.utcnow(),
            updated_at=row.updated_at or datetime.utcnow(),
        )

    def close(self) -> None:
        pass


def _make_key(text: str) -> str:
    key = re.sub(r'[^\w一-鿿]', '_', (text or "").lower())[:40]
    return key.strip('_') or "untitled"
