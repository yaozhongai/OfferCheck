"""
Trace Store — SQLAlchemy 后端

表: agent_trace_runs / agent_trace_events
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from server.persistence.database import get_session, init_db
from server.persistence.models import AgentTraceRun as TraceRunModel, AgentTraceEvent as TraceEventModel
from nexa_agent.trace.schema import AgentTraceRun, AgentTraceEvent
from nexa_agent.logger import get_logger

logger = get_logger("trace_store")


class TraceStore:
    """Trace 事件存储 (SQLAlchemy)"""

    def __init__(self, db_path: str = ""):
        init_db()
        logger.info("TraceStore 初始化 (SQLAlchemy)")

    # ── Run ──

    def create_run(self, run: AgentTraceRun) -> AgentTraceRun:
        s = get_session()
        try:
            m = TraceRunModel(
                trace_id=run.trace_id, request_id=run.request_id,
                session_id=run.session_id, user_id=run.user_id,
                route_type=run.route_type, status=run.status.value,
                current_node=run.current_node,
                started_at=run.started_at.timestamp() if run.started_at else time.time(),
                created_at=time.time(), updated_at=time.time(),
            )
            s.add(m)
            s.commit()
            return run
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def complete_run(self, trace_id: str, final_answer_summary: Optional[str] = None) -> None:
        s = get_session()
        try:
            m = s.query(TraceRunModel).filter_by(trace_id=trace_id).first()
            if m:
                m.status = "completed"
                m.finished_at = time.time()
                if m.started_at:
                    m.duration_ms = int((time.time() - m.started_at) * 1000)
                m.final_answer_summary = final_answer_summary
                m.updated_at = time.time()
                s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def fail_run(self, trace_id: str) -> None:
        s = get_session()
        try:
            m = s.query(TraceRunModel).filter_by(trace_id=trace_id).first()
            if m:
                m.status = "failed"
                m.finished_at = time.time()
                m.updated_at = time.time()
                s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Event ──

    def emit_event(self, event: AgentTraceEvent) -> AgentTraceEvent:
        s = get_session()
        try:
            m = TraceEventModel(
                event_id=event.event_id, trace_id=event.trace_id,
                request_id=event.request_id, session_id=event.session_id,
                seq=event.seq, event_type=event.event_type.value,
                event_status=event.event_status.value, event_level=event.event_level.value,
                visibility=event.visibility.value, node_name=event.node_name,
                title=event.title, message=event.message,
                input_summary=event.input_summary, output_summary=event.output_summary,
                payload_json=json.dumps(event.payload, ensure_ascii=False) if event.payload else None,
                duration_ms=event.duration_ms, error_type=event.error_type,
                error_message=event.error_message,
                created_at=event.created_at.timestamp() if event.created_at else time.time(),
            )
            s.add(m)
            # 更新 runs 计数
            run = s.query(TraceRunModel).filter_by(trace_id=event.trace_id).first()
            if run:
                run.event_count = (run.event_count or 0) + 1
                run.updated_at = time.time()
                run.current_node = event.node_name
                if event.event_status.value == "failed" or "_failed" in event.event_type.value:
                    run.error_count = (run.error_count or 0) + 1
            s.commit()
            return event
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def get_events(
        self, trace_id: str, after_seq: Optional[int] = None, limit: int = 200,
    ) -> list[AgentTraceEvent]:
        s = get_session()
        try:
            q = s.query(TraceEventModel).filter_by(trace_id=trace_id).order_by(TraceEventModel.seq.asc())
            if after_seq is not None:
                q = q.filter(TraceEventModel.seq > after_seq)
            rows = q.limit(limit).all()
            return [_event_from_row(r) for r in rows]
        finally:
            s.close()

    def get_run(self, trace_id: str) -> Optional[AgentTraceRun]:
        s = get_session()
        try:
            m = s.query(TraceRunModel).filter_by(trace_id=trace_id).first()
            if not m:
                return None
            return AgentTraceRun(
                trace_id=m.trace_id, request_id=m.request_id,
                session_id=m.session_id, user_id=m.user_id,
                route_type=m.route_type, status=m.status,
                current_node=m.current_node,
                started_at=datetime.fromtimestamp(m.started_at) if m.started_at else None,
                finished_at=datetime.fromtimestamp(m.finished_at) if m.finished_at else None,
                duration_ms=m.duration_ms, event_count=m.event_count or 0,
                error_count=m.error_count or 0,
                model_call_count=m.model_call_count or 0,
                tool_call_count=m.tool_call_count or 0,
                final_answer_summary=m.final_answer_summary,
            )
        finally:
            s.close()

    def close(self):
        pass


def _event_from_row(r: TraceEventModel) -> AgentTraceEvent:
    payload = {}
    if r.payload_json:
        try:
            payload = json.loads(r.payload_json)
        except Exception:
            pass
    return AgentTraceEvent(
        event_id=r.event_id, trace_id=r.trace_id,
        request_id=r.request_id, session_id=r.session_id,
        seq=r.seq, event_type=r.event_type,
        event_status=r.event_status, event_level=r.event_level,
        visibility=r.visibility, node_name=r.node_name,
        title=r.title, message=r.message,
        input_summary=r.input_summary, output_summary=r.output_summary,
        payload=payload, duration_ms=r.duration_ms,
        error_type=r.error_type, error_message=r.error_message,
        created_at=datetime.fromtimestamp(r.created_at) if r.created_at else None,
    )
