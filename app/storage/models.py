"""
SQLAlchemy ORM 模型

image_analysis_cache / agent_trace_runs / agent_trace_events
ltm_memory_items / ltm_memory_events / ltm_forget_requests / kb_documents
"""

import json
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Index,
)
from app.storage.database import Base


# ======================================================================
# 图片分析缓存
# ======================================================================

class ImageAnalysisCache(Base):
    __tablename__ = "image_analysis_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(128), unique=True, nullable=False, index=True)
    file_sha256 = Column(String(128), unique=True, nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    image_path = Column(String(1024), nullable=False)
    filename = Column(String(512))
    content_type = Column(String(128))
    model_name = Column(String(256))
    vlm_text = Column(Text)
    structured_data_json = Column(Text, default="{}")
    status = Column(String(32), nullable=False, default="success")
    latency_ms = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def structured_data(self) -> dict:
        return json.loads(self.structured_data_json) if self.structured_data_json else {}

    @structured_data.setter
    def structured_data(self, value: dict):
        self.structured_data_json = json.dumps(value or {}, ensure_ascii=False)


# ======================================================================
# LTM 表
# ======================================================================

class LTMMemoryItem(Base):
    __tablename__ = "ltm_memory_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(String(128), nullable=False, index=True)
    scope = Column(String(32), nullable=False, default="user")
    project_id = Column(String(128), index=True)
    memory_type = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="active", index=True)

    title = Column(String(256))
    content = Column(Text, nullable=False)
    normalized_key = Column(String(256), index=True)

    subject = Column(String(256))
    predicate = Column(String(256))
    object_value = Column(Text)

    tags_json = Column(Text, default="[]")
    confidence = Column(Float, default=0.8)
    importance = Column(Float, default=0.5)
    sensitivity = Column(String(32), default="low")

    source_type = Column(String(32), default="assistant_summary")
    source_session_id = Column(String(128))
    source_request_id = Column(String(128))
    source_trace_id = Column(String(128))
    source_turn_id = Column(String(128))
    source_entry_ids_json = Column(Text, default="[]")
    source_summary = Column(Text)

    version = Column(Integer, default=1)
    supersedes_memory_id = Column(String(128))
    superseded_by_memory_id = Column(String(128))

    valid_from = Column(DateTime)
    valid_until = Column(DateTime)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    use_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(Text, default="{}")

    @property
    def tags(self) -> list:
        return json.loads(self.tags_json) if self.tags_json else []

    @tags.setter
    def tags(self, value: list):
        self.tags_json = json.dumps(value, ensure_ascii=False)

    @property
    def source_entry_ids(self) -> list:
        return json.loads(self.source_entry_ids_json) if self.source_entry_ids_json else []

    @source_entry_ids.setter
    def source_entry_ids(self, value: list):
        self.source_entry_ids_json = json.dumps(value, ensure_ascii=False)


class LTMMemoryEvent(Base):
    __tablename__ = "ltm_memory_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(128), unique=True, nullable=False)
    memory_id = Column(String(128), nullable=False, index=True)
    user_id = Column(String(128), nullable=False, index=True)

    event_type = Column(String(32), nullable=False)
    request_id = Column(String(128))
    session_id = Column(String(128))
    trace_id = Column(String(128))

    actor = Column(String(32), nullable=False, default="assistant")
    reason = Column(Text)

    old_snapshot_json = Column(Text)
    new_snapshot_json = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, default="{}")


class LTMForgetRequest(Base):
    __tablename__ = "ltm_forget_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forget_request_id = Column(String(128), unique=True, nullable=False)
    user_id = Column(String(128), nullable=False, index=True)

    request_id = Column(String(128))
    session_id = Column(String(128))
    trace_id = Column(String(128))

    query = Column(Text)
    target_memory_ids_json = Column(Text, default="[]")
    status = Column(String(32), nullable=False, default="pending")
    strategy = Column(String(32), nullable=False, default="soft_forget")
    reason = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    metadata_json = Column(Text, default="{}")


# ======================================================================
# Trace 表
# ======================================================================

class AgentTraceRun(Base):
    __tablename__ = "agent_trace_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(128), unique=True, nullable=False, index=True)
    request_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    user_id = Column(String(128))
    route_type = Column(String(64))
    status = Column(String(32), nullable=False, default="running")
    current_node = Column(String(64))
    started_at = Column(Float, nullable=False)
    finished_at = Column(Float)
    duration_ms = Column(Integer)
    event_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    model_call_count = Column(Integer, default=0)
    tool_call_count = Column(Integer, default=0)
    final_answer_summary = Column(Text)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)


class AgentTraceEvent(Base):
    __tablename__ = "agent_trace_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(128), unique=True, nullable=False)
    trace_id = Column(String(128), nullable=False, index=True)
    request_id = Column(String(128), nullable=False)
    session_id = Column(String(128), nullable=False)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(64), nullable=False)
    event_status = Column(String(32), nullable=False, default="success")
    event_level = Column(String(32), nullable=False, default="info")
    visibility = Column(String(32), nullable=False, default="user")
    node_name = Column(String(64))
    title = Column(String(256), nullable=False)
    message = Column(Text)
    input_summary = Column(Text)
    output_summary = Column(Text)
    payload_json = Column(Text)
    duration_ms = Column(Integer)
    error_type = Column(String(64))
    error_message = Column(Text)
    created_at = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_trace_events_seq", "trace_id", "seq"),
    )


# ======================================================================
# 知识库表
# ======================================================================

class KBDocument(Base):
    __tablename__ = "kb_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(128), unique=True, nullable=False, index=True)
    title = Column(String(512))
    content = Column(Text, nullable=False)
    content_full = Column(Text)
    content_hash = Column(String(128), unique=True)
    source_url = Column(Text)
    doc_type = Column(String(32), nullable=False, default="web_article")
    tags_json = Column(Text, default="[]")
    extracted_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    use_count = Column(Integer, default=0)
    status = Column(String(16), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
