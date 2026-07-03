"""
Agent Trace Schema — 执行轨迹协议

AgentState 负责状态协议，Agent Trace 负责过程轨迹。
AgentTraceEvent 是细粒度执行事件，不等同于 ActionTraceItem（轻量动作摘要）。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ======================================================================
# 枚举
# ======================================================================

class TraceStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TraceEventType(str, Enum):
    TRACE_STARTED = "trace_started"
    TRACE_COMPLETED = "trace_completed"
    TRACE_FAILED = "trace_failed"

    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"

    ROUTE_DECIDED = "route_decided"

    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_COMPLETED = "model_call_completed"
    MODEL_CALL_FAILED = "model_call_failed"

    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_FAILED = "retrieval_failed"

    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"

    MEMORY_READ_STARTED = "memory_read_started"
    MEMORY_READ_COMPLETED = "memory_read_completed"
    MEMORY_WRITE_STARTED = "memory_write_started"
    MEMORY_WRITE_COMPLETED = "memory_write_completed"
    MEMORY_WRITE_SKIPPED = "memory_write_skipped"
    MEMORY_WRITE_FAILED = "memory_write_failed"

    FALLBACK_TRIGGERED = "fallback_triggered"

    HUMAN_CONFIRM_REQUIRED = "human_confirm_required"
    HUMAN_CONFIRM_COMPLETED = "human_confirm_completed"

    TOOL_CALL_PLANNED = "tool_call_planned"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"


class TraceEventStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class TraceEventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceVisibility(str, Enum):
    USER = "user"
    DEV = "dev"
    DEBUG = "debug"


class TraceNodeName(str, Enum):
    NORMALIZE_INPUT = "normalize_input"
    LOAD_SHORT_TERM_CONTEXT = "load_short_term_context"
    ROUTE_TASK = "route_task"
    VISION_DIRECT = "vision_direct"
    VISION_SCHEMA = "vision_schema"
    VISION_PERCEIVE = "vision_perceive"
    VALIDATE_DIRECT = "validate_direct"
    VALIDATE_SCHEMA = "validate_schema"
    RETRIEVE = "retrieve"
    REASON = "reason"
    VERIFY = "verify"
    RESPOND = "respond"
    UPDATE_MEMORY = "update_memory"
    FALLBACK = "fallback"
    TOOL_ACT_PLACEHOLDER = "tool_act_placeholder"


# ======================================================================
# Payload 子 Schema
# ======================================================================

class RouteTracePayload(BaseModel):
    route_type: str
    confidence: Optional[float] = None
    reason: Optional[str] = None
    matched_rules: list[str] = Field(default_factory=list)
    need_retrieve: bool = False
    need_reason: bool = False
    need_verify: bool = False
    need_memory_write: bool = False
    risk_level: Optional[str] = None


class ModelCallTracePayload(BaseModel):
    provider: Optional[str] = None
    model_name: str
    node_name: str
    purpose: str
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    latency_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class RetrievalTracePayload(BaseModel):
    query_summary: Optional[str] = None
    source_type: str
    source_name: Optional[str] = None
    result_count: int = 0
    top_scores: list[float] = Field(default_factory=list)
    used_short_term_memory: bool = False
    used_long_term_memory: bool = False
    used_document_context: bool = False


class ValidationTracePayload(BaseModel):
    validator_name: str
    passed: bool
    score: Optional[float] = None
    issues: list[str] = Field(default_factory=list)
    revised: bool = False
    revised_answer_summary: Optional[str] = None


class MemoryWriteTracePayload(BaseModel):
    need_memory_write: bool
    target: Optional[str] = None
    written: bool = False
    skipped_reason: Optional[str] = None
    memory_item_count: int = 0


class ErrorTracePayload(BaseModel):
    error_type: str
    message: str
    node_name: Optional[str] = None
    recoverable: bool = True
    retryable: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


# ======================================================================
# 主线对象
# ======================================================================

class AgentTraceRun(BaseModel):
    trace_id: str
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    route_type: Optional[str] = None
    status: TraceStatus = TraceStatus.RUNNING
    current_node: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    event_count: int = 0
    error_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    final_answer_summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentTraceEvent(BaseModel):
    event_id: str
    trace_id: str
    request_id: str
    session_id: str
    seq: int
    event_type: TraceEventType
    event_status: TraceEventStatus
    event_level: TraceEventLevel = TraceEventLevel.INFO
    visibility: TraceVisibility = TraceVisibility.USER
    node_name: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    title: str
    message: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentTimelineItem(BaseModel):
    """前端展示 DTO，由 AgentTraceEvent 聚合，不落库"""
    item_id: str
    trace_id: str
    node_name: str
    title: str
    description: Optional[str] = None
    status: TraceEventStatus
    level: TraceEventLevel = TraceEventLevel.INFO
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    events: list[AgentTraceEvent] = Field(default_factory=list)
    expandable: bool = False
    default_open: bool = False
