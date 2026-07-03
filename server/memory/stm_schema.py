"""
Short-Term Memory Schema — 枚举 + Pydantic 数据模型

对齐 docs/Short-Term_Memory_Schema.md §4-§6
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ======================================================================
# 枚举
# ======================================================================

class STMSessionStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class STMTurnStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class STMEntryRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    OBSERVER = "observer"


class STMEntryType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    FINAL_ANSWER = "final_answer"
    ROUTE_SUMMARY = "route_summary"
    DECISION_SUMMARY = "decision_summary"
    ACTION_SUMMARY = "action_summary"
    OBSERVATION_SUMMARY = "observation_summary"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_OBSERVATION = "tool_observation"
    VISION_OBSERVATION = "vision_observation"
    DOCUMENT_CONTEXT = "document_context"
    USER_CLARIFICATION = "user_clarification"
    HUMAN_CONFIRM_RESULT = "human_confirm_result"
    ERROR_SUMMARY = "error_summary"


class STMAssetType(str, Enum):
    IMAGE = "image"
    FILE = "file"
    URL = "url"


class STMSourceModule(str, Enum):
    FASTAPI = "fastapi"
    AGENT_NODE = "agent_node"
    REACT_SUBGRAPH = "react_subgraph"
    TOOL = "tool"
    VLM = "vlm"
    LLM = "llm"
    MEMORY = "memory"
    SYSTEM = "system"


# ======================================================================
# ObservationSource → STMSourceModule 映射
# ======================================================================

OBSERVATION_TO_STM_SOURCE = {
    "user":      STMSourceModule.FASTAPI,
    "vlm":       STMSourceModule.VLM,
    "llm":       STMSourceModule.LLM,
    "memory":    STMSourceModule.MEMORY,
    "tool":      STMSourceModule.TOOL,
    "verifier":  STMSourceModule.SYSTEM,
    "document":  STMSourceModule.MEMORY,
    "system":    STMSourceModule.SYSTEM,
}


# ======================================================================
# Pydantic 模型
# ======================================================================

class STMAssetRef(BaseModel):
    asset_id: str
    asset_type: STMAssetType
    image_id: Optional[str] = None
    file_id: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class STMSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    status: STMSessionStatus = STMSessionStatus.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    suspended_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    turn_count: int = 0
    entry_count: int = 0
    last_request_id: Optional[str] = None
    last_trace_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class STMTurn(BaseModel):
    turn_id: str
    session_id: str
    request_id: str
    trace_id: Optional[str] = None
    turn_index: int
    status: STMTurnStatus = STMTurnStatus.STARTED
    input_type: str = "text"
    user_input_summary: Optional[str] = None
    route_type: Optional[str] = None
    final_answer_summary: Optional[str] = None
    asset_refs: list[STMAssetRef] = Field(default_factory=list)
    token_estimate: Optional[int] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class STMEntry(BaseModel):
    entry_id: str
    session_id: str
    turn_id: str
    request_id: str
    trace_id: Optional[str] = None
    entry_index: int
    role: STMEntryRole
    entry_type: STMEntryType
    source_module: Optional[STMSourceModule] = None
    node_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    content: str
    structured_data: Optional[dict[str, Any]] = None
    asset_refs: list[STMAssetRef] = Field(default_factory=list)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    token_estimate: Optional[int] = None
    visible_to_llm: bool = True
    visible_to_user: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class STMContextItem(BaseModel):
    """注入 AgentState.short_term_context 的 DTO — 5 核心字段"""
    role: str
    content: str
    entry_type: str
    created_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    asset_refs: list[dict[str, Any]] = Field(default_factory=list)
    # 可选扩展
    entry_id: Optional[str] = None
    turn_id: Optional[str] = None
    request_id: Optional[str] = None
    structured_data: Optional[dict[str, Any]] = None
    source_module: Optional[str] = None
    node_name: Optional[str] = None
    importance: Optional[float] = None
    confidence: Optional[float] = None
