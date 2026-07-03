"""
Long-Term Memory Schema — 枚举 + Pydantic 数据模型

对齐 docs/Long-Term_Memory_Schema.md §5-§7
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ======================================================================
# 枚举
# ======================================================================

class LTMMemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    EXPERIENCE = "experience"


class LTMMemoryScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    SESSION = "session"
    SYSTEM = "system"


class LTMMemoryStatus(str, Enum):
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    DELETED = "deleted"


class LTMSourceType(str, Enum):
    USER_EXPLICIT = "user_explicit"
    USER_MESSAGE = "user_message"
    ASSISTANT_SUMMARY = "assistant_summary"
    REFLECTION = "reflection"
    SYSTEM_IMPORT = "system_import"
    MANUAL_EDIT = "manual_edit"


class LTMSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    RESTRICTED = "restricted"


class LTMEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    RESTORED = "restored"
    DELETED = "deleted"
    USED = "used"


class LTMForgetStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# ======================================================================
# Pydantic 模型
# ======================================================================

class MemoryWriteCandidate(BaseModel):
    """Reflection / Memory Gate 输出给 LTM 的写入候选"""
    user_id: str
    memory_type: LTMMemoryType
    scope: LTMMemoryScope = LTMMemoryScope.USER
    project_id: Optional[str] = None
    content: str
    normalized_key: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: LTMSensitivity = LTMSensitivity.LOW
    source_type: LTMSourceType = LTMSourceType.ASSISTANT_SUMMARY
    source_session_id: Optional[str] = None
    source_request_id: Optional[str] = None
    source_trace_id: Optional[str] = None
    source_turn_id: Optional[str] = None
    source_entry_ids: list[str] = Field(default_factory=list)
    source_summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LTMMemoryItem(BaseModel):
    memory_id: str
    user_id: str
    scope: LTMMemoryScope = LTMMemoryScope.USER
    project_id: Optional[str] = None
    memory_type: LTMMemoryType
    status: LTMMemoryStatus = LTMMemoryStatus.ACTIVE
    title: Optional[str] = None
    content: str
    normalized_key: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: LTMSensitivity = LTMSensitivity.LOW
    source_type: LTMSourceType = LTMSourceType.ASSISTANT_SUMMARY
    source_session_id: Optional[str] = None
    source_request_id: Optional[str] = None
    source_trace_id: Optional[str] = None
    source_turn_id: Optional[str] = None
    source_entry_ids: list[str] = Field(default_factory=list)
    source_summary: Optional[str] = None
    version: int = 1
    supersedes_memory_id: Optional[str] = None
    superseded_by_memory_id: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LTMRetrievalResult(BaseModel):
    """检索结果 DTO，注入 AgentState 前转为 EvidenceItem"""
    memory_id: str
    memory_type: LTMMemoryType
    scope: LTMMemoryScope
    content: str
    title: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyword_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recency_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LTMMemoryEvent(BaseModel):
    event_id: str
    memory_id: str
    user_id: str
    event_type: LTMEventType
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    actor: str = "assistant"
    reason: Optional[str] = None
    old_snapshot: Optional[dict[str, Any]] = None
    new_snapshot: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LTMForgetRequest(BaseModel):
    forget_request_id: str
    user_id: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    query: Optional[str] = None
    target_memory_ids: list[str] = Field(default_factory=list)
    status: LTMForgetStatus = LTMForgetStatus.PENDING
    strategy: str = "soft_forget"
    reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ======================================================================
# 注入 AgentState 的转换
# ======================================================================

def to_evidence_item(result: LTMRetrievalResult) -> dict:
    """长期记忆检索结果 → EvidenceItem dict"""
    return {
        "source_type": "memory",
        "source_id": result.memory_id,
        "title": result.title or result.memory_type.value,
        "content": result.content[:500],
        "score": result.score,
        "metadata": {
            "memory_type": result.memory_type.value,
            "scope": result.scope.value,
            "tags": result.tags,
            "importance": result.importance,
            "confidence": result.confidence,
        },
    }
