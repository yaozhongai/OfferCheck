"""
Pydantic 请求/响应模型 — V0

FastAPI 接口层的入参校验和出参序列化。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: str = Field(..., description="错误描述")
    detail: Optional[str] = Field(None, description="详细堆栈（仅开发环境）")
    session_id: Optional[str] = Field(None)


class HealthResponse(BaseModel):
    """健康检查"""
    status: str = "ok"
    version: str = "0.1.0"
    vlm_available: bool = False
    llm_available: bool = False
    llm_model: str = ""


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., min_length=1, max_length=128,
                            description="会话 ID，用于上下文关联")
    message: str = Field(..., min_length=1, max_length=10000,
                         description="用户输入文本")
    user_id: Optional[str] = Field(None, description="可选：用户 ID，用于 LTM 隔离")
    image_path: Optional[str] = Field(None, description="可选：关联图片路径")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元信息")


class ChatResponse(BaseModel):
    """对话响应 — 对齐 AgentState_Schema.md to_public_response"""
    request_id: str = ""
    session_id: str
    response: str
    status: str = "ok"
    task_type: str = ""
    confidence: Optional[float] = None
    execution_path: List[str] = Field(default_factory=list)
    llm_calls: int = 0
    vlm_calls: int = 0
    latency_ms: float = 0.0
    state: str = Field("ok", description="兼容旧字段")
    reflection: Optional[Dict[str, Any]] = Field(None, description="兼容旧字段")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 文件上传
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """文件上传响应"""
    session_id: str
    filename: str
    file_path: str = Field(..., description="服务器端保存路径")
    file_size: int
    content_type: Optional[str] = None
    file_id: str = ""
    file_sha256: str = ""


class ImageAnalysisRequest(BaseModel):
    """上传文件的一次性图片识别请求"""
    session_id: str = Field(..., min_length=1, max_length=128)
    file_id: Optional[str] = None
    file_sha256: Optional[str] = None
    file_path: str
    filename: Optional[str] = None
    content_type: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    """图片识别缓存响应"""
    session_id: str
    file_id: str
    file_sha256: str = ""
    file_path: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    status: str = "success"
    cached: bool = False
    model_name: str = ""
    vlm_text: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 记忆管理
# ---------------------------------------------------------------------------

class MemoryListResponse(BaseModel):
    """记忆列表响应"""
    total: int
    items: List[Dict[str, Any]]
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------

class StateMachineStatus(BaseModel):
    """状态机状态"""
    session_id: str
    current_state: str
    transition_count: int
    history: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
