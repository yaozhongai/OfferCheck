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
# OfferCheck 阶段执行（瘦调用 nexa_agent 核心）
# ---------------------------------------------------------------------------

class RunStageRequest(BaseModel):
    """OfferCheck 阶段执行请求

    server 瘦 handler：把输入透传给 nexa_agent 核心 ReflexionReActAgent.execute()，
    按 stage 加载对应阶段任务定义 prompt。stage=None 时为纯通用引擎问答。
    """
    input: str = Field(..., min_length=1, max_length=20000,
                       description="用户输入：offer/JD 文本、公司名或聊天记录")
    stage: Optional[str] = Field(
        None, description="阶段：stage1=选岗调研 | stage4=offer证伪（None=通用引擎）"
    )
    image_path: Optional[str] = Field(None, description="可选：关联图片路径")
    session_id: Optional[str] = Field(None, description="可选：会话 ID")
    user_id: Optional[str] = Field(None, description="可选：用户 ID（LTM 隔离）")
    max_trials: Optional[int] = Field(None, ge=1, le=5, description="覆盖最大 Trial 数")
    max_steps: Optional[int] = Field(None, ge=1, le=32, description="覆盖每轮最大步数")
    answer_mode: Optional[bool] = Field(
        False, description="追问回答模式：允许基于已有结论对话式作答并逐 token 流式（用于 followup）"
    )
    auto_route: Optional[bool] = Field(
        False, description="followup 轻量 stage 路由：追问明显属于其他阶段能力时自动切换 stage prompt（关键词门 + fast 层确认）"
    )
    output_lang: Optional[str] = Field(
        None, description="可选：显式输出语言 'en'|'zh'（评审 1.10）。指定则优先，避免内容检测阈值把混合语言判翻；未指定回退内容检测"
    )


class RunStageResponse(BaseModel):
    """OfferCheck 阶段执行响应"""
    success: bool
    stage: Optional[str] = None
    answer: str
    trials_used: int = 0
    trial_details: List[Dict[str, Any]] = Field(default_factory=list)
    reflections: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    verdict: Optional[Dict[str, Any]] = Field(None, description="结构化裁定（评审 3.2）：submit_verdict 路径直传，供前端裁定卡免文本解析")


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
