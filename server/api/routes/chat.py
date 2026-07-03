"""
对话接口 — 待接入 nexa_agent 核心（server 变薄）

原实现驱动 app/agent 的 LangGraph orchestration，该模块已按 core+app 分层重构删除。
按开发计划，server 的 HTTP handler 将变薄为：
    载入用户 STM/LTM → 调 nexa_agent 核心 ReflexionReActAgent.execute()
    （注入 memory + trace emitter）→ 流式推事件 + 落盘 → 回写记忆。

在核心 wiring 完成前（7/4 计划），本端点返回 501，其余服务层（trace_store /
persistence / memory）保持可用，不阻塞对已完成核心引擎的独立测试。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.api.deps import ChatRequest, ChatResponse, ErrorResponse
from nexa_agent.logger import get_logger

logger = get_logger("api_chat")
router = APIRouter(prefix="/api/v0", tags=["chat"])


@router.post("/chat", response_model=ChatResponse,
             responses={501: {"model": ErrorResponse}},
             summary="多轮对话（待接入核心）",
             description="待改为瘦调用 nexa_agent 核心 ReflexionReActAgent.execute()")
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("请求 session=%s msg_len=%d（handler 待接入核心）",
                request.session_id, len(request.message))
    raise HTTPException(
        status_code=501,
        detail=(
            "chat 端点待接入 nexa_agent 核心引擎（旧 app/agent LangGraph 已移除，"
            "见 7/4 server 变薄计划）。核心引擎本身可经 CLI 独立运行："
            "python -m nexa_agent.reflexion_agent \"...\" --stage stage4"
        ),
    )
