"""
Nexa Agent V0 — FastAPI 入口

启动方式:
    python -m server.main
    或
    uvicorn server.main:app --reload

所有日志统一使用 nexa_agent.logger.get_logger。
"""

from __future__ import annotations

import sys
import os

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.deps import (
    HealthResponse,
    get_config,
    get_extraction_pipeline,
    get_llm_client,
    get_long_term_memory,
    get_short_term_memory,
    reset_all,
)
from server.api.routes import memory, upload, trace, run_stage
from nexa_agent.logger import get_logger

logger = get_logger("nexa_agent")


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期"""
    config = get_config()
    logger.info("══════════════════════════════════════════")
    logger.info("  Nexa Agent V0 启动中...")
    logger.info("  API: http://%s:%d", config.api_host, config.api_port)
    logger.info("  Docs: http://%s:%d/docs", config.api_host, config.api_port)
    logger.info("  DB: %s", config.db_path)
    logger.info("══════════════════════════════════════════")

    # 预热单例（extraction pipeline 已随 app/pipeline 删除，见 deps 桩说明）
    get_short_term_memory()
    get_long_term_memory()

    # 引擎模型链路（唯一真源 = nexa_agent/config.py）
    try:
        from nexa_agent.config import LLM_PROVIDER, MODEL_TIER
        logger.info("  引擎 LLM: provider=%s strong=%s fast=%s",
                    LLM_PROVIDER, MODEL_TIER["strong"]["model"],
                    MODEL_TIER["fast"]["model"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("  引擎模型信息读取失败: %s", exc)

    yield

    # 关闭
    ltm = get_long_term_memory()
    ltm.close()
    logger.info("Nexa Agent V0 已关闭")


# ---------------------------------------------------------------------------
# 创建应用
# ---------------------------------------------------------------------------

config = get_config()

app = FastAPI(
    title=config.api_title,
    version="0.1.0",
    description="Nexa Agent — 渐进式智能助手 V0: 发票/图片票据识别与问答",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(run_stage.router)
app.include_router(upload.router)
app.include_router(memory.router)
app.include_router(trace.router)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.get("/api/v0/health", response_model=HealthResponse, tags=["system"])
async def health():
    """系统健康检查 — 报告引擎模型链路（唯一真源 = nexa_agent/config.py）"""
    from nexa_agent.config import LLM_PROVIDER, MODEL_TIER, MODEL_CONFIG
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vlm_available=False,
        llm_available=bool(MODEL_CONFIG.get("api_key")),
        llm_model=f"{LLM_PROVIDER}:{MODEL_TIER['strong']['model']}",
    )


@app.post("/api/v0/reset", tags=["system"])
async def reset():
    """重置所有全局状态（仅开发调试用）"""
    reset_all()
    logger.warning("全局状态已重置")
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# 直接启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        log_level=config.log_level.lower(),
    )
