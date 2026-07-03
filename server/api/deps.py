"""
FastAPI 依赖注入 — V0

提供全局单例（状态机、记忆、提取管线等）的懒加载获取函数。
"""

from __future__ import annotations

from functools import lru_cache

from server.config import AppConfig, get_config
from server.memory.long_term import LongTermMemory
from server.memory.short_term import ShortTermMemory
from server.api.schemas import (  # noqa: F401   — re-export for route convenience
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MemoryListResponse,
    UploadResponse,
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    StateMachineStatus,
)
from nexa_agent.llm.client import BaseLLMClient, create_llm_client
from nexa_agent.logger import get_logger

logger = get_logger("deps")

# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_short_term_memory: ShortTermMemory | None = None
_long_term_memory: LongTermMemory | None = None
_llm_client: BaseLLMClient | None = None


def get_short_term_memory() -> ShortTermMemory:
    """获取短期记忆单例"""
    global _short_term_memory
    if _short_term_memory is None:
        config = get_config()
        _short_term_memory = ShortTermMemory(
            max_sessions=config.stm_max_sessions,
            session_ttl_seconds=config.stm_session_ttl_seconds,
        )
        logger.info("短期记忆单例已初始化")
    return _short_term_memory


def get_long_term_memory() -> LongTermMemory:
    """获取长期记忆单例"""
    global _long_term_memory
    if _long_term_memory is None:
        config = get_config()
        _long_term_memory = LongTermMemory(db_path=config.db_path)
        logger.info("长期记忆单例已初始化")
    return _long_term_memory


def get_extraction_pipeline():
    """获取图像/文档提取管线单例。

    TODO(7/4 server 变薄): 旧 app/pipeline（未实现的 VLM 抽象）已删除。
    图像理解改由 nexa_agent 核心的 analyze_image / analyze_image_cloud 工具承担，
    upload 路由后续直接调核心工具，不再需要独立 pipeline 单例。
    """
    raise NotImplementedError(
        "extraction pipeline 已随 app/pipeline 删除；upload 路由待改调 "
        "nexa_agent.tools.analyze_image（见 7/4 server 变薄计划）"
    )


def get_llm_client() -> BaseLLMClient:
    """获取 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        config = get_config()
        kwargs = {"timeout": config.llm_timeout}

        # 按 backend 传入对应的 api_key 和 base_url
        if config.llm_backend == "deepseek":
            kwargs["api_key"] = config.deepseek_api_key
            kwargs["base_url"] = config.deepseek_base_url
        elif config.llm_backend == "kimi":
            kwargs["api_key"] = config.kimi_api_key
            kwargs["base_url"] = config.kimi_base_url
        elif config.llm_backend == "glm":
            kwargs["api_key"] = config.glm_api_key

        _llm_client = create_llm_client(
            backend=config.llm_backend,
            model=config.llm_model,
            **kwargs,
        )
        logger.info("LLM 客户端单例已初始化 backend=%s model=%s",
                     config.llm_backend, config.llm_model)
    return _llm_client


def reset_all():
    """重置所有单例（仅测试用）"""
    global _short_term_memory, _long_term_memory, _llm_client
    _short_term_memory = None
    _long_term_memory = None
    _llm_client = None
    from server.config import reset_config
    reset_config()
    logger.info("所有全局单例已重置")
