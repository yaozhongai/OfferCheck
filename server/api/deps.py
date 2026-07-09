"""
FastAPI 依赖注入 — V0

提供全局单例（状态机、记忆、提取管线等）的懒加载获取函数。
"""

from __future__ import annotations

from server.config import get_config
from server.memory.long_term import LongTermMemory
from server.memory.short_term import ShortTermMemory
from server.api.schemas import (  # noqa: F401   — re-export for route convenience
    ErrorResponse,
    HealthResponse,
    MemoryListResponse,
    UploadResponse,
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    StateMachineStatus,
)
from nexa_agent.logger import get_logger

logger = get_logger("deps")

# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_short_term_memory: ShortTermMemory | None = None
_long_term_memory: LongTermMemory | None = None


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


def reset_all():
    """重置所有单例（仅测试用）"""
    global _short_term_memory, _long_term_memory
    _short_term_memory = None
    _long_term_memory = None
    from server.config import reset_config
    reset_config()
    logger.info("所有全局单例已重置")
