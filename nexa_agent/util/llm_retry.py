"""LLM 瞬时错误分类 + 指数退避重试（共享 helper）。

GMI 网关的 Connection error / 超时 / 5xx / 429 常见，重试可自愈；4xx 请求错误
（400/422）是协议/参数问题，不重试直接抛（见 CLAUDE.md GMI 约束）。

原实现内嵌在 react_agent 的 call_llm_with_tools，仅保护了主循环；抽出为共享 helper
后，evaluator / verifier / reflexion / stage_router 等质量门也走同一套重试与可观测
（评审 1.9：重试策略不一致 + 质量门在 GMI 抖动时 fail-open）。
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple, TypeVar

from nexa_agent.logger import get_logger

logger = get_logger("llm_retry")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # 秒；指数退避 base（1s → 2s → 4s）

try:
    from openai import (
        APIConnectionError as _APIConnectionError,
        APITimeoutError as _APITimeoutError,
        InternalServerError as _InternalServerError,
        RateLimitError as _RateLimitError,
    )
    _TRANSIENT_LLM_ERRORS: Tuple[type, ...] = (
        _APIConnectionError, _APITimeoutError, _InternalServerError, _RateLimitError,
    )
except ImportError:  # openai 版本差异兜底
    _TRANSIENT_LLM_ERRORS = ()

T = TypeVar("T")


def is_transient_llm_error(exc: Exception) -> bool:
    """是否为可重试的瞬时错误（连接/超时/5xx/429）。4xx（400/422）返回 False。"""
    if _TRANSIENT_LLM_ERRORS and isinstance(exc, _TRANSIENT_LLM_ERRORS):
        return True
    # 兜底：按错误文本判断（未装到具体异常类型时）
    msg = str(exc).lower()
    if any(k in msg for k in ("connection", "timeout", "timed out", "temporarily", "econnreset")):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return isinstance(status, int) and status in (429, 500, 502, 503, 504)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """对零参可调用 fn 做瞬时错误指数退避重试。

    - 瞬时错误（连接/超时/5xx/429）：退避后重试，最多 max_retries 次。
    - 非瞬时错误（4xx 等）或重试耗尽：直接抛出（由调用方决定 fail-safe 兜底）。
    - on_retry(attempt, exc)：每次重试前回调（供上层发 trace 事件）；回调异常被吞。
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not is_transient_llm_error(exc):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("LLM 瞬时错误，%.1fs 后重试 (%d/%d): %s",
                           delay, attempt + 1, max_retries, exc)
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(delay)
    # 逻辑上不可达（循环内要么 return 要么 raise）
    raise RuntimeError("call_with_retry 异常退出")
