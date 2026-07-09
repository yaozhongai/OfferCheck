"""LLM Gateway —— 文本 LLM 调用的单一入口（评审 3.1）。

全仓此前有 ~9 个各自为政的 LLM 调用点，每个都重复「构造 OpenAI client + 拼
kwargs + thinking 注入 + retry + token 记账」。这带来批次 1 修过的一整类不一致
（温度没传 / thinking 旧守卫 / retry 只护主循环 / 空响应只有 verifier 处理）。

本模块把这套横切逻辑收敛到一处 `complete()`：
  - role → 模型路由（复用 config.get_model_for_role），或显式 model
  - 统一 thinking 注入（thinking_extra_body）、温度、stop
  - 统一瞬时错误重试（util.llm_retry.call_with_retry）+ 可选空响应重试
  - 统一 token 记账与日志（为 3.6 成本指标铺垫）

返回归一化的 LLMResult：文本调用方取 `.content`；将来 tool-calling 调用方
（react_loop，3.1b）取 `.message`（原始 choice.message，保留 tool_calls /
reasoning_content 语义）。

注：视觉调用（analyze_image*）走 VISION_CONFIG 的另一 provider，不归本网关。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from nexa_agent.config import MODEL_CONFIG, thinking_extra_body, get_model_for_role
from nexa_agent.logger import get_logger
from nexa_agent.util.llm_retry import call_with_retry, DEFAULT_MAX_RETRIES

logger = get_logger("llm_gateway")


@dataclass
class LLMResult:
    """归一化的 LLM 返回。

    - content: 面向文本调用方的正文（可能为空串）
    - message: 原始 choice.message（tool-calling 调用方用它取 .tool_calls /
      .reasoning_content，供 _assistant_msg_to_dict 回传）
    """
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    reasoning_content: Optional[str] = None
    message: Any = None

    @property
    def tool_calls(self):
        return getattr(self.message, "tool_calls", None) if self.message is not None else None


class LLMGateway:
    """文本 LLM 调用网关。默认单例（GATEWAY），也可按需构造隔离实例。"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or MODEL_CONFIG["api_key"]
        self.base_url = base_url or MODEL_CONFIG["base_url"]
        self._client = None  # 惰性构造

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            # 客户端层不做 SDK 自动重试（统一由 call_with_retry 管，避免双重重试）
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=120.0, max_retries=0)
        return self._client

    def complete(
        self,
        messages: List[dict],
        *,
        role: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        stop: Optional[List[str]] = None,
        timeout: float = 60.0,
        max_retries: Optional[int] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        retry_on_empty: bool = False,
    ) -> LLMResult:
        """执行一次（非流式）LLM 调用并返回归一化结果。

        Args:
            role/model: 二选一决定模型；role 经 get_model_for_role 解析，model 直接用。
            enable_thinking: 是否允许思考（经 thinking_extra_body 按 provider 落参）。
            timeout: 本次请求超时（秒）。
            max_retries: 瞬时错误重试次数（None=DEFAULT；stage_router 传 1 走快失败）。
            retry_on_empty: 成功但 content 为空时再试一次（GMI 大 prompt 偶发空返回）。
        """
        actual_model = model or get_model_for_role(role or "react_main")
        kwargs: dict = {
            "model": actual_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
        if stop:
            kwargs["stop"] = stop
        _eb = thinking_extra_body(actual_model, enable_thinking)
        if _eb:
            kwargs["extra_body"] = _eb

        client = self._get_client().with_options(timeout=timeout)
        retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries

        t0 = time.time()
        attempts = 2 if retry_on_empty else 1
        response = None
        for i in range(attempts):
            response = call_with_retry(
                lambda: client.chat.completions.create(**kwargs),
                max_retries=retries, on_retry=on_retry,
            )
            content = response.choices[0].message.content or ""
            if content.strip() or i == attempts - 1:
                break
            logger.warning("LLM 返回空 content（model=%s finish=%s），重试 (%d/%d)",
                           actual_model, response.choices[0].finish_reason, i + 1, attempts - 1)
        elapsed_ms = (time.time() - t0) * 1000

        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        logger.info("LLM 调用完成 model=%s elapsed=%.0fms tokens(in=%d out=%d) "
                    "tool_calls=%s thinking=%s",
                    actual_model, elapsed_ms, pt, ct, bool(getattr(msg, "tool_calls", None)),
                    "on" if enable_thinking else "off")

        return LLMResult(
            content=msg.content or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            finish_reason=choice.finish_reason or "stop",
            reasoning_content=getattr(msg, "reasoning_content", None),
            message=msg,
        )


# 默认单例：全仓文本调用共享（provider 唯一真源 = MODEL_CONFIG）
GATEWAY = LLMGateway()


def complete(messages: List[dict], **kwargs) -> LLMResult:
    """模块级便捷入口，委托默认单例。"""
    return GATEWAY.complete(messages, **kwargs)
