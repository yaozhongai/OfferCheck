"""
LLM 客户端模块 — V0

支持三个云端 LLM 提供商，统一通过 OpenAI-compatible 接口调用：

- DeepSeek V4  (deepseek-v4-pro / deepseek-v4-flash)
- Kimi K2.6    (kimi-k2.6)
- GLM-5.1      (glm-5.1)

所有实现不启用 thinking / reasoning 模式，保持输出直接可读。

日志统一使用 nexa_agent.logger.get_logger。
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexa_agent.logger import get_logger

logger = get_logger("llm_client")


# ======================================================================
# 数据模型
# ======================================================================

@dataclass
class LLMMessage:
    """LLM 对话消息"""
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "LLMMessage":
        return cls(role="assistant", content=content)


@dataclass
class LLMResponse:
    """LLM 统一响应"""
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: float = 0.0
    finish_reason: str = "stop"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content[:500],
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_ms": self.elapsed_ms,
            "finish_reason": self.finish_reason,
        }


# ======================================================================
# 抽象接口
# ======================================================================

class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """发送对话消息，返回模型响应"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查 API 是否可用（通过 API key 是否配置判断）"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名"""
        ...


# ======================================================================
# 辅助函数
# ======================================================================

def _get_env_key(var_name: str, config_value: str) -> str:
    """优先从环境变量获取 key，fallback 到配置值"""
    return os.environ.get(var_name, config_value)


def _log_tokens(logger_instance, model: str, response: LLMResponse) -> None:
    logger_instance.info(
        "LLM 调用完成 model=%s elapsed=%.0fms tokens(in=%d out=%d total=%d)",
        model, response.elapsed_ms,
        response.prompt_tokens, response.completion_tokens, response.total_tokens,
    )


# ======================================================================
# DeepSeek V4
# ======================================================================

class DeepSeekClient(BaseLLMClient):
    """DeepSeek V4 API 客户端

    环境变量: DEEPSEEK_API_KEY
    API 文档: https://api.deepseek.com

    可用模型:
      - deepseek-v4-pro    (高精度)
      - deepseek-v4-flash  (快速)
    """

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 120.0,
    ):
        self._model = model
        self._api_key = _get_env_key("DEEPSEEK_API_KEY", api_key)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

        logger.info("DeepSeekClient 初始化 model=%s base_url=%s", model, base_url)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=3,  # SDK-level retry on transient errors (2nd line of defense)
            )
        return self._client

    def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        t0 = time.time()
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )

        elapsed = (time.time() - t0) * 1000
        choice = response.choices[0]

        result = LLMResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            elapsed_ms=elapsed,
            finish_reason=choice.finish_reason or "stop",
        )

        _log_tokens(logger, self._model, result)
        return result

    def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model


# ======================================================================
# Kimi K2.6
# ======================================================================

class KimiClient(BaseLLMClient):
    """Kimi K2.6 API 客户端

    环境变量: MOONSHOT_API_KEY
    API 文档: https://api.moonshot.cn/v1

    可用模型:
      - kimi-k2.6
    """

    def __init__(
        self,
        model: str = "kimi-k2.6",
        api_key: str = "",
        base_url: str = "https://api.moonshot.cn/v1",
        timeout: float = 120.0,
    ):
        self._model = model
        self._api_key = _get_env_key("MOONSHOT_API_KEY", api_key)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

        logger.info("KimiClient 初始化 model=%s base_url=%s", model, base_url)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=3,  # SDK-level retry on transient errors (2nd line of defense)
            )
        return self._client

    def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        t0 = time.time()
        client = self._get_client()

        # Kimi K2.6 默认启用思考模式，显式关闭。
        # 非思考模式下 temperature 必须为 0.6，否则 API 报错。
        response = client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            max_tokens=max_tokens,
            temperature=0.6,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

        elapsed = (time.time() - t0) * 1000
        choice = response.choices[0]

        result = LLMResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            elapsed_ms=elapsed,
            finish_reason=choice.finish_reason or "stop",
        )

        _log_tokens(logger, self._model, result)
        return result

    def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model


# ======================================================================
# GLM-5.1 (智谱)
# ======================================================================

class GLMClient(BaseLLMClient):
    """GLM-5.1 (智谱) API 客户端

    环境变量: ZHIPU_API_KEY
    SDK: zai (智谱官方)

    可用模型:
      - glm-5.1
    """

    def __init__(
        self,
        model: str = "glm-5.1",
        api_key: str = "",
        timeout: float = 120.0,
    ):
        self._model = model
        self._api_key = _get_env_key("ZHIPU_API_KEY", api_key)
        self._timeout = timeout
        self._client = None

        logger.info("GLMClient 初始化 model=%s", model)

    def _get_client(self):
        if self._client is None:
            from zai import ZhipuAiClient
            self._client = ZhipuAiClient(api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        t0 = time.time()
        client = self._get_client()

        # GLM 不启用 thinking 模式
        response = client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = (time.time() - t0) * 1000
        choice = response.choices[0]

        result = LLMResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            elapsed_ms=elapsed,
            finish_reason=choice.finish_reason or "stop",
        )

        _log_tokens(logger, self._model, result)
        return result

    def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model


# ======================================================================
# 工厂
# ======================================================================

# 每个 provider 支持的模型列表
PROVIDER_MODELS = {
    "deepseek": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "kimi":     ["kimi-k2.6"],
    "glm":      ["glm-5.1"],
}


def create_llm_client(
    backend: str = "deepseek",
    model: str = "",
    **kwargs,
) -> BaseLLMClient:
    """LLM 客户端工厂

    Args:
        backend: "deepseek" | "kimi" | "glm"
        model: 模型名。若为空，使用该 provider 的默认模型

    Returns:
        BaseLLMClient 实例
    """
    backend = backend.lower()

    if backend == "deepseek":
        default_model = "deepseek-v4-pro"
        return DeepSeekClient(
            model=model or default_model,
            api_key=kwargs.pop("api_key", ""),
            base_url=kwargs.pop("base_url", "https://api.deepseek.com"),
            **kwargs,
        )

    elif backend == "kimi":
        return KimiClient(
            model=model or "kimi-k2.6",
            api_key=kwargs.pop("api_key", ""),
            base_url=kwargs.pop("base_url", "https://api.moonshot.cn/v1"),
            **kwargs,
        )

    elif backend == "glm":
        return GLMClient(
            model=model or "glm-5.1",
            api_key=kwargs.pop("api_key", ""),
            **kwargs,
        )

    raise ValueError(
        f"不支持的 LLM 后端: {backend}。当前支持: {', '.join(PROVIDER_MODELS.keys())}"
    )
