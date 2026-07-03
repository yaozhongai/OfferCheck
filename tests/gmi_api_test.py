"""GMI Cloud Inference Engine 连通性冒烟测试。

验证引擎默认 provider（GMI, OpenAI 兼容）可正常完成一次 chat。
- Key 从环境变量读取（GMI_API_KEY），绝不硬编码。
- 无 key 时自动 skip，不阻塞 CI；不在收集期打真实 API。
- 标记 network，可用 `-m "not network"` 跳过。

运行::

    pytest tests/gmi_api_test.py -v            # 有 GMI_API_KEY 时真实调用
    pytest -m "not network"                    # 跳过所有联网测试
"""

from __future__ import annotations

import os

import pytest

# 与引擎 config 一致地加载 .env，使本地有 key 时能真实联网验证
try:
    from dotenv import load_dotenv

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv = os.path.join(_root, ".env")
    if os.path.exists(_dotenv):
        load_dotenv(_dotenv)
except ImportError:
    pass

GMI_API_KEY = os.environ.get("GMI_API_KEY", "")
GMI_BASE_URL = os.environ.get("GMI_BASE_URL", "https://api.gmi-serving.com/v1")

requires_gmi = pytest.mark.skipif(
    not GMI_API_KEY,
    reason="GMI_API_KEY 未配置，跳过 GMI 联网冒烟测试",
)


@pytest.mark.network
@requires_gmi
def test_gmi_chat_completion_smoke():
    """一次最小 chat completion，确认 GMI 网关 + 模型 + key 三者可用。"""
    from openai import OpenAI

    client = OpenAI(api_key=GMI_API_KEY, base_url=GMI_BASE_URL, timeout=60.0)
    resp = client.chat.completions.create(
        model=os.environ.get("STRONG_MODEL", "deepseek-ai/DeepSeek-V4-Pro"),
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=0,
        max_tokens=200,
    )

    assert resp.choices, "GMI 返回空 choices"
    content = resp.choices[0].message.content or ""
    assert content.strip(), "GMI 返回空内容"
    assert resp.usage and resp.usage.total_tokens > 0, "GMI 未返回 token 用量"


@pytest.mark.network
@requires_gmi
def test_engine_config_defaults_to_gmi():
    """有 GMI_API_KEY 时，引擎配置应默认选中 GMI provider。"""
    import importlib

    import nexa_agent.config as cfg
    importlib.reload(cfg)

    assert cfg.LLM_PROVIDER == "gmi", f"预期默认 gmi，实际 {cfg.LLM_PROVIDER}"
    assert "gmi-serving" in cfg.MODEL_CONFIG["base_url"]
    # GMI 是 OpenAI 兼容网关，不支持 DeepSeek 私有 thinking 扩展参数
    assert cfg.SUPPORTS_THINKING_PARAM is False
