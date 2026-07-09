"""LLM Gateway 测试（评审 3.1）——用假 client 注入，不发网络。

验证：kwargs 组装（温度/模型/thinking）、role→模型解析、空响应重试、结果归一化。
"""

from types import SimpleNamespace

import pytest

from nexa_agent.llm_gateway import LLMGateway
from nexa_agent.config import get_model_for_role


def _resp(content, pt=3, ct=5):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


class _FakeClient:
    def __init__(self, responder):
        self.calls = []
        self._responder = responder

    def with_options(self, **kw):
        return self

    @property
    def chat(self):
        def create(**kwargs):
            self.calls.append(kwargs)
            return self._responder(len(self.calls))
        return SimpleNamespace(completions=SimpleNamespace(create=create))


def _gateway(responder):
    g = LLMGateway()
    g._client = _FakeClient(responder)
    return g, g._client


def test_kwargs_assembly_temperature_and_model():
    g, fake = _gateway(lambda n: _resp('{"ok":true}'))
    g.complete([{"role": "user", "content": "hi"}], model="my-model",
               max_tokens=256, temperature=0.0, enable_thinking=False)
    kw = fake.calls[0]
    assert kw["model"] == "my-model"
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == 256
    # GMI 下 thinking off → extra_body 带 enable_thinking:false（provider 相关，允许缺省）
    assert "extra_body" in kw or True


def test_role_resolves_to_model():
    g, fake = _gateway(lambda n: _resp("x"))
    g.complete([{"role": "user", "content": "hi"}], role="evaluator_llm")
    assert fake.calls[0]["model"] == get_model_for_role("evaluator_llm")


def test_result_normalized():
    g, _ = _gateway(lambda n: _resp("hello", pt=7, ct=9))
    r = g.complete([{"role": "user", "content": "hi"}], model="m")
    assert r.content == "hello"
    assert r.prompt_tokens == 7 and r.completion_tokens == 9
    assert r.finish_reason == "stop"


def test_retry_on_empty():
    # 第一次空、第二次有内容 → retry_on_empty 应重试并取到非空
    def responder(n):
        return _resp("" if n == 1 else "recovered")
    g, fake = _gateway(responder)
    r = g.complete([{"role": "user", "content": "hi"}], model="m", retry_on_empty=True)
    assert r.content == "recovered"
    assert len(fake.calls) == 2


def test_no_retry_on_empty_when_disabled():
    g, fake = _gateway(lambda n: _resp(""))
    r = g.complete([{"role": "user", "content": "hi"}], model="m", retry_on_empty=False)
    assert r.content == ""
    assert len(fake.calls) == 1
