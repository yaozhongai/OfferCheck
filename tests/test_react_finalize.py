"""react_loop 采样温度与兜底收尾测试（评审 1.1 / 1.5）

- 1.1：call_llm_with_tools 现在显式传 temperature（此前跑在 provider 默认温度）。
- 1.5：max_steps 兜底汇总统一走 _finalize，返回结果含 source_attribution（AIS 对账），
       此前手工拼 dict 绕过了接地层。
两者都用假 LLM 客户端/函数离线验证，不发网络。
"""

from types import SimpleNamespace

import pytest

import nexa_agent.react_agent as ra


# ── 1.1 温度显式传入 ────────────────────────────────────────────────────

def test_call_llm_with_tools_passes_temperature(monkeypatch):
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content="hi", tool_calls=None, reasoning_content=None)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    class _FakeClient:
        chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(ra, "_get_llm_client", lambda: _FakeClient())
    ra.call_llm_with_tools([{"role": "user", "content": "hi"}], tools=[], enable_thinking=False)
    assert "temperature" in captured
    assert captured["temperature"] == ra.MODEL_CONFIG["react_temperature"] == 0.0


# ── 1.5 兜底汇总走 _finalize ─────────────────────────────────────────────

def _toolcall_choice():
    tc = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="get_current_time", arguments="{}"),
    )
    msg = SimpleNamespace(content="继续", tool_calls=[tc], reasoning_content=None)
    return SimpleNamespace(message=msg, finish_reason="tool_calls")


def test_max_steps_fallback_routes_through_finalize(monkeypatch):
    # 每步都返回一个安全工具调用（get_current_time，无网络），逼到 max_steps
    monkeypatch.setattr(ra, "call_llm_with_tools",
                        lambda *a, **k: (_toolcall_choice(), 1, 1))
    # 兜底汇总的纯文本 LLM 调用
    monkeypatch.setattr(ra, "call_llm",
                        lambda *a, **k: ("兜底最终结论：基于已有信息作答。", 1, 1))

    result = ra.react_loop("测试问题", max_steps=2, verbose=False)

    assert result["terminated_reason"] == "max_steps"
    # 关键：_finalize 才会附加 source_attribution（AIS 对账）——证明未绕过接地层
    assert "source_attribution" in result
    assert result["answer"]
