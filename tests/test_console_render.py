"""CLI 控制台渲染测试（评审 3.5：库代码去 print，CLI 入口自渲染）。

引擎 react_loop 不再直接 print——控制台进度由 CLI 把 render_react_event 挂成
on_event 回调驱动。本测试验证：① react 级事件渲染成预期行；② trial 级事件被忽略
（留给 reflexion_agent 的 print，不重复）；③ react_loop(verbose=True) 不再向
stdout 直出（库代码 stdout 洁净）。
"""

from types import SimpleNamespace

import nexa_agent.react_agent as ra
from nexa_agent.react_agent import render_react_event


def test_renders_react_events(capsys):
    render_react_event({"type": "step_start", "step": 2, "max_steps": 12, "model": "m"})
    render_react_event({"type": "action", "step": 2, "tool": "web_search",
                        "args": "字节跳动", "thought": "先查公司"})
    render_react_event({"type": "observation", "step": 2, "tool": "web_search",
                        "ok": True, "observation": "结果……"})
    render_react_event({"type": "final_answer", "answer": "[Verdict] 靠谱 —— ok"})
    out = capsys.readouterr().out
    assert "Step 2/12" in out and "web_search" in out
    assert "先查公司" in out            # thought 渲染
    assert "Observation" in out
    assert "✅ Final Answer" in out and "[Verdict] 靠谱" in out


def test_ignores_trial_level_events(capsys):
    # trial 级事件由 reflexion_agent 自己 print；渲染器忽略，避免重复
    for t in ("trial_start", "trial_evaluated", "verifier_start", "verifier_result",
              "usage", "started", "done", "stage_routed", "error"):
        render_react_event({"type": t, "trial": 1})
    assert capsys.readouterr().out == ""


def test_submit_verdict_action_not_rendered(capsys):
    # submit_verdict 是终止工具，不作为普通 Action 行渲染
    render_react_event({"type": "action", "tool": ra.FINALIZE_TOOL, "args": "{}"})
    assert capsys.readouterr().out == ""


def test_react_loop_is_stdout_silent(capsys, monkeypatch):
    """react_loop(verbose=True) 不再直出 stdout（库洁净）——即便详细模式也只走 logger。"""
    monkeypatch.setattr(ra, "call_llm_with_tools",
                        lambda *a, **k: (_text_choice("直接结论：已知足够。"), 1, 1))
    res = ra.react_loop("2+2 等于几", max_steps=2, verbose=True)
    assert res.answer  # 有答案
    assert capsys.readouterr().out == ""  # 但没有任何 stdout 直出


def _text_choice(text: str):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(message=msg, finish_reason="stop")
