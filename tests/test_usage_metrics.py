"""Token 用量指标端到端透出测试（评审 3.6）

react_loop 已累加 total_prompt_tokens/total_completion_tokens，但此前只 print。
本批次让它经 ReflexionResult 透出 + 发 additive `usage` 事件 + Eval Harness 报告
「每裁定平均 token」。全部离线（假 react_loop / 假评估器），不发网络。
"""

from types import SimpleNamespace

from nexa_agent.reflexion_agent import ReflexionReActAgent, ReflexionResult
import nexa_agent.reflexion_agent as rx


def _fake_react_result(pt: int, ct: int) -> dict:
    """伪造一条 react_loop 返回（带 token 计数），字段够 execute() 消费即可。"""
    return {
        "answer": "[Verdict] 靠谱 —— 已核实",
        "trajectory": "### Step 1\nAction: web_search(x)\nObservation: ok",
        "steps_used": 3,
        "terminated_reason": "submit_verdict",
        "total_prompt_tokens": pt,
        "total_completion_tokens": ct,
        "step_utilities": [],
        "critical_step": None,
        "source_attribution": {"total_sources": 1, "unverified": 0},
        "action_history": [("web_search", "x")],
        "seen_urls": ["https://example.com"],
        "successful_retrievals": 1,
        "evidence_registry": {},
        "verdict": {"verdict": "靠谱", "verdict_level": "reliable"},
    }


def _pass_agent(monkeypatch) -> ReflexionReActAgent:
    """构造一个 trial-1 直接判成功、跳过 verifier 的 agent。"""
    agent = ReflexionReActAgent(max_trials=3, evaluator_mode="heuristic")
    monkeypatch.setattr(
        agent.evaluator, "evaluate",
        lambda **kw: SimpleNamespace(success=True, reason="ok", feedback_signal="", failure_mode=None),
    )
    monkeypatch.setattr(rx, "should_trigger_verifier", lambda *a, **k: False)
    return agent


def test_result_exposes_tokens(monkeypatch):
    agent = _pass_agent(monkeypatch)
    monkeypatch.setattr(rx, "react_loop", lambda **kw: _fake_react_result(1200, 300))

    result = agent.execute("测试任务", verbose=False)

    assert isinstance(result, ReflexionResult)
    assert result.total_prompt_tokens == 1200
    assert result.total_completion_tokens == 300
    assert result.total_tokens == 1500  # property = in + out


def test_usage_event_emitted(monkeypatch):
    agent = _pass_agent(monkeypatch)
    monkeypatch.setattr(rx, "react_loop", lambda **kw: _fake_react_result(500, 100))

    events: list[dict] = []
    agent.execute("测试任务", verbose=False, on_event=lambda e: events.append(e))

    usage = [e for e in events if e.get("type") == "usage"]
    assert len(usage) == 1
    u = usage[0]
    assert u["prompt_tokens"] == 500 and u["completion_tokens"] == 100
    assert u["total_tokens"] == 600
    assert u["cumulative_total_tokens"] == 600  # 单 Trial：累计=本轮


def test_tokens_accumulate_across_trials(monkeypatch):
    """多轮 Trial：token 跨 Trial 累加，且每轮都发一次 usage 事件。"""
    # 前两轮判失败（触发反思→重试），第三轮成功
    calls = {"n": 0}

    def _eval(**kw):
        calls["n"] += 1
        ok = calls["n"] >= 3
        return SimpleNamespace(success=ok, reason="", feedback_signal="fb",
                               failure_mode=None if ok else "loop")

    agent = ReflexionReActAgent(max_trials=3, evaluator_mode="heuristic")
    monkeypatch.setattr(agent.evaluator, "evaluate", _eval)
    monkeypatch.setattr(rx, "should_trigger_verifier", lambda *a, **k: False)
    # 失败轮会走反思/教训/scratchpad 提取的 LLM 调用——一律短路，避免网络
    monkeypatch.setattr(agent, "_generate_reflection", lambda **kw: "反思")
    monkeypatch.setattr(agent, "_extract_lessons", lambda r: [])
    monkeypatch.setattr(agent, "_extract_scratchpad_facts", lambda *a, **k: [])
    monkeypatch.setattr(rx, "react_loop", lambda **kw: _fake_react_result(100, 50))

    events: list[dict] = []
    result = agent.execute("测试任务", verbose=False, on_event=lambda e: events.append(e))

    assert result.trials_used == 3
    assert result.total_prompt_tokens == 300   # 3 × 100
    assert result.total_completion_tokens == 150  # 3 × 50
    assert result.total_tokens == 450
    usage = [e for e in events if e.get("type") == "usage"]
    assert len(usage) == 3
    assert usage[-1]["cumulative_total_tokens"] == 450


def test_eval_report_avg_tokens():
    """EvalReport 汇总每裁定平均 token；顺带证 avg_steps 不再恒 0（steps_used 键修复）。"""
    from nexa_agent.eval_harness import EvalHarness, EvalRecord

    recs = [
        EvalRecord(case_id="a", question="q", expected_answer="", prediction="p",
                   correct=True, trials_used=1, elapsed_seconds=1.0,
                   step_count=4, prompt_tokens=1000, completion_tokens=200),
        EvalRecord(case_id="b", question="q", expected_answer="", prediction="p",
                   correct=True, trials_used=1, elapsed_seconds=1.0,
                   step_count=6, prompt_tokens=2000, completion_tokens=400),
    ]
    report = EvalHarness(max_trials=1, max_steps=8)._build_report("rid", "suite", recs)
    assert report.avg_prompt_tokens == 1500
    assert report.avg_completion_tokens == 300
    assert report.avg_total_tokens == 1800
    assert report.avg_steps == 5.0  # (4+6)/2 —— 修复前恒 0
