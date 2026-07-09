"""B′ 无来源硬门 + A′-1 裁定 doctrine + 解析回退测试（事故 20260709 复盘修复）。

线上 Trial 1：prompt/schema 已写三遍「evidence 必须绑定来源」，模型仍交出 8 条
无来源事实 → Verifier 驳回整轮（白烧 ~87K tokens）。硬门把纠正压到 step 级：
submit_verdict 拦截点检出「全部条目缺来源」→ 拒一次要求补 [Source]。全部离线。
"""

import json
import os
from types import SimpleNamespace

import nexa_agent.react_agent as ra
from nexa_agent.react_agent import _evidence_all_unsourced
from nexa_agent.verifier import _parse_facts_from_output


# ── helper：无来源判定（保守：全部缺失才拦）─────────────────────────────────

def test_unsourced_detection():
    assert _evidence_all_unsourced(["纯断言一", "纯断言二，没有任何来源"]) is True
    assert _evidence_all_unsourced([]) is False  # 空列表不拦（空提交另有防护）


def test_sourced_variants_pass():
    assert not _evidence_all_unsourced(["事实 [Source] https://x.com/page"])
    assert not _evidence_all_unsourced(["官网 https://a.com/jobs 列出该职位"])   # 行内 URL
    assert not _evidence_all_unsourced(["域名注册于 2001 domain_whois_lookup(anthropic.com)"])  # 工具引用
    assert not _evidence_all_unsourced(["官网 anthropic.com 要求 25% 到岗"])     # 裸域名
    # 部分有来源 → 放行交 Verifier 权衡（硬门不做新误杀源）
    assert not _evidence_all_unsourced(["无来源断言", "有来源 [Source] https://b.com"])


# ── react_loop 集成：无来源提交被拒一次 → 补来源后收尾 ─────────────────────

def _tc(name, args_dict):
    return SimpleNamespace(id=f"c_{name}", type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args_dict, ensure_ascii=False)))


def _choice(tool_calls=None, content=""):
    return SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=None),
        finish_reason="tool_calls" if tool_calls else "stop")


def test_gate_rejects_once_then_accepts(monkeypatch):
    seq = [
        _choice([_tc("web_search", {"input": "Anthropic 招聘"})], "先查证"),
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "待核",
                                        "evidence": ["无来源断言一", "无来源断言二"]})]),
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "待核",
                                        "evidence": ["官网要求到岗 [Source] https://www.anthropic.com/careers"]})]),
    ]
    calls = {"i": 0}
    def _fake_llm(*a, **k):
        c = seq[calls["i"]]; calls["i"] += 1; return c, 5, 3
    monkeypatch.setattr(ra, "call_llm_with_tools", _fake_llm)
    monkeypatch.setattr(ra, "execute_tool",
                        lambda name, args: "结果 https://www.anthropic.com/careers 官方招聘页")

    events = []
    res = ra.react_loop("Anthropic 招聘真实吗", max_steps=6, verbose=False,
                        stage="offercheck_stage1", on_event=lambda e: events.append(e))

    assert res.terminated_reason == "submit_verdict"
    assert res.steps_used == 3                      # 拒 1 次只花 1 步，不烧整轮 Trial
    nags = [e for e in events if e.get("type") == "correction" and "来源" in e.get("message", "")]
    assert len(nags) == 1                            # 硬门发过一次纠正事件
    assert "[Verdict] 存疑" in res.answer


def test_gate_only_nags_once_no_deadlock(monkeypatch):
    # 模型执意不补来源：第二次照样放行（交 Verifier 权衡），绝不死锁
    seq = [
        _choice([_tc("web_search", {"input": "q"})], "查"),
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "s", "evidence": ["无来源"]})]),
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "s", "evidence": ["还是无来源"]})]),
    ]
    calls = {"i": 0}
    monkeypatch.setattr(ra, "call_llm_with_tools",
                        lambda *a, **k: (seq[calls.__setitem__("i", calls["i"]+1) or calls["i"]-1], 1, 1))
    monkeypatch.setattr(ra, "execute_tool", lambda n, a: "ok https://x.com")

    res = ra.react_loop("q", max_steps=6, verbose=False, stage="offercheck_stage1")
    assert res.terminated_reason == "submit_verdict"
    assert res.steps_used == 3


def test_gate_skipped_when_zero_retrieval(monkeypatch):
    # 零检索时轮不到来源门——强制取证 gate（更早的层）先拦「无检索下裁定」
    seq = [
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "s", "evidence": ["无来源"]})]),
        _choice([_tc("web_search", {"input": "q"})], "补查"),
        _choice([_tc("submit_verdict", {"verdict": "存疑", "summary": "s",
                                        "evidence": ["x [Source] https://a.com"]})]),
    ]
    calls = {"i": 0}
    monkeypatch.setattr(ra, "call_llm_with_tools",
                        lambda *a, **k: (seq[calls.__setitem__("i", calls["i"]+1) or calls["i"]-1], 1, 1))
    monkeypatch.setattr(ra, "execute_tool", lambda n, a: "ok https://a.com")

    events = []
    res = ra.react_loop("q", max_steps=6, verbose=False,
                        stage="offercheck_stage1", on_event=lambda e: events.append(e))
    assert any(e.get("type") == "evidence_gate" for e in events)  # 走的是取证 gate
    assert res.terminated_reason == "submit_verdict"


# ── Verifier 解析回退：[Fact] 行内嵌 URL 不再判「未标注」───────────────────

def test_parse_facts_inline_url_as_source():
    facts = _parse_facts_from_output(
        "[Verdict] 存疑\n[Fact] 官网 https://www.anthropic.com/careers 要求至少 25% 到岗\n")
    assert facts[0]["source"] == "https://www.anthropic.com/careers"   # 此前是「未标注」


def test_parse_facts_explicit_source_still_wins():
    facts = _parse_facts_from_output(
        "[Fact] 域名注册于 2001（见 https://inline.example/x）\n[Source] domain_whois_lookup(anthropic.com)\n")
    assert facts[0]["source"] == "domain_whois_lookup(anthropic.com)"


# ── A′-1 prompt 护栏：接地铁律第 7 条在位（防误删回归）───────────────────────

def test_grounding_rule7_present():
    path = os.path.join(os.path.dirname(ra.__file__), "prompts", "react_system.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "裁定对象必须是你实际检验过的东西" in text
    assert "只能取恰好一档" in text
    assert "通用背景" in text          # 「常被仿冒」≠「这一次就是仿冒」
    assert "照常下「大概率有坑」" in text  # 反向保护：已提供材料的确凿信号不弱化
