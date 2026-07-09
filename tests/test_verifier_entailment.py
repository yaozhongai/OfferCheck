"""Verifier entailment 内容核实测试（评审 2.2）

补 AIS 存在性对账抓不到的 misattribution：来源真实、但正文不支持断言。
覆盖域名提取/匹配 + supported=false 强制驳回的解析逻辑（不发网络）。
"""

import pytest

from nexa_agent.verifier import (
    VerifierAgent, _source_domain, _match_evidence,
)


# ── 域名提取/匹配 ──

def test_source_domain_from_url():
    assert _source_domain("https://jobs.bytedance.com/apply — 官网") == "jobs.bytedance.com"


def test_source_domain_from_bare():
    assert _source_domain("domain_whois_lookup(bytedance-recruit.com) — RDAP") == "bytedance-recruit.com"


def test_source_domain_none():
    assert _source_domain("calculator 工具计算") == ""


def test_match_evidence_exact_and_subdomain():
    ev = {"bytedance.com": "正文A"}
    assert _match_evidence("bytedance.com", ev) == "正文A"
    # 父/子域宽松匹配
    assert _match_evidence("jobs.bytedance.com", ev) == "正文A"
    assert _match_evidence("other.com", ev) == ""


# ── 解析：misattribution 强制驳回 ──

@pytest.fixture
def v():
    return VerifierAgent()


def test_misattribution_forces_failed(v):
    # D3 校准后：只有「摘录直接矛盾」(contradicted) 才是真 misattribution、才强制驳回
    facts = [{"fact": "该页要求交押金", "source": "https://bytedance.com/jobs", "confidence": "High"}]
    text = ('{"fact_verdicts":[{"index":1,"reliable":true,"supported":"contradicted",'
            '"reason":"页面正文明确说无需任何费用，与押金要求矛盾"}],'
            '"overall":"verified","reason":"来源可靠","feedback":""}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[1])
    assert res.status == "failed"           # 即使 LLM 说 verified，也被强制驳回
    assert res.unreliable_facts
    assert "misattribution" in res.unreliable_facts[0]["reject_reason"]


def test_supported_true_passes(v):
    facts = [{"fact": "官方域名注册于 2011", "source": "https://bytedance.com", "confidence": "High"}]
    text = ('{"fact_verdicts":[{"index":1,"reliable":true,"supported":true,"reason":"摘录印证"}],'
            '"overall":"verified","reason":"ok","feedback":""}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[1])
    assert res.status == "verified"
    assert not res.unreliable_facts


def test_supported_false_ignored_when_not_entailment_checked(v):
    # 没做 entailment 的事实（idx 不在集合里），supported 字段不参与判定
    facts = [{"fact": "x", "source": "https://a.com", "confidence": "Low"}]
    text = ('{"fact_verdicts":[{"index":1,"reliable":true,"supported":false,"reason":"r"}],'
            '"overall":"verified","reason":"ok","feedback":""}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[])
    assert res.status == "verified"
    assert not res.unreliable_facts


def test_unreliable_source_still_flagged(v):
    facts = [{"fact": "x", "source": "https://quora.com/q", "confidence": "Low"}]
    text = ('{"fact_verdicts":[{"index":1,"reliable":false,"supported":null,"reason":"UGC"}],'
            '"overall":"failed","reason":"来源不可靠","feedback":"换权威来源"}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[])
    assert res.status == "failed"
    assert res.unreliable_facts
