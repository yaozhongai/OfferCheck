"""Verifier 误杀复盘修复测试（P0 D1–D3 + P1 D4–D5）。

线上 Trial 1 好答案（真公司 + 诈骗预警）被 CoVe 判 misattribution 误杀。根因：
搜索列表污染 registry（D1）+ 只喂前 300 字（D2）+「查无」当「矛盾」硬毙（D3）+
stage1 落进严格来源标准（D4）+ 全败回传最后一轮错误串（D5）。全部离线。
"""

from nexa_agent.react_agent import register_evidence
from nexa_agent.verifier import (
    VerifierAgent, _relevant_window, _source_criteria, _norm_supported,
)
from nexa_agent.reflexion_agent import ReflexionReActAgent


# ── D1: registry 分级，真实抓取正文顶替搜索列表占位 ────────────────────────
def test_d1_fetched_content_beats_search_placeholder():
    reg, strength = {}, {}
    # 先 web_search：列表里出现 anthropic.com，登记列表文本为 weak 占位
    register_evidence(reg, strength, "web_search", "Anthropic",
                      "结果:\n1. https://www.anthropic.com/ ...\n2. https://reddit.com/r/x ...")
    assert strength["anthropic.com"] == "weak"
    # 再 web_fetch 官网：真实正文，strong，顶替 weak
    register_evidence(reg, strength, "web_fetch", "https://www.anthropic.com/careers",
                      "REAL_PAGE >>> Software Engineer roles ...")
    assert strength["anthropic.com"] == "strong"
    assert reg["anthropic.com"].startswith("REAL_PAGE")


def test_d1_outbound_links_not_registered():
    reg, strength = {}, {}
    # 抓 anthropic 页，正文里含出站链接 greenhouse.io —— 不应把 anthropic 正文错配给 greenhouse
    register_evidence(reg, strength, "web_fetch", "https://www.anthropic.com/careers",
                      "Anthropic careers. Apply at https://job-boards.greenhouse.io/anthropic")
    assert "anthropic.com" in reg
    assert "job-boards.greenhouse.io" not in reg


def test_d1_strong_not_overwritten_by_search():
    reg, strength = {}, {}
    register_evidence(reg, strength, "web_fetch", "https://a.com/x", "REAL a-content")
    register_evidence(reg, strength, "web_search", "q", "list ... https://a.com/y ...")
    assert reg["a.com"] == "REAL a-content"   # 搜索 weak 不覆盖已有 strong


def test_d1_whois_is_strong():
    reg, strength = {}, {}
    register_evidence(reg, strength, "domain_whois_lookup", "bytedance-recruit.com",
                      "RDAP: registered 2025-01-01, registrar NameCheap")
    assert strength["bytedance-recruit.com"] == "strong"


# ── D2: 取与断言重叠度最高的窗口，而非机械前 N 字 ──────────────────────────
def test_d2_relevant_window_finds_assertion():
    head = "导航 主页 关于 联系 " * 20  # 前段是无关导航
    body = "Anthropic 官网 careers 页列出 remote Software Engineer 职位，坐标美国。"
    text = head + body + ("页脚 版权 " * 20)
    win = _relevant_window(text, "官网列出 remote Software Engineer 职位", size=60)
    assert "Software Engineer" in win           # 命中断言所在段
    assert win != text[:60]                       # 不是机械取头部


def test_d2_short_text_returned_whole():
    assert _relevant_window("短文本", "任意事实", size=300) == "短文本"


# ── D3: 三态判定，只有 contradicted 硬毙，no_evidence 放行 ───────────────────
def test_d3_norm_supported_three_state():
    assert _norm_supported("contradicted") == "contradicted"
    assert _norm_supported("yes") == "yes"
    assert _norm_supported("no_evidence") == "no_evidence"
    assert _norm_supported(True) == "yes"
    assert _norm_supported(False) == "no_evidence"   # 旧格式布尔保守转 no_evidence
    assert _norm_supported(None) is None


def test_d3_no_evidence_does_not_fail():
    v = VerifierAgent()
    facts = [{"fact": "官网列出该职位", "source": "https://anthropic.com/careers", "confidence": "High"}]
    # judge 说 no_evidence（300 字窗口没提到）—— 不得驳回
    text = ('{"fact_verdicts":[{"index":1,"reliable":true,"supported":"no_evidence",'
            '"reason":"该窗口未提及此职位"}],"overall":"verified","reason":"ok","feedback":""}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[1])
    assert res.status == "verified"
    assert not res.unreliable_facts


def test_d3_contradicted_forces_failed():
    v = VerifierAgent()
    facts = [{"fact": "自称来自官网", "source": "https://anthropic.com/careers", "confidence": "High"}]
    text = ('{"fact_verdicts":[{"index":1,"reliable":true,"supported":"contradicted",'
            '"reason":"摘录实为论坛帖，与官网来源矛盾"}],"overall":"verified","reason":"ok","feedback":""}')
    res = v._parse_cove_response(text, facts, entailment_idxs=[1])
    assert res.status == "failed"
    assert "misattribution" in res.unreliable_facts[0]["reject_reason"]


# ── D4: stage1 走宽松来源标准 ─────────────────────────────────────────────
def test_d4_stage1_uses_lenient_criteria():
    for stg in ("offercheck_stage1", "stage1", "offercheck_stage3", "offercheck_stage4"):
        assert "OfferCheck 调查证伪专用" in _source_criteria(stg)
    # stage2 / 通用 走严格标准
    for stg in ("offercheck_stage2", None):
        assert "OfferCheck 调查证伪专用" not in _source_criteria(stg)


# ── D5: 全败回传最佳答案（跳过 llm_error 轮）───────────────────────────────
def test_d5_prefers_verifier_caveat_over_llm_error():
    trials = [
        {"trial": 1, "answer": "[Verdict] 靠谱 —— Anthropic 真实", "terminated_reason": "submit_verdict",
         "failure_mode": "unreliable_source", "verdict": {"verdict_level": "reliable"}},
        {"trial": 2, "answer": "[错误] 推理模型调用失败 (step 6): 400", "terminated_reason": "llm_error",
         "failure_mode": "llm_error", "verdict": None},
    ]
    ans, verdict = ReflexionReActAgent._pick_best_final(trials)
    assert ans.startswith("[Verdict] 靠谱")        # 拿 Trial 1 好答案，不是 Trial 2 错误串
    assert verdict == {"verdict_level": "reliable"}


def test_d5_falls_back_to_last_when_all_errors():
    trials = [
        {"trial": 1, "answer": "[错误] a", "terminated_reason": "llm_error", "verdict": None},
        {"trial": 2, "answer": "[错误] b", "terminated_reason": "llm_error", "verdict": None},
    ]
    ans, _ = ReflexionReActAgent._pick_best_final(trials)
    assert ans == "[错误] b"


def test_d5_empty():
    assert ReflexionReActAgent._pick_best_final([]) == ("", None)
