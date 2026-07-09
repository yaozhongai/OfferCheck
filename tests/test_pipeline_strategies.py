"""react_loop pipeline 策略单测（评审 3.5：从闭包抽出的纯函数可脱离主循环单测）。

should_gate_block（强制取证 gate 决策）+ build_structured_sources（结构化来源）。
"""

from nexa_agent.react_agent import should_gate_block, build_structured_sources


# ── 强制取证 gate 决策 ────────────────────────────────────────────────────
def test_gate_blocks_verdict_without_retrieval():
    # stage 调查 + 零成功检索 + 未超额度 → 拦截
    assert should_gate_block("[Verdict] 靠谱 —— ok", stage="offercheck_stage4",
                             answer_mode=False, successful_retrievals=0, evidence_gate_nags=0)


def test_gate_passes_when_retrieval_done():
    assert not should_gate_block("[Verdict] 靠谱", stage="offercheck_stage4",
                                 answer_mode=False, successful_retrievals=2, evidence_gate_nags=0)


def test_gate_respects_nag_budget():
    # 已达提醒上限 → 放行（防死锁）
    assert not should_gate_block("[Verdict] 靠谱", stage="offercheck_stage4",
                                 answer_mode=False, successful_retrievals=0,
                                 evidence_gate_nags=2, max_nags=2)


def test_gate_answer_mode_conversational_bypass():
    # answer-mode 非裁定对话式回答 → 不强制新检索
    assert not should_gate_block("你的 offer 年薪不错，竞业两年偏长。", stage="offercheck_stage4",
                                 answer_mode=True, successful_retrievals=0, evidence_gate_nags=0)
    # 但 answer-mode 下仍下裁定 → 照拦
    assert should_gate_block("[Verdict] 大概率有坑", stage="offercheck_stage4",
                             answer_mode=True, successful_retrievals=0, evidence_gate_nags=0)


def test_gate_generic_nonverdict_not_forced():
    # 无 stage、无裁定/溯源标签 → 纯常识问答不强制取证
    assert not should_gate_block("2+2=4", stage=None, answer_mode=False,
                                 successful_retrievals=0, evidence_gate_nags=0)


# ── 结构化来源 ────────────────────────────────────────────────────────────
def test_sources_marks_verified_and_backfills():
    ans = "见 https://bytedance.com/careers 与 https://made-up-unseen.example/x"
    seen = {"https://bytedance.com/careers", "https://news.site/report"}
    out = build_structured_sources(ans, seen)
    by_url = {s["domain"]: s for s in out}
    assert by_url["bytedance.com"]["verified"] is True       # 引用且见过
    assert by_url["made-up-unseen.example"]["verified"] is False  # 引用但没见过
    assert "news.site" in by_url                              # seen_urls 回填


def test_sources_filters_localhost_and_caps():
    ans = " ".join(f"https://d{i}.com/x" for i in range(20)) + " http://localhost:8000/api"
    out = build_structured_sources(ans, set(), limit=12)
    assert len(out) == 12                                     # 上限
    assert all(s["domain"] != "localhost" for s in out)      # 本地地址过滤
