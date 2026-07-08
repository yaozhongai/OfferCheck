"""AIS 联动降级测试（评审 2.1）

>1/3 来源未验证时：靠谱→存疑（只增加谨慎）；有坑/存疑不弱化，只补置信说明。
分类走 label-first（复用 _classify_verdict_level），reason 里的否定词不误伤。
"""

import pytest

from nexa_agent.react_agent import apply_ais_confidence_downgrade


def _attr(total, unver):
    return {"total_sources": total, "unverified": unver}


def test_reliable_downgraded_when_over_one_third():
    ans = "[Verdict] 靠谱 —— 公司真实可信\n[Fact] a\n[Source] http://a.com"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(3, 2), lang="zh")
    assert triggered is True
    assert "[Verdict] 存疑" in out
    assert "自动降级" in out
    assert "[NeedUserConfirm]" in out


def test_exactly_one_third_not_triggered():
    ans = "[Verdict] 靠谱 —— 公司真实"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(3, 1), lang="zh")  # 1/3，不 > 1/3
    assert triggered is False
    assert out == ans


def test_scam_verdict_not_weakened():
    ans = "[Verdict] 大概率有坑 —— 域名仿冒 + 预付款\n[Source] http://x.com"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(4, 3), lang="zh")
    assert triggered is True
    # label 不被弱化（仍是大概率有坑），但补了置信说明
    assert "[Verdict] 大概率有坑" in out
    assert "存疑" not in out.split("\n")[0]   # 首行裁定未被改成存疑
    assert "[NeedUserConfirm]" in out


def test_suspicious_verdict_gets_caveat_only():
    ans = "[Verdict] 存疑 —— 信息不足"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(2, 2), lang="zh")
    assert triggered is True
    assert "[Verdict] 存疑" in out
    assert "[NeedUserConfirm]" in out


def test_no_sources_no_change():
    ans = "[Verdict] 靠谱 —— 无引用来源"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(0, 0), lang="zh")
    assert triggered is False
    assert out == ans


def test_label_first_classification_no_negation_misfire():
    # label 是「靠谱」，reason 含「有坑」否定语境——应按 label 判 reliable 并降级
    ans = "[Verdict] 靠谱 —— 未发现任何有坑或诈骗迹象\n[Source] http://a.com"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(3, 2), lang="zh")
    assert triggered is True
    assert out.split("\n")[0].startswith("[Verdict] 存疑")


def test_english_downgrade_label():
    ans = "[Verdict] Looks Legit — company verified\n[Source] http://a.com"
    out, triggered = apply_ais_confidence_downgrade(ans, _attr(3, 2), lang="en")
    assert triggered is True
    assert "[Verdict] Suspicious" in out
    assert "auto-downgraded" in out
