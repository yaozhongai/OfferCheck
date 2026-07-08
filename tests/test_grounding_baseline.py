"""批次 0 — 接地层纯函数特征化（回归护栏）

锁定 attribute_sources / _truncate_observation / _render_verdict 的当前行为。
这些函数在批次 1 里不应被改动；本测试若变红即说明接地层被意外触碰。
"""

import pytest

from nexa_agent.react_agent import (
    attribute_sources, _truncate_observation, _render_verdict,
)


# ── AIS 来源对账 ────────────────────────────────────────────────────────

def test_attribute_sources_flags_unseen_url():
    answer = (
        "[Verdict] 存疑\n"
        "[Fact] 公司融资信息\n"
        "[Source] https://fake-unseen.example.com/funding\n"
    )
    annotated, report = attribute_sources(answer, seen_urls=set(), called_tools=set())
    assert report["total_sources"] == 1
    assert report["unverified"] == 1
    assert "⚠️" in annotated


def test_attribute_sources_passes_seen_url():
    url = "https://real.example.com/page"
    answer = f"[Fact] x\n[Source] {url}\n"
    annotated, report = attribute_sources(answer, seen_urls={url}, called_tools=set())
    assert report["unverified"] == 0
    assert "⚠️" not in annotated


# ── 观察截断分档 ────────────────────────────────────────────────────────

def test_truncate_short_tool_capped_at_3000():
    obs = "y" * 5000
    out = _truncate_observation("web_search", obs)
    assert len(out) < 5000
    assert "已截断" in out


def test_truncate_long_tool_sandwich():
    obs = "z" * 20000
    out = _truncate_observation("web_fetch", obs)
    assert "省略" in out  # 三明治截断保留首尾


def test_truncate_long_tool_small_untouched():
    obs = "z" * 1000
    assert _truncate_observation("web_fetch", obs) == obs


# ── 裁定渲染 ────────────────────────────────────────────────────────────

def test_render_verdict_labels():
    out = _render_verdict({
        "verdict": "大概率有坑",
        "summary": "仿冒域名 + 预付款",
        "evidence": ["域名近期注册"],
        "red_flags": ["要求押金"],
        "need_user_confirm": ["核对 HR 邮箱域名"],
    })
    assert "[Verdict] 大概率有坑" in out
    assert "[Fact] 域名近期注册" in out
    assert "[RedFlag] 要求押金" in out
    assert "[NeedUserConfirm] 核对 HR 邮箱域名" in out
