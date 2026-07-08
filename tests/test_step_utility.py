"""信用分配对现役工具的效用测试（评审 1.3）

web_fetch / read_pdf / read_xlsx / domain_whois_lookup 此前全落默认 0.0，
critical_step 定位与中途纠偏因此漏掉这些工具的成败。
"""

import pytest

from nexa_agent.react_agent import _compute_step_utility

_LONG = "x" * 600
_SHORT = "x" * 20


@pytest.mark.parametrize("tool", ["web_fetch", "read_pdf", "read_xlsx", "tavily_extract"])
def test_content_tools_reward_long_content(tool):
    assert _compute_step_utility(tool, "http://a", _LONG, []) == 1.0
    assert _compute_step_utility(tool, "http://a", _SHORT, []) == -0.3


def test_whois_counts_as_retrieval():
    # 成功返回注册信息 → 正效用（此前落 0.0）
    assert _compute_step_utility(
        "domain_whois_lookup", "acme.com", "Registrar: GoDaddy, created 2026-06", []
    ) == 0.5
    # 查无 → 0.0（合法的"未找到"）
    assert _compute_step_utility(
        "domain_whois_lookup", "acme.com", "未找到该域名的注册信息", []
    ) == 0.0


def test_error_still_negative():
    assert _compute_step_utility("web_fetch", "http://a", "[错误] 抓取失败", []) == -0.5


def test_repeat_penalized():
    hist = [("web_fetch", "http://a")]
    assert _compute_step_utility("web_fetch", "http://a", _LONG, hist) == -0.5
