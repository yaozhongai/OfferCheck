"""启发式评估规则测试（评审 1.4）

- 已移除 wrong_reasoning/不确定性规则：含「可能/存疑/信息不足」的合法裁定不再被误判失败。
- tool_gap 现在正确作用于 trajectory（旧 dispatch 曾把它误喂 answer）。
"""

import pytest

from nexa_agent.evaluator import Evaluator, HEURISTIC_RULES


def test_uncertainty_rule_removed():
    names = [r[0] for r in HEURISTIC_RULES]
    assert "wrong_reasoning" not in names


def test_hedged_verdict_not_flagged_as_failure():
    ev = Evaluator(mode="heuristic")
    # 合法的「存疑」裁定，含「可能」——符合 SPEC「宁可存疑」教义，不应判失败
    answer = "[Verdict] 存疑 —— 公司信息无法完全核实，可能存在风险，建议进一步确认"
    trajectory = (
        "### Step 1\nAction: web_search(acme company)\n"
        "Observation: 找到一些零散信息\n"
        "### Step 2\nAction: domain_whois_lookup(acme.com)\nObservation: 注册于近期"
    )
    result = ev.evaluate(task="核实这家公司", answer=answer, trajectory=trajectory,
                         terminated_reason="final_answer")
    assert result.success is True


def test_tool_gap_detected_from_trajectory():
    ev = Evaluator(mode="heuristic")
    url = "https://example.com/report.pdf"
    # 同一 PDF 被访问 3 次 + 轨迹含「无法」——tool_gap 需从 trajectory 取 URL
    trajectory = "\n".join(
        f"### Step {i}\nAction: web_fetch({url})\nObservation: 无法解析，乱码"
        for i in range(1, 4)
    )
    result = ev.evaluate(task="读取报告", answer="", trajectory=trajectory,
                         terminated_reason="final_answer")
    assert result.success is False
    assert result.failure_mode == "tool_gap"
