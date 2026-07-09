"""结构化 Verdict 构建测试（评审 3.2）

引擎在 submit_verdict 处直接构建权威结构化裁定（label + 归一化等级 + 摘要 + 红旗），
供 SSE→前端免文本解析。verdict_level 走 label-first 分类，否定语境不误伤。
"""

import pytest

from nexa_agent.react_agent import _build_verdict_struct


def test_scam_verdict():
    s = _build_verdict_struct({
        "verdict": "大概率有坑",
        "summary": "仿冒域名 + 预付款",
        "red_flags": ["要求押金", "仅 Telegram 沟通"],
        "need_user_confirm": ["核对 HR 邮箱域名"],
    })
    assert s["verdict"] == "大概率有坑"
    assert s["verdict_level"] == "likely_scam"
    assert s["summary"] == "仿冒域名 + 预付款"
    assert s["red_flags"] == ["要求押金", "仅 Telegram 沟通"]
    assert s["need_user_confirm"] == ["核对 HR 邮箱域名"]


def test_reliable_with_negation_reason_not_misclassified():
    # label 是「靠谱」，summary 含「诈骗」否定语境——等级须按 label 判 reliable（不误伤）
    s = _build_verdict_struct({
        "verdict": "靠谱",
        "summary": "多渠道核实，未发现任何诈骗或有坑迹象",
    })
    assert s["verdict"] == "靠谱"
    assert s["verdict_level"] == "reliable"


def test_english_label():
    s = _build_verdict_struct({"verdict": "Likely a Scam", "summary": "impersonation"})
    assert s["verdict_level"] == "likely_scam"


def test_suspicious_and_empty():
    assert _build_verdict_struct({"verdict": "存疑", "summary": "无法核实"})["verdict_level"] == "suspicious"
    empty = _build_verdict_struct({})
    assert empty["verdict"] == "" and empty["verdict_level"] == "unknown"


def test_list_fields_normalized():
    # 模型可能把 red_flags 传成单字符串——须规整为字符串列表（复用 _as_str_list）
    s = _build_verdict_struct({"verdict": "存疑", "summary": "x", "red_flags": "单条红旗"})
    assert s["red_flags"] == ["单条红旗"]
