"""共享 JSON 抽取工具测试（评审 1.7）

覆盖脆弱正则 `\\{[^}]+\\}` 治不了的三类：嵌套对象、markdown 围栏、max_tokens 截断。
"""

import json

import pytest

from nexa_agent.util.json_extract import extract_json_block, repair_truncated_json


def test_nested_object_not_truncated_at_first_brace():
    # 旧正则 \{[^}]+\} 会在第一个 } 处停下，拿到残缺 JSON
    text = '前言 {"verdict": "存疑", "meta": {"k": 1}, "arr": [1, 2]} 后记'
    block = extract_json_block(text)
    data = json.loads(block)
    assert data["meta"]["k"] == 1
    assert data["arr"] == [1, 2]


def test_markdown_code_fence_stripped():
    text = '```json\n{"success": true, "reason": "ok"}\n```'
    data = json.loads(extract_json_block(text))
    assert data["success"] is True


def test_truncated_json_repaired():
    # 模拟被 max_tokens 截断：字符串与数组、对象都未闭合
    text = '{"fact_verdicts": [{"index": 1, "reliable": true, "reason": "来源可靠'
    block = extract_json_block(text)
    repaired = repair_truncated_json(block)
    data = json.loads(repaired)  # 不应抛
    assert data["fact_verdicts"][0]["index"] == 1


def test_no_json_returns_none():
    assert extract_json_block("完全没有大括号的一段话") is None
    assert extract_json_block("") is None
