"""LLM JSON 响应的健壮抽取 + 截断修复。

instruct 模型（GMI 上的 DeepSeek/Kimi/Qwen）没有可靠的 structured-output 模式，
JSON 常夹带 markdown 围栏、前后说明文字，或在 max_tokens 处被截断。本模块提供
两个纯函数供所有需要解析 LLM JSON 的地方复用（verifier / evaluator / stage_router）：

- extract_json_block: 从任意文本里抠出最外层 JSON 对象（括号配对，忽略字符串内括号）
- repair_truncated_json: 补齐被截断的 JSON（闭合未结束的字符串与括号）

原实现位于 verifier.py，此处抽出为共享工具（评审 1.7：消灭同仓两套 JSON 容错）。
"""

from __future__ import annotations

import re
from typing import Optional


def extract_json_block(text: str) -> Optional[str]:
    """从 LLM 响应里抠出最外层 JSON 对象。

    容错 markdown 代码围栏（```json ... ```）与前后夹带的说明文字，
    从第一个 `{` 起做括号配对（忽略字符串内的括号）取到匹配的 `}`。
    截断（无匹配闭合）时返回从 `{` 到结尾的整段，交给 repair_truncated_json 补齐。
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    # 未闭合 → 截断，返回整段供修复
    return t[start:]


def repair_truncated_json(s: str) -> str:
    """补齐被 max_tokens 截断的 JSON：闭合未结束的字符串与括号。

    尽力而为——去掉尾部残缺 token 后，按栈补上缺失的 `]` / `}`。
    修复后仍可能非法（如遗留尾逗号），由调用方 try/except 兜底。
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    repaired = s
    if in_str:
        repaired += '"'
    # 去掉尾部残缺片段（截断在逗号/冒号/未完成键值处）
    repaired = re.sub(r"[,:]\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired
