"""
校验器 — V0

分两级:
  L1 规则校验 — 0ms, 零成本
  L2 LLM 校验 — 仅复杂推理/高风险时触发

区分 VISION_DIRECT (自然语言回答) 和 VISION_SCHEMA (dict 结构化结果)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from nexa_agent.logger import get_logger

logger = get_logger("validate")


@dataclass
class ValidateResult:
    passed: bool
    validator_name: str = ""
    issues: List[str] = field(default_factory=list)
    revised_answer: str | None = None
    confidence: float | None = None


# ======================================================================
# VISION_DIRECT — 自然语言回答校验
# ======================================================================

def validate_direct_answer(response: str) -> ValidateResult:
    """校验 VLM 直答结果

    检查点:
      - 非空
      - 不是明显的失败标记（error/失败/抱歉/无法）
      - 长度合理（>5 字符）
    """
    issues = []

    if not response or not response.strip():
        issues.append("VLM 返回空结果")
        return ValidateResult(passed=False, validator_name="direct_answer", issues=issues)

    text = response.strip()

    # 明显失败标记
    failure_markers = [
        "error", "failed", "failure", "cannot", "unable to",
        "无法识别", "无法处理", "无法读取", "无法理解",
        "图片不清晰", "图片无法识别", "图片格式不支持",
        "抱歉，我无法", "sorry",
    ]
    for marker in failure_markers:
        if marker.lower() in text.lower()[:100]:
            issues.append(f"VLM 返回失败标记: {marker}")
            break

    # 长度检查
    if len(text) < 5:
        issues.append("VLM 回答过短")

    ok = len(issues) == 0
    logger.debug("直答校验: passed=%s issues=%d", ok, len(issues))
    return ValidateResult(passed=ok, validator_name="direct_answer", issues=issues)


# ======================================================================
# VISION_SCHEMA — 结构化结果校验
# ======================================================================

def validate_schema_result(response: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """校验 VLM 结构化提取结果

    检查点:
      - JSON 可解析
      - 必填字段存在（自动补空值，不报错）
      - 金额/日期格式是否正确

    Returns:
        (passed, parsed_dict, issues)
    """
    issues = []
    data = _parse_json(response)

    if data is None:
        return False, {}, ["VLM 输出无法解析为 JSON"]

    # 必填字段补充默认值
    defaults = {
        "invoice_code": "", "invoice_date": "", "amount": "",
        "tax_number": "", "seller_name": "", "buyer_name": "",
        "invoice_type": "", "items": [],
    }
    for key, default in defaults.items():
        if key not in data:
            data[key] = default

    # 金额格式校验（宽松）
    amount = str(data.get("amount", ""))
    if amount and not re.search(r"\d+(\.\d{1,2})?", amount):
        issues.append(f"金额格式异常: {amount}")

    # 日期格式校验
    date = str(data.get("invoice_date", ""))
    if date and not re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", date):
        issues.append(f"日期格式异常: {date}")

    ok = len(issues) == 0
    logger.debug("结构化校验: passed=%s issues=%d", ok, len(issues))
    return ok, data, issues


def _parse_json(text: str) -> Dict[str, Any] | None:
    """从 VLM 输出中提取 JSON"""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
