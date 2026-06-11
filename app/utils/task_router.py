"""
任务路由 — V0

根据用户输入和上下文，将请求分配到最短执行路径。
仅用规则匹配，不调用 LLM。

路径:
  VISION_DIRECT   — VLM 直答 → 轻量校验 → 返回
  VISION_SCHEMA   — VLM 结构化提取 → 规则校验 → 返回
  VISION_REASON   — VLM 感知 → 检索/规则 → LLM 推理 → 按需校验
  TEXT_QA         — 检索 → LLM
  TOOL_ACT        — 规划 → 风险校验 → 确认/执行 (V1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from app.utils.logger_config import get_logger

logger = get_logger("task_router")


# ======================================================================
# 枚举
# ======================================================================

class RouteType(str, Enum):
    VISION_DIRECT = "vision_direct"
    VISION_SCHEMA = "vision_schema"
    VISION_REASON = "vision_reason"
    TEXT_QA = "text_qa"
    TOOL_ACT = "tool_act"
    REACT = "react"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ======================================================================
# 关键词
# ======================================================================

# 图片 → 结构化提取（不是简单问答）
_SCHEMA_KEYWORDS = [
    "提取", "结构化", "json", "字段", "表格",
    "发票代码", "发票号码", "发票号", "税号",
    "整理", "汇总", "列出",
]

# 图片 → 复杂推理（需要业务知识/规则/历史上下文）
_REASON_KEYWORDS = [
    "是否可以", "能不能", "合不合规", "符不符合",
    "原因", "为什么", "怎么回事", "什么原因",
    "风险", "隐患", "异常", "是否有问题",
    "建议", "应该怎么", "如何处理", "怎么办",
    "判断", "评估", "分析",
]

# 工具执行（强动作词，不用泛词如"查询""生成"）
_TOOL_KEYWORDS = [
    "重启设备", "下发配置", "修改参数", "修改设置",
    "生成工单", "创建工单", "发送邮件", "发邮件",
    "调用接口", "执行命令", "部署", "发布",
    "禁用", "启用", "切换模式",
]


# ======================================================================
# 路由函数
# ======================================================================

@dataclass
class RouteResult:
    """路由决策结果"""
    route_type: RouteType = RouteType.UNKNOWN
    confidence: float = 0.0
    reason: str = ""
    matched_rules: list = field(default_factory=list)
    should_use_react: bool = False
    risk_level: Any = None  # RiskLevel

    # 各阶段是否需要执行
    need_retrieve: bool = False
    need_reason: bool = False
    need_verify: bool = False
    need_memory: bool = False


def route_task(user_input: str, image_path: str | None = None, parsed_content: str = "") -> RouteResult:
    """路由入口

    Args:
        user_input: 用户文本输入
        image_path: 图片路径（None 表示纯文本）
        parsed_content: 已有识别内容（重试场景）
    """
    text = (user_input or "").strip().lower()

    # ── 有图片 ──
    if image_path:
        return _route_vision(text)

    # ── 纯文本：工具执行 ──
    if any(kw in text for kw in _TOOL_KEYWORDS):
        logger.info("路由: TOOL_ACT")
        return RouteResult(
            route_type=RouteType.TOOL_ACT,
            risk_level=RiskLevel.MEDIUM,
            need_reason=True,
            need_verify=True,
            need_memory=True,
            confidence=0.95,
            reason="tool_keyword",
            matched_rules=["tool_keyword"],
        )

    # ── 纯文本：TOOL_ACT ──
    logger.info("路由: TOOL_ACT (文本)")
    return RouteResult(
        route_type=RouteType.TOOL_ACT,
        confidence=0.85,
        risk_level=RiskLevel.LOW,
        need_retrieve=True,
        need_reason=True,
        need_memory=False,
        reason="default_text",
        matched_rules=["default_text"],
    )


def _route_vision(text: str) -> RouteResult:
    """有图片时的路由判断"""

    # TOOL_ACT 优先
    if any(kw in text for kw in _TOOL_KEYWORDS):
        logger.info("路由: TOOL_ACT (vision)")
        return RouteResult(
            route_type=RouteType.TOOL_ACT,
            risk_level=RiskLevel.MEDIUM,
            need_reason=True,
            need_verify=True,
            need_memory=True,
            confidence=0.95,
            reason="tool_keyword",
            matched_rules=["tool_keyword"],
        )

    # 结构化提取
    if any(kw in text for kw in _SCHEMA_KEYWORDS):
        logger.info("路由: VISION_SCHEMA")
        return RouteResult(
            route_type=RouteType.VISION_SCHEMA,
            confidence=0.95,
            risk_level=RiskLevel.LOW,
            need_retrieve=False,
            need_reason=False,
            need_verify=False,
            need_memory=False,
            reason="schema_keyword",
            matched_rules=["schema_keyword"],
        )

    # 复杂推理
    if any(kw in text for kw in _REASON_KEYWORDS):
        logger.info("路由: VISION_REASON")
        return RouteResult(
            route_type=RouteType.VISION_REASON,
            confidence=0.90,
            risk_level=RiskLevel.MEDIUM,
            need_retrieve=True,
            need_reason=True,
            need_verify=True,
            need_memory=False,
            reason="reason_keyword",
            matched_rules=["reason_keyword"],
        )

    # 默认：图片直答
    logger.info("路由: VISION_DIRECT")
    return RouteResult(
        route_type=RouteType.VISION_DIRECT,
        confidence=0.90,
        risk_level=RiskLevel.LOW,
        need_retrieve=False,
        need_reason=False,
        need_verify=False,
        need_memory=False,
        reason="default_vision",
        matched_rules=["default_vision"],
    )
