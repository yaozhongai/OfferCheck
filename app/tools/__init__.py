"""
ReAct 工具集 — Tool Registry + 执行入口

对齐 react_exp/tools.py 的 @register 装饰器 + TOOLS 注册表模式。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.utils.logger_config import get_logger

logger = get_logger("tools")

# 全局工具注册表
TOOLS: Dict[str, Callable[[str], str]] = {}
TOOL_META: Dict[str, Dict[str, Any]] = {}


def register(name: str, description: str, signature: str, examples: List[str]):
    """装饰器：将函数注册到 TOOLS 全局表"""
    def decorator(func: Callable[[str], str]):
        TOOLS[name] = func
        TOOL_META[name] = {
            "description": description,
            "signature": signature,
            "examples": examples,
        }
        return func
    return decorator


def execute_tool(tool_name: str, tool_args: str) -> str:
    """根据工具名和参数执行对应的工具函数"""
    if tool_name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"[错误] 未知工具: {tool_name}。可用工具: {available}"

    tool_func = TOOLS[tool_name]
    try:
        return tool_func(tool_args)
    except Exception as exc:
        logger.error("工具 %s 执行异常: %s", tool_name, exc, exc_info=True)
        return f"[错误] 工具 {tool_name} 执行失败: {exc}"


def get_tools_description() -> str:
    """生成工具列表描述（用于 System Prompt）"""
    lines = []
    for name, meta in TOOL_META.items():
        lines.append(f"- **{meta['signature']}**: {meta['description']}")
        for ex in meta["examples"]:
            lines.append(f"  - 示例: `{ex}`")
    return "\n".join(lines)


# 延迟导入各工具模块，触发 @register 装饰器
def _import_tools():
    import app.tools.web_search       # noqa
    import app.tools.calculator       # noqa
    import app.tools.time_tool        # noqa
    import app.tools.image_tools      # noqa
    import app.tools.content_extract  # noqa


_import_tools()
