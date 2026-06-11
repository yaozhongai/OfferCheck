"""calculator — 数学表达式计算"""

from __future__ import annotations

from app.tools import register
from app.utils.logger_config import get_logger

logger = get_logger("tools.calculator")


@register(
    name="calculator",
    description="安全计算数学表达式，支持 sqrt/log/sin/cos/pow 等函数",
    signature="calculator(expression)",
    examples=["calculator(sqrt(144) + pow(2, 10))"],
)
def calculator(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        return "[错误] calculator: 表达式不能为空"

    logger.info("calculator expression=%s", expression)

    try:
        import numexpr
        result = numexpr.evaluate(expression)
        if hasattr(result, "item"):
            result = result.item()
        return f"计算结果: {expression} = {result}"
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("numexpr 失败: %s，尝试 eval fallback", exc)

    # eval fallback
    import math
    safe_dict = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "pow": pow, "abs": abs, "pi": 3.141592653589793, "e": 2.718281828459045,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {expression} = {result}"
    except Exception as exc:
        return f"[错误] calculator: 计算失败 - {exc}"
