"""get_current_time — 时间查询工具"""

from __future__ import annotations

from datetime import datetime

from app.tools import register
from app.utils.logger_config import get_logger

logger = get_logger("tools.time")


@register(
    name="get_current_time",
    description="返回当前本地日期和时间",
    signature="get_current_time()",
    examples=["get_current_time()"],
)
def get_current_time(_param: str = "") -> str:
    logger.info("get_current_time 调用")
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"当前时间: %Y年%m月%d日 {weekdays[now.weekday()]} %H:%M:%S")
