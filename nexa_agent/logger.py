"""
nexa_agent 日志配置

- 所有日志输出到控制台 + 汇总文件 logs/react_agent_exp.log
- 每次运行 (start_run_log) 额外写入独立的 logs/run_<timestamp>.log
"""

import logging
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_run_file_handler = None  # Optional[logging.FileHandler]


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """获取配置好的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    # 汇总文件
    log_file = os.path.join(_LOG_DIR, "react_agent_exp.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    # 如果当前有活跃的 run log，也挂上
    if _run_file_handler is not None:
        logger.addHandler(_run_file_handler)

    return logger


def start_run_log(tag: str = "") -> str:
    """开启本次运行的独立日志文件

    为 nexa_agent 下所有 logger 添加一个指向 run_<timestamp>.log 的 handler。

    Args:
        tag: 可选标签，拼入文件名（如 "gaia_l1"）

    Returns:
        日志文件路径
    """
    global _run_file_handler

    stop_run_log()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"run_{ts}{suffix}.log"
    filepath = os.path.join(_LOG_DIR, filename)

    _run_file_handler = logging.FileHandler(filepath, encoding="utf-8")
    _run_file_handler.setLevel(logging.DEBUG)
    _run_file_handler.setFormatter(_FORMATTER)

    # 挂到 nexa_agent 相关的所有 logger 上
    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        if lg.handlers:
            lg.addHandler(_run_file_handler)

    logging.getLogger("run_log").addHandler(_run_file_handler)
    logging.getLogger("run_log").info("=== Run log started: %s ===", filepath)

    return filepath


def stop_run_log() -> None:
    """关闭本次运行的独立日志 handler"""
    global _run_file_handler

    if _run_file_handler is None:
        return

    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        if _run_file_handler in lg.handlers:
            lg.removeHandler(_run_file_handler)

    _run_file_handler.close()
    _run_file_handler = None
