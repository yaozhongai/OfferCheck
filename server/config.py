"""
Nexa Agent V0 核心配置

所有敏感配置 (API Key 等) 从环境变量 / .env 文件读取。
非敏感配置保留默认值，可被环境变量覆盖。

加载优先级: .env 文件 < 环境变量 < 代码默认值
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# 加载 .env 文件 (若 python-dotenv 可用)
try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    """读取环境变量，fallback 到默认值"""
    return os.environ.get(key, default)


@dataclass
class AppConfig:
    """应用全局配置

    所有字段均可通过环境变量覆盖，敏感字段无硬编码默认值。
    """

    # --- 项目路径 ---
    project_root: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ))

    # --- 数据库 ---
    db_path: str = field(default_factory=lambda: _env(
        "DB_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "nexa_agent.db",
        ),
    ))

    # --- 日志 ---
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
    ))

    # NOTE: 模型选择唯一真源是引擎 nexa_agent/config.py（MODEL_TIER + MODEL_ROUTING）。
    # LLM provider key/endpoint、VLM 端点均由引擎侧（nexa_agent/config.py /
    # nexa_agent/tools.py）直接读环境变量，server 层不再持有任何 LLM/VLM 配置。
    # 新增模型/provider 配置一律加到引擎侧，勿在此处扩充。

    # --- 记忆 ---
    stm_max_sessions: int = 1000        # STM 最大同时会话数
    stm_session_ttl_seconds: int = 3600  # STM 会话过期时间
    long_term_sync_interval: int = 10

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Nexa Agent V0"

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# 全局单例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config():
    """重置配置 (测试用)"""
    global _config
    _config = None
