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

    # --- LLM（LEGACY：待 server 变薄后移除）---
    # 模型选择的唯一真源是引擎 nexa_agent/config.py（MODEL_TIER + MODEL_ROUTING，
    # 含 GMI provider 与 strong/fast/upgrade 分层）。以下字段仅服务于尚未迁移的
    # server/llm/client.py + health/lifespan 旧路径；server 变薄（HTTP handler 直调
    # nexa_agent 核心 ReflexionReActAgent）完成后，这些 LLM 字段整体删除，不再由
    # server 决定模型。新增模型/provider 配置一律加到引擎侧，勿在此处扩充。
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "deepseek"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-v4-flash"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "4096")))
    llm_timeout: float = 120.0

    # DeepSeek
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    # Kimi
    kimi_api_key: str = field(default_factory=lambda: _env("MOONSHOT_API_KEY", ""))
    kimi_base_url: str = field(default_factory=lambda: _env("KIMI_BASE_URL", "https://api.moonshot.cn/v1"))

    # GLM (智谱)
    glm_api_key: str = field(default_factory=lambda: _env("ZHIPU_API_KEY", ""))

    # --- 图片识别 ---
    vlm_enabled: bool = True
    vlm_backend: str = "llamacpp"
    vlm_model_name: str = field(default_factory=lambda: _env("VLM_MODEL_NAME", "minicpm-v"))
    vlm_base_url: str = field(default_factory=lambda: _env("VLM_BASE_URL", "http://127.0.0.1:8080/v1"))
    vlm_timeout: float = 120.0
    vlm_ctx_size: int = 4096

    # --- 记忆 ---
    stm_max_sessions: int = 1000        # STM 最大同时会话数
    stm_session_ttl_seconds: int = 3600  # STM 会话过期时间
    long_term_sync_interval: int = 10

    # --- 自我反思 ---
    reflection_max_retries: int = 2
    reflection_prompt: str = (
        "请对上述回答进行质量检查，判断：\n"
        "1. 回答是否准确？\n"
        "2. 是否有遗漏？\n"
        "3. 是否需要补充？\n"
        "如果存在问题，请指出并给出修正意见。"
    )

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
