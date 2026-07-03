"""
Reflexion + ReAct 统一配置管理

所有超参数集中管理，可通过环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

# 加载 .env（config 可能被独立 import，不依赖上游先加载）
try:
    from dotenv import load_dotenv

    _dotenv_path = Path(__file__).parent.parent / ".env"
    if _dotenv_path.exists():
        load_dotenv(_dotenv_path)
except ImportError:
    pass

# ==========================================================================
# LLM Provider（gmi | deepseek）
# ==========================================================================
# 比赛期主推理经 GMI Cloud Inference Engine（OpenAI 兼容）承载；
# LLM_PROVIDER 未显式指定时，优先 GMI（有 key 时），否则回落 DeepSeek 官方 API。

_PROVIDER_PRESETS = {
    "gmi": {
        "base_url": os.environ.get("GMI_BASE_URL", "https://api.gmi-serving.com/v1"),
        "api_key": os.environ.get("GMI_API_KEY", ""),
        # 全非思考模型：GMI 无法用参数关闭 V4 思考，思考模式在多轮 tool-calling
        # 下强制回传 reasoning_content（否则 400），且常出现 reasoning-only 空 content。
        # 换用非思考 instruct 模型从根上规避（实测见 docs/run_20260702_234927_failure_analysis.md）。
        "strong_model": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "fast_model": "deepseek-ai/DeepSeek-V3.2",
    },
    "deepseek": {
        "base_url": os.environ.get(
            "DEEPSEEK_BASE_URL",
            os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com"),
        ),
        "api_key": os.environ.get(
            "DEEPSEEK_API_KEY",
            os.environ.get("KIMI_API_KEY", ""),
        ),
        "strong_model": "deepseek-v4-pro",
        "fast_model": "deepseek-v4-flash",
    },
}

_default_provider = "gmi" if _PROVIDER_PRESETS["gmi"]["api_key"] else "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", _default_provider).lower()
if LLM_PROVIDER not in _PROVIDER_PRESETS:
    LLM_PROVIDER = "deepseek"

_PRESET = _PROVIDER_PRESETS[LLM_PROVIDER]

# thinking 开关是 DeepSeek 官方 API 的私有扩展参数；
# GMI 等 OpenAI 兼容网关收到会报 422，必须跳过
SUPPORTS_THINKING_PARAM = LLM_PROVIDER == "deepseek"

# ==========================================================================
# 模型配置
# ==========================================================================

MODEL_CONFIG = {
    "provider": LLM_PROVIDER,
    "model": os.environ.get("LLM_MODEL", _PRESET["strong_model"]),
    "base_url": _PRESET["base_url"],
    "api_key": _PRESET["api_key"],
    "react_temperature": 0.0,
    "reflection_temperature": 0.3,
}

# ==========================================================================
# 模型分层（任务感知的动态模型选择）
# ==========================================================================

MODEL_TIER = {
    "strong": {
        "model": os.environ.get("STRONG_MODEL", _PRESET["strong_model"]),
        "description": "深度推理 — 首步规划、规则演化等需要强推理能力的任务",
    },
    "fast": {
        "model": os.environ.get("FAST_MODEL", _PRESET["fast_model"]),
        "description": "快速执行 — 后续步骤、反思生成、评估判断等成本敏感任务",
    },
    "upgrade": {
        # 备援模型：连续 N 步 LLM 未发 tool_calls 时动态切入。
        # 必须是非思考模型——旧选 Qwen3.6-35B-A3B 实为思考模型，升级后照样
        # 只输出 reasoning 不发 tool_calls（方向反了）。Kimi-K2-Instruct 是
        # 非思考且工具调用强的备援（实测见失败分析文档）。
        "model": os.environ.get("UPGRADE_MODEL", "moonshotai/Kimi-K2-Instruct-0905"),
        "description": "备援 tool-call — 连续 N 步未发 tool_calls 时自动切入",
    },
}

MODEL_ROUTING = {
    "react_first":       "strong",   # 首步规划 — 需要深度思考
    "react_main":        "fast",     # 后续步骤 — 多步执行，成本敏感
    "reflection":        "fast",     # 反思生成 — 结构化输出，复杂度低
    "evaluator_llm":     "fast",     # LLM 评估 — 判断逻辑简单
    "lesson_extract":    "fast",     # 教训提取 — 归纳总结
    "evolver":           "strong",   # 规则演化 — 低频高要求
    "tool_call_upgrade": "upgrade",  # 备援 — 动态升级触发时使用
}

# 动态升级阈值：连续 N 步 LLM 未发 tool_calls（且无 Final Answer）后自动切 upgrade 层
DYNAMIC_UPGRADE_THRESHOLD = 2

# ==========================================================================
# 视觉模型（云端图片理解 — analyze_image_cloud）
# ==========================================================================
# 文档「GMI模型选择」规定云端图片提取走 GMI 的 Gemini 3.1：
#   普通图片 → gemini-3.1-flash-lite-preview；复杂/模糊 → gemini-3.1-pro-preview
# 有 GMI key 时优先 GMI；否则回落 Kimi 多模态（旧路径），保证无 GMI 时仍可用。

_VISION_GMI_KEY = os.environ.get("GMI_API_KEY", "")

VISION_CONFIG = {
    "provider": os.environ.get(
        "VISION_PROVIDER", "gmi" if _VISION_GMI_KEY else "kimi"
    ).lower(),
    "gmi": {
        "base_url": os.environ.get("GMI_BASE_URL", "https://api.gmi-serving.com/v1"),
        "api_key": _VISION_GMI_KEY,
        # 普通图片默认模型
        "model": os.environ.get("VISION_MODEL", "google/gemini-3.1-flash-lite-preview"),
        # 复杂/模糊图片升级模型（flash-lite 空响应时自动重试）
        "model_complex": os.environ.get(
            "VISION_MODEL_COMPLEX", "google/gemini-3.1-pro-preview"
        ),
    },
    "kimi": {
        "base_url": os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "api_key": os.environ.get("MOONSHOT_API_KEY", ""),
        "model": "kimi-k2.6",
    },
}


# ==========================================================================
# ReAct 配置
# ==========================================================================

REACT_CONFIG = {
    "max_steps": int(os.environ.get("REACT_MAX_STEPS", "16")),
    "observation_max_chars": int(os.environ.get("REACT_OBS_MAX_CHARS", "2000")),
    "compression_after_steps": int(os.environ.get("REACT_COMPRESS_AFTER", "5")),
}

# ==========================================================================
# Reflexion 配置
# ==========================================================================

REFLEXION_CONFIG = {
    "enabled": os.environ.get("REFLEXION_ENABLED", "true").lower() == "true",
    "max_trials": int(os.environ.get("REFLEXION_MAX_TRIALS", "3")),
    "max_memory_size": int(os.environ.get("REFLEXION_MAX_MEMORY", "3")),
    "evaluator_mode": os.environ.get("REFLEXION_EVAL_MODE", "hybrid"),  # heuristic | llm | hybrid
    "persist_memory": os.environ.get("REFLEXION_PERSIST_MEMORY", "false").lower() == "true",
    "reflection_model": os.environ.get("REFLEXION_MODEL", "deepseek-v4-pro"),
    "reflection_temperature": float(os.environ.get("REFLEXION_TEMPERATURE", "0.3")),
    "min_reflection_length": int(os.environ.get("REFLEXION_MIN_LENGTH", "20")),
    "max_reflection_length": int(os.environ.get("REFLEXION_MAX_LENGTH", "500")),
}

# ==========================================================================
# 记忆配置
# ==========================================================================

MEMORY_CONFIG = {
    "episodic": {
        "max_size": int(os.environ.get("MEM_EPISODIC_MAX", "10")),
        "eviction_policy": os.environ.get("MEM_EVICTION", "composite"),
        "injection_top_k": int(os.environ.get("MEM_INJECT_TOP_K", "3")),
    },
    "persist_dir": os.environ.get(
        "MEM_PERSIST_DIR",
        str(Path(__file__).parent / "memory"),
    ),
}

# ==========================================================================
# 搜索配置（可插拔 provider 抽象层）
# ==========================================================================

SEARCH_CONFIG = {
    # provider 优先级（有序降级）。第一个"可用且健康"的 provider 胜出。
    # 默认 Tavily 优先（额度够时质量最好）→ 额度尽时无缝切自建 SearXNG。
    "provider_order": [
        p.strip()
        for p in os.environ.get("SEARCH_PROVIDER_ORDER", "tavily,searxng,exa,ddg").split(",")
        if p.strip()
    ],
    "max_results": int(os.environ.get("SEARCH_MAX_RESULTS", "5")),
    "snippet_max_chars": int(os.environ.get("SEARCH_SNIPPET_MAX_CHARS", "300")),
    "request_timeout": int(os.environ.get("SEARCH_TIMEOUT", "15")),
    # 自建 SearXNG 后端（见 searxng/ 目录）
    "searxng_base_url": os.environ.get("SEARXNG_BASE_URL", "http://localhost:8888"),
    # Exa（可选，1000/月免费，免信用卡）
    "exa_api_key": os.environ.get("EXA_API_KEY", ""),
    # 健康熔断：单 provider 连续失败 N 次后，冷却一段时间内跳过
    "health_fail_threshold": int(os.environ.get("SEARCH_HEALTH_FAIL_THRESHOLD", "3")),
    "health_cooldown_sec": int(os.environ.get("SEARCH_HEALTH_COOLDOWN", "120")),
    # 增强层：对 top-k 结果抓取正文替换摘要，使自建源质量贴近 Tavily
    "enrich_enabled": os.environ.get("SEARCH_ENRICH_ENABLED", "true").lower() == "true",
    "enrich_top_k": int(os.environ.get("SEARCH_ENRICH_TOP_K", "3")),
    "enrich_max_chars": int(os.environ.get("SEARCH_ENRICH_MAX_CHARS", "1200")),
    "enrich_timeout": int(os.environ.get("SEARCH_ENRICH_TIMEOUT", "12")),
    # 仅增强摘要偏弱的 provider；Tavily/Exa 自带优质正文，无需重复抓取
    "enrich_providers": [
        p.strip()
        for p in os.environ.get("SEARCH_ENRICH_PROVIDERS", "searxng,ddg").split(",")
        if p.strip()
    ],
}

# ==========================================================================
# 路径配置
# ==========================================================================

_PROJECT_DIR = Path(__file__).parent

PATH_CONFIG = {
    "prompts_dir": str(_PROJECT_DIR / "prompts"),
    "memory_dir": str(_PROJECT_DIR / "memory"),
    "reflections_file": str(_PROJECT_DIR / "memory" / "reflections.json"),
    "logs_dir": str(_PROJECT_DIR / "logs"),
}


def get_model_for_role(role: str) -> str:
    """根据角色获取对应的模型名

    Args:
        role: 角色标识 — "react_first" | "react_main" | "reflection" |
              "evaluator_llm" | "lesson_extract" | "evolver"

    Returns:
        模型名称字符串
    """
    tier = MODEL_ROUTING.get(role, "strong")
    return MODEL_TIER[tier]["model"]


def get_config_summary() -> str:
    """生成配置摘要（用于调试和日志）"""
    lines = [
        "=" * 50,
        "Reflexion + ReAct 配置",
        "=" * 50,
        f"LLM Provider: {LLM_PROVIDER} ({MODEL_CONFIG['base_url']})",
        f"模型层级: strong={MODEL_TIER['strong']['model']}, fast={MODEL_TIER['fast']['model']}",
        f"路由策略: react_first→{MODEL_ROUTING['react_first']}, react_main→{MODEL_ROUTING['react_main']}",
        f"          reflection→{MODEL_ROUTING['reflection']}, eval→{MODEL_ROUTING['evaluator_llm']}",
        f"ReAct 最大步数: {REACT_CONFIG['max_steps']}",
        f"Reflexion 启用: {REFLEXION_CONFIG['enabled']}",
        f"最大 Trial 数: {REFLEXION_CONFIG['max_trials']}",
        f"记忆容量: {REFLEXION_CONFIG['max_memory_size']}",
        f"评估模式: {REFLEXION_CONFIG['evaluator_mode']}",
        f"记忆持久化: {REFLEXION_CONFIG['persist_memory']}",
        f"动态升级阈值: 连续 {DYNAMIC_UPGRADE_THRESHOLD} 次解析失败",
        "=" * 50,
    ]
    return "\n".join(lines)
