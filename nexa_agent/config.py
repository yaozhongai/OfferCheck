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
# 官方 API Provider（去 GMI 化，双 Provider）
# ==========================================================================
# 固定职责：
#   strong / fast → DeepSeek 官方
#   upgrade / vision → Moonshot 官方
# Provider 不再由一个全局 LLM_PROVIDER 环境变量切换，避免模型名被发送到错误 endpoint。

PROVIDER_CONFIG = {
    "deepseek": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    },
    "kimi": {
        "base_url": os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "api_key": os.environ.get("MOONSHOT_API_KEY", ""),
    },
}

def thinking_extra_body(
    model: str,
    enable_thinking: bool,
    provider: str | None = None,
) -> dict:
    """生成官方 API 的 thinking 参数。

    DeepSeek V4 与 Kimi K2.6 官方 OpenAI 兼容接口都使用
    ``thinking.type=enabled|disabled``。provider 显式传入时优先；旧调用方未传时
    根据模型归属推断。upgrade 的 disabled 策略由 MODEL_TIER/Gateway 强制执行。
    """
    resolved_provider = provider or get_provider_for_model(model)
    if resolved_provider in ("deepseek", "kimi"):
        return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
    return {}

# ==========================================================================
# 模型配置
# ==========================================================================

MODEL_TIER = {
    "strong": {
        **PROVIDER_CONFIG["deepseek"],
        "provider": "deepseek",
        "model": os.environ.get("STRONG_MODEL", "deepseek-v4-pro"),
        "thinking": "enabled",
        "description": "深度推理 — 首步规划、规则演化等需要强推理能力的任务",
    },
    "fast": {
        **PROVIDER_CONFIG["deepseek"],
        "provider": "deepseek",
        "model": os.environ.get("FAST_MODEL", "deepseek-v4-flash"),
        "thinking": "disabled",
        "description": "快速执行 — 后续步骤、反思生成、评估判断等成本敏感任务",
    },
    "upgrade": {
        **PROVIDER_CONFIG["kimi"],
        "provider": "kimi",
        "model": os.environ.get("UPGRADE_MODEL", "kimi-k2.6"),
        "thinking": "disabled",
        "description": "备援 tool-call — 连续 N 步未发 tool_calls 时自动切入",
    },
}

# 兼容仍读取 MODEL_CONFIG 的调用方：它代表默认 strong 路由，而不是全局 Provider。
MODEL_CONFIG = {
    **MODEL_TIER["strong"],
    "react_temperature": float(os.environ.get("REACT_TEMPERATURE", "0.0")),
}

MODEL_ROUTING = {
    "react_first":       "strong",   # 首步规划 — 需要深度思考
    "react_main":        "fast",     # 后续步骤 — 多步执行，成本敏感
    "reflection":        "fast",     # 反思生成 — 结构化输出，复杂度低
    "evaluator_llm":     "fast",     # LLM 评估 — 判断逻辑简单
    "lesson_extract":    "fast",     # 教训提取 — 归纳总结
    "tool_call_upgrade": "upgrade",  # 备援 — 动态升级触发时使用
    # （已删 evolver 路由：无对应实现，get_model_for_role("evolver") 从未被调用）
}

# 动态升级阈值：连续 N 步 LLM 未发 tool_calls（且无 Final Answer）后自动切 upgrade 层
DYNAMIC_UPGRADE_THRESHOLD = 2

# ==========================================================================
# 视觉模型（Moonshot 官方 Kimi K2.6）
# ==========================================================================
VISION_CONFIG = {
    "provider": "kimi",
    "kimi": {
        **PROVIDER_CONFIG["kimi"],
        "model": os.environ.get("VISION_MODEL", "kimi-k2.6"),
    },
}


# ==========================================================================
# ReAct 配置
# ==========================================================================

REACT_CONFIG = {
    "max_steps": int(os.environ.get("REACT_MAX_STEPS", "16")),
    "observation_max_chars": int(os.environ.get("REACT_OBS_MAX_CHARS", "2000")),
    # （已删 compression_after_steps：从未被引用；观察截断由 _truncate_observation 分档处理）
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
    "reflection_model": os.environ.get("REFLEXION_MODEL", "deepseek-v4-flash"),
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
    # Tavily 优先（额度够时质量最好，1000/月）→ Exa（1000/月免费，配 key 即用，机房 IP 也可）
    # → DDG（无 key 兜底，机房 IP 易限流）。未配 key 的 provider 经 is_available() 自动跳过。
    "provider_order": [
        p.strip()
        for p in os.environ.get("SEARCH_PROVIDER_ORDER", "tavily,exa,ddg").split(",")
        if p.strip()
    ],
    "max_results": int(os.environ.get("SEARCH_MAX_RESULTS", "5")),
    "snippet_max_chars": int(os.environ.get("SEARCH_SNIPPET_MAX_CHARS", "300")),
    "request_timeout": int(os.environ.get("SEARCH_TIMEOUT", "15")),
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
    # 仅增强摘要偏弱的 provider（DDG）；Tavily/Exa 自带优质正文，无需重复抓取
    "enrich_providers": [
        p.strip()
        for p in os.environ.get("SEARCH_ENRICH_PROVIDERS", "ddg").split(",")
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
              "evaluator_llm" | "lesson_extract" | "tool_call_upgrade"

    Returns:
        模型名称字符串
    """
    tier = MODEL_ROUTING.get(role, "strong")
    return MODEL_TIER[tier]["model"]


def get_model_config_for_role(role: str) -> dict:
    """返回角色对应的完整路由（model/provider/base_url/api_key/thinking）。"""
    tier = MODEL_ROUTING.get(role, "strong")
    return {**MODEL_TIER[tier], "tier": tier}


def get_provider_for_model(model: str) -> str:
    """根据当前 tier 模型名反查 provider；未知显式模型安全回落 DeepSeek。"""
    for cfg in MODEL_TIER.values():
        if cfg["model"] == model:
            return cfg["provider"]
    return "kimi" if (model or "").lower().startswith("kimi-") else "deepseek"


def resolve_model_config(
    *,
    role: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """解析一次调用应使用的完整模型路由。

    role 优先确定 tier；显式 model 与现有 tier 同名时继承该 tier 的 provider。
    对临时显式模型可同时传 provider，避免跨 Provider 猜测。
    """
    if role:
        cfg = get_model_config_for_role(role)
    else:
        matched_tier = next(
            (name for name, item in MODEL_TIER.items() if item["model"] == model),
            None,
        )
        cfg = (
            {**MODEL_TIER[matched_tier], "tier": matched_tier}
            if matched_tier
            else {**MODEL_TIER["strong"], "tier": None}
        )

    actual_provider = provider or (
        cfg["provider"] if model is None or cfg.get("model") == model
        else get_provider_for_model(model)
    )
    if actual_provider not in PROVIDER_CONFIG:
        raise ValueError(f"不支持的模型 provider: {actual_provider}")
    provider_cfg = PROVIDER_CONFIG[actual_provider]
    return {
        **cfg,
        **provider_cfg,
        "provider": actual_provider,
        "model": model or cfg["model"],
    }


def get_config_summary() -> str:
    """生成配置摘要（用于调试和日志）"""
    lines = [
        "=" * 50,
        "Reflexion + ReAct 配置",
        "=" * 50,
        "LLM Providers: DeepSeek 官方 + Moonshot 官方",
        f"模型层级: strong=deepseek/{MODEL_TIER['strong']['model']}, "
        f"fast=deepseek/{MODEL_TIER['fast']['model']}, "
        f"upgrade=kimi/{MODEL_TIER['upgrade']['model']}",
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
