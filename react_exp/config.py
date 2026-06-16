"""
Reflexion + ReAct 统一配置管理

所有超参数集中管理，可通过环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

# ==========================================================================
# 模型配置
# ==========================================================================

MODEL_CONFIG = {
    "model": os.environ.get("LLM_MODEL", "deepseek-v4-pro"),
    "base_url": os.environ.get(
        "DEEPSEEK_BASE_URL",
        os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com"),
    ),
    "api_key": os.environ.get(
        "DEEPSEEK_API_KEY",
        os.environ.get("KIMI_API_KEY", ""),
    ),
    "react_temperature": 0.0,
    "reflection_temperature": 0.3,
}

# ==========================================================================
# 模型分层（任务感知的动态模型选择）
# ==========================================================================

MODEL_TIER = {
    "strong": {
        "model": os.environ.get("STRONG_MODEL", "deepseek-v4-pro"),
        "description": "深度推理 — 首步规划、规则演化等需要强推理能力的任务",
    },
    "fast": {
        "model": os.environ.get("FAST_MODEL", "deepseek-v4-flash"),
        "description": "快速执行 — 后续步骤、反思生成、评估判断等成本敏感任务",
    },
}

MODEL_ROUTING = {
    "react_first":   "strong",   # 首步规划 — 需要深度思考
    "react_main":    "fast",     # 后续步骤 — 多步执行，成本敏感
    "reflection":    "fast",     # 反思生成 — 结构化输出，复杂度低
    "evaluator_llm": "fast",     # LLM 评估 — 判断逻辑简单
    "lesson_extract": "fast",    # 教训提取 — 归纳总结
    "evolver":       "strong",   # 规则演化 — 低频高要求
}

# 动态升级阈值：连续 N 次 Action 解析失败后自动升级为强模型
DYNAMIC_UPGRADE_THRESHOLD = 2

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
