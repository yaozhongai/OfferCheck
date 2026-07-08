"""nexa_agent 通用工具子包（无第三方依赖、可被引擎任意层 import）。

- json_extract: LLM JSON 响应的健壮抽取 + 截断修复（原在 verifier，抽出共享）
- llm_retry:    LLM 瞬时错误的分类 + 指数退避重试（原在 react_agent，抽出共享）
"""

from .json_extract import extract_json_block, repair_truncated_json
from .llm_retry import is_transient_llm_error, call_with_retry

__all__ = [
    "extract_json_block",
    "repair_truncated_json",
    "is_transient_llm_error",
    "call_with_retry",
]
