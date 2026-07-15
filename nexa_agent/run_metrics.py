"""Absolute, user-facing metrics for one investigation run.

The engine already records latency and token usage.  This module turns those
raw counters into one stable object that can be returned by both HTTP paths,
persisted with the trace, and rendered by the web UI.

Cost is deliberately an *estimate*.  A run can route across several models,
while the current usage contract is aggregated.  Deployments may configure a
blended input/output price via environment variables; when no price is
configured we return ``None`` rather than inventing a number.
"""

from __future__ import annotations

import os
import re
from typing import Any


_URL_RE = re.compile(r"https?://[^\s\]\[(){}<>\"']+", re.IGNORECASE)
_SOURCE_RE = re.compile(r"\[Source\]\s*([^\n]+)", re.IGNORECASE)


def _configured_rate(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost from configurable blended USD-per-million-token rates."""
    input_rate = _configured_rate("OFFERCHECK_INPUT_USD_PER_1M_TOKENS")
    output_rate = _configured_rate("OFFERCHECK_OUTPUT_USD_PER_1M_TOKENS")
    if input_rate is None and output_rate is None:
        return None
    value = (
        prompt_tokens * (input_rate or 0.0)
        + completion_tokens * (output_rate or 0.0)
    ) / 1_000_000
    return round(value, 6)


def _source_count(answer: str) -> int:
    """Count unique URL/tool source references in the rendered final answer."""
    refs: set[str] = set()
    for url in _URL_RE.findall(answer or ""):
        refs.add(url.rstrip(".,;:!?，。；：！？"))
    for source in _SOURCE_RE.findall(answer or ""):
        cleaned = source.strip().rstrip(".,;:!?，。；：！？")
        if cleaned:
            refs.add(cleaned)
    return len(refs)


def build_run_metrics(result: Any, latency_ms: float) -> dict[str, Any]:
    """Build the single absolute-metrics contract for API, trace, and UI."""
    prompt_tokens = int(getattr(result, "total_prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(result, "total_completion_tokens", 0) or 0)
    details = getattr(result, "trial_details", None) or []
    steps = sum(int(d.get("steps_used") or 0) for d in details if isinstance(d, dict))
    cost = estimate_cost_usd(prompt_tokens, completion_tokens)
    return {
        "success": bool(getattr(result, "success", False)),
        "duration_ms": round(float(latency_ms)),
        "duration_seconds": round(float(latency_ms) / 1000, 1),
        "trials": int(getattr(result, "trials_used", 0) or 0),
        "steps": steps,
        "sources": _source_count(str(getattr(result, "answer", "") or "")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": cost,
        "cost_configured": cost is not None,
        "cost_note": "configured blended rate; auxiliary evaluator/verifier calls excluded",
    }
