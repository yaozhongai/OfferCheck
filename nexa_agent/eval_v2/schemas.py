"""Versioned schemas for independent, reconstructable Nexa evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


PROTOCOL_VERSION = "nexa-eval-v2.0"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    task_profile: str
    grader: str
    expected_answer: Optional[str] = None
    expected_verdict: Optional[str] = None
    stage: Optional[str] = None
    tags: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetLimits:
    max_steps_per_internal_trial: int = 16
    max_internal_trials: int = 3
    max_tokens_per_trial: Optional[int] = None
    max_wall_seconds_per_trial: Optional[float] = None
    max_llm_attempts_per_trial: Optional[int] = 64
    max_tool_calls_per_trial: Optional[int] = 64
    max_total_tokens: Optional[int] = None
    max_total_wall_seconds: Optional[float] = None


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    runtime_profile: str
    max_internal_trials: int
    evaluator_mode: str = "hybrid"
    max_steps: int = 16
    excluded_tools: tuple[str, ...] = ()
    model_tier_override: Optional[str] = None
    enable_curation: bool = False
    reflexion_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    suite_id: str
    seed: int
    repeats: int
    code_commit: str
    dirty_worktree: bool
    source_snapshot_sha256: str
    model_snapshot: dict[str, Any]
    budgets: BudgetLimits
    configs: tuple[ExperimentConfig, ...]
    tasks: tuple[EvalTask, ...]
    created_at: str = field(default_factory=utc_now)
    protocol_version: str = PROTOCOL_VERSION
    snapshot_policy: str = "paired_record_replay"

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_hash(self) -> str:
        payload = self.snapshot()
        payload.pop("created_at", None)
        return stable_hash(payload, length=64)


@dataclass
class TrialRecord:
    trial_id: str
    run_id: str
    task_id: str
    config_id: str
    repeat_index: int
    status: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    answer: str = ""
    correct: Optional[bool] = None
    grade: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_breakdown: dict[str, Any] = field(default_factory=dict)
    internal_trials_used: int = 0
    steps_used: int = 0
    terminated_reasons: list[str] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    claim_bindings: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)
    infra_error: Optional[dict[str, str]] = None
    budget_violations: list[str] = field(default_factory=list)
    snapshot_audit: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def make_trial_id(
    run_id: str,
    task_id: str,
    config_id: str,
    repeat_index: int,
) -> str:
    digest = stable_hash(
        {
            "protocol": PROTOCOL_VERSION,
            "run_id": run_id,
            "task_id": task_id,
            "config_id": config_id,
            "repeat_index": repeat_index,
        }
    )
    return f"{task_id}__{config_id}__r{repeat_index:02d}__{digest}"
