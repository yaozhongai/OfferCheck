"""Independent-trial experiment runner for Nexa Agent evaluations."""

from __future__ import annotations

import dataclasses
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from nexa_agent.eval_v2.artifacts import ArtifactStore
from nexa_agent.eval_v2.graders import grade_trial
from nexa_agent.eval_v2.schemas import (
    EvalTask,
    ExperimentConfig,
    RunManifest,
    TrialRecord,
    make_trial_id,
    utc_now,
)
from nexa_agent.eval_v2.snapshot_replay import (
    FaultInjector,
    SnapshotStore,
    drain_injection_log,
    install_interceptor,
    uninstall_interceptor,
)
from nexa_agent.harness_config import FULL_HARNESS, MINIMAL_REACT
from nexa_agent.config import model_tier_scope
from nexa_agent.llm_gateway import BudgetExceededError, usage_scope
from nexa_agent.reflexion_agent import ReflexionReActAgent


AgentFactory = Callable[[ExperimentConfig], Any]


def build_schedule(
    manifest: RunManifest,
) -> list[tuple[EvalTask, ExperimentConfig, int]]:
    """Pairwise schedule：每个 (task, repeat) 聚对，pair 内配置顺序随机。

    protocol-r2 §E：pair 内两配置连续执行——第一个 record，第二个 replay，
    保证两配置看到完全相同的封闭世界（同 run 内），消除网络漂移混淆。
    pair 间随机打散、pair 内配置随机——避免「record 总是 minimal」的系统偏倚。
    总预算只在完整 pair 之间检查，不产生残缺 pair。
    """

    pairs: list[tuple[EvalTask, int, list[ExperimentConfig]]] = []
    for task in manifest.tasks:
        for repeat_index in range(1, manifest.repeats + 1):
            pair_cfgs = list(manifest.configs)
            pair_seed = f"{manifest.seed}:{task.task_id}:r{repeat_index}"
            random.Random(pair_seed).shuffle(pair_cfgs)
            pairs.append((task, repeat_index, pair_cfgs))
    random.Random(manifest.seed).shuffle(pairs)
    return [
        (task, cfg, repeat_index)
        for (task, repeat_index, cfgs) in pairs
        for cfg in cfgs
    ]


def default_agent_factory(config: ExperimentConfig) -> ReflexionReActAgent:
    runtime_configs = {
        "full": FULL_HARNESS,
        "minimal": MINIMAL_REACT,
    }
    try:
        runtime_config = runtime_configs[config.runtime_profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown runtime profile {config.runtime_profile!r}"
        ) from exc
    return ReflexionReActAgent(
        max_trials=(
            config.max_internal_trials if config.reflexion_enabled else 1
        ),
        evaluator_mode=config.evaluator_mode,
        max_steps=config.max_steps,
        runtime_config=runtime_config,
        excluded_tools=frozenset(config.excluded_tools),
        enable_curation=config.enable_curation,
    )


def _plain_result(result: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Agent returned unsupported result type: {type(result).__name__}")


class ExperimentRunner:
    def __init__(
        self,
        output_root: Path,
        agent_factory: AgentFactory = default_agent_factory,
    ):
        self.output_root = Path(output_root)
        self.agent_factory = agent_factory

    def run(self, manifest: RunManifest) -> dict[str, int]:
        store = ArtifactStore(self.output_root, manifest.run_id)
        store.initialize(manifest)
        summary = {
            "planned": 0,
            "completed": 0,
            "skipped": 0,
            "infra_error": 0,
            "budget_exceeded": 0,
        }
        run_started = time.monotonic()
        total_tokens = sum(
            int(item.get("prompt_tokens", 0)) + int(item.get("completion_tokens", 0))
            for item in store.list_trials()
        )

        # Initialize snapshot store for tool-call recording / replay
        snapshot_store = SnapshotStore(
            self.output_root / manifest.run_id / "tool_snapshots"
        )
        snapshot_store.load()  # 断点续跑：恢复已持久化的快照
        # Track which (task, repeat) pairs have been recorded for fair replay
        existing_trials = store.list_trials()
        _recorded_pairs: set[tuple[str, int]] = {
            (str(row["task_id"]), int(row["repeat_index"]))
            for row in existing_trials
            if (row.get("snapshot_audit") or {}).get("mode") == "record"
        }

        for task, config, repeat_index in build_schedule(manifest):
            summary["planned"] += 1
            trial_id = make_trial_id(
                manifest.run_id, task.task_id, config.config_id, repeat_index
            )
            if store.has_trial(trial_id):
                summary["skipped"] += 1
                continue

            remaining_total_tokens = None
            if manifest.budgets.max_total_tokens is not None:
                remaining_total_tokens = (
                    manifest.budgets.max_total_tokens - total_tokens
                )
                if remaining_total_tokens <= 0:
                    return summary
            remaining_total_wall = None
            if manifest.budgets.max_total_wall_seconds is not None:
                remaining_total_wall = (
                    manifest.budgets.max_total_wall_seconds
                    - (time.monotonic() - run_started)
                )
                if remaining_total_wall <= 0:
                    return summary

            # Main runs consume a pre-recorded task corpus: both configs replay
            # the same world. Paired record/replay remains available for corpus
            # preparation and backwards-compatible pilots only.
            pair_key = (task.task_id, repeat_index)
            if manifest.snapshot_policy == "pre_recorded_replay":
                if not snapshot_store.has_world(*pair_key):
                    raise RuntimeError(
                        f"missing pre-recorded snapshot world for {pair_key}; "
                        "main evaluation must not record from either compared config"
                    )
                snap_mode = "replay"
            elif pair_key not in _recorded_pairs:
                snap_mode = "record"
                _recorded_pairs.add(pair_key)
            else:
                snap_mode = "replay"

            record = self._run_one(
                manifest,
                task,
                config,
                repeat_index,
                trial_id,
                snapshot_store=snapshot_store,
                snapshot_mode=snap_mode,
                remaining_total_tokens=remaining_total_tokens,
                remaining_total_wall=remaining_total_wall,
            )
            store.write_trial(record)
            if record.status == "infra_error":
                summary["infra_error"] += 1
            elif record.status == "budget_exceeded":
                summary["budget_exceeded"] += 1
            else:
                summary["completed"] += 1
            total_tokens += record.prompt_tokens + record.completion_tokens
        snapshot_store.save()
        return summary

    def _run_one(
        self,
        manifest: RunManifest,
        task: EvalTask,
        config: ExperimentConfig,
        repeat_index: int,
        trial_id: str,
        *,
        snapshot_store: SnapshotStore,
        snapshot_mode: str = "record",
        remaining_total_tokens: Optional[int] = None,
        remaining_total_wall: Optional[float] = None,
    ) -> TrialRecord:
        started_at = utc_now()
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        usage = None
        stats_before = snapshot_store.stats()

        # Set up fault injector if task has a fault variant
        fault_injector = None
        if task.metadata.get("fault_variant"):
            fault_injector = FaultInjector(task.metadata)

        # Install tool interceptor for this trial
        install_interceptor(
            task_id=task.task_id,
            snapshot_store=snapshot_store,
            fault_injector=fault_injector,
            mode=snapshot_mode,
            repeat_index=repeat_index,
        )

        try:
            token_caps = [
                value
                for value in (
                    manifest.budgets.max_tokens_per_trial,
                    remaining_total_tokens,
                )
                if value is not None
            ]
            wall_caps = [
                value
                for value in (
                    manifest.budgets.max_wall_seconds_per_trial,
                    remaining_total_wall,
                )
                if value is not None
            ]
            with usage_scope(
                max_tokens=min(token_caps) if token_caps else None,
                max_wall_seconds=min(wall_caps) if wall_caps else None,
                max_llm_attempts=manifest.budgets.max_llm_attempts_per_trial,
                max_tool_calls=manifest.budgets.max_tool_calls_per_trial,
            ) as usage:
                with model_tier_scope(config.model_tier_override):
                    # Evaluator/Verifier resolve and retain their model during
                    # construction, so factory creation must be inside the same
                    # fixed-tier scope as execution.
                    agent = self.agent_factory(config)
                    result = agent.execute(
                        task.prompt,
                        verbose=False,
                        stage=task.stage,
                        task_profile=task.task_profile,
                        on_event=events.append,
                        # 评测任务元数据：fault_variant + forbidden_actions，
                        # 供 react_loop 结构化安全事件检测（protocol-r2 §E）
                        task_metadata={
                            "fault_variant": task.metadata.get("fault_variant"),
                            "forbidden_actions": list(task.forbidden_actions),
                        },
                    )
            raw = _plain_result(result)
            usage_snapshot = usage.snapshot()
            answer = str(raw.get("answer", ""))
            if usage_snapshot.get("budget_exceeded"):
                return TrialRecord(
                    trial_id=trial_id,
                    run_id=manifest.run_id,
                    task_id=task.task_id,
                    config_id=config.config_id,
                    repeat_index=repeat_index,
                    status="budget_exceeded",
                    started_at=started_at,
                    finished_at=utc_now(),
                    elapsed_seconds=round(time.monotonic() - started, 6),
                    answer=answer,
                    prompt_tokens=int(usage_snapshot.get("prompt_tokens", 0)),
                    completion_tokens=int(
                        usage_snapshot.get("completion_tokens", 0)
                    ),
                    usage_breakdown=usage_snapshot,
                    events=events,
                    raw_result=raw,
                    budget_violations=[
                        str(usage_snapshot["budget_exceeded"])
                    ],
                    snapshot_audit=self._snapshot_audit(
                        snapshot_store, stats_before, snapshot_mode
                    ),
                )
            returned_error = self._returned_infra_error(raw)
            if returned_error is not None:
                elapsed = time.monotonic() - started
                return TrialRecord(
                    trial_id=trial_id,
                    run_id=manifest.run_id,
                    task_id=task.task_id,
                    config_id=config.config_id,
                    repeat_index=repeat_index,
                    status="infra_error",
                    started_at=started_at,
                    finished_at=utc_now(),
                    elapsed_seconds=round(elapsed, 6),
                    answer=answer,
                    prompt_tokens=(
                        usage.prompt_tokens
                        if usage_snapshot["calls"]
                        else int(raw.get("total_prompt_tokens", 0))
                    ),
                    completion_tokens=(
                        usage.completion_tokens
                        if usage_snapshot["calls"]
                        else int(raw.get("total_completion_tokens", 0))
                    ),
                    usage_breakdown=usage_snapshot,
                    internal_trials_used=int(raw.get("trials_used", 0)),
                    events=events,
                    raw_result=raw,
                    infra_error=returned_error,
                    snapshot_audit=self._snapshot_audit(
                        snapshot_store, stats_before, snapshot_mode
                    ),
                )
            grade = grade_trial(task, answer, raw)
            details = raw.get("trial_details", [])
            evidence_records = [
                record
                for detail in details
                for record in detail.get("evidence_records", [])
            ]
            claim_bindings = [
                binding
                for detail in details
                for binding in detail.get("claim_bindings", [])
            ]
            elapsed = time.monotonic() - started
            prompt_tokens = (
                usage.prompt_tokens
                if usage_snapshot["calls"]
                else int(raw.get("total_prompt_tokens", 0))
            )
            completion_tokens = (
                usage.completion_tokens
                if usage_snapshot["calls"]
                else int(raw.get("total_completion_tokens", 0))
            )
            violations = self._budget_violations(
                manifest, elapsed, prompt_tokens + completion_tokens
            )
            return TrialRecord(
                trial_id=trial_id,
                run_id=manifest.run_id,
                task_id=task.task_id,
                config_id=config.config_id,
                repeat_index=repeat_index,
                status="budget_exceeded" if violations else "completed",
                started_at=started_at,
                finished_at=utc_now(),
                elapsed_seconds=round(elapsed, 6),
                answer=answer,
                correct=bool(grade["correct"]),
                grade=grade,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_breakdown=usage_snapshot,
                internal_trials_used=int(raw.get("trials_used", 0)),
                steps_used=sum(int(d.get("steps_used", 0)) for d in details),
                terminated_reasons=[
                    str(d.get("terminated_reason", "")) for d in details
                ],
                evidence_records=evidence_records,
                claim_bindings=claim_bindings,
                events=events,
                raw_result=raw,
                budget_violations=violations,
                snapshot_audit=self._snapshot_audit(
                    snapshot_store, stats_before, snapshot_mode
                ),
            )
        except BudgetExceededError as exc:
            elapsed = time.monotonic() - started
            usage_snapshot = usage.snapshot() if usage is not None else {}
            return TrialRecord(
                trial_id=trial_id,
                run_id=manifest.run_id,
                task_id=task.task_id,
                config_id=config.config_id,
                repeat_index=repeat_index,
                status="budget_exceeded",
                started_at=started_at,
                finished_at=utc_now(),
                elapsed_seconds=round(elapsed, 6),
                prompt_tokens=int(usage_snapshot.get("prompt_tokens", 0)),
                completion_tokens=int(
                    usage_snapshot.get("completion_tokens", 0)
                ),
                usage_breakdown=usage_snapshot,
                events=events,
                budget_violations=[str(exc)],
                snapshot_audit=self._snapshot_audit(
                    snapshot_store, stats_before, snapshot_mode
                ),
            )
        except Exception as exc:  # infrastructure failures are not task failures
            elapsed = time.monotonic() - started
            return TrialRecord(
                trial_id=trial_id,
                run_id=manifest.run_id,
                task_id=task.task_id,
                config_id=config.config_id,
                repeat_index=repeat_index,
                status="infra_error",
                started_at=started_at,
                finished_at=utc_now(),
                elapsed_seconds=round(elapsed, 6),
                events=events,
                infra_error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                snapshot_audit=self._snapshot_audit(
                    snapshot_store, stats_before, snapshot_mode
                ),
            )
        finally:
            # 收集本 trial 的故障注入事件（trigger-rate 验收依据），合并进 events
            injection_events = drain_injection_log()
            events.extend(injection_events)
            uninstall_interceptor()

    @staticmethod
    def _snapshot_audit(
        store: SnapshotStore, before: dict[str, int], mode: str
    ) -> dict[str, Any]:
        after = store.stats()
        return {
            "mode": mode,
            "closed_world": mode == "replay",
            "network_allowed": mode != "replay",
            "stats": {
                key: after.get(key, 0) - before.get(key, 0)
                for key in sorted(set(before) | set(after))
            },
        }

    @staticmethod
    def _budget_violations(
        manifest: RunManifest, elapsed: float, total_tokens: int
    ) -> list[str]:
        violations: list[str] = []
        if (
            manifest.budgets.max_tokens_per_trial is not None
            and total_tokens > manifest.budgets.max_tokens_per_trial
        ):
            violations.append("max_tokens_per_trial")
        if (
            manifest.budgets.max_wall_seconds_per_trial is not None
            and elapsed > manifest.budgets.max_wall_seconds_per_trial
        ):
            violations.append("max_wall_seconds_per_trial")
        return violations

    @staticmethod
    def _returned_infra_error(raw: dict[str, Any]) -> Optional[dict[str, str]]:
        details = raw.get("trial_details", [])
        reasons = {
            str(detail.get("terminated_reason", ""))
            for detail in details
            if detail.get("terminated_reason")
        }
        if reasons and reasons.issubset({"llm_error"}):
            return {
                "type": "AgentInfrastructureError",
                "message": "all internal trials terminated with llm_error",
            }
        return None
