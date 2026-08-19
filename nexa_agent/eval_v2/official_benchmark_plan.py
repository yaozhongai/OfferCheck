"""Validate the preregistered low-cost official benchmark pilot.

This module intentionally does not download benchmarks or execute paid model
calls. It validates the experiment contract before any formal run is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PLAN_PATH = (
    Path(__file__).parents[1]
    / "results"
    / "official_benchmark_pilot_v1"
    / "preregistration.json"
)


class PlanValidationError(ValueError):
    """Raised when the preregistration contains an invalid experiment design."""


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_plan(plan: dict[str, Any], *, require_frozen: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    benchmarks = plan.get("benchmarks", [])
    if {item.get("id") for item in benchmarks} != {
        "agentdojo-workspace",
        "bfcl-v4-targeted",
    }:
        errors.append("The plan must contain exactly the AgentDojo and BFCL pilots.")

    computed_trials = 0
    computed_tasks = 0
    for benchmark in benchmarks:
        task_count = benchmark.get("task_count", 0)
        harness_count = benchmark.get("harness_count", 0)
        scenario_count = benchmark.get("scenario_count", 0)
        expected = task_count * harness_count * scenario_count
        computed_trials += expected
        computed_tasks += task_count

        if benchmark.get("planned_trials") != expected:
            errors.append(
                f"{benchmark.get('id')}: planned_trials must equal "
                "task_count × harness_count × scenario_count."
            )

        selection = benchmark.get("selection", {})
        frozen_ids = selection.get("frozen_task_ids", [])
        if frozen_ids and len(set(frozen_ids)) != len(frozen_ids):
            errors.append(f"{benchmark.get('id')}: frozen task ids contain duplicates.")
        if frozen_ids and len(frozen_ids) != task_count:
            errors.append(
                f"{benchmark.get('id')}: frozen task id count must equal task_count."
            )
        if not frozen_ids:
            warnings.append(f"{benchmark.get('id')}: task ids are not frozen yet.")

        if not benchmark.get("version_pin"):
            warnings.append(f"{benchmark.get('id')}: benchmark version is not pinned yet.")

    totals = plan.get("totals", {})
    if computed_trials != 56 or totals.get("formal_trials") != computed_trials:
        errors.append("The formal design must contain exactly 56 trials.")
    if computed_tasks != 20:
        errors.append("The formal design must contain exactly 20 unique benchmark tasks.")

    run_config = plan.get("shared_run_config", {})
    token_cap = run_config.get("max_model_tokens_per_trial")
    if token_cap != 30_000:
        errors.append("The per-trial token cap must remain 30,000 (revision 2 decision).")

    overrides = run_config.get("budget_overrides", {})
    multi_turn_cap = overrides.get(
        "bfcl_multi_turn_max_model_tokens_per_trial", token_cap
    )
    multi_turn_categories = set(overrides.get("bfcl_multi_turn_categories", []))
    computed_max_tokens = 0
    for benchmark in benchmarks:
        multiplier = benchmark.get("harness_count", 0) * benchmark.get(
            "scenario_count", 0
        )
        frozen_ids = benchmark.get("selection", {}).get("frozen_task_ids", [])
        for task_id in frozen_ids:
            category = task_id.rsplit("_", 1)[0]
            cap = (
                multi_turn_cap
                if category in multi_turn_categories
                else token_cap
            )
            computed_max_tokens += cap * multiplier
    if not computed_max_tokens:
        computed_max_tokens = computed_trials * token_cap
    if totals.get("maximum_formal_model_tokens") != computed_max_tokens:
        errors.append("maximum_formal_model_tokens is inconsistent with the trial plan.")

    if run_config.get("temperature") != 0:
        errors.append("Temperature must remain 0 for paired formal evaluation.")

    if not run_config.get("resolved_provider") or not run_config.get("resolved_model_id"):
        warnings.append("The exact provider and model id are not resolved yet.")

    if plan.get("status") not in {"smoke_passed", "formal_running", "complete"}:
        warnings.append("The 20 unscored smoke trials have not passed yet.")

    if require_frozen and warnings:
        errors.extend(f"Formal-run gate: {warning}" for warning in warnings)

    if errors:
        raise PlanValidationError("\n".join(errors))

    return {
        "valid": True,
        "formal_run_ready": not warnings,
        "formal_trials": computed_trials,
        "unique_tasks": computed_tasks,
        "maximum_formal_model_tokens": computed_max_tokens,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Fail until versions, task ids, and exact model identity are frozen.",
    )
    args = parser.parse_args()

    try:
        result = validate_plan(load_plan(args.plan), require_frozen=args.require_frozen)
    except PlanValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
