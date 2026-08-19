"""Recompute deterministic GTS from immutable raw trials without mutating them."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from nexa_agent.eval_v2.graders import grade_trial
from nexa_agent.eval_v2.schemas import EvalTask
from nexa_agent.react_agent import build_claim_bindings


def _task_from_snapshot(item: dict) -> EvalTask:
    payload = dict(item)
    payload["tags"] = tuple(payload.get("tags", ()))
    payload["forbidden_actions"] = tuple(payload.get("forbidden_actions", ()))
    return EvalTask(**payload)


def regrade_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks = {
        task.task_id: task
        for task in (_task_from_snapshot(item) for item in manifest["tasks"])
    }
    grader_files = [
        Path(__file__).with_name("graders.py"),
        Path(__file__).parents[1] / "react_agent.py",
    ]
    digest = hashlib.sha256()
    for path in grader_files:
        digest.update(path.read_bytes())

    records = []
    for path in sorted((run_dir / "trials").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] != "completed":
            continue
        raw = copy.deepcopy(row["raw_result"])
        rebuilt = []
        for detail in raw.get("trial_details", []):
            old = detail.get("claim_bindings", [])
            bindings = build_claim_bindings(
                [item.get("claim", "") for item in old],
                [item.get("declared_sources", []) for item in old],
                detail.get("evidence_records", []),
            )
            detail["claim_bindings"] = bindings
            rebuilt.extend(bindings)
        cached_entailment = (row.get("grade") or {}).get("entailment")
        if not isinstance(cached_entailment, dict):
            raise ValueError(
                f"{row['trial_id']}: missing cached entailment verdict; "
                "offline regrade refuses to call a live judge"
            )
        grade = grade_trial(
            tasks[row["task_id"]],
            row["answer"],
            raw,
            cached_claim_entailment=cached_entailment,
        )
        records.append(
            {
                "trial_id": row["trial_id"],
                "task_id": row["task_id"],
                "config_id": row["config_id"],
                "outcome_correct": grade["outcome_correct"],
                "gts": grade["gts"],
                "grounding": grade["grounding"],
                "claim_entailment_source": "cached_immutable_trial_grade",
                "bindings": len(rebuilt),
                "supported_bindings": sum(
                    item.get("supported_by_visited_source", False)
                    for item in rebuilt
                ),
                "invalid_claim_scope": sum(
                    not item.get("claim_scope_valid", True) for item in rebuilt
                ),
            }
        )
    return {
        "run_id": manifest["run_id"],
        "original_manifest_hash": manifest["manifest_hash"],
        "grader_snapshot_sha256": digest.hexdigest(),
        "records": records,
        "gts_by_config": {
            config_id: {
                "gts": sum(
                    item["gts"]
                    for item in records
                    if item["config_id"] == config_id
                ),
                "trials": sum(
                    item["config_id"] == config_id for item in records
                ),
            }
            for config_id in sorted({item["config_id"] for item in records})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="regrade_current.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = regrade_run(run_dir)
    output = run_dir / args.output
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
