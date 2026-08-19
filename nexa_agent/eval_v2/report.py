"""Generate machine-readable and Markdown reports from immutable trial artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from nexa_agent.eval_v2.artifacts import ArtifactStore


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _mean(rows: list[dict[str, Any]], key) -> float:
    return statistics.mean(key(row) for row in rows) if rows else 0.0


def build_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = ArtifactStore(run_dir.parent, run_dir.name).list_trials()
    configs: dict[str, Any] = {}
    # Only completed task executions belong in capability metrics. Infrastructure
    # and budget-control outcomes are reported separately and never become failures.
    valid_rows = [row for row in rows if row["status"] == "completed"]

    for config_id in sorted({row["config_id"] for row in rows}):
        config_rows = [row for row in valid_rows if row["config_id"] == config_id]
        correct = sum(row["correct"] is True for row in config_rows)
        role_tokens: Counter[str] = Counter()
        failure_modes: Counter[str] = Counter()
        for row in config_rows:
            for role, usage in row.get("usage_breakdown", {}).get(
                "by_role", {}
            ).items():
                role_tokens[role] += int(usage.get("prompt_tokens", 0))
                role_tokens[role] += int(usage.get("completion_tokens", 0))
            for detail in row.get("raw_result", {}).get("trial_details", []):
                if detail.get("failure_mode"):
                    failure_modes[str(detail["failure_mode"])] += 1

        configs[config_id] = {
            "trials": len(config_rows),
            "correct": correct,
            "accuracy": correct / len(config_rows) if config_rows else 0.0,
            "accuracy_wilson_95": _wilson(correct, len(config_rows)),
            "avg_total_tokens": _mean(
                config_rows,
                lambda row: row["prompt_tokens"] + row["completion_tokens"],
            ),
            "avg_react_only_tokens": _mean(
                config_rows,
                lambda row: row.get("raw_result", {}).get(
                    "total_prompt_tokens", 0
                )
                + row.get("raw_result", {}).get("total_completion_tokens", 0),
            ),
            "avg_elapsed_seconds": _mean(
                config_rows, lambda row: row["elapsed_seconds"]
            ),
            "avg_steps": _mean(config_rows, lambda row: row["steps_used"]),
            "internal_success_rate": _mean(
                config_rows,
                lambda row: 1.0
                if row.get("raw_result", {}).get("success")
                else 0.0,
            ),
            "avg_llm_attempts": _mean(
                config_rows,
                lambda row: row.get("usage_breakdown", {}).get(
                    "llm_attempts", 0
                ),
            ),
            "avg_tool_calls": _mean(
                config_rows,
                lambda row: row.get("usage_breakdown", {}).get(
                    "tool_calls", 0
                ),
            ),
            "gts": sum(
                row.get("grade", {}).get("gts", row.get("correct")) is True
                for row in config_rows
            ),
            "role_tokens": dict(role_tokens),
            "failure_modes": dict(failure_modes),
        }

    by_pair: dict[tuple[str, int], dict[str, bool]] = {}
    for row in valid_rows:
        by_pair.setdefault(
            (row["task_id"], row["repeat_index"]), {}
        )[row["config_id"]] = bool(row["correct"])
    paired = Counter()
    config_ids = sorted(configs)
    if len(config_ids) == 2:
        left, right = config_ids
        for outcomes in by_pair.values():
            if left not in outcomes or right not in outcomes:
                continue
            if outcomes[left] == outcomes[right]:
                paired["tie"] += 1
            elif outcomes[left]:
                paired[f"{left}_only"] += 1
            else:
                paired[f"{right}_only"] += 1

    return {
        "run_id": manifest["run_id"],
        "protocol_version": manifest["protocol_version"],
        "manifest_hash": manifest["manifest_hash"],
        "source_snapshot_sha256": manifest.get("source_snapshot_sha256"),
        "planned_trials": len(manifest["tasks"])
        * len(manifest["configs"])
        * manifest["repeats"],
        "recorded_trials": len(rows),
        "infra_errors": sum(row["status"] == "infra_error" for row in rows),
        "budget_exceeded": sum(row["status"] == "budget_exceeded" for row in rows),
        "configs": configs,
        "paired_outcomes": dict(paired),
        "trial_index": [
            {
                "trial_id": row["trial_id"],
                "task_id": row["task_id"],
                "config_id": row["config_id"],
                "repeat_index": row["repeat_index"],
                "status": row["status"],
                "correct": row["correct"],
                "tokens": row["prompt_tokens"] + row["completion_tokens"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for row in rows
        ],
        "interpretation_limit": (
            "Pilot validates protocol and surfaces failure modes; its sample is "
            "too small for resume-grade comparative claims."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Nexa Agent Pilot Report — {report['run_id']}",
        "",
        f"- Protocol: `{report['protocol_version']}`",
        f"- Manifest: `{report['manifest_hash']}`",
        f"- Source snapshot: `{report.get('source_snapshot_sha256')}`",
        f"- Infrastructure errors: {report['infra_errors']}",
        f"- Budget violations: {report['budget_exceeded']}",
        "",
        "## Configuration summary",
        "",
        "| Config | GTS/Trials | 95% Wilson CI | Avg tokens (all LLM roles) | Avg latency | Avg steps | Avg LLM attempts | Avg tool calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_id, item in report["configs"].items():
        low, high = item["accuracy_wilson_95"]
        lines.append(
            f"| {config_id} | {item['gts']}/{item['trials']} "
            f"({item['accuracy']:.1%}) | {low:.1%}–{high:.1%} | "
            f"{item['avg_total_tokens']:.0f} | "
            f"{item['avg_elapsed_seconds']:.1f}s | {item['avg_steps']:.1f} | "
            f"{item['avg_llm_attempts']:.1f} | {item['avg_tool_calls']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Paired outcomes",
            "",
            json.dumps(report["paired_outcomes"], ensure_ascii=False, sort_keys=True),
            "",
            "## Interpretation boundary",
            "",
            report["interpretation_limit"],
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path) -> tuple[Path, Path]:
    report = build_report(run_dir)
    json_path = Path(run_dir) / "report.json"
    md_path = Path(run_dir) / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    json_path, md_path = write_report(Path(args.run_dir))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
