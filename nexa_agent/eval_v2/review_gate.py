"""Audit and finalize two independent reviews for replacement tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_packet(packet: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for task in packet.get("tasks", []):
        task_id = str(task.get("task_id", "unknown"))
        snapshot = Path(task["snapshot_path"])
        if not snapshot.exists():
            issues.append(f"{task_id}: snapshot missing")
        elif _sha256(snapshot) != task.get("expected_snapshot_sha256"):
            issues.append(f"{task_id}: snapshot hash changed")
        reviews = task.get("reviews") or []
        reviewers = {
            str(item.get("reviewer_id", "")).strip()
            for item in reviews
            if str(item.get("reviewer_id", "")).strip()
        }
        if len(reviewers) < 2:
            issues.append(f"{task_id}: two distinct reviewers required")
        for index, review in enumerate(reviews, 1):
            prefix = f"{task_id}/review-{index}"
            if review.get("decision") not in {"pass", "fail"}:
                issues.append(f"{prefix}: decision must be pass or fail")
            checks = review.get("checks") or {}
            for key in (
                "question_unambiguous",
                "official_sources_support_answer",
                "snapshot_contains_required_facts",
                "rubric_matches_question",
                "no_suite_overlap",
            ):
                if not isinstance(checks.get(key), bool):
                    issues.append(f"{prefix}: missing boolean check {key}")
            if not str(review.get("evidence_note", "")).strip():
                issues.append(f"{prefix}: evidence_note required")
            if not str(review.get("reviewed_at", "")).strip():
                issues.append(f"{prefix}: reviewed_at required")
        if len(reviews) >= 2:
            decisions = [item.get("decision") for item in reviews]
            if decisions != ["pass", "pass"]:
                issues.append(f"{task_id}: both independent decisions must pass")
    return {
        "schema_version": "replacement-review-audit-r3",
        "pass": not issues,
        "issues": issues,
    }


def finalize(packet_path: Path, suites: list[Path]) -> None:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    report = audit_packet(packet)
    if not report["pass"]:
        raise ValueError("review packet does not pass: " + "; ".join(report["issues"]))
    reviewed = {item["task_id"]: item for item in packet["tasks"]}
    found: set[str] = set()
    for suite in suites:
        output: list[str] = []
        for line in suite.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                output.append(line)
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id in reviewed:
                found.add(task_id)
                metadata = row.setdefault("metadata", {})
                metadata["round1_review_status"] = "pass"
                metadata["round2_review_status"] = "pass"
                metadata["review"] = {
                    "requires_revalidation": False,
                    "schema_version": "replacement-review-r3",
                    "reviews": reviewed[task_id]["reviews"],
                    "packet_sha256": _sha256(packet_path),
                }
            output.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        suite.write_text("\n".join(output) + "\n", encoding="utf-8")
    missing = set(reviewed) - found
    if missing:
        raise ValueError(f"reviewed tasks not found in supplied suites: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--suites", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.finalize:
        if not args.suites:
            parser.error("--finalize requires --suites")
        finalize(args.packet, args.suites)
        return
    report = audit_packet(json.loads(args.packet.read_text(encoding="utf-8")))
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
