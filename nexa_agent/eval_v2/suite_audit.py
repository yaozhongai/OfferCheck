"""Machine-readable pre-main audit for frozen resume evaluation suites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_suite(path: Path, snapshot_dir: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    tasks = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tasks += 1
        row = json.loads(line)
        task_id = str(row.get("task_id", f"line-{line_number}"))
        if "eval_card" in row:
            issues.append({
                "task_id": task_id,
                "code": "duplicate_top_level_eval_card",
                "detail": "eval_card must exist only under metadata",
            })
        metadata = row.get("metadata") or {}
        card = metadata.get("eval_card")
        if not isinstance(card, dict):
            issues.append({
                "task_id": task_id,
                "code": "missing_runtime_eval_card",
                "detail": "metadata.eval_card is missing",
            })
            continue
        package = card.get("evidential_package") or {}
        expected_hash = str(package.get("content_hash", "")).strip()
        snapshot_at = str(package.get("snapshot_at", "")).strip()
        if not expected_hash:
            issues.append({
                "task_id": task_id,
                "code": "missing_content_hash",
                "detail": "runtime evidential package has no content hash",
            })
        if not snapshot_at:
            issues.append({
                "task_id": task_id,
                "code": "missing_snapshot_at",
                "detail": "runtime evidential package has no snapshot timestamp",
            })
        candidates = sorted(snapshot_dir.glob(f"{task_id}*"))
        if not candidates:
            issues.append({
                "task_id": task_id,
                "code": "snapshot_missing",
                "detail": "no evidence snapshot matches task id",
            })
        elif expected_hash and expected_hash not in {_sha256(item) for item in candidates}:
            issues.append({
                "task_id": task_id,
                "code": "snapshot_hash_mismatch",
                "detail": "content hash does not match any task snapshot",
            })
        if (metadata.get("review") or {}).get("requires_revalidation"):
            issues.append({
                "task_id": task_id,
                "code": "replacement_review_stale",
                "detail": "replacement task has not completed two new independent reviews",
            })
    return {
        "suite": str(path),
        "suite_sha256": _sha256(path),
        "tasks": tasks,
        "pass": not issues,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [audit_suite(path, args.snapshot_dir) for path in args.suites]
    report = {
        "schema_version": "step-c-audit-r3",
        "pass": all(item["pass"] for item in results),
        "suites": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
