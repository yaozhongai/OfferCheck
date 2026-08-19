"""Load frozen JSONL task suites into the V2 schema."""

from __future__ import annotations

import json
from pathlib import Path

from nexa_agent.eval_v2.schemas import EvalTask
from nexa_agent.eval_v2.snapshot_replay import validate_fault_variant


def load_jsonl_suite(path: Path) -> tuple[EvalTask, ...]:
    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        payload = json.loads(stripped)
        task_id = str(payload["task_id"])
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id {task_id!r} at line {line_number}")
        seen_ids.add(task_id)
        metadata = dict(payload.get("metadata", {}))
        top_level_card = payload.get("eval_card")
        metadata_card = metadata.get("eval_card")
        if top_level_card is not None:
            if metadata_card is None:
                raise ValueError(
                    f"{task_id}: eval_card must be stored at metadata.eval_card; "
                    "run the suite migration before evaluation"
                )
            if top_level_card != metadata_card:
                raise ValueError(
                    f"{task_id}: conflicting top-level eval_card and "
                    "metadata.eval_card (split-brain suite)"
                )
            raise ValueError(
                f"{task_id}: duplicate top-level eval_card is forbidden; "
                "metadata.eval_card is the canonical location"
            )
        requires_frozen_card = task_id.startswith("nexa_v0_")
        if not isinstance(metadata_card, dict) and requires_frozen_card:
            raise ValueError(f"{task_id}: missing metadata.eval_card")
        if not requires_frozen_card:
            metadata_card = metadata_card or {}
        evidential = metadata_card.get("evidential_package")
        if not isinstance(evidential, dict) and requires_frozen_card:
            raise ValueError(
                f"{task_id}: missing metadata.eval_card.evidential_package"
            )
        evidential = evidential or {}
        for required in ("content_hash", "snapshot_at") if requires_frozen_card else ():
            if not str(evidential.get(required, "")).strip():
                raise ValueError(
                    f"{task_id}: empty metadata.eval_card.evidential_package."
                    f"{required}"
                )
        review = metadata.get("review", {})
        if review.get("requires_revalidation"):
            raise ValueError(
                f"{task_id}: replacement task requires independent revalidation"
            )
        # fault_variant 契约校验：非法规格在加载阶段直接失败（protocol-r2 §G）
        validate_fault_variant(task_id, metadata)
        tasks.append(
            EvalTask(
                task_id=task_id,
                prompt=str(payload["prompt"]),
                task_profile=str(payload["task_profile"]),
                grader=str(payload["grader"]),
                expected_answer=payload.get("expected_answer"),
                expected_verdict=payload.get("expected_verdict"),
                stage=payload.get("stage"),
                tags=tuple(payload.get("tags", [])),
                forbidden_actions=tuple(payload.get("forbidden_actions", [])),
                metadata=metadata,
            )
        )
    if not tasks:
        raise ValueError(f"Suite {path} contains no tasks")
    return tuple(tasks)
