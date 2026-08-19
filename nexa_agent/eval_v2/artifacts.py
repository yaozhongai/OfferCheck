"""Immutable manifest and atomic per-trial artifact storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from nexa_agent.eval_v2.schemas import RunManifest, TrialRecord, canonical_json


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ArtifactStore:
    def __init__(self, root: Path, run_id: str):
        self.run_dir = Path(root) / run_id
        self.trials_dir = self.run_dir / "trials"
        self.manifest_path = self.run_dir / "manifest.json"

    def initialize(self, manifest: RunManifest) -> None:
        payload = manifest.snapshot()
        payload["manifest_hash"] = manifest.manifest_hash
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("manifest_hash") != manifest.manifest_hash:
                raise ValueError(
                    f"Run {manifest.run_id!r} already has a different immutable manifest"
                )
            return
        _atomic_write(self.manifest_path, serialized + "\n")

    def trial_path(self, trial_id: str) -> Path:
        return self.trials_dir / f"{trial_id}.json"

    def has_trial(self, trial_id: str) -> bool:
        return self.trial_path(trial_id).exists()

    def load_trial(self, trial_id: str) -> Optional[dict[str, Any]]:
        path = self.trial_path(trial_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_trial(self, record: TrialRecord) -> Path:
        path = self.trial_path(record.trial_id)
        serialized = json.dumps(
            record.snapshot(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(existing) != canonical_json(record.snapshot()):
                raise ValueError(f"Trial artifact {record.trial_id!r} is immutable")
            return path
        _atomic_write(path, serialized)
        return path

    def list_trials(self) -> list[dict[str, Any]]:
        if not self.trials_dir.exists():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.trials_dir.glob("*.json"))
        ]
