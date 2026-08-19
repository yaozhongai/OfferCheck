"""Build a configuration-independent closed-world corpus before comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from nexa_agent.eval_v2.snapshot_replay import _normalize_url_for_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_suite(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build(
    suite: Path,
    task_ids: set[str],
    evidence_dir: Path,
    output_dir: Path,
    repeats: int,
    seed_store: Optional[Path] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_pools: dict = {}
    seed_fetches: dict = {}
    if seed_store:
        pool_path = seed_store / "_search_pools.json"
        fetch_path = seed_store / "_fetch_cache.json"
        if pool_path.exists():
            seed_pools = json.loads(pool_path.read_text(encoding="utf-8"))
        if fetch_path.exists():
            seed_fetches = json.loads(fetch_path.read_text(encoding="utf-8"))

    pools: dict = {}
    fetches: dict = {}
    coverage: list[dict] = []
    for task in _load_suite(suite):
        task_id = task["task_id"]
        if task_ids and task_id not in task_ids:
            continue
        metadata = task.get("metadata") or {}
        reference_url = str((metadata.get("provenance") or {}).get("reference_url", ""))
        snapshot = evidence_dir / f"{task_id}.txt"
        if not reference_url or not snapshot.exists():
            raise ValueError(f"{task_id}: reference URL or evidence snapshot missing")
        content = snapshot.read_text(encoding="utf-8")
        normalized_reference = _normalize_url_for_snapshot(reference_url)
        required_domains = {
            str(item).lower().removeprefix("www.")
            for item in metadata.get("required_sources", [])
        }
        available_domains: set[str] = set()

        for repeat in range(1, repeats + 1):
            seed_pool = seed_pools.get(f"{task_id}__r1", {"entries": []})
            entries = list(seed_pool.get("entries", []))
            entries.append({
                "query": task["prompt"],
                "tokens": [],
                "result": (
                    f"Frozen authoritative result for {task_id}: "
                    f"{reference_url}"
                ),
            })
            # SnapshotStore can rebuild missing token lists only for newly added
            # corpora if they are populated here.
            from nexa_agent.eval_v2.snapshot_replay import _tokenize_query
            for entry in entries:
                entry["tokens"] = list(_tokenize_query(str(entry["query"])))
            pools[f"{task_id}__r{repeat}"] = {"entries": entries}

            prefix = f"{task_id}__r1__"
            for key, value in seed_fetches.items():
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    fetches[f"{task_id}__r{repeat}__{suffix}"] = value
                    available_domains.add(
                        urlparse("https://" + suffix).netloc.removeprefix("www.")
                    )
            fetches[
                f"{task_id}__r{repeat}__{normalized_reference}"
            ] = content
            available_domains.add(
                urlparse(reference_url).netloc.lower().removeprefix("www.")
            )

        missing = sorted(
            domain for domain in required_domains
            if not any(domain in available for available in available_domains)
        )
        coverage.append({
            "task_id": task_id,
            "required_domains": sorted(required_domains),
            "available_domains": sorted(available_domains),
            "missing_required_domains": missing,
            "pass": not missing,
        })

    pool_path = output_dir / "_search_pools.json"
    fetch_path = output_dir / "_fetch_cache.json"
    pool_path.write_text(
        json.dumps(pools, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fetch_path.write_text(
        json.dumps(fetches, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "closed-world-corpus-r3",
        "suite": str(suite),
        "suite_sha256": _sha256(suite),
        "repeats": repeats,
        "configuration_independent": True,
        "built_before_comparison": True,
        "search_pool_sha256": _sha256(pool_path),
        "fetch_cache_sha256": _sha256(fetch_path),
        "coverage": coverage,
        "pass": bool(coverage) and all(item["pass"] for item in coverage),
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--task-ids", nargs="*", default=[])
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-store", type=Path)
    args = parser.parse_args()
    report = build(
        args.suite,
        set(args.task_ids),
        args.evidence_dir,
        args.output_dir,
        args.repeats,
        args.seed_store,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
