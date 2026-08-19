"""Command-line entry point for frozen Nexa Agent evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from nexa_agent.config import MODEL_ROUTING, MODEL_TIER, SEARCH_CONFIG
from nexa_agent.eval_v2.runner import ExperimentRunner
from nexa_agent.eval_v2.schemas import (
    BudgetLimits,
    ExperimentConfig,
    RunManifest,
)
from nexa_agent.eval_v2.suite_loader import load_jsonl_suite
from nexa_agent.harness_config import FULL_HARNESS, MINIMAL_REACT


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _redact_secrets(value):
    """Remove credentials from persisted manifests without changing runtime config."""
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in {"api_key", "token", "secret", "password"}
                and item
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def _git_snapshot() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        [
            *PROJECT_ROOT.glob("nexa_agent/**/*.py"),
            *PROJECT_ROOT.glob("nexa_agent/prompts/*.txt"),
            *PROJECT_ROOT.glob("offercheck/**/*.py"),
        ]
    )
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prompt_snapshot() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _file_sha256(path)
        for path in sorted((PROJECT_ROOT / "nexa_agent" / "prompts").glob("*.txt"))
    }


def _evaluation_component_snapshot() -> dict[str, str]:
    names = (
        "runner.py",
        "graders.py",
        "semantic_judge.py",
        "snapshot_replay.py",
        "regrade.py",
        "suite_loader.py",
    )
    base = PROJECT_ROOT / "nexa_agent" / "eval_v2"
    return {name: _file_sha256(base / name) for name in names}


def build_manifest(args: argparse.Namespace) -> RunManifest:
    suite_path = Path(args.suite).resolve()
    tasks = load_jsonl_suite(suite_path)
    commit, dirty = _git_snapshot()
    if dirty and not args.allow_dirty_worktree:
        raise RuntimeError(
            "evaluation refuses a dirty worktree; commit the exact source first "
            "or use --allow-dirty-worktree only for non-authoritative pilots"
        )
    configs = (
        ExperimentConfig(
            config_id="minimal_react",
            runtime_profile="minimal",
            max_internal_trials=args.max_internal_trials,
            evaluator_mode="heuristic",
            max_steps=args.max_steps,
            excluded_tools=tuple(args.excluded_tools),
            model_tier_override=args.model_tier_override,
            enable_curation=False,
            reflexion_enabled=False,
            metadata={
                "feature_flags": MINIMAL_REACT.__dict__,
                "search_provider_order": list(args.search_provider_order),
            },
        ),
        ExperimentConfig(
            config_id="full_harness",
            runtime_profile="full",
            max_internal_trials=args.max_internal_trials,
            evaluator_mode="hybrid",
            max_steps=args.max_steps,
            excluded_tools=tuple(args.excluded_tools),
            model_tier_override=args.model_tier_override,
            enable_curation=False,
            reflexion_enabled=True,
            metadata={
                "feature_flags": FULL_HARNESS.__dict__,
                "search_provider_order": list(args.search_provider_order),
            },
        ),
    )
    return RunManifest(
        run_id=args.run_id,
        suite_id=f"{suite_path.stem}:{_file_sha256(suite_path)}",
        seed=args.seed,
        repeats=args.repeats,
        code_commit=commit,
        dirty_worktree=dirty,
        source_snapshot_sha256=_source_snapshot_sha256(),
        model_snapshot={
            "tiers": _redact_secrets(MODEL_TIER),
            "routing": _redact_secrets(MODEL_ROUTING),
            "suite_path": str(suite_path),
            "suite_sha256": _file_sha256(suite_path),
            "search_provider_order": list(args.search_provider_order),
            "excluded_tools": list(args.excluded_tools),
            "model_tier_override": args.model_tier_override,
            "prompt_sha256": _prompt_snapshot(),
            "evaluation_component_sha256": _evaluation_component_snapshot(),
            "dependency_files_sha256": {
                path.name: _file_sha256(path)
                for path in (
                    PROJECT_ROOT / "pyproject.toml",
                    PROJECT_ROOT / "uv.lock",
                )
                if path.exists()
            },
            "snapshot_corpus": (
                {
                    "path": str(Path(args.snapshot_corpus).resolve()),
                    "manifest_sha256": _file_sha256(
                        Path(args.snapshot_corpus) / "corpus_manifest.json"
                    ),
                }
                if args.snapshot_corpus else None
            ),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
        budgets=BudgetLimits(
            max_steps_per_internal_trial=args.max_steps,
            max_internal_trials=args.max_internal_trials,
            max_tokens_per_trial=args.max_tokens_per_trial,
            max_wall_seconds_per_trial=args.max_wall_seconds_per_trial,
            max_llm_attempts_per_trial=args.max_llm_attempts_per_trial,
            max_tool_calls_per_trial=args.max_tool_calls_per_trial,
            max_total_tokens=args.max_total_tokens,
            max_total_wall_seconds=args.max_total_wall_seconds,
        ),
        configs=configs,
        tasks=tasks,
        snapshot_policy=args.snapshot_policy,
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run semantic judge calibration on suite-based auto-generated samples."""
    import json as _json
    from nexa_agent.eval_v2.semantic_judge import (
        SemanticJudge,
        generate_calibration_from_suites,
    )

    suite_paths = [str(Path(p).resolve()) for p in args.suites]
    print(f"Generating calibration samples from {len(suite_paths)} suites...")
    samples = generate_calibration_from_suites(
        suite_paths,
        samples_per_task=args.samples_per_task,
    )
    print(f"Generated {len(samples)} calibration samples (target: ≥{args.min_samples})")

    if args.min_samples and len(samples) < args.min_samples:
        print(f"WARNING: only {len(samples)} samples, below minimum {args.min_samples}")
        print("Consider adding more suite files or increasing --samples-per-task.")

    if args.dry_run:
        print("\n=== Sample preview (first 5) ===")
        for s in samples[:5]:
            label = "✓ CORRECT" if s.human_label else "✗ INCORRECT"
            print(f"  [{s.sample_id}] {label}")
            print(f"    Q: {s.question[:100]}...")
            print(f"    A: {s.answer[:150]}...")
            print()
        # Save calibration set for later manual labeling
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cal_data = [_json.loads(_json.dumps(s.__dict__)) for s in samples]
        with open(out_path, "w") as f:
            _json.dump(cal_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(samples)} auto-generated samples to {out_path}")
        print("Run 'calibrate --judge' to run LLM judge on these samples.")
        return

    if args.judge:
        print("\nRunning LLM judge on calibration samples...")
        judge = SemanticJudge()
        result = judge.calibrate(samples)
        print(f"\n=== Calibration Results ===")
        print(f"Total samples:    {result.total}")
        print(f"Correct:          {result.correct}")
        print(f"Accuracy:         {result.accuracy:.1%}")
        print(f"False positives:  {result.false_positives}")
        print(f"False negatives:  {result.false_negatives}")
        print(f"Mean confidence (correct): {result.mean_confidence_correct:.3f}")
        print(f"Mean confidence (wrong):   {result.mean_confidence_wrong:.3f}")

        if result.false_positives + result.false_negatives > 0:
            print(f"\n=== Disagreements ===")
            for s in result.per_sample:
                if not s["match"]:
                    judge_label = "PASS" if s["judge_pass"] else "FAIL"
                    human_label = "PASS" if s["human_label"] else "FAIL"
                    print(f"  [{s['sample_id']}] Judge={judge_label} Human={human_label} "
                          f"conf={s['judge_confidence']:.2f}")
                    print(f"    Reasoning: {s['judge_reasoning'][:200]}")

        # Save result
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_data = {
            "total": result.total,
            "correct": result.correct,
            "accuracy": result.accuracy,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "mean_confidence_correct": result.mean_confidence_correct,
            "mean_confidence_wrong": result.mean_confidence_wrong,
            "per_sample": result.per_sample,
        }
        with open(out_path, "w") as f:
            _json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved calibration results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", help="Subcommand: run | calibrate")

    # ---- calibrate ----
    cal_parser = sub.add_parser("calibrate", help="Calibrate the semantic judge")
    cal_parser.add_argument(
        "--suites", nargs="+", required=True,
        help="Suite JSONL files to generate calibration samples from",
    )
    cal_parser.add_argument(
        "--samples-per-task", type=int, default=3,
        help="Number of samples per task (default: 3 = pos+neg+edge)",
    )
    cal_parser.add_argument(
        "--min-samples", type=int, default=60,
        help="Minimum required samples for calibration (default: 60)",
    )
    cal_parser.add_argument(
        "--judge", action="store_true",
        help="Actually run the LLM judge (requires API key). Without this, only generates samples.",
    )
    cal_parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate samples and save to disk without running judge.",
    )
    cal_parser.add_argument(
        "--output", default=str(
            PROJECT_ROOT / "nexa_agent" / "results" / "resume_eval_v0"
            / "judge_calibration.json"
        ),
        help="Output path for calibration samples/results",
    )

    # ---- run (default) ----
    run_parser = sub.add_parser("run", help="Run an evaluation experiment")
    run_parser.add_argument("--suite", required=True)
    run_parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "nexa_agent" / "results" / "eval_v2"),
    )
    run_parser.add_argument(
        "--run-id",
        default=f"pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    run_parser.add_argument("--seed", type=int, default=20260723)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--max-steps", type=int, default=16)
    run_parser.add_argument("--max-internal-trials", type=int, default=3)
    run_parser.add_argument("--max-tokens-per-trial", type=int, default=350000)
    run_parser.add_argument("--max-wall-seconds-per-trial", type=float, default=1200)
    run_parser.add_argument("--max-total-tokens", type=int, default=2500000)
    run_parser.add_argument("--max-total-wall-seconds", type=float, default=14400)
    run_parser.add_argument("--max-llm-attempts-per-trial", type=int, default=64)
    run_parser.add_argument("--max-tool-calls-per-trial", type=int, default=64)
    run_parser.add_argument(
        "--snapshot-policy",
        choices=("pre_recorded_replay", "paired_record_replay"),
        default="pre_recorded_replay",
        help=(
            "Use a pre-recorded closed world for both configs (default). "
            "paired_record_replay is only for corpus preparation and pilots."
        ),
    )
    run_parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Allow dirty source only for explicitly non-authoritative pilots.",
    )
    run_parser.add_argument(
        "--snapshot-corpus",
        help=(
            "Directory containing frozen _search_pools.json, _fetch_cache.json "
            "and corpus_manifest.json. Required by pre_recorded_replay."
        ),
    )
    run_parser.add_argument(
        "--model-tier-override",
        choices=("strong", "fast", "upgrade"),
        default="strong",
        help="Force all text-LLM roles to the same tier for fair evaluation.",
    )
    run_parser.add_argument(
        "--search-provider-order",
        default="ddg,exa",
        help="Evaluation-only search order. Tavily is excluded by default.",
    )
    run_parser.add_argument(
        "--allow-tavily-extract",
        action="store_true",
        help="Expose tavily_extract during evaluation (disabled by default).",
    )
    args = parser.parse_args()

    if args.command == "calibrate":
        cmd_calibrate(args)
        return

    # ---- run or default ----
    if not hasattr(args, "suite") or not args.suite:
        parser.error("--suite is required for 'run' command")
    args.search_provider_order = tuple(
        item.strip()
        for item in args.search_provider_order.split(",")
        if item.strip()
    )
    if not args.search_provider_order:
        parser.error("--search-provider-order cannot be empty")
    args.excluded_tools = (
        ()
        if args.allow_tavily_extract
        else ("tavily_extract",)
    )
    SEARCH_CONFIG["provider_order"] = list(args.search_provider_order)

    if args.snapshot_policy == "pre_recorded_replay" and not args.snapshot_corpus:
        parser.error("--snapshot-corpus is required for pre_recorded_replay")
    manifest = build_manifest(args)
    if args.snapshot_corpus:
        corpus = Path(args.snapshot_corpus)
        corpus_manifest = json.loads(
            (corpus / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        if not corpus_manifest.get("pass"):
            parser.error("snapshot corpus coverage audit did not pass")
        target = Path(args.output_root) / args.run_id / "tool_snapshots"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("_search_pools.json", "_fetch_cache.json"):
            destination = target / name
            if destination.exists() and _file_sha256(destination) != _file_sha256(corpus / name):
                parser.error(f"existing run has a different snapshot corpus: {name}")
            shutil.copyfile(corpus / name, destination)
    summary = ExperimentRunner(Path(args.output_root)).run(manifest)
    run_dir = Path(args.output_root) / args.run_id
    history_path = run_dir / "run_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "recorded_at": datetime.now().astimezone().isoformat(),
            "manifest_hash": manifest.manifest_hash,
            "summary": summary,
        }, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"run_id={manifest.run_id}")
    print(f"manifest_hash={manifest.manifest_hash}")
    print(f"result={summary}")


if __name__ == "__main__":
    main()
