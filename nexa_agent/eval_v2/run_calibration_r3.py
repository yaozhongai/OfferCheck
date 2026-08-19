"""Run the entailment judge on fully human-labeled r3 calibration samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexa_agent.eval_v2.semantic_judge import SemanticJudge


def _final_label(sample: dict) -> bool:
    if isinstance(sample.get("adjudicated_label"), bool):
        return sample["adjudicated_label"]
    labels = [item.get("label") for item in sample.get("annotations", [])]
    if len(labels) >= 2 and all(isinstance(item, bool) for item in labels):
        if len(set(labels)) == 1:
            return labels[0]
    raise ValueError(f"{sample.get('sample_id')}: no agreed/adjudicated label")


def run(samples: list[dict]) -> dict:
    judge = SemanticJudge()
    rows = []
    fp = fn = correct = safety_fn = 0
    positive = negative = 0
    for sample in samples:
        label = _final_label(sample)
        positive += int(label)
        negative += int(not label)
        verdict = judge.judge_claim_entailment(
            question=sample["question"],
            claim=sample["claim"],
            evidence_excerpt=sample["evidence_excerpt"],
            source_refs=sample.get("declared_sources", []),
            rubric=sample.get("rubric", ""),
            is_negative_task=bool(sample.get("is_negative_task")),
        )
        match = verdict.pass_ == label
        correct += int(match)
        fp += int(verdict.pass_ and not label)
        fn += int(not verdict.pass_ and label)
        if sample.get("category") == "injection_followed" and verdict.pass_:
            safety_fn += 1
        rows.append({
            "sample_id": sample["sample_id"],
            "category": sample.get("category"),
            "human_label": label,
            "judge_pass": verdict.pass_,
            "match": match,
            "judge_confidence": verdict.confidence,
            "judge_reasoning": verdict.reasoning,
        })
    return {
        "schema_version": "judge-calibration-results-r3",
        "total": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
        "human_positive": positive,
        "human_negative": negative,
        "safety_false_negatives": safety_fn,
        "per_sample": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(json.loads(args.samples.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
