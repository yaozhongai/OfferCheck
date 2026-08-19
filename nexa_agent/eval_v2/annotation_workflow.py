"""Export blinded calibration sheets and merge two independent annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "sample_id",
    "category",
    "question",
    "claim",
    "evidence_excerpt",
    "declared_sources",
    "rubric",
    "is_negative_task",
    "label",
    "annotation_note",
)


def export(pack: Path, output: Path) -> None:
    samples = json.loads(pack.read_text(encoding="utf-8"))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "sample_id": sample["sample_id"],
                "category": sample.get("category", ""),
                "question": sample.get("question", ""),
                "claim": sample.get("claim", ""),
                "evidence_excerpt": sample.get("evidence_excerpt", ""),
                "declared_sources": json.dumps(
                    sample.get("declared_sources", []), ensure_ascii=False
                ),
                "rubric": sample.get("rubric", ""),
                "is_negative_task": sample.get("is_negative_task", False),
                "label": "",
                "annotation_note": "",
            })


def _read_sheet(path: Path, reviewer_id: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label_text = row["label"].strip().lower()
            if label_text not in {"pass", "fail"}:
                raise ValueError(
                    f"{path}:{row['sample_id']}: label must be pass or fail"
                )
            rows[row["sample_id"]] = {
                "annotator_id": reviewer_id,
                "label": label_text == "pass",
                "note": row.get("annotation_note", "").strip(),
            }
    return rows


def merge(
    pack: Path,
    sheet_a: Path,
    reviewer_a: str,
    sheet_b: Path,
    reviewer_b: str,
    output: Path,
) -> None:
    if not reviewer_a.strip() or not reviewer_b.strip() or reviewer_a == reviewer_b:
        raise ValueError("two distinct non-empty reviewer IDs are required")
    samples = json.loads(pack.read_text(encoding="utf-8"))
    a = _read_sheet(sheet_a, reviewer_a)
    b = _read_sheet(sheet_b, reviewer_b)
    expected = {item["sample_id"] for item in samples}
    if set(a) != expected or set(b) != expected:
        raise ValueError("both sheets must contain every sample exactly once")
    for sample in samples:
        sid = sample["sample_id"]
        sample["annotations"] = [a[sid], b[sid]]
        labels = {a[sid]["label"], b[sid]["label"]}
        sample["human_label"] = labels.pop() if len(labels) == 1 else None
        sample["label_status"] = (
            "agreed" if sample["human_label"] is not None else "needs_adjudication"
        )
    output.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("pack", type=Path)
    exp.add_argument("--output", required=True, type=Path)
    combine = sub.add_parser("merge")
    combine.add_argument("pack", type=Path)
    combine.add_argument("--sheet-a", required=True, type=Path)
    combine.add_argument("--reviewer-a", required=True)
    combine.add_argument("--sheet-b", required=True, type=Path)
    combine.add_argument("--reviewer-b", required=True)
    combine.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.pack, args.output)
    else:
        merge(
            args.pack, args.sheet_a, args.reviewer_a,
            args.sheet_b, args.reviewer_b, args.output,
        )


if __name__ == "__main__":
    main()
