"""
系统化 Eval Harness — 回归评测流水线

将一次性 GAIA 跑分升级为可复用的回归评测框架：
- 固定测试集 + 自定义用例
- 多维度评分: accuracy / failure_mode / cost / pass^k
- JSONL 结果持久化 + 增量对比
- Failure mode 归因统计

用法::

    # 运行完整评测
    python -m nexa_agent.eval_harness --suite gaia_l1

    # 运行回归子集 (快速门禁)
    python -m nexa_agent.eval_harness --suite gaia_l1 --subset regression

    # 对比两次运行
    python -m nexa_agent.eval_harness compare --baseline results/run_A.jsonl --current results/run_B.jsonl

    # 查看 failure mode 分布
    python -m nexa_agent.eval_harness analyze --input results/run_A.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from nexa_agent.logger import get_logger
from nexa_agent.config import (
    REACT_CONFIG, REFLEXION_CONFIG, get_config_summary, get_model_for_role,
)

logger = get_logger("eval_harness")

# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """单个评测用例"""
    case_id: str
    question: str
    expected_answer: str
    level: int = 1
    tags: list[str] = field(default_factory=list)
    file_path: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class EvalRecord:
    """单个评测结果记录"""
    case_id: str
    question: str
    expected_answer: str
    prediction: str
    correct: bool
    trials_used: int
    elapsed_seconds: float
    failure_mode: Optional[str] = None
    reflections: list[str] = field(default_factory=list)
    step_count: int = 0
    level: int = 1
    tags: list[str] = field(default_factory=list)
    agent_success: bool = False
    timestamp: str = ""


@dataclass
class EvalReport:
    """评测汇总报告"""
    run_id: str
    suite_name: str
    timestamp: str
    total: int
    correct: int
    accuracy: float
    avg_elapsed: float
    avg_trials: float
    avg_steps: float
    failure_mode_dist: dict[str, int] = field(default_factory=dict)
    per_tag_accuracy: dict[str, float] = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)
    records: list[EvalRecord] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 答案匹配 (复用 GAIA 官方逻辑)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.,/-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def match_answer(prediction: str, reference: str) -> bool:
    if not prediction or not prediction.strip():
        return False

    pred_norm = _normalize_text(prediction.strip())
    ref_norm = _normalize_text(reference.strip())

    if pred_norm == ref_norm:
        return True
    if ref_norm in pred_norm:
        return True

    pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred_norm)
    ref_nums = re.findall(r"[-+]?\d*\.?\d+", ref_norm)
    if ref_nums and pred_nums:
        if ref_nums[-1] == pred_nums[-1]:
            return True
        try:
            if abs(float(ref_nums[-1]) - float(pred_nums[-1])) < 1e-3:
                return True
        except ValueError:
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 测试集加载
# ═══════════════════════════════════════════════════════════════════════════

GAIA_DIR = Path(__file__).parent.parent / "GAIA" / "2023" / "validation"
CUSTOM_SUITE_DIR = Path(__file__).parent / "eval_suites"


def load_gaia_suite(levels: list[int] = None, subset: str = None) -> list[EvalCase]:
    """加载 GAIA 数据集作为评测套件"""
    try:
        import pandas as pd
    except ImportError:
        logger.error("需要 pandas: pip install pandas pyarrow")
        return []

    parquet_path = GAIA_DIR / "metadata.parquet"
    if not parquet_path.exists():
        logger.error("GAIA 数据集不存在: %s", parquet_path)
        return []

    df = pd.read_parquet(parquet_path)
    if levels:
        df["Level"] = df["Level"].astype(int)
        df = df[df["Level"].isin(levels)]

    if subset == "regression":
        regression_ids = _load_regression_subset()
        if regression_ids:
            df = df[df["task_id"].isin(regression_ids)]

    cases = []
    for _, row in df.iterrows():
        file_path = None
        image_path = None
        file_name = row.get("file_name", "")
        if file_name and not (isinstance(file_name, float)):
            fp = GAIA_DIR / file_name
            if fp.exists():
                if str(fp).lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = str(fp)
                else:
                    file_path = str(fp)

        cases.append(EvalCase(
            case_id=row["task_id"],
            question=row["Question"],
            expected_answer=str(row["Final answer"]),
            level=int(row["Level"]),
            tags=[f"level_{int(row['Level'])}"],
            file_path=file_path,
            image_path=image_path,
        ))

    return cases


def load_custom_suite(suite_path: str) -> list[EvalCase]:
    """从 JSONL 文件加载自定义测试套件"""
    cases = []
    with open(suite_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            cases.append(EvalCase(
                case_id=data.get("case_id", data.get("task_id", "")),
                question=data["question"],
                expected_answer=data["expected_answer"],
                level=data.get("level", 1),
                tags=data.get("tags", []),
                file_path=data.get("file_path"),
                image_path=data.get("image_path"),
            ))
    return cases


def _load_regression_subset() -> set[str]:
    """加载回归测试子集 ID（从历史失败用例中选取）"""
    subset_file = CUSTOM_SUITE_DIR / "regression_ids.json"
    if subset_file.exists():
        with open(subset_file, "r") as f:
            return set(json.load(f))
    return set()


# ═══════════════════════════════════════════════════════════════════════════
# Failure Mode 提取
# ═══════════════════════════════════════════════════════════════════════════

def extract_failure_mode(record: dict) -> Optional[str]:
    """从评测记录中提取 failure mode"""
    if record.get("correct"):
        return None

    for detail in record.get("trial_details", []):
        fm = detail.get("failure_mode")
        if fm:
            return fm

    reflections = record.get("reflections", [])
    if reflections:
        text = " ".join(reflections).lower()
        if "循环" in text or "重复" in text or "loop" in text:
            return "loop"
        if "工具" in text or "tool" in text:
            return "tool_misuse"
        if "步数" in text or "max_step" in text:
            return "context_overflow"

    prediction = record.get("prediction", "")
    if not prediction or len(prediction.strip()) < 5:
        return "empty_answer"
    if any(kw in prediction for kw in ["无法", "抱歉", "不支持"]):
        return "tool_gap"

    return "wrong_reasoning"


# ═══════════════════════════════════════════════════════════════════════════
# 核心评测引擎
# ═══════════════════════════════════════════════════════════════════════════

class EvalHarness:
    """系统化评测执行器"""

    def __init__(
        self,
        max_trials: int = None,
        max_steps: int = None,
        evaluator_mode: str = "heuristic",
    ):
        self.max_trials = max_trials or REFLEXION_CONFIG["max_trials"]
        self.max_steps = max_steps or REACT_CONFIG["max_steps"]
        self.evaluator_mode = evaluator_mode

    def run(
        self,
        cases: list[EvalCase],
        suite_name: str = "custom",
        output_path: str = None,
        resume_from: str = None,
    ) -> EvalReport:
        """执行评测套件"""
        from nexa_agent.reflexion_agent import ReflexionReActAgent

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(results_dir / f"eval_{suite_name}_{run_id}.jsonl")

        completed_ids = set()
        if resume_from and Path(resume_from).exists():
            with open(resume_from, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    completed_ids.add(item["case_id"])
            output_path = resume_from
            logger.info("断点恢复: 已完成 %d 条", len(completed_ids))

        records: list[EvalRecord] = []
        total = len(cases)

        print(f"\n{'═' * 60}")
        print(f"Eval Harness — {suite_name}")
        print(f"Cases: {total} | Trials: {self.max_trials} | Steps: {self.max_steps}")
        print(f"Evaluator: {self.evaluator_mode}")
        print(f"Output: {output_path}")
        print(f"{'═' * 60}\n")

        for idx, case in enumerate(cases):
            if case.case_id in completed_ids:
                continue

            print(f"[{idx+1}/{total}] {case.case_id[:8]}... L{case.level}")
            print(f"  Q: {case.question[:100]}...")

            agent = ReflexionReActAgent(
                max_trials=self.max_trials,
                evaluator_mode=self.evaluator_mode,
                max_steps=self.max_steps,
            )

            task_prompt = case.question
            if case.file_path:
                task_prompt += f"\n\n[附件文件路径: {case.file_path}]"

            start_time = time.time()
            try:
                result = agent.execute(
                    task=task_prompt,
                    image_path=case.image_path,
                    max_steps=self.max_steps,
                )
                elapsed = time.time() - start_time
                prediction = result.answer
                agent_success = result.success
                trials_used = result.trials_used
                reflections = result.reflections
                step_count = sum(
                    d.get("steps", 0) for d in result.trial_details
                )
            except Exception as e:
                elapsed = time.time() - start_time
                prediction = ""
                agent_success = False
                trials_used = 0
                reflections = []
                step_count = 0
                logger.error("Case %s 执行异常: %s", case.case_id[:8], e)

            is_correct = match_answer(prediction, case.expected_answer)

            record_dict = {
                "case_id": case.case_id,
                "question": case.question[:200],
                "expected_answer": case.expected_answer,
                "prediction": prediction,
                "correct": is_correct,
                "trials_used": trials_used,
                "elapsed_seconds": round(elapsed, 1),
                "reflections": reflections,
                "step_count": step_count,
                "level": case.level,
                "tags": case.tags,
                "agent_success": agent_success,
                "timestamp": datetime.now().isoformat(),
            }

            failure_mode = extract_failure_mode(record_dict)
            record_dict["failure_mode"] = failure_mode

            record = EvalRecord(**record_dict)
            records.append(record)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

            status = "✅" if is_correct else "❌"
            fm_str = f" [{failure_mode}]" if failure_mode else ""
            print(f"  {status}{fm_str} Pred: {prediction[:80]} | GT: {case.expected_answer}")

            done = len(records)
            correct_so_far = sum(1 for r in records if r.correct)
            acc = correct_so_far / done * 100 if done > 0 else 0
            print(f"  [进度] {done}/{total} | Acc: {acc:.1f}%\n")

        return self._build_report(run_id, suite_name, records)

    def _build_report(
        self, run_id: str, suite_name: str, records: list[EvalRecord],
    ) -> EvalReport:
        """构建评测报告"""
        total = len(records)
        correct = sum(1 for r in records if r.correct)
        accuracy = correct / total * 100 if total > 0 else 0

        elapsed_list = [r.elapsed_seconds for r in records]
        trials_list = [r.trials_used for r in records]
        steps_list = [r.step_count for r in records]

        failure_modes = Counter(
            r.failure_mode for r in records if r.failure_mode
        )

        per_tag: dict[str, list[bool]] = {}
        for r in records:
            for tag in r.tags:
                per_tag.setdefault(tag, []).append(r.correct)
        per_tag_accuracy = {
            tag: sum(vals) / len(vals) * 100
            for tag, vals in per_tag.items()
        }

        report = EvalReport(
            run_id=run_id,
            suite_name=suite_name,
            timestamp=datetime.now().isoformat(),
            total=total,
            correct=correct,
            accuracy=round(accuracy, 2),
            avg_elapsed=round(statistics.mean(elapsed_list), 1) if elapsed_list else 0,
            avg_trials=round(statistics.mean(trials_list), 2) if trials_list else 0,
            avg_steps=round(statistics.mean(steps_list), 1) if steps_list else 0,
            failure_mode_dist=dict(failure_modes),
            per_tag_accuracy={k: round(v, 1) for k, v in per_tag_accuracy.items()},
            config_snapshot={
                "max_trials": self.max_trials,
                "max_steps": self.max_steps,
                "evaluator_mode": self.evaluator_mode,
                "react_first_model": get_model_for_role("react_first"),
                "react_main_model": get_model_for_role("react_main"),
            },
            records=records,
        )

        self._print_report(report)
        return report

    def _print_report(self, report: EvalReport):
        """打印评测报告"""
        print(f"\n{'═' * 60}")
        print(f"  Eval Report — {report.suite_name}")
        print(f"{'═' * 60}")
        print(f"  Run ID:    {report.run_id}")
        print(f"  Total:     {report.total}")
        print(f"  Correct:   {report.correct}")
        print(f"  Accuracy:  {report.accuracy:.1f}%")
        print(f"  Avg Time:  {report.avg_elapsed:.1f}s")
        print(f"  Avg Trials: {report.avg_trials:.1f}")
        print(f"  Avg Steps: {report.avg_steps:.0f}")

        if report.failure_mode_dist:
            print(f"\n  Failure Modes:")
            for mode, count in sorted(
                report.failure_mode_dist.items(), key=lambda x: -x[1]
            ):
                pct = count / (report.total - report.correct) * 100
                print(f"    {mode:<20} {count:>3} ({pct:.0f}%)")

        if report.per_tag_accuracy:
            print(f"\n  Per-Tag Accuracy:")
            for tag, acc in sorted(report.per_tag_accuracy.items()):
                print(f"    {tag:<20} {acc:.1f}%")

        print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 结果分析与对比
# ═══════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> list[dict]:
    """加载 JSONL 结果文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_results(path: str):
    """分析单次评测结果"""
    records = load_results(path)
    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    accuracy = correct / total * 100 if total > 0 else 0

    failure_modes = Counter()
    for r in records:
        fm = r.get("failure_mode") or extract_failure_mode(r)
        if fm:
            failure_modes[fm] += 1

    elapsed_list = [r["elapsed_seconds"] for r in records if "elapsed_seconds" in r]
    trials_list = [r.get("trials_used", 1) for r in records]

    print(f"\n{'═' * 60}")
    print(f"  Analysis: {path}")
    print(f"{'═' * 60}")
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct} ({accuracy:.1f}%)")
    print(f"  Avg Time: {statistics.mean(elapsed_list):.1f}s" if elapsed_list else "")
    print(f"  Avg Trials: {statistics.mean(trials_list):.1f}")

    if failure_modes:
        print(f"\n  Failure Mode Distribution:")
        failed = total - correct
        for mode, count in failure_modes.most_common():
            pct = count / failed * 100 if failed > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {mode:<20} {count:>3} ({pct:4.0f}%) {bar}")

    print(f"{'═' * 60}\n")


def compare_runs(baseline_path: str, current_path: str):
    """对比两次评测结果"""
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    base_by_id = {r.get("case_id", r.get("task_id")): r for r in baseline}
    curr_by_id = {r.get("case_id", r.get("task_id")): r for r in current}

    common_ids = set(base_by_id.keys()) & set(curr_by_id.keys())

    if not common_ids:
        print("⚠️ 没有共同的 case_id，无法对比")
        return

    base_correct = sum(1 for cid in common_ids if base_by_id[cid].get("correct"))
    curr_correct = sum(1 for cid in common_ids if curr_by_id[cid].get("correct"))
    n = len(common_ids)

    base_acc = base_correct / n * 100
    curr_acc = curr_correct / n * 100
    delta = curr_acc - base_acc

    regressions = []
    improvements = []
    for cid in common_ids:
        b = base_by_id[cid].get("correct", False)
        c = curr_by_id[cid].get("correct", False)
        if b and not c:
            regressions.append(cid)
        elif not b and c:
            improvements.append(cid)

    print(f"\n{'═' * 60}")
    print(f"  Run Comparison")
    print(f"{'═' * 60}")
    print(f"  Common cases: {n}")
    print(f"  Baseline:  {base_correct}/{n} ({base_acc:.1f}%)")
    print(f"  Current:   {curr_correct}/{n} ({curr_acc:.1f}%)")

    delta_sign = "+" if delta >= 0 else ""
    delta_color = "✅" if delta >= 0 else "🔴"
    print(f"  Delta:     {delta_color} {delta_sign}{delta:.1f}pp")

    if regressions:
        print(f"\n  ⚠️ Regressions ({len(regressions)}):")
        for cid in regressions[:10]:
            q = base_by_id[cid].get("question", "")[:60]
            print(f"    - {cid[:8]}... {q}")

    if improvements:
        print(f"\n  ✅ Improvements ({len(improvements)}):")
        for cid in improvements[:10]:
            q = curr_by_id[cid].get("question", "")[:60]
            print(f"    + {cid[:8]}... {q}")

    print(f"{'═' * 60}\n")

    if delta < -2.0:
        print("🚨 REGRESSION GATE: 准确率下降超过 2pp，建议阻断提交")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nexa Agent Eval Harness — 系统化回归评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── run ──
    run_parser = subparsers.add_parser("run", help="运行评测")
    run_parser.add_argument("--suite", type=str, default="gaia_l1",
                            help="测试套件: gaia_l1, gaia_l2, 或自定义 JSONL 路径")
    run_parser.add_argument("--subset", type=str, default=None,
                            choices=["regression", "all"],
                            help="子集: regression (快速回归) 或 all")
    run_parser.add_argument("--limit", type=int, default=None,
                            help="限制评测数量")
    run_parser.add_argument("--max-trials", type=int, default=None)
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--evaluator", default="heuristic",
                            choices=["heuristic", "llm", "hybrid"])
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.add_argument("--resume", type=str, default=None)

    # ── analyze ──
    analyze_parser = subparsers.add_parser("analyze", help="分析评测结果")
    analyze_parser.add_argument("--input", "-i", type=str, required=True)

    # ── compare ──
    compare_parser = subparsers.add_parser("compare", help="对比两次评测")
    compare_parser.add_argument("--baseline", "-b", type=str, required=True)
    compare_parser.add_argument("--current", "-c", type=str, required=True)

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        suite = getattr(args, "suite", "gaia_l1")
        cases = []

        if suite == "gaia_l1":
            cases = load_gaia_suite(levels=[1], subset=getattr(args, "subset", None))
        elif suite == "gaia_l2":
            cases = load_gaia_suite(levels=[2], subset=getattr(args, "subset", None))
        elif suite == "gaia_all":
            cases = load_gaia_suite(levels=[1, 2, 3], subset=getattr(args, "subset", None))
        elif Path(suite).exists():
            cases = load_custom_suite(suite)
        else:
            print(f"❌ 未知测试套件: {suite}")
            sys.exit(1)

        if not cases:
            print("❌ 没有加载到测试用例")
            sys.exit(1)

        limit = getattr(args, "limit", None)
        if limit:
            cases = cases[:limit]

        harness = EvalHarness(
            max_trials=getattr(args, "max_trials", None),
            max_steps=getattr(args, "max_steps", None),
            evaluator_mode=getattr(args, "evaluator", "heuristic"),
        )
        report = harness.run(
            cases=cases,
            suite_name=suite,
            output_path=getattr(args, "output", None),
            resume_from=getattr(args, "resume", None),
        )

        report_path = Path(getattr(args, "output", None) or "").parent / f"report_{report.run_id}.json"
        if str(report_path) != ".":
            report_dict = asdict(report)
            report_dict.pop("records", None)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            print(f"📊 Report saved: {report_path}")

    elif args.command == "analyze":
        analyze_results(args.input)

    elif args.command == "compare":
        success = compare_runs(args.baseline, args.current)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
