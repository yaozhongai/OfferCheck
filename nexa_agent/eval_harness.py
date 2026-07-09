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
    REACT_CONFIG, REFLEXION_CONFIG, get_model_for_role,
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
    # OfferCheck 裁定级评测：expected_verdict ∈ reliable | suspicious | likely_scam
    # 设置后按裁定归一化匹配（而非 GAIA 子串匹配），并透传 stage 加载阶段 prompt
    expected_verdict: Optional[str] = None
    stage: Optional[str] = None
    # 关键词召回评测（stage2 简历定向等非裁定型）：预置「必须被指出的差距关键词」，
    # 按 Agent 输出是否命中这些关键词打召回分（详见 score_keyword_recall）
    expected_keywords: Optional[list[str]] = None


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
    # Token 用量（评审 3.6）：单条用例的调查主循环 in/out token，作成本指标
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 裁定级评测：期望/预测裁定归一化等级（None = 非 OfferCheck 用例）
    expected_verdict: Optional[str] = None
    predicted_verdict: Optional[str] = None
    # 关键词召回评测（stage2）：期望关键词 + 实测召回率（0~1）
    expected_keywords: Optional[list[str]] = None
    keyword_recall: Optional[float] = None


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
    # 成本指标（评审 3.6）：每裁定平均 token（in/out/合计），与准确率同为可量化 harness 硬通货
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    config_snapshot: dict = field(default_factory=dict)
    # 裁定级指标（仅当套件含 expected_verdict 用例时填充）
    verdict_metrics: dict = field(default_factory=dict)
    # 关键词召回指标（仅当套件含 expected_keywords 用例时填充）
    keyword_metrics: dict = field(default_factory=dict)
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
# OfferCheck 裁定级评分
# ═══════════════════════════════════════════════════════════════════════════

VERDICT_LEVELS = ("reliable", "suspicious", "likely_scam")


def classify_prediction_verdict(prediction: str) -> str:
    """把 Agent 输出归一化为裁定等级

    Returns:
        reliable | suspicious | likely_scam | unknown（拒答/无裁定）

    注意：`_classify_verdict_level` 对整行 [Verdict] 文本做子串匹配，会被
    「未发现…诈骗案例」这类**否定语境**里的 scam 关键词误伤（把「靠谱」判成
    likely_scam）。评分只信任裁定**标签本身**——即 [Verdict] 行分隔符
    （——/—/-/:）之前的那截 label，对它做有序关键词匹配；仅当没有 [Verdict]
    标签（自由文本回答）时才回退到全文兜底分类。
    """
    if not prediction or not prediction.strip():
        return "unknown"

    verdict_match = re.search(r"\[Verdict\]\s*(.*?)(?:\n|$)", prediction, re.IGNORECASE)
    if verdict_match:
        label = re.split(r"——|—|:|：|\s-\s", verdict_match.group(1).strip(), maxsplit=1)[0]
        level = _classify_verdict_label(label)
        if level != "unknown":
            return level
        # label 本身无法判定时，退回到整条 verdict 行
        return _classify_verdict_label(verdict_match.group(1))

    # 无 [Verdict] 标签 → 自由文本兜底
    from nexa_agent.verifier import _classify_verdict_level
    return _classify_verdict_level(prediction)


def _classify_verdict_label(label: str) -> str:
    """对裁定 label 片段做有序关键词匹配（scam → suspicious → reliable）"""
    t = label.lower()
    if any(k in label for k in ("大概率有坑", "有坑", "诈骗", "骗局", "不推荐")) or \
       any(k in t for k in ("scam", "fraud", "high risk")):
        return "likely_scam"
    if any(k in label for k in ("存疑", "谨慎", "可疑")) or \
       any(k in t for k in ("suspicious", "caution", "uncertain")):
        return "suspicious"
    if any(k in label for k in ("靠谱", "可靠", "推荐")) or \
       any(k in t for k in ("reliable", "legit", "trustworthy", "safe")):
        return "reliable"
    return "unknown"


def compute_verdict_metrics(records: list["EvalRecord"]) -> dict:
    """裁定级评测指标：准确率 / 误报率 / 漏报率 / 拒答率 + 混淆矩阵

    - accuracy:            裁定等级精确匹配率
    - false_positive_rate: 误报——把「靠谱(reliable)」判成 suspicious/likely_scam 的比例
                           （吓退好机会，产品体验杀手）
    - miss_rate:           漏报——把「大概率有坑(likely_scam)」判成 reliable 的比例
                           （最危险，用户可能因此受骗）
    - refusal_rate:        拒答——预测为 unknown（无裁定）的比例
    """
    verdict_records = [r for r in records if r.expected_verdict]
    n = len(verdict_records)
    if n == 0:
        return {}

    correct = sum(
        1 for r in verdict_records if r.predicted_verdict == r.expected_verdict
    )

    reliable_cases = [r for r in verdict_records if r.expected_verdict == "reliable"]
    scam_cases = [r for r in verdict_records if r.expected_verdict == "likely_scam"]

    false_positives = sum(
        1 for r in reliable_cases
        if r.predicted_verdict in ("suspicious", "likely_scam")
    )
    misses = sum(1 for r in scam_cases if r.predicted_verdict == "reliable")
    refusals = sum(1 for r in verdict_records if r.predicted_verdict == "unknown")

    # 混淆矩阵：expected -> {predicted: count}
    confusion: dict[str, dict[str, int]] = {}
    for r in verdict_records:
        row = confusion.setdefault(r.expected_verdict, {})
        row[r.predicted_verdict or "unknown"] = row.get(r.predicted_verdict or "unknown", 0) + 1

    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n * 100, 1),
        "false_positive_rate": round(false_positives / len(reliable_cases) * 100, 1)
        if reliable_cases else 0.0,
        "false_positive_n": f"{false_positives}/{len(reliable_cases)}",
        "miss_rate": round(misses / len(scam_cases) * 100, 1) if scam_cases else 0.0,
        "miss_n": f"{misses}/{len(scam_cases)}",
        "refusal_rate": round(refusals / n * 100, 1),
        "refusal_n": f"{refusals}/{n}",
        "confusion": confusion,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 关键词召回评分（stage2 简历定向等非裁定型）
# ═══════════════════════════════════════════════════════════════════════════

# 召回 ≥ 该阈值即判该用例 correct（驱动 accuracy 与回归门禁）
KEYWORD_RECALL_THRESHOLD = 0.6


def score_keyword_recall(prediction: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """关键词召回：Agent 输出是否指出了我们预置的「必须命中的差距关键词」

    stage2 的产物是自由文本修改清单，没有可枚举裁定；改用**可判定的代理指标**——
    用例构造时让 JD 要求而简历缺失若干**独特关键词**，正确的定向分析应把这些差距点
    显式指出。逐个关键词做大小写无关的子串匹配，返回 (召回率, 未命中列表)。

    Returns:
        (recall ∈ [0,1], missed_keywords)
    """
    if not expected_keywords:
        return 0.0, []
    pred_lower = (prediction or "").lower()
    missed = [kw for kw in expected_keywords if kw.lower() not in pred_lower]
    hit = len(expected_keywords) - len(missed)
    return hit / len(expected_keywords), missed


def compute_keyword_metrics(records: list["EvalRecord"]) -> dict:
    """关键词召回聚合指标：平均召回率 + 达标率（recall ≥ 阈值）"""
    kw_records = [r for r in records if r.expected_keywords]
    n = len(kw_records)
    if n == 0:
        return {}
    recalls = [r.keyword_recall or 0.0 for r in kw_records]
    passed = sum(1 for r in recalls if r >= KEYWORD_RECALL_THRESHOLD)
    return {
        "total": n,
        "avg_recall": round(statistics.mean(recalls) * 100, 1),
        "pass_rate": round(passed / n * 100, 1),
        "pass_n": f"{passed}/{n}",
        "threshold": KEYWORD_RECALL_THRESHOLD,
    }


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
                # 裁定级用例可只给 expected_verdict，不给 expected_answer
                expected_answer=data.get("expected_answer", data.get("expected_verdict", "")),
                level=data.get("level", 1),
                tags=data.get("tags", []),
                file_path=data.get("file_path"),
                image_path=data.get("image_path"),
                expected_verdict=data.get("expected_verdict"),
                stage=data.get("stage"),
                expected_keywords=data.get("expected_keywords"),
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
                    stage=case.stage,
                )
                elapsed = time.time() - start_time
                prediction = result.answer
                agent_success = result.success
                trials_used = result.trials_used
                reflections = result.reflections
                # trial_details 的步数键是 "steps_used"（此前误用 "steps" → avg_steps 恒 0）
                step_count = sum(
                    d.get("steps_used", 0) for d in result.trial_details
                )
                prompt_tokens = result.total_prompt_tokens       # 评审 3.6：成本指标
                completion_tokens = result.total_completion_tokens
            except Exception as e:
                elapsed = time.time() - start_time
                prediction = ""
                agent_success = False
                trials_used = 0
                reflections = []
                step_count = 0
                prompt_tokens = 0
                completion_tokens = 0
                logger.error("Case %s 执行异常: %s", case.case_id[:8], e)

            # 评分模式三选一：裁定级 / 关键词召回 / GAIA 子串
            predicted_verdict = None
            keyword_recall = None
            if case.expected_verdict:
                predicted_verdict = classify_prediction_verdict(prediction)
                is_correct = predicted_verdict == case.expected_verdict
            elif case.expected_keywords:
                keyword_recall, _missed = score_keyword_recall(prediction, case.expected_keywords)
                is_correct = keyword_recall >= KEYWORD_RECALL_THRESHOLD
            else:
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
                "expected_verdict": case.expected_verdict,
                "predicted_verdict": predicted_verdict,
                "expected_keywords": case.expected_keywords,
                "keyword_recall": round(keyword_recall, 3) if keyword_recall is not None else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            failure_mode = extract_failure_mode(record_dict)
            record_dict["failure_mode"] = failure_mode

            record = EvalRecord(**record_dict)
            records.append(record)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

            status = "✅" if is_correct else "❌"
            fm_str = f" [{failure_mode}]" if failure_mode else ""
            if case.expected_verdict:
                print(f"  {status}{fm_str} Verdict: {predicted_verdict} | GT: {case.expected_verdict}")
            elif case.expected_keywords:
                print(f"  {status}{fm_str} Keyword recall: {keyword_recall:.0%} (need ≥{KEYWORD_RECALL_THRESHOLD:.0%})")
            else:
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
        prompt_tok_list = [r.prompt_tokens for r in records]
        completion_tok_list = [r.completion_tokens for r in records]
        total_tok_list = [r.prompt_tokens + r.completion_tokens for r in records]

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
            avg_prompt_tokens=round(statistics.mean(prompt_tok_list), 0) if prompt_tok_list else 0,
            avg_completion_tokens=round(statistics.mean(completion_tok_list), 0) if completion_tok_list else 0,
            avg_total_tokens=round(statistics.mean(total_tok_list), 0) if total_tok_list else 0,
            verdict_metrics=compute_verdict_metrics(records),
            keyword_metrics=compute_keyword_metrics(records),
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
        if report.avg_total_tokens:
            print(f"  Avg Tokens: {report.avg_total_tokens:.0f} "
                  f"(in {report.avg_prompt_tokens:.0f} / out {report.avg_completion_tokens:.0f}) — 每裁定成本")

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

        vm = report.verdict_metrics
        if vm:
            print(f"\n  Verdict Metrics (n={vm['total']}):")
            print(f"    Accuracy:           {vm['accuracy']:.1f}%  ({vm['correct']}/{vm['total']})")
            print(f"    False-positive rate:{vm['false_positive_rate']:>6.1f}%  ({vm['false_positive_n']}) — 把靠谱判成有坑")
            print(f"    Miss rate:          {vm['miss_rate']:>6.1f}%  ({vm['miss_n']}) — 把诈骗判成靠谱")
            print(f"    Refusal rate:       {vm['refusal_rate']:>6.1f}%  ({vm['refusal_n']}) — 无裁定")
            print(f"\n  Confusion (expected → predicted):")
            for exp in VERDICT_LEVELS:
                row = vm["confusion"].get(exp)
                if row:
                    cells = ", ".join(f"{k}:{v}" for k, v in sorted(row.items()))
                    print(f"    {exp:<14} → {cells}")

        km = report.keyword_metrics
        if km:
            print(f"\n  Keyword-Recall Metrics (n={km['total']}, stage2 定向清单):")
            print(f"    Avg recall:  {km['avg_recall']:.1f}%  — 平均命中预置差距关键词比例")
            print(f"    Pass rate:   {km['pass_rate']:.1f}%  ({km['pass_n']}) — recall ≥ {km['threshold']:.0%} 判达标")

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
    token_list = [r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in records]

    print(f"\n{'═' * 60}")
    print(f"  Analysis: {path}")
    print(f"{'═' * 60}")
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct} ({accuracy:.1f}%)")
    print(f"  Avg Time: {statistics.mean(elapsed_list):.1f}s" if elapsed_list else "")
    print(f"  Avg Trials: {statistics.mean(trials_list):.1f}")
    if any(token_list):
        print(f"  Avg Tokens: {statistics.mean(token_list):.0f} — 每裁定成本")

    if failure_modes:
        print(f"\n  Failure Mode Distribution:")
        failed = total - correct
        for mode, count in failure_modes.most_common():
            pct = count / failed * 100 if failed > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {mode:<20} {count:>3} ({pct:4.0f}%) {bar}")

    # 裁定级指标（从 JSONL dict 重建 EvalRecord 后复用 compute_verdict_metrics）
    verdict_dicts = [r for r in records if r.get("expected_verdict")]
    if verdict_dicts:
        recs = [
            EvalRecord(
                case_id=r.get("case_id", ""), question=r.get("question", ""),
                expected_answer=r.get("expected_answer", ""),
                prediction=r.get("prediction", ""), correct=r.get("correct", False),
                trials_used=r.get("trials_used", 0), elapsed_seconds=r.get("elapsed_seconds", 0),
                expected_verdict=r.get("expected_verdict"),
                predicted_verdict=r.get("predicted_verdict"),
            )
            for r in verdict_dicts
        ]
        vm = compute_verdict_metrics(recs)
        print(f"\n  Verdict Metrics (n={vm['total']}):")
        print(f"    Accuracy:            {vm['accuracy']:.1f}%  ({vm['correct']}/{vm['total']})")
        print(f"    False-positive rate: {vm['false_positive_rate']:.1f}%  ({vm['false_positive_n']})")
        print(f"    Miss rate:           {vm['miss_rate']:.1f}%  ({vm['miss_n']})")
        print(f"    Refusal rate:        {vm['refusal_rate']:.1f}%  ({vm['refusal_n']})")

    keyword_dicts = [r for r in records if r.get("expected_keywords")]
    if keyword_dicts:
        recs = [
            EvalRecord(
                case_id=r.get("case_id", ""), question=r.get("question", ""),
                expected_answer=r.get("expected_answer", ""),
                prediction=r.get("prediction", ""), correct=r.get("correct", False),
                trials_used=r.get("trials_used", 0), elapsed_seconds=r.get("elapsed_seconds", 0),
                expected_keywords=r.get("expected_keywords"),
                keyword_recall=r.get("keyword_recall"),
            )
            for r in keyword_dicts
        ]
        km = compute_keyword_metrics(recs)
        print(f"\n  Keyword-Recall Metrics (n={km['total']}):")
        print(f"    Avg recall:  {km['avg_recall']:.1f}%")
        print(f"    Pass rate:   {km['pass_rate']:.1f}%  ({km['pass_n']}, recall ≥ {km['threshold']:.0%})")

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
        elif suite == "offercheck":
            # OfferCheck 裁定级评测集（求职诈骗/存疑/正常，带 expected_verdict）
            oc_path = Path(__file__).parent.parent / "offercheck" / "eval_suite" / "cases.jsonl"
            if not oc_path.exists():
                print(f"❌ OfferCheck 评测集不存在: {oc_path}")
                sys.exit(1)
            cases = load_custom_suite(str(oc_path))
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
