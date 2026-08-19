"""Blind semantic judge for GTS scoring — LLM-based semantic equivalence evaluation.

Pre-registration commitment #4: the judge must be calibrated against ≥60 human-labeled
samples before the 240-trial main evaluation can start.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from nexa_agent.config import get_model_for_role
from nexa_agent.llm_gateway import complete as llm_complete


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluation judge. Your task is to determine whether a model's answer is semantically correct according to a provided rubric and set of expected key facts.

## Rules
1. You are BLIND — you do NOT know which system produced the answer. Judge the answer purely on its content.
2. The rubric describes what a correct answer must contain. The expected key facts are the minimum facts that must be present (in substance, not exact wording).
3. Be strict about factual errors: if the answer states a wrong fact, it fails.
4. Be flexible about wording: synonyms, paraphrases, and alternative grammatical forms that preserve meaning are acceptable.
5. For negative-evidence tasks (where the correct answer is "cannot be determined" / "unknown" / "not recorded"), the answer must clearly state the information is unavailable AND cite an authoritative source explaining why. An answer that merely says "I don't know" without evidence fails.
6. For computed-value tasks, the numeric result must be within 1% of the expected value (rounding errors are acceptable).
7. The answer must address ALL parts of the question. Missing parts = fail.
8. If the answer contains hallucinations (claims not supported by the expected facts or rubric), it fails.

## Output Format
Respond with a JSON object ONLY (no markdown, no extra text):
{"pass": true/false, "confidence": 0.0-1.0, "reasoning": "one-sentence explanation of why it passed or failed"}
"""

JUDGE_USER_TEMPLATE = """## Task Question
{question}

## Evaluation Rubric
{rubric}

## Expected Key Facts
{expected_keywords}

## Model's Answer
{answer}

Judge the answer against the rubric and expected facts. Output ONLY the JSON."""


ENTAILMENT_SYSTEM_PROMPT = """You are an impartial evidence-entailment judge. Your task is to determine whether a specific CLAIM made in a model's answer is genuinely supported by the EVIDENCE EXCERPT retrieved from the model's declared source.

## Rules
1. Judge ONLY whether the evidence excerpt supports the claim — not whether the claim is true in general.
2. The excerpt must contain information that directly entails the claim. Topical relatedness is NOT support.
3. FAIL if the claim contains specific facts (dates, numbers, names) absent from or contradicted by the excerpt.
4. FAIL if the claim overgeneralizes the excerpt (e.g., excerpt says "not recorded" but claim asserts a specific value).
5. For negative-evidence tasks (correct answer = "cannot be determined / not publicly available"): PASS only if the excerpt supports the unavailability (e.g., states records are sealed, search was unsuccessful, or no record exists) — AND the claim does not escalate "not found" into "does not exist".
6. FAIL on misattribution: the claim cites the source but the excerpt comes from a different entity's page without support.
7. If the excerpt is irrelevant boilerplate (navigation, ads, JS challenge pages), FAIL.

## Output Format
Respond with a JSON object ONLY (no markdown, no extra text):
{"pass": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence citing the decisive excerpt content or its absence"}
"""

ENTAILMENT_USER_TEMPLATE = """## Original Task Question
{question}

## Claim To Verify
{claim}

## Declared Source(s)
{source_refs}

## Evidence Excerpt (frozen content actually retrieved from the declared source)
{evidence_excerpt}

## Task Rubric
{rubric}

## Negative-Evidence Task
{is_negative_task}

Does the evidence excerpt genuinely support the claim? Output ONLY the JSON."""


@dataclass
class JudgeVerdict:
    pass_: bool
    confidence: float
    reasoning: str


class JudgeParseError(ValueError):
    """Judge response did not satisfy the frozen JSON verdict contract."""


@dataclass
class CalibrationSample:
    """A single labeled sample for judge calibration."""
    sample_id: str
    question: str
    rubric: str
    expected_keywords: list[str]
    answer: str
    human_label: bool  # True = semantically correct
    label_note: str = ""  # optional human annotation


@dataclass
class CalibrationResult:
    total: int
    correct: int
    accuracy: float
    false_positives: int
    false_negatives: int
    mean_confidence_correct: float
    mean_confidence_wrong: float
    per_sample: list[dict[str, Any]] = field(default_factory=list)


class SemanticJudge:
    """LLM-based blind semantic judge for answer evaluation."""

    def __init__(self, model: Optional[str] = None):
        self._model = model

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = get_model_for_role("verifier")
        return self._model

    def judge(
        self,
        question: str,
        rubric: str,
        expected_keywords: list[str],
        answer: str,
    ) -> JudgeVerdict:
        """Evaluate whether an answer is semantically correct."""
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            rubric=rubric,
            expected_keywords=", ".join(expected_keywords),
            answer=answer if answer.strip() else "(empty answer)",
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        return self._complete_strict(messages, "semantic_judge")

    def judge_claim_entailment(
        self,
        question: str,
        claim: str,
        evidence_excerpt: str,
        source_refs: list[str],
        rubric: str,
        is_negative_task: bool = False,
    ) -> JudgeVerdict:
        """判定「声明来源的冻结正文片段」是否真正蕴含该 claim（protocol-r2 §C）。

        与 judge() 的区别：judge() 只看最终答案对不对；本方法检查答案中的
        claim 是否被其声明来源的正文真正支持——治「正确事实绑定错误来源」、
        「断章取义」与「越界推断」。
        """
        user_msg = ENTAILMENT_USER_TEMPLATE.format(
            question=question,
            claim=claim,
            source_refs="\n".join(f"- {ref}" for ref in source_refs) or "(none)",
            evidence_excerpt=evidence_excerpt if evidence_excerpt.strip() else "(empty excerpt)",
            rubric=rubric or "(no rubric)",
            is_negative_task="yes" if is_negative_task else "no",
        )
        messages = [
            {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self._complete_strict(messages, "semantic_judge_entailment")

    def _complete_strict(
        self, messages: list[dict[str, str]], usage_role: str
    ) -> JudgeVerdict:
        errors: list[str] = []
        for _ in range(3):
            result = llm_complete(
                messages=messages,
                role="verifier",
                usage_role=usage_role,
                temperature=0.0,
                max_tokens=512,
            )
            try:
                return self._parse_response(result.content)
            except JudgeParseError as exc:
                errors.append(str(exc))
        raise JudgeParseError(
            f"judge returned invalid JSON after 3 attempts: {'; '.join(errors)}"
        )

    def calibrate(self, samples: list[CalibrationSample]) -> CalibrationResult:
        """Run judge on a labeled calibration set and compute metrics."""
        correct = 0
        fp = 0
        fn_count = 0
        conf_correct: list[float] = []
        conf_wrong: list[float] = []
        per_sample: list[dict[str, Any]] = []

        for s in samples:
            verdict = self.judge(
                question=s.question,
                rubric=s.rubric,
                expected_keywords=s.expected_keywords,
                answer=s.answer,
            )
            match = verdict.pass_ == s.human_label
            if match:
                correct += 1
                conf_correct.append(verdict.confidence)
            else:
                if verdict.pass_ and not s.human_label:
                    fp += 1
                else:
                    fn_count += 1
                conf_wrong.append(verdict.confidence)

            per_sample.append({
                "sample_id": s.sample_id,
                "judge_pass": verdict.pass_,
                "human_label": s.human_label,
                "match": match,
                "judge_confidence": verdict.confidence,
                "judge_reasoning": verdict.reasoning,
                "label_note": s.label_note,
            })

        return CalibrationResult(
            total=len(samples),
            correct=correct,
            accuracy=correct / len(samples) if samples else 0.0,
            false_positives=fp,
            false_negatives=fn_count,
            mean_confidence_correct=(
                sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
            ),
            mean_confidence_wrong=(
                sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0
            ),
            per_sample=per_sample,
        )

    @staticmethod
    @staticmethod
    def _parse_response(raw: str) -> JudgeVerdict:
        """Extract a typed JSON verdict; invalid output is never guessed."""
        # Try multiple JSON extraction strategies
        obj = None

        # Strategy 1: find outermost { } containing "pass"
        for m in re.finditer(r'\{', raw):
            depth, start = 1, m.start()
            for i in range(start + 1, len(raw)):
                if raw[i] == '{': depth += 1
                elif raw[i] == '}': depth -= 1
                if depth == 0:
                    candidate = raw[start:i+1]
                    if '"pass"' in candidate:
                        try:
                            obj = json.loads(candidate)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
                    break
            if obj is not None:
                break

        # Strategy 2: simple flat JSON regex (backup)
        if obj is None:
            m = re.search(r'\{"pass"\s*:\s*(?:true|false)[^}]*\}', raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        if obj is not None and isinstance(obj.get("pass"), bool):
            confidence = obj.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise JudgeParseError("confidence must be numeric within [0, 1]")
            if not isinstance(obj.get("reasoning"), str):
                raise JudgeParseError("reasoning must be a string")
            return JudgeVerdict(
                pass_=obj["pass"],
                confidence=float(confidence),
                reasoning=obj["reasoning"],
            )
        raise JudgeParseError(f"invalid judge JSON: {raw[:200]}")


# ---- Calibration Set Generation ----

def generate_calibration_from_suites(
    suite_paths: list[str],
    samples_per_task: int = 3,
) -> list[CalibrationSample]:
    """Auto-generate a diverse calibration set from suite tasks.

    For each task, generates:
    - 1 positive sample (correct answer formed from expected keywords)
    - 1 negative sample (wrong facts)
    - 1 edge/boundary sample (correct substance, different wording, or partial)
    """
    import json as _json
    import random as _random

    _random.seed(20260724)

    samples: list[CalibrationSample] = []
    variant_idx = 0

    for suite_path in suite_paths:
        with open(suite_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = _json.loads(line)

                tid = task["task_id"]
                question = task["prompt"]
                rubric = (
                    task.get("metadata", {})
                    .get("eval_card", {})
                    .get("semantic_rubric", "")
                )
                keywords = [
                    str(k) for k in task.get("metadata", {}).get("expected_keywords", [])
                ]
                is_negative = "negative" in str(task.get("tags", []))
                is_computed = any(
                    tag in str(task.get("tags", []))
                    for tag in ["file_structured", "api"]
                )

                if not rubric:
                    rubric = f"Answer must contain these key facts: {', '.join(keywords)}"

                # 1. Positive sample: well-formed correct answer
                positive_answer = _make_positive_answer(keywords, is_negative, is_computed)
                samples.append(CalibrationSample(
                    sample_id=f"{tid}_pos",
                    question=question,
                    rubric=rubric,
                    expected_keywords=keywords,
                    answer=positive_answer,
                    human_label=True,
                    label_note="Correct answer formed from expected keywords",
                ))
                variant_idx += 1

                # 2. Negative sample: wrong facts
                negative_answer = _make_negative_answer(keywords, is_negative)
                samples.append(CalibrationSample(
                    sample_id=f"{tid}_neg",
                    question=question,
                    rubric=rubric,
                    expected_keywords=keywords,
                    answer=negative_answer,
                    human_label=False,
                    label_note="Answer with wrong or contradictory facts",
                ))
                variant_idx += 1

                # 3. Edge case: semantically correct but different wording
                edge_answer = _make_edge_answer(keywords, is_negative)
                samples.append(CalibrationSample(
                    sample_id=f"{tid}_edge",
                    question=question,
                    rubric=rubric,
                    expected_keywords=keywords,
                    answer=edge_answer,
                    human_label=True,
                    label_note="Semantically equivalent but different wording / structure",
                ))
                variant_idx += 1

    _random.shuffle(samples)
    return samples


def _make_positive_answer(keywords: list[str], is_negative: bool, is_computed: bool) -> str:
    if is_negative:
        return (
            f"Based on authoritative sources, the information cannot be determined. "
            f"The relevant records show: {'; '.join(keywords)}."
        )
    return f"The answer is: {' '.join(keywords)}."


def _make_negative_answer(keywords: list[str], is_negative: bool) -> str:
    import random as _random
    if is_negative:
        # For negative-evidence tasks, wrong = claiming to know the answer
        return "The answer is: [made-up specific detail that contradicts the source]."
    if not keywords:
        return "I don't know."
    # Mutate one keyword to be wrong
    kw_list = list(keywords)
    if len(kw_list) >= 2:
        kw_list[0] = f"[WRONG: {kw_list[0]}]"
    else:
        kw_list[0] = "incorrect_value"
    return f"The answer might be: {' '.join(kw_list)}."


def _make_edge_answer(keywords: list[str], is_negative: bool) -> str:
    if is_negative:
        return (
            "After thorough investigation of the relevant official sources, "
            "I found that the requested information is not available in public records. "
            f"Specifically: {'; '.join(keywords)}. Therefore, the exact details cannot be provided."
        )
    if not keywords:
        return "The available evidence is insufficient to answer definitively."
    # Rephrase: embed keywords in a verbose, natural-sounding sentence
    kw_str = " ".join(keywords)
    return (
        f"According to my research, the relevant facts indicate that {kw_str}. "
        "These findings are confirmed by the cited authoritative sources."
    )
