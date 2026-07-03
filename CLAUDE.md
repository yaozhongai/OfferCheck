# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Nexa Agent is a **scenario-agnostic autonomous-investigation Agent engine** (ReAct + Reflexion + fact-gating + Eval Harness). **OfferCheck** is its first application — a job-offer fraud-investigation agent. Read `SPEC.md` first: it is the project "constitution" (core goals / explicit non-goals / key decisions with rationale). `README.md` describes only what actually exists; `docs/` holds design deep-dives (paradigm/hallucination research, failure analyses).

## Environment (critical, non-obvious)

**Always run under conda env `agent` (Python 3.10).** Base miniconda is 3.9, where the FastAPI server fails to import (pydantic can't evaluate PEP-604 `int | None` in route signatures). Every python/uvicorn/pytest command must be preceded by:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent
```

`pytest` lives in base env, not `agent` — run tests from base, or install pytest into `agent`. Tests do not import `server.main`, so the 3.9 issue doesn't affect them.

## Commands

```bash
# CLI — run the engine headless (no server needed)
python -m nexa_agent.reflexion_agent "<task>" --stage stage4   # stage1|stage4; omit --stage for generic
python -m nexa_agent.reflexion_agent --file q.txt --max-trials 1 --max-steps 12
make agent Q="..." STAGE=stage4                                 # Makefile wrapper

# Full stack
python -m uvicorn server.main:app --port 8000 --reload          # backend (--reload matters: stale process = old code)
cd web && npm install && npm run dev                            # Next.js frontend :3000 (proxies /api/v0/* to :8000)

# Eval Harness (GAIA parquet not in repo; use custom JSONL suite)
python -m nexa_agent.eval_harness run --suite path/to/cases.jsonl
python -m nexa_agent.eval_harness analyze --input results/eval_xxx.jsonl
python -m nexa_agent.eval_harness compare --baseline A.jsonl --current B.jsonl

# Tests (base env has pytest)
python -m pytest tests/ -q
python -m pytest tests/gmi_api_test.py::test_gmi_chat_completion_smoke -v   # single test; network tests skip without GMI_API_KEY
python -m pytest -m "not network"                              # skip live-API tests
```

## Architecture (the big picture)

Three layers in one repo — dependency direction is **server/offercheck → nexa_agent** (never the reverse):

- **`nexa_agent/`** — the engine. Headless, importable, no FastAPI/DB/UI. `react_agent.py` (ReAct inner loop, native function calling) → `reflexion_agent.py` (`ReflexionReActAgent.execute()`, the Trial→Evaluate→Verify→Reflect outer loop) → `evaluator.py` / `verifier.py` / `memory.py` (Reflexion episodic) / `tools.py` (12 tools) / `eval_harness.py` / `search/` (pluggable providers) / `llm/` (multi-provider client) / `trace/` (event schema) / `config.py` / `logger.py`.
- **`offercheck/`** — scenario layer (currently skeleton). Stages are **not** separate code paths: the same engine gets a different `--stage` prompt (`nexa_agent/prompts/offercheck_stage{1,4}.txt`) appended after the generic `react_system.txt`.
- **`server/`** — thin FastAPI. `/api/v0/run_stage` and `/run_stage/stream` (SSE) forward to `ReflexionReActAgent.execute()`. `chat`/`upload` endpoints return 501 (legacy, pre-refactor). `server/memory/` = STM/LTM product memory; `server/trace_store/` = persistence + SSE; `server/persistence/` = SQLAlchemy.
- **`web/`** — Next.js App Router; `page.tsx` streams the SSE trace and renders a verdict card.

### Model routing (single source of truth)

`nexa_agent/config.py` `MODEL_TIER` + `MODEL_ROUTING` is the **only** place models are chosen. Tiers: `strong` (first-step planning / verifier), `fast` (subsequent steps / reflection / eval), `upgrade` (tool-call fallback via dynamic upgrade when N consecutive steps emit no tool_calls). `get_model_for_role(role)` resolves a role to a tier to a model. `server/config.py` also has legacy LLM fields — do **not** add model config there.

### GMI + non-thinking-model constraints (easy to break)

The project runs on GMI Cloud with **non-thinking (instruct) models** (`Qwen3-235B-A22B-Instruct`, `DeepSeek-V3.2`, `Kimi-K2-Instruct`). Two hard rules learned from GMI 400/422 failures:

1. `extra_body={"thinking": ...}` must always be guarded by `if SUPPORTS_THINKING_PARAM and ...` — GMI 422s on the `thinking` param. `SUPPORTS_THINKING_PARAM` is `False` for GMI. (Multiple past 422 bugs came from unguarded sites in evaluator/reflexion.)
2. When appending an assistant message that has tool_calls, use `_assistant_msg_to_dict(msg)` (react_agent.py) — DeepSeek reasoning models require `reasoning_content` to be passed back or the next turn 400s.

### Grounding layer (anti-hallucination — do not weaken casually)

The engine enforces "no evidence, no answer" through four layers (see `docs/run_20260702_234927_failure_analysis.md`, `docs/agent_paradigms_and_hallucination.md`):

- **Grounding rules** at the top of `react_system.txt` (highest priority prompt section).
- **Evidence gate** in `react_loop`: verdict/fact-type answers (`_answer_requires_evidence`) with zero successful retrieval are blocked (`MAX_EVIDENCE_GATE_NAGS` times) and re-prompted to investigate.
- **Termination**: native convention — a text response with **no tool_calls** is the final answer; the literal `Final Answer:` sentinel is legacy and no longer required (kept only for back-compat). `submit_verdict` is a structured terminal tool (defined in `get_openai_tool_definitions`, intercepted in `react_loop`, not a normal executed tool) that carries the verdict schema and is the natural home for the gate + attribution.
- **AIS source attribution** (`attribute_sources`): `[Source]` lines are cross-checked against `seen_urls` (URLs actually observed) and called tools; unmatched sources are annotated `⚠️[未验证]`. The annotated answer flows through `react_result["answer"]`.

### Trace emission

`react_loop`/`execute` take an `on_event(dict)` callback; structured events (trial_start / step_start / action / observation / evidence_gate / final_answer …) are emitted at existing instrumentation points. The server's SSE endpoint bridges these to the browser via a background thread + `queue.Queue`.

## Git

Remote uses SSH via `ssh://git@github.com/...` form (a global `insteadOf` rule rewrites `git@github.com:` back to HTTPS, which breaks push; the `ssh://` long form bypasses it). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
