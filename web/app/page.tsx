"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { flushSync } from "react-dom";
import type { FormData } from "./ui";
import {
  Stage, EngineEvent, StepStatus, TraceItem, RunState, StageState,
  STAGE_META, UI, tm, TOOL_LEGEND, summarizeArgs, withEngineSources,
  parseStructuredAnswer, buildFollowupInput, buildInput, valid, summary, inputStyle,
  buildCrossStageContext, completedEarlierStages, ConvTurn,
  InlineTrace, ChatSummary, StructuredResult, StageForm,
  apiUrl, DEMO_FORMS, STAGE_FIELDS,
} from "./ui";

// ─── Main page ────────────────────────────────────────────────────────────────

const EMPTY_RUN: RunState = { status: "idle", trace: [] };
const EMPTY_STAGE_STATE = (): StageState => ({ initialRun: { ...EMPTY_RUN }, followups: [], chatInput: "" });
const EMPTY_STAGE_STATES = (): Record<Stage, StageState> => ({
  stage1: EMPTY_STAGE_STATE(), stage2: EMPTY_STAGE_STATE(),
  stage3: EMPTY_STAGE_STATE(), stage4: EMPTY_STAGE_STATE(),
});
const EMPTY_FORMS: FormData = { company: "", position: "", resume: "", context: "", jd: "", target: "", message: "", recruiter: "", offer: "", hr: "", link: "" };

// A Case = one job opportunity (a company / role) followed across all four stages.
// The engine is fed different inputs per stage; a Case is the container for that
// opportunity's forms, per-stage runs, and cross-stage memory.
type Case = {
  id: string;
  name: string;         // company name, or a placeholder until the user renames / starts stage 1
  createdAt: number;
  activeStage: Stage;
  forms: FormData;
  stageStates: Record<Stage, StageState>;
};

const PLACEHOLDER_NAMES = new Set(["New case", "Untitled case"]);
// Deterministic default so SSR and first client render match (no Math.random / Date.now in initial state).
const DEFAULT_CASE = (): Case => ({
  id: "case-1", name: "New case", createdAt: 0, activeStage: "stage1",
  forms: { ...EMPTY_FORMS }, stageStates: EMPTY_STAGE_STATES(),
});
function makeCase(): Case {
  return {
    id: "case-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    name: "Untitled case", createdAt: Date.now(), activeStage: "stage1",
    forms: { ...EMPTY_FORMS }, stageStates: EMPTY_STAGE_STATES(),
  };
}

const LS_KEY = "offercheck.cases.v1";

// On hydrate, in-flight runs are stale — coerce them so the UI isn't stuck "running".
function sanitizeRun(r: RunState): RunState {
  if (r.status !== "running") return r;
  return { ...r, status: r.answer ? "done" : "error",
    errorMsg: r.answer ? r.errorMsg : "Interrupted — please re-run",
    trace: r.trace.map(t => t.status === "running" ? { ...t, status: "error" as StepStatus } : t) };
}
function sanitizeCase(c: Case): Case {
  const ss: Record<Stage, StageState> = { ...c.stageStates };
  (Object.keys(ss) as Stage[]).forEach(s => {
    ss[s] = { ...ss[s], initialRun: sanitizeRun(ss[s].initialRun),
      followups: ss[s].followups.map(f => ({ ...f, runState: sanitizeRun(f.runState) })) };
  });
  return { ...c, stageStates: ss };
}
// Strip heavy per-step text when persisting under quota pressure (keeps summary counts + labels).
function lightenCase(c: Case): Case {
  const strip = (t: TraceItem): TraceItem => ({ ...t, observation: undefined, args: undefined, thought: undefined });
  const lightRun = (r: RunState): RunState => ({ ...r, trace: r.trace.map(strip) });
  const ss: Record<Stage, StageState> = { ...c.stageStates };
  (Object.keys(ss) as Stage[]).forEach(s => {
    ss[s] = { ...ss[s], initialRun: lightRun(ss[s].initialRun),
      followups: ss[s].followups.map(f => ({ ...f, runState: lightRun(f.runState) })) };
  });
  return { ...c, stageStates: ss };
}

let _idSeq = 0;
function nextId() { return ++_idSeq; }

export default function Home() {
  const [cases, setCases] = useState<Case[]>(() => [DEFAULT_CASE()]);
  const [activeCaseId, setActiveCaseId] = useState<string>("case-1");
  const [hydrated, setHydrated] = useState(false);
  // 默认布局用纯 CSS flex 比例（56:44）驱动——服务端/客户端渲染一致，首帧即最终布局，
  // 零闪烁。仅当用户手动拖拽分隔条后才切到像素宽度模式（userResized=true）。
  const [chatWidth, setChatWidth] = useState(560);
  const [userResized, setUserResized] = useState(false);
  // Start composer: guided-fields mode vs free-text ("skip, just type") mode.
  const [rawMode, setRawMode] = useState(false);
  const [rawInput, setRawInput] = useState("");
  // Manual collapse of the Evidence Board (user can hide it to widen the chat).
  const [boardCollapsed, setBoardCollapsed] = useState(false);
  // Attached screenshot/PDF for the next submission (uploaded to the server).
  const [attachedImage, setAttachedImage] = useState<{ path: string; name: string; preview: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function uploadImage(file: File) {
    if (!/\.(png|jpe?g|webp|gif|pdf)$/i.test(file.name)) {
      window.alert("Only images (PNG/JPG/WebP/GIF) or PDF are supported.");
      return;
    }
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(apiUrl("/api/v0/upload"), { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
      const data = await resp.json();
      setAttachedImage({ path: data.file_path, name: data.filename || file.name, preview: URL.createObjectURL(file) });
    } catch (e) {
      window.alert("Upload failed: " + String(e));
    } finally {
      setUploading(false);
    }
  }
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);

  // ─── Persistence: hydrate once on mount, then save on change ──────
  const activeCaseIdRef = useRef(activeCaseId);
  useEffect(() => { activeCaseIdRef.current = activeCaseId; }, [activeCaseId]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as { cases: Case[]; activeCaseId: string };
        if (parsed.cases?.length) {
          const clean = parsed.cases.map(sanitizeCase);
          setCases(clean);
          setActiveCaseId(clean.some(c => c.id === parsed.activeCaseId) ? parsed.activeCaseId : clean[0].id);
        }
      }
    } catch { /* corrupt storage — ignore, keep default */ }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return; // don't overwrite storage before we've loaded it
    const payload = { cases, activeCaseId };
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(payload));
    } catch {
      // Quota exceeded — retry with trace detail stripped.
      try { localStorage.setItem(LS_KEY, JSON.stringify({ cases: cases.map(lightenCase), activeCaseId })); }
      catch { /* give up silently */ }
    }
  }, [cases, activeCaseId, hydrated]);

  // ─── Active-case derived state + stable updater ──────────────────
  const activeCase = cases.find(c => c.id === activeCaseId) ?? cases[0];
  const stage = activeCase.activeStage;
  const forms = activeCase.forms;
  const stageStates = activeCase.stageStates;

  const ss = stageStates[stage];
  const meta = STAGE_META[stage];
  const formValid = valid(stage, forms);
  // An attached screenshot alone is enough to start (e.g. just upload an offer image).
  const startValid = (rawMode ? rawInput.trim().length > 0 : formValid) || !!attachedImage;

  // The "active" run: the most recent followup if any, else initialRun
  const activeRun: RunState = ss.followups.length > 0
    ? ss.followups[ss.followups.length - 1].runState
    : ss.initialRun;

  const isInvestigating = activeRun.status === "running";
  const hasStarted = ss.initialRun.status !== "idle";

  // Stable patcher for the active case (reads id from ref to avoid stale closures).
  const patchActiveCase = useCallback((fn: (c: Case) => Case) => {
    setCases(prev => prev.map(c => c.id === activeCaseIdRef.current ? fn(c) : c));
  }, []);
  const patchStageStates = useCallback((upd: (prev: Record<Stage, StageState>) => Record<Stage, StageState>) => {
    patchActiveCase(c => ({ ...c, stageStates: upd(c.stageStates) }));
  }, [patchActiveCase]);

  const setStage = useCallback((s: Stage) => {
    patchActiveCase(c => ({ ...c, activeStage: s }));
  }, [patchActiveCase]);

  // Update a stage's initial run (within the active case)
  const setInitialRun = useCallback((s: Stage, upd: Partial<RunState> | ((prev: RunState) => RunState)) => {
    patchStageStates(prev => {
      const cur = prev[s];
      const next = typeof upd === "function" ? upd(cur.initialRun) : { ...cur.initialRun, ...upd };
      return { ...prev, [s]: { ...cur, initialRun: next } };
    });
  }, [patchStageStates]);

  // Update the last followup's run state (within the active case)
  const setLastFollowupRun = useCallback((s: Stage, upd: Partial<RunState> | ((prev: RunState) => RunState)) => {
    patchStageStates(prev => {
      const cur = prev[s];
      if (cur.followups.length === 0) return prev;
      const followups = [...cur.followups];
      const last = followups[followups.length - 1];
      const nextRun = typeof upd === "function" ? upd(last.runState) : { ...last.runState, ...upd };
      followups[followups.length - 1] = { ...last, runState: nextRun };
      return { ...prev, [s]: { ...cur, followups } };
    });
  }, [patchStageStates]);

  // ─── Case management ─────────────────────────────────────────────
  function createCase() {
    const c = makeCase();
    setCases(prev => [...prev, c]);
    setActiveCaseId(c.id);
    setRawMode(false);
    setRawInput("");
  }
  function selectCase(id: string) {
    setActiveCaseId(id);
    setRawMode(false);
    setRawInput("");
  }
  function renameCase(id: string) {
    const cur = cases.find(c => c.id === id);
    const next = window.prompt("Rename case", cur?.name ?? "")?.trim();
    if (next) setCases(prev => prev.map(c => c.id === id ? { ...c, name: next } : c));
  }
  function deleteCase(id: string) {
    setCases(prev => {
      const rest = prev.filter(c => c.id !== id);
      const next = rest.length ? rest : [makeCase()];
      if (id === activeCaseIdRef.current) setActiveCaseId(next[0].id);
      return next;
    });
  }

  // ─── Resize ──────────────────────────────────────────────────────

  const resizing = useRef(false);
  const resizeStart = useRef({ x: 0, w: 0 });

  function onResizeDown(e: React.MouseEvent) {
    e.preventDefault();
    resizing.current = true;
    // flex 模式下 chatWidth 不是真实宽度——从实际渲染宽度起拖，避免跳变
    const startW = chatPanelRef.current?.offsetWidth ?? chatWidth;
    resizeStart.current = { x: e.clientX, w: startW };
    if (!userResized) setUserResized(true);
    setChatWidth(startW);
    const mv = (ev: MouseEvent) => {
      if (!resizing.current) return;
      const maxW = window.innerWidth - 232 - 6 - 380; // 画布至少留 380
      setChatWidth(Math.max(320, Math.min(maxW, resizeStart.current.w + (ev.clientX - resizeStart.current.x))));
    };
    const up = () => { resizing.current = false; window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", mv);
    window.addEventListener("mouseup", up);
  }

  function updateForm(k: string, v: string) {
    patchActiveCase(c => ({ ...c, forms: { ...c.forms, [k]: v } }));
  }
  // Fill the current stage's form with a ready-made demo case (one-click try).
  function fillDemo(s: Stage) {
    setRawMode(false);
    patchActiveCase(c => ({ ...c, forms: { ...c.forms, ...DEMO_FORMS[s] } }));
  }

  // ─── Event reducer (pure) ─────────────────────────────────────────

  function reduceRun(prev: RunState, evt: EngineEvent): RunState {
    switch (evt.type) {
      case "started":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "step_start",
            metaDetail: `Engine started · ${evt.stage ?? stage}` }] };

      case "trial_start":
        return { ...prev, currentTrial: evt.trial, maxTrials: evt.max_trials,
          trace: [...prev.trace, { id: nextId(), type: "meta", metaKind: "trial",
            metaDetail: `${evt.trial}/${evt.max_trials}` }] };

      case "step_start":
        return { ...prev, maxSteps: evt.max_steps ?? prev.maxSteps,
          currentStep: evt.step ?? prev.currentStep, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "step_start",
            metaDetail: `${evt.model ?? "unknown"} · Step ${evt.step}/${evt.max_steps}` }] };

      case "action": {
        if (!evt.tool) return prev;
        const m = tm(evt.tool);
        // Summary from args (query / url / domain) — stable every step, unlike the
        // model's optional `thought`. Prefer the engine-provided `query` if present.
        const brief = (evt.query || summarizeArgs(evt.tool, evt.args) || "").slice(0, 160);
        return { ...prev, trace: [...prev.trace, {
          id: nextId(), type: "step", stepNum: evt.step, tool: evt.tool,
          label: brief ? `${m.label}: ${brief}` : m.label, args: evt.args,
          status: "running" as StepStatus, brief, model: evt.model,
        }] };
      }

      case "observation":
        return { ...prev, trace: prev.trace.map(t =>
          t.type === "step" && t.status === "running"
            ? { ...t, status: (evt.ok === false ? "notfound" : "success") as StepStatus,
                observation: evt.observation?.slice(0, 1500) ?? "" }
            : t) };

      case "retry":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "warn", metaKind: "retry",
            metaDetail: `Retrying LLM call (${evt.attempt ?? "?"})… ${evt.error ?? ""}`.trim() }] };

      case "stage_routed": {
        // Followup auto-routed to another stage capability (multi-capability chat).
        const to = evt.to_stage as Stage | undefined;
        const label = to && STAGE_META[to] ? `${STAGE_META[to].num} ${STAGE_META[to].en}` : evt.to_stage;
        return { ...prev, routedStage: to, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "step_start",
            metaDetail: `↪ Routed to ${label} capability` }] };
      }

      case "answer_delta": {
        // Accumulate streamed answer tokens; reset when the step changes so we
        // only ever hold the latest step's text (guards against pre-tool leakage).
        const sameStep = prev.streamStep === evt.step;
        return { ...prev, streamStep: evt.step,
          streamedAnswer: (sameStep ? (prev.streamedAnswer ?? "") : "") + (evt.text ?? "") };
      }

      case "evidence_gate":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "warn", metaKind: "evidence_gate",
            metaDetail: evt.reason ?? "evidence required" }] };

      case "correction":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "warn", metaKind: "correction",
            metaDetail: evt.message ?? "" }] };

      case "verifier_start":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "verifier",
            metaDetail: "Verifying facts…", ok: undefined }] };

      case "verifier_result":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "verifier",
            metaDetail: evt.reason ?? (evt.passed ? "passed" : "rejected"),
            ok: evt.passed ? true : false }] };

      case "trial_evaluated":
        return { ...prev, trace: [...prev.trace,
          { id: nextId(), type: "meta", metaKind: "trial",
            metaDetail: `${evt.success ? "✓" : "✗"} Trial ${evt.trial} — ${evt.reason ?? ""}` }] };

      case "final_answer":
        return { ...prev, answer: evt.answer,
          sources: evt.sources ?? prev.sources,
          summaryForUser: evt.summary_for_user ?? prev.summaryForUser,
          suggestedFollowups: evt.suggested_followups ?? prev.suggestedFollowups,
          streamedAnswer: undefined,  // authoritative answer takes over the live stream
          trace: prev.trace.map(t => t.status === "running" ? { ...t, status: "success" as const } : t) };

      case "done": {
        // A produced answer is worth showing even if the Verifier wasn't fully
        // satisfied — `success:false` is a quality caveat (surfaced via ⚠️ on
        // sources), not a hard failure. Only treat as error when there's no answer.
        const hasAnswer = !!(evt.answer ?? prev.answer);
        return { ...prev,
          status: hasAnswer ? "done" : (evt.success === false ? "error" : "done") as RunState["status"],
          answer: evt.answer ?? prev.answer,
          sources: evt.sources ?? prev.sources,
          success: evt.success, latency_ms: evt.latency_ms, trials_used: evt.trials_used,
          trace: prev.trace.map(t => t.status === "running" ? { ...t, status: "success" as const } : t) };
      }

      case "error":
        return { ...prev, status: "error", errorMsg: evt.message ?? "Unknown error",
          trace: prev.trace.map(t => t.status === "running" ? { ...t, status: "error" as const } : t) };
    }
    return prev;
  }

  // ─── SSE runner (shared for initial + followup) ───────────────────

  async function runSSE(
    input: string,
    currentStage: Stage,
    isFollowup: boolean,
    imagePath?: string,
    requestStage?: Stage,  // sticky-routed capability; UI state stays keyed to currentStage
  ) {
    const setRun = isFollowup ? setLastFollowupRun : setInitialRun;

    try {
      const resp = await fetch(apiUrl("/api/v0/run_stage/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          stage: requestStage ?? currentStage,
          max_trials: isFollowup ? 1 : 2,
          max_steps: isFollowup ? 8 : 12,
          answer_mode: isFollowup,  // followups may answer conversationally + stream
          auto_route: isFollowup,   // followups may switch to another stage capability
          image_path: imagePath,    // attached screenshot/PDF (engine OCRs via analyze_image)
        }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done: sd, value } = await reader.read();
        if (sd) {
          flushSync(() => {
            setRun(currentStage, prev => ({
              ...prev,
              status: prev.status === "running" ? "done" : prev.status,
              trace: prev.trace.map(t => t.status === "running" ? { ...t, status: "success" as const } : t),
            }));
          });
          break;
        }

        buf += decoder.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop() ?? "";

        for (const c of chunks) {
          const line = c.trim();
          if (!line.startsWith("data:")) continue;
          let evt: EngineEvent;
          try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }

          // Process each event individually to maximise streaming granularity
          flushSync(() => {
            setRun(currentStage, prev => reduceRun(prev, evt));
          });

          if (evt.type === "error") break;
        }
      }
    } catch (err) {
      flushSync(() => {
        setRun(currentStage, prev => ({
          ...prev, status: "error", errorMsg: String(err),
          trace: prev.trace.map(t => t.status === "running" ? { ...t, status: "error" as const } : t),
        }));
      });
    }
  }

  // ─── Start initial investigation ─────────────────────────────────

  async function startInitial() {
    let input = rawMode ? rawInput.trim() : buildInput(stage, forms);
    if (!input.trim() && attachedImage) input = "请分析我上传的这张截图/图片，并据此展开调查。";
    if (!input.trim()) return;

    // Cross-stage carryover: pack completed earlier stages' findings into the task
    // so this run starts with the case's accumulated context (real wiring for the
    // "findings are carried over" promise — the engine still re-verifies evidence).
    const crossCtx = buildCrossStageContext(stageStates, stage);
    if (crossCtx) input = `${crossCtx}\n\n[本阶段任务]\n${input}`;
    _idSeq = 0;
    setBoardCollapsed(false);  // a fresh result should surface the board

    const label = rawMode
      ? (rawInput.trim().length > 160 ? rawInput.trim().slice(0, 160) + "…" : rawInput.trim())
      : summary(stage, forms);

    // Derive a case name from the opportunity if it's still a placeholder.
    // Prefer the company; otherwise use the first line of this stage's primary
    // field (JD / recruiter message / offer) so non-Research stages get a title too.
    const firstLine = (s: string) => {
      const t = s.trim().split(/\r?\n/)[0].trim();
      return t.length > 44 ? t.slice(0, 44) + "…" : t;
    };
    const primaryKey = STAGE_FIELDS[stage]?.[0]?.key;
    const primaryVal = primaryKey ? (forms[primaryKey] ?? "").trim() : "";
    const derivedName = forms.company?.trim()
      || (rawMode ? firstLine(rawInput) : (primaryVal ? firstLine(primaryVal) : ""));

    const img = attachedImage;
    flushSync(() => {
      patchActiveCase(c => ({
        ...c,
        name: PLACEHOLDER_NAMES.has(c.name) && derivedName ? derivedName : c.name,
        stageStates: {
          ...c.stageStates,
          [stage]: { ...c.stageStates[stage], taskLabel: label, taskImageName: img?.name,
            initialRun: { status: "running", trace: [] } },
        },
      }));
    });
    setAttachedImage(null);

    // Auto-scroll chat to bottom
    setTimeout(() => chatBottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);

    await runSSE(input, stage, false, img?.path);
  }

  // ─── Send follow-up ───────────────────────────────────────────────

  async function runFollowup(q: string) {
    if ((!q && !attachedImage) || isInvestigating) return;
    if (!q && attachedImage) q = "请分析我新上传的这张截图/图片。";
    setBoardCollapsed(false);  // a fresh result should surface the board

    // Rolling conversation window: initial task + every completed followup turn.
    // Conversational (non-verdict) replies are preserved as raw text inside
    // buildFollowupInput — a "please upload your JD" turn must survive so the
    // next "如图 + attachment" reply keeps its meaning.
    const formSummary = summary(stage, forms);
    const turns: ConvTurn[] = [];
    if (ss.initialRun.answer) {
      turns.push({ question: ss.taskLabel ?? formSummary, answer: ss.initialRun.answer });
    }
    for (const f of ss.followups) {
      if (f.runState.answer) turns.push({ question: f.question, answer: f.runState.answer });
    }

    const input = buildFollowupInput(q, turns, formSummary, {
      resume: forms.resume, jd: forms.jd,
    });

    // Sticky routed stage: if the previous turn was answered by another stage
    // capability (e.g. routed to Resume Fit), the thread stays in that mode —
    // a bare "如图" reply has no routing keywords and would otherwise fall back
    // to the panel's stage. The engine router can still switch away explicitly.
    const lastRun = ss.followups.length > 0
      ? ss.followups[ss.followups.length - 1].runState : undefined;
    const requestStage: Stage = lastRun?.routedStage ?? stage;

    const img = attachedImage;
    // Add followup slot and clear chat input. Pre-seed routedStage when the
    // thread is sticky-routed so the badge shows and the NEXT turn inherits it
    // (the engine only emits stage_routed on a *change* from the request stage).
    flushSync(() => {
      patchStageStates(prev => ({
        ...prev,
        [stage]: {
          ...prev[stage],
          chatInput: "",
          followups: [...prev[stage].followups, {
            question: q,
            runState: { status: "running", trace: [],
              routedStage: requestStage !== stage ? requestStage : undefined },
            isFollowup: true,
            imageName: img?.name,
          }],
        },
      }));
    });
    setAttachedImage(null);

    setTimeout(() => chatBottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    await runSSE(input, stage, true, img?.path, requestStage);
  }

  function sendFollowup() { void runFollowup(ss.chatInput.trim()); }
  // A suggested-followup chip (problem 4): fire the canned question as a real
  // investigation turn (still fully grounded — goes through the normal loop).
  function sendSuggestedFollowup(q: string) { void runFollowup(q.trim()); }

  // ─── Reset stage ─────────────────────────────────────────────────

  function resetStage(s: Stage) {
    patchStageStates(prev => ({ ...prev, [s]: EMPTY_STAGE_STATE() }));
  }

  // ─── Derived state ───────────────────────────────────────────────

  const initialDone = ss.initialRun.status === "done" || ss.initialRun.status === "error";
  // A case is "fresh" when no stage has been run yet — show the hero pitch.
  const caseFresh = (Object.keys(activeCase.stageStates) as Stage[])
    .every(s => activeCase.stageStates[s].initialRun.status === "idle");

  // ─── Chat input key handler ───────────────────────────────────────

  function onChatKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendFollowup();
    }
  }

  // Canvas: which run to show trace for — show all runs for context
  const allRuns: Array<{ label: string; run: RunState; isFollowup: boolean; question?: string; anchorId: string }> = [
    { label: "Investigation", run: ss.initialRun, isFollowup: false, anchorId: "nexa-result-initial" },
    ...ss.followups.map((f, i) => ({ label: `Follow-up ${i + 1}`, run: f.runState, isFollowup: true, question: f.question, anchorId: `nexa-result-followup-${i}` })),
  ].filter(r => r.run.status !== "idle");

  // Evidence Board has content only once a run has produced an answer.
  const hasBoardContent = allRuns.some(r =>
    (r.run.status === "done" || r.run.status === "error") && (r.run.answer || r.run.status === "error"));
  // Show the full board only when it has content AND the user hasn't hidden it.
  const showBoard = hasBoardContent && !boardCollapsed;

  // ─── Render ──────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden",
      background: "oklch(98% 0.01 70)", fontFamily: "var(--font-sans, Manrope, sans-serif)", color: "oklch(24% 0.02 50)" }}>

      {/* ═══ SIDEBAR ═══════════════════════════════════════════════ */}
      <div style={{ width: 232, flexShrink: 0, borderRight: "1px solid oklch(90% 0.012 70)",
        padding: "28px 16px", display: "flex", flexDirection: "column", gap: 24,
        height: "100%", overflowY: "auto", background: "white" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <svg width="26" height="26" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="OfferCheck logo">
              <defs>
                <linearGradient id="oc-shield" x1="16" y1="2.5" x2="16" y2="30" gradientUnits="userSpaceOnUse">
                  <stop offset="0" stopColor="#CF6A44" />
                  <stop offset="1" stopColor="#A94A2C" />
                </linearGradient>
              </defs>
              <path d="M16 2.5 L27 6.6 V15 C27 22 22.2 27.8 16 30 C9.8 27.8 5 22 5 15 V6.6 Z" fill="url(#oc-shield)" />
              <circle cx="14.2" cy="14" r="5" stroke="#FFFFFF" strokeWidth="2.3" fill="none" />
              <line x1="17.9" y1="17.7" x2="21.6" y2="21.4" stroke="#FFFFFF" strokeWidth="2.6" strokeLinecap="round" />
              <path d="M11.7 14.1 L13.5 15.9 L16.9 12" stroke="#FFFFFF" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            </svg>
            <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.02em" }}>OfferCheck</div>
          </div>
          <div style={{ fontSize: 11.5, color: "oklch(48% 0.02 50)", marginTop: 6, lineHeight: 1.55 }}>
            {UI.tagline}
          </div>
        </div>

        {/* Cases — each is one job opportunity tracked across the four stages */}
        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <div style={{ fontSize: 10.5, fontWeight: 700, color: "oklch(45% 0.02 50)",
              textTransform: "uppercase", letterSpacing: "0.05em" }}>Cases</div>
            <button onClick={createCase} title="New case" style={{
              background: "transparent", border: "1px solid oklch(88% 0.012 70)", cursor: "pointer",
              borderRadius: 7, padding: "3px 9px", fontSize: 11.5, fontWeight: 700,
              fontFamily: "var(--font-sans)", color: "oklch(45% 0.05 250)" }}>
              + New
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {cases.map(c => {
              const isActive = c.id === activeCaseId;
              return (
                <div key={c.id} onClick={() => selectCase(c.id)} style={{
                  padding: "9px 10px", borderRadius: 9, cursor: "pointer",
                  border: isActive ? "1px solid oklch(80% 0.06 40)" : "1px solid transparent",
                  background: isActive ? "oklch(96% 0.02 40)" : "transparent" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div onDoubleClick={e => { e.stopPropagation(); renameCase(c.id); }}
                      title="Double-click to rename"
                      style={{ flex: 1, minWidth: 0, fontSize: 13, fontWeight: 700,
                        color: PLACEHOLDER_NAMES.has(c.name) ? "oklch(58% 0.02 50)" : "oklch(28% 0.02 50)",
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {c.name}
                    </div>
                    {cases.length > 1 && (
                      <button onClick={e => { e.stopPropagation(); deleteCase(c.id); }} title="Delete case"
                        style={{ background: "transparent", border: "none", cursor: "pointer", padding: 0,
                          fontSize: 13, lineHeight: 1, color: "oklch(65% 0.02 50)", flexShrink: 0 }}>
                        ×
                      </button>
                    )}
                  </div>
                  {/* Per-case four-stage journey dots */}
                  <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 7 }}>
                    {(Object.keys(STAGE_META) as Stage[]).map(s => {
                      const st = c.stageStates[s].initialRun.status;
                      const dstStageActive = isActive && s === c.activeStage;
                      const bg = st === "done" ? "oklch(52% 0.13 145)"
                        : st === "running" ? "oklch(58% 0.13 40)"
                        : st === "error" ? "oklch(60% 0.14 25)"
                        : dstStageActive ? "oklch(70% 0.08 40)" : "oklch(90% 0.012 70)";
                      return (
                        <div key={s} title={`${STAGE_META[s].num} ${STAGE_META[s].en} — ${st}`}
                          style={{ width: 20, height: 5, borderRadius: 3, background: bg,
                            animation: st === "running" ? "pulse 1.1s ease-in-out infinite" : undefined }} />
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ marginTop: "auto", paddingTop: 16, borderTop: "1px solid oklch(90% 0.012 70)" }}>
          <div style={{ fontSize: 10.5, fontWeight: 700, color: "oklch(45% 0.02 50)",
            textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>Investigation tools</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {TOOL_LEGEND.map(t => (
              <div key={t.label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <div style={{ width: 8, height: 8, borderRadius: 3, background: t.fg, flexShrink: 0 }} />
                <div style={{ fontSize: 11, color: "oklch(42% 0.02 50)" }}>{t.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ═══ CHAT PANEL ════════════════════════════════════════════ */}
      <div ref={chatPanelRef} style={{
        ...(userResized
          ? { width: chatWidth, flexShrink: 0 }
          : { flexGrow: 56, flexBasis: 0, minWidth: 360 }),
        display: "flex", flexDirection: "column",
        overflow: "hidden", background: "oklch(99.3% 0.004 70)" }}>

        {/* Header */}
        <div style={{ flexShrink: 0, padding: "18px 20px", borderBottom: "1px solid oklch(90% 0.012 70)",
          background: "white", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.07em",
              color: "oklch(58% 0.13 40)", textTransform: "uppercase" }}>
              Stage {meta.num}
            </div>
            <div style={{ fontSize: 17, fontWeight: 800, marginTop: 3 }}>{meta.en}</div>
          </div>
          {hasStarted && (
            <button onClick={() => resetStage(stage)} style={{
              background: "transparent", border: "1px solid oklch(88% 0.012 70)",
              color: "oklch(48% 0.02 50)", padding: "5px 12px", borderRadius: 7,
              fontSize: 12, fontFamily: "var(--font-sans)", cursor: "pointer" }}>
              {UI.reset}
            </button>
          )}
        </div>

        {/* Messages */}
        {/* minHeight:0 是关键——flex 子项默认 min-height:auto 会被内容撑高、
            使 overflowY:auto 失效（尤其调查结束展开长 trace 时无法滚动）。 */}
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "16px 18px",
          display: "flex", flexDirection: "column", gap: 10 }}>

          {/* Hero pitch — only on a brand-new case */}
          {caseFresh && (
            <div style={{ padding: "18px 20px 6px" }}>
              <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em",
                lineHeight: 1.25, color: "oklch(24% 0.02 50)" }}>
                Verify before you trust.
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.6, color: "oklch(45% 0.02 50)", marginTop: 8 }}>
                OfferCheck actively investigates <b>this specific opportunity</b> — not just a static
                company credit score — to catch impersonation scams (fake offers, look-alike domains,
                bogus HR). Around <b>38% of remote job postings</b> involve some form of scam; run one
                free check before you sign or pay anything.
              </div>
            </div>
          )}

          {/* Intro bubble */}
          <div style={{ maxWidth: "92%", background: "white", border: "1px solid oklch(90% 0.012 70)",
            fontSize: 13.5, lineHeight: 1.6, color: "oklch(30% 0.02 50)",
            padding: "11px 14px", borderRadius: "3px 12px 12px 12px" }}>
            {meta.chatIntro}
          </div>

          {/* Cross-stage context banner — only shown when carryover will really happen */}
          {ss.initialRun.status === "idle" && completedEarlierStages(stageStates, stage).length > 0 && (
            <div style={{ maxWidth: "92%", display: "flex", gap: 8, alignItems: "flex-start",
              background: "oklch(95% 0.03 80)", border: "1px solid oklch(85% 0.05 80)",
              borderRadius: "3px 12px 12px 12px", padding: "11px 14px" }}>
              <span style={{ fontSize: 14, lineHeight: 1.3 }}>◆</span>
              <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "oklch(34% 0.05 80)" }}>
                Carrying findings from {completedEarlierStages(stageStates, stage)
                  .map(s => `${STAGE_META[s].num} ${STAGE_META[s].en}`).join(" · ")} into this
                investigation — verdicts will still be re-verified with fresh evidence.
              </div>
            </div>
          )}

          {/* User's initial submission bubble */}
          {hasStarted && (
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <div style={{ maxWidth: "88%", background: "oklch(58% 0.13 40)", color: "white",
                fontSize: 13, lineHeight: 1.6, padding: "10px 14px",
                borderRadius: "12px 3px 12px 12px", whiteSpace: "pre-wrap" }}>
                {ss.taskImageName && (
                  <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 4 }}>📎 {ss.taskImageName}</div>
                )}
                {ss.taskLabel ?? summary(stage, forms)}
              </div>
            </div>
          )}

          {/* Initial run: inline trace + verdict summary */}
          {hasStarted && ss.initialRun.trace.length > 0 && (
            <InlineTrace
              trace={ss.initialRun.trace}
              isDone={initialDone}
              stepCount={ss.initialRun.currentStep ?? ss.initialRun.trace.filter(t => t.type === "step").length}
              maxSteps={ss.initialRun.maxSteps}
              currentTrial={ss.initialRun.currentTrial}
              maxTrials={ss.initialRun.maxTrials}
              label="Investigating"
            />
          )}
          {ss.initialRun.status === "done" && ss.initialRun.answer && (
            <div style={{ maxWidth: "92%", background: "white",
              border: "1px solid oklch(90% 0.012 70)", fontSize: 13.5, lineHeight: 1.6,
              color: "oklch(30% 0.02 50)", padding: "11px 14px",
              borderRadius: "3px 12px 12px 12px" }}>
              <ChatSummary
                parsed={withEngineSources(parseStructuredAnswer(ss.initialRun.answer), ss.initialRun.sources)}
                summaryForUser={ss.initialRun.summaryForUser}
                suggestedFollowups={ss.initialRun.suggestedFollowups}
                onFollowup={sendSuggestedFollowup}
                latencyMs={ss.initialRun.latency_ms}
                trials={ss.initialRun.trials_used}
                jumpTargetId="nexa-result-initial"
              />
            </div>
          )}
          {ss.initialRun.status === "error" && (
            <div style={{ maxWidth: "92%", background: "white",
              border: "1px solid oklch(90% 0.012 70)", fontSize: 13.5, lineHeight: 1.6,
              color: "oklch(50% 0.14 25)", padding: "11px 14px",
              borderRadius: "3px 12px 12px 12px" }}>
              Investigation failed: {ss.initialRun.errorMsg}
            </div>
          )}

          {/* Follow-up Q&A thread */}
          {ss.followups.map((f, i) => {
            const fDone = f.runState.status === "done" || f.runState.status === "error";
            return (
              <div key={i} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {/* User question bubble */}
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <div style={{ maxWidth: "88%", background: "oklch(58% 0.13 40)", color: "white",
                    fontSize: 13, lineHeight: 1.6, padding: "10px 14px",
                    borderRadius: "12px 3px 12px 12px", whiteSpace: "pre-wrap" }}>
                    {f.imageName && (
                      <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 4 }}>📎 {f.imageName}</div>
                    )}
                    {f.question}
                  </div>
                </div>
                {/* Routed badge — the followup was answered by another stage capability */}
                {f.runState.routedStage && (
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 6,
                    background: "oklch(95% 0.03 250)", border: "1px solid oklch(87% 0.05 250)",
                    borderRadius: 999, padding: "3px 11px", fontSize: 11.5, fontWeight: 600,
                    color: "oklch(42% 0.09 250)", width: "fit-content" }}>
                    ↪ {STAGE_META[f.runState.routedStage].num} {STAGE_META[f.runState.routedStage].en} capability
                  </div>
                )}
                {/* Inline trace for the follow-up run */}
                {f.runState.trace.length > 0 && (
                  <InlineTrace
                    trace={f.runState.trace}
                    isDone={fDone}
                    stepCount={f.runState.currentStep ?? f.runState.trace.filter(t => t.type === "step").length}
                    maxSteps={f.runState.maxSteps}
                    currentTrial={f.runState.currentTrial}
                    maxTrials={f.runState.maxTrials}
                    label="Following up"
                  />
                )}
                {/* Live streaming answer (answer-mode, while running) */}
                {f.runState.status === "running" && f.runState.streamedAnswer && (
                  <div style={{ maxWidth: "92%", background: "white",
                    border: "1px solid oklch(90% 0.012 70)", fontSize: 13.5, lineHeight: 1.65,
                    color: "oklch(30% 0.02 50)", padding: "11px 14px",
                    borderRadius: "3px 12px 12px 12px", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {f.runState.streamedAnswer}
                    <span style={{ animation: "pulse 1s ease-in-out infinite" }}>▍</span>
                  </div>
                )}
                {/* Agent response bubble (done): conversational answer-mode text OR verdict summary */}
                {f.runState.status === "done" && f.runState.answer && (() => {
                  const parsed = withEngineSources(parseStructuredAnswer(f.runState.answer), f.runState.sources);
                  const bubbleStyle: React.CSSProperties = { maxWidth: "92%", background: "white",
                    border: "1px solid oklch(90% 0.012 70)", fontSize: 13.5, lineHeight: 1.65,
                    color: "oklch(30% 0.02 50)", padding: "11px 14px", borderRadius: "3px 12px 12px 12px" };
                  // Non-verdict → conversational answer-mode reply: render the full text.
                  if (!parsed.verdictLabel) {
                    return (
                      <div style={{ ...bubbleStyle, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                        {f.runState.answer}
                        {f.runState.latency_ms ? (
                          <div style={{ fontSize: 12, color: "oklch(58% 0.02 50)", marginTop: 6 }}>
                            ({Math.round(f.runState.latency_ms / 1000)}s)
                          </div>
                        ) : null}
                      </div>
                    );
                  }
                  // Verdict → the model chose to re-investigate: show the structured summary.
                  const prevAnswer = i > 0
                    ? (ss.followups[i - 1].runState.answer ?? "")
                    : (ss.initialRun.answer ?? "");
                  const prevVerdict = parseStructuredAnswer(prevAnswer).verdictLabel || undefined;
                  return (
                    <div style={bubbleStyle}>
                      <ChatSummary
                        parsed={parsed}
                        prevVerdict={prevVerdict}
                        summaryForUser={f.runState.summaryForUser}
                        suggestedFollowups={f.runState.suggestedFollowups}
                        onFollowup={sendSuggestedFollowup}
                        latencyMs={f.runState.latency_ms}
                        trials={f.runState.trials_used}
                        jumpTargetId={`nexa-result-followup-${i}`}
                      />
                    </div>
                  );
                })()}
                {f.runState.status === "error" && (
                  <div style={{ maxWidth: "92%", background: "white",
                    border: "1px solid oklch(90% 0.012 70)", fontSize: 13.5, lineHeight: 1.6,
                    color: "oklch(50% 0.14 25)", padding: "11px 14px",
                    borderRadius: "3px 12px 12px 12px" }}>
                    Follow-up failed: {f.runState.errorMsg}
                  </div>
                )}
              </div>
            );
          })}

          <div ref={chatBottomRef} />
        </div>

        {/* ─── Composer: stage chips + start / follow-up ─────────── */}
        <div style={{ flexShrink: 0, padding: "12px 18px", borderTop: "1px solid oklch(90% 0.012 70)",
          background: "white", display: "flex", flexDirection: "column", gap: 10 }}>

          {/* Stage chips */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {(Object.keys(STAGE_META) as Stage[]).map(s => {
              const m = STAGE_META[s];
              const active = s === stage;
              const done = stageStates[s].initialRun.status === "done";
              return (
                <button key={s} onClick={() => setStage(s)} style={{
                  display: "inline-flex", alignItems: "center", gap: 5, padding: "5px 11px",
                  borderRadius: 999, fontSize: 12, fontWeight: 700, fontFamily: "var(--font-sans)",
                  cursor: "pointer", whiteSpace: "nowrap",
                  border: active ? "1px solid oklch(58% 0.13 40)" : "1px solid oklch(88% 0.012 70)",
                  background: active ? "oklch(58% 0.13 40)" : "white",
                  color: active ? "white" : "oklch(40% 0.02 50)" }}>
                  {done && <span style={{ color: active ? "white" : "oklch(48% 0.11 145)" }}>✓</span>}
                  <span style={{ opacity: 0.7, fontSize: 10.5 }}>{m.num}</span>
                  {m.chip}
                  {m.engine === "soon" && <span style={{ opacity: 0.6, fontSize: 10 }}>· soon</span>}
                </button>
              );
            })}
          </div>

          {/* Attachment (screenshot / PDF) — engine OCRs it via analyze_image */}
          {meta.engine === "live" && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input ref={fileInputRef} type="file" accept="image/*,.pdf" style={{ display: "none" }}
                onChange={e => { const f = e.target.files?.[0]; if (f) uploadImage(f); e.target.value = ""; }} />
              {attachedImage ? (
                <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "4px 8px",
                  border: "1px solid oklch(88% 0.012 70)", borderRadius: 9, background: "oklch(98% 0.01 70)" }}>
                  {/\.pdf$/i.test(attachedImage.name)
                    ? <span style={{ fontSize: 16 }}>📄</span>
                    : <img src={attachedImage.preview} alt="" style={{ width: 28, height: 28, objectFit: "cover", borderRadius: 5 }} />}
                  <span style={{ fontSize: 12, color: "oklch(35% 0.02 50)", maxWidth: 160,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{attachedImage.name}</span>
                  <button onClick={() => setAttachedImage(null)} title="Remove"
                    style={{ background: "transparent", border: "none", cursor: "pointer", color: "oklch(60% 0.02 50)",
                      fontSize: 14, lineHeight: 1, padding: 0 }}>×</button>
                </div>
              ) : (
                <button onClick={() => fileInputRef.current?.click()} disabled={uploading || isInvestigating}
                  style={{ display: "inline-flex", alignItems: "center", gap: 5, background: "transparent",
                    border: "1px solid oklch(88% 0.012 70)", borderRadius: 9, padding: "5px 11px",
                    fontSize: 12, fontWeight: 600, fontFamily: "var(--font-sans)", color: "oklch(45% 0.02 50)",
                    cursor: uploading ? "default" : "pointer", opacity: uploading ? 0.6 : 1 }}>
                  📎 {uploading ? "Uploading…" : "Attach screenshot / PDF"}
                </button>
              )}
            </div>
          )}

          {/* Start mode: guided fields (or free text) + start button */}
          {!hasStarted ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {rawMode ? (
                <textarea
                  value={rawInput}
                  onChange={e => setRawInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && meta.engine === "live") { e.preventDefault(); startInitial(); } }}
                  rows={3}
                  placeholder={UI.rawPlaceholder}
                  style={{ ...inputStyle, resize: "vertical" }}
                />
              ) : (
                <StageForm stage={stage} forms={forms} onChange={updateForm} />
              )}

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <button onClick={() => setRawMode(m => !m)} style={{
                    background: "transparent", border: "none", padding: 0, cursor: "pointer",
                    fontSize: 12, fontFamily: "var(--font-sans)", color: "oklch(50% 0.05 250)", fontWeight: 600 }}>
                    {rawMode ? UI.useGuided : UI.skipToType}
                  </button>
                  {meta.engine === "live" && !isInvestigating && (
                    <button onClick={() => fillDemo(stage)} style={{
                      background: "transparent", border: "none", padding: 0, cursor: "pointer",
                      fontSize: 12, fontFamily: "var(--font-sans)", color: "oklch(52% 0.02 50)", fontWeight: 600 }}>
                      ✨ {UI.tryExample}
                    </button>
                  )}
                </div>

                {meta.engine === "soon" ? (
                  <span style={{ fontSize: 12, color: "oklch(52% 0.02 50)" }}>{UI.comingSoon}</span>
                ) : (
                  <button onClick={startInitial} disabled={!startValid || isInvestigating} style={{
                    background: startValid ? "oklch(58% 0.13 40)" : "oklch(80% 0.012 70)",
                    color: "white", border: "none", padding: "9px 20px", borderRadius: 9,
                    fontSize: 13, fontWeight: 700, fontFamily: "var(--font-sans)",
                    cursor: startValid ? "pointer" : "not-allowed", opacity: startValid ? 1 : 0.65 }}>
                    {isInvestigating ? meta.ctaRunning : meta.ctaIdle}
                  </button>
                )}
              </div>
            </div>
          ) : (
            /* Follow-up mode */
            meta.engine === "live" && (
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                <textarea
                  value={ss.chatInput}
                  onChange={e => patchStageStates(prev => ({ ...prev, [stage]: { ...prev[stage], chatInput: e.target.value } }))}
                  onKeyDown={onChatKeyDown}
                  disabled={isInvestigating}
                  rows={1}
                  placeholder={
                    isInvestigating ? UI.runningPlaceholder :
                    !initialDone ? UI.waitPlaceholder :
                    UI.followupPlaceholder
                  }
                  style={{ flex: 1, boxSizing: "border-box", padding: "10px 12px", borderRadius: 9,
                    border: "1px solid oklch(88% 0.012 70)", fontSize: 13.5, fontFamily: "var(--font-sans)",
                    resize: "none", outline: "none", lineHeight: 1.5,
                    opacity: isInvestigating || !initialDone ? 0.6 : 1 }}
                />
                <button
                  onClick={sendFollowup}
                  disabled={isInvestigating || !initialDone || (!ss.chatInput.trim() && !attachedImage)}
                  style={{ background: "oklch(58% 0.13 40)", color: "white", border: "none",
                    padding: "10px 16px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                    fontFamily: "var(--font-sans)", cursor: "pointer", flexShrink: 0,
                    opacity: (isInvestigating || !initialDone || (!ss.chatInput.trim() && !attachedImage)) ? 0.5 : 1 }}>
                  {UI.send}
                </button>
              </div>
            )
          )}
        </div>
      </div>

      {/* ═══ RESIZE HANDLE (only when board is expanded) ════════════ */}
      {showBoard && (
        <div onMouseDown={onResizeDown} style={{ width: 6, flexShrink: 0, cursor: "col-resize",
          background: "oklch(90% 0.012 70)", position: "relative", userSelect: "none" }}>
          <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
            width: 2, height: 36, borderRadius: 2, background: "oklch(75% 0.012 70)" }} />
        </div>
      )}

      {/* ═══ EVIDENCE BOARD ══════════════════════════════════════════ */}
      {!showBoard ? (
        /* Collapsed strip — no findings yet, or manually hidden (then clickable to reopen) */
        <div onClick={hasBoardContent ? () => setBoardCollapsed(false) : undefined}
          title={hasBoardContent ? "Show Evidence Board" : undefined}
          style={{ width: 48, flexShrink: 0, borderLeft: "1px solid oklch(90% 0.012 70)",
            background: "oklch(98% 0.01 70)", display: "flex", alignItems: "center", justifyContent: "center",
            gap: 8, flexDirection: "column", cursor: hasBoardContent ? "pointer" : "default" }}>
          {hasBoardContent && <span style={{ fontSize: 13, color: "oklch(55% 0.02 50)" }}>‹</span>}
          <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 11,
            fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
            color: "oklch(62% 0.02 50)" }}>
            {UI.boardCollapsed}
          </div>
        </div>
      ) : (
        <div style={{ ...(userResized ? { flexGrow: 1 } : { flexGrow: 44 }),
          flexBasis: 0, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 340 }}>
          <div style={{ flexShrink: 0, padding: "24px 28px 16px", borderBottom: "1px solid oklch(90% 0.012 70)",
            background: "oklch(98% 0.01 70)", display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.07em",
                color: "oklch(58% 0.13 40)", textTransform: "uppercase" }}>
                {UI.boardTitle}
              </div>
              <div style={{ fontSize: 13.5, color: "oklch(45% 0.02 50)", marginTop: 5 }}>
                {UI.boardSub}
              </div>
            </div>
            <button onClick={() => setBoardCollapsed(true)} title="Hide Evidence Board" style={{
              flexShrink: 0, background: "transparent", border: "1px solid oklch(88% 0.012 70)",
              borderRadius: 7, width: 28, height: 28, cursor: "pointer", color: "oklch(50% 0.02 50)",
              fontSize: 15, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
              ✕
            </button>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "20px 28px 80px" }}>
            <div style={{ maxWidth: 820, margin: "0 auto" }}>

              {/* Runs: verdict + evidence only (trace now lives inline in the chat) */}
              {allRuns.map((r, idx) => {
                const parsed = r.run.answer ? withEngineSources(parseStructuredAnswer(r.run.answer), r.run.sources) : null;

                // Conversational answer-mode follow-ups (no verdict) live in the chat only —
                // the board updates only when there's genuinely new structured evidence.
                if (r.isFollowup && r.run.status !== "error" && parsed && !parsed.verdictLabel) return null;

                return (
                  <div key={idx}>
                    {/* Divider for follow-ups */}
                    {r.isFollowup && (
                      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "20px 0 16px" }}>
                        <div style={{ flex: 1, height: 1, background: "oklch(88% 0.012 70)" }} />
                        <div style={{ fontSize: 11.5, fontWeight: 700, color: "oklch(52% 0.02 50)",
                          whiteSpace: "nowrap" }}>
                          Follow-up {idx} · {r.question && r.question.length > 40 ? r.question.slice(0, 40) + "…" : r.question}
                        </div>
                        <div style={{ flex: 1, height: 1, background: "oklch(88% 0.012 70)" }} />
                      </div>
                    )}

                    {/* Result (anchor target for the「see full evidence」jump) */}
                    {r.run.answer && parsed && (
                      <div id={r.anchorId} style={{ scrollMarginTop: 12 }}>
                        <StructuredResult parsed={parsed} isFollowup={r.isFollowup} anchorBase={r.anchorId} />
                      </div>
                    )}

                    {/* Error */}
                    {r.run.status === "error" && (
                      <div style={{ background: "oklch(96% 0.035 25)", border: "1px solid oklch(85% 0.05 25)",
                        borderRadius: 14, padding: "16px 20px", color: "oklch(35% 0.12 25)", fontSize: 13.5,
                        marginBottom: 16 }}>
                        <div style={{ fontWeight: 700, marginBottom: 4 }}>Investigation failed</div>
                        <div>{r.run.errorMsg}</div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

