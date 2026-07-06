"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ─── Backend base URL ───────────────────────────────────────────────────────────
// In production the browser talks to the FastAPI backend directly — set
// NEXT_PUBLIC_API_BASE to the Railway URL (e.g. https://nexa-xxx.up.railway.app)
// so long-running SSE runs bypass Vercel's 60s serverless-function limit.
// Empty (local dev) → relative paths hit the Next.js proxy route on :3000.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");
export const apiUrl = (path: string) => `${API_BASE}${path}`;

// ─── Types ────────────────────────────────────────────────────────────────────

export type Stage = "stage1" | "stage2" | "stage3" | "stage4";

export type EngineEvent = {
  type: string;
  stage?: string;
  step?: number;
  max_steps?: number;
  model?: string;
  tool?: string;
  args?: string;
  query?: string;          // engine-provided human-readable query for a step (problem 2)
  thought?: string;
  observation?: string;
  ok?: boolean;
  message?: string;
  trial?: number;
  max_trials?: number;
  attempt?: number;        // LLM retry attempt number (problem 3)
  error?: string;          // retry error text (problem 3)
  from_stage?: string;     // stage_routed: the stage the followup came from
  to_stage?: string;       // stage_routed: the capability the followup was routed to
  text?: string;           // answer_delta streamed token chunk (answer-mode)
  passed?: boolean;
  reason?: string;
  success?: boolean;
  answer?: string;
  sources?: SourceItem[];  // structured sources from the engine (problem 5)
  summary_for_user?: string;      // problem 4
  suggested_followups?: string[]; // problem 4
  trials_used?: number;
  latency_ms?: number;
};

export type SourceItem = { url: string; domain: string; context?: string; verified?: boolean };

export type StepStatus = "running" | "success" | "notfound" | "error" | "cache";

export type TraceItem = {
  id: number;
  type: "step" | "meta" | "warn";
  stepNum?: number;
  tool?: string;
  label?: string;
  brief?: string;
  thought?: string;
  args?: string;
  status?: StepStatus;
  observation?: string;
  model?: string;
  metaKind?: string;
  metaDetail?: string;
  ok?: boolean;
};

export type RunState = {
  status: "idle" | "running" | "done" | "error";
  trace: TraceItem[];
  answer?: string;
  success?: boolean;
  latency_ms?: number;
  trials_used?: number;
  errorMsg?: string;
  maxSteps?: number;
  currentStep?: number;            // per-trial step number from the latest step_start event
  currentTrial?: number;
  maxTrials?: number;
  sources?: SourceItem[];          // engine-provided structured sources (problem 5)
  summaryForUser?: string;         // engine-provided user-facing summary (problem 4)
  suggestedFollowups?: string[];   // engine-provided follow-up suggestions (problem 4)
  streamedAnswer?: string;         // live answer-mode text accumulated from answer_delta
  streamStep?: number;             // which step the current streamedAnswer belongs to
  routedStage?: Stage;             // followup auto-routed to another stage capability
};

export type FollowupItem = {
  question: string;
  runState: RunState;
  isFollowup: true;
  imageName?: string; // attached screenshot/PDF filename, for the request bubble
};

export type StageState = {
  initialRun: RunState;
  followups: FollowupItem[];
  chatInput: string;
  taskLabel?: string; // what the user submitted (form summary or raw text), for the request bubble
  taskImageName?: string; // attached screenshot/PDF filename on the initial submission
};

export type FormData = { [field: string]: string };

// Parsed structured answer from LLM output
export type ParsedAnswer = {
  verdictLabel: string;  // canonical key for VERDICT_STYLES, e.g. "靠谱"
  verdictRaw: string;    // label exactly as the model wrote it (e.g. "Likely a Scam") — for the subtitle
  verdictReason: string; // text after " — " on the verdict line
  facts: { text: string; confidence: string; url?: string }[];
  redFlags: { text: string }[];
  sources: SourceItem[];
  rawText: string;
};

// ─── Stage metadata ────────────────────────────────────────────────────────────

export const STAGE_META: Record<Stage, {
  num: string; cn: string; en: string; chip: string; chatIntro: string;
  ctaIdle: string; ctaRunning: string; ctaDone: string;
  engine: "live" | "soon";
}> = {
  stage1: { num: "01", cn: "选岗调研", en: "Role Research", chip: "Research",
    chatIntro: "Tell me the company and role you're weighing. I'll independently investigate whether the company is real, how healthy its funding looks, and whether the posting is an active opening or a long-standing ghost job.",
    ctaIdle: "Start investigation", ctaRunning: "Investigating…", ctaDone: "Re-run", engine: "live" },
  stage2: { num: "02", cn: "简历定向", en: "Resume Fit", chip: "Resume",
    chatIntro: "Paste the JD and your resume. I'll return an actionable, prioritized list of what to emphasize and what to add — not a full rewrite.",
    ctaIdle: "Analyze fit", ctaRunning: "Analyzing…", ctaDone: "Re-analyze", engine: "live" },
  stage3: { num: "03", cn: "沟通证伪", en: "Message Check", chip: "Messages",
    chatIntro: "Paste the recruiter's messages. I'll verify their identity and flag abnormal patterns. Keep pasting new messages as they arrive — I'll reason across the whole thread.",
    ctaIdle: "Run check", ctaRunning: "Checking…", ctaDone: "Re-check", engine: "live" },
  stage4: { num: "04", cn: "Offer 证伪", en: "Offer Verification", chip: "Offer",
    chatIntro: "Paste the offer or contract. I'll run the deepest cross-verification and return a verdict: Looks Legit, Suspicious, or Likely a Scam — with the evidence behind it.",
    ctaIdle: "Verify offer", ctaRunning: "Verifying…", ctaDone: "Re-verify", engine: "live" },
};

// Data-driven guided fields per stage (compact composer form).
// Keys must match those consumed by buildInput / valid / summary.
export type FieldSpec = { key: string; label: string; placeholder: string; required?: boolean; area?: boolean; rows?: number };
export const STAGE_FIELDS: Record<Stage, FieldSpec[]> = {
  stage1: [
    { key: "company", label: "Target company", placeholder: "e.g. Anthropic, ByteDance", required: true },
    { key: "position", label: "Role (optional)", placeholder: "e.g. Senior Software Engineer" },
    { key: "context", label: "Anything else (optional)", placeholder: "JD link, recruiter email, your background…", area: true, rows: 2 },
  ],
  stage2: [
    { key: "jd", label: "Job description", placeholder: "Paste the full JD", required: true, area: true, rows: 4 },
    { key: "resume", label: "Your resume", placeholder: "Paste your resume (or the key parts)", required: true, area: true, rows: 4 },
    { key: "target", label: "Focus (optional)", placeholder: "What to emphasize or worry about" },
  ],
  stage3: [
    { key: "message", label: "Recruiter messages", placeholder: "Paste the chat / email, or describe the exchange", required: true, area: true, rows: 4 },
    { key: "recruiter", label: "Recruiter info (optional)", placeholder: "Name, email, platform…", area: true, rows: 2 },
  ],
  stage4: [
    { key: "offer", label: "Offer / contract", placeholder: "Paste the full text — company, role, salary, location, signing entity", required: true, area: true, rows: 5 },
    { key: "hr", label: "HR contact (optional)", placeholder: "Email, phone, etc." },
    { key: "link", label: "Company site / job link (optional)", placeholder: "https://" },
  ],
};

// One-click demo presets — lets a judge/first-time visitor run a real
// investigation without typing. stage1 is a grounded "happy path" (a real,
// reputable company → low-risk verdict); stage3/4 use clearly fictitious
// entities showing textbook overseas-job-scam red flags. Never name a real
// company in a fraud example.
export const DEMO_FORMS: Record<Stage, Record<string, string>> = {
  stage1: {
    company: "Anthropic",
    position: "Software Engineer",
    context: "A recruiter reached out on LinkedIn about a remote role. I want to confirm the company and role are legitimate before I reply.",
  },
  stage2: {
    jd: "Senior Backend Engineer — build and scale distributed data pipelines in Python/Go. Requirements: 5+ years backend, strong SQL, experience with Kafka and Kubernetes, cloud (AWS/GCP).",
    resume: "Backend engineer, 6 years. Python & Go. Built real-time data pipelines (Kafka, Flink) on AWS. Led migration to Kubernetes. Strong SQL / PostgreSQL.",
    target: "Is this role a genuine fit, and are there gaps I should address?",
  },
  stage3: {
    message: "Hi! We saw your profile and want to offer you a remote data-entry job, $35/hour, no interview needed. To get started, please pay a one-time $150 onboarding/equipment deposit via USDT. Reply on Telegram @hr_fastjobs to receive your contract.",
    recruiter: "‘HR Manager’, contact only via Telegram @hr_fastjobs, no company email.",
  },
  stage4: {
    offer: "Offer Letter — Nexora Global Solutions Ltd.\nPosition: Remote Financial Operations Associate (no interview required)\nCompensation: USD 9,200 / month, paid weekly in USDT (crypto).\nLocation: 100% remote, flexible hours, immediate start.\nSigning entity: Everbright Holdings Group (Hong Kong).\nOnboarding: purchase a $320 equipment & software kit from our partner vendor; fully reimbursed in your first paycheck.\nNote: limited positions, please confirm within 24 hours.",
    hr: "Telegram @nexora_hr only (no corporate email address provided)",
    link: "",
  },
};

// UI chrome strings (English-first). Investigation content language stays adaptive.
export const UI = {
  tryExample: "Try an example",
  tagline: "Deep-research any job offer for scams — before you trust it.",
  taglineSub: "A suspicious research co-pilot for the whole job hunt.",
  boardTitle: "Evidence Board",
  boardSub: "Verdicts, red flags and sources land here as the investigation runs.",
  boardEmpty: "No findings yet",
  boardEmptyHint: "Start an investigation on the left. Verdict and evidence will appear here.",
  boardCollapsed: "Evidence Board",
  skipToType: "Skip — just type",
  useGuided: "Use guided form",
  followupPlaceholder: "Ask a follow-up or add new info…  (Enter to send, Shift+Enter for newline)",
  runningPlaceholder: "Investigating…",
  waitPlaceholder: "Waiting for the investigation to finish…",
  send: "Send",
  reset: "Reset",
  comingSoon: "Engine in development — coming soon",
  rawPlaceholder: "Describe what you want me to investigate…",
};

// ─── Tool meta ─────────────────────────────────────────────────────────────────

export const TOOL_META: Record<string, { label: string; bg: string; fg: string }> = {
  web_search:        { label: "Web Search",  bg: "oklch(93% 0.03 250)", fg: "oklch(38% 0.12 250)" },
  wikipedia_search:  { label: "Wikipedia",   bg: "oklch(93% 0.03 265)", fg: "oklch(38% 0.1 265)" },
  web_fetch:         { label: "Page Fetch",  bg: "oklch(93% 0.03 300)", fg: "oklch(38% 0.12 300)" },
  tavily_extract:    { label: "Page Fetch",  bg: "oklch(93% 0.03 300)", fg: "oklch(38% 0.12 300)" },
  extract_url_text:  { label: "Page Fetch",  bg: "oklch(93% 0.03 300)", fg: "oklch(38% 0.12 300)" },
  domain_whois_lookup:{ label: "WHOIS",      bg: "oklch(93% 0.03 190)", fg: "oklch(35% 0.1 190)" },
  analyze_image:     { label: "Image OCR",   bg: "oklch(93% 0.03 335)", fg: "oklch(40% 0.13 335)" },
  analyze_image_cloud:{ label: "Image OCR",  bg: "oklch(93% 0.03 335)", fg: "oklch(40% 0.13 335)" },
  read_pdf:          { label: "PDF Parse",   bg: "oklch(93% 0.03 35)",  fg: "oklch(40% 0.12 35)" },
  calculator:        { label: "Analysis",    bg: "oklch(93% 0.01 50)",  fg: "oklch(38% 0.02 50)" },
  submit_verdict:    { label: "Verdict",     bg: "oklch(93% 0.03 40)",  fg: "oklch(38% 0.1 40)" },
};
export function tm(name: string) { return TOOL_META[name] ?? { label: name, bg: "oklch(93% 0.01 50)", fg: "oklch(40% 0.02 50)" }; }

// Human-readable one-liner for a step, derived from the tool's JSON args (query / url /
// domain / …). Stable across turns — unlike the model's optional reasoning `thought`.
export function summarizeArgs(_tool: string | undefined, args: string | undefined): string {
  if (!args) return "";
  let obj: unknown = null;
  try { obj = JSON.parse(args); } catch { return args.length > 140 ? args.slice(0, 140) + "…" : args; }
  if (obj && typeof obj === "object") {
    const o = obj as Record<string, unknown>;
    const preferred = o.query ?? o.q ?? o.url ?? o.domain ?? o.expression ?? o.term ?? o.text ?? o.path ?? o.file ?? o.verdict;
    if (typeof preferred === "string" && preferred.trim()) return preferred.trim();
    for (const v of Object.values(o)) if (typeof v === "string" && v.trim()) return v.trim();
  }
  return args.length > 140 ? args.slice(0, 140) + "…" : args;
}

export const TOOL_LEGEND = [
  { label: "Web Search",  fg: "oklch(38% 0.12 250)" },
  { label: "Page Fetch",  fg: "oklch(38% 0.12 300)" },
  { label: "WHOIS",       fg: "oklch(35% 0.1 190)" },
  { label: "Image OCR",   fg: "oklch(40% 0.13 335)" },
  { label: "PDF Parse",   fg: "oklch(40% 0.12 35)" },
  { label: "Wikipedia",   fg: "oklch(38% 0.1 265)" },
];

// ─── Verdict styles ────────────────────────────────────────────────────────────

export type VerdictKey = keyof typeof VERDICT_STYLES;
export const VERDICT_STYLES = {
  "值得投递":  { bg: "oklch(96% 0.03 145)",  fg: "oklch(28% 0.1 145)",  accent: "oklch(52% 0.13 145)", en: "Worth Applying" },
  "谨慎投递":  { bg: "oklch(96% 0.035 80)",  fg: "oklch(34% 0.1 80)",   accent: "oklch(60% 0.14 80)",  en: "Proceed with Caution" },
  "建议放弃":  { bg: "oklch(96% 0.035 25)",  fg: "oklch(33% 0.12 25)",  accent: "oklch(52% 0.16 25)",  en: "Recommend Skipping" },
  "靠谱":      { bg: "oklch(96% 0.03 145)",  fg: "oklch(28% 0.1 145)",  accent: "oklch(52% 0.13 145)", en: "Looks Legit" },
  "存疑":      { bg: "oklch(96% 0.035 80)",  fg: "oklch(34% 0.1 80)",   accent: "oklch(60% 0.14 80)",  en: "Suspicious" },
  "大概率有坑":{ bg: "oklch(96% 0.035 25)", fg: "oklch(33% 0.12 25)",  accent: "oklch(52% 0.16 25)",  en: "Likely a Scam" },
};

export function detectVerdict(text: string): VerdictKey | null {
  if (!text) return null;
  if (text.includes("大概率有坑") || text.toLowerCase().includes("likely a scam")) return "大概率有坑";
  if (text.includes("存疑") || text.toLowerCase().includes("suspicious")) return "存疑";
  if (text.includes("靠谱") || text.toLowerCase().includes("looks legit")) return "靠谱";
  if (text.includes("值得投递") || text.toLowerCase().includes("worth applying")) return "值得投递";
  if (text.includes("谨慎投递") || text.toLowerCase().includes("caution")) return "谨慎投递";
  if (text.includes("建议放弃") || text.toLowerCase().includes("recommend skipping")) return "建议放弃";
  return null;
}

// ─── Answer parsing ────────────────────────────────────────────────────────────

export function extractURLs(text: string): { url: string; domain: string; context: string }[] {
  const urls = text.match(/https?:\/\/[^\s\)\]\}，。、；""'']+/g) ?? [];
  const seen = new Set<string>();
  const out: { url: string; domain: string; context: string }[] = [];
  for (const url of urls) {
    const clean = url.replace(/[.,;:!?]+$/, "");
    if (seen.has(clean)) continue;
    seen.add(clean);
    let domain = clean;
    try { domain = new URL(clean).hostname; } catch {}
    const idx = text.indexOf(clean);
    const ctx = idx >= 0 ? text.slice(Math.max(0, idx - 40), idx + clean.length + 40).trim() : "";
    out.push({ url: clean, domain, context: ctx });
  }
  return out;
}

export function parseStructuredAnswer(text: string): ParsedAnswer {
  if (!text) return { verdictLabel: "", verdictRaw: "", verdictReason: "", facts: [], redFlags: [], sources: [], rawText: "" };

  const lines = text.split(/\r?\n/);
  const facts: ParsedAnswer["facts"] = [];
  const redFlags: ParsedAnswer["redFlags"] = [];
  let verdictLabel = "";
  let verdictRaw = "";
  let verdictReason = "";

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.startsWith("[Verdict]")) {
      const content = line.slice("[Verdict]".length).trim();
      const knownVerdicts = Object.keys(VERDICT_STYLES);
      // Separator between label and reason: the model emits a run of em/en
      // dashes ("——", " — ", "–") or a spaced ASCII hyphen. Split on the first.
      const sepMatch = content.match(/\s*[—–]+\s*|\s+-{1,2}\s+/);
      if (sepMatch && sepMatch.index !== undefined) {
        const candidate = content.slice(0, sepMatch.index).trim();
        const rest = content.slice(sepMatch.index + sepMatch[0].length).trim();
        const matched = knownVerdicts.find(v => candidate.includes(v));
        // 英文裁定标签（如 "Likely a Scam"）经 detectVerdict 归一到中文 canonical key，
        // 使 VERDICT_STYLES 样式查找始终命中（否则英文输出时裁定卡失去配色/徽章）。
        verdictRaw = candidate.replace(/^(裁定|verdict|判断)[：:]\s*/i, "").trim();
        verdictLabel = matched ?? detectVerdict(candidate) ?? verdictRaw;
        verdictReason = rest;
      } else {
        const matched = knownVerdicts.find(v => content.includes(v));
        verdictRaw = content.replace(/^(裁定|verdict|判断)[：:]\s*/i, "").trim();
        verdictLabel = matched ?? detectVerdict(content) ?? verdictRaw;
        verdictReason = "";
      }
    } else if (line.startsWith("[Fact]")) {
      let content = line.slice("[Fact]".length).trim();
      // Confidence at end: "— High", "— Medium", "— Low"
      const confMatch = content.match(/\s+[—–-]{1,2}\s+(High|Medium|Low)\s*$/);
      let confidence = "";
      if (confMatch) {
        confidence = confMatch[1];
        content = content.slice(0, content.length - confMatch[0].length).trim();
      }
      // Extract URL if present inline
      const urlMatch = content.match(/https?:\/\/[^\s，。、；""'']+/);
      const url = urlMatch ? urlMatch[0].replace(/[.,;:!?]+$/, "") : undefined;
      const cleanText = url ? content.replace(url, "").replace(/\s*[—–-]{1,2}\s*$/, "").trim() : content;
      facts.push({ text: cleanText, confidence, url });
    } else if (line.startsWith("[RedFlag]")) {
      const content = line.slice("[RedFlag]".length).trim();
      if (content) redFlags.push({ text: content });
    }
  }

  const sources = extractURLs(text);

  // Fallback: if no structured markers found, try to detect verdict from full text
  if (!verdictLabel) {
    const detected = detectVerdict(text);
    if (detected) {
      verdictLabel = detected;
      // Find reason after the verdict keyword
      const idx = text.indexOf(detected);
      if (idx >= 0) {
        verdictReason = text.slice(idx + detected.length).replace(/^\s*[—–-]\s*/, "").slice(0, 200).trim();
      }
    }
  }

  return { verdictLabel, verdictRaw, verdictReason, facts, redFlags, sources, rawText: text };
}

// Prefer engine-provided structured sources over regex-extracted ones (problem 5).
// Guarantees stable citations whenever the run actually retrieved anything.
export function withEngineSources(parsed: ParsedAnswer, sources?: SourceItem[]): ParsedAnswer {
  if (!sources || sources.length === 0) return parsed;
  return { ...parsed, sources };
}

// One conversation turn in a stage thread (user question + assistant answer).
export type ConvTurn = { question: string; answer: string };

/**
 * Pack the followup request with a ROLLING CONVERSATION WINDOW, not just the
 * last answer. Two failure modes this fixes (both hit in real use):
 *  - Conversational (non-verdict) replies like "请提供 JD 全文和简历" parse to
 *    zero structured fields — the old parsed-only context dropped them entirely,
 *    so a reply of "如图 + attachment" lost its meaning. Non-verdict answers now
 *    carry raw text excerpts.
 *  - Multi-turn requests ("upload your JD" → user uploads) need the *asking*
 *    turn visible, not just the latest one. We keep the last 3 turns.
 */
export function buildFollowupInput(
  question: string,
  turns: ConvTurn[],
  formSummary: string,
  materials?: { resume?: string; jd?: string },
): string {
  // Recent turns: verdict answers → compact structured form; conversational
  // answers → raw excerpt (they often contain requests/instructions to the user).
  const history = turns.slice(-3).map(t => {
    const parsed = parseStructuredAnswer(t.answer);
    const assistant = parsed.verdictLabel
      ? {
          verdict: parsed.verdictLabel,
          reason: parsed.verdictReason.slice(0, 250),
          key_facts: parsed.facts.slice(0, 6).map(f => f.text),
          red_flags: parsed.redFlags.slice(0, 4).map(f => f.text),
        }
      : t.answer.slice(0, 600);
    return { user: t.question.slice(0, 300), assistant };
  });

  // Citation continuity: sources from the most recent verdict-bearing turn.
  let priorSources: string[] = [];
  for (let i = turns.length - 1; i >= 0; i--) {
    const parsed = parseStructuredAnswer(turns[i].answer);
    if (parsed.verdictLabel) { priorSources = parsed.sources.slice(0, 8).map(s => s.url); break; }
  }

  const ctx: Record<string, unknown> = {
    original_task: formSummary.slice(0, 400),
    conversation_history: history,
  };
  if (priorSources.length > 0) ctx.prior_sources = priorSources;
  // User-provided materials from the case forms — so "compare my resume with
  // this JD" works even when the resume was entered turns/stages ago.
  const mat: Record<string, string> = {};
  if (materials?.resume?.trim()) mat.resume = materials.resume.trim().slice(0, 1500);
  if (materials?.jd?.trim()) mat.jd = materials.jd.trim().slice(0, 1500);
  if (Object.keys(mat).length > 0) ctx.user_materials = mat;

  return `[对话上下文 - 供参考]\n${JSON.stringify(ctx, null, 2)}\n\n[追问/补充信息]\n${question}`;
}

// ─── Cross-stage carryover ────────────────────────────────────────────────────
// One case = one job opportunity journeyed across the four stages. When a later
// stage starts, findings from completed earlier stages are packed into the task
// input as reference context, so e.g. stage 4 verifies the offer WITH the stage 1
// company research in hand ("持续辅助", not four isolated buttons). The engine's
// grounding gate still demands fresh evidence for any new verdict — carryover is
// context, not a substitute for retrieval.

const STAGE_ORDER: Stage[] = ["stage1", "stage2", "stage3", "stage4"];

/** Latest completed answer of a stage: last done followup wins over initial run. */
export function latestStageAnswer(ss: StageState): string | null {
  for (let i = ss.followups.length - 1; i >= 0; i--) {
    const r = ss.followups[i].runState;
    if (r.status === "done" && r.answer) return r.answer;
  }
  if (ss.initialRun.status === "done" && ss.initialRun.answer) return ss.initialRun.answer;
  return null;
}

/** Stages earlier in the journey than `current` that have a completed answer. */
export function completedEarlierStages(
  stageStates: Record<Stage, StageState>, current: Stage,
): Stage[] {
  const cut = STAGE_ORDER.indexOf(current);
  return STAGE_ORDER.slice(0, cut).filter(s => latestStageAnswer(stageStates[s]) !== null);
}

/** Compact JSON context block from completed earlier stages; null if none. */
export function buildCrossStageContext(
  stageStates: Record<Stage, StageState>, current: Stage,
): string | null {
  const carried = completedEarlierStages(stageStates, current);
  if (carried.length === 0) return null;

  const entries = carried.map(s => {
    const answer = latestStageAnswer(stageStates[s])!;
    const meta = STAGE_META[s];
    if (s === "stage2") {
      // Resume-fit output is a free-text checklist, not a verdict — carry an excerpt.
      return { stage: `${meta.num} ${meta.en}`, summary: answer.slice(0, 500) };
    }
    const parsed = parseStructuredAnswer(answer);
    return {
      stage: `${meta.num} ${meta.en}`,
      verdict: parsed.verdictLabel || "未知",
      reason: parsed.verdictReason.slice(0, 250),
      key_facts: parsed.facts.slice(0, 6).map(f => f.text),
      red_flags: parsed.redFlags.slice(0, 4).map(f => f.text),
      sources: parsed.sources.slice(0, 6).map(src => src.url),
    };
  });

  return `[本案早前阶段的已取证结论 - 供参考，新裁定仍须独立取证核实]\n${JSON.stringify(entries, null, 2)}`;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function buildInput(stage: Stage, forms: FormData): string {
  if (stage === "stage1") {
    const parts = [`Company: ${forms.company}`];
    if (forms.position) parts.push(`Position: ${forms.position}`);
    if (forms.resume) parts.push(`My background: ${forms.resume}`);
    if (forms.context) parts.push(`Additional context: ${forms.context}`);
    return parts.join("\n");
  }
  if (stage === "stage2") {
    return [`JD:\n${forms.jd}`, `Resume:\n${forms.resume}`, forms.target && `Target focus: ${forms.target}`].filter(Boolean).join("\n\n");
  }
  if (stage === "stage3") {
    return [`Recruiter messages:\n${forms.message}`, forms.recruiter && `Recruiter info: ${forms.recruiter}`, forms.context && `Context: ${forms.context}`].filter(Boolean).join("\n\n");
  }
  return [`Offer / contract:\n${forms.offer}`, forms.hr && `HR contact: ${forms.hr}`, forms.link && `Company website: ${forms.link}`].filter(Boolean).join("\n\n");
}

export function valid(stage: Stage, f: FormData): boolean {
  if (stage === "stage1") return !!f.company?.trim();
  if (stage === "stage2") return !!f.jd?.trim() && !!f.resume?.trim();
  if (stage === "stage3") return !!f.message?.trim();
  return !!f.offer?.trim();
}

export function summary(stage: Stage, f: FormData): string {
  const tr = (s: string, n = 80) => s && s.length > n ? s.slice(0, n) + "…" : s;
  if (stage === "stage1") return [f.company, f.position && `Role: ${f.position}`, f.context && tr(f.context)].filter(Boolean).join(" · ");
  if (stage === "stage2") return `JD: ${tr(f.jd, 60)} · Resume: ${tr(f.resume, 60)}`;
  if (stage === "stage3") return tr(f.message);
  return [tr(f.offer), f.hr && `HR: ${f.hr}`].filter(Boolean).join(" · ");
}

// ─── Shared styles ────────────────────────────────────────────────────────────

export const inputStyle: React.CSSProperties = {
  width: "100%", padding: "9px 11px", borderRadius: 8,
  border: "1px solid oklch(88% 0.012 70)", fontSize: 13.5,
  fontFamily: "var(--font-sans)", background: "white", color: "oklch(24% 0.02 50)", outline: "none",
  boxSizing: "border-box",
};
export const labelStyle: React.CSSProperties = {
  display: "block", fontSize: 12, fontWeight: 700, color: "oklch(35% 0.02 50)", marginBottom: 5,
};

// ─── Sub-components ───────────────────────────────────────────────────────────

export function Pulse({ color = "oklch(58% 0.13 40)" }: { color?: string }) {
  return <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%",
    background: color, animation: "pulse 1.1s ease-in-out infinite", flexShrink: 0 }} />;
}

export function Badge({ tool }: { tool: string }) {
  const m = tm(tool);
  return <span style={{ fontSize: 10.5, fontWeight: 700, padding: "3px 8px", borderRadius: 6,
    background: m.bg, color: m.fg, flexShrink: 0, whiteSpace: "nowrap" }}>{m.label}</span>;
}

export function DetailBlock({ label, text, maxH }: { label: string; text: string; maxH?: number }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, fontWeight: 700, color: "oklch(50% 0.02 50)", textTransform: "uppercase",
        letterSpacing: "0.04em", marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 12.5, fontFamily: "var(--font-mono)", color: "oklch(32% 0.02 50)",
        whiteSpace: "pre-wrap", lineHeight: 1.5, maxHeight: maxH ?? 400, overflowY: "auto" }}>{text}</div>
    </div>
  );
}

export function MetaEvents({ items }: { items: TraceItem[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ fontSize: 11.5, padding: "4px 0" }}>
      <div onClick={() => setOpen(p => !p)} style={{
        color: "oklch(52% 0.02 50)", cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          Engine Log ({items.length})
        </span>
        <span style={{ fontSize: 10 }}>{open ? "▾" : "▸"}</span>
      </div>
      {open && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 4, marginLeft: 12,
          borderLeft: "2px solid oklch(92% 0.01 70)", paddingLeft: 12,
          maxHeight: 240, overflowY: "auto" }}>
          {items.map((m, i) => (
            <div key={i} style={{ fontSize: 11, fontFamily: "var(--font-mono)",
              color: m.type === "warn" ? "oklch(50% 0.13 80)" : "oklch(48% 0.02 50)", padding: "2px 0" }}>
              {m.metaKind === "trial" && <><span style={{ fontWeight: 600 }}>🔬 Trial {m.metaDetail}</span></>}
              {m.metaKind === "verifier" && <>
                <span style={{ fontWeight: 600, color: m.ok ? "oklch(45% 0.12 145)" : "oklch(50% 0.14 25)" }}>
                  {m.ok ? "✓" : "✗"} Verifier</span>
                {m.metaDetail && <span style={{ color: "oklch(48% 0.02 50)" }}> — {m.metaDetail}</span>}
              </>}
              {m.metaKind === "correction" && <><span style={{ fontWeight: 600 }}>⚡ Correction:</span> {m.metaDetail}</>}
              {m.metaKind === "retry" && <><span style={{ fontWeight: 600, color: "oklch(52% 0.13 40)" }}>🔄 {m.metaDetail}</span></>}
              {m.metaKind === "evidence_gate" && <><span style={{ fontWeight: 600 }}>🔒 Evidence gate:</span> {m.metaDetail}</>}
              {m.metaKind === "step_start" && <span style={{ color: "oklch(55% 0.02 50)" }}>Model: {m.metaDetail}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function TraceStepCard({ s, index }: { s: TraceItem; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const running = s.status === "running";
  const statusLabel = running ? "Running…" :
    s.status === "success" ? "Success" : s.status === "notfound" ? "No result" :
    s.status === "cache" ? "Cache hit" : "Error";
  const statusFg = running ? "oklch(58% 0.13 40)" :
    s.status === "success" ? "oklch(45% 0.12 145)" :
    s.status === "notfound" ? "oklch(52% 0.02 50)" :
    s.status === "cache" ? "oklch(45% 0.1 265)" : "oklch(50% 0.14 25)";

  return (
    <div style={{ border: "1px solid oklch(90% 0.012 70)", borderRadius: 11,
      background: "white", overflow: "hidden" }}>
      <div onClick={() => setExpanded(p => !p)} style={{ display: "flex", alignItems: "center", gap: 10,
        padding: "10px 14px", cursor: "pointer" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "oklch(55% 0.02 50)",
          width: 20, flexShrink: 0 }}>{index + 1}</span>
        {s.tool && <Badge tool={s.tool} />}
        <span style={{ fontSize: 13, color: "oklch(28% 0.02 50)", flex: 1, minWidth: 0,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          fontFamily: "var(--font-mono)" }}>
          {s.label ?? s.tool}
        </span>
        {running && <Pulse />}
        <span style={{ fontSize: 11, fontWeight: 600, color: statusFg, flexShrink: 0 }}>{statusLabel}</span>
        <span style={{ fontSize: 11, color: "oklch(60% 0.02 50)", flexShrink: 0, width: 14, textAlign: "center" }}>
          {expanded ? "▾" : "▸"}
        </span>
      </div>
      {expanded && (
        <div style={{ padding: "0 14px 14px 44px", display: "flex", flexDirection: "column", gap: 10 }}>
          {s.args && <DetailBlock label="Input" text={s.args} />}
          {s.observation && <DetailBlock label="Result" text={s.observation} maxH={200} />}
        </div>
      )}
    </div>
  );
}

export function TraceView({ trace, isDone }: { trace: TraceItem[]; isDone: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);
  const steps = trace.filter(t => t.type === "step");
  const metas = trace.filter(t => t.type !== "step");

  // Auto-scroll to latest item ONLY while the investigation is live. Once done,
  // expanding the trace is a "look back" — auto-scrolling then would jump the
  // collapse toggle out of view (esp. for an earlier turn higher in the thread).
  useEffect(() => {
    if (isDone) return;
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [trace.length, isDone]);

  if (trace.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 12 }}>
      {metas.length > 0 && <MetaEvents items={metas} />}
      {steps.map((s, i) => <TraceStepCard key={s.id} s={s} index={i} />)}
      <div ref={endRef} />
    </div>
  );
}

// ─── Structured result cards ───────────────────────────────────────────────────

export function CollapsibleCard({ title, defaultOpen = true, accent, children }: {
  title: string; defaultOpen?: boolean; accent?: string; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ background: "white", border: `1px solid ${accent ?? "oklch(90% 0.012 70)"}`,
      borderRadius: 14, marginBottom: 12 }}>
      <div onClick={() => setOpen(p => !p)} style={{ display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "14px 20px", cursor: "pointer", userSelect: "none" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "oklch(26% 0.02 50)" }}>{title}</div>
        <span style={{ fontSize: 12, color: "oklch(55% 0.02 50)" }}>{open ? "Hide ▾" : "Show ▸"}</span>
      </div>
      {open && <div style={{ padding: "0 20px 18px" }}>{children}</div>}
    </div>
  );
}

// Citation marker ([n]) shared by facts and the chat summary.
// Verified source → opens the original page directly (Perplexity pattern).
// Unverified / no-URL → warning tint, jumps to the source strip so the user sees the ⚠️.
export function Cite({ n, source, onJump }: { n: number; source?: SourceItem; onJump?: () => void }) {
  const base: React.CSSProperties = {
    fontSize: 10.5, fontWeight: 700, padding: "0 3px", borderRadius: 4,
    fontFamily: "var(--font-mono)", lineHeight: 1.6, textDecoration: "none", whiteSpace: "nowrap",
  };
  const verified = source ? source.verified !== false : true; // regex fallback treated as clickable
  if (source?.url && verified) {
    return (
      <a href={source.url} target="_blank" rel="noopener noreferrer" title={source.domain}
        style={{ ...base, cursor: "pointer", color: "oklch(45% 0.13 250)", background: "oklch(95% 0.03 250)" }}>
        [{n}]
      </a>
    );
  }
  return (
    <span onClick={onJump} title={source ? "未验证来源 · 点击查看" : "查看来源"}
      style={{ ...base, cursor: "pointer", color: "oklch(48% 0.13 80)", background: "oklch(95% 0.04 80)" }}>
      [{n}]{source && "⚠"}
    </span>
  );
}

// Render LLM markdown (headings / bold / lists / tables) with compact, inline-styled
// elements. Color is inherited from the parent so callers control it.
export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: (p) => <div style={{ fontSize: 15, fontWeight: 800, margin: "12px 0 6px" }} {...p} />,
        h2: (p) => <div style={{ fontSize: 14, fontWeight: 800, margin: "12px 0 5px" }} {...p} />,
        h3: (p) => <div style={{ fontSize: 13.5, fontWeight: 700, margin: "10px 0 4px" }} {...p} />,
        p: (p) => <p style={{ margin: "5px 0" }} {...p} />,
        ul: (p) => <ul style={{ margin: "5px 0", paddingLeft: 18 }} {...p} />,
        ol: (p) => <ol style={{ margin: "5px 0", paddingLeft: 18 }} {...p} />,
        li: (p) => <li style={{ margin: "2px 0" }} {...p} />,
        strong: (p) => <strong style={{ fontWeight: 700 }} {...p} />,
        a: (p) => <a target="_blank" rel="noopener noreferrer" style={{ color: "oklch(45% 0.13 250)" }} {...p} />,
        code: (p) => <code style={{ background: "oklch(95% 0.01 70)", padding: "1px 4px", borderRadius: 4, fontFamily: "var(--font-mono)", fontSize: "0.92em" }} {...p} />,
        hr: () => <div style={{ height: 1, background: "oklch(90% 0.012 70)", margin: "12px 0" }} />,
      }}
    >
      {children}
    </ReactMarkdown>
  );
}

export function StructuredResult({ parsed, isFollowup = false, anchorBase }: {
  parsed: ParsedAnswer; isFollowup?: boolean; anchorBase?: string;
}) {
  const verdict = parsed.verdictLabel as VerdictKey | "";
  const vs = verdict ? VERDICT_STYLES[verdict as VerdictKey] : null;
  const [sourcesOpen, setSourcesOpen] = useState(true);
  // Map each source URL to its index so facts can cite [n] into the source list.
  const srcIndex = new Map(parsed.sources.map((s, i) => [s.url, i] as const));
  const citeJump = (i: number) => { if (anchorBase) jumpToAnchor(`${anchorBase}-src-${i}`); };

  return (
    <div>
      {/* Verdict header card */}
      {verdict && vs ? (
        <div style={{ border: `1.5px solid ${vs.accent}`, background: vs.bg, borderRadius: 16,
          padding: "18px 22px", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={{ display: "inline-flex", alignItems: "center", background: vs.accent,
              color: "white", padding: "7px 16px", borderRadius: 999, fontSize: 15, fontWeight: 800 }}>
              {vs.en}
            </span>
            {parsed.verdictRaw && parsed.verdictRaw !== vs.en && (
              <span style={{ fontSize: 12, fontWeight: 700, color: vs.fg,
                opacity: 0.7 }}>{parsed.verdictRaw}</span>
            )}
          </div>
          {parsed.verdictReason && (
            <div style={{ marginTop: 12, fontSize: 14, lineHeight: 1.7, color: vs.fg,
              wordBreak: "break-word" }}>
              <Markdown>{parsed.verdictReason}</Markdown>
            </div>
          )}
        </div>
      ) : (
        /* No structured verdict — check for any answer text */
        parsed.rawText && !parsed.verdictLabel && (
          <div style={{ background: "white", border: "1px solid oklch(90% 0.012 70)", borderRadius: 14,
            padding: "16px 20px", marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "oklch(35% 0.02 50)", marginBottom: 8 }}>
              {isFollowup ? "Follow-up Result" : "Investigation Result"}
            </div>
            <div style={{ fontSize: 13.5, lineHeight: 1.7, color: "oklch(28% 0.02 50)",
              wordBreak: "break-word", maxHeight: 500, overflowY: "auto" }}>
              <Markdown>{parsed.rawText}</Markdown>
            </div>
          </div>
        )
      )}

      {/* Facts */}
      {parsed.facts.length > 0 && (
        <CollapsibleCard title={`Verified Facts · ${parsed.facts.length}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {parsed.facts.map((f, i) => (
              <div key={i} style={{ display: "flex", gap: 10, paddingBottom: 10,
                borderBottom: i < parsed.facts.length - 1 ? "1px solid oklch(93% 0.01 70)" : "none" }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "oklch(52% 0.13 145)",
                  marginTop: 7, flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: 13.5, lineHeight: 1.6, color: "oklch(28% 0.02 50)" }}>{f.text}</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 3, alignItems: "center" }}>
                    {(() => { const fi = f.url ? srcIndex.get(f.url) : undefined;
                      return fi !== undefined
                        ? <Cite n={fi + 1} source={parsed.sources[fi]} onJump={() => citeJump(fi)} /> : null; })()}
                    {f.url && (
                      <a href={f.url} target="_blank" rel="noopener noreferrer" style={{
                        fontSize: 11, color: "oklch(38% 0.1 250)", fontFamily: "var(--font-mono)" }}>
                        {(() => { try { return new URL(f.url).hostname; } catch { return f.url.slice(0, 40); } })()}
                      </a>
                    )}
                    {f.confidence && (
                      <span style={{ fontSize: 10.5, fontWeight: 700, padding: "1px 6px", borderRadius: 4,
                        background: f.confidence === "High" ? "oklch(94% 0.04 145)" : f.confidence === "Low" ? "oklch(94% 0.04 25)" : "oklch(94% 0.03 80)",
                        color: f.confidence === "High" ? "oklch(38% 0.11 145)" : f.confidence === "Low" ? "oklch(42% 0.13 25)" : "oklch(40% 0.1 80)" }}>
                        {f.confidence}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CollapsibleCard>
      )}

      {/* RedFlags */}
      {parsed.redFlags.length > 0 && (
        <CollapsibleCard title={`Red Flags · ${parsed.redFlags.length}`} accent="oklch(82% 0.05 25)">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {parsed.redFlags.map((f, i) => (
              <div key={i} style={{ display: "flex", gap: 10 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "oklch(52% 0.16 25)",
                  marginTop: 7, flexShrink: 0 }} />
                <div style={{ fontSize: 13.5, lineHeight: 1.6, color: "oklch(30% 0.02 50)" }}>{f.text}</div>
              </div>
            ))}
          </div>
        </CollapsibleCard>
      )}

      {/* If no structured facts/flags but has raw text with verdict, show raw text collapsed */}
      {verdict && parsed.facts.length === 0 && parsed.redFlags.length === 0 && parsed.rawText && (
        <CollapsibleCard title="Details" defaultOpen={false}>
          <div style={{ fontSize: 13.5, lineHeight: 1.7, color: "oklch(28% 0.02 50)",
            whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 400, overflowY: "auto" }}>
            {parsed.rawText}
          </div>
        </CollapsibleCard>
      )}

      {/* Sources */}
      {parsed.sources.length > 0 && (
        <div style={{ background: "oklch(99.5% 0.004 70)", border: "1px solid oklch(90% 0.012 70)",
          borderRadius: 14, marginBottom: 12 }}>
          <div onClick={() => setSourcesOpen(p => !p)} style={{ display: "flex", alignItems: "center",
            justifyContent: "space-between", padding: "14px 20px", cursor: "pointer" }}>
            <div style={{ fontSize: 13.5, fontWeight: 700 }}>Sources · {parsed.sources.length}</div>
            <span style={{ fontSize: 12, color: "oklch(55% 0.02 50)" }}>{sourcesOpen ? "Hide ▾" : "Show ▸"}</span>
          </div>
          {sourcesOpen && (
            <div style={{ padding: "0 20px 16px", display: "flex", flexDirection: "column", gap: 2 }}>
              <div style={{ fontSize: 11, color: "oklch(58% 0.02 50)", marginBottom: 6 }}>
                Click a citation ([n]) above to open its source; ⚠️ = not found in the retrieval log.
              </div>
              {/* Compact numbered domain strip — legend + trust overview + ⚠️ surfacing */}
              {parsed.sources.map((s, i) => (
                <a key={i} id={anchorBase ? `${anchorBase}-src-${i}` : undefined}
                  href={s.url} target="_blank" rel="noopener noreferrer"
                  style={{ display: "flex", alignItems: "center", gap: 7, padding: "5px 6px",
                    scrollMarginTop: 12, borderRadius: 6, textDecoration: "none",
                    background: s.verified === false ? "oklch(97% 0.02 80)" : "transparent" }}>
                  <span style={{ fontSize: 10.5, fontWeight: 700, color: "oklch(55% 0.02 50)",
                    fontFamily: "var(--font-mono)", flexShrink: 0 }}>[{i + 1}]</span>
                  <span style={{ color: "oklch(38% 0.1 250)", fontWeight: 600, fontFamily: "var(--font-mono)",
                    fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {s.domain}
                  </span>
                  {s.verified === false && (
                    <span title="Not found in this run's retrieval log" style={{ fontSize: 10, fontWeight: 700,
                      color: "oklch(45% 0.13 80)", flexShrink: 0 }}>⚠️</span>
                  )}
                </a>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Inline trace (in the chat flow) ────────────────────────────────────────────
// Running: a live one-liner of the current step. Done: a collapsed summary bar
// ("Investigated in N steps · X searches · …") that expands to the full TraceView.

export function traceCounts(trace: TraceItem[]) {
  const steps = trace.filter(t => t.type === "step");
  let search = 0, fetch = 0, whois = 0, ocr = 0, pdf = 0, cache = 0, other = 0;
  for (const s of steps) {
    if (s.status === "cache") { cache++; continue; }
    const label = tm(s.tool ?? "").label;
    if (label === "Web Search" || label === "Wikipedia") search++;
    else if (label === "Page Fetch") fetch++;
    else if (label === "WHOIS") whois++;
    else if (label === "Image OCR") ocr++;
    else if (label === "PDF Parse") pdf++;
    else other++;
  }
  return { total: steps.length, search, fetch, whois, ocr, pdf, cache, other };
}

export function summaryParts(c: ReturnType<typeof traceCounts>): string[] {
  const p: string[] = [];
  const s = (n: number, one: string, many = one + "s") => n > 0 && p.push(`${n} ${n > 1 ? many : one}`);
  s(c.search, "search", "searches");
  s(c.fetch, "page fetched", "pages fetched");
  s(c.whois, "WHOIS lookup", "WHOIS lookups");
  s(c.ocr, "image read");
  s(c.pdf, "PDF parsed");
  s(c.cache, "cache hit");
  return p;
}

export function InlineTrace({ trace, isDone, stepCount, maxSteps, currentTrial, maxTrials, label }: {
  trace: TraceItem[]; isDone: boolean; stepCount: number; maxSteps?: number;
  currentTrial?: number; maxTrials?: number; label?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const steps = trace.filter(t => t.type === "step");
  const last = steps[steps.length - 1];
  const c = traceCounts(trace);

  // Running: live status line + last-step badge.
  if (!isDone) {
    return (
      <div style={{ maxWidth: "92%", border: "1px solid oklch(90% 0.012 70)",
        background: "oklch(99.5% 0.004 70)", borderRadius: "3px 12px 12px 12px", padding: "11px 14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: 700,
          color: "oklch(30% 0.02 50)" }}>
          <Pulse />
          <span>{label ?? "Investigating"}…</span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, fontWeight: 600,
            color: "oklch(52% 0.02 50)" }}>
            {maxTrials && maxTrials > 1 && currentTrial ? `Trial ${currentTrial}/${maxTrials} · ` : ""}
            Step {stepCount}{maxSteps ? `/${maxSteps}` : ""}
          </span>
        </div>
        {last && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8,
            fontSize: 12.5, color: "oklch(40% 0.02 50)" }}>
            {last.tool && <Badge tool={last.tool} />}
            <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis",
              whiteSpace: "nowrap", fontFamily: "var(--font-mono)" }}>
              {last.brief || last.label || last.tool}
            </span>
          </div>
        )}
      </div>
    );
  }

  // Done: collapsed summary bar, expandable to full step detail.
  const parts = summaryParts(c);
  return (
    <div style={{ maxWidth: "92%", border: "1px solid oklch(90% 0.012 70)",
      background: "oklch(99.5% 0.004 70)", borderRadius: "3px 12px 12px 12px", overflow: "hidden" }}>
      <div onClick={() => setExpanded(p => !p)} style={{ display: "flex", alignItems: "center",
        gap: 8, padding: "10px 14px", cursor: "pointer", userSelect: "none" }}>
        <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%",
          background: "oklch(52% 0.13 145)", flexShrink: 0 }} />
        <span style={{ fontSize: 12.5, color: "oklch(38% 0.02 50)", flex: 1, minWidth: 0 }}>
          <b style={{ color: "oklch(30% 0.02 50)" }}>Investigated in {c.total} step{c.total !== 1 ? "s" : ""}</b>
          {parts.length > 0 && <span> · {parts.join(" · ")}</span>}
        </span>
        <span style={{ fontSize: 11.5, color: "oklch(55% 0.02 50)", flexShrink: 0 }}>
          {expanded ? "Hide steps ▾" : "View steps ▸"}
        </span>
      </div>
      {expanded && (
        <div style={{ padding: "0 14px 12px" }}>
          <TraceView trace={trace} isDone={isDone} />
        </div>
      )}
    </div>
  );
}

// ─── Stage forms ──────────────────────────────────────────────────────────────

export function StageForm({ stage, forms, onChange }: {
  stage: Stage; forms: FormData; onChange: (f: string, v: string) => void;
}) {
  const f = (k: string) => (forms[k] ?? "");
  const ch = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => onChange(k, e.target.value);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {STAGE_FIELDS[stage].map(fld => (
        <div key={fld.key}>
          <label style={labelStyle}>
            {fld.label}
            {fld.required && <span style={{ color: "oklch(55% 0.16 25)" }}> *</span>}
          </label>
          {fld.area ? (
            <textarea value={f(fld.key)} onChange={ch(fld.key)} rows={fld.rows ?? 3}
              placeholder={fld.placeholder} style={{ ...inputStyle, resize: "vertical" }} />
          ) : (
            <input value={f(fld.key)} onChange={ch(fld.key)}
              placeholder={fld.placeholder} style={inputStyle} />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Chat summary (Phase 1) ─────────────────────────────────────────────────
// 从已解析的裁定数据拼出对话摘要——不新增 LLM 调用，只重排 agent 已取证的内容，
// 因此不会引入幻觉（见 SPEC.md：聊天摘要复用 submit_verdict 结论）。

export function firstSentences(text: string, n: number): string {
  if (!text) return "";
  const stripped = text.replace(/\[(Verdict|Fact|RedFlag|Source)\][^\n]*/g, "").trim();
  if (!stripped) return "";
  const sentences = stripped.split(/(?<=[。！？.!?])\s*/).filter(Boolean);
  return sentences.slice(0, n).join("").slice(0, 300).trim();
}

export function jumpToAnchor(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  // 找最近的可滚动祖先（画布是独立滚动容器）
  let scroller: HTMLElement | null = el.parentElement;
  while (scroller) {
    const oy = getComputedStyle(scroller).overflowY;
    if (oy === "auto" || oy === "scroll") break;
    scroller = scroller.parentElement;
  }
  if (scroller) {
    const s = scroller;
    const target = s.scrollTop
      + el.getBoundingClientRect().top - s.getBoundingClientRect().top - 12;
    s.scrollTo?.({ top: target, behavior: "smooth" });
    // 兜底：部分环境平滑滚动会静默失效，200ms 后仍未到位则直接定位
    window.setTimeout(() => {
      if (Math.abs(s.scrollTop - target) > 8) s.scrollTop = target;
    }, 200);
  } else {
    el.scrollIntoView({ block: "start" });
  }
  // 高亮闪一下：即使内容已在视口内、无需滚动，也给用户明确的点击反馈
  el.style.transition = "box-shadow 0.35s ease";
  el.style.boxShadow = "0 0 0 3px oklch(62% 0.15 40)";
  window.setTimeout(() => { el.style.boxShadow = "none"; }, 1100);
}

export function ChatSummary({ parsed, prevVerdict, latencyMs, trials, jumpTargetId,
  summaryForUser, suggestedFollowups, onFollowup }: {
  parsed: ParsedAnswer; prevVerdict?: string; latencyMs?: number; trials?: number; jumpTargetId?: string;
  summaryForUser?: string; suggestedFollowups?: string[]; onFollowup?: (q: string) => void;
}) {
  const vs = parsed.verdictLabel && (parsed.verdictLabel in VERDICT_STYLES)
    ? VERDICT_STYLES[parsed.verdictLabel as VerdictKey] : null;
  const factCount = parsed.facts.length;
  const srcCount = parsed.sources.length;
  const flags = parsed.redFlags;

  // Compare against the prior verdict on follow-ups: "unchanged" vs "updated to".
  const verdictEn = parsed.verdictLabel && (parsed.verdictLabel in VERDICT_STYLES)
    ? VERDICT_STYLES[parsed.verdictLabel as VerdictKey].en : parsed.verdictLabel;
  let verdictVerb = "Verdict";
  if (prevVerdict !== undefined && parsed.verdictLabel) {
    verdictVerb = parsed.verdictLabel === prevVerdict ? "Verdict holds" : "Verdict updated to";
  }

  // Non-verdict answers (e.g. Stage 2 resume analysis) have no structured card —
  // show the FULL text here (rendered markdown), not a 2-sentence teaser that cuts off.
  const fallback = parsed.rawText;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Verdict — a compact colored pill for a glance. The full labelled card
          (with the reason) lives on the Evidence Board; we don't repeat it here. */}
      {parsed.verdictLabel ? (
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {prevVerdict !== undefined && (
            <span style={{ fontSize: 12, color: "oklch(52% 0.02 50)" }}>
              {verdictVerb === "Verdict holds" ? "Verdict holds:" : "Updated to:"}
            </span>
          )}
          <span style={{ display: "inline-flex", alignItems: "center",
            background: vs?.accent ?? "oklch(52% 0.02 50)", color: "white",
            padding: "3px 11px", borderRadius: 999, fontSize: 12.5, fontWeight: 800 }}>
            {verdictEn}
          </span>
        </div>
      ) : (
        <div style={{ lineHeight: 1.6 }}>{fallback ? <Markdown>{fallback}</Markdown> : "Investigation complete — see the Evidence Board."}</div>
      )}

      {/* User-facing summary (engine-provided, grounded). Falls back to the verdict
          reason only when no summary was produced (older / free-text answers). */}
      {summaryForUser ? (
        <div style={{ lineHeight: 1.65, color: "oklch(32% 0.02 50)" }}><Markdown>{summaryForUser}</Markdown></div>
      ) : (
        parsed.verdictLabel && parsed.verdictReason && (
          <div style={{ lineHeight: 1.65, color: "oklch(32% 0.02 50)" }}><Markdown>{parsed.verdictReason}</Markdown></div>
        )
      )}

      {/* Red-flag heads-up */}
      {flags.length > 0 && (
        <div style={{ color: "oklch(45% 0.12 25)", lineHeight: 1.6 }}>
          ⚠️ {flags.length} risk signal{flags.length > 1 ? "s" : ""}: {flags.slice(0, 2).map(f => f.text).join("; ")}
          {flags.length > 2 ? " …" : ""}
        </div>
      )}

      {/* Evidence count + click-through to the board */}
      {(factCount > 0 || srcCount > 0) && (
        <div style={{ fontSize: 12.5, color: "oklch(52% 0.02 50)" }}>
          {factCount} fact{factCount !== 1 ? "s" : ""} verified{srcCount > 0 ? ` · ${srcCount} source${srcCount !== 1 ? "s" : ""}` : ""} —{" "}
          {jumpTargetId ? (
            <span onClick={() => jumpToAnchor(jumpTargetId)}
              style={{ color: "oklch(50% 0.15 250)", cursor: "pointer",
                textDecoration: "underline", textUnderlineOffset: 2, fontWeight: 600 }}>
              see full evidence →
            </span>
          ) : "see the Evidence Board →"}
        </div>
      )}

      {/* Inline citations — verified opens the source directly; unverified jumps to the board */}
      {jumpTargetId && srcCount > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
          <span style={{ fontSize: 11.5, color: "oklch(58% 0.02 50)" }}>Cited:</span>
          {parsed.sources.slice(0, 10).map((s, i) => (
            <Cite key={i} n={i + 1} source={s} onJump={() => jumpToAnchor(`${jumpTargetId}-src-${i}`)} />
          ))}
        </div>
      )}

      {/* Suggested follow-ups (engine-provided) — click to run as a grounded turn */}
      {suggestedFollowups && suggestedFollowups.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 2 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "oklch(52% 0.02 50)",
            textTransform: "uppercase", letterSpacing: "0.04em" }}>Suggested next steps</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {suggestedFollowups.slice(0, 3).map((q, i) => (
              <button key={i} onClick={() => onFollowup?.(q)} disabled={!onFollowup}
                style={{ textAlign: "left", background: "oklch(97% 0.015 250)",
                  border: "1px solid oklch(88% 0.03 250)", borderRadius: 9, padding: "7px 11px",
                  fontSize: 12.5, lineHeight: 1.45, color: "oklch(35% 0.06 250)",
                  fontFamily: "var(--font-sans)", cursor: onFollowup ? "pointer" : "default" }}>
                ↳ {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Follow-up invite + latency */}
      <div style={{ fontSize: 12.5, color: "oklch(55% 0.02 50)" }}>
        Ask a follow-up to dig into any point.
        {latencyMs ? (
          <span style={{ marginLeft: 6 }}>
            ({Math.round(latencyMs / 1000)}s{trials ? ` · ${trials} trial` : ""})
          </span>
        ) : null}
      </div>
    </div>
  );
}
