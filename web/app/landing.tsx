"use client";

import { useEffect, useState } from "react";

// ─── Landing / entry screen ─────────────────────────────────────────────────
// Doubles as the product's promo frame and the gate in front of the chat UI.
// The right panel replays the engine's own stage-4 demo case (DEMO_FORMS.stage4)
// using real engine vocabulary — tool chips, evidence gate, verifier, verdict —
// so what the landing promises is exactly what the app delivers.

const INK = "oklch(24% 0.02 50)";
const FOG = "oklch(46% 0.02 50)";
const FAINT = "oklch(58% 0.02 50)";
const LINE = "oklch(90% 0.012 70)";
const PAPER = "oklch(98% 0.01 70)";
const ACCENT = "oklch(58% 0.13 40)";
const ALARM = "oklch(52% 0.16 25)";
const ALARM_INK = "oklch(33% 0.12 25)";
const OK_GREEN = "oklch(52% 0.13 145)";
const MONO = "var(--font-mono, 'JetBrains Mono', monospace)";
const SANS = "var(--font-sans, Manrope, sans-serif)";

const HEADLINE = "Verify before\nyou trust.";
const ACCENT_CHARS = 6; // "Verify" renders in terracotta

// Simulated trace — mirrors the tool set / event kinds the engine actually emits.
type TraceRow =
  | { kind: "step"; tool: string; chipBg: string; chipFg: string; text: string; dot: string }
  | { kind: "gate"; text: string }
  | { kind: "verify"; text: string }
  | { kind: "verdict" };

const SEARCH_BG = "oklch(93% 0.03 250)";
const SEARCH_FG = "oklch(38% 0.12 250)";

const TRACE: TraceRow[] = [
  { kind: "step", tool: "Web Search", chipBg: SEARCH_BG, chipFg: SEARCH_FG, dot: "oklch(62% 0.14 80)",
    text: "“Nexora Global Solutions Ltd” — no official site, no registry record" },
  { kind: "gate", text: "Evidence gate — verdict blocked: 0 verified sources. Keep investigating." },
  { kind: "step", tool: "Web Search", chipBg: SEARCH_BG, chipFg: SEARCH_FG, dot: OK_GREEN,
    text: "signing entity “Everbright Holdings (HK)” — no tie to the hiring company" },
  { kind: "step", tool: "Web Search", chipBg: SEARCH_BG, chipFg: SEARCH_FG, dot: OK_GREEN,
    text: "“$320 equipment kit” + USDT weekly pay — advance-fee scam pattern" },
  { kind: "verify", text: "Verifier — 4/4 claims entailed by cited sources" },
  { kind: "verdict" },
];
// Static promo export is read at ~0.42× inside WeChat articles: keep the
// process layer terse (one row fewer) and let the verdict layer dominate.
const PROMO_TRACE = TRACE.filter((_, i) => i !== 2);
const STEP_MS = 1050;
const HOLD_MS = 6200;

const RED_FLAGS = [
  "Pay-to-work: $320 up-front “equipment kit”",
  "Salary in USDT, weekly — Telegram-only HR, no corporate email",
  "Signing entity ≠ hiring company · 24-hour pressure deadline",
];
// Promo (CN poster) rule: process rows stay in English (engine-log authenticity);
// the conclusion layer — slogan, stamp, red flags — speaks Chinese.
const RED_FLAGS_CN = [
  "先交钱才上岗：$320 预付“设备费”",
  "周薪用 USDT 加密货币，HR 只有 Telegram，无企业邮箱",
  "签约主体 ≠ 招聘公司 · “24 小时内答复”限时施压",
];
const CJK = "'PingFang SC', 'Noto Sans SC', sans-serif";

const STAGES = [
  { num: "01", name: "Role Research", desc: "Real company? Live opening, or a ghost job?" },
  { num: "02", name: "Resume Fit", desc: "Prioritized gaps to close before you apply." },
  { num: "03", name: "Message Check", desc: "Is the recruiter who they claim to be?" },
  { num: "04", name: "Offer Verification", desc: "The deepest cross-check, before you sign." },
];
// Promo band shows stage names only — descriptions don't survive WeChat scaling.
const STAGES_CN = ["选岗调研", "简历定向", "沟通证伪", "Offer 核验"];

// Hero workflow — a real four-step component, not a line of dev-note text.
const FLOW_EN = [
  { t: "Submit", d: "offer · JD · message" },
  { t: "Investigate", d: "multi-source research" },
  { t: "Verify", d: "independent re-check" },
  { t: "Verdict", d: "evidence-backed" },
];
const FLOW_CN = [
  { t: "提交机会", d: "Offer / JD / 沟通记录" },
  { t: "多源调查", d: "搜索并交叉验证" },
  { t: "独立复核", d: "复核关键判断" },
  { t: "风险结论", d: "结论均附依据" },
];


function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="OfferCheck logo">
      <defs>
        <linearGradient id="oc-shield-landing" x1="16" y1="2.5" x2="16" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#CF6A44" />
          <stop offset="1" stopColor="#A94A2C" />
        </linearGradient>
      </defs>
      <path d="M16 2.5 L27 6.6 V15 C27 22 22.2 27.8 16 30 C9.8 27.8 5 22 5 15 V6.6 Z" fill="url(#oc-shield-landing)" />
      <circle cx="14.2" cy="14" r="5" stroke="#FFFFFF" strokeWidth="2.3" fill="none" />
      <line x1="17.9" y1="17.7" x2="21.6" y2="21.4" stroke="#FFFFFF" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M11.7 14.1 L13.5 15.9 L16.9 12" stroke="#FFFFFF" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

export default function Landing({ onEnter }: { onEnter: (opts?: { demo?: boolean }) => void }) {
  const [reduced, setReduced] = useState(false);
  const [typedN, setTypedN] = useState(0);
  const [phase, setPhase] = useState(0);   // how many TRACE rows are visible
  const [fading, setFading] = useState(false);
  // ?promo — promo-poster variant: bigger type, hides dev chrome. Chinese by
  // default (?promo / ?promo=1); ?promo=en renders the same poster in English.
  // The regular entry screen stays English-only.
  const [promo, setPromo] = useState(false);
  const [promoZh, setPromoZh] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const fn = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (!params.has("promo")) return;
    setPromo(true);
    setPromoZh(params.get("promo") !== "en");   // ?promo=en → English poster
    const s = document.createElement("style");
    s.textContent = "nextjs-portal, #devtools-indicator { display: none !important; }";
    document.head.appendChild(s);
    return () => s.remove();
  }, []);

  // Headline types itself — the agent writing its conclusion.
  useEffect(() => {
    if (reduced) { setTypedN(HEADLINE.length); return; }
    let i = 0;
    const iv = setInterval(() => {
      i += 1;
      setTypedN(i);
      if (i >= HEADLINE.length) clearInterval(iv);
    }, 42);
    return () => clearInterval(iv);
  }, [reduced]);

  const rows = promo ? PROMO_TRACE : TRACE;
  const total = rows.length;

  // Trace loop: steps land one by one → verdict stamps → hold → fade → replay.
  useEffect(() => {
    if (reduced) { setPhase(total); setFading(false); return; }
    const timers: ReturnType<typeof setTimeout>[] = [];
    const run = () => {
      timers.length = 0;
      setFading(false);
      setPhase(0);
      for (let i = 1; i <= total; i += 1) {
        timers.push(setTimeout(() => setPhase(i), 700 + i * STEP_MS));
      }
      const end = 700 + total * STEP_MS + HOLD_MS;
      timers.push(setTimeout(() => setFading(true), end));
      timers.push(setTimeout(run, end + 480));
    };
    run();
    return () => timers.forEach(clearTimeout);
  }, [reduced, total]);

  // Smooth-scroll between the two screens (header link, hero cue, back-to-top).
  const scrollToStages = () => {
    document.getElementById("oc-stages")?.scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  };
  const scrollToTop = () => {
    document.getElementById("oc-top")?.scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  };

  const typed = HEADLINE.slice(0, typedN);
  const rest = HEADLINE.slice(typedN);
  const typedAccent = typed.slice(0, ACCENT_CHARS);
  const typedInk = typed.slice(ACCENT_CHARS);
  const done = phase >= total;

  const monoTag: React.CSSProperties = {
    fontFamily: MONO, fontSize: 10.5, letterSpacing: "0.04em", color: FAINT,
  };

  return (
    <div style={{ height: "100vh", overflowY: "auto", background: PAPER, color: INK, fontFamily: SANS }}>

      {/* ═══ First screen: brand → thesis → workflow → live case ══ */}
      <section id="oc-top" style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>

      {/* ═══ Header ═══════════════════════════════════════════════ */}
      <header className="oc-header" style={{ flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "11px 40px", borderBottom: `1px solid ${LINE}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: promo ? 13 : 10 }}>
          {/* Promo poster is consumed at ~0.42× scale inside WeChat articles —
              the product name has to survive that shrink. */}
          <Logo size={promo ? 48 : 34} />
          <span style={{ fontSize: promo ? 36 : 24, fontWeight: 800, letterSpacing: "-0.02em" }}>OfferCheck</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          {/* The competition's required inference base gets the header slot —
              a bottom footnote dies at WeChat scale. Two-tone: quiet "Powered by",
              emphatic engine name. */}
          <span className="oc-header-tag" style={{ fontFamily: MONO, display: "flex", alignItems: "center", gap: 8,
            whiteSpace: "nowrap", fontSize: promo ? 17 : 12, letterSpacing: "0.02em" }}>
            <span style={{ width: promo ? 8 : 6, height: promo ? 8 : 6, borderRadius: 99, background: OK_GREEN, display: "inline-block" }} />
            <span>
              <span style={{ color: FAINT }}>Powered by </span>
              <b style={{ color: "oklch(30% 0.04 40)" }}>GMI Cloud Inference Engine</b>
            </span>
          </span>
          {!promo && (
            <button className="oc-cue" onClick={scrollToStages} style={{
              background: "transparent", border: "none", padding: "4px 2px",
              fontSize: 12.5, fontWeight: 700, fontFamily: SANS, color: FOG,
              cursor: "pointer", whiteSpace: "nowrap" }}>
              How it works
            </button>
          )}
          {/* Same entry as the hero CTA — just a shortcut in the corner. */}
          {!promo && (
            <button className="oc-quiet" onClick={() => onEnter()} style={{
              background: "transparent", border: `1px solid ${LINE}`, borderRadius: 9,
              padding: "7px 15px", fontSize: 12.5, fontWeight: 700, fontFamily: SANS,
              color: FOG, cursor: "pointer", whiteSpace: "nowrap" }}>
              Start a check
            </button>
          )}
        </div>
      </header>

      {/* ═══ Hero ═════════════════════════════════════════════════ */}
      <main style={{ flex: 1, display: "flex", alignItems: "center", width: "100%" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 460px), 1fr))",
          gap: "36px 56px", alignItems: "center", maxWidth: 1280, margin: "0 auto",
          padding: "8px 56px", width: "100%" }}>

          {/* ─── Thesis ─────────────────────────────────────────── */}
          <div>
            <div style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, letterSpacing: "0.16em",
              textTransform: "uppercase", color: ACCENT }}>
              Job-offer due diligence · evidence-first
            </div>

            <h1 aria-label={HEADLINE.replace("\n", " ")} style={{ fontFamily: MONO, fontWeight: 800,
              fontSize: "clamp(38px, 4.4vw, 58px)", lineHeight: 1.12, letterSpacing: "-0.045em",
              whiteSpace: "pre-line", marginTop: 14 }}>
              <span aria-hidden="true">
                <span style={{ color: ACCENT }}>{typedAccent}</span>
                <span>{typedInk}</span>
                <span className="oc-anim" style={{ color: ACCENT, animation: "ocCaret 1.06s step-end infinite" }}>▍</span>
                <span style={{ visibility: "hidden" }}>{rest}</span>
              </span>
            </h1>

            {promoZh && (
              <div className="oc-anim" style={{ fontFamily: CJK,
                fontSize: 36, fontWeight: 700, letterSpacing: "0.2em", color: ACCENT, marginTop: 18,
                animation: "ocFadeUp 0.55s ease 0.1s both" }}>
                信不信，先核验
              </div>
            )}

            {promoZh ? (
              <p className="oc-anim" style={{ fontFamily: CJK, fontSize: 16.5, lineHeight: 1.8, color: FOG,
                maxWidth: 470, marginTop: 16, animation: "ocFadeUp 0.55s ease 0.15s both" }}>
                粘贴 Offer、JD、招聘沟通或公司名称。OfferCheck 围绕这个具体机会独立调查，
                给出有出处、可复核的判断。
              </p>
            ) : (
              <p className="oc-anim" style={{ fontSize: 15.5, lineHeight: 1.66, color: FOG, maxWidth: 470,
                marginTop: 16, animation: "ocFadeUp 0.55s ease 0.15s both" }}>
                Paste an offer, a JD, a recruiter message — or just a company name.
                A skeptical research agent investigates that specific opportunity and
                returns a verdict backed by sources it actually saw.
              </p>
            )}

            <div className="oc-anim" style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap",
              marginTop: 22, animation: "ocFadeUp 0.55s ease 0.3s both" }}>
              <button className="oc-cta" onClick={() => onEnter()} style={{
                background: ACCENT, color: "white", border: "none", borderRadius: 11,
                padding: "13px 26px", fontSize: 15, fontWeight: 800, fontFamily: SANS,
                cursor: "pointer", boxShadow: "0 6px 22px -6px oklch(58% 0.13 40 / 0.55)" }}>
                Start a check →
              </button>
              {promoZh
                ? <span style={{ fontFamily: CJK, fontSize: 13, color: FAINT }}>免费使用 · 无需注册 · 每个结论都有出处</span>
                : <span style={monoTag}>free · no sign-up · every claim sourced</span>}
            </div>

            {/* How it works — a proper four-step card. No metrics strip: the
                product has no operating data yet, and dressing principles up
                as numbers ("0", "2-pass") invites questions we can't answer. */}
            <div className="oc-anim" style={{ background: "white", border: `1px solid ${LINE}`,
              borderRadius: 14, padding: "14px 18px 16px", marginTop: 28, maxWidth: 560,
              animation: "ocFadeUp 0.55s ease 0.45s both" }}>
              <div style={{ fontFamily: promoZh ? CJK : MONO, fontSize: promo ? 12 : 10, fontWeight: 700,
                letterSpacing: promo ? "0.2em" : "0.16em", textTransform: "uppercase", color: FAINT }}>
                {promoZh ? "工作流程" : "How a check runs"}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(112px, 1fr))",
                gap: 12, marginTop: 11 }}>
                {(promoZh ? FLOW_CN : FLOW_EN).map((st, i, arr) => (
                  <div key={st.t}>
                    {/* number + connector — the stepper reads as flow without arrow glyphs */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontFamily: MONO, fontSize: promo ? 14 : 13, fontWeight: 800, color: ACCENT }}>
                        0{i + 1}
                      </span>
                      {i < arr.length - 1 && <span style={{ flex: 1, height: 1, background: LINE }} />}
                    </div>
                    <div style={{ fontFamily: promoZh ? CJK : SANS, fontSize: promo ? 15.5 : 14.5, fontWeight: 800,
                      color: INK, marginTop: 6, letterSpacing: "-0.01em", whiteSpace: "nowrap" }}>{st.t}</div>
                    <div style={{ fontFamily: promoZh ? CJK : SANS, fontSize: promo ? 11.5 : 11, color: FAINT,
                      marginTop: 3, lineHeight: 1.45 }}>{st.d}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ─── Live case replay (signature) ───────────────────── */}
          <div style={{ justifySelf: "center", width: "100%", maxWidth: promo ? 565 : 530 }}>
            <p className="oc-sr">
              Example investigation: the agent researches a suspicious offer from
              “Nexora Global Solutions”, is blocked by the evidence gate until sources
              are verified, then returns the verdict “Likely a Scam” with three red flags.
            </p>

            <div aria-hidden="true" style={{ background: "white", border: `1px solid ${LINE}`,
              borderRadius: 16, overflow: "hidden",
              boxShadow: "0 24px 60px -28px oklch(40% 0.06 40 / 0.5)" }}>

              {/* Title bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 16px",
                borderBottom: `1px solid ${LINE}`, background: "oklch(99.3% 0.004 70)" }}>
                <div style={{ display: "flex", gap: 5 }}>
                  {[0, 1, 2].map(i => (
                    <span key={i} style={{ width: 9, height: 9, borderRadius: 99, background: LINE, display: "inline-block" }} />
                  ))}
                </div>
                <span style={{ fontFamily: MONO, fontSize: 11.5, color: FOG }}>
                  case-004 · Offer Verification
                </span>
                <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6,
                  fontFamily: MONO, fontSize: 10.5, color: done ? OK_GREEN : ACCENT }}>
                  <span className="oc-anim" style={{ width: 6, height: 6, borderRadius: 99,
                    background: done ? OK_GREEN : ACCENT, display: "inline-block",
                    animation: done ? "none" : "pulse 1.1s ease-in-out infinite" }} />
                  {done ? "verdict ready" : "investigating"}
                </span>
              </div>

              {/* Body */}
              <div style={{ padding: "13px 14px", minHeight: 420, display: "flex", flexDirection: "column", gap: 7,
                opacity: fading ? 0 : 1, transition: "opacity 0.42s ease" }}>

                {/* Task bubble */}
                <div style={{ alignSelf: "flex-end", maxWidth: "92%", background: ACCENT, color: "white",
                  fontSize: promo ? 12.5 : 11.5, lineHeight: 1.5, padding: "8px 11px", borderRadius: "12px 3px 12px 12px" }}>
                  Offer — Nexora Global Solutions Ltd. · $9,200/mo in USDT ·
                  $320 “equipment kit” up-front · reply within 24h. Legit?
                </div>

                {rows.slice(0, phase).map((row, i) => {
                  const isLast = i === phase - 1 && !done;
                  if (row.kind === "step") {
                    return (
                      <div key={i} className="oc-anim" style={{ display: "flex", alignItems: "flex-start", gap: 8,
                        animation: "ocStepIn 0.32s ease-out both" }}>
                        <span style={{ flexShrink: 0, fontFamily: MONO, fontSize: promo ? 11 : 10, fontWeight: 700,
                          background: row.chipBg, color: row.chipFg, borderRadius: 6, padding: "3px 8px",
                          marginTop: 1 }}>{row.tool}</span>
                        <span style={{ fontSize: promo ? 13 : 12, lineHeight: 1.55, color: "oklch(32% 0.02 50)" }}>{row.text}</span>
                        <span className="oc-anim" style={{ flexShrink: 0, width: 7, height: 7, borderRadius: 99,
                          background: row.dot, marginLeft: "auto", marginTop: 5,
                          animation: isLast ? "pulse 1.1s ease-in-out infinite" : "none" }} />
                      </div>
                    );
                  }
                  if (row.kind === "gate") {
                    return (
                      <div key={i} className="oc-anim" style={{ display: "flex", gap: 8, alignItems: "flex-start",
                        background: "oklch(95% 0.035 80)", border: "1px solid oklch(85% 0.05 80)",
                        borderRadius: 9, padding: "8px 11px", animation: "ocStepIn 0.32s ease-out both" }}>
                        <span style={{ fontSize: 12, lineHeight: 1.4 }}>⚠</span>
                        <span style={{ fontFamily: MONO, fontSize: promo ? 12 : 11, lineHeight: 1.55, color: "oklch(34% 0.1 80)" }}>
                          {row.text}
                        </span>
                      </div>
                    );
                  }
                  if (row.kind === "verify") {
                    return (
                      <div key={i} className="oc-anim" style={{ display: "flex", gap: 8, alignItems: "center",
                        animation: "ocStepIn 0.32s ease-out both" }}>
                        <span style={{ color: OK_GREEN, fontSize: 12, fontWeight: 800 }}>✓</span>
                        <span style={{ fontFamily: MONO, fontSize: promo ? 12 : 11, color: "oklch(38% 0.09 145)" }}>{row.text}</span>
                      </div>
                    );
                  }
                  // Verdict card — the conclusion layer. Promo poster speaks Chinese
                  // here (stamp + flags + footer); the live page stays English.
                  return (
                    <div key={i} className="oc-anim" style={{ background: "oklch(96% 0.035 25)",
                      border: "1px solid oklch(85% 0.05 25)", borderRadius: 12, padding: promo ? "13px 15px" : "11px 13px",
                      marginTop: 2, animation: "ocStepIn 0.3s ease-out both" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                        <span className="oc-anim" style={{ display: "inline-block", border: `3px solid ${ALARM}`,
                          color: ALARM, fontFamily: promoZh ? CJK : MONO, fontSize: promo ? 21 : 14.5, fontWeight: 800,
                          letterSpacing: promo ? "0.12em" : "0.07em", textTransform: "uppercase",
                          padding: promo ? "5px 14px" : "4px 12px",
                          borderRadius: 6, transform: "rotate(-3deg)",
                          animation: "ocStampIn 0.42s cubic-bezier(0.2, 1.4, 0.4, 1) both" }}>
                          {promoZh ? "大概率有坑" : "Likely a Scam"}
                        </span>
                        <span style={{ fontFamily: MONO, fontSize: promo ? 11.5 : 10.5, color: "oklch(45% 0.08 25)" }}>
                          {promoZh ? "Likely a Scam · high confidence" : "high confidence · re-verified"}
                        </span>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: promo ? 6 : 4, marginTop: promo ? 11 : 9 }}>
                        {(promoZh ? RED_FLAGS_CN : RED_FLAGS).map(f => (
                          <div key={f} style={{ display: "flex", gap: 8, fontSize: promo ? 13.5 : 11.5,
                            lineHeight: 1.5, color: ALARM_INK,
                            fontFamily: promoZh ? CJK : SANS }}>
                            <span style={{ color: ALARM, flexShrink: 0 }}>⚑</span>
                            <span>{f}</span>
                          </div>
                        ))}
                      </div>
                      <div style={{ fontFamily: promoZh ? CJK : MONO, fontSize: promo ? 11.5 : 10.5, color: "oklch(45% 0.08 25)",
                        marginTop: promo ? 10 : 8, paddingTop: promo ? 9 : 7, borderTop: "1px dashed oklch(82% 0.06 25)" }}>
                        {promoZh ? "6 步调查 · 引用 4 个来源 · 已独立复核" : "6 steps · 4 sources cited · re-verified independently"}
                      </div>
                    </div>
                  );
                })}

                {/* Status line */}
                <div style={{ marginTop: "auto", paddingTop: 5, fontFamily: MONO, fontSize: 10.5, color: FAINT }}>
                  {done
                    ? "done · 6 steps · evidence-gated · verifier ✓"
                    : `DeepSeek-V4-Flash · step ${Math.min(phase + 1, total)}/12 · trial 1/2`}
                </div>
              </div>
            </div>

            <button className="oc-quiet" onClick={() => onEnter({ demo: true })} style={{
              display: "block", margin: "8px auto 0", background: "transparent", border: "none",
              fontFamily: MONO, fontSize: 11, fontWeight: 700, color: ACCENT, cursor: "pointer",
              padding: "3px 8px", borderRadius: 6 }}>
              ▸ Run this exact case yourself — pre-filled, live engine
            </button>
          </div>
        </div>
      </main>

      {/* Scroll cue — the hero's Offer case reads as "this only checks offers";
          the cue points at the screen that corrects that. */}
      {!promo && (
        <button className="oc-cue oc-anim" onClick={scrollToStages} style={{
          alignSelf: "center", margin: "0 0 14px", display: "inline-flex", alignItems: "center",
          gap: 9, background: "transparent", border: "none", cursor: "pointer",
          fontFamily: MONO, fontSize: 14.5, fontWeight: 700, color: FOG,
          padding: "6px 12px", borderRadius: 8,
          animation: "ocFadeUp 0.55s ease 0.8s both" }}>
          See all 4 stages
          <span className="oc-bob" aria-hidden="true" style={{ color: ACCENT, fontSize: 15 }}>↓</span>
        </button>
      )}

      {/* Promo poster stays a single screen: one roomy band of stage names —
          long descriptions and tech chips don't survive WeChat scaling. */}
      {promo && (
        <footer style={{ flexShrink: 0, borderTop: `1px solid ${LINE}` }}>
          <div style={{ maxWidth: 1280, margin: "0 auto", padding: "18px 56px 24px" }}>
            <div style={{ fontFamily: promoZh ? CJK : MONO, fontSize: 13, fontWeight: 600,
              letterSpacing: "0.18em", color: FAINT, textTransform: promoZh ? "none" : "uppercase" }}>
              {promoZh ? "覆盖求职全流程" : "Covers the whole job search"}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 44px", marginTop: 12 }}>
              {(promoZh ? STAGES_CN : STAGES.map(s => s.name)).map((n, i) => (
                <div key={n} style={{ display: "flex", alignItems: "baseline", gap: 9 }}>
                  <span style={{ fontFamily: MONO, fontSize: 13, fontWeight: 800, color: ACCENT }}>0{i + 1}</span>
                  <span style={{ fontFamily: promoZh ? CJK : SANS, fontSize: 18, fontWeight: 700, color: INK }}>{n}</span>
                </div>
              ))}
            </div>
          </div>
        </footer>
      )}
      </section>

      {/* ═══ Second screen (live only): the four stages, given room.
          One job: correct the "it only checks offers" impression. ═══ */}
      {!promo && (
        <section id="oc-stages" style={{ borderTop: `1px solid ${LINE}`, background: "oklch(99.3% 0.004 70)" }}>
          <div style={{ maxWidth: 1280, margin: "0 auto", padding: "56px 56px 60px" }}>
            <div style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, letterSpacing: "0.16em",
              textTransform: "uppercase", color: ACCENT }}>
              Four stages · one agent
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em", marginTop: 8 }}>
              Covering the whole job-search journey
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
              gap: 18, marginTop: 26 }}>
              {STAGES.map(s => (
                <div key={s.num} className="oc-card" style={{ background: "white", border: `1px solid ${LINE}`,
                  borderRadius: 14, padding: "18px 18px 20px" }}>
                  <span style={{ fontFamily: MONO, fontSize: 12, fontWeight: 800, color: ACCENT }}>{s.num}</span>
                  <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: "-0.01em", marginTop: 8 }}>{s.name}</div>
                  <div style={{ fontSize: 13, lineHeight: 1.55, color: FOG, marginTop: 6 }}>{s.desc}</div>
                </div>
              ))}
            </div>
            {/* The reader who scrolled this far is convinced — hand them the door. */}
            <div style={{ display: "flex", justifyContent: "center", marginTop: 44 }}>
              <button className="oc-cta" onClick={() => onEnter()} style={{
                background: ACCENT, color: "white", border: "none", borderRadius: 11,
                padding: "13px 26px", fontSize: 15, fontWeight: 800, fontFamily: SANS,
                cursor: "pointer", boxShadow: "0 6px 22px -6px oklch(58% 0.13 40 / 0.55)" }}>
                Start a check →
              </button>
            </div>
            {/* Way back up — mirror of the hero's scroll cue */}
            <button className="oc-cue" onClick={scrollToTop} style={{
              display: "flex", alignItems: "center", gap: 8, margin: "26px auto 0",
              background: "transparent", border: "none", cursor: "pointer",
              fontFamily: MONO, fontSize: 13, fontWeight: 700, color: FOG,
              padding: "5px 10px", borderRadius: 8 }}>
              Back to top
              <span aria-hidden="true" style={{ color: ACCENT, fontSize: 13.5 }}>↑</span>
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
