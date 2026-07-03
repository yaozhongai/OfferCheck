"use client";

import { useRef, useState } from "react";

type EngineEvent = {
  type: string;
  step?: number;
  max_steps?: number;
  model?: string;
  tool?: string;
  args?: string;
  thought?: string;
  observation?: string;
  ok?: boolean;
  message?: string;
  trial?: number;
  max_trials?: number;
  passed?: boolean;
  reason?: string;
  success?: boolean;
  answer?: string;
  trials_used?: number;
  latency_ms?: number;
};

const STAGES = [
  { value: "", label: "通用引擎 (no stage)" },
  { value: "stage1", label: "① 选岗调研 (stage1)" },
  { value: "stage4", label: "④ Offer 证伪 (stage4)" },
];

function eventLabel(e: EngineEvent): { cls: string; type: string; detail: string } {
  switch (e.type) {
    case "started":
      return { cls: "", type: "started", detail: "调查启动…" };
    case "trial_start":
      return { cls: "", type: `Trial ${e.trial}/${e.max_trials}`, detail: "开始一轮调查" };
    case "step_start":
      return { cls: "", type: `Step ${e.step}`, detail: `模型 ${e.model ?? ""}` };
    case "action":
      return {
        cls: "tool",
        type: `🔧 ${e.tool}`,
        detail: (e.thought ? `想法: ${e.thought}\n` : "") + `参数: ${e.args ?? ""}`,
      };
    case "observation":
      return {
        cls: "tool",
        type: `👁 ${e.tool} ${e.ok ? "✓" : "✗"}`,
        detail: e.observation ?? "",
      };
    case "correction":
      return { cls: "error", type: "⚡ 中途纠偏", detail: e.message ?? "" };
    case "verifier_start":
      return { cls: "verifier", type: "🔍 Verifier", detail: "事实核查中…" };
    case "verifier_result":
      return {
        cls: "verifier",
        type: `🔍 Verifier ${e.passed ? "通过" : "驳回"}`,
        detail: e.reason ?? "",
      };
    case "trial_evaluated":
      return {
        cls: "",
        type: `Trial ${e.trial} 评估`,
        detail: `${e.success ? "成功" : "失败"} — ${e.reason ?? ""}`,
      };
    case "final_answer":
      return { cls: "verifier", type: "✅ 最终答案", detail: "" };
    case "error":
      return { cls: "error", type: "❌ 错误", detail: e.message ?? "" };
    default:
      return { cls: "", type: e.type, detail: "" };
  }
}

export default function Home() {
  const [input, setInput] = useState("");
  const [stage, setStage] = useState("");
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<EngineEvent[]>([]);
  const [done, setDone] = useState<EngineEvent | null>(null);
  const timelineEnd = useRef<HTMLDivElement>(null);

  async function run() {
    if (!input.trim() || running) return;
    setRunning(true);
    setEvents([]);
    setDone(null);

    try {
      const resp = await fetch("/api/v0/run_stage/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          stage: stage || null,
          max_trials: 2,
          max_steps: 12,
        }),
      });

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });

        const chunks = buf.split("\n\n");
        buf = chunks.pop() || "";
        for (const chunk of chunks) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const evt: EngineEvent = JSON.parse(line.slice(5).trim());
          if (evt.type === "done") {
            setDone(evt);
          } else {
            setEvents((prev) => [...prev, evt]);
          }
          timelineEnd.current?.scrollIntoView({ behavior: "smooth" });
        }
      }
    } catch (err) {
      setEvents((prev) => [
        ...prev,
        { type: "error", message: String(err) },
      ]);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="container">
      <div className="header">
        <h1>OfferCheck</h1>
        <p>
          Skeptical research agent for job offers — paste an offer / JD / company
          name, watch it investigate live, get a verdict.
        </p>
      </div>

      <div className="card">
        <textarea
          placeholder="粘贴 offer / JD / 公司名，例如：某远程岗位要求先交 $500 设备押金，公司叫 Acme Remote Inc，邮箱 hr@acme-remote-jobs.com …"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={running}
        />
        <div className="row">
          <select
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            disabled={running}
          >
            {STAGES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <button className="primary" onClick={run} disabled={running || !input.trim()}>
            {running ? "调查中…" : "开始调查"}
          </button>
          {running && <span className="muted">引擎在实时调查，事件会逐条出现 ↓</span>}
        </div>
      </div>

      {events.length > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>🔬 调查轨迹 (实时 Trace)</h3>
          <ul className="timeline">
            {events.map((e, i) => {
              const { cls, type, detail } = eventLabel(e);
              return (
                <li className={`event ${cls}`} key={i}>
                  <div className="etype">{type}</div>
                  {detail && <div className="edetail">{detail}</div>}
                </li>
              );
            })}
          </ul>
          <div ref={timelineEnd} />
        </div>
      )}

      {done && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>
            📋 裁定
            <span className={`badge ${done.success ? "ok" : "fail"}`}>
              {done.success ? "成功" : "未达成"}
            </span>
            <span className="muted" style={{ marginLeft: 12 }}>
              {done.trials_used} trials · {done.latency_ms}ms
            </span>
          </h3>
          <div className="verdict">{done.answer}</div>
        </div>
      )}
    </div>
  );
}
