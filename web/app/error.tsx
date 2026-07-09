"use client";

import { useEffect } from "react";

/**
 * Route-level error boundary (Next.js App Router).
 *
 * Contains client-side RENDER crashes so a single bad render (e.g. an unexpected
 * shape in a verdict/sources payload during the live SSE stream) degrades to this
 * fallback instead of a blank/dev-overlay crash. The investigation result is
 * already persisted to localStorage — reloading rehydrates it (a "running" run
 * with an answer is coerced to "done" on hydrate), so "Reload" recovers the
 * verdict. (Network/read errors during streaming are handled separately in
 * page.tsx's runSSE catch, which keeps the answer instead of erroring out.)
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface for debugging (P2: pinpoint the exact throwing render).
    console.error("OfferCheck render error:", error);
  }, [error]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg)",
        fontFamily: "var(--font-sans)",
        padding: 24,
      }}
    >
      <div
        style={{
          maxWidth: 440,
          width: "100%",
          background: "white",
          border: "1px solid var(--border)",
          borderRadius: 14,
          padding: "28px 26px",
          textAlign: "center",
          boxShadow: "0 6px 24px oklch(50% 0.02 50 / 0.08)",
        }}
      >
        <div style={{ fontSize: 30, marginBottom: 8 }}>⚠️</div>
        <div style={{ fontSize: 17, fontWeight: 800, color: "var(--text)", marginBottom: 8 }}>
          Something went wrong displaying this view
        </div>
        <div style={{ fontSize: 13.5, color: "var(--muted)", lineHeight: 1.55, marginBottom: 20 }}>
          Your investigation result is saved. Reload to view it — nothing was lost.
        </div>
        <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: "9px 18px",
              borderRadius: 9,
              border: "none",
              background: "var(--accent)",
              color: "white",
              fontSize: 13.5,
              fontWeight: 700,
              fontFamily: "var(--font-sans)",
              cursor: "pointer",
            }}
          >
            Reload &amp; recover result
          </button>
          <button
            onClick={reset}
            style={{
              padding: "9px 18px",
              borderRadius: 9,
              border: "1px solid var(--border)",
              background: "white",
              color: "var(--text)",
              fontSize: 13.5,
              fontWeight: 600,
              fontFamily: "var(--font-sans)",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}
