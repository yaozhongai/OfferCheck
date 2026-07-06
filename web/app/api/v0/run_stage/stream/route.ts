import { NextRequest } from "next/server";

const NEXA_API = process.env.NEXA_API_BASE || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  const body = await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${NEXA_API}/api/v0/run_stage/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      // Node 18+ fetch supports streaming response bodies
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return new Response(text, { status: upstream.status });
  }

  // Pipe the upstream ReadableStream directly — no buffering
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
      "Transfer-Encoding": "chunked",
    },
  });
}

// Opt into Node.js runtime (not Edge) so fetch() supports streaming bodies
export const runtime = "nodejs";
// Never cache this route
export const dynamic = "force-dynamic";
