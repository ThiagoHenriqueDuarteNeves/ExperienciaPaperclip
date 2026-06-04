import { NextRequest } from "next/server";

export const runtime = "edge";

const MEMORY_API_BASE = process.env.MEMORY_API_URL || "http://localhost:8001";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const conversationId = req.headers.get("x-conversation-id") || "default";
  const authorization = req.headers.get("authorization") || "";

  let upstream: Response;
  try {
    upstream = await fetch(`${MEMORY_API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authorization,
        "x-conversation-id": conversationId,
        skip_zrok_interstitial: "true",
      },
      body,
      signal: req.signal,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "upstream unreachable";
    return new Response(
      `data: ${JSON.stringify({ event: "error", message: msg })}\n\n`,
      { status: 502, headers: { "Content-Type": "text/event-stream" } },
    );
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(
      `data: ${JSON.stringify({ event: "error", message: `upstream ${upstream.status}: ${text.slice(0, 200)}` })}\n\n`,
      { status: 502, headers: { "Content-Type": "text/event-stream" } },
    );
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
