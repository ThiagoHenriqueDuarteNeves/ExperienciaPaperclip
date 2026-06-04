import { NextRequest } from "next/server";

export const runtime = "edge";

// Proxy auth calls through this same-origin edge route to avoid browser CORS /
// the zrok interstitial (the preflight OPTIONS can't carry skip_zrok_interstitial,
// so a direct browser→zrok call hangs). Server-side fetch has no such problem.
function backendBase(): string {
  const chat = process.env.NEXT_PUBLIC_CHAT_API_URL;
  const fromChat = chat ? chat.replace(/\/api\/chat\/?$/, "") : "";
  return fromChat || process.env.MEMORY_API_URL || "http://localhost:8001";
}

const ALLOWED = new Set(["register", "login", "me"]);

async function forward(req: NextRequest, action: string, method: "GET" | "POST") {
  if (!ALLOWED.has(action)) {
    return new Response(JSON.stringify({ detail: "not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  const init: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
      skip_zrok_interstitial: "true",
      Authorization: req.headers.get("authorization") || "",
    },
  };
  if (method === "POST") init.body = await req.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${backendBase()}/auth/${action}`, init);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "upstream unreachable";
    return new Response(JSON.stringify({ detail: msg }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ action: string }> }) {
  const { action } = await ctx.params;
  return forward(req, action, "POST");
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ action: string }> }) {
  const { action } = await ctx.params;
  return forward(req, action, "GET");
}
