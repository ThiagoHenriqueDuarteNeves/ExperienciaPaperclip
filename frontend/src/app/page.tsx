"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ProfileGate from "../components/ProfileGate";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { clearSession, getSession, type Session } from "../lib/auth";

const BUILD_TAG = "v14";
const CHAT_URL = process.env.NEXT_PUBLIC_CHAT_API_URL ?? "/api/chat";

function genId(): string {
  try {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function")
      return crypto.randomUUID();
  } catch { /* unavailable */ }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function parseSSELine(line: string): { text?: string; error?: string } {
  if (!line.startsWith("data: ")) return {};
  try {
    const event = JSON.parse(line.slice(6));
    if (event.event === "text") return { text: event.content as string };
    if (event.event === "error") return { error: event.message as string };
  } catch { /* incomplete frame */ }
  return {};
}

function parseSSEBuffer(raw: string): string {
  let text = "";
  for (const line of raw.split("\n")) {
    const p = parseSSELine(line);
    if (p.text) text += p.text;
  }
  return text;
}

function convKey(userId: string) { return `memory-chat-conversation-id:${userId}`; }

function getConversationId(userId: string): string {
  if (typeof window === "undefined") return "default-conversation";
  const key = convKey(userId);
  try {
    const existing = window.localStorage.getItem(key);
    if (existing) return existing;
    const id = genId();
    window.localStorage.setItem(key, id);
    return id;
  } catch { return genId(); }
}

function clearConversationId(userId: string): void {
  try { window.localStorage.removeItem(convKey(userId)); } catch { /* ignore */ }
}

interface ChatMessage { role: "user" | "assistant"; content: string; }

// ── Glass constants ──────────────────────────────────────────────────────────

const GLASS_PANEL: React.CSSProperties = {
  background: "rgba(8, 6, 28, 0.62)",
  backdropFilter: "blur(24px)",
  WebkitBackdropFilter: "blur(24px)",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  boxShadow: "0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.07)",
};

const GLASS_MSG_ASSISTANT: React.CSSProperties = {
  background: "rgba(255, 255, 255, 0.05)",
  backdropFilter: "blur(14px)",
  WebkitBackdropFilter: "blur(14px)",
  border: "1px solid rgba(255, 255, 255, 0.09)",
  boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
};

// ── Avatars ──────────────────────────────────────────────────────────────────

function AssistantAvatar() {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: 8, flexShrink: 0, fontSize: 14,
      background: "rgba(129, 140, 248, 0.14)",
      border: "1px solid rgba(129, 140, 248, 0.3)",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>🧠</div>
  );
}

function UserAvatar({ name }: { name: string }) {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
      fontSize: 12, fontWeight: 700, color: "#c0c8ff", userSelect: "none",
      background: "rgba(99, 102, 241, 0.28)",
      backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
      border: "1px solid rgba(129, 140, 248, 0.42)",
      display: "flex", alignItems: "center", justifyContent: "center",
    } as React.CSSProperties}>
      {name.charAt(0).toUpperCase()}
    </div>
  );
}

// ── Typing dots ──────────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <span style={{ display: "inline-flex", gap: 5, alignItems: "center", height: 18 }}>
      {[0, 1, 2].map((i) => (
        <span key={i} style={{
          width: 5, height: 5, borderRadius: "50%", display: "inline-block",
          background: "rgba(160, 160, 220, 0.6)",
          animation: "pulseDot 1.2s ease-in-out infinite",
          animationDelay: `${i * 0.18}s`,
        }} />
      ))}
    </span>
  );
}

// ── Message bubbles ──────────────────────────────────────────────────────────

function MessageBubble({
  role, content, userName = "U", isFirst = true, isLast = true, plain = false,
}: {
  role: "user" | "assistant"; content: string;
  userName?: string; isFirst?: boolean; isLast?: boolean;
  /** While streaming, render as plain text to avoid re-parsing markdown on
      every token (O(n^2) over the stream). Full parse runs once on completion. */
  plain?: boolean;
}) {
  const gap = isLast ? 16 : 3;

  if (role === "user") {
    const r = !isFirst && !isLast ? "18px 4px 4px 18px"
            : !isFirst            ? "18px 4px 18px 18px"
            :                       "18px 18px 4px 18px";
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "flex-end", gap: 8, marginBottom: gap, animation: "fadeSlide 0.18s ease-out" }}>
        <div style={{
          maxWidth: "72%", padding: "10px 15px",
          background: "rgba(99, 102, 241, 0.24)",
          backdropFilter: "blur(14px)", WebkitBackdropFilter: "blur(14px)",
          border: "1px solid rgba(129, 140, 248, 0.38)",
          borderRadius: r, color: "#dde0ff",
          fontSize: 14.5, lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word",
          boxShadow: "0 4px 16px rgba(99, 102, 241, 0.15)",
        } as React.CSSProperties}>
          {content}
        </div>
        <div style={{ visibility: isLast ? "visible" : "hidden", flexShrink: 0 } as React.CSSProperties}>
          <UserAvatar name={userName} />
        </div>
      </div>
    );
  }

  const cr = !isFirst && !isLast ? "4px 14px 4px 14px"
           : !isFirst             ? "4px 14px 14px 14px"
           :                        "4px 14px 14px 14px";

  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: gap, animation: "fadeSlide 0.18s ease-out" }}>
      <div style={{ visibility: isFirst ? "visible" : "hidden", flexShrink: 0, width: 28 } as React.CSSProperties}>
        <AssistantAvatar />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        {isFirst && (
          <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(130, 130, 200, 0.55)", marginBottom: 5, letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Memory Chat
          </div>
        )}
        <div style={{ ...GLASS_MSG_ASSISTANT, borderRadius: cr, padding: "12px 16px" }}>
          {plain ? (
            <div style={{ fontSize: 14.5, lineHeight: 1.65, color: "var(--text)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {content}
            </div>
          ) : (
            <MarkdownRenderer content={content} />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Typing indicator ─────────────────────────────────────────────────────────

function TypingCard() {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 16, animation: "fadeSlide 0.18s ease-out" }}>
      <AssistantAvatar />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(130, 130, 200, 0.55)", marginBottom: 5, letterSpacing: "0.06em", textTransform: "uppercase" }}>
          Memory Chat
        </div>
        <div style={{ ...GLASS_MSG_ASSISTANT, borderRadius: "4px 14px 14px 14px", padding: "12px 16px", display: "inline-block" } as React.CSSProperties}>
          <TypingDots />
        </div>
      </div>
    </div>
  );
}

// ── Empty state ──────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "O que você lembra de mim?",
  "Resuma nossas últimas conversas",
  "Me ajude a organizar uma ideia",
];

function EmptyState({ name, onPrompt }: { name: string; onPrompt: (t: string) => void }) {
  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: 20, padding: "40px 16px", textAlign: "center",
      animation: "fadeIn 0.5s ease-out",
    }}>
      <div style={{
        width: 70, height: 70, borderRadius: 20, fontSize: 30,
        background: "rgba(129, 140, 248, 0.14)",
        backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)",
        border: "1px solid rgba(129, 140, 248, 0.28)",
        boxShadow: "0 8px 32px rgba(99, 102, 241, 0.2)",
        display: "flex", alignItems: "center", justifyContent: "center",
      } as React.CSSProperties}>🧠</div>

      <div>
        <p style={{ margin: "0 0 8px", fontSize: 20, fontWeight: 700, color: "#dde0ff", letterSpacing: "-0.03em" }}>
          Olá, {name}!
        </p>
        <p style={{ margin: 0, fontSize: 14, maxWidth: 300, lineHeight: 1.65, color: "rgba(140, 140, 200, 0.75)" }}>
          Lembro de tudo que conversamos. Continue de onde paramos.
        </p>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", maxWidth: 420 }}>
        {SUGGESTIONS.map((s) => (
          <button key={s} type="button" className="glass-chip" onClick={() => onPrompt(s)} style={{
            padding: "7px 16px", borderRadius: 20, cursor: "pointer",
            background: "rgba(255, 255, 255, 0.06)",
            backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
            border: "1px solid rgba(255, 255, 255, 0.11)",
            color: "rgba(180, 180, 240, 0.85)", fontSize: 13,
            transition: "all 0.15s", boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
            touchAction: "manipulation", WebkitTapHighlightColor: "transparent",
          } as React.CSSProperties}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Header glass button ───────────────────────────────────────────────────────

function GlassBtn({ onClick, title, children }: { onClick: () => void; title?: string; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} title={title} className="glass-btn" style={{
      display: "flex", alignItems: "center", gap: 5,
      padding: "5px 11px", borderRadius: 8,
      border: "1px solid rgba(255, 255, 255, 0.1)",
      background: "rgba(255, 255, 255, 0.06)",
      color: "rgba(180, 180, 240, 0.9)", fontSize: 12,
      cursor: "pointer", touchAction: "manipulation",
      WebkitTapHighlightColor: "transparent", userSelect: "none",
      transition: "all 0.15s", letterSpacing: "0.01em",
    } as React.CSSProperties}>
      {children}
    </button>
  );
}

// ── Send / Stop ───────────────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M8 13V3M4 7l4-4 4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden>
      <rect x="2" y="2" width="9" height="9" rx="2" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function ActionButton({ kind, onClick, enabled }: { kind: "send" | "stop"; onClick: () => void; enabled?: boolean }) {
  const isSend = kind === "send";
  return (
    <button type="button" onClick={onClick}
      aria-label={isSend ? "Enviar" : "Parar"}
      style={{
        flexShrink: 0, width: 38, height: 38, borderRadius: 10,
        display: "flex", alignItems: "center", justifyContent: "center",
        cursor: isSend ? (enabled ? "pointer" : "default") : "pointer",
        touchAction: "manipulation", WebkitTapHighlightColor: "transparent",
        userSelect: "none", transition: "all 0.15s",
        backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
        ...(isSend
          ? {
              border: enabled ? "1px solid rgba(129, 140, 248, 0.45)" : "1px solid rgba(255,255,255,0.08)",
              background: enabled ? "rgba(99, 102, 241, 0.3)" : "rgba(255,255,255,0.04)",
              color: enabled ? "#a5b4fc" : "rgba(140,140,200,0.35)",
            }
          : {
              border: "1px solid rgba(248, 113, 113, 0.42)",
              background: "rgba(248, 113, 113, 0.13)",
              color: "#f87171",
            }),
      } as React.CSSProperties}
    >
      {isSend ? <SendIcon /> : <StopIcon />}
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hasText, setHasText] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const streamingRef = useRef(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const sessionRef = useRef<Session | null>(null);

  useEffect(() => { setSession(getSession()); setAuthChecked(true); }, []);
  useEffect(() => { sessionRef.current = session; }, [session]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streamingText]);

  function readText(): string { return (textareaRef.current?.value ?? "").trim(); }

  function clearTextarea() {
    if (textareaRef.current) textareaRef.current.value = "";
    setHasText(false);
  }

  // Fixed-height input: no auto-resize. Text scrolls inside the textarea so the
  // floating bar never grows/moves as you type.
  function handleInput() { setHasText(readText().length > 0); }

  function handlePromptClick(text: string) {
    if (textareaRef.current) {
      textareaRef.current.value = text;
      setHasText(true); textareaRef.current.focus();
    }
  }

  const doSubmit = useCallback(async () => {
    const text = readText();
    if (!text || streamingRef.current) return;
    const sess = sessionRef.current;
    if (!sess) return;

    const userMessage: ChatMessage = { role: "user", content: text };
    const newMessages = [...messagesRef.current, userMessage];
    messagesRef.current = newMessages;
    setMessages(newMessages);
    clearTextarea();
    streamingRef.current = true; setStreaming(true); setStreamingText("");

    const controller = new AbortController();
    abortRef.current = controller;
    let finalReply = "", errorMsg = "";

    try {
      const conversationId = getConversationId(sess.userId);
      const res = await fetch(CHAT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${sess.token}`,
          "x-conversation-id": conversationId,
          skip_zrok_interstitial: "true",
        },
        body: JSON.stringify({ messages: newMessages }),
        signal: controller.signal,
      });

      if (res.status === 401) { clearSession(); setSession(null); return; }
      if (!res.ok) {
        errorMsg = `HTTP ${res.status} ${res.statusText}`;
      } else if (!res.body || typeof res.body.getReader !== "function") {
        const raw = await res.text();
        finalReply = parseSSEBuffer(raw);
        setStreamingText(finalReply);
        if (!finalReply) errorMsg = "resposta vazia (fallback sem stream)";
      } else {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            const p = parseSSELine(line);
            if (p.text) { finalReply += p.text; setStreamingText(finalReply); }
            else if (p.error) { errorMsg = p.error; }
          }
        }
        const tail = parseSSELine(buffer);
        if (tail.text) { finalReply += tail.text; setStreamingText(finalReply); }
        if (!finalReply && !errorMsg) errorMsg = "resposta vazia (stream sem dados)";
      }
    } catch (err) {
      if (!(err instanceof Error && err.name === "AbortError"))
        errorMsg = `falha de conexão — ${err instanceof Error ? err.message : "desconhecido"}`;
    } finally {
      streamingRef.current = false; setStreaming(false); setStreamingText(""); abortRef.current = null;
    }

    const content = finalReply.trim() || (errorMsg ? `⚠️ ${errorMsg}` : "");
    if (content) {
      const reply: ChatMessage = { role: "assistant", content };
      messagesRef.current = [...messagesRef.current, reply];
      setMessages([...messagesRef.current]);
    }
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSubmit(); }
  }
  function handleStop() { abortRef.current?.abort(); }
  function handleNewConversation() {
    if (streamingRef.current) abortRef.current?.abort();
    if (session) clearConversationId(session.userId);
    messagesRef.current = []; setMessages([]); setStreamingText(""); clearTextarea();
  }
  function handleLogout() {
    if (streamingRef.current) abortRef.current?.abort();
    clearSession(); messagesRef.current = []; setMessages([]); setStreamingText(""); clearTextarea(); setSession(null);
  }

  if (!authChecked) return null;
  if (!session) return <ProfileGate onAuthenticated={(s) => setSession(s)} />;

  const isEmpty = messages.length === 0 && !streamingText;

  return (
    <>
      {/* Animated background — fixed, behind everything */}
      <div className="bg-root">
        <div className="bg-blob bg-blob-1" />
        <div className="bg-blob bg-blob-2" />
        <div className="bg-blob bg-blob-3" />
      </div>

      {/* ── Scrollable messages — the ONLY scroll container.
          position:fixed anchors it to the viewport directly, so it never depends
          on a parent's resolved height. The inner wrapper centers content (maxWidth)
          and reserves space for the floating header/input via padding. ── */}
      <div
        className="msgs"
        style={{ position: "fixed", inset: 0, zIndex: 1, overflowY: "auto", overflowX: "hidden" }}
      >
        <div
          style={{
            maxWidth: 840,
            margin: "0 auto",
            minHeight: "100%",
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "column",
            paddingTop: 82,     /* clear fixed header */
            paddingBottom: "calc(96px + env(safe-area-inset-bottom, 0px))", /* clear fixed input */
            paddingLeft: 10,
            paddingRight: 10,
          }}
        >
          {isEmpty && <EmptyState name={session.displayName} onPrompt={handlePromptClick} />}

          {messages.map((msg, i) => {
            const prevSame = i > 0 && messages[i - 1].role === msg.role;
            const nextSame = i < messages.length - 1 && messages[i + 1].role === msg.role;
            return (
              <MessageBubble
                key={i} role={msg.role} content={msg.content}
                userName={session.displayName}
                isFirst={!prevSame} isLast={!nextSame}
              />
            );
          })}

          {streaming && !streamingText && <TypingCard />}
          {streamingText && (
            <MessageBubble role="assistant" content={streamingText} userName={session.displayName} plain />
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Fixed header — viewport-anchored, centered. The outer wrapper is
          full-width with pointer-events:none so it never blocks message scroll;
          only the glass bar itself is interactive. ── */}
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
        display: "flex", justifyContent: "center",
        padding: "10px 10px 0", pointerEvents: "none",
      } as React.CSSProperties}>
        <header style={{
          width: "100%", maxWidth: 820, pointerEvents: "auto",
          padding: "10px 14px", borderRadius: 16,
          display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
          ...GLASS_PANEL,
        } as React.CSSProperties}>
          {/* minWidth:0 lets this flex child shrink so the name ellipsizes
              instead of wrapping and growing the floating header's height. */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 9, fontSize: 16, flexShrink: 0,
              background: "rgba(129, 140, 248, 0.14)",
              border: "1px solid rgba(129, 140, 248, 0.3)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>🧠</div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 15, lineHeight: 1.2, letterSpacing: "-0.025em", color: "#dde0ff", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                Memory Chat{" "}
                <span style={{ fontSize: 10, fontWeight: 400, color: "rgba(129,140,248,0.65)", letterSpacing: "0.02em" }}>
                  {BUILD_TAG}
                </span>
              </div>
              <div style={{ fontSize: 11, color: "rgba(130, 130, 200, 0.6)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {session.displayName}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
            <GlassBtn onClick={handleNewConversation}>
              <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden>
                <path d="M5.5 1v9M1 5.5h9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
              <span className="hdr-label">Nova conversa</span>
            </GlassBtn>
            <GlassBtn onClick={handleLogout} title="Trocar de perfil">
              <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden>
                <path d="M7 1.5l3 3-3 3M1 4.5h9M4 9.5L1 6.5l3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Trocar
            </GlassBtn>
          </div>
        </header>
      </div>

      {/* ── Fixed input bar — viewport-anchored, centered, never moves ── */}
      <div style={{
        position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 50,
        display: "flex", justifyContent: "center",
        padding: "0 10px",
        paddingBottom: "calc(10px + env(safe-area-inset-bottom, 0px))",
        pointerEvents: "none",
      } as React.CSSProperties}>
        <div style={{
          width: "100%", maxWidth: 820, pointerEvents: "auto",
          borderRadius: 16,
          padding: "10px 10px 10px 16px",
          ...GLASS_PANEL,
        } as React.CSSProperties}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
            <textarea
              ref={textareaRef}
              className="glass-input"
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Digite uma mensagem…"
              rows={1}
              style={{
                flex: 1, border: "none", outline: "none",
                resize: "none", background: "transparent",
                color: "#dde0ff", lineHeight: 1.5, fontSize: 16,
                padding: "4px 0",
                /* border-box: height must cover the 24px line (16*1.5) + 8px
                   padding, else a single line gets clipped/scrolls. */
                height: 32,        /* fixed: one line — bar never grows */
                overflowY: "auto", /* extra text scrolls inside */
              }}
            />
            <ActionButton
              kind={streaming ? "stop" : "send"}
              onClick={streaming ? handleStop : doSubmit}
              enabled={hasText}
            />
          </div>
          <p style={{
            margin: "6px 0 0", fontSize: 11, textAlign: "center",
            color: "rgba(120, 120, 180, 0.38)", letterSpacing: "0.01em",
          }}>
            Enter para enviar · Shift+Enter para nova linha
          </p>
        </div>
      </div>
    </>
  );
}
