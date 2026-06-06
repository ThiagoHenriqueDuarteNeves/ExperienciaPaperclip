"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ProfileGate from "../components/ProfileGate";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { clearSession, getSession, type Session } from "../lib/auth";

const BUILD_TAG = "v14";
const CHAT_URL = process.env.NEXT_PUBLIC_CHAT_API_URL ?? "/api/chat";

function genId(): string {
  try {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
  } catch { /* unavailable */ }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
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

function convKey(userId: string): string {
  return `memory-chat-conversation-id:${userId}`;
}

function getConversationId(userId: string): string {
  if (typeof window === "undefined") return "default-conversation";
  const key = convKey(userId);
  try {
    const existing = window.localStorage.getItem(key);
    if (existing) return existing;
    const id = genId();
    window.localStorage.setItem(key, id);
    return id;
  } catch {
    return genId();
  }
}

function clearConversationId(userId: string): void {
  try { window.localStorage.removeItem(convKey(userId)); } catch { /* ignore */ }
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// ─── Avatars ────────────────────────────────────────────────────────────────

function AssistantAvatar() {
  return (
    <div
      style={{
        width: 30,
        height: 30,
        borderRadius: 8,
        background: "var(--accent-dim)",
        border: "1px solid var(--accent-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        fontSize: 15,
      }}
    >
      🧠
    </div>
  );
}

function UserAvatar({ name }: { name: string }) {
  return (
    <div
      style={{
        width: 30,
        height: 30,
        borderRadius: "50%",
        background: "var(--accent)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        fontSize: 13,
        fontWeight: 700,
        color: "var(--bg)",
        userSelect: "none",
      }}
    >
      {name.charAt(0).toUpperCase()}
    </div>
  );
}

// ─── Typing indicator ────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <span style={{ display: "inline-flex", gap: 5, alignItems: "center", height: 20 }}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          style={{
            width: 5,
            height: 5,
            borderRadius: "50%",
            background: "var(--text-muted)",
            display: "inline-block",
            animation: "pulseDot 1.2s ease-in-out infinite",
            animationDelay: `${i * 0.18}s`,
          }}
        />
      ))}
    </span>
  );
}

// ─── Message bubbles ─────────────────────────────────────────────────────────

function MessageBubble({
  role,
  content,
  userName = "U",
}: {
  role: "user" | "assistant";
  content: string;
  userName?: string;
}) {
  if (role === "user") {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          alignItems: "flex-end",
          gap: 10,
          marginBottom: 18,
          animation: "fadeSlide 0.18s ease-out",
        }}
      >
        <div
          style={{
            maxWidth: "70%",
            padding: "10px 15px",
            background: "var(--user-bg)",
            color: "var(--user-text)",
            borderRadius: "18px 18px 4px 18px",
            fontSize: 14.5,
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
          }}
        >
          {content}
        </div>
        <UserAvatar name={userName} />
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        marginBottom: 22,
        animation: "fadeSlide 0.18s ease-out",
      }}
    >
      <AssistantAvatar />
      <div style={{ flex: 1, minWidth: 0, paddingTop: 4 }}>
        <MarkdownRenderer content={content} />
      </div>
    </div>
  );
}

// ─── Empty state ─────────────────────────────────────────────────────────────

function EmptyState({ name }: { name: string }) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 14,
        padding: "40px 24px",
        textAlign: "center",
        animation: "fadeIn 0.4s ease-out",
      }}
    >
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: 16,
          background: "var(--accent-dim)",
          border: "1px solid var(--accent-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 26,
          marginBottom: 4,
        }}
      >
        🧠
      </div>
      <p
        style={{
          margin: 0,
          fontSize: 17,
          fontWeight: 700,
          color: "var(--text)",
          letterSpacing: "-0.025em",
        }}
      >
        Olá, {name}!
      </p>
      <p
        style={{
          margin: 0,
          fontSize: 13.5,
          maxWidth: 300,
          lineHeight: 1.65,
          color: "var(--text-muted)",
        }}
      >
        Lembro de tudo que conversamos antes. Pode perguntar ou continuar de onde paramos.
      </p>
    </div>
  );
}

// ─── Header buttons ───────────────────────────────────────────────────────────

function HeaderBtn({
  onClick,
  title,
  children,
}: {
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        padding: "5px 11px",
        borderRadius: 7,
        border: "1px solid var(--border)",
        background: "transparent",
        color: "var(--text-secondary)",
        fontSize: 12,
        cursor: "pointer",
        touchAction: "manipulation",
        WebkitTapHighlightColor: "transparent",
        userSelect: "none",
        transition: "background 0.12s, color 0.12s",
        letterSpacing: "0.01em",
      } as React.CSSProperties}
    >
      {children}
    </button>
  );
}

// ─── Send / Stop buttons ─────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M8 13V3M4 7l4-4 4 4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <rect x="2.5" y="2.5" width="9" height="9" rx="2" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hasText, setHasText] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const streamingRef = useRef(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const sessionRef = useRef<Session | null>(null);

  useEffect(() => {
    setSession(getSession());
    setAuthChecked(true);
  }, []);

  useEffect(() => { sessionRef.current = session; }, [session]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  function resizeTextarea() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function readText(): string {
    return (textareaRef.current?.value ?? "").trim();
  }

  function clearTextarea() {
    if (textareaRef.current) {
      textareaRef.current.value = "";
      textareaRef.current.style.height = "auto";
    }
    setHasText(false);
  }

  function handleInput() {
    setHasText(readText().length > 0);
    resizeTextarea();
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
    streamingRef.current = true;
    setStreaming(true);
    setStreamingText("");

    const controller = new AbortController();
    abortRef.current = controller;

    let finalReply = "";
    let errorMsg = "";

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

      if (res.status === 401) {
        clearSession();
        setSession(null);
        return;
      }
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
      if (!(err instanceof Error && err.name === "AbortError")) {
        errorMsg = `falha de conexão — ${err instanceof Error ? err.message : "desconhecido"}`;
      }
    } finally {
      streamingRef.current = false;
      setStreaming(false);
      setStreamingText("");
      abortRef.current = null;
    }

    const content = finalReply.trim() || (errorMsg ? `⚠️ ${errorMsg}` : "");
    if (content) {
      const reply: ChatMessage = { role: "assistant", content };
      messagesRef.current = [...messagesRef.current, reply];
      setMessages([...messagesRef.current]);
    }
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSubmit();
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleNewConversation() {
    if (streamingRef.current) abortRef.current?.abort();
    if (session) clearConversationId(session.userId);
    messagesRef.current = [];
    setMessages([]);
    setStreamingText("");
    clearTextarea();
  }

  function handleLogout() {
    if (streamingRef.current) abortRef.current?.abort();
    clearSession();
    messagesRef.current = [];
    setMessages([]);
    setStreamingText("");
    clearTextarea();
    setSession(null);
  }

  if (!authChecked) return null;
  if (!session) {
    return <ProfileGate onAuthenticated={(s) => setSession(s)} />;
  }

  const isEmpty = messages.length === 0 && !streamingText;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100dvh",
        maxWidth: 820,
        margin: "0 auto",
        background: "var(--surface)",
        borderLeft: "1px solid var(--border)",
        borderRight: "1px solid var(--border)",
        overflow: "hidden",
      }}
    >
      {/* ── Header ── */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 20px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 9,
              background: "var(--accent-dim)",
              border: "1px solid var(--accent-border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              flexShrink: 0,
            }}
          >
            🧠
          </div>
          <div>
            <div
              style={{
                fontWeight: 700,
                fontSize: 15,
                lineHeight: 1.2,
                letterSpacing: "-0.025em",
                color: "var(--text)",
              }}
            >
              Memory Chat{" "}
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 400,
                  color: "var(--accent)",
                  letterSpacing: "0.02em",
                  opacity: 0.7,
                }}
              >
                {BUILD_TAG}
              </span>
            </div>
            <div
              style={{
                fontSize: 11,
                color: "var(--text-muted)",
                letterSpacing: "0.01em",
              }}
            >
              {session.displayName}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <HeaderBtn onClick={handleNewConversation}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
              <path d="M6 2v8M2 6h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
            Nova conversa
          </HeaderBtn>
          <HeaderBtn onClick={handleLogout} title="Trocar de perfil">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
              <path
                d="M8 2l3 3-3 3M1 5h10M4 10l-3-3 3-3"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Trocar
          </HeaderBtn>
        </div>
      </header>

      {/* ── Messages ── */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "24px 20px 8px",
          display: "flex",
          flexDirection: "column",
          WebkitOverflowScrolling: "touch",
        } as React.CSSProperties}
      >
        {isEmpty && <EmptyState name={session.displayName} />}

        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            role={msg.role}
            content={msg.content}
            userName={session.displayName}
          />
        ))}

        {/* Typing indicator */}
        {streaming && !streamingText && (
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
              marginBottom: 22,
              animation: "fadeSlide 0.18s ease-out",
            }}
          >
            <AssistantAvatar />
            <div style={{ paddingTop: 8 }}>
              <TypingDots />
            </div>
          </div>
        )}

        {streamingText && (
          <MessageBubble role="assistant" content={streamingText} userName={session.displayName} />
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input area ── */}
      <div
        style={{
          borderTop: "1px solid var(--border)",
          padding: "12px 20px",
          background: "var(--surface)",
          flexShrink: 0,
          paddingBottom: "max(12px, env(safe-area-inset-bottom))",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 8,
            background: "var(--surface-alt)",
            border: `1.5px solid ${inputFocused ? "var(--accent-border)" : "var(--border)"}`,
            borderRadius: 14,
            padding: "8px 8px 8px 16px",
            transition: "border-color 0.15s",
          }}
        >
          <textarea
            ref={textareaRef}
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            placeholder="Digite uma mensagem…"
            rows={1}
            style={{
              flex: 1,
              border: "none",
              outline: "none",
              resize: "none",
              background: "transparent",
              color: "var(--text)",
              lineHeight: 1.5,
              fontSize: 16,
              padding: "4px 0",
              maxHeight: 160,
              overflowY: "auto",
            }}
          />

          {streaming ? (
            <button
              type="button"
              onClick={handleStop}
              aria-label="Parar resposta"
              style={{
                flexShrink: 0,
                width: 38,
                height: 38,
                borderRadius: 10,
                border: "1px solid var(--danger-border)",
                background: "var(--danger-dim)",
                color: "var(--danger)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                touchAction: "manipulation",
                WebkitTapHighlightColor: "transparent",
                userSelect: "none",
                transition: "all 0.15s",
              } as React.CSSProperties}
            >
              <StopIcon />
            </button>
          ) : (
            <button
              type="button"
              onClick={doSubmit}
              aria-label="Enviar mensagem"
              style={{
                flexShrink: 0,
                width: 38,
                height: 38,
                borderRadius: 10,
                border: "none",
                background: hasText ? "var(--accent)" : "var(--surface-hover)",
                color: hasText ? "var(--bg)" : "var(--text-muted)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: hasText ? "pointer" : "default",
                touchAction: "manipulation",
                WebkitTapHighlightColor: "transparent",
                userSelect: "none",
                transition: "all 0.15s",
              } as React.CSSProperties}
            >
              <SendIcon />
            </button>
          )}
        </div>
        <p
          style={{
            margin: "6px 0 0",
            fontSize: 11,
            color: "var(--text-muted)",
            textAlign: "center",
            letterSpacing: "0.01em",
          }}
        >
          Enter para enviar · Shift+Enter para nova linha
        </p>
      </div>
    </div>
  );
}
