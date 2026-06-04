"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ProfileGate from "../components/ProfileGate";
import { clearSession, getSession, type Session } from "../lib/auth";

const BUILD_TAG = "v13";
const CHAT_URL = process.env.NEXT_PUBLIC_CHAT_API_URL ?? "/api/chat";

/**
 * Gera um UUID. crypto.randomUUID() só existe em contexto seguro
 * (HTTPS ou localhost) — em HTTP via IP de rede (celular) ela é undefined.
 * Por isso há fallback puro em JS.
 */
function genId(): string {
  try {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
  } catch { /* indisponível — usa fallback */ }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Parse de uma linha SSE "data: {...}". */
function parseSSELine(line: string): { text?: string; error?: string } {
  if (!line.startsWith("data: ")) return {};
  try {
    const event = JSON.parse(line.slice(6));
    if (event.event === "text") return { text: event.content as string };
    if (event.event === "error") return { error: event.message as string };
  } catch { /* frame incompleto/ignorável */ }
  return {};
}

/** Parse do corpo SSE inteiro de uma vez (fallback sem streaming). */
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

function TypingDots() {
  return (
    <span style={{ display: "inline-flex", gap: 4, alignItems: "center", height: 18 }}>
      {[0, 1, 2].map((i) => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: "50%",
          background: "var(--text-muted)", display: "inline-block",
          animation: "blink 1.2s ease-in-out infinite",
          animationDelay: `${i * 0.2}s`,
        }} />
      ))}
    </span>
  );
}

function MessageBubble({ role, content }: {
  role: "user" | "assistant"; content: string;
}) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: isUser ? "row-reverse" : "row",
      alignItems: "flex-end", gap: 8, marginBottom: 12,
      animation: "fadeSlide 0.18s ease-out",
    }}>
      <div style={{
        flexShrink: 0, width: 32, height: 32, borderRadius: "50%",
        background: isUser ? "var(--user-bg)" : "var(--surface-alt)",
        border: "1px solid var(--border)",
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
      }}>
        {isUser ? "👤" : "🧠"}
      </div>
      <div style={{
        maxWidth: "72%", padding: "10px 14px",
        borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        background: isUser ? "var(--user-bg)" : "var(--assistant-bg)",
        color: isUser ? "var(--user-text)" : "var(--assistant-text)",
        boxShadow: "var(--shadow-sm)",
        border: isUser ? "none" : "1px solid var(--border)",
        whiteSpace: "pre-wrap", wordBreak: "break-word",
        lineHeight: 1.55, fontSize: 14,
      }}>
        {content}
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hasText, setHasText] = useState(false);   // cosmético: cor do botão
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

  // Load any persisted session on mount (localStorage is client-only).
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

  /** Lê o texto direto do DOM — fonte única de verdade, imune a falhas de onChange no mobile. */
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

  // onInput dispara em TODO input (digitação, colar, swipe, autocorreção) em qualquer navegador.
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
        // Token expired/invalid — drop session and show the gate again.
        clearSession();
        setSession(null);
        return;
      }
      if (!res.ok) {
        errorMsg = `HTTP ${res.status} ${res.statusText}`;
      } else if (!res.body || typeof res.body.getReader !== "function") {
        // Navegador sem streaming de body (alguns mobile): lê tudo de uma vez.
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

    // Sempre mostra algo: a resposta, ou um erro visível (nunca silêncio).
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
    setSession(null); // keeps the profile list — only ends the session
  }

  // Auth gate: wait for the localStorage check, then require a session.
  if (!authChecked) return null;
  if (!session) {
    return <ProfileGate onAuthenticated={(s) => { setSession(s); }} />;
  }

  const isEmpty = messages.length === 0 && !streamingText;

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100dvh", maxWidth: 760, margin: "0 auto",
      background: "var(--surface)", boxShadow: "var(--shadow-md)",
      overflow: "hidden",
    }}>

      {/* Header */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 20px", borderBottom: "1px solid var(--border)",
        background: "var(--surface)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, lineHeight: 1.2 }}>
              Memory Chat <span style={{ fontSize: 10, fontWeight: 400, color: "var(--btn-primary)" }}>{BUILD_TAG}</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              👤 {session.displayName}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            type="button"
            onClick={handleNewConversation}
            style={{
              padding: "6px 12px", borderRadius: 8,
              border: "1px solid var(--border)", background: "transparent",
              color: "var(--text-muted)", fontSize: 12,
              display: "flex", alignItems: "center", gap: 5,
              cursor: "pointer",
              touchAction: "manipulation",
              WebkitTapHighlightColor: "transparent",
              userSelect: "none",
            } as React.CSSProperties}
          >
            ✏️ Nova conversa
          </button>
          <button
            type="button"
            onClick={handleLogout}
            title="Trocar de perfil"
            style={{
              padding: "6px 12px", borderRadius: 8,
              border: "1px solid var(--border)", background: "transparent",
              color: "var(--text-muted)", fontSize: 12,
              display: "flex", alignItems: "center", gap: 5,
              cursor: "pointer",
              touchAction: "manipulation",
              WebkitTapHighlightColor: "transparent",
              userSelect: "none",
            } as React.CSSProperties}
          >
            🔄 Trocar
          </button>
        </div>
      </header>

      {/* Messages */}
      <div style={{
        flex: 1, overflowY: "auto", padding: "20px 16px",
        display: "flex", flexDirection: "column",
        WebkitOverflowScrolling: "touch",
      } as React.CSSProperties}>
        {isEmpty && (
          <div style={{
            flex: 1, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            gap: 12, color: "var(--text-muted)", textAlign: "center", padding: "0 24px",
          }}>
            <span style={{ fontSize: 48 }}>🧠</span>
            <p style={{ margin: 0, fontSize: 16, fontWeight: 600, color: "var(--text)" }}>
              Olá! Eu tenho memória.
            </p>
            <p style={{ margin: 0, fontSize: 13, maxWidth: 360 }}>
              Lembro de conversas anteriores, nomes e contextos. Comece digitando sua mensagem abaixo.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} />
        ))}

        {streaming && !streamingText && (
          <div style={{
            display: "flex", alignItems: "flex-end", gap: 8,
            marginBottom: 12, animation: "fadeSlide 0.18s ease-out",
          }}>
            <div style={{
              flexShrink: 0, width: 32, height: 32, borderRadius: "50%",
              background: "var(--surface-alt)", border: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
            }}>🧠</div>
            <div style={{
              padding: "10px 14px", borderRadius: "18px 18px 18px 4px",
              background: "var(--assistant-bg)", border: "1px solid var(--border)",
              boxShadow: "var(--shadow-sm)",
            }}>
              <TypingDots />
            </div>
          </div>
        )}

        {streamingText && <MessageBubble role="assistant" content={streamingText} />}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div style={{
        borderTop: "1px solid var(--border)", padding: "12px 16px",
        background: "var(--surface)", flexShrink: 0,
        paddingBottom: "max(12px, env(safe-area-inset-bottom))",
      }}>
        <div style={{
          display: "flex", alignItems: "flex-end", gap: 8,
          background: "var(--input-bg)",
          border: `1.5px solid var(--input-border)`,
          borderRadius: 14, padding: "6px 6px 6px 14px",
        }}>
          <textarea
            ref={textareaRef}
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Digite uma mensagem…"
            rows={1}
            style={{
              flex: 1, border: "none", outline: "none",
              resize: "none", background: "transparent",
              color: "var(--text)", lineHeight: 1.5,
              fontSize: 16,
              padding: "4px 0", maxHeight: 160, overflowY: "auto",
            }}
          />

          {streaming ? (
            <button
              type="button"
              onClick={handleStop}
              style={{
                flexShrink: 0, width: 44, height: 44, borderRadius: 10,
                border: "none", background: "var(--btn-stop)", color: "#fff",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 18, cursor: "pointer",
                touchAction: "manipulation",
                WebkitTapHighlightColor: "transparent",
                userSelect: "none",
              } as React.CSSProperties}
              aria-label="Parar resposta"
            >
              ⏹
            </button>
          ) : (
            <button
              type="button"
              onClick={doSubmit}
              style={{
                flexShrink: 0, width: 44, height: 44, borderRadius: 10,
                border: "none",
                background: "var(--btn-primary)",
                color: "#fff",
                opacity: hasText ? 1 : 0.55,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 18, cursor: "pointer",
                touchAction: "manipulation",
                WebkitTapHighlightColor: "transparent",
                userSelect: "none",
                transition: "opacity 0.15s",
              } as React.CSSProperties}
              aria-label="Enviar mensagem"
            >
              ➤
            </button>
          )}
        </div>
        <p style={{
          margin: "6px 0 0", fontSize: 11,
          color: "var(--text-muted)", textAlign: "center",
        }}>
          Enter para enviar · Shift+Enter para nova linha
        </p>
      </div>
    </div>
  );
}
