"use client";

import { useState, useCallback, useRef } from "react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim() || streaming) return;

      const userMessage: ChatMessage = { role: "user", content: input.trim() };
      const newMessages = [...messages, userMessage];
      setMessages(newMessages);
      setInput("");
      setStreaming(true);
      setStreamingText("");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: newMessages }),
          signal: controller.signal,
        });

        if (!res.ok) {
          setStreamingText(`Error: ${res.status} ${res.statusText}`);
          return;
        }

        const reader = res.body?.getReader();
        if (!reader) {
          setStreamingText("Error: No response stream");
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const event = JSON.parse(line.slice(6));
                if (event.event === "text") {
                  fullText += event.content;
                  setStreamingText(fullText);
                } else if (event.event === "done") {
                  // finalize
                } else if (event.event === "error") {
                  setStreamingText(`Error: ${event.message}`);
                }
              } catch {
                // skip unparseable events
              }
            }
          }
        }

        // Commit the streaming text as a final assistant message
        if (fullText) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullText },
          ]);
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setStreamingText(
          `Error: ${err instanceof Error ? err.message : "Unknown"}`,
        );
      } finally {
        setStreaming(false);
        setStreamingText("");
        abortRef.current = null;
      }
    },
    [input, messages, streaming],
  );

  const handleStop = () => {
    abortRef.current?.abort();
  };

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: 24,
        display: "flex",
        flexDirection: "column",
        height: "100dvh",
      }}
    >
      <h1 style={{ fontSize: 20, fontWeight: 600, margin: "0 0 16px" }}>
        Memory Chat
      </h1>

      <div
        style={{
          flex: 1,
          overflowY: "auto",
          marginBottom: 16,
          padding: "8px 0",
        }}
      >
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: 12,
              padding: "8px 12px",
              borderRadius: 8,
              background: msg.role === "user" ? "#e8f0fe" : "#f0f0f0",
              whiteSpace: "pre-wrap",
            }}
          >
            <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong>{" "}
            {msg.content}
          </div>
        ))}
        {streamingText && (
          <div
            style={{
              marginBottom: 12,
              padding: "8px 12px",
              borderRadius: 8,
              background: "#f0f0f0",
              whiteSpace: "pre-wrap",
            }}
          >
            <strong>Assistant:</strong> {streamingText}
          </div>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", gap: 8, padding: "8px 0" }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          style={{
            flex: 1,
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #ccc",
            fontSize: 14,
          }}
          disabled={streaming}
        />
        {streaming ? (
          <button
            type="button"
            onClick={handleStop}
            style={{
              padding: "8px 16px",
              borderRadius: 8,
              border: "none",
              background: "#e44",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            style={{
              padding: "8px 16px",
              borderRadius: 8,
              border: "none",
              background: "#0070f3",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            Send
          </button>
        )}
      </form>
    </div>
  );
}
