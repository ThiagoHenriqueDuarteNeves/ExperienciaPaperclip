"use client";

import { useState, useCallback } from "react";

type Segment =
  | { type: "text"; content: string }
  | { type: "code"; lang: string; content: string };

function parseSegments(text: string): Segment[] {
  const segments: Segment[] = [];
  const regex = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: "text", content: text.slice(lastIndex, match.index) });
    }
    segments.push({
      type: "code",
      lang: match[1].trim() || "code",
      content: match[2].trimEnd(),
    });
    lastIndex = match.index + match[0].length;
  }

  const remaining = text.slice(lastIndex);
  if (remaining) {
    const openIdx = remaining.indexOf("```");
    if (openIdx !== -1) {
      if (openIdx > 0) {
        segments.push({ type: "text", content: remaining.slice(0, openIdx) });
      }
      const afterOpen = remaining.slice(openIdx + 3);
      const nlIdx = afterOpen.indexOf("\n");
      const lang = nlIdx > -1 ? afterOpen.slice(0, nlIdx).trim() : afterOpen.trim();
      const code = nlIdx > -1 ? afterOpen.slice(nlIdx + 1) : "";
      segments.push({ type: "code", lang: lang || "code", content: code });
    } else {
      segments.push({ type: "text", content: remaining });
    }
  }

  return segments;
}

function InlineText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\n]+\*)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4)
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
          return (
            <code
              key={i}
              style={{
                fontFamily: "var(--jetbrains), monospace",
                fontSize: "0.84em",
                padding: "2px 6px",
                borderRadius: 4,
                background: "var(--code-inline-bg)",
                color: "var(--code-inline-text)",
                border: "1px solid var(--code-inline-border)",
                letterSpacing: "0.01em",
              }}
            >
              {part.slice(1, -1)}
            </code>
          );
        if (part.startsWith("*") && part.endsWith("*") && part.length > 2)
          return <em key={i}>{part.slice(1, -1)}</em>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function TextSegment({ content }: { content: string }) {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let listKeyCounter = 0;

  function flushList() {
    if (!listItems.length) return;
    const Tag = listType === "ol" ? "ol" : "ul";
    const key = `list-${listKeyCounter++}`;
    elements.push(
      <Tag
        key={key}
        style={{
          margin: "6px 0",
          paddingLeft: 22,
          display: "flex",
          flexDirection: "column",
          gap: 3,
        }}
      >
        {listItems.map((item, idx) => (
          <li key={idx} style={{ lineHeight: 1.6 }}>
            <InlineText text={item} />
          </li>
        ))}
      </Tag>
    );
    listItems = [];
    listType = null;
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const olMatch = line.match(/^(\d+)\.\s+(.+)/);
    if (olMatch) {
      if (listType !== "ol") flushList();
      listType = "ol";
      listItems.push(olMatch[2]);
      continue;
    }

    const ulMatch = line.match(/^[-*+]\s+(.+)/);
    if (ulMatch) {
      if (listType !== "ul") flushList();
      listType = "ul";
      listItems.push(ulMatch[1]);
      continue;
    }

    flushList();

    if (line.startsWith("### ")) {
      elements.push(
        <div
          key={i}
          style={{ margin: "14px 0 4px", fontSize: "0.95em", fontWeight: 700, color: "var(--text)" }}
        >
          <InlineText text={line.slice(4)} />
        </div>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <div
          key={i}
          style={{ margin: "18px 0 6px", fontSize: "1.05em", fontWeight: 700, color: "var(--text)" }}
        >
          <InlineText text={line.slice(3)} />
        </div>
      );
    } else if (line.startsWith("# ")) {
      elements.push(
        <div
          key={i}
          style={{ margin: "20px 0 8px", fontSize: "1.15em", fontWeight: 700, color: "var(--text)" }}
        >
          <InlineText text={line.slice(2)} />
        </div>
      );
    } else if (line.startsWith("> ")) {
      elements.push(
        <blockquote
          key={i}
          style={{
            borderLeft: "2px solid var(--accent)",
            margin: "6px 0",
            paddingLeft: 12,
            color: "var(--text-secondary)",
            fontStyle: "italic",
          }}
        >
          <InlineText text={line.slice(2)} />
        </blockquote>
      );
    } else if (line.trim() === "") {
      if (i > 0 && i < lines.length - 1) {
        elements.push(<div key={i} style={{ height: 6 }} />);
      }
    } else {
      elements.push(
        <p key={i} style={{ margin: 0, lineHeight: 1.65 }}>
          <InlineText text={line} />
        </p>
      );
    }
  }

  flushList();
  return <>{elements}</>;
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <rect x="4.5" y="1" width="6.5" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
      <path
        d="M7.5 1V3H1V11H6.5"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path
        d="M2 6.5L4.5 9L10 3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CodeBlock({ lang, content }: { lang: string; content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      } else {
        const el = document.createElement("textarea");
        el.value = content;
        el.style.cssText = "position:fixed;opacity:0";
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  }, [content]);

  return (
    <div
      style={{
        margin: "10px 0",
        borderRadius: 10,
        overflow: "hidden",
        border: "1px solid var(--code-border)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "7px 14px",
          background: "var(--code-header-bg)",
          borderBottom: "1px solid var(--code-border)",
          gap: 8,
        }}
      >
        <span
          style={{
            fontFamily: "var(--jetbrains), monospace",
            fontSize: 11,
            color: "var(--text-muted)",
            letterSpacing: "0.06em",
          }}
        >
          {lang}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          aria-label="Copiar código"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 5,
            padding: "3px 10px",
            borderRadius: 5,
            border: `1px solid ${copied ? "var(--accent-border)" : "var(--code-border)"}`,
            background: copied ? "var(--accent-dim)" : "transparent",
            color: copied ? "var(--accent)" : "var(--text-muted)",
            fontSize: 11,
            fontFamily: "var(--outfit), system-ui, sans-serif",
            cursor: "pointer",
            transition: "all 0.15s ease",
            letterSpacing: "0.02em",
            flexShrink: 0,
          }}
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? "copiado" : "copiar"}
        </button>
      </div>
      <pre
        style={{
          margin: 0,
          padding: "14px 16px",
          overflowX: "auto",
          background: "var(--code-bg)",
          color: "var(--code-text)",
          fontFamily: "var(--jetbrains), monospace",
          lineHeight: 1.7,
          fontSize: 13,
          WebkitOverflowScrolling: "touch",
        } as React.CSSProperties}
      >
        <code>{content}</code>
      </pre>
    </div>
  );
}

export function MarkdownRenderer({ content }: { content: string }) {
  const segments = parseSegments(content);
  return (
    <div style={{ fontSize: 14.5, lineHeight: 1.65, color: "var(--text)" }}>
      {segments.map((seg, i) =>
        seg.type === "code" ? (
          // Key on type+content so a CodeBlock's copied state can't bleed to a
          // different block if segment indices shift.
          <CodeBlock key={`c${i}:${seg.content.length}`} lang={seg.lang} content={seg.content} />
        ) : (
          <TextSegment key={`t${i}`} content={seg.content} />
        )
      )}
    </div>
  );
}
