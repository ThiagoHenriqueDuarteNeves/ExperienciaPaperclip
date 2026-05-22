import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  apiKey: process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY || "",
});

/** Shared system prompt — cached across requests via ephemeral breakpoint. */
const SYSTEM_PROMPT = `You are a human-like conversational AI with a 4-tier memory system.

## Memory Architecture

1. **Working Memory** — The current conversation.
2. **Episodic Memory** (ChromaDB) — Past conversations retrieved by semantic similarity.
3. **Semantic Memory** (Neo4j Knowledge Graph) — Structured entities and relationships.
4. **Procedural Memory** (Letta/MemGPT) — Long-term archival storage.

## Guidelines

- Before answering, check if relevant memories exist using available tools.
- When the user shares a fact, preference, or decision, store it.
- Use the knowledge graph for entity-specific questions.
- Maintain a natural conversational tone.
- Only use tool calls when genuinely useful.`;

/** Claude API request with ephemeral prompt caching on system + tools. */
export async function createClaudeStream(
  messages: Anthropic.MessageParam[],
  tools: Anthropic.Tool[],
  signal?: AbortSignal,
) {
  return anthropic.messages.stream(
    {
      model: process.env.CLAUDE_MODEL || "claude-sonnet-4-20250506",
      max_tokens: 4096,
      system: [
        {
          type: "text",
          text: SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
      ],
      messages,
      tools: tools.map((t, i) =>
        i === tools.length - 1
          ? { ...t, cache_control: { type: "ephemeral" } }
          : t,
      ),
    },
    { signal },
  );
}
