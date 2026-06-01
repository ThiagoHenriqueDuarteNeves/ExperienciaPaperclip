import Anthropic from "@anthropic-ai/sdk";

// Support both Anthropic and Deepseek (Deepseek provides Anthropic-compatible API)
const getAnthropicClient = () => {
  const apiKey = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY || "";
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const baseURL = provider === "deepseek" 
    ? "https://api.deepseek.com/anthropic"
    : process.env.ANTHROPIC_BASE_URL || undefined;

  return new Anthropic({
    apiKey,
    ...(baseURL && { baseURL }),
  });
};

const anthropic = getAnthropicClient();

/** Shared system prompt — cached across requests via ephemeral breakpoint. */
const SYSTEM_PROMPT = `You are a warm, human-like conversational assistant with persistent memory across conversations.

## How your memory works
Relevant records from past conversations are retrieved automatically and given to you in a system section titled "Retrieved memories" — when relevant memory exists, it is ALREADY provided to you. You do not call any tool to fetch it.

## Grounding and honesty — CRITICAL, follow strictly
- The "Retrieved memories" section together with the current conversation are your ONLY sources of truth about the user, past events, names, stories, dates and facts.
- Reproduce facts, names, dates, events and stories EXACTLY as they appear in memory. Never alter, embellish, dramatize, summarize away or contradict them. If memory records a story, retell that exact story — do not "improve" or reinvent it.
- If the information needed to answer is NOT in the retrieved memories or in the current conversation, say plainly that you do not have it recorded, or ask the user. NEVER invent, guess, or fill in missing details and present them as real.
- A short, honest "I don't have that recorded" is always better than a confident answer that might be wrong.
- If a memory conflicts with your own assumptions, the memory always wins.
- Never claim something happened, or describe events, unless it is supported by memory or the conversation.

## Style
- Natural, conversational tone. Reply in the same language the user is using (Portuguese by default).
- Be concise unless the user asks for detail.`;

/** Claude API request with ephemeral prompt caching on system. */
export async function createClaudeStream(
  messages: Anthropic.MessageParam[],
  signal?: AbortSignal,
  memoryContext?: string,
) {
  const system: Anthropic.TextBlockParam[] = [
    {
      type: "text",
      text: SYSTEM_PROMPT,
      cache_control: { type: "ephemeral" },
    },
  ];

  // Memória recuperada entra como bloco de sistema AUTORITATIVO (não como fala
  // do assistente). Muda por requisição, então fica fora do bloco cacheado.
  if (memoryContext && memoryContext.trim()) {
    system.push({
      type: "text",
      text:
        "## Retrieved memories (AUTHORITATIVE — your only record of the past)\n" +
        "These are real records from previous conversations with this user. " +
        "Treat them as ground truth. Answer using ONLY these records plus the " +
        "current conversation. Reproduce any story or detail faithfully — do not " +
        "alter or invent anything. If the answer is not here, say you do not have " +
        "it recorded instead of guessing.\n\n" +
        memoryContext,
    });
  }

  return anthropic.messages.stream(
    {
      model: process.env.CLAUDE_MODEL || "claude-sonnet-4-20250506",
      max_tokens: 4096,
      system,
      messages,
    },
    { signal },
  );
}
