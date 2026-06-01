const MEMORY_API_BASE =
  process.env.MEMORY_API_URL || "http://localhost:8001";

// Headers padrão para a memory-api. O `skip_zrok_interstitial` (qualquer valor)
// pula a página de aviso do zrok quando a API é exposta por um share público —
// sem ele, requisições de API recebem HTML da interstitial em vez do JSON.
const MEMORY_API_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  skip_zrok_interstitial: "true",
};

export interface MemorySearchResult {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
}

export interface ConversationMemoryResult {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
}

export interface GraphEntity {
  name: string;
  type: string;
  description: string;
}

export interface ConversationEvent {
  event: string;
  content?: string;
  text?: string;
  episodic?: MemorySearchResult[];
  graph?: GraphEntity[];
  name?: string;
  input?: Record<string, unknown>;
  tool_calls?: unknown[];
  message?: string;
}

/** Store a conversation turn in episodic memory. */
export async function storeMemory(
  userId: string,
  conversationId: string,
  content: string,
): Promise<string> {
  const res = await fetch(`${MEMORY_API_BASE}/store`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({
      user_id: userId,
      conversation_id: conversationId,
      content,
      metadata: { source: "chat" },
      extract_kg: true,
    }),
  });
  if (!res.ok) throw new Error(`Store failed: ${res.status}`);
  const data = await res.json();
  return data.memory_id;
}

/** Search episodic memory by semantic similarity. */
export async function recallMemories(
  query: string,
  userId?: string,
  topK?: number,
): Promise<MemorySearchResult[]> {
  const res = await fetch(`${MEMORY_API_BASE}/retrieve`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({ query, user_id: userId, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Retrieve failed: ${res.status}`);
  const data = await res.json();
  return data.memories;
}

/** Search conversation memory in pgvector. */
export async function recallConversationMemories(
  query: string,
  userId?: string,
  topK?: number,
): Promise<ConversationMemoryResult[]> {
  const res = await fetch(`${MEMORY_API_BASE}/memory/search`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({ query, user_id: userId, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Conversation memory search failed: ${res.status}`);
  const data = await res.json();
  return data.results || [];
}

/** Store a conversation message in pgvector. */
export async function storeConversationMessage(
  threadId: string,
  role: "user" | "assistant" | "system",
  content: string,
  userId: string,
): Promise<string> {
  const res = await fetch(`${MEMORY_API_BASE}/memory/messages`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({
      thread_id: threadId,
      role,
      content,
      metadata: { user_id: userId, source: "chat" },
    }),
  });
  if (!res.ok) throw new Error(`Conversation message store failed: ${res.status}`);
  const data = await res.json();
  return data.message_id;
}

/** Search semantic memory facts (e.g. user name, preferences). */
export async function searchSemanticFacts(
  query: string,
  userId?: string,
  topK = 3,
  minSimilarity = 0.6,
): Promise<MemorySearchResult[]> {
  const res = await fetch(`${MEMORY_API_BASE}/memory/semantic/search`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({
      query,
      user_id: userId,
      top_k: topK,
      min_similarity: minSimilarity,
    }),
  });
  if (!res.ok) throw new Error(`Semantic search failed: ${res.status}`);
  const data = await res.json();
  return data.results || [];
}

/** Check whether an assistant response is confident enough to be stored. */
export async function checkResponseConfidence(
  text: string,
  threshold = 0.72,
): Promise<boolean> {
  const res = await fetch(`${MEMORY_API_BASE}/memory/confidence`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({ text, threshold }),
  });
  if (!res.ok) return true; // fail open: persist on error
  const data = await res.json();
  return data.confident as boolean;
}

/** Search the knowledge graph. */
export async function searchGraph(
  query: string,
  typeFilter?: string,
): Promise<GraphEntity[]> {
  const res = await fetch(`${MEMORY_API_BASE}/graph/search`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({ query, type_filter: typeFilter }),
  });
  if (!res.ok) throw new Error(`Graph search failed: ${res.status}`);
  const data = await res.json();
  return data.entities;
}

/** Run the full conversation loop via SSE and yield events. */
export async function* runConversationLoop(
  userId: string,
  conversationId: string,
  message: string,
  history?: { role: string; content: string }[],
  agentId?: string,
  signal?: AbortSignal,
): AsyncGenerator<ConversationEvent> {
  const res = await fetch(`${MEMORY_API_BASE}/conversation`, {
    method: "POST",
    headers: MEMORY_API_HEADERS,
    body: JSON.stringify({
      user_id: userId,
      conversation_id: conversationId,
      message,
      history,
      agent_id: agentId,
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Conversation loop failed: ${res.status} — ${text}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // skip unparseable frames
        }
      }
    }
  }
}
