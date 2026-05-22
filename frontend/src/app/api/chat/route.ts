import { NextRequest } from "next/server";
import { createClaudeStream } from "@/lib/claude/client";
import { MEMORY_TOOLS } from "@/lib/claude/tools";
import { storeMemory } from "@/lib/memory/client";
import type Anthropic from "@anthropic-ai/sdk";

export const runtime = "edge";

/** Map of tool names to executor functions. */
const toolExecutors: Record<
  string,
  (args: Record<string, unknown>, req: NextRequest) => Promise<string>
> = {
  remember_conversation: async (args, req) => {
    const userId = req.headers.get("x-user-id") || "default";
    const conversationId = req.headers.get("x-conversation-id") || "default";
    const content = args.content as string;
    const category = (args.category as string) || "general";
    const memoryId = await storeMemory(userId, conversationId, content);
    return JSON.stringify({ stored: true, memory_id: memoryId, category });
  },

  recall_memories: async (args, _req) => {
    const userId = _req.headers.get("x-user-id") || undefined;
    const { recallMemories } = await import("@/lib/memory/client");
    const memories = await recallMemories(
      args.query as string,
      userId,
      (args.top_k as number) || 5,
    );
    return JSON.stringify({ memories, count: memories.length });
  },

  search_knowledge_graph: async (args, _req) => {
    const { searchGraph } = await import("@/lib/memory/client");
    const entities = await searchGraph(
      args.query as string,
      args.type_filter as string | undefined,
    );
    return JSON.stringify({ entities, count: entities.length });
  },

  store_archival_memory: async (args, _req) => {
    const memoryApiBase =
      process.env.MEMORY_API_URL || "http://localhost:8001";
    const agentId = _req.headers.get("x-agent-id") || "default";
    const res = await fetch(
      `${memoryApiBase}/letta/agents/${agentId}/archival`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: args.content }),
      },
    );
    const ok = res.ok;
    return JSON.stringify({ stored: ok });
  },
};

type ChatRequest = {
  messages: { role: "user" | "assistant"; content: string }[];
};

export async function POST(req: NextRequest) {
  const { messages } = (await req.json()) as ChatRequest;

  const encoder = new TextEncoder();
  let aborted = false;
  req.signal.addEventListener("abort", () => {
    aborted = true;
  });

  const stream = new ReadableStream({
    async start(controller) {
      const enqueue = (event: Record<string, unknown>) => {
        if (!aborted) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
          );
        }
      };

      try {
        const anthropicMessages: Anthropic.MessageParam[] = [];
        for (const msg of messages) {
          anthropicMessages.push({ role: msg.role, content: msg.content });
        }

        // Tool-use loop
        let maxRounds = 5;
        let finalText = "";

        while (maxRounds-- > 0 && !aborted) {
          const response = createClaudeStream(
            anthropicMessages,
            MEMORY_TOOLS,
            req.signal,
          );
          let currentBlock: Anthropic.TextBlock | Anthropic.ToolUseBlock | null = null;
          const toolRequests: Anthropic.ToolUseBlock[] = [];

          for await (const event of await response) {
            if (
              event.type === "content_block_start" &&
              event.content_block.type === "text"
            ) {
              currentBlock = event.content_block;
            }

            if (
              event.type === "content_block_start" &&
              event.content_block.type === "tool_use"
            ) {
              currentBlock = event.content_block;
              toolRequests.push(event.content_block);
            }

            if (
              event.type === "content_block_delta" &&
              event.delta.type === "text_delta"
            ) {
              enqueue({ event: "text", content: event.delta.text });
            }

            if (
              event.type === "content_block_delta" &&
              event.delta.type === "input_json_delta"
            ) {
              // Tool input accumulates — we'll send at tool execution
            }
          }

          // After the stream completes, check the message stop reason
          // by checking if any tool_use blocks were accumulated
          if (toolRequests.length > 0) {
            // Add assistant message with tool_use blocks
            const assistantContent: Anthropic.ContentBlock[] = toolRequests.map(
              (tr) => ({
                type: "tool_use" as const,
                id: tr.id,
                name: tr.name,
                input: tr.input,
              }),
            );
            anthropicMessages.push({
              role: "assistant",
              content: assistantContent,
            });

            // Execute tools and add results
            const toolResults: Anthropic.ToolResultBlockParam[] = [];
            for (const tr of toolRequests) {
              enqueue({
                event: "tool_call",
                name: tr.name,
                input: tr.input,
              });

              const executor = toolExecutors[tr.name];
              const result = executor
                ? await executor(tr.input as Record<string, unknown>, req)
                : JSON.stringify({ error: `Unknown tool: ${tr.name}` });

              toolResults.push({
                type: "tool_result",
                tool_use_id: tr.id,
                content: result,
              });
            }

            anthropicMessages.push({
              role: "user",
              content: toolResults,
            });
          } else {
            // No tool calls — we're done
            break;
          }
        }

        enqueue({ event: "done", text: finalText });
      } catch (err) {
        enqueue({
          event: "error",
          message: err instanceof Error ? err.message : "Unknown error",
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
