import { NextRequest } from "next/server";
import { createClaudeStream } from "@/lib/claude/client";
import {
  recallConversationMemories,
  recallMemories,
  searchSemanticFacts,
  storeConversationMessage,
  storeMemory,
} from "@/lib/memory/client";
import type Anthropic from "@anthropic-ai/sdk";

export const runtime = "edge";

type ChatRequest = {
  messages: { role: "user" | "assistant"; content: string }[];
};

export async function POST(req: NextRequest) {
  const { messages } = (await req.json()) as ChatRequest;
  const userId = req.headers.get("x-user-id") || "default";
  const conversationId = req.headers.get("x-conversation-id") || "default";
  const latestUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user")?.content;

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
        let memoryContext = "";

        if (latestUserMessage) {
          try {
            // Recall episodic + conversation memories and stored identity facts in parallel.
            const [episodicMemories, conversationMemories, identityFacts] =
              await Promise.all([
                recallMemories(latestUserMessage, userId, 8),
                recallConversationMemories(latestUserMessage, userId, 8),
                searchSemanticFacts("user name assistant name", userId, 4, 0.55),
              ]);

            const allMemories = [...episodicMemories, ...conversationMemories];

            if (identityFacts.length > 0) {
              memoryContext = identityFacts
                .map((f) => f.content)
                .join("\n");
            }

            if (allMemories.length > 0) {
              const episodicContext = allMemories
                .map(
                  (memory, index) =>
                    `Memory ${index + 1} (similarity ${memory.similarity}): ${memory.content}`,
                )
                .join("\n");
              memoryContext = memoryContext
                ? `${memoryContext}\n${episodicContext}`
                : episodicContext;
            }
          } catch (error) {
            console.error("[chat] silent memory recall failed", error);
          }
        }

        const anthropicMessages: Anthropic.MessageParam[] = messages.map(
          (msg) => ({ role: msg.role, content: msg.content }),
        );

        let finalText = "";

        if (latestUserMessage) {
          void storeConversationMessage(
            conversationId,
            "user",
            latestUserMessage,
            userId,
          ).catch((error) => {
            console.error("[chat] conversation message store failed", error);
          });
        }

        const response = createClaudeStream(anthropicMessages, req.signal, memoryContext);

        for await (const event of await response) {
          if (
            event.type === "content_block_delta" &&
            event.delta.type === "text_delta"
          ) {
            finalText += event.delta.text;
            enqueue({ event: "text", content: event.delta.text });
          }
        }

        const finalReply = finalText.trim();

        // Sinaliza o fim ao cliente IMEDIATAMENTE — o botão reabilita assim que
        // o texto termina, sem esperar a persistência (que inclui extração de
        // entidades via LLM e leva segundos). A memória é salva em segundo
        // plano, igual à mensagem do usuário acima.
        enqueue({ event: "done", text: finalText });

        if (latestUserMessage && finalReply) {
          void storeConversationMessage(
            conversationId,
            "assistant",
            finalReply,
            userId,
          ).catch((error) => {
            console.error("[chat] assistant message store failed", error);
          });
          void storeMemory(
            userId,
            conversationId,
            `User: ${latestUserMessage}\nAssistant: ${finalReply}`,
          ).catch((error) => {
            console.error("[chat] memory store failed", error);
          });
        }
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
