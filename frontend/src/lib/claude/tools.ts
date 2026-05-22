import type Anthropic from "@anthropic-ai/sdk";

/** Memory tool definitions for the 4-tier memory system. */
export const MEMORY_TOOLS: Anthropic.Tool[] = [
  {
    name: "remember_conversation",
    description:
      "Store a conversation or important fact in episodic memory. " +
      "Use when the user shares something worth remembering — " +
      "preferences, decisions, personal facts, project context.",
    input_schema: {
      type: "object",
      properties: {
        content: {
          type: "string",
          description: "The fact or conversation snippet to remember.",
        },
        category: {
          type: "string",
          enum: ["fact", "preference", "decision", "context", "general"],
          description: "Category for the stored memory.",
        },
      },
      required: ["content"],
    },
  },
  {
    name: "recall_memories",
    description:
      "Search episodic memory for semantically similar past conversations " +
      "or facts. Use to recall what the user previously mentioned about a topic.",
    input_schema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query.",
        },
        top_k: {
          type: "integer",
          description: "Number of results to return (default 5).",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "search_knowledge_graph",
    description:
      "Search the semantic knowledge graph for entities, concepts, " +
      "or relationships. Use for structured knowledge about people, " +
      "technologies, projects, or organizations.",
    input_schema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Entity or concept to look up.",
        },
        type_filter: {
          type: "string",
          description: "Optional entity type filter (person, technology, etc.).",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "store_archival_memory",
    description:
      "Store a detailed passage in Letta procedural/archival memory. " +
      "Use for long-term knowledge that should persist across sessions.",
    input_schema: {
      type: "object",
      properties: {
        content: {
          type: "string",
          description: "Detailed passage to archive.",
        },
      },
      required: ["content"],
    },
  },
];
