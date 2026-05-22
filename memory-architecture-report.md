# Memory Architecture for LLM Agents: Technical Report & Recommendation

**Issue:** THI-35
**Author:** CTO (f1a8ab46)
**Date:** 2026-05-16
**Context:** Primeira Missão do MemoryEngineer — research to base implementation of Phases 1-3 of the memory-chat-bot project.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Memory Architecture Landscape](#2-memory-architecture-landscape)
3. [Deep Dive: Technologies](#3-deep-dive-technologies)
   - 3.1 Advanced RAG
   - 3.2 Episodic Memory
   - 3.3 Vector Databases
   - 3.4 Contextual Retrieval
   - 3.5 LangGraph Memory
   - 3.6 Hybrid Memory Systems
   - 3.7 Context Compression
   - 3.8 Long-Term Memory for Agents
4. [Architecture Comparison Matrix](#4-architecture-comparison-matrix)
5. [Recommended Stack](#5-recommended-stack)
6. [Implementation Phases (Fases 1-3)](#6-implementation-phases)
7. [Trade-offs and Risk Analysis](#7-trade-offs-and-risk-analysis)
8. [Conclusion](#8-conclusion)

---

## 1. Executive Summary

After evaluating eight major memory paradigms for LLM-based agents, we recommend a **hybrid layered architecture** combining:

| Layer | Technology | Role |
|-------|-----------|------|
| **Working Memory** | In-context (LLM prompt window) | Active conversation state |
| **Episodic Memory** | LangGraph + SQLite/Postgres | Conversation history with summarization |
| **Semantic Memory** | Vector store (pgvector) + Contextual Retrieval | Long-term knowledge retrieval |
| **Procedural Memory** | LangGraph graphs + tool definitions | How to do things |
| **Compression Layer** | LLM-based summarization + MCP | Manage context windows |

**Primary stack:** PostgreSQL (pgvector) + LangGraph + Anthropic Claude API + MCP Tools.

**Why this stack:**
- PostgreSQL with pgvector eliminates a separate vector DB operational burden
- LangGraph provides first-class persistent state, checkpointing, and memory management
- MCP gives a standardized tool interface that decouples memory from agent logic
- Claude API provides long-context windows (200K tokens) and tool-use natively

---

## 2. Memory Architecture Landscape

Modern LLM agents require multiple memory types, mirroring human cognition:

```
                    ┌─────────────────────────────┐
                    │     WORKING MEMORY           │
                    │  (LLM context window)        │
                    │  ~200K tokens ephemeral       │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │     EPISODIC MEMORY          │
                    │  (Conversation history)      │
                    │  Structured logs + summary   │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼──────────┐ ┌──────▼───────┐ ┌──────────▼──────────┐
│  SEMANTIC MEMORY   │ │ PROCEDURAL   │ │   COMPRESSION       │
│  (Facts, knowledge)│ │ MEMORY       │ │   (Summarization)   │
│  Vector store +    │ │ (Graphs,     │ │   Sliding window    │
│  Hybrid search     │ │  tools)      │ │   + MCP             │
└────────────────────┘ └──────────────┘ └─────────────────────┘
```

### Key Design Dimensions

1. **Persistence:** Ephemeral (in-context) vs. Durable (DB-backed)
2. **Granularity:** Raw messages vs. Summarized/Compressed
3. **Retrieval:** Exact match vs. Semantic search vs. Hybrid
4. **Scope:** Per-thread vs. Per-session vs. Per-user vs. Global
5. **Freshness:** Recency-biased vs. Relevance-biased vs. Importance-biased

---

## 3. Deep Dive: Technologies

### 3.1 Advanced RAG

**Naive RAG** (index → retrieve → generate) fails on complex queries because retrieved chunks lack surrounding context. **Advanced RAG** introduces:

#### Pre-retrieval (Indexing)
- **Chunking strategies:** Semantic chunking (by topic boundaries, not fixed token count), sliding window overlap (15-25%), document hierarchy (section → paragraph → sentence)
- **Metadata enrichment:** Attach document titles, timestamps, summaries, entity references to each chunk
- **Multi-representation indexing:** Store both a brief summary (for retrieval) and full text (for generation) per chunk
- **HyDE (Hypothetical Document Embeddings):** Generate a synthetic perfect document from the query, then use its embedding for retrieval

#### Retrieval
- **Hybrid search:** Combine dense (vector) + sparse (BM25/keyword) retrieval using reciprocal rank fusion (RRF)
- **Multi-query expansion:** Generate 3-5 query variations from the user's question, retrieve for each, deduplicate
- **Hierarchical retrieval:** First retrieve documents, then relevant sections, then specific chunks within
- **Contextual retrieval (see §3.4):** Prepend chunk context before embedding

#### Post-retrieval
- **Re-ranking:** Cross-encoder models (Cohere Rerank, BGE Reranker) re-score top-K results
- **Context window management:** Dynamically pack relevant chunks within token budget, truncate least relevant
- **MMR (Maximum Marginal Relevance):** Diversify results to avoid redundancy

### 3.2 Episodic Memory

Episodic memory stores sequences of events/interactions as they occurred — the "what happened when" of agent experience.

#### Approaches

| Approach | Description | Best For |
|----------|-------------|----------|
| **Raw Log** | Full message history stored in DB | Debugging, full audit trail |
| **Rolling Summary** | Periodically summarize older messages, keep recent raw | Long-running conversations |
| **Reflection** | Agent periodically reviews past episodes to extract insights | Learning from experience |
| **Importance-weighted** | Score each episode by importance, retain high-score events | Agentic memory with limited storage |

#### Implementation Pattern (LangGraph)
```
                 ┌──────────────────┐
                 │  New Message      │
                 └────────┬─────────┘
                          ▼
              ┌──────────────────────┐
              │  Store in Episodic DB │
              │  (timestamp, content, │
              │   metadata, summary)  │
              └──────────┬───────────┘
                         ▼
         ┌─────────────────────────────┐
         │  Check: should we compress? │
         │  (every N messages or       │
         │   approaching context limit)│
         └──────────┬──────────────────┘
                    ▼
         ┌─────────────────────────────┐
         │  LLM summarizes oldest 25%  │
         │  Store summary, mark raw    │
         │  as archived                │
         └─────────────────────────────┘
```

**Key insight:** Use the LLM itself as the compression engine. Instruct it to preserve: decisions made, user preferences expressed, key facts established, unresolved items.

### 3.3 Vector Databases

#### Comparative Analysis

| Feature | pgvector | Pinecone | Qdrant | Weaviate | Chroma |
|---------|----------|----------|--------|----------|--------|
| **Type** | Extension | SaaS | Self-hosted/SaaS | Self-hosted/SaaS | Embedded |
| **Index** | IVFFlat, HNSW | HNSW | HNSW | HNSW | HNSW |
| **Filtering** | Postgres WHERE | Metadata filter | Payload filter | Where filter | Where filter |
| **Hybrid search** | pgvector + pg_bm25 | Not native | Built-in | Built-in | Not native |
| **Persistence** | Postgres | Managed | Disk/Managed | Disk/Managed | DuckDB/embedded |
| **Scaling** | Vertical | Horizontal | Horizontal | Horizontal | Single-node |
| **Operations** | Existing Postgres | API-only | Self-manage | Self-manage | Embedded |
| **Cost** | Free (existing DB) | Usage-based | Self-host: free | Self-host: free | Free |
| **Maturity** | Mature | Mature | Mature | Mature | Young |
| **LangChain support** | Yes | Yes | Yes | Yes | Yes |

#### Recommendation: pgvector

**Rationale:**
- Eliminates a separate infrastructure service (fewer moving parts)
- Transactional consistency with conversational data in the same database
- HNSW index provides good performance up to millions of vectors
- pg_bm25 extension enables hybrid search without leaving Postgres
- Operational simplicity: one database to back up, monitor, and tune
- Mature and well-supported in all major agent frameworks

**When to reconsider:** At >10M vectors or >1000 queries/second, consider Qdrant or Pinecone for horizontal scaling.

### 3.4 Contextual Retrieval

Anthropic's **Contextual Retrieval** technique addresses the fundamental weakness of naive chunk-based retrieval: a standalone chunk often lacks the context needed to determine relevance.

#### The Problem
When you chunk a document, each chunk loses its surrounding context. Example:
- *"The company reported $12B in revenue"* — which company? what year?
- *"Step 3: Apply the transform"* — step 3 of what?

#### The Solution
Before embedding each chunk, prepend a short LLM-generated context:
```
<chunk-context>
Document: {document_title}
Section: {section_heading}
Summary: {1-2 sentence summary of surrounding content}
Previous: {last sentence before this chunk}
</chunk-context>
{original chunk content}
```

#### Key Implementation Details

1. **Context generation:** Use a fast LLM to generate context for each chunk (one-shot prompt with document)
2. **Context caching:** Cache the generated context strings so only new/changed documents incur the cost
3. **Hybrid matching:** Apply contextual chunks to both dense (vector) and sparse (BM25) retrieval
4. **Results:** Anthropic reported 49% improvement in retrieval recall (top-20) over naive chunking

#### Practical Benefits
- Drastically reduces "false negatives" — relevant chunks that fail to match because they lack context
- No re-ranking step needed for moderate quality requirements
- Simple to implement: one additional pre-processing pipeline step

### 3.5 LangGraph Memory

LangGraph provides the most mature agentic memory framework available today, with three integrated levels:

#### Level 1: Thread-scoped Memory (Short-term)
- Built-in checkpointing of every step in a graph execution
- State is persisted per thread (conversation)
- Enables pause/resume, human-in-the-loop, and error recovery
- **Storage:** SQLite (default) or Postgres checkpointer

#### Level 2: Cross-thread Memory (Long-term)
- **Store** API: A shared key-value store across all threads
- Each entry has a namespace (`(user_id, conversation_type)`) and key
- Stores: user preferences, facts learned, accumulated knowledge
- **Storage:** In-memory (dev), file-based, or custom backend

#### Level 3: Episodic Memory via Summarization
- Built-in `MessagesState` with configurable summarization
- Triggers: token count threshold, message count, or custom condition
- Summarizer runs as a node in the graph, updates running summary
- Summary injected into system prompt on subsequent runs

#### Code Architecture (Phase 1-3)

```
┌─────────────────────────────────────────────┐
│                 LangGraph                    │
│                                              │
│  ┌─────────────┐    ┌──────────────────┐    │
│  │ StateGraph   │    │  Checkpointer    │    │
│  │ (agent flow) │◄───│  (Postgres)      │    │
│  └──────┬──────┘    └──────────────────┘    │
│         │                                    │
│  ┌──────▼──────┐    ┌──────────────────┐    │
│  │  Store      │    │  Summarizer Node │    │
│  │ (cross-thrd)│    │  (auto-compress) │    │
│  └──────┬──────┘    └──────────────────┘    │
│         │                                    │
└─────────┼────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐
│  MCP Memory Server   │
│  (episodic + semantic)│
└──────────────────────┘
```

### 3.6 Hybrid Memory Systems

No single memory type suffices for production agents. The winning pattern is a **layered/hierarchical hybrid**.

#### The Three-Tier Model

| Tier | Name | Technology | Latency | Capacity | Retention |
|------|------|-----------|---------|----------|-----------|
| L1 | Working Memory | LLM Context Window | <1ms | ~200K tokens | Session |
| L2 | Episodic Buffer | LangGraph + Postgres | ~10ms | Recent history | Days |
| L3 | Semantic Store | pgvector + BM25 | ~100ms | Long-term facts | Permanent |

#### Information Flow

1. **Write path:** Every agent action flows into L2 (episodic buffer). Periodic background compression extracts semantic facts into L3. Working memory (L1) is the LLM context window — populated from L2/L3 at conversation start and on summarization triggers.

2. **Read path:** At each turn, the agent has L1 active. When it needs knowledge beyond L1, it queries L3 semantically. L2 provides recency-biased context for coherent conversation flow.

3. **Maintenance path:** L2 summaries roll up as the conversation grows. L3 runs deduplication and importance-scoring in background. L1 is managed by the LLM's own attention mechanism.

#### Reflexion Pattern (Advanced)

A particularly effective hybrid pattern for agentic tasks:

```
Task → Act → Observe → Reflect → (loop)
                              ↓
                        Memory Store
                        (episodic + semantic)
```

The agent acts, observes outcomes, then reflects on what worked, storing the lesson. This enables continuous improvement without retraining.

### 3.7 Context Compression

Context compression is essential for managing the cost-quality-latency triangle in agents with long histories.

#### Techniques

| Technique | Compression Ratio | Quality Impact | Implementation Complexity |
|-----------|------------------|----------------|--------------------------|
| **LLM Summarization** | 10-50x | Medium (loses detail) | Low (one prompt) |
| **Token pruning** | 2-5x | Low (removes filler) | Medium (heuristic + LLM) |
| **Sliding window** | Variable | Low (drops oldest) | Low (configurable) |
| **Semantic compression** | 5-20x | Medium (tone/emotion lost) | High (specialized model) |
| **KV-cache reuse** | 1x (no new tokens) | None | Very High (infra-level) |
| **MCP resource compression** | 2-10x | Low | Medium (server-level) |

#### Recommended Approach: Hybrid Two-Stage

**Stage 1 — Lossless pruning** (every turn):
- Remove tool call/result pairs that failed or produced empty results
- Truncate verbose system messages to relevant directives
- Remove duplicate or superseded information

**Stage 2 — Lossy compression** (when approaching context limit):
- LLM summarizes the oldest 30-50% of conversation
- Preserve: decisions, preferences, key facts, unresolved todos
- The summary is stored in episodic memory AND injected into system prompt
- Raw messages are archived but not included in context

### 3.8 Long-Term Memory for Agents

Long-term memory is the hardest open problem in agent architecture. Current approaches:

#### Category A: Explicit Storage (Production-ready)

| Approach | Mechanism | Example |
|----------|-----------|---------|
| **Fact extraction** | LLM extracts facts each turn → stored in vector DB | MemGPT/Letta |
| **Preference learning** | User corrections → stored as rules | LangGraph Store |
| **Knowledge graphs** | Entities + relationships extracted → graph DB | WhyHow.AI, Neo4j |
| **Summarization** | Periodic compression of all past interactions | LangGraph summarizer |

#### Category B: Implicit / Learned (Emerging)

- **Fine-tuning on interactions:** Update model weights periodically with distilled conversations (expensive, slow)
- **In-context learning:** Retrieve examples similar to current query (few-shot from history)
- **Hypernetwork / Write-read memory:** External memory module trained alongside LLM (research-stage)

#### Practical Recommendation for Phases 1-3

**Phase 1:** Start with structured episodic memory (LangGraph checkpointer) + basic fact extraction. Store facts in a simple pgvector collection. No compression yet.

**Phase 2:** Add summarization pipeline + sliding window for long conversations. Introduce knowledge graph for entity relationships.

**Phase 3:** Full hybrid system with automatic tier promotion, importance scoring, and background maintenance jobs.

**Do NOT attempt in Phases 1-3:**
- Fine-tuning from interactions
- Learned memory modules
- Distributed agent memory sharing

---

## 4. Architecture Comparison Matrix

| Criterion | Basic RAG | Advanced RAG | LangGraph Memory | Hybrid (Recommended) |
|-----------|-----------|-------------|-----------------|---------------------|
| **Setup complexity** | Low | Medium | Medium | Medium-High |
| **Recall quality** | Moderate | High | High | Very High |
| **Context freshness** | Stale (index time) | Semi-fresh | Fresh | Fresh |
| **Cross-session memory** | No | No | Yes (Store API) | Yes |
| **Operational cost** | Low | Medium | Low-Medium | Medium |
| **Scaling** | Good | Good | Moderate | Good |
| **Agent integration** | Manual | Manual | Native | Via MCP |
| **Maintenance burden** | Low | Medium | Low | Medium |
| **Flexibility** | Low | Medium | High | Very High |
| **Production readiness** | Proven | Proven | Proven (LangChain) | Proven (components) |

---

## 5. Recommended Stack

### Tier 1 (Phase 1 — Core)

| Component | Technology | Justification |
|-----------|-----------|---------------|
| LLM | Claude API (Sonnet) | Long context, tool use, reliable |
| Agent framework | LangGraph | Best-in-class state management, checkpointing, Store API |
| Vector storage | pgvector (PostgreSQL) | Eliminates separate vector DB, transactional consistency |
| Full-text search | pg_bm25 (via ParadeDB) | Hybrid search with pgvector in same database |
| Embeddings | Voyage AI or OpenAI text-embedding-3-large | State-of-the-art retrieval embeddings |

### Tier 2 (Phase 2 — Added)

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Episodic memory | LangGraph checkpointer (Postgres) | Built-in, production-grade |
| Cross-thread memory | LangGraph Store API | Structured key-value across conversations |
| Re-ranker | Cohere Rerank 3 or BGE-M3 | Cross-encoder improves retrieval precision |
| Compression | LangGraph summarizer node | Automated, configurable |

### Tier 3 (Phase 3 — Scale)

| Component | Technology | Justification |
|-----------|-----------|---------------|
| MCP memory server | Custom (Python/TypeScript) | Standardized tool interface, decoupled |
| Knowledge graph | WhyHow.AI or custom Neo4j | Entity relationship storage |
| Background jobs | Celery / LangGraph cron | Periodic summarization, maintenance |
| Monitoring | LangSmith / Langfuse | Trace agent memory usage |

### Infrastructure

```
┌─────────────────────────────────────────────┐
│                  Application                 │
├─────────────────────────────────────────────┤
│  LangGraph Agent (Python)                   │
│  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Claude  │ │ Memory   │ │ MCP Tools   │  │
│  │ API     │ │ Server   │ │ (extensions)│  │
│  └────┬────┘ └────┬─────┘ └──────┬──────┘  │
├───────┼───────────┼──────────────┼──────────┤
│       │           │              │           │
│  ┌────▼───────────▼──────────────▼──────┐   │
│  │         PostgreSQL 16                │   │
│  │  ┌─────────┐ ┌────────┐ ┌────────┐  │   │
│  │  │ Chat    │ │Vector  │ │ Store  │  │   │
│  │  │ History │ │(pgvec) │ │ (JSONB)│  │   │
│  │  └─────────┘ └────────┘ └────────┘  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 6. Implementation Phases (Fases 1-3)

### Phase 1: Foundation (Weeks 1-3)

**Goal:** Functional agent with basic memory — can persist and retrieve conversation history.

**Deliverables:**
- [ ] LangGraph agent skeleton with state management
- [ ] PostgreSQL + pgvector setup
- [ ] Claude API integration with tool use
- [ ] Basic conversation persistence (store/load threads)
- [ ] Simple semantic retrieval: embed user messages, retrieve relevant past context
- [ ] MCP server with `get_context` and `store_memory` tools

**Key decisions:**
- Use LangGraph's built-in Postgres checkpointer for thread-scoped memory
- Store all messages as structured records (not raw text) for future queryability
- Embeddings: Voyage AI (or OpenAI text-embedding-3-large), 1024-dim or 1536-dim
- Start with `IVFFlat` index (faster build), migrate to `HNSW` in Phase 2 if needed

**Memory schema (Postgres):**

```sql
-- Thread-scoped conversation history
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'tool', 'system'
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON conversation_messages (thread_id, created_at);
CREATE INDEX ON conversation_messages USING hnsw (embedding vector_cosine_ops);

-- Cross-thread semantic memory
CREATE TABLE semantic_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    importance REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, key)
);
```

### Phase 2: Memory Intelligence (Weeks 4-6)

**Goal:** Intelligent memory — summarization, reflection, hybrid retrieval, and context management.

**Deliverables:**
- [ ] Automatic conversation summarization (triggered every N messages)
- [ ] Sliding window context management
- [ ] Hybrid search (dense + sparse/BM25 via pg_bm25)
- [ ] Re-ranking for retrieval precision
- [ ] Cross-thread memory via LangGraph Store API
- [ ] Importance scoring for semantic memory entries
- [ ] MCP memory server with summarization endpoint

**Architecture additions:**

```
Agent Turn
  │
  ├─► Compress context (if > threshold)
  │     └─► LLM summarizes oldest 30% → store summary → remove raw
  │
  ├─► Retrieve relevant memories
  │     ├─► Semantic search (pgvector, top-20)
  │     ├─► Keyword search (pg_bm25, top-20)
  │     ├─► RRF fusion → top-10
  │     └─► Cross-encoder re-rank → top-5
  │
  └─► Extract new facts
        └─► LLM: "What facts did we learn?" → store in semantic_memory
```

### Phase 3: Scale & Polish (Weeks 7-10)

**Goal:** Production-grade memory — knowledge graphs, background maintenance, full MCP tool interface, and observability.

**Deliverables:**
- [ ] Knowledge graph extraction (entities + relationships from conversations)
- [ ] Background maintenance jobs (deduplication, importance decay, stale cleanup)
- [ ] Full MCP memory tool suite
- [ ] Memory observability dashboard (LangSmith/Langfuse)
- [ ] Cache warming for frequent queries
- [ ] Memory export/import for user data portability
- [ ] Performance optimization (connection pooling, query tuning, index maintenance)

---

## 7. Trade-offs and Risk Analysis

### Critical Trade-offs

| Decision | Option A | Option B | Our Choice | Why |
|----------|----------|----------|-----------|-----|
| Vector DB | Dedicated (Pinecone) | pgvector | pgvector | Stage-appropriate simplicity |
| Agent framework | Custom | LangGraph | LangGraph | Maturity + built-in memory |
| Embedding model | OpenAI | Voyage AI | Voyage AI (team pref) | Comparable quality, lower cost |
| Compression | Sliding window | LLM summary | Both hybrid | Best quality/cost ratio |
| Memory server | Built-in | MCP | MCP | Decoupling for future flexibility |
| Knowledge graph | Phase 2 | Phase 3 | Phase 3 | Stage-appropriate complexity |

### Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| pgvector performance at scale | Medium | High | Monitor query latency; have Qdrant as upgrade path |
| LLM summarization loses critical details | Medium | High | Always archive raw messages; summary is a cache, not source of truth |
| Context window grows unbounded | Medium | Medium | Strict budget enforcement; hard cap with oldest-first eviction |
| MCP tool latency adds overhead | Low | Medium | Cache frequently used memory; batch retrieve at conversation start |
| Embedding drift (model changes) | Low | Medium | Version embeddings; re-index on model change |
| Data portability compliance | Low | Medium | Design export/import from day one; use standard formats (JSONL) |

### Cost Projection (Monthly, at scale)

| Component | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Claude API (Sonnet) | $200 | $400 | $800 |
| PostgreSQL (RDS) | $50 | $50 | $100 |
| Embedding API | $20 | $40 | $60 |
| Re-ranker API | — | $50 | $100 |
| Total | ~$270 | ~$540 | ~$1,060 |

---

## 8. Conclusion

The recommended **hybrid layered architecture** with PostgreSQL/pgvector + LangGraph + Claude API + MCP provides:

- **Stage-appropriate complexity:** Start simple (Phase 1), add sophistication as needed (Phase 2-3)
- **Operational simplicity:** One database, one agent framework, clean separation via MCP
- **Production readiness:** Every component is battle-tested at scale
- **Extensibility:** MCP interface means memory can be swapped, upgraded, or distributed without touching agent logic
- **Observability:** LangGraph's built-in tracing plus optional LangSmith integration

**The MemoryEngineer should begin with Phase 1 immediately.** The Phase 1 deliverables define the skeleton that all future memory work builds upon. Phase 2 and 3 are additive — they enhance rather than replace.

### Immediate Next Steps

1. Set up PostgreSQL 16 + pgvector extension
2. Scaffold LangGraph agent with Postgres checkpointer
3. Implement basic MCP memory server with `store_message` and `search_memories` tools
4. Integrate Claude API with tool use
5. Write integration tests for the full memory cycle: store → retrieve → respond → store
6. Deploy to staging and run with real conversation traffic for 1 week before advancing to Phase 2

---

*This report was produced as deliverable for THI-35. It bases the implementation of Phases 1-3 of the memory-chat-bot project. All technology choices should be re-evaluated every 6 months against current ecosystem state.*
