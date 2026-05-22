# THI-21: Backend Memory Systems Engineer — Hire Request (Ready to Submit)

## Status
AGENTS.md drafted, all hire parameters defined. **Blocked on:** Paperclip API server migration issue (`relation "agent_runtime_state" already exists`).

## Hire Configuration

```json
{
  "name": "BackendMemorySystemsEngineer",
  "role": "engineer",
  "title": "Backend Memory Systems Engineer",
  "icon": "database",
  "reportsTo": "f1a8ab46-9337-41c5-ae71-6049f9e485b3",
  "capabilities": "Designs and implements backend memory infrastructure for LLM-based conversational systems: memory APIs (FastAPI), vector databases (pgvector, Qdrant, FAISS), hybrid retrieval, context compression, episodic memory, multi-agent communication (RabbitMQ, MCP), and cognitive persistence.",
  "desiredSkills": [],
  "adapterType": "claude_local",
  "adapterConfig": {
    "cwd": "C:\\Users\\thiag\\.paperclip\\instances\\default\\projects\\cb49ee99-6bba-402c-a2da-3df52294f8ed\\f3651b84-292a-4c3b-92fd-5a7bdff3de80\\_default",
    "model": "claude-sonnet-4-6"
  },
  "instructionsBundle": {
    "files": {
      "AGENTS.md": "<see memory-engineer-AGENTS.md in project root>"
    }
  },
  "runtimeConfig": {
    "heartbeat": {
      "enabled": false,
      "wakeOnDemand": true
    }
  },
  "sourceIssueId": "723bff84-7bb3-49be-9692-734e71a14f66"
}
```

## Template Path
**Adjacent template** — adapted from `coder.md` with:
- Charter rewritten for backend memory systems engineering
- Domain lenses swapped to memory/backend-specific (10 lenses: Memory Hierarchy Fit, Hybrid Retrieval, Context Window Economics, Persistence Durability, Recall Precision vs Recall, Temporal Relevance Decay, Modularity Over Monolith, Open Source First, Observability-Driven Development, Stage-Appropriate Complexity)
- Company context (human memory simulation project) + tech stack added
- Collaboration/handoffs updated for CTO reporting line

## Server Issue
`npx paperclipai run` fails with: `relation "agent_runtime_state" already exists`
- Cause: Drizzle migrations tracking is out of sync with actual database state
- 79 migrations listed as pending, but tables already exist from a prior successful run
- CLI `doctor --repair` passes but doesn't resolve migration state
- No `--skip-migrations` or `--force` flag available on `run`
- This likely needs: resetting the migrations tracking table, or reseeding the database

## Next Steps
1. Fix server migration issue (CTO or board action)
2. Verify `llms/agent-icons.txt` for exact icon name
3. Verify `llms/agent-configuration/claude_local.txt` for exact adapter config
4. Run the draft-review checklist against AGENTS.md
5. Submit POST /api/companies/{id}/agent-hires with the payload above

## Files
- Draft AGENTS.md: `./memory-engineer-AGENTS.md` (in project root)
