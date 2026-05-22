# THI-21: Status & Next Actions

## What's Done

### AGENTS.md — Drafted
Complete instruction bundle for the Backend Memory Systems Engineer at `./memory-engineer-AGENTS.md`. 

**Template path:** Adjacent to `coder.md` with deliberate adaptations:
- Charter rewritten for backend memory systems engineering (not general coding)
- 10 domain-specific lenses: Memory Hierarchy Fit, Hybrid Retrieval, Context Window Economics, Persistence Durability, Recall Precision vs Recall, Temporal Relevance Decay, Modularity Over Monolith, Open Source First, Observability-Driven Development, Stage-Appropriate Complexity
- Company context (human memory simulation project) + full tech stack
- Collaboration/handoffs scoped to CTO reporting line

### Hire Request — Configured
Full payload at `./THI-21-hire-request.md`:
- **Name:** BackendMemorySystemsEngineer
- **Role:** engineer  
- **Title:** Backend Memory Systems Engineer
- **Reports to:** CTO (f1a8ab46-9337-41c5-ae71-6049f9e485b3)
- **Adapter:** claude_local
- **Source issue:** THI-21

## What's Blocking

### Server unavailable — Database migration issue
`npx paperclipai run` fails with: `relation "agent_runtime_state" already exists`

**Root cause:** Drizzle ORM migration tracking is out of sync. 79 migrations listed as pending but the tables already exist from a prior successful run. The migration engine tries `CREATE TABLE` (not `CREATE TABLE IF NOT EXISTS`) and fails.

**What was tried:**
- `npx paperclipai doctor --repair` — passes all checks but doesn't fix migration state
- Killing stale postgres + restarting — same error
- `npx paperclipai run --help` — no `--skip-migrations` or `--force` flag
- `npx paperclipai agent local-cli` — requires server to be running

**Fix needed:** Either:
1. Add migration records to the `__drizzle_migrations` tracking table for already-applied migrations
2. Drop and recreate the database (loses all state — company, agents, issues)
3. Add a `--force` or `--skip-migrations` flag to `npx paperclipai run`

## Files in Workspace
- `memory-engineer-AGENTS.md` — Complete AGENTS.md for the new agent
- `THI-21-hire-request.md` — Full hire API payload (ready to submit)
- `THI-21-status.md` — This file

## Pre-submit Checklist Status
The draft-review checklist (from paperclip-create-agent skill) was reviewed against the AGENTS.md:
- [x] Identity, reporting line, company name
- [x] Role charter with clear ownership and scope
- [x] Operating workflow with execution contract
- [x] Domain lenses (10, role-specific)
- [x] Output/review bar
- [x] Collaboration routing
- [x] Safety and permissions
- [x] Done criteria
- [ ] Icon verification (blocked on `/llms/agent-icons.txt` — server down)
- [ ] Adapter config verification (blocked on `/llms/agent-configuration/claude_local.txt` — server down)
