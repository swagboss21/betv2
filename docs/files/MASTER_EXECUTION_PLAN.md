# THE BRAIN - MASTER EXECUTION PLAN

*Applying MAKER Framework to Sprint Decomposition*

---

## Core Principle: Maximal Agentic Decomposition (MAD)

From the MAKER paper: "Tasks should be broken up into the smallest possible elements, so that an LLM agent can focus on them one step at a time."

**Translation for Claude Code:**
- Each task = one sub-agent
- Each sub-agent produces ONE artifact (file, script, migration)
- Each artifact has verification criteria
- Human approves at sprint boundaries, not task boundaries

---

## Execution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│  (You, via Claude Code)                                      │
│  - Spawns sub-agents for each Task Card                     │
│  - Collects outputs, runs verification                      │
│  - Gates progression to next sprint                         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Sub-Agent A │    │  Sub-Agent B │    │  Sub-Agent C │
│  Task: DB    │    │  Task: API   │    │  Task: Test  │
│  Schema      │    │  Function    │    │  Script      │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
   schema.sql          api.py             test_api.py
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                    VERIFICATION
                    (tests pass?)
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
                   ✅              ❌
               Continue        Red-flag,
                              retry with
                              different
                              approach
```

---

## Sprint Decomposition Summary

| Sprint | Tasks | Parallelizable | Human Checkpoints |
|--------|-------|----------------|-------------------|
| 0: Foundation | 6 | 4 parallel, 2 sequential | 1 (approve schema) |
| 1: Factory | 5 | 2 parallel, 3 sequential | 1 (approve pipeline) |
| 2: Warehouse | 6 | 5 parallel, 1 sequential | 1 (approve API) |
| 3: Chat MVP | 5 | 3 parallel, 2 sequential | 2 (approve UI + tools) |
| 4: Results | 4 | 3 parallel, 1 sequential | 1 (approve grading) |
| 5: Bots | 4 | 2 parallel, 2 sequential | 1 (approve bot logic) |
| 6: Auth | 5 | 3 parallel, 2 sequential | 2 (OAuth + paywall) |

**Total: 35 tasks, ~20 can run in parallel batches**

---

## How to Execute with Claude Code

### Command Pattern
```bash
# Spawn a sub-agent for a specific task
claude "Execute task SPRINT0-T1 from execution_plan/sprints/SPRINT_0.md"

# Or run a whole sprint (orchestrator mode)
claude "Execute all tasks in SPRINT_0.md in order, verify each before proceeding"
```

### Verification Pattern (Red-Flagging)
Each task includes a `verify.sh` or `verify.py` that:
1. Checks if output file exists
2. Runs syntax validation
3. Runs unit tests (if applicable)
4. Outputs ✅ PASS or ❌ FAIL with reason

If FAIL: task gets re-attempted with error context (MAKER's "error correction")

---

## File Structure After Execution

```
the-brain/
├── brain.db                    # SQLite database (Sprint 0)
├── db/
│   ├── schema.sql              # Sprint 0
│   └── migrations/             # Future migrations
├── batch/
│   ├── precompute.py           # Sprint 1 - nightly job
│   ├── scrape_injuries.py      # Sprint 1
│   └── grade_bets.py           # Sprint 4
├── api/
│   ├── queries.py              # Sprint 2 - warehouse layer
│   └── probability.py          # Sprint 2 - math helpers
├── brain_mcp/
│   └── server.py               # Sprint 3 - MCP server (5 tools)
├── bots/
│   ├── strategies.py           # Sprint 5
│   └── runner.py               # Sprint 5
├── auth/
│   └── oauth.py                # Sprint 6
├── simulation/                 # EXISTING - don't touch
│   ├── engine.py
│   ├── models.py
│   └── ...
├── models/                     # EXISTING - don't touch
│   ├── game_model.pkl
│   └── ...
└── tests/
    ├── test_db.py              # Generated per sprint
    ├── test_api.py
    └── ...
```

---

## Dependencies & Sequencing

```
SPRINT 0 ──► SPRINT 1 ──► SPRINT 2 ──► SPRINT 3 ──► SPRINT 4
   │                                       │
   │                                       └──► SPRINT 5 ──► SPRINT 6
   │
   └── [Can start Sprint 1 immediately after Schema approved]
```

**Critical Path:** 0 → 1 → 2 → 3 (Core loop)
**Optional Path:** 4, 5, 6 (Enhancement)

---

## Next: Individual Sprint PRDs

See the `/sprints/` folder for detailed Task Cards:
- `SPRINT_0.md` - Foundation (DB + Structure)
- `SPRINT_1.md` - Factory (Batch Precompute)
- `SPRINT_2.md` - Warehouse (Query Functions)
- `SPRINT_3.md` - Chat MVP
- `SPRINT_4.md` - Results Grading
- `SPRINT_5.md` - House Bots
- `SPRINT_6.md` - Authentication
