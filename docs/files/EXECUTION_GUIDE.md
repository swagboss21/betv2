# EXECUTION GUIDE: Using This with Claude Code

## Quick Start

### Option 1: Execute One Task at a Time
```bash
# Start with Sprint 0, Task 1
claude "Read execution_plan/sprints/SPRINT_0.md and execute task S0-T1. Output to db/schema.sql"

# Then task 2
claude "Execute task S0-T2 from SPRINT_0.md"

# And so on...
```

### Option 2: Execute Full Sprint
```bash
# Run all of Sprint 0
claude "Execute all tasks in execution_plan/sprints/SPRINT_0.md in sequence. Verify each task passes before proceeding to the next."
```

### Option 3: Parallel Execution (Advanced)
Tasks marked "Parallel: Yes" can run simultaneously:
```bash
# Sprint 0 parallel tasks
claude "Execute S0-T3, S0-T4, S0-T5 in parallel from SPRINT_0.md"
```

---

## The MAKER Pattern Applied

Each task in this plan follows the MAKER paper principles:

### 1. Maximal Agentic Decomposition (MAD)
- Each task is ONE focused unit of work
- One sub-agent = one output file
- No task depends on more than 1-2 other tasks

### 2. Verification (Error Correction)
Every task has a verification step:
```bash
# Run the verify script
./verify_s0t1.sh

# If ❌ FAIL: Re-run the task with error context
claude "Task S0-T1 failed with error: [paste error]. Fix and retry."
```

### 3. Red-Flagging
If a sub-agent produces:
- Incorrect file format → Re-run
- Missing required functions → Re-run
- Test failures → Re-run with error context

---

## Critical Path (Minimum Viable Product)

To get to MVP fastest:
```
Sprint 0 (2-3 days) → Sprint 1 (3-4 days) → Sprint 2 (2-3 days) → Sprint 3 (3-4 days)
```

Total: ~2 weeks to MVP

Sprints 4, 5, 6 are enhancements that can wait.

---

## Human Checkpoints

These require YOUR approval before proceeding:

| Sprint | Checkpoint | What to Review |
|--------|------------|----------------|
| 0 | After S0-T1 | Database schema looks right? |
| 1 | After S1-T5 | Projections populated correctly? |
| 3 | After S3-T3 | System prompt sounds right? |
| 3 | After S3-T5 | Full flow works end-to-end? |
| 6 | Before S6-T1 | Pricing decision confirmed? |

---

## Folder Structure After All Sprints

```
the-brain/
├── execution_plan/           # This plan
│   ├── MASTER_EXECUTION_PLAN.md
│   └── sprints/
│       ├── SPRINT_0.md
│       ├── SPRINT_1.md
│       ├── SPRINT_2.md
│       ├── SPRINT_3.md
│       └── SPRINTS_4_5_6.md
├── brain.db                  # SQLite database
├── db/
│   ├── schema.sql
│   └── init_db.py
├── batch/
│   ├── precompute.py
│   ├── scrape_injuries.py
│   └── grade_bets.py
├── api/
│   ├── queries.py
│   └── probability.py
├── brain_mcp/
│   └── server.py
├── bots/
│   ├── strategies.py
│   └── runner.py
├── auth/
│   └── oauth.py
├── simulation/               # EXISTING (don't touch)
├── models/                   # EXISTING (don't touch)
└── tests/
    ├── test_sprint0.py
    ├── test_sprint1.py
    └── test_sprint2.py
```

---

## Troubleshooting

### "No games found today"
- Run `python batch/precompute.py` first
- Check if NBA season is active

### "No projection for player"
- Player may not be playing tonight
- Check spelling matches exactly
- Verify precompute ran successfully

### "Tool execution failed"
- Check database exists: `sqlite3 brain.db ".tables"`
- Check API modules import: `python -c "from api import queries"`

### "Anthropic API error"
- Verify `ANTHROPIC_API_KEY` is set
- Check you have API credits

---

## Cost Estimates

| Sprint | Primary Cost |
|--------|--------------|
| 0-2 | Free (local dev) |
| 3+ | ~$0.001 per chat message (Haiku) |
| Production | ~$30/month for 1000 daily users |

---

## Next Steps

1. Copy this execution_plan folder to your project root
2. Start with Sprint 0, Task 1
3. Work through systematically
4. Get to Sprint 3 MVP
5. Test with 5 friends
6. Iterate based on feedback
