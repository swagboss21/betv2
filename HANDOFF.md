# The Brain - Handoff Context

**Use this to onboard a new Claude session or continue development.**

---

## Quick Summary

NBA betting co-pilot with working Monte Carlo simulation + LLM integration.
**Core system COMPLETE.** Ready to discuss productization.

---

## What's Built

| Component | Status | Description |
|-----------|--------|-------------|
| Monte Carlo Engine | ✅ Working | Simulates 10K games, outputs probability distributions |
| 3 Trained Models | ✅ Working | Game scores, player minutes, player stats (7 types) |
| LLM Integration | ✅ Working | Claude Sonnet 4 with 7 tools via function calling |
| Parlay Analyzer | ✅ Working | Detects correlated bets, generates "thesis" explanations |

---

## What Works (Tested Dec 5, 2024)

```
User: "Should I bet Tatum over 27.5 points tonight?"
→ Checks injuries for both teams
→ Runs 10K Monte Carlo simulations
→ Returns: "My model has Tatum at 23.1 pts, only 29% chance of over.
   I'd pass or take the under."
```

**Verified features:**
- ✅ Prop simulation with edge calculation
- ✅ Tonight's games via nba_api
- ✅ Injury checking (ESPN scraper + manual override)
- ✅ Parlay correlation detection (shootout thesis, etc.)
- ✅ Natural language explanations

---

## Current Cost

| Metric | Value |
|--------|-------|
| Per query (Sonnet 4) | $0.02 |
| Per query (with caching) | $0.008 |
| Your $5 free credits | ~250 queries |
| Break-even at $20/user | 1000 queries/month |

---

## Open Questions to Explore

### Business Model
- Charge per BET delivered (not per query)?
- $15-20/month for 10-25 researched bets?
- How to define a "bet" vs a "query"?

### Product Direction
- Target user wants RISKY PARLAYS (entangled, high upside)
- Need bet tracking + outcome memory
- Need to optimize research flow (generate bet in <10 queries)

### Technical Decisions Made
- Interface: MCP Server (Claude Desktop integration) ✅
- Database: PostgreSQL ✅
- Auth: TBD (pending SaaS phase)
- Hosting: TBD (pending SaaS phase)

---

## To Test Locally

**Via Claude Desktop (Recommended):**
1. Add `.mcp.json` config to Claude Desktop settings
2. Restart Claude Desktop
3. Tools available: `get_games_today`, `get_projection`, `get_best_props`, `get_injuries`, `lock_bet`

**Direct MCP test:**
```bash
cd "/Users/noahcantu/Desktop/the-brain-organized 2"
python -m brain_mcp.server
```

Example prompts:
- "What games are on tonight?"
- "Should I bet LeBron over 25.5 points vs Boston?"
- "Check Lakers injuries"
- "Best props for tonight?"

---

## Key Files

| File | Purpose |
|------|---------|
| `brain_mcp/server.py` | MCP server (5 tools) |
| `api/queries.py` | Database query functions |
| `simulation/engine.py` | Monte Carlo simulation |
| `models/*.pkl` | Trained XGBoost models |
| `CLAUDE.md` | Project documentation |

---

## What NOT to Do

- Don't rebuild the simulation engine (it works)
- Don't switch LLM providers without testing
- Don't add features before validating core loop with real users
- Don't make the LLM do math (engine does math, LLM explains)

---

## GitHub

https://github.com/swagboss21/betv2.git

---

## Suggested Next Steps

1. **Validate with users** - Have 3-5 people test via Claude Desktop, gather feedback
2. **Add live odds** - Integrate SportsGameOdds API for real lines
3. **Improve data pipeline** - Automate batch jobs, add outcome tracking
4. **Add bet memory** - Track locked bets and results
5. **Build web API** - REST/GraphQL for future web/mobile clients
