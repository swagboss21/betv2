# The Brain - NBA Betting Co-Pilot

## What This Is
AI-powered NBA player prop betting co-pilot using Monte Carlo simulation + LLM reasoning.

---

## Current State: Sprint 3 COMPLETE ✅

All core components built and tested:

| Sprint | Status | What's Done |
|--------|--------|-------------|
| 0. Database | ✅ COMPLETE | PostgreSQL schema, 6 tables |
| 1. Batch Pipeline | ✅ COMPLETE | Monte Carlo precompute + ESPN scraper |
| 2. API Layer | ✅ COMPLETE | 13 query functions, 30/30 tests passing |
| 3. MCP Server | ✅ COMPLETE | Claude Desktop integration via MCP protocol |

---

## Architecture

```
User Query → LLM (Claude) → Tools → Monte Carlo Engine → Response
```

### Models (trained, in `models/`)
- `game_model.pkl` - Predicts team scores (1.6MB)
- `minutes_model.pkl` - Predicts player minutes (514KB)
- `stats_models.pkl` - Predicts 7 stats: pts/reb/ast/stl/blk/tov/fg3m (6.2MB)

### Simulation Engine (`simulation/`)
- `engine.py` - MonteCarloEngine class, 10K simulations
- `feature_transformer.py` - Fetch live player/team data from nba_api
- `injury_adjustment.py` - Boost remaining players when someone OUT
- `edge_calculator.py` - Convert odds, calculate betting edge
- `parlay_analyzer.py` - Detect correlated legs, generate thesis
- `models.py` - Data classes for inputs/outputs

### MCP Server (`brain_mcp/`)
- `server.py` - MCP server exposing 5 tools to Claude Desktop
- `.mcp.json` - Claude Desktop configuration

Tools available:
- `get_games_today` - Tonight's NBA schedule
- `get_projection` - Player stat projection with probability
- `get_best_props` - Top edge props ranked
- `get_injuries` - Team injury report
- `lock_bet` - Save user's bet

---

## Using MCP Server

The Brain integrates with Claude Desktop via MCP (Model Context Protocol).

**Setup:**
1. Copy `.mcp.json` contents to Claude Desktop config
2. Restart Claude Desktop
3. Tools are automatically available in conversation

**Manual test:**
```bash
python -m brain_mcp.server
```

API keys loaded from `.env` automatically.

---

## Cost

| Optimization | Cost/Query | Monthly (50/day) |
|--------------|------------|------------------|
| None (current) | $0.02 | $30 |
| Prompt caching | $0.008 | $12 |

---

## What's Next (Phase 5+)

Building toward SaaS product:
- [ ] User accounts + auth
- [ ] Bet tracking + memory
- [ ] Outcome tracking (did it hit?)
- [ ] Web interface
- [ ] Usage metering (bets, not queries)

---

## Locked Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Odds API | SportsGameOdds v2 | Per-event pricing |
| LLM | Claude Sonnet 4 | Best reasoning for cost |
| Simulation | 10K Monte Carlo | Distribution, not point estimate |
| Data | 3 seasons (2022-25) | NBA changed too much before |

---

## Key Files

```
brain_mcp/
  server.py                 # MCP server (5 tools)
api/
  queries.py                # Database query functions
  probability.py            # Probability calculations
simulation/
  engine.py                 # Monte Carlo engine
  parlay_analyzer.py        # Correlation detection
batch/
  precompute.py             # Nightly projection job
  scrape_injuries.py        # ESPN injury scraper
```

---

## DO NOT

- Suggest switching APIs (SGO is LOCKED)
- Use >4 seasons of historical data
- Make LLM do math (engine does math, LLM explains)
- Push parlays unless user asks (response format rule)
