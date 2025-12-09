# The Brain - NBA Betting Co-Pilot

## What This Is
AI-powered NBA player prop betting co-pilot using Monte Carlo simulation + LLM reasoning.

---

## Handoff: Dec 8, 2025

### What Was Just Done (Sprint 5)
1. Fixed `batch/pull_sgo_odds.py` - added `startsAfter` parameter (line 140)
2. Created `batch/fetch_results.py` - morning-after bet grading using nba_api box scores

### Test Run Results
```bash
# Injuries: WORKING
python3 batch/scrape_injuries.py  # Found 114 injuries across 29 teams

# SGO Odds: NOT MATCHING
python3 batch/pull_sgo_odds.py    # Received 2 events, matched 0 to our games
# Issue: SGO returns events but team names don't match our DB codes

# Results Fetch: TIMING ISSUE
python3 batch/fetch_results.py    # "No team stats found" for game 0022500364
# Issue: Game still in progress or too recent for nba_api
```

### Environment Note
System python is 3.9, MCP needs 3.10+. Dependencies installed to Homebrew python:
```bash
/opt/homebrew/bin/python3.11 batch/precompute.py
```

### Next Agent TODO
1. Debug SGO event matching - log what team names SGO returns vs our DB
2. Fix fetch_results.py to only process truly completed games (check nba_api game status)
3. Consider adding shell alias or updating PATH for python3.11

---

## Current State: Sprint 5 COMPLETE (with known issues)

All core components built and tested:

| Sprint | Status | What's Done |
|--------|--------|-------------|
| 0. Database | ✅ COMPLETE | PostgreSQL schema, 6 tables |
| 1. Batch Pipeline | ✅ COMPLETE | Monte Carlo precompute + ESPN scraper |
| 2. API Layer | ✅ COMPLETE | 13 query functions, 30/30 tests passing |
| 3. MCP Server | ✅ COMPLETE | Claude Desktop integration via MCP protocol |
| 4. Data Hardening | ✅ COMPLETE | Bug fixes, odds schema, SGO integration, bulk endpoint |
| 5. Results Tracking | ⚠️ COMPLETE | SGO startsAfter fix, bet grading pipeline (see known issues) |

---

## Known Issues (Sprint 6 TODO)

### 1. SGO Event Matching Fails
**Symptom:** `pull_sgo_odds.py` returns "Matched 0 SGO events to our games"
**Cause:** SGO API returns events but team abbreviations don't match our DB
**Debug needed:** Log SGO event team names vs our DB team codes

### 2. Box Score Fetch Fails for Live/Recent Games
**Symptom:** `fetch_results.py` says "No team stats found for {game_id}"
**Cause:** nba_api BoxScoreTraditionalV2 may not have data for games still in progress or very recently completed
**Fix needed:** Add delay or check game status before fetching

### 3. Python Version Mismatch
**Note:** System python is 3.9, but MCP requires 3.10+. Use Homebrew python:
```bash
/opt/homebrew/bin/python3.11 batch/precompute.py
# or add to PATH: export PATH="/opt/homebrew/bin:$PATH"
```

---

## Sprint 4: Data Layer Hardening (Dec 2024)

### What Was Done

1. **Fixed `get_best_props()` bug** - Added min_lines filter to prevent fake props like "UNDER 0 fg3m"
   - File: `api/queries.py` (lines 150-165)
   - Min thresholds: pts≥5.5, reb≥2.5, ast≥1.5, others≥0.5

2. **Added deviation signal** - Hot/cold streak indicator
   - Schema: Added `l5_avg`, `szn_avg`, `l5_std` columns to projections table
   - Formula: `deviation = (l5_avg - szn_avg) / max(l5_std, 0.5)`
   - Populated during `batch/precompute.py` run

3. **Created odds table** - Schema ready for real sportsbook lines
   - File: `db/schema.sql`
   - Columns: game_id, player_name, stat_type, line, over_odds, under_odds, book

4. **Built SGO integration** - SportsGameOdds API integration
   - Files: `batch/pull_sgo_odds.py`, `batch/player_matcher.py`
   - API VERIFIED WORKING (see below)

5. **Created bulk endpoint** - `get_tonight_analysis()` MCP tool
   - Returns games, players, projections, odds, injuries in one call
   - File: `brain_mcp/server.py`

6. **Created validation script** - `scripts/validate_data.py`

### SGO API - VERIFIED WORKING

**Key finding:** Must use `startsAfter` param or API returns old data from April 2024.

```python
# Working API call:
params = {
    'leagueID': 'NBA',
    'started': 'false',
    'startsAfter': '2025-12-08',  # REQUIRED - use today's date
    'oddIDs': 'points-PLAYER_ID-game-ou-over,rebounds-PLAYER_ID-game-ou-over',
    'includeOpposingOdds': 'true'
}
```

**Response structure:**
- Player IDs format: `ANDREW_WIGGINS_1_NBA`, `BAM_ADEBAYO_1_NBA`
- Lines are in `closeOverUnder` field (may be None if not yet set)
- Odds in `closeOdds` field (American format, e.g., -110)

---

## Sprint 5: Results Tracking (Dec 2024)

### What Was Done

1. **Fixed SGO `startsAfter` parameter** - Critical fix to get today's odds
   - File: `batch/pull_sgo_odds.py` (line 140)
   - Without this param, API returns stale April 2024 data

2. **Built NBA results fetcher** - Morning-after job to grade bets
   - File: `batch/fetch_results.py`
   - Fetches box scores from nba_api after games complete
   - Updates `games.status` to 'final' with scores
   - Grades pending bets: WIN/LOSS/PUSH based on actual stats

### Usage

```bash
# Run in morning after games (e.g., 6am ET)
python batch/fetch_results.py

# Dry run (fetch but don't update DB)
python batch/fetch_results.py --dry-run
```

### Bet Grading Logic

```python
# Grade based on direction and actual value vs line
if actual_value == line:
    return "PUSH"
elif direction == "OVER":
    return "WIN" if actual_value > line else "LOSS"
else:  # UNDER
    return "WIN" if actual_value < line else "LOSS"
```

### Data Flow

```
Night before:    precompute.py → projections
                 pull_sgo_odds.py → odds
                 User locks bets via MCP → bets table (result=NULL)

Morning after:   fetch_results.py →
                   - Fetch box scores from nba_api
                   - Update games (status='final', scores)
                   - Grade bets (result=WIN/LOSS/PUSH, actual_value)
```

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

## What's Next (Sprint 6+)

**Immediate fixes needed:**
- [ ] Fix SGO event matching (debug team name mismatch)
- [ ] Fix box score fetch timing (wait for game to be truly final)
- [ ] Add Python 3.11 to PATH or create shell alias

**Building toward SaaS product:**
- [ ] User accounts + auth
- [x] Bet tracking + memory (done: bets table, lock_bet tool)
- [x] Outcome tracking (done: fetch_results.py grades bets)
- [ ] MCP tool to show user's bet history + win rate
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
  server.py                 # MCP server (7 tools including get_tonight_analysis)
api/
  queries.py                # Database query functions (get_projection, get_best_props, etc.)
  probability.py            # Probability calculations
simulation/
  engine.py                 # Monte Carlo engine
batch/
  precompute.py             # Nightly projection job (populates projections + deviation cols)
  scrape_injuries.py        # ESPN injury scraper
  pull_sgo_odds.py          # SGO API odds puller (uses startsAfter param)
  player_matcher.py         # Fuzzy name matching for SGO player IDs
  fetch_results.py          # Morning-after box score fetcher + bet grader
scripts/
  validate_data.py          # Data validation checks
db/
  schema.sql                # PostgreSQL schema (games, projections, odds, injuries, bets)
```

---

## DO NOT

- Make LLM do math (engine does math, LLM explains)
- Push parlays unless user asks (response format rule)
