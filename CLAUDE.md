# The Brain v2 - NBA Betting Co-Pilot

## What This Is

Data-first NBA player prop betting assistant. Aggregates odds from multiple sportsbooks, finds best lines, serves structured data to LLMs via MCP.

**Philosophy:** Sportsbooks have already done the math. Our value is curation, structure, and finding the best available lines.

---

## Architecture

```
User Query → LLM (Claude) → MCP Tools → PostgreSQL → Response
                                ↑
                          SGO API (batch refresh)
```

No predictions. No machine learning. Just smart aggregation.

---

## Database Schema

### 3 Tables

```sql
-- Games scheduled for today
games (
    id TEXT PRIMARY KEY,
    home_team VARCHAR(3),
    away_team VARCHAR(3),
    starts_at TIMESTAMPTZ,
    status VARCHAR(20),  -- scheduled, final
    home_score INT,
    away_score INT
)

-- Player injuries linked to games
injuries (
    game_id TEXT REFERENCES games(id),
    player_name TEXT,
    team VARCHAR(3),
    status VARCHAR(20),  -- OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    injury TEXT,
    UNIQUE(game_id, player_name)
)

-- Multi-book player props
props (
    game_id TEXT REFERENCES games(id),
    player_name TEXT,
    player_id_sgo TEXT,
    stat_type VARCHAR(10),  -- pts, reb, ast, stl, blk, tov, fg3m
    consensus_line REAL,
    consensus_over_odds TEXT,
    consensus_under_odds TEXT,
    by_book JSONB,    -- {"fanduel": {"line": 25.5, "over": "-115", "under": "-105"}, ...}
    alt_lines JSONB,  -- [{"book": "fanduel", "line": 23.5, "over": "-170", "under": "+140"}, ...]
    UNIQUE(game_id, player_name, stat_type)
)
```

---

## MCP Tools

5 tools exposed to Claude Desktop:

| Tool | Purpose |
|------|---------|
| `get_games` | Tonight's NBA schedule |
| `get_props` | Player props with best lines across books |
| `get_injuries` | Team injury report |
| `get_tonight_injuries` | All injuries for tonight's games |
| `get_player_analysis` | Deep dive on one player (all stats, all games) |

### Key Features

- **Best line calculation**: Returns best over (lowest line) and best under (highest line) across all books
- **Line spread**: Flags props where books disagree significantly
- **Injury cross-reference**: Props include player injury status

---

## Pipeline

```bash
# Run full pipeline
/opt/homebrew/bin/python3.11 batch/orchestrate.py
```

### 4 Steps

1. **GAMES** - Fetch tonight's schedule, cleanup stale games
2. **INJURIES** - Scrape ESPN injury reports
3. **PROPS** - Pull SGO odds (multi-book)
4. **VALIDATE** - Check data integrity

### Options

```bash
--lenient       # Log errors but continue (don't fail)
--skip-props    # Skip SGO odds pull
```

---

## Key Files

```
batch/
  orchestrate.py        # Pipeline entry point (4 steps)
  pull_sgo_odds.py      # SGO API client (multi-book)
  scrape_injuries.py    # ESPN injury scraper
  player_matcher.py     # Fuzzy name matching
  fetch_results.py      # Morning-after bet grading

api/
  queries.py            # Database queries (props, injuries, games)

brain_mcp/
  server.py             # MCP server (5 tools)

db/
  schema.sql            # PostgreSQL schema (3 tables)
  init_db.py            # Database initialization
```

---

## SGO API Reference

### Team Mapping

SGO uses different abbreviations:
```python
SGO_TO_NBA_TEAM = {
    "PHO": "PHX",  "BRK": "BKN",  "CHO": "CHA",
    "GS": "GSW",   "NY": "NYK",   "SA": "SAS",
    "NO": "NOP",   "WSH": "WAS"
}
```

### API Parameters

```python
params = {
    'leagueID': 'NBA',
    'started': 'false',
    'startsAfter': '2025-12-11',  # REQUIRED - today's date
    'includeOpposingOdds': 'true',
    'includeAlternateLines': 'true'
}
```

### Response Fields

- **Lines**: `bookOverUnder` or `fairOverUnder` (not `closeOverUnder`)
- **Odds**: `bookOdds` or `fairOdds` (not `closeOdds`)
- **Player IDs**: Format `LEBRON_JAMES_1_NBA`

---

## Environment

System python is 3.9, MCP needs 3.10+. Use Homebrew:

```bash
/opt/homebrew/bin/python3.11 batch/orchestrate.py
```

---

## DO NOT

- Recommend props for injured players (always check injuries first)
- Assume consensus line is best (use best_over/best_under)
- Push parlays unless user asks
- Make LLM do math (data layer does math, LLM explains)
