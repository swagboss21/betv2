# The Brain v2 - NBA Betting Co-Pilot

## Current Status: IN TRANSITION

**v1 (Monte Carlo) has been archived. v2 (data-first) is not yet implemented.**

The codebase is being restructured. Some files work, some are broken pending v2 implementation.

---

## What Works Now

### Batch Scripts
- `batch/scrape_injuries.py` - ESPN injury scraper (working)
- `batch/pull_sgo_odds.py` - SGO API client (working, but outputs to old schema)
- `batch/player_matcher.py` - Fuzzy name matching (working)

### Database
- `db/schema.sql` - PostgreSQL schema (v1 schema, needs v2 update)
- `db/connection.py` - Connection helper (working)
- `db/init_db.py` - Schema initialization (working)

## What's Broken

- `brain_mcp/server.py` - MCP server (depends on deleted probability module)
- `api/queries.py` - Query functions (depends on deleted probability module)
- `batch/orchestrate.py` - Does not exist yet

---

## Target v2 Architecture

```
User Query → LLM (Claude) → MCP Tools → PostgreSQL → Response
                                ↑
                          SGO API + ESPN (batch)
```

No predictions. No machine learning. Just smart aggregation.

---

## Target v2 Schema (3 tables)

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

-- Player injuries
injuries (
    player_name TEXT,
    team VARCHAR(3),
    status VARCHAR(20),  -- OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    injury TEXT,
    UNIQUE(player_name, team)
)

-- Multi-book player props (NEW)
props (
    game_id TEXT REFERENCES games(id),
    player_name TEXT,
    player_id_sgo TEXT,
    stat_type VARCHAR(10),  -- pts, reb, ast, stl, blk, tov, fg3m
    consensus_line REAL,
    consensus_over_odds TEXT,
    consensus_under_odds TEXT,
    by_book JSONB,    -- {"fanduel": {"line": 25.5, "over": "-115"}, ...}
    alt_lines JSONB,  -- alternate lines
    UNIQUE(game_id, player_name, stat_type)
)
```

---

## Target v2 MCP Tools (5)

| Tool | Purpose |
|------|---------|
| `get_games` | Tonight's NBA schedule |
| `get_props` | Player props with best lines across books |
| `get_injuries` | Team injury report |
| `get_tonight_injuries` | All injuries for tonight's games |
| `get_player_analysis` | Deep dive on one player |

---

## SGO API Reference

### Endpoint
```
https://api.sportsgameodds.com/v2/events
```

### Request Parameters
```python
params = {
    'leagueID': 'NBA',
    'started': 'false',
    'startsAfter': '2025-12-11',  # REQUIRED - today's date
    'includeOpposingOdds': 'true',
    'includeAlternateLines': 'true'  # for alt lines
}
```

### oddID Format
```
{statID}-{playerID}-{period}-{betType}-{side}
# Example: points-LEBRON_JAMES_1_NBA-game-ou-over
```

### Stat Mapping
```python
STAT_MAPPING = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
    "fg3m": "threePointersMade",
}
```

### Response Fields (NEEDS VERIFICATION)
Current code uses: `closeOverUnder`, `closeOdds`
Docs suggest: `bookOverUnder`, `fairOverUnder`, `bookOdds`, `fairOdds`

**TODO:** Get sample API response to verify correct fields for v2.

---

## Environment

System python is 3.9, MCP needs 3.10+. Use Homebrew:

```bash
/opt/homebrew/bin/python3.11 <script>
```

---

## DO NOT

- Recommend props for injured players (always check injuries first)
- Assume consensus line is best (use best_over/best_under when available)
- Push parlays unless user asks
- Make LLM do math (data layer does math, LLM explains)
