# Sprint 2 Complete - Handoff Document

**Date:** 2025-12-07
**Status:** ✅ COMPLETE (30/30 tests passing)

---

## What Was Built

Sprint 2 implemented the **API Layer (The Warehouse)** - Python module functions for querying precomputed Monte Carlo projections with support for **alternate line queries** (PrizePicks Goblin/Demon style).

### Key Feature: Arbitrary Line Probability
Users can now query probability for ANY line value, not just stored percentiles:
```python
from api.queries import get_projection

# Query any line - this is the "Goblin/Demon" feature
proj = get_projection("Stephen Curry", "pts", line=22.5)
# Returns: prob_over=0.486 (48.6% chance of OVER 22.5)
```

---

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `api/probability.py` | Modified | 6 probability math functions implemented |
| `api/queries.py` | Modified | 13 database query functions implemented |
| `tests/test_sprint2.py` | Created | 30 integration tests |

---

## Functions Implemented

### `api/probability.py` (S2-T1)
```python
prob_over(mean, std, line) -> float      # P(X > line) using scipy norm.sf()
prob_under(mean, std, line) -> float     # P(X < line) using scipy norm.cdf()
american_to_probability(odds) -> float   # "-115" -> 0.535
calculate_edge(model_prob, book_prob) -> float
format_edge(edge) -> str                 # "+5.2%" or "PASS"
devig_odds(over_odds, under_odds) -> tuple[float, float]
```

### `api/queries.py` - Projection Queries (S2-T2)
```python
get_games_today() -> list[dict]
get_projection(player_name, stat_type, game_id=None, line=None) -> dict | None
get_all_projections(game_id) -> list[dict]
get_injuries(team) -> list[dict]
```

### `api/queries.py` - Best Props (S2-T3)
```python
get_best_props(min_edge=0.05, limit=10) -> list[dict]
# Returns props sorted by edge descending
# Uses p50 (median) as market line placeholder (v1)
# Assumes -110 juice (52.4% breakeven)
```

### `api/queries.py` - Bet Operations (S2-T4)
```python
save_bet(user_id, bet) -> int
get_user_bets(user_id, pending_only=False) -> list[dict]
get_user_record(user_id) -> dict  # {total, wins, losses, pushes, win_rate}
```

### `api/queries.py` - User Operations (S2-T5)
```python
create_user(email, auth_provider='google') -> str  # Returns UUID
get_user(email) -> dict | None
get_or_create_user(email, auth_provider='google') -> dict
increment_message_count(user_id) -> int
is_user_paid(user_id) -> bool
set_user_paid(user_id, is_paid=True) -> None
```

---

## Response Format for Alternate Lines

```python
get_projection("LeBron James", "pts", line=22.5)

# Returns:
{
    "player_name": "LeBron James",
    "stat_type": "pts",
    "game_id": "0022500362",
    "mean": 26.3,
    "std": 5.2,
    "p10": 19.5,
    "p25": 22.8,
    "p50": 26.1,
    "p75": 29.5,
    "p90": 33.2,
    "line": 22.5,
    "prob_over": 0.766,   # 76.6% chance (Goblin = easier line)
    "prob_under": 0.234
}
```

---

## Verification Commands

```bash
cd "/Users/noahcantu/Desktop/the-brain-organized 2"
export DATABASE_URL="postgresql://postgres:brain123@localhost:5432/brain"

# Run all Sprint 2 tests
python3 -m pytest tests/test_sprint2.py -v

# Quick smoke test
python3 -c "
from api.probability import prob_over
from api.queries import get_projection

print(f'P(26.3 > 25.5) = {prob_over(26.3, 5.2, 25.5):.3f}')  # ~0.561

proj = get_projection('Stephen Curry', 'pts', line=22.5)
if proj:
    print(f'P(Curry > 22.5) = {proj[\"prob_over\"]:.3f}')
"
```

---

## Database State

| Table | Rows | Notes |
|-------|------|-------|
| games | 7 | Tonight's NBA schedule |
| projections | 1,211 | 5/7 games have projections (LAL@PHI, OKC@UTA missing) |
| injuries | 121 | ESPN scraped data (verified) |
| users | ~10 | Test users created during testing |
| bets | ~5 | Test bets created during testing |
| house_bots | 3 | Seeded in Sprint 0 |

---

## Known Issues / Tech Debt

1. **2 games missing projections** - LAL@PHI and OKC@UTA weren't simulated (user stopped early)
2. **Game times NULL** - `starts_at` column not populated in games table
3. **p50 as market line** - `get_best_props()` uses median as placeholder; needs real odds API later
4. **Test cleanup** - Test users/bets accumulate in database (no cleanup fixture)

---

## Dependencies

- `scipy` - For normal distribution CDF/SF calculations
- `psycopg2` - PostgreSQL connection
- `pytest` - Testing

---

## What's Next: Sprint 3 (COMPLETE)

Sprint 3 was the **MCP Server** - integrating the API layer with Claude Desktop via MCP protocol.

**Completed:** `brain_mcp/server.py` exposes 5 tools to Claude Desktop.

Key deliverables:
- `get_games_today` - Tonight's NBA schedule
- `get_projection` - Player stat projection with probability
- `get_best_props` - Top edge props ranked
- `get_injuries` - Team injury report
- `lock_bet` - Save user's bet
- Update `build_parlay` to use stored projections

---

## Completed Sprints Summary

| Sprint | Status | Summary |
|--------|--------|---------|
| Sprint 0 | ✅ COMPLETE | PostgreSQL schema (6 tables), folder structure, db/connection.py |
| Sprint 1 | ✅ COMPLETE | Batch precompute pipeline + ESPN injury scraper |
| Sprint 2 | ✅ COMPLETE | API layer with alternate line support (30/30 tests) |
| Sprint 3 | 🔜 NEXT | Chat interface integration |

---

## Docker/DB Access

```bash
# PostgreSQL container
docker ps | grep postgres  # brain-postgres on port 5432

# Connection string
DATABASE_URL="postgresql://postgres:brain123@localhost:5432/brain"
```
