# SPRINT 1: THE FACTORY (Batch Precompute)

**Goal:** Daily pre-computation pipeline that fills the projections table.
**Duration:** 3-4 days
**Blocker Risk:** Medium — nba_api quirks

---

## Sprint Summary

| Task ID | Task | Parallel? | Depends On | Output |
|---------|------|-----------|------------|--------|
| S1-T1 | Implement fetch_tonights_games | No | Sprint 0 | `batch/precompute.py` |
| S1-T2 | Wire Monte Carlo to batch | No | S1-T1 | Updated `precompute.py` |
| S1-T3 | Implement save_projections | Yes | S1-T1 | Updated `precompute.py` |
| S1-T4 | Implement injury scraper | Yes | Sprint 0 | `batch/scrape_injuries.py` |
| S1-T5 | Integration test + dry run | No | S1-T1-T4 | Test pass + sample data |

---

## 🔲 HUMAN CHECKPOINT

Before executing Sprint 1, confirm:
- [ ] nba_api is installed (`pip install nba_api`)
- [ ] You understand the existing Monte Carlo engine in `simulation/engine.py`
- [ ] OK to run ~5 minute batch job that populates ~1000 rows per game slate

---

## Task Cards

---

### S1-T1: Implement fetch_tonights_games

**Type:** API Integration
**Input:** nba_api docs, existing `poc.py` (has `get_todays_games` function)
**Output:** Updated `batch/precompute.py`

**Task Prompt for Sub-Agent:**
```
Implement the fetch_tonights_games() function in batch/precompute.py.

Reference: Look at poc.py which already has a working get_todays_games() function using nba_api.

Requirements:
1. Use nba_api.live.endpoints.scoreboard to get today's games
2. Return list of dicts with structure:
   {
     "game_id": "0022400123",  # NBA game ID
     "home_team": "LAL",
     "away_team": "BOS",
     "starts_at": "2025-12-06T19:30:00Z",
     "status": "scheduled"
   }
3. Handle the case where no games today (return empty list)
4. Add try/except for API failures with logging
5. Insert games into the `games` table (upsert on game_id)

Database connection pattern:
```python
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "brain.db"

def get_db():
    return sqlite3.connect(DB_PATH)
```

Test by running:
```python
python -c "from batch.precompute import fetch_tonights_games; print(fetch_tonights_games())"
```
```

**Verification:**
```bash
# verify_s1t1.sh
python -c "
from batch.precompute import fetch_tonights_games
games = fetch_tonights_games()
assert isinstance(games, list), 'Must return list'
if games:
    assert 'game_id' in games[0], 'Missing game_id'
    assert 'home_team' in games[0], 'Missing home_team'
print(f'✅ PASS: Found {len(games)} games')
"
```

---

### S1-T2: Wire Monte Carlo to Batch

**Type:** Integration
**Input:** Existing `simulation/engine.py`, S1-T1 output
**Output:** Updated `batch/precompute.py`

**Task Prompt for Sub-Agent:**
```
Implement run_simulations() in batch/precompute.py.

This function must:
1. Accept a game dict from fetch_tonights_games()
2. Import and use the existing MonteCarloEngine from simulation/
3. Run engine.simulate_game(home_team, away_team)
4. Extract projections for all players, all stats

Key insight: The existing engine already outputs a SimulationResult with:
- result.players[player_name].pts.mean, .std, .p10, .p25, .p50, .p75, .p90
- Same for reb, ast, stl, blk, tov, fg3m

Transform this into a list of projection dicts:
[
  {
    "game_id": "0022400123",
    "player_name": "LeBron James",
    "stat_type": "pts",
    "mean": 27.3,
    "std": 4.8,
    "p10": 20.1,
    "p25": 23.5,
    "p50": 27.2,
    "p75": 31.0,
    "p90": 33.9,
    "computed_at": "2025-12-06T15:00:00Z"
  },
  ...
]

Return this list (saving to DB happens in S1-T3).

Important:
- Set n_simulations=10000 (current default)
- Capture timing: print how long simulation took
- Handle edge case where roster fetch fails
```

**Verification:**
```bash
# verify_s1t2.sh
python -c "
from batch.precompute import fetch_tonights_games, run_simulations
games = fetch_tonights_games()
if games:
    projections = run_simulations(games[0])
    assert len(projections) > 0, 'No projections returned'
    assert 'mean' in projections[0], 'Missing mean'
    print(f'✅ PASS: Generated {len(projections)} projections for game')
else:
    print('⚠️ SKIP: No games today to test')
"
```

---

### S1-T3: Implement save_projections

**Type:** Database Operations
**Input:** S1-T2 output structure
**Output:** Updated `batch/precompute.py`

**Task Prompt for Sub-Agent:**
```
Implement save_projections() in batch/precompute.py.

Requirements:
1. Accept list of projection dicts from run_simulations()
2. Insert/update into projections table using INSERT OR REPLACE
3. Use executemany for efficiency
4. Return count of rows inserted
5. Log: "Saved {n} projections for game {game_id}"

SQL pattern:
```sql
INSERT OR REPLACE INTO projections 
(game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90, computed_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

Also implement main() function:
```python
def main():
    """Orchestrate nightly precompute."""
    print(f"Starting precompute at {datetime.now()}")
    
    games = fetch_tonights_games()
    print(f"Found {len(games)} games")
    
    total_projections = 0
    for game in games:
        print(f"Simulating {game['away_team']} @ {game['home_team']}...")
        projections = run_simulations(game)
        count = save_projections(projections)
        total_projections += count
    
    print(f"Complete. {total_projections} projections saved.")

if __name__ == "__main__":
    main()
```
```

**Verification:**
```bash
# verify_s1t3.sh
python batch/precompute.py  # Run the main function
sqlite3 brain.db "SELECT COUNT(*) FROM projections"  # Should show rows
```

---

### S1-T4: Implement Injury Scraper

**Type:** Web Scraping
**Input:** Existing `poc.py` (has `scrape_espn_injuries`)
**Output:** Updated `batch/scrape_injuries.py`

**Task Prompt for Sub-Agent:**
```
Implement batch/scrape_injuries.py by porting logic from poc.py.

The existing poc.py has a working scrape_espn_injuries() function. Port it to:

scrape_espn() -> dict:
    Returns {
        "LAL": [
            {"player": "Anthony Davis", "status": "OUT", "injury": "Knee"},
            ...
        ],
        ...
    }

update_injuries_table(injuries: dict):
    1. Clear old injuries (DELETE FROM injuries WHERE source = 'ESPN')
    2. Insert fresh data
    3. Don't touch source='USER' injuries (those are manual overrides)

main():
    injuries = scrape_espn()
    update_injuries_table(injuries)
    print(f"Updated injuries for {len(injuries)} teams")

Run as: python batch/scrape_injuries.py
```

**Verification:**
```bash
# verify_s1t4.sh
python batch/scrape_injuries.py
sqlite3 brain.db "SELECT team, COUNT(*) FROM injuries GROUP BY team"
```

---

### S1-T5: Sprint 1 Integration Test

**Type:** End-to-End Test
**Input:** All S1 outputs
**Output:** Test report + sample data

**Task Prompt for Sub-Agent:**
```
Create tests/test_sprint1.py with:

1. test_fetch_games(): Verify fetch_tonights_games returns valid structure
2. test_simulation_runs(): Run simulation for one game (mock if no games)
3. test_projections_saved(): Verify projections table has data after run
4. test_injuries_scraped(): Verify injuries table has data

Also create verify_sprint1.sh that:
1. Runs python batch/precompute.py
2. Runs python batch/scrape_injuries.py
3. Queries database to show summary:
   - Number of games
   - Number of projections
   - Number of injuries
4. Prints sample projection (first 5 rows)
```

**Verification:**
```bash
pytest tests/test_sprint1.py -v
./verify_sprint1.sh
```

---

## Success Criteria

Sprint 1 is COMPLETE when:
- [ ] `python batch/precompute.py` runs without error
- [ ] Projections table has ~150-200 rows per game
- [ ] Injuries table has ESPN data
- [ ] Full pipeline runs in <10 minutes for 8-game slate
- [ ] Human has run and approved output

---

## Key Insight

From your Sprint Plan:
> "You run one command at 3pm. Five minutes later, your database has projections for every player in every game tonight."

This sprint delivers exactly that. The "Factory" runs once per day and fills inventory.

---

## Proceed to Sprint 2?

Run final verification:
```bash
./verify_sprint1.sh
```

If all ✅: Execute Sprint 2
If any ❌: Re-run failed task with error context
