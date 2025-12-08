# SPRINT 0: FOUNDATION

**Goal:** Set up the filing system before you start filing.
**Duration:** 2-3 days
**Blocker Risk:** Low — just setup

---

## Sprint Summary

| Task ID | Task | Parallel? | Depends On | Output |
|---------|------|-----------|------------|--------|
| S0-T1 | Create database schema | No | — | `db/schema.sql` |
| S0-T2 | Initialize SQLite database | No | S0-T1 | `brain.db` |
| S0-T3 | Create folder structure | Yes | — | Directories |
| S0-T4 | Create batch script stubs | Yes | S0-T3 | `batch/*.py` |
| S0-T5 | Create API module stubs | Yes | S0-T3 | `api/*.py` |
| S0-T6 | Verify setup | No | S0-T1-T5 | Test pass |

---

## 🔲 HUMAN CHECKPOINT

Before executing Sprint 0, confirm:
- [ ] Database will live at project root as `brain.db`
- [ ] Using SQLite (not Postgres) for v1
- [ ] OK to create new folder structure alongside existing `simulation/` and `models/`

---

## Task Cards

---

### S0-T1: Create Database Schema

**Type:** Code Generation
**Input:** Sprint Plan requirements (6 tables)
**Output:** `db/schema.sql`

**Task Prompt for Sub-Agent:**
```
Create a SQLite schema file with the following 6 tables:

1. games - Tonight's games
   - id (PRIMARY KEY, TEXT - use SGO eventID format)
   - home_team (TEXT, 3-letter abbrev)
   - away_team (TEXT)
   - starts_at (DATETIME)
   - home_score (INTEGER, nullable - filled after game)
   - away_score (INTEGER, nullable)
   - status (TEXT: 'scheduled', 'live', 'final')
   - created_at (DATETIME DEFAULT CURRENT_TIMESTAMP)

2. projections - Monte Carlo results (the gold)
   - id (INTEGER PRIMARY KEY AUTOINCREMENT)
   - game_id (TEXT, FK to games.id)
   - player_name (TEXT)
   - stat_type (TEXT: pts, reb, ast, stl, blk, tov, fg3m)
   - mean (REAL)
   - std (REAL)
   - p10 (REAL)
   - p25 (REAL)
   - p50 (REAL)
   - p75 (REAL)
   - p90 (REAL)
   - computed_at (DATETIME)
   - UNIQUE(game_id, player_name, stat_type)

3. injuries - Current injury status
   - id (INTEGER PRIMARY KEY AUTOINCREMENT)
   - player_name (TEXT)
   - team (TEXT, 3-letter abbrev)
   - status (TEXT: 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE')
   - injury (TEXT, description)
   - source (TEXT: 'ESPN', 'USER')
   - updated_at (DATETIME)
   - UNIQUE(player_name, team)

4. users
   - id (TEXT PRIMARY KEY - UUID)
   - email (TEXT UNIQUE)
   - auth_provider (TEXT: 'google', 'apple')
   - is_paid (INTEGER DEFAULT 0)
   - message_count (INTEGER DEFAULT 0)
   - created_at (DATETIME)
   - last_active_at (DATETIME)

5. bets - User locked bets
   - id (INTEGER PRIMARY KEY AUTOINCREMENT)
   - user_id (TEXT, FK to users.id)
   - game_id (TEXT, FK to games.id)
   - player_name (TEXT)
   - stat_type (TEXT)
   - line (REAL)
   - direction (TEXT: 'OVER', 'UNDER')
   - odds (TEXT)
   - edge_pct (REAL)
   - locked_at (DATETIME)
   - result (TEXT, nullable: 'WIN', 'LOSS', 'PUSH')
   - actual_value (REAL, nullable)
   - graded_at (DATETIME, nullable)

6. house_bots - Automated bettors
   - id (TEXT PRIMARY KEY)
   - name (TEXT)
   - strategy (TEXT: 'aggressive', 'conservative', 'balanced')
   - bankroll (REAL DEFAULT 1000.0)
   - starting_bankroll (REAL DEFAULT 1000.0)
   - is_alive (INTEGER DEFAULT 1)
   - created_at (DATETIME)

Include:
- Appropriate indexes for common queries
- Foreign key constraints
- Comments explaining each table
```

**Verification:**
```bash
# verify_s0t1.sh
sqlite3 :memory: < db/schema.sql && echo "✅ PASS: Schema syntax valid" || echo "❌ FAIL: Schema has errors"
```

---

### S0-T2: Initialize SQLite Database

**Type:** Script Execution
**Input:** `db/schema.sql`
**Output:** `brain.db`

**Task Prompt for Sub-Agent:**
```
Create a Python script `db/init_db.py` that:
1. Checks if brain.db exists
2. If exists, prompts for confirmation before overwriting (or add --force flag)
3. Creates brain.db by executing schema.sql
4. Prints table list to confirm creation

Run the script to initialize the database.
```

**Verification:**
```bash
# verify_s0t2.sh
sqlite3 brain.db ".tables" | grep -q "games projections injuries users bets house_bots" && echo "✅ PASS" || echo "❌ FAIL"
```

---

### S0-T3: Create Folder Structure

**Type:** File System Setup
**Input:** None
**Output:** Empty directories

**Task Prompt for Sub-Agent:**
```
Create the following directory structure (if directories don't exist):
- db/
- db/migrations/
- batch/
- api/
- brain_mcp/
- tests/

Do NOT touch existing directories: simulation/, models/, archive/

Create empty __init__.py files in each Python package directory.
```

**Verification:**
```bash
# verify_s0t3.sh
for dir in db batch api brain_mcp tests; do
  [ -d "$dir" ] && echo "✅ $dir exists" || echo "❌ $dir missing"
done
```

---

### S0-T4: Create Batch Script Stubs

**Type:** Code Generation
**Input:** Sprint 1 requirements
**Output:** `batch/__init__.py`, `batch/precompute.py`, `batch/scrape_injuries.py`, `batch/grade_bets.py`

**Task Prompt for Sub-Agent:**
```
Create stub files in batch/ with docstrings and placeholder functions:

batch/precompute.py:
- def fetch_tonights_games() -> list[dict]: """Fetch games from nba_api"""
- def run_simulations(game_id: str) -> dict: """Run Monte Carlo for one game"""
- def save_projections(projections: list[dict]): """Insert into projections table"""
- def main(): """Orchestrate nightly precompute"""

batch/scrape_injuries.py:
- def scrape_espn() -> dict: """Scrape ESPN injury page"""
- def update_injuries_table(injuries: dict): """Upsert injuries"""
- def main(): """Run injury scrape"""

batch/grade_bets.py:
- def fetch_actuals(game_id: str) -> dict: """Get actual stats from nba_api"""
- def grade_bets(game_id: str): """Compare actuals to locked bets, update results"""
- def update_bot_bankrolls(): """Adjust bot bankrolls based on results"""
- def main(): """Run morning grading"""

Each function should:
- Have a TODO comment: "# TODO: Implement in Sprint 1/4"
- Pass without error (return empty dict/list or None)
- Include type hints
```

**Verification:**
```bash
# verify_s0t4.sh
python -c "from batch import precompute, scrape_injuries, grade_bets" && echo "✅ PASS" || echo "❌ FAIL"
```

---

### S0-T5: Create API Module Stubs

**Type:** Code Generation
**Input:** Sprint 2 requirements
**Output:** `api/__init__.py`, `api/queries.py`, `api/probability.py`

**Task Prompt for Sub-Agent:**
```
Create stub files in api/:

api/queries.py - Database query functions:
- def get_games_today() -> list[dict]
- def get_projection(player_name: str, stat: str, game_id: str) -> dict | None
- def get_all_projections(game_id: str) -> list[dict]
- def get_best_props(min_edge: float = 0.05) -> list[dict]
- def get_injuries(team: str) -> list[dict]
- def save_bet(user_id: str, bet: dict) -> int
- def get_user_bets(user_id: str) -> list[dict]
- def get_user_record(user_id: str) -> dict

api/probability.py - Math helpers:
- def prob_over(mean: float, std: float, line: float) -> float:
    """Given normal dist params, calculate P(X > line) using scipy.stats.norm.sf"""
- def prob_under(mean: float, std: float, line: float) -> float
- def implied_probability(american_odds: str) -> float
- def calculate_edge(model_prob: float, implied_prob: float) -> float

Each function should have:
- TODO comment
- Type hints
- Docstring
- Pass without error (return empty/placeholder)
```

**Verification:**
```bash
# verify_s0t5.sh
python -c "from api import queries, probability" && echo "✅ PASS" || echo "❌ FAIL"
```

---

### S0-T6: Sprint 0 Integration Test

**Type:** Test Execution
**Input:** All S0 outputs
**Output:** Test report

**Task Prompt for Sub-Agent:**
```
Create tests/test_sprint0.py that verifies:

1. brain.db exists and has 6 tables
2. All module stubs import without error
3. Folder structure is correct
4. Schema has correct columns (spot check 2-3 tables)

Run pytest and report results.
```

**Verification:**
```bash
pytest tests/test_sprint0.py -v
```

---

## Success Criteria

Sprint 0 is COMPLETE when:
- [ ] `brain.db` exists with 6 empty tables
- [ ] All directories created
- [ ] All stub modules import without error
- [ ] Integration test passes
- [ ] Human has reviewed schema and approved

---

## Proceed to Sprint 1?

Run final verification:
```bash
./verify_sprint0.sh
```

If all ✅: Execute Sprint 1
If any ❌: Re-run failed task with error context
