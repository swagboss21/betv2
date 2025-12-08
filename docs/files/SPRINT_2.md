# SPRINT 2: THE WAREHOUSE (Query Functions)

**Goal:** Instant query functions that read from database.
**Duration:** 2-3 days
**Blocker Risk:** Low — straightforward

---

## Sprint Summary

| Task ID | Task | Parallel? | Depends On | Output |
|---------|------|-----------|------------|--------|
| S2-T1 | Implement probability math | Yes | — | `api/probability.py` |
| S2-T2 | Implement get_projection | Yes | Sprint 1 | `api/queries.py` |
| S2-T3 | Implement get_best_props | Yes | S2-T1, S2-T2 | `api/queries.py` |
| S2-T4 | Implement bet operations | Yes | Sprint 0 | `api/queries.py` |
| S2-T5 | Implement user queries | Yes | Sprint 0 | `api/queries.py` |
| S2-T6 | Integration test | No | S2-T1-T5 | Test pass |

---

## Key Insight

From your Sprint Plan:
> "The probability calculation is just: 'Given mean=26.3 and std=5.2, what's the chance of exceeding 25.5?' This is one line of statistics, not 10,000 simulations."

The Monte Carlo already ran. Now we just do quick lookups and math.

---

## Task Cards

---

### S2-T1: Implement Probability Math

**Type:** Math Functions
**Input:** Statistics knowledge
**Output:** Updated `api/probability.py`

**Task Prompt for Sub-Agent:**
```
Implement api/probability.py with pure math functions (no DB access):

from scipy import stats

def prob_over(mean: float, std: float, line: float) -> float:
    """
    Calculate probability of exceeding the line given normal distribution.
    
    P(X > line) = 1 - CDF(line)
    
    Example:
        mean=26.3, std=5.2, line=25.5
        returns 0.561 (56.1% chance of going over)
    """
    if std <= 0:
        return 1.0 if mean > line else 0.0
    return stats.norm.sf(line, loc=mean, scale=std)

def prob_under(mean: float, std: float, line: float) -> float:
    """P(X < line) = CDF(line)"""
    if std <= 0:
        return 1.0 if mean < line else 0.0
    return stats.norm.cdf(line, loc=mean, scale=std)

def american_to_probability(odds: str) -> float:
    """
    Convert American odds to implied probability.
    
    Examples:
        "-115" -> 0.535 (53.5%)
        "+150" -> 0.40 (40%)
    """
    odds_val = float(odds.replace('+', ''))
    if odds_val > 0:
        return 100 / (100 + odds_val)
    else:
        return abs(odds_val) / (abs(odds_val) + 100)

def calculate_edge(model_prob: float, book_prob: float) -> float:
    """
    Calculate betting edge.
    
    Edge = model_prob - book_prob
    
    Positive edge = we think over is more likely than book implies
    Negative edge = book has it right or overstates probability
    
    Example:
        model says 56% over, book implies 53.5%
        edge = 0.025 (2.5% edge)
    """
    return model_prob - book_prob

def format_edge(edge: float) -> str:
    """Format edge as percentage string."""
    if edge >= 0.05:
        return f"+{edge*100:.1f}%"
    elif edge <= -0.05:
        return f"{edge*100:.1f}%"
    else:
        return "PASS"
```

**Verification:**
```bash
python -c "
from api.probability import prob_over, american_to_probability, calculate_edge
# Test prob_over
p = prob_over(26.3, 5.2, 25.5)
assert 0.55 < p < 0.60, f'Expected ~0.56, got {p}'
# Test odds conversion
p = american_to_probability('-115')
assert 0.53 < p < 0.54, f'Expected ~0.535, got {p}'
print('✅ PASS: Probability math correct')
"
```

---

### S2-T2: Implement get_projection

**Type:** Database Query
**Input:** Projections table structure
**Output:** Updated `api/queries.py`

**Task Prompt for Sub-Agent:**
```
Implement core query functions in api/queries.py:

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "brain.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Dict-like access
    return conn

def get_games_today() -> list[dict]:
    """Get all games for today from games table."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM games 
            WHERE DATE(starts_at) = DATE('now')
            ORDER BY starts_at
        """).fetchall()
        return [dict(row) for row in rows]

def get_projection(player_name: str, stat_type: str, game_id: str = None) -> Optional[dict]:
    """
    Get projection for a specific player/stat.
    
    If game_id not provided, get most recent projection.
    
    Returns:
        {"mean": 26.3, "std": 5.2, "p10": 19.5, ...} or None if not found
    """
    with get_db() as conn:
        if game_id:
            row = conn.execute("""
                SELECT * FROM projections 
                WHERE player_name = ? AND stat_type = ? AND game_id = ?
            """, (player_name, stat_type, game_id)).fetchone()
        else:
            row = conn.execute("""
                SELECT * FROM projections 
                WHERE player_name = ? AND stat_type = ?
                ORDER BY computed_at DESC LIMIT 1
            """, (player_name, stat_type)).fetchone()
        
        return dict(row) if row else None

def get_all_projections(game_id: str) -> list[dict]:
    """Get all projections for a game."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM projections WHERE game_id = ?
        """, (game_id,)).fetchall()
        return [dict(row) for row in rows]

def get_injuries(team: str) -> list[dict]:
    """Get injury report for a team."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM injuries WHERE team = ?
        """, (team,)).fetchall()
        return [dict(row) for row in rows]
```

**Verification:**
```bash
python -c "
from api.queries import get_games_today, get_projection
games = get_games_today()
print(f'Found {len(games)} games today')
# Try to get a projection (may be empty if no games)
proj = get_projection('LeBron James', 'pts')
print(f'Sample projection: {proj}')
print('✅ PASS: Queries work')
"
```

---

### S2-T3: Implement get_best_props

**Type:** Complex Query
**Input:** S2-T1, S2-T2
**Output:** Updated `api/queries.py`

**Task Prompt for Sub-Agent:**
```
Add get_best_props() to api/queries.py.

This is the key function for "Tonight's best bets" feature.

from api.probability import prob_over, prob_under, calculate_edge

def get_best_props(min_edge: float = 0.05, limit: int = 10) -> list[dict]:
    """
    Get props with highest edge across all games tonight.
    
    For each projection, we need to calculate edge against market lines.
    Since we don't have live odds in DB, we'll use a placeholder approach:
    
    For v1: Return props where model is most confident (highest z-score from line)
    
    Returns list sorted by edge descending:
    [
        {
            "player_name": "LeBron James",
            "stat_type": "pts",
            "game_id": "0022400123",
            "mean": 27.3,
            "std": 4.8,
            "line": 25.5,  # Placeholder - would come from odds API
            "direction": "OVER",
            "probability": 0.65,
            "edge": 0.12
        },
        ...
    ]
    """
    with get_db() as conn:
        # Get all projections for today's games
        rows = conn.execute("""
            SELECT p.*, g.home_team, g.away_team
            FROM projections p
            JOIN games g ON p.game_id = g.id
            WHERE DATE(g.starts_at) = DATE('now')
        """).fetchall()
        
        results = []
        for row in rows:
            proj = dict(row)
            mean = proj['mean']
            std = proj['std']
            
            # For v1: Use median as "market line" placeholder
            # In production, this would come from SGO API
            line = proj['p50']
            
            # Calculate probability and edge
            p_over = prob_over(mean, std, line)
            p_under = prob_under(mean, std, line)
            
            # Pick direction with higher probability
            if p_over > p_under:
                direction = "OVER"
                probability = p_over
                edge = probability - 0.52  # Assume -110 juice (52% breakeven)
            else:
                direction = "UNDER"
                probability = p_under
                edge = probability - 0.52
            
            if edge >= min_edge:
                results.append({
                    "player_name": proj['player_name'],
                    "stat_type": proj['stat_type'],
                    "game_id": proj['game_id'],
                    "mean": mean,
                    "std": std,
                    "line": line,
                    "direction": direction,
                    "probability": round(probability, 3),
                    "edge": round(edge, 3)
                })
        
        # Sort by edge descending
        results.sort(key=lambda x: x['edge'], reverse=True)
        return results[:limit]
```

**Verification:**
```bash
python -c "
from api.queries import get_best_props
props = get_best_props(min_edge=0.03)
print(f'Found {len(props)} props with edge > 3%')
for p in props[:3]:
    print(f'  {p[\"player_name\"]} {p[\"stat_type\"]} {p[\"direction\"]}: edge={p[\"edge\"]}')
print('✅ PASS')
"
```

---

### S2-T4: Implement Bet Operations

**Type:** Database CRUD
**Input:** Bets table structure
**Output:** Updated `api/queries.py`

**Task Prompt for Sub-Agent:**
```
Add bet operations to api/queries.py:

from datetime import datetime

def save_bet(user_id: str, bet: dict) -> int:
    """
    Save a locked bet.
    
    bet = {
        "game_id": "0022400123",
        "player_name": "LeBron James",
        "stat_type": "pts",
        "line": 25.5,
        "direction": "OVER",
        "odds": "-115",
        "edge_pct": 0.08
    }
    
    Returns: bet_id
    """
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO bets 
            (user_id, game_id, player_name, stat_type, line, direction, odds, edge_pct, locked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            bet['game_id'],
            bet['player_name'],
            bet['stat_type'],
            bet['line'],
            bet['direction'],
            bet.get('odds', '-110'),
            bet.get('edge_pct', 0),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        return cursor.lastrowid

def get_user_bets(user_id: str, pending_only: bool = False) -> list[dict]:
    """Get all bets for a user."""
    with get_db() as conn:
        if pending_only:
            rows = conn.execute("""
                SELECT * FROM bets 
                WHERE user_id = ? AND result IS NULL
                ORDER BY locked_at DESC
            """, (user_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM bets WHERE user_id = ?
                ORDER BY locked_at DESC
            """, (user_id,)).fetchall()
        return [dict(row) for row in rows]

def get_user_record(user_id: str) -> dict:
    """Get user's betting record."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'PUSH' THEN 1 ELSE 0 END) as pushes
            FROM bets WHERE user_id = ? AND result IS NOT NULL
        """, (user_id,)).fetchone()
        
        if not row or row['total'] == 0:
            return {"total": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": 0}
        
        return {
            "total": row['total'],
            "wins": row['wins'],
            "losses": row['losses'],
            "pushes": row['pushes'],
            "win_rate": round(row['wins'] / (row['wins'] + row['losses']), 3) if (row['wins'] + row['losses']) > 0 else 0
        }
```

**Verification:**
```bash
python -c "
from api.queries import save_bet, get_user_bets, get_user_record
# Save a test bet
bet_id = save_bet('test-user-123', {
    'game_id': 'test-game',
    'player_name': 'Test Player',
    'stat_type': 'pts',
    'line': 20.5,
    'direction': 'OVER'
})
print(f'Saved bet with id: {bet_id}')
bets = get_user_bets('test-user-123')
print(f'User has {len(bets)} bets')
record = get_user_record('test-user-123')
print(f'Record: {record}')
print('✅ PASS')
"
```

---

### S2-T5: Implement User Queries

**Type:** Database CRUD
**Input:** Users table structure
**Output:** Updated `api/queries.py`

**Task Prompt for Sub-Agent:**
```
Add user operations to api/queries.py:

import uuid
from datetime import datetime

def create_user(email: str, auth_provider: str = 'google') -> str:
    """Create a new user, return user_id."""
    user_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO users (id, email, auth_provider, created_at, last_active_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, email, auth_provider, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
        conn.commit()
    return user_id

def get_user(email: str) -> Optional[dict]:
    """Get user by email."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

def get_or_create_user(email: str, auth_provider: str = 'google') -> dict:
    """Get existing user or create new one."""
    user = get_user(email)
    if user:
        return user
    user_id = create_user(email, auth_provider)
    return get_user(email)

def increment_message_count(user_id: str) -> int:
    """Increment user's message count, return new count."""
    with get_db() as conn:
        conn.execute("""
            UPDATE users 
            SET message_count = message_count + 1, 
                last_active_at = ?
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), user_id))
        conn.commit()
        row = conn.execute("SELECT message_count FROM users WHERE id = ?", (user_id,)).fetchone()
        return row['message_count'] if row else 0

def is_user_paid(user_id: str) -> bool:
    """Check if user has paid subscription."""
    with get_db() as conn:
        row = conn.execute("SELECT is_paid FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(row['is_paid']) if row else False

def set_user_paid(user_id: str, is_paid: bool = True):
    """Update user's paid status."""
    with get_db() as conn:
        conn.execute("UPDATE users SET is_paid = ? WHERE id = ?", (1 if is_paid else 0, user_id))
        conn.commit()
```

**Verification:**
```bash
python -c "
from api.queries import get_or_create_user, increment_message_count, is_user_paid
user = get_or_create_user('test@example.com')
print(f'User: {user[\"id\"]}')
count = increment_message_count(user['id'])
print(f'Message count: {count}')
paid = is_user_paid(user['id'])
print(f'Is paid: {paid}')
print('✅ PASS')
"
```

---

### S2-T6: Sprint 2 Integration Test

**Type:** End-to-End Test
**Input:** All S2 outputs
**Output:** Test pass

**Task Prompt for Sub-Agent:**
```
Create tests/test_sprint2.py:

1. test_probability_math(): Verify prob_over, prob_under, edge calculations
2. test_get_projection(): Verify we can fetch projections
3. test_get_best_props(): Verify returns sorted by edge
4. test_bet_lifecycle(): Save bet, retrieve, verify
5. test_user_lifecycle(): Create, increment, check paid status

Run: pytest tests/test_sprint2.py -v
```

**Verification:**
```bash
pytest tests/test_sprint2.py -v
```

---

## Success Criteria

Sprint 2 is COMPLETE when:
- [ ] All query functions work with test data
- [ ] Probability math is verified correct
- [ ] Can save and retrieve bets
- [ ] Can create and query users
- [ ] Integration tests pass

---

## Key Insight

After this sprint, the Chat interface (Sprint 3) just needs to call these functions. No more simulation latency — just database lookups.

```python
# Before (slow): 
result = engine.simulate_game("LAL", "BOS")  # 2-3 seconds

# After (instant):
proj = get_projection("LeBron James", "pts")  # <10ms
```

---

## Proceed to Sprint 3?

Run final verification:
```bash
./verify_sprint2.sh
```
