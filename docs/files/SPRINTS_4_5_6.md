# SPRINT 4: CLOSING THE LOOP (Results Grading)

**Goal:** Grade bets and show results.
**Duration:** 2-3 days
**Blocker Risk:** Low — similar to factory

---

## Sprint Summary

| Task ID | Task | Output |
|---------|------|--------|
| S4-T1 | Implement fetch_actuals | `batch/grade_bets.py` |
| S4-T2 | Implement grade_bets | `batch/grade_bets.py` |
| S4-T3 | Display user record in sidebar | `chat/app.py` |
| S4-T4 | Integration test | Test pass |

---

## Key Files from nba_api

Use `boxscoretraditionalv3` to get actual player stats after game completes:

```python
from nba_api.stats.endpoints import boxscoretraditionalv3

def fetch_actuals(game_id: str) -> dict:
    """Get actual stats from completed game."""
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    players = box.player_stats.get_data_frame()
    
    # Map to our stat names
    stat_map = {
        'pts': 'points',
        'reb': 'reboundsTotal', 
        'ast': 'assists',
        'stl': 'steals',
        'blk': 'blocks',
        'tov': 'turnovers',
        'fg3m': 'threePointersMade'
    }
    
    results = {}
    for _, player in players.iterrows():
        name = f"{player['firstName']} {player['familyName']}"
        results[name] = {
            stat: player[api_field] for stat, api_field in stat_map.items()
        }
    return results
```

---

## Grading Logic

```python
def grade_bets(game_id: str):
    """Grade all bets for a completed game."""
    actuals = fetch_actuals(game_id)
    
    with get_db() as conn:
        bets = conn.execute("""
            SELECT * FROM bets 
            WHERE game_id = ? AND result IS NULL
        """, (game_id,)).fetchall()
        
        for bet in bets:
            player_stats = actuals.get(bet['player_name'])
            if not player_stats:
                continue  # Player didn't play
            
            actual = player_stats[bet['stat_type']]
            line = bet['line']
            direction = bet['direction']
            
            # Determine result
            if actual == line:
                result = 'PUSH'
            elif direction == 'OVER' and actual > line:
                result = 'WIN'
            elif direction == 'UNDER' and actual < line:
                result = 'WIN'
            else:
                result = 'LOSS'
            
            # Update bet
            conn.execute("""
                UPDATE bets 
                SET result = ?, actual_value = ?, graded_at = ?
                WHERE id = ?
            """, (result, actual, datetime.utcnow().isoformat(), bet['id']))
        
        conn.commit()
```

---

## Success Criteria

- [ ] Morning after games, run `python batch/grade_bets.py`
- [ ] All bets have result (WIN/LOSS/PUSH)
- [ ] User record shows in sidebar (e.g., "5-3 (62.5%)")

---

# SPRINT 5: HOUSE BOTS (Marketing Track Record)

**Goal:** Automated bettors that build your public track record.
**Duration:** 2-3 days
**Blocker Risk:** Low — uses existing pieces

---

## Sprint Summary

| Task ID | Task | Output |
|---------|------|--------|
| S5-T1 | Create bot strategies | `bots/strategies.py` |
| S5-T2 | Implement bot runner | `bots/runner.py` |
| S5-T3 | Add bot grading to batch | `batch/grade_bets.py` |
| S5-T4 | Display leaderboard | `chat/app.py` |

---

## Bot Strategies

```python
# bots/strategies.py

def aggressive_strategy(props: list[dict]) -> list[dict]:
    """Picks props with edge >= 8%"""
    return [p for p in props if p['edge'] >= 0.08]

def conservative_strategy(props: list[dict]) -> list[dict]:
    """Picks only the top 2 props with edge >= 10%"""
    filtered = [p for p in props if p['edge'] >= 0.10]
    return sorted(filtered, key=lambda x: x['edge'], reverse=True)[:2]

def balanced_strategy(props: list[dict]) -> list[dict]:
    """Picks top 5 props with edge >= 5%"""
    filtered = [p for p in props if p['edge'] >= 0.05]
    return sorted(filtered, key=lambda x: x['edge'], reverse=True)[:5]

STRATEGIES = {
    'aggressive': aggressive_strategy,
    'conservative': conservative_strategy,
    'balanced': balanced_strategy
}
```

---

## Bot Runner (runs after precompute)

```python
# bots/runner.py

def run_bots():
    """Have each bot place their picks."""
    from api.queries import get_best_props, save_bet, get_db
    from bots.strategies import STRATEGIES
    
    props = get_best_props(min_edge=0.03, limit=20)
    
    with get_db() as conn:
        bots = conn.execute("SELECT * FROM house_bots WHERE is_alive = 1").fetchall()
        
        for bot in bots:
            strategy = STRATEGIES.get(bot['strategy'], balanced_strategy)
            picks = strategy(props)
            
            for pick in picks:
                save_bet(bot['id'], pick)
            
            print(f"{bot['name']} locked {len(picks)} bets")
```

---

## Bankroll Updates (after grading)

```python
def update_bot_bankrolls():
    """Update bot bankrolls based on results."""
    with get_db() as conn:
        bots = conn.execute("SELECT * FROM house_bots").fetchall()
        
        for bot in bots:
            # Calculate profit/loss from today's graded bets
            row = conn.execute("""
                SELECT 
                    SUM(CASE WHEN result = 'WIN' THEN 100 ELSE 0 END) as winnings,
                    SUM(CASE WHEN result = 'LOSS' THEN 110 ELSE 0 END) as losses
                FROM bets 
                WHERE user_id = ? 
                AND DATE(graded_at) = DATE('now')
            """, (bot['id'],)).fetchone()
            
            net = (row['winnings'] or 0) - (row['losses'] or 0)
            new_bankroll = bot['bankroll'] + net
            
            # Check if bot died (bankroll <= 0)
            is_alive = 1 if new_bankroll > 0 else 0
            
            conn.execute("""
                UPDATE house_bots 
                SET bankroll = ?, is_alive = ?
                WHERE id = ?
            """, (max(new_bankroll, 0), is_alive, bot['id']))
        
        conn.commit()
```

---

## Responsible Gambling: Dead Bots

From your Sprint Plan:
> "If bankroll hits $0, bot 'dies' (responsible gambling messaging)"

Display dead bots as cautionary tales:
```python
# In leaderboard display:
dead_bots = [b for b in bots if not b['is_alive']]
if dead_bots:
    st.warning("☠️ These bots went broke:")
    for bot in dead_bots:
        st.caption(f"{bot['name']} - Started with ${bot['starting_bankroll']}, busted after X bets")
```

---

## Success Criteria

- [ ] 3 bots created with different strategies
- [ ] Bots automatically lock picks after precompute
- [ ] Bots get graded with users
- [ ] Leaderboard shows bot records
- [ ] Dead bots display as warnings

---

# SPRINT 6: MULTI-USER AUTH

**Goal:** Real users with accounts and limits.
**Duration:** 3-4 days
**Blocker Risk:** Medium — OAuth setup

---

## Sprint Summary

| Task ID | Task | Output |
|---------|------|--------|
| S6-T1 | Add Google OAuth | `auth/oauth.py` |
| S6-T2 | Wire auth to Streamlit | `chat/app.py` |
| S6-T3 | Implement free tier limit | `chat/app.py` |
| S6-T4 | Add upgrade prompt | `chat/app.py` |
| S6-T5 | Integration test | Test pass |

---

## 🔲 HUMAN DECISIONS NEEDED

Before executing Sprint 6, decide:

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Free tier | 3 per day / 3 per week / 3 ever | **3 ever** (trial-style) |
| Price | $5 / $10 / $15 / $20 / month | **$15/month** (can lower later) |
| Payment | Stripe / Gumroad / Manual | Start with **manual** for v1 |

---

## Google OAuth with Streamlit

Use `streamlit-authenticator` or simple OAuth flow:

```python
# auth/oauth.py
import streamlit as st
from google.oauth2 import id_token
from google.auth.transport import requests

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

def verify_google_token(token: str) -> dict:
    """Verify Google ID token and return user info."""
    idinfo = id_token.verify_oauth2_token(
        token, requests.Request(), GOOGLE_CLIENT_ID
    )
    return {
        "email": idinfo['email'],
        "name": idinfo.get('name', ''),
        "picture": idinfo.get('picture', '')
    }
```

Alternatively, use Streamlit's built-in secrets + components:
```python
# Simpler approach for v1:
st.sidebar.text_input("Email (for tracking)")
```

---

## Free Tier Limit

```python
# In chat/app.py, before processing message:

from api.queries import get_user, increment_message_count, is_user_paid

FREE_LIMIT = 3

if st.session_state.user_id:
    user = get_user(st.session_state.user_id)
    
    if not is_user_paid(user['id']) and user['message_count'] >= FREE_LIMIT:
        st.error("🚫 Free trial complete!")
        st.info("""
        You've used all 3 free messages. 
        
        **Upgrade to Pro ($15/month):**
        - Unlimited prop analysis
        - Full bet tracking
        - Priority support
        
        [Upgrade Now](https://your-payment-link.com)
        """)
        st.stop()
    
    # Track this message
    increment_message_count(user['id'])
```

---

## Success Criteria

- [ ] Users can sign in with Google
- [ ] Free users hit limit after 3 messages
- [ ] Paid users have unlimited access
- [ ] Upgrade prompt shows for free users at limit
- [ ] Multiple users can use simultaneously

---

## Post-Sprint 6: You Have a Product!

After Sprint 6:
- ✅ Database with projections
- ✅ Pre-computed daily simulations
- ✅ Chat interface with tools
- ✅ Bet tracking + grading
- ✅ Bot leaderboards
- ✅ User auth + paywall

**What's NOT built (v2):**
- Real money tracking
- Platform integrations (DraftKings API)
- Mobile app
- Multiple sports
- Social features
