# The Brain v2: Monte Carlo Game Simulation System

## Vision
Build a system that simulates 10,000 NBA games to predict player stats, with inputs for matchups, injuries, and context. Output feeds an LLM for reasoning.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  "What will LeBron score tonight? AD is questionable."      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM REASONING LAYER                      │
│  - Interprets user query                                    │
│  - Calls simulation API                                     │
│  - Explains results in natural language                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MONTE CARLO SIMULATION ENGINE                  │
│                                                             │
│  Inputs:                                                    │
│    - Team A vs Team B                                       │
│    - Home/Away                                              │
│    - Injury Report (who's out/questionable)                 │
│    - Date (for rest days calc)                              │
│                                                             │
│  Process:                                                   │
│    1. Predict game outcome distribution                     │
│    2. For each of 10,000 simulations:                       │
│       - Sample game score                                   │
│       - Distribute minutes to players                       │
│       - Distribute stats based on minutes + usage           │
│    3. Aggregate into player stat distributions              │
│                                                             │
│  Outputs:                                                   │
│    - Player stat predictions (mean, std, percentiles)       │
│    - Game outcome probabilities                             │
│    - Scenario comparisons ("if AD plays vs sits")           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINED MODELS                           │
│                                                             │
│  Model 1: GAME MODEL                                        │
│    Input: Team A features, Team B features, home/away       │
│    Output: Team A score, Team B score (with variance)       │
│                                                             │
│  Model 2: MINUTES MODEL                                     │
│    Input: Player features, game context, blowout prob       │
│    Output: Minutes played (with variance)                   │
│                                                             │
│  Model 3: STATS MODEL (per stat type)                       │
│    Input: Player features, predicted minutes, game pace     │
│    Output: PTS, REB, AST, etc. (with variance)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING DATA                            │
│                                                             │
│  Player-level: 89,030 rows (have)                           │
│  Team-level: 8,386 rows (have)                              │
│  Game-level: ~4,200 games (need to structure)               │
│  Advanced stats: usage rate, pace (need to pull)            │
└─────────────────────────────────────────────────────────────┘
```

---

# PHASE 1: DATA ENRICHMENT

## Goal
Add missing features to training data for better predictions.

## 1.1 Add Team Offensive Features (from existing data)

**Source:** `data/raw/team_game_logs.csv`

| Feature | Description | Computation |
|---------|-------------|-------------|
| `team_pts_scored_L10` | Team's scoring avg (last 10) | Rolling mean, shifted |
| `team_pace_proxy` | Possessions estimate | (team_pts + opp_pts) / 2 |
| `team_fg_pct_L10` | Team shooting % | Rolling mean |

## 1.2 Add Player Context Features (from existing data)

| Feature | Description | Computation |
|---------|-------------|-------------|
| `player_team_pts_share` | % of team's points | player_pts / team_pts |
| `player_usage_proxy` | Usage estimate | (FGA + TOV + FTA*0.44) / minutes |

## 1.3 Pull Advanced Stats (NEW API calls)

**Source:** `nba_api` endpoints

| Feature | Endpoint | Description |
|---------|----------|-------------|
| `usage_pct` | PlayerDashboard | Official usage rate |
| `pace` | TeamDashboard | Possessions per 48 min |
| `pie` | PlayerDashboard | Player Impact Estimate |
| `def_rating` | TeamDashboard | Defensive rating |
| `off_rating` | TeamDashboard | Offensive rating |

## 1.4 Create Game-Level Dataset (NEW structure)

Transform player logs into game-level data:

```
game_id | date | home_team | away_team | home_score | away_score |
home_pace | away_pace | home_off_rtg | away_off_rtg | home_def_rtg | away_def_rtg |
home_rest_days | away_rest_days | home_injuries | away_injuries
```

**Expected:** ~4,200 games across 3 seasons

---

# PHASE 2: MODEL TRAINING

## 2.1 Game Outcome Model

**Purpose:** Predict final scores for both teams

```
Input features:
  - home_off_rtg_L10, home_def_rtg_L10
  - away_off_rtg_L10, away_def_rtg_L10
  - home_pace_L10, away_pace_L10
  - home_rest_days, away_rest_days
  - historical_matchup_diff (optional)

Output:
  - home_score (mean + std)
  - away_score (mean + std)

Model: XGBoost Regressor (2 outputs)
Training rows: ~4,200 games
```

## 2.2 Player Minutes Model

**Purpose:** Predict minutes given game context

```
Input features:
  - player_min_L5_avg, player_min_szn_avg
  - is_starter (derived from min > 25)
  - predicted_game_total
  - predicted_margin (blowout = less star minutes)
  - rest_days

Output:
  - minutes (mean + std)

Model: XGBoost Regressor
Training rows: ~89,000 player-games
```

## 2.3 Player Stats Model (Multi-output)

**Purpose:** Predict stats given minutes and context

```
Input features:
  - predicted_minutes
  - player_pts_L5_avg, player_reb_L5_avg, etc.
  - player_usage_proxy
  - team_pace
  - opp_def_rtg

Output:
  - pts, reb, ast, stl, blk, tov, fg3m, ftm (means + stds)

Model: XGBoost MultiOutput OR separate models
Training rows: ~89,000 player-games
```

---

# PHASE 3: INJURY ADJUSTMENT MODULE

## Purpose
Redistribute minutes and usage when players are injured.

## Logic

```python
def adjust_for_injuries(team_roster, injury_report, models):

    # 1. Identify injured players
    injured = [p for p in team_roster if p in injury_report]
    healthy = [p for p in team_roster if p not in injury_report]

    # 2. Calculate missing production
    missing_minutes = sum(p.avg_minutes for p in injured)
    missing_usage = sum(p.usage_rate for p in injured)

    # 3. Redistribute to healthy players
    # Minutes redistribution (proportional to current minutes)
    for player in healthy:
        player.adjusted_minutes = player.avg_minutes + (
            missing_minutes * (player.avg_minutes / total_healthy_minutes)
        )

    # 4. Usage boost (stars get bigger boost)
    for player in healthy:
        usage_boost = missing_usage * (player.usage_rate / total_healthy_usage)
        player.adjusted_usage = player.usage_rate + usage_boost

    return healthy_with_adjustments
```

---

# PHASE 4: MONTE CARLO SIMULATION ENGINE

## Purpose
Generate 10,000 game simulations to produce stat distributions.

## Algorithm

```python
def simulate_game(team_a, team_b, home_team, injury_report, n_simulations=10000):

    results = []

    for sim in range(n_simulations):

        # 1. Sample game outcome from game model
        game_pred = game_model.predict(team_a, team_b, home_team)
        score_a = np.random.normal(game_pred.score_a_mean, game_pred.score_a_std)
        score_b = np.random.normal(game_pred.score_b_mean, game_pred.score_b_std)

        # 2. Determine game context
        margin = abs(score_a - score_b)
        is_blowout = margin > 15
        pace = (score_a + score_b) / 2  # Simplified

        # 3. Adjust rosters for injuries
        roster_a = adjust_for_injuries(team_a.roster, injury_report)
        roster_b = adjust_for_injuries(team_b.roster, injury_report)

        # 4. Predict minutes for each player
        for player in roster_a + roster_b:
            min_pred = minutes_model.predict(player, pace, is_blowout)
            player.sim_minutes = np.random.normal(min_pred.mean, min_pred.std)

        # 5. Predict stats given minutes
        for player in roster_a + roster_b:
            stats_pred = stats_model.predict(player, player.sim_minutes, pace)
            player.sim_stats = {
                'pts': np.random.normal(stats_pred.pts_mean, stats_pred.pts_std),
                'reb': np.random.normal(stats_pred.reb_mean, stats_pred.reb_std),
                'ast': np.random.normal(stats_pred.ast_mean, stats_pred.ast_std),
                # ... etc
            }

        results.append({
            'score_a': score_a,
            'score_b': score_b,
            'player_stats': {p.id: p.sim_stats for p in roster_a + roster_b}
        })

    # 6. Aggregate results
    return aggregate_simulations(results)
```

## Output Format

```json
{
  "game": {
    "team_a": "LAL",
    "team_b": "BOS",
    "simulations": 10000
  },
  "team_a_score": {
    "mean": 112.4,
    "std": 8.2,
    "p10": 101,
    "p50": 112,
    "p90": 123
  },
  "players": {
    "lebron_james": {
      "pts": {"mean": 26.3, "std": 5.1, "p10": 19, "p50": 26, "p90": 33},
      "reb": {"mean": 7.8, "std": 2.3, "p10": 5, "p50": 8, "p90": 11},
      "ast": {"mean": 8.1, "std": 2.5, "p10": 5, "p50": 8, "p90": 11}
    }
  },
  "scenarios": {
    "if_ad_out": {
      "lebron_pts": {"mean": 29.8, "std": 5.5}
    }
  }
}
```

---

# PHASE 5: LLM INTEGRATION

## Purpose
Natural language interface for simulation system.

## Flow

```
User: "What do you think LeBron scores tonight against Boston?
       AD is listed as questionable."

LLM:
  1. Parse query → extract: player=LeBron, opponent=BOS, injury=AD questionable
  2. Call simulation API twice:
     - Scenario A: AD plays
     - Scenario B: AD sits
  3. Analyze results
  4. Generate response:

"Based on 10,000 simulated games:

**If AD plays:**
LeBron projects for 25.8 points (range: 18-34)
- 70% chance he scores 22+
- 25% chance he scores 30+

**If AD sits:**
LeBron projects for 29.2 points (range: 21-38)
- His usage increases from 31% to 37%
- 85% chance he scores 22+
- 40% chance he scores 30+

The 3.4 point boost when AD sits is significant. If you're
betting the over on a 27.5 line, you want AD OUT."
```

---

# EXECUTION PLAN

## Phase 1: Data Enrichment (Next Step)

| Task | Description | Time Est |
|------|-------------|----------|
| 1.1 | Add team offensive features from existing data | 30 min |
| 1.2 | Add player context features | 30 min |
| 1.3 | Pull advanced stats (usage, pace) from nba_api | 1 hr |
| 1.4 | Create game-level dataset | 1 hr |

**Output:** Enhanced training datasets ready for modeling

## Phase 2: Model Training

| Task | Description | Time Est |
|------|-------------|----------|
| 2.1 | Train game outcome model | 1 hr |
| 2.2 | Train player minutes model | 1 hr |
| 2.3 | Train player stats models | 2 hr |
| 2.4 | Evaluate and tune | 2 hr |

**Output:** Trained models saved to `models/`

## Phase 3: Simulation Engine

| Task | Description | Time Est |
|------|-------------|----------|
| 3.1 | Build injury adjustment module | 1 hr |
| 3.2 | Build Monte Carlo engine | 2 hr |
| 3.3 | Build aggregation/output formatting | 1 hr |

**Output:** `simulate_game()` function that returns distributions

## Phase 4: Integration

| Task | Description | Time Est |
|------|-------------|----------|
| 4.1 | Create simulation API | 1 hr |
| 4.2 | Build LLM prompt templates | 1 hr |
| 4.3 | End-to-end testing | 2 hr |

**Output:** Working system that answers natural language queries

---

# FILES TO CREATE

```
the-brain-organized 2/
├── training_data_v2.csv          # EXISTS - player data
├── data/
│   ├── raw/
│   │   ├── player_game_logs.csv  # EXISTS
│   │   ├── team_game_logs.csv    # EXISTS
│   │   └── advanced_stats.csv    # NEW - usage, pace
│   └── processed/
│       ├── player_features.csv   # NEW - enriched player data
│       ├── team_features.csv     # NEW - enriched team data
│       └── game_features.csv     # NEW - game-level data
├── models/
│   ├── game_model.pkl            # NEW
│   ├── minutes_model.pkl         # NEW
│   └── stats_model.pkl           # NEW
├── scripts/
│   ├── 1_enrich_data.py          # NEW - Phase 1
│   ├── 2_train_models.py         # NEW - Phase 2
│   └── 3_simulate.py             # NEW - Phase 3
└── api/
    └── simulation_api.py         # NEW - Phase 4
```

---

# READY TO EXECUTE

**Next step:** Phase 1.1 - Add team offensive features to training data

Approve to begin?
