# The Brain - NBA Betting Co-Pilot

## What This Is
AI-powered NBA player prop betting co-pilot using Monte Carlo simulation + LLM reasoning.

---

## Current State: v2 COMPLETE ✅

All core components built and tested:

| Phase | Status | What's Done |
|-------|--------|-------------|
| 1. Data Pipeline | ✅ COMPLETE | 89K player-games, 3 seasons (2022-25) |
| 2. Model Training | ✅ COMPLETE | Game/Minutes/Stats XGBoost models |
| 3. Simulation Engine | ✅ COMPLETE | Monte Carlo + injury adjustments |
| 4. LLM Integration | ✅ COMPLETE | Claude Sonnet 4 + 7 tools |

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

### LLM Integration (`poc.py`)
Tools available:
- `simulate_prop` - Simulate any player prop bet
- `get_player_projection` - Full stat projections for a player
- `check_injuries` - ESPN scraper + user overrides
- `set_injury` - Manual injury input ("AD is out")
- `get_lineup` - Team roster via nba_api
- `get_tonight_games` - Today's schedule
- `build_parlay` - Correlated parlay analysis with thesis

---

## Running the POC

```bash
cd "/path/to/the-brain-organized 2"
export ANTHROPIC_API_KEY="your-key"
python poc.py
```

Commands:
- `quit` - Exit
- `clear` - Reset conversation
- `injuries` - Clear injury overrides

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
poc.py                      # LLM chat interface (working)
simulation/
  engine.py                 # Monte Carlo engine
  feature_transformer.py    # Live data fetching
  injury_adjustment.py      # Injury boost logic
  edge_calculator.py        # Odds conversion
  parlay_analyzer.py        # Correlation detection
  models.py                 # Data classes
models/
  game_model.pkl            # Team score model
  minutes_model.pkl         # Player minutes model
  stats_models.pkl          # 7 stat models
training_data_v2.csv        # 89K rows training data
```

---

## DO NOT

- Suggest switching APIs (SGO is LOCKED)
- Use >4 seasons of historical data
- Make LLM do math (engine does math, LLM explains)
- Build UI before core loop is validated
