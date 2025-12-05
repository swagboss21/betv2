# The Brain - Development Timeline

## Dec 5, 2025

### Session: Model v1.1 Training + Inference Pipeline

**Trained Model v1.1** with improvements:
- Filtered out team totals (player_id = "home"/"away")
- Added `class_weight='balanced'` to fix under-bias
- Added `player_id` as categorical feature (279 qualified players)
- Filtered to players with 50+ props

**Results - v1.0 vs v1.1**:
| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| Accuracy | 61.76% | 57.65% | -4.11pp |
| Over Recall | 14.27% | 65.39% | **+51.12pp** |
| Under Recall | 92.56% | 52.63% | -39.93pp |

**Trade-off**: v1.1 catches 4.5x more overs but at cost of overall accuracy.

**Built Inference Pipeline** (`scripts/inference.py`):
- Pulls tonight's NBA props from SGO API
- Runs predictions through both v1.0 and v1.1
- Outputs ranked list (agreement first, then confidence)
- Saves to `predictions/YYYY-MM-DD.csv`

**First Live Test**: 6 games, 1,939 props predicted

**Files Created**:
- `scripts/3_train_model_v1.1.py`
- `scripts/inference.py`
- `models/logistic_v1.1.pkl`
- `predictions/2025-12-04.csv`

---

## Dec 3, 2025

### Session 1: Data Analysis & Name Matching Fix
- Analyzed 1,264 props from Dec 2 games
- Fixed hyphenated name matching (Caldwell-Pope, Karl-Anthony Towns, Gilgeous-Alexander)
- Added player alias mapping (Q Post → Quinten Post, T Vukcevic → Tristan Vukcevic)
- **Result**: 1,149 matched (90.9%), up from 1,095 (86.6%)
- Remaining unlabeled: 81 DEN@IND (not played), 34 DNP players

### Session 2: Historical Data Discovery
**Question**: Can we pull historical SGO data instead of daily pulls?

**Investigation**:
1. Found SGO v2 API supports `finalized=true` parameter
2. Finalized events include:
   - `score` - actual player stat result
   - `closeBookOverUnder` - closing line
   - `bookOverUnder` - opening line
3. Tested with 1 finalized event (LAC@MIA)

**Key Finding - Zero Line Movement**:
```
Total player props analyzed: 311
Zero movement:    311 (100.0%)
Average movement: 0.000 points
```

### Session 3: API Usage Research & Capacity Planning

**Question**: How many games can we pull on the free tier?

**Discovery - 1 Entity = 1 Game!**
```
Test: Fetched 2 events with 2,031 total odds
Result: Used only 2 entities (not 2,031!)
```

**Current Usage**:
| Metric | Value |
|--------|-------|
| Tier | amateur (free) |
| Monthly limit | 2,500 entities |
| Current usage | 566 entities |
| Remaining | 1,934 entities |

**Capacity Calculation**:
- 2024-25 season: ~400 games
- 2023-24 season: ~1,230 games
- Total needed: ~1,630 games
- Available: 1,934 games
- **Both seasons fit!** ✅

**Created**:
- `scripts/fetch_historical_sgo.py` - ready to pull both seasons
- Verified with `--dry-run` flag

**Ready to Execute**:
```bash
python3 scripts/fetch_historical_sgo.py --season both
```

This will create ~160K labeled training rows.

---

## Earlier History

### Nov 2024
- Initial chatbot testing (archived)

### Dec 2, 2025
- First SGO API pull
- Created JSON → CSV transformation
- Created labeling pipeline with nba_api
