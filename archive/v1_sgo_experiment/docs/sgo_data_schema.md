# SGO NBA Props Data Schema

Reference documentation for SportsGameOdds API data transformation.

---

## Files

| File | Location | Description |
|------|----------|-------------|
| Raw JSON | `data/raw/nba_all_YYYYMMDD_HHMMSS.json` | ~6MB per pull |
| Processed CSV | `data/processed/props_YYYYMMDD.csv` | ~1,200 rows per 7-game slate |

---

## JSON Structure (SGO API Response)

```
{
  "success": true,
  "data": [
    {
      "eventID": "l0kBhXg8nNwFPqqLTSLL",      // Unique game ID
      "sportID": "BASKETBALL",
      "leagueID": "NBA",
      "status": {
        "startsAt": "2025-12-03T01:00:00.000Z",
        "started": false,
        "completed": false
      },
      "teams": {
        "away": { "names": { "short": "MEM" } },
        "home": { "names": { "short": "SAS" } }
      },
      "players": {
        "CAM_SPENCER_1_NBA": { "name": "Cam Spencer", "teamID": "..." }
      },
      "odds": {
        "{stat}-{playerID}-game-ou-over": { ... },
        "{stat}-{playerID}-game-ou-under": { ... }
      },
      "results": {
        "game": { ... }  // Team-level only, NO player stats
      }
    }
  ]
}
```

### Odds Object Structure

```
odds["assists-CAM_SPENCER_1_NBA-game-ou-over"] = {
  "oddID": "assists-CAM_SPENCER_1_NBA-game-ou-over",
  "playerID": "CAM_SPENCER_1_NBA",
  "statID": "assists",
  "sideID": "over",
  "fairOdds": "+100",           // Market-derived (use for fair_odds column)
  "bookOdds": "-156",           // Aggregated from books (use for over_odds)
  "fairOverUnder": "4",         // Market line
  "bookOverUnder": "3.5"        // Book line (use for line column)
}
```

---

## CSV Schema

| Column | Type | Source | Example |
|--------|------|--------|---------|
| `date` | Date | `status.startsAt` parsed | `2025-12-03` |
| `player` | String | `players[id].name` | `Cam Spencer` |
| `player_id` | String | `odds[].playerID` | `CAM_SPENCER_1_NBA` |
| `game` | String | `away@home` | `MEM@SAS` |
| `stat_type` | String | `odds[].statID` | `assists` |
| `line` | Float | `odds[].bookOverUnder` | `3.5` |
| `over_odds` | String | `odds[...over].bookOdds` | `-156` |
| `under_odds` | String | `odds[...under].bookOdds` | `+118` |
| `over_implied` | Float | Calculated | `0.6094` |
| `under_implied` | Float | Calculated | `0.4587` |
| `fair_odds` | String | `odds[].fairOdds` | `+100` |
| `game_id` | String | `eventID` | `l0kBhXg8nNwFPqqLTSLL` |
| `starts_at` | ISO8601 | `status.startsAt` | `2025-12-03T01:00:00.000Z` |

### Columns to Add (After Results)

| Column | Type | Description |
|--------|------|-------------|
| `actual` | Float | Player's actual stat |
| `hit` | Int | 1 if actual > line, else 0 |
| `margin` | Float | actual - line |

---

## Stat Types (12 Supported)

| stat_type | NBA API V3 Field | Calculation |
|-----------|------------------|-------------|
| `points` | `points` | Direct |
| `assists` | `assists` | Direct |
| `rebounds` | `reboundsTotal` | Direct |
| `blocks` | `blocks` | Direct |
| `steals` | `steals` | Direct |
| `turnovers` | `turnovers` | Direct |
| `threePointersMade` | `threePointersMade` | Direct |
| `points+assists` | `points + assists` | Sum |
| `points+rebounds` | `points + reboundsTotal` | Sum |
| `points+rebounds+assists` | `points + reboundsTotal + assists` | Sum |
| `rebounds+assists` | `reboundsTotal + assists` | Sum |
| `blocks+steals` | `blocks + steals` | Sum |

---

## Transformation Logic

### Implied Probability

```python
def american_to_implied(odds_str):
    odds = float(odds_str.replace('+', ''))
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return abs(odds) / (abs(odds) + 100)

# -156 → 0.6094 (60.94%)
# +118 → 0.4587 (45.87%)
```

### Odds Key Pattern

```
"{statID}-{playerID}-{periodID}-{betTypeID}-{sideID}"
```

Example: `assists-CAM_SPENCER_1_NBA-game-ou-over`

### Row Consolidation

JSON has **2 objects** per prop (over + under).
CSV consolidates to **1 row** with both `over_odds` and `under_odds`.

---

## Primary Key

```
(game_id, player_id, stat_type) → UNIQUE
```

Use this to join props to results.

---

## Player ID Matching

**SGO format**: `{FIRSTNAME}_{LASTNAME}_{N}_NBA`
- Example: `CAM_SPENCER_1_NBA`

**NBA API**: Uses numeric `PLAYER_ID` and `PLAYER_NAME`

**Match strategy**: Match by player name + game date + team

---

## nba_api Reference

**IMPORTANT:** Use V3 endpoints (V2 returns empty player stats).

### Get Games by Date
```python
from nba_api.stats.endpoints import scoreboardv2

games = scoreboardv2.ScoreboardV2(
    game_date='2025-12-02',
    league_id='00'
)
game_header = games.game_header.get_data_frame()
# Returns: GAME_ID, GAMECODE (e.g., "20251202/MEMSAS"), etc.
```

### Get Box Score for Game (USE V3!)
```python
from nba_api.stats.endpoints import boxscoretraditionalv3

box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id='0022500322')
players = box.player_stats.get_data_frame()
# Returns: firstName, familyName, points, assists, reboundsTotal, etc.
```

### Field Mapping (nba_api V3 → CSV)
| nba_api V3 Field | CSV stat_type |
|------------------|---------------|
| `points` | `points` |
| `assists` | `assists` |
| `reboundsTotal` | `rebounds` |
| `steals` | `steals` |
| `blocks` | `blocks` |
| `turnovers` | `turnovers` |
| `threePointersMade` | `threePointersMade` |

### Player Name (V3)
```python
# V3 splits name into firstName + familyName
player_name = f"{player['firstName']} {player['familyName']}"
```

### Matching Flow
```
1. ScoreboardV2(game_date) → Get NBA game_ids + GAMECODE
2. Parse GAMECODE to get team codes (e.g., "MEMSAS" → MEM@SAS)
3. BoxScoreTraditionalV3(game_id) → Get player stats
4. Match players by normalized name (lowercase, no punctuation)
5. Calculate: actual, hit, margin
```

### Scripts
```bash
# Step 1: Fetch + cache box scores
python scripts/fetch_nba_results.py 2025-12-02
# Output: data/raw/nba_boxscores_20251202.json

# Step 2: Label props from cache
python scripts/label_results.py data/processed/props_fullgame_20251202.csv
# Output: data/processed/props_fullgame_20251202_labeled.csv
```

---

## Critical Notes

1. **JSON has NO player-level results** - must use external source (nba_api)
2. **SGO game_id ≠ NBA game_id** - match by date + teams via GAMECODE
3. **Book odds vs Fair odds** - CSV uses book odds for over/under, fair odds separate
4. **Line uses bookOverUnder** - not fairOverUnder
5. **nba_api rate limiting** - Scripts use 0.6s delays between requests
6. **USE V3 ENDPOINTS** - BoxScoreTraditionalV2 returns empty, V3 works
7. **UTC vs ET dates** - SGO uses UTC, NBA API uses ET. Props dated "Dec 3 UTC" = "Dec 2 ET" games

---

## Name Matching (RESOLVED Dec 3, 2025)

### Issues Fixed

| Issue | Players | Fix |
|-------|---------|-----|
| Hyphenated names | Caldwell-Pope, Karl-Anthony Towns, Gilgeous-Alexander | Normalize removes ALL spaces & hyphens |
| Abbreviated names | Q Post → Quinten Post, T Vukcevic → Tristan Vukcevic | PLAYER_ALIASES mapping |

### Remaining Unlabeled (115 total)

| Issue | Count | Notes |
|-------|-------|-------|
| Game not played (DEN@IND) | 81 | Fetch after Dec 3 ET game |
| Player DNP | 34 | Zion Williamson, Quentin Grimes, Yves Missi didn't play |

### Name Normalization Logic
```python
# Removes ALL punctuation, hyphens, and spaces
# "Kentavious Caldwell-Pope" → "kentaviouscaldwellpope"
# "Kentavious Caldwellpope" → "kentaviouscaldwellpope"
# MATCH!
```
