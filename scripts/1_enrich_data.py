"""
Phase 1: Data Enrichment for NBA Betting Model

Processes raw player and team game logs to create enriched feature sets.
Implements strict anti-leakage protocols using .shift(1) for all rolling calculations.

Inputs:
    - data/raw/player_game_logs.csv
    - data/raw/team_game_logs.csv

Outputs:
    - data/processed/player_features_enriched.csv
    - data/processed/game_features.csv
    - data/processed/team_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set pandas display options for better debugging
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Define paths
BASE_DIR = Path("/Users/noahcantu/Desktop/the-brain-organized 2")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create output directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 1: DATA ENRICHMENT")
print("=" * 80)


# ==============================================================================
# STEP 1: LOAD RAW DATA
# ==============================================================================

print("\n[1/6] Loading raw data...")

player_df = pd.read_csv(RAW_DIR / "player_game_logs.csv")
team_df = pd.read_csv(RAW_DIR / "team_game_logs.csv")

print(f"  - Loaded {len(player_df):,} player-game rows")
print(f"  - Loaded {len(team_df):,} team-game rows")

# Convert GAME_DATE to datetime
player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])

# Sort by date (critical for rolling calculations)
player_df = player_df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
team_df = team_df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)


# ==============================================================================
# STEP 2: ENGINEER TEAM FEATURES
# ==============================================================================

print("\n[2/6] Engineering team features...")

def parse_matchup(row):
    """
    Parse MATCHUP column to extract opponent and home/away status.
    Format: "LAL vs. GSW" (LAL home) or "LAL @ GSW" (LAL away)
    """
    matchup = row['MATCHUP']
    if 'vs.' in matchup:
        is_home = 1
        opponent = matchup.split('vs.')[1].strip()
    elif '@' in matchup:
        is_home = 0
        opponent = matchup.split('@')[1].strip()
    else:
        is_home = None
        opponent = None
    return pd.Series({'is_home': is_home, 'opponent_abbr': opponent})

# Add home/away and opponent to team data
team_df[['is_home', 'opponent_abbr']] = team_df.apply(parse_matchup, axis=1)

# Calculate rest days for teams
team_df['team_rest_days'] = team_df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days - 1
team_df['team_rest_days'] = team_df['team_rest_days'].fillna(0)  # First game of season

# Calculate opponent points (for defensive rating)
# We'll need to join with opponent's actual score
# Create a mapping of game_id + team -> opponent points
team_scores = team_df[['GAME_ID', 'TEAM_ID', 'PTS']].copy()
team_scores.columns = ['GAME_ID', 'opponent_team_id', 'opp_pts']

# For each team game, find the opponent's points
# First, create a mapping from team_abbr to team_id
team_abbr_to_id = team_df[['TEAM_ABBREVIATION', 'TEAM_ID']].drop_duplicates()
abbr_to_id_dict = dict(zip(team_abbr_to_id['TEAM_ABBREVIATION'], team_abbr_to_id['TEAM_ID']))

# Map opponent abbreviation to opponent ID
team_df['opponent_id'] = team_df['opponent_abbr'].map(abbr_to_id_dict)

# Join to get opponent's points
team_df = team_df.merge(
    team_scores,
    left_on=['GAME_ID', 'opponent_id'],
    right_on=['GAME_ID', 'opponent_team_id'],
    how='left'
)
team_df = team_df.drop('opponent_team_id', axis=1)

# Calculate pace (approximation: (team_pts + opp_pts) / 2)
team_df['pace'] = (team_df['PTS'] + team_df['opp_pts']) / 2

# Calculate rolling features (SHIFTED by 1 to prevent leakage)
print("  - Calculating rolling team features (L10, shifted)...")

team_rolling_features = []

for team_id in team_df['TEAM_ID'].unique():
    team_games = team_df[team_df['TEAM_ID'] == team_id].copy()

    # Rolling 10-game averages (shifted by 1)
    team_games['team_pts_L10'] = team_games['PTS'].shift(1).rolling(10, min_periods=1).mean()
    team_games['team_opp_pts_L10'] = team_games['opp_pts'].shift(1).rolling(10, min_periods=1).mean()
    team_games['team_pace_L10'] = team_games['pace'].shift(1).rolling(10, min_periods=1).mean()
    team_games['team_fg_pct_L10'] = team_games['FG_PCT'].shift(1).rolling(10, min_periods=1).mean()

    team_rolling_features.append(team_games)

team_df = pd.concat(team_rolling_features, ignore_index=True)

print(f"  - Created team features for {team_df['TEAM_ID'].nunique()} teams")


# ==============================================================================
# STEP 3: CREATE GAME-LEVEL DATASET
# ==============================================================================

print("\n[3/6] Creating game-level dataset...")

# Each game appears twice in team_df (once for each team)
# Create a deduplicated game dataset with home/away split

games_list = []

for game_id in team_df['GAME_ID'].unique():
    game_data = team_df[team_df['GAME_ID'] == game_id]

    if len(game_data) != 2:
        # Some games might have issues, skip
        continue

    home_team = game_data[game_data['is_home'] == 1]
    away_team = game_data[game_data['is_home'] == 0]

    if len(home_team) == 0 or len(away_team) == 0:
        continue

    home_team = home_team.iloc[0]
    away_team = away_team.iloc[0]

    game_row = {
        'game_id': game_id,
        'game_date': home_team['GAME_DATE'],
        'season': home_team['SEASON_YEAR'],
        'home_team_id': home_team['TEAM_ID'],
        'away_team_id': away_team['TEAM_ID'],
        'home_team_abbr': home_team['TEAM_ABBREVIATION'],
        'away_team_abbr': away_team['TEAM_ABBREVIATION'],
        'home_pts': home_team['PTS'],
        'away_pts': away_team['PTS'],
        'home_pts_L10': home_team['team_pts_L10'],
        'away_pts_L10': away_team['team_pts_L10'],
        'home_opp_pts_L10': home_team['team_opp_pts_L10'],
        'away_opp_pts_L10': away_team['team_opp_pts_L10'],
        'home_pace_L10': home_team['team_pace_L10'],
        'away_pace_L10': away_team['team_pace_L10'],
        'home_fg_pct_L10': home_team['team_fg_pct_L10'],
        'away_fg_pct_L10': away_team['team_fg_pct_L10'],
        'home_rest_days': home_team['team_rest_days'],
        'away_rest_days': away_team['team_rest_days'],
    }

    games_list.append(game_row)

game_df = pd.DataFrame(games_list)
game_df = game_df.sort_values('game_date').reset_index(drop=True)

print(f"  - Created {len(game_df):,} unique games")


# ==============================================================================
# STEP 4: SAVE TEAM FEATURES
# ==============================================================================

print("\n[4/6] Saving team features...")

team_features_df = team_df[[
    'GAME_ID', 'GAME_DATE', 'SEASON_YEAR', 'TEAM_ID', 'TEAM_ABBREVIATION',
    'opponent_id', 'opponent_abbr', 'is_home', 'PTS', 'opp_pts',
    'team_pts_L10', 'team_opp_pts_L10', 'team_pace_L10', 'team_fg_pct_L10',
    'team_rest_days'
]].copy()

team_features_df.columns = [
    'game_id', 'game_date', 'season', 'team_id', 'team_abbr',
    'opponent_id', 'opponent_abbr', 'is_home', 'team_pts', 'opp_pts',
    'team_pts_L10', 'team_opp_pts_L10', 'team_pace_L10', 'team_fg_pct_L10',
    'team_rest_days'
]

team_features_df.to_csv(PROCESSED_DIR / "team_features.csv", index=False)
print(f"  - Saved team_features.csv ({len(team_features_df):,} rows)")


# ==============================================================================
# STEP 5: ENGINEER PLAYER FEATURES
# ==============================================================================

print("\n[5/6] Engineering player features...")

# Parse matchup for players
player_df[['is_home', 'opponent_abbr']] = player_df.apply(parse_matchup, axis=1)
player_df['opponent_id'] = player_df['opponent_abbr'].map(abbr_to_id_dict)

# Calculate rest days for players
player_df['rest_days'] = player_df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days - 1
player_df['rest_days'] = player_df['rest_days'].fillna(0)

# Get team points for each game (for pts_share calculation)
# Join with team_df to get team's total points
team_pts_lookup = team_df[['GAME_ID', 'TEAM_ID', 'PTS']].copy()
team_pts_lookup.columns = ['GAME_ID', 'TEAM_ID', 'team_pts']

player_df = player_df.merge(
    team_pts_lookup,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Calculate usage proxy: (FGA + 0.44*FTA + TOV) / MIN
# Handle cases where MIN is 0
player_df['usage_calc'] = (
    player_df['FGA'] + 0.44 * player_df['FTA'] + player_df['TOV']
) / player_df['MIN'].replace(0, np.nan)

# Calculate pts share: player_pts / team_pts
player_df['pts_share_calc'] = player_df['PTS'] / player_df['team_pts'].replace(0, np.nan)

print("  - Calculating rolling player features (L5, shifted)...")

# Stats to calculate rolling features for
rolling_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'MIN']

player_rolling_features = []

for player_id in player_df['PLAYER_ID'].unique():
    player_games = player_df[player_df['PLAYER_ID'] == player_id].copy()

    # Sort by date
    player_games = player_games.sort_values('GAME_DATE')

    # Rolling 5-game features (shifted by 1)
    for stat in rolling_stats:
        stat_lower = stat.lower()
        player_games[f'{stat_lower}_L5_avg'] = player_games[stat].shift(1).rolling(5, min_periods=1).mean()
        player_games[f'{stat_lower}_L5_std'] = player_games[stat].shift(1).rolling(5, min_periods=1).std()

    # Usage proxy and pts share rolling averages
    player_games['player_usage_proxy'] = player_games['usage_calc'].shift(1).rolling(5, min_periods=1).mean()
    player_games['player_team_pts_share'] = player_games['pts_share_calc'].shift(1).rolling(5, min_periods=1).mean()

    # Season averages (expanding mean, shifted)
    for stat in rolling_stats:
        stat_lower = stat.lower()
        # Group by season and calculate expanding mean
        for season in player_games['SEASON_YEAR'].unique():
            season_mask = player_games['SEASON_YEAR'] == season
            player_games.loc[season_mask, f'{stat_lower}_szn_avg'] = (
                player_games.loc[season_mask, stat].shift(1).expanding(min_periods=1).mean()
            )

    # Games played in season (cumulative, shifted)
    for season in player_games['SEASON_YEAR'].unique():
        season_mask = player_games['SEASON_YEAR'] == season
        player_games.loc[season_mask, 'games_played_szn'] = range(len(player_games[season_mask]))

    player_rolling_features.append(player_games)

player_df = pd.concat(player_rolling_features, ignore_index=True)

# Calculate is_starter (min_L5_avg > 25)
player_df['is_starter'] = (player_df['min_L5_avg'] > 25).astype(int)

print(f"  - Created player features for {player_df['PLAYER_ID'].nunique():,} players")


# ==============================================================================
# STEP 6: JOIN OPPONENT FEATURES AND CREATE FINAL DATASET
# ==============================================================================

print("\n[6/6] Joining opponent features and creating final dataset...")

# Join opponent defensive features
# Create lookup for opponent features
opp_features = team_features_df[[
    'game_id', 'team_id', 'team_opp_pts_L10', 'team_pace_L10'
]].copy()

opp_features.columns = [
    'GAME_ID', 'opponent_id', 'opp_pts_allowed_L10', 'opp_pace_L10'
]

player_df = player_df.merge(
    opp_features,
    on=['GAME_ID', 'opponent_id'],
    how='left'
)

# Rename actual stats to have 'actual_' prefix
actual_stats_map = {
    'PTS': 'actual_pts',
    'REB': 'actual_reb',
    'AST': 'actual_ast',
    'STL': 'actual_stl',
    'BLK': 'actual_blk',
    'TOV': 'actual_tov',
    'FG3M': 'actual_fg3m',
    'MIN': 'actual_min',
    'FGM': 'actual_fgm',
    'FGA': 'actual_fga',
    'FG_PCT': 'actual_fg_pct',
    'FG3A': 'actual_fg3a',
    'FG3_PCT': 'actual_fg3_pct',
    'FTM': 'actual_ftm',
    'FTA': 'actual_fta',
    'FT_PCT': 'actual_ft_pct',
    'OREB': 'actual_oreb',
    'DREB': 'actual_dreb',
    'PF': 'actual_pf',
    'PLUS_MINUS': 'actual_plus_minus',
    'WL': 'actual_wl'
}

# Select final columns for player features
player_features_df = player_df.rename(columns=actual_stats_map)

# Select columns to keep
feature_columns = [
    # Identifiers
    'PLAYER_ID', 'PLAYER_NAME', 'GAME_ID', 'GAME_DATE', 'SEASON_YEAR',
    'TEAM_ID', 'TEAM_ABBREVIATION', 'opponent_id', 'opponent_abbr', 'MATCHUP',

    # Context
    'is_home', 'rest_days', 'is_starter', 'games_played_szn',

    # Rolling L5 features (avg and std)
    'pts_L5_avg', 'pts_L5_std',
    'reb_L5_avg', 'reb_L5_std',
    'ast_L5_avg', 'ast_L5_std',
    'stl_L5_avg', 'stl_L5_std',
    'blk_L5_avg', 'blk_L5_std',
    'tov_L5_avg', 'tov_L5_std',
    'fg3m_L5_avg', 'fg3m_L5_std',
    'min_L5_avg', 'min_L5_std',

    # Season averages
    'pts_szn_avg', 'reb_szn_avg', 'ast_szn_avg', 'stl_szn_avg',
    'blk_szn_avg', 'tov_szn_avg', 'fg3m_szn_avg', 'min_szn_avg',

    # Player context
    'player_usage_proxy', 'player_team_pts_share',

    # Opponent features
    'opp_pts_allowed_L10', 'opp_pace_L10',

    # Actual stats (targets)
    'actual_pts', 'actual_reb', 'actual_ast', 'actual_stl', 'actual_blk',
    'actual_tov', 'actual_fg3m', 'actual_min', 'actual_fgm', 'actual_fga',
    'actual_fg_pct', 'actual_fg3a', 'actual_fg3_pct', 'actual_ftm',
    'actual_fta', 'actual_ft_pct', 'actual_oreb', 'actual_dreb',
    'actual_pf', 'actual_plus_minus', 'actual_wl'
]

# Rename columns to standardize
column_rename = {
    'PLAYER_ID': 'player_id',
    'PLAYER_NAME': 'player_name',
    'GAME_ID': 'game_id',
    'GAME_DATE': 'game_date',
    'SEASON_YEAR': 'season',
    'TEAM_ID': 'team_id',
    'TEAM_ABBREVIATION': 'team_abbr',
    'MATCHUP': 'matchup'
}

player_features_df = player_features_df[feature_columns].rename(columns=column_rename)

# Sort by date
player_features_df = player_features_df.sort_values('game_date').reset_index(drop=True)

print(f"  - Created enriched player features ({len(player_features_df):,} rows)")


# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Save player features
player_features_df.to_csv(PROCESSED_DIR / "player_features_enriched.csv", index=False)
print(f"\n[SAVED] player_features_enriched.csv")
print(f"  - Rows: {len(player_features_df):,}")
print(f"  - Columns: {len(player_features_df.columns)}")

# Save game features
game_df.to_csv(PROCESSED_DIR / "game_features.csv", index=False)
print(f"\n[SAVED] game_features.csv")
print(f"  - Rows: {len(game_df):,}")
print(f"  - Columns: {len(game_df.columns)}")

# Already saved team features earlier
print(f"\n[SAVED] team_features.csv")
print(f"  - Rows: {len(team_features_df):,}")
print(f"  - Columns: {len(team_features_df.columns)}")


# ==============================================================================
# DATA QUALITY SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY SUMMARY")
print("=" * 80)

print("\n--- Player Features ---")
print(f"Unique players: {player_features_df['player_id'].nunique():,}")
print(f"Unique games: {player_features_df['game_id'].nunique():,}")
print(f"Date range: {player_features_df['game_date'].min()} to {player_features_df['game_date'].max()}")
print(f"Seasons: {sorted(player_features_df['season'].unique())}")

print("\n--- Missing Values (Player Features) ---")
missing_pct = (player_features_df.isnull().sum() / len(player_features_df) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_pct) > 0:
    print(missing_pct.head(10))
else:
    print("No missing values!")

print("\n--- Game Features ---")
print(f"Unique games: {len(game_df):,}")
print(f"Date range: {game_df['game_date'].min()} to {game_df['game_date'].max()}")
print(f"Seasons: {sorted(game_df['season'].unique())}")

print("\n--- Missing Values (Game Features) ---")
missing_pct_games = (game_df.isnull().sum() / len(game_df) * 100).round(2)
missing_pct_games = missing_pct_games[missing_pct_games > 0].sort_values(ascending=False)
if len(missing_pct_games) > 0:
    print(missing_pct_games.head(10))
else:
    print("No missing values!")

# Sample of engineered features
print("\n--- Sample of Engineered Features (first player-game with full features) ---")
# Find first row with no missing L5 values
sample_row = player_features_df[
    player_features_df['pts_L5_avg'].notna() &
    player_features_df['games_played_szn'] >= 5
].iloc[0]

print(f"\nPlayer: {sample_row['player_name']}")
print(f"Date: {sample_row['game_date']}")
print(f"Matchup: {sample_row['matchup']}")
print(f"Is Home: {sample_row['is_home']}")
print(f"Rest Days: {sample_row['rest_days']}")
print(f"Is Starter: {sample_row['is_starter']}")
print(f"\nRecent Performance (L5):")
print(f"  PTS: {sample_row['pts_L5_avg']:.1f} ± {sample_row['pts_L5_std']:.1f}")
print(f"  REB: {sample_row['reb_L5_avg']:.1f} ± {sample_row['reb_L5_std']:.1f}")
print(f"  AST: {sample_row['ast_L5_avg']:.1f} ± {sample_row['ast_L5_std']:.1f}")
print(f"  MIN: {sample_row['min_L5_avg']:.1f}")
print(f"\nSeason Averages:")
print(f"  PTS: {sample_row['pts_szn_avg']:.1f}")
print(f"  REB: {sample_row['reb_szn_avg']:.1f}")
print(f"  AST: {sample_row['ast_szn_avg']:.1f}")
print(f"\nContext:")
print(f"  Usage Proxy: {sample_row['player_usage_proxy']:.3f}")
print(f"  Team PTS Share: {sample_row['player_team_pts_share']:.3f}")
print(f"  Opp Defense (pts allowed L10): {sample_row['opp_pts_allowed_L10']:.1f}")
print(f"  Opp Pace L10: {sample_row['opp_pace_L10']:.1f}")
print(f"\nActual Performance:")
print(f"  PTS: {sample_row['actual_pts']:.0f}")
print(f"  REB: {sample_row['actual_reb']:.0f}")
print(f"  AST: {sample_row['actual_ast']:.0f}")
print(f"  MIN: {sample_row['actual_min']:.0f}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("  1. Review the processed data files")
print("  2. Verify no data leakage (all features use .shift(1))")
print("  3. Proceed to Phase 2: Model Training")
print("\n")
