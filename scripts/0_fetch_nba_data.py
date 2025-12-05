#!/usr/bin/env python3
"""
Fetch NBA Player and Team Game Logs from nba_api

Purpose: Pull fresh player and team game logs for 4 seasons (2022-23 through 2025-26)
Output: data/raw/player_game_logs.csv and data/raw/team_game_logs.csv

Usage:
    python scripts/0_fetch_nba_data.py
"""

import os
import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import PlayerGameLogs, TeamGameLogs

# Configuration
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
SEASON_TYPE = 'Regular Season'
RATE_LIMIT_DELAY = 0.6  # seconds between API calls

# Output paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PLAYER_OUTPUT = RAW_DATA_DIR / 'player_game_logs.csv'
TEAM_OUTPUT = RAW_DATA_DIR / 'team_game_logs.csv'


def ensure_directory_exists(directory: Path) -> None:
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {directory}")


def fetch_player_logs(season: str) -> pd.DataFrame:
    """
    Fetch player game logs for a given season.

    Args:
        season: Season string in format 'YYYY-YY' (e.g., '2022-23')

    Returns:
        DataFrame with player game logs
    """
    print(f"Fetching {season} player logs...")
    try:
        player_logs = PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=SEASON_TYPE
        )
        df = player_logs.get_data_frames()[0]
        df['SEASON_YEAR'] = season
        print(f"  -> Retrieved {len(df)} player game logs for {season}")
        return df
    except Exception as e:
        print(f"  -> ERROR fetching player logs for {season}: {e}")
        return pd.DataFrame()


def fetch_team_logs(season: str) -> pd.DataFrame:
    """
    Fetch team game logs for a given season.

    Args:
        season: Season string in format 'YYYY-YY' (e.g., '2022-23')

    Returns:
        DataFrame with team game logs
    """
    print(f"Fetching {season} team logs...")
    try:
        team_logs = TeamGameLogs(
            season_nullable=season,
            season_type_nullable=SEASON_TYPE
        )
        df = team_logs.get_data_frames()[0]
        df['SEASON_YEAR'] = season
        print(f"  -> Retrieved {len(df)} team game logs for {season}")
        return df
    except Exception as e:
        print(f"  -> ERROR fetching team logs for {season}: {e}")
        return pd.DataFrame()


def main():
    """Main execution function."""
    print("=" * 70)
    print("NBA Data Fetch Script")
    print("=" * 70)
    print(f"Seasons to fetch: {', '.join(SEASONS)}")
    print(f"Season type: {SEASON_TYPE}")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s between calls")
    print("=" * 70)
    print()

    # Ensure output directory exists
    ensure_directory_exists(RAW_DATA_DIR)
    print()

    # Fetch player logs for all seasons
    print("STEP 1: Fetching Player Game Logs")
    print("-" * 70)
    all_player_logs = []

    for season in SEASONS:
        df = fetch_player_logs(season)
        if not df.empty:
            all_player_logs.append(df)
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting

    # Combine and save player logs
    if all_player_logs:
        player_df = pd.concat(all_player_logs, ignore_index=True)
        player_df.to_csv(PLAYER_OUTPUT, index=False)
        print()
        print(f"SUCCESS: Saved {len(player_df)} total player game logs to:")
        print(f"  -> {PLAYER_OUTPUT}")
    else:
        print()
        print("WARNING: No player logs were fetched")

    print()
    print()

    # Fetch team logs for all seasons
    print("STEP 2: Fetching Team Game Logs")
    print("-" * 70)
    all_team_logs = []

    for season in SEASONS:
        df = fetch_team_logs(season)
        if not df.empty:
            all_team_logs.append(df)
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting

    # Combine and save team logs
    if all_team_logs:
        team_df = pd.concat(all_team_logs, ignore_index=True)
        team_df.to_csv(TEAM_OUTPUT, index=False)
        print()
        print(f"SUCCESS: Saved {len(team_df)} total team game logs to:")
        print(f"  -> {TEAM_OUTPUT}")
    else:
        print()
        print("WARNING: No team logs were fetched")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_player_logs:
        print(f"Player Logs: {len(player_df):,} rows")
        print(f"  Seasons: {player_df['SEASON_YEAR'].unique().tolist()}")
        print(f"  Date range: {player_df['GAME_DATE'].min()} to {player_df['GAME_DATE'].max()}")
    else:
        print("Player Logs: 0 rows (FAILED)")

    print()

    if all_team_logs:
        print(f"Team Logs: {len(team_df):,} rows")
        print(f"  Seasons: {team_df['SEASON_YEAR'].unique().tolist()}")
        print(f"  Date range: {team_df['GAME_DATE'].min()} to {team_df['GAME_DATE'].max()}")
    else:
        print("Team Logs: 0 rows (FAILED)")

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
