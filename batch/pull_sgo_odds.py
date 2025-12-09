#!/usr/bin/env python3
"""
Pull player prop odds from SportsGameOdds API.

Usage:
    python batch/pull_sgo_odds.py

This populates the odds table with real sportsbook lines for tonight's games.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from batch.player_matcher import (
    extract_player_name_from_sgo_id,
    match_player,
    apply_name_override
)

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")

# SGO API config
SGO_API_KEY = os.environ.get("sgo_api")
SGO_BASE_URL = "https://api.sportsgameodds.com/v2/events"

# Stat type mapping: our stat -> SGO stat ID
STAT_MAPPING = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
    "fg3m": "threePointersMade",
}

# Reverse mapping
SGO_TO_OUR_STAT = {v: k for k, v in STAT_MAPPING.items()}


def get_db():
    """Get database connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = RealDictCursor
    return conn


def get_player_names_from_db() -> list[str]:
    """Get all player names from projections table."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT player_name FROM projections")
    names = [row['player_name'] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return names


def get_games_from_db() -> list[dict]:
    """Get tonight's games with team info for matching."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, home_team, away_team
        FROM games
        WHERE status = 'scheduled'
        ORDER BY starts_at
    """)
    games = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return games


def match_sgo_event_to_game(sgo_event: dict, db_games: list[dict]) -> Optional[str]:
    """
    Match SGO event to our game by team abbreviations.

    Returns our game_id if match found, None otherwise.
    """
    teams = sgo_event.get("teams", {})
    home_team = teams.get("home", {})
    away_team = teams.get("away", {})

    sgo_home = home_team.get("names", {}).get("short", "")
    sgo_away = away_team.get("names", {}).get("short", "")

    if not sgo_home or not sgo_away:
        return None

    for game in db_games:
        if game["home_team"] == sgo_home and game["away_team"] == sgo_away:
            return game["id"]

    return None


def fetch_sgo_events(league: str = "NBA") -> list[dict]:
    """
    Fetch events with odds from SGO API.

    Args:
        league: League ID (default NBA)

    Returns:
        List of event dicts with odds
    """
    if not SGO_API_KEY:
        print("ERROR: No SGO API key found in environment (sgo_api)")
        return []

    headers = {"X-Api-Key": SGO_API_KEY}

    # Build oddIDs for player props we care about
    # Format: {statID}-PLAYER_ID-game-ou-over
    odd_ids = []
    for sgo_stat in SGO_TO_OUR_STAT.keys():
        # Request all players for each stat
        odd_ids.append(f"{sgo_stat}-PLAYER_ID-game-ou-over")

    params = {
        "leagueID": league,
        "started": "false",  # Only upcoming games
        "startsAfter": datetime.now().strftime("%Y-%m-%d"),  # Required: get today's games only
        "oddIDs": ",".join(odd_ids),
        "includeOpposingOdds": "true",  # Get both over and under
    }

    all_events = []
    next_cursor = None

    while True:
        if next_cursor:
            params["cursor"] = next_cursor

        try:
            response = requests.get(SGO_BASE_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            events = data.get("data", data.get("events", []))
            all_events.extend(events)

            next_cursor = data.get("nextCursor")
            if not next_cursor:
                break

        except requests.RequestException as e:
            print(f"Error fetching SGO events: {e}")
            break

    return all_events


def parse_sgo_odds(events: list[dict], player_names: list[str], db_games: list[dict]) -> list[dict]:
    """
    Parse SGO events into odds records.

    Args:
        events: Raw events from SGO API
        player_names: List of player names from our DB for matching
        db_games: Games from our database for event matching

    Returns:
        List of odds dicts ready for database insert
    """
    odds_records = []
    unmatched_players = set()
    matched_events = 0

    for event in events:
        # Match SGO event to our game ID
        our_game_id = match_sgo_event_to_game(event, db_games)
        if not our_game_id:
            continue  # Skip events not in our database

        matched_events += 1
        odds_data = event.get("odds", {})

        for odd_id, odd_info in odds_data.items():
            # Parse oddID: {statID}-{playerID}-{period}-{betType}-{side}
            parts = odd_id.split("-")
            if len(parts) < 5:
                continue

            # Extract components
            sgo_stat = parts[0]
            sgo_player_id = "-".join(parts[1:-3])  # Player ID may contain dashes
            period = parts[-3]
            bet_type = parts[-2]
            side = parts[-1]

            # Only process full-game over/under props
            if period != "game" or bet_type != "ou":
                continue

            # Map SGO stat to our stat
            our_stat = SGO_TO_OUR_STAT.get(sgo_stat)
            if not our_stat:
                continue

            # Extract player name and match
            sgo_name = extract_player_name_from_sgo_id(sgo_player_id)
            sgo_name = apply_name_override(sgo_name)

            match_result = match_player(sgo_player_id, player_names, threshold=0.85)
            if not match_result:
                unmatched_players.add(sgo_name)
                continue

            matched_name, score = match_result

            # Extract line and odds
            line = odd_info.get("closeOverUnder") or odd_info.get("line")
            if line is None:
                continue

            # Get American odds
            close_odds = odd_info.get("closeOdds", -110)
            # Format as string with sign
            if isinstance(close_odds, (int, float)):
                odds_str = f"+{int(close_odds)}" if close_odds > 0 else str(int(close_odds))
            else:
                odds_str = str(close_odds)

            # Determine over/under odds based on side
            if side == "over":
                over_odds = odds_str
                under_odds = "-110"  # Default, will be overwritten if opposing is found
            else:
                under_odds = odds_str
                over_odds = "-110"

            # Check for existing record to merge over/under
            existing_key = (our_game_id, matched_name, our_stat)
            existing_record = None
            for rec in odds_records:
                if (rec["game_id"], rec["player_name"], rec["stat_type"]) == existing_key:
                    existing_record = rec
                    break

            if existing_record:
                # Update existing record with this side's odds
                if side == "over":
                    existing_record["over_odds"] = odds_str
                else:
                    existing_record["under_odds"] = odds_str
            else:
                # Create new record
                odds_records.append({
                    "game_id": our_game_id,
                    "player_name": matched_name,
                    "stat_type": our_stat,
                    "line": float(line),
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "book": "consensus",
                })

    print(f"  Matched {matched_events} SGO events to our games")

    if unmatched_players:
        print(f"  Warning: {len(unmatched_players)} unmatched players:")
        for name in list(unmatched_players)[:5]:
            print(f"    - {name}")
        if len(unmatched_players) > 5:
            print(f"    ... and {len(unmatched_players) - 5} more")

    return odds_records


def save_odds(odds_records: list[dict]) -> int:
    """
    Save odds to database using upsert.

    Args:
        odds_records: List of odds dicts

    Returns:
        Number of rows inserted/updated
    """
    if not odds_records:
        return 0

    conn = get_db()
    cursor = conn.cursor()

    sql = """
        INSERT INTO odds (game_id, player_name, stat_type, line, over_odds, under_odds, book, pulled_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (game_id, player_name, stat_type, book)
        DO UPDATE SET
            line = EXCLUDED.line,
            over_odds = EXCLUDED.over_odds,
            under_odds = EXCLUDED.under_odds,
            pulled_at = EXCLUDED.pulled_at
    """

    data = [
        (
            rec["game_id"],
            rec["player_name"],
            rec["stat_type"],
            rec["line"],
            rec["over_odds"],
            rec["under_odds"],
            rec["book"],
        )
        for rec in odds_records
    ]

    cursor.executemany(sql, data)
    conn.commit()

    count = len(data)
    cursor.close()
    conn.close()

    return count


def main():
    """Pull odds from SGO and save to database."""
    print("=" * 60)
    print(f"Pulling SGO odds at {datetime.now()}")
    print("=" * 60)

    # Get player names for matching
    print("\nFetching player names from database...")
    player_names = get_player_names_from_db()
    print(f"  Found {len(player_names)} players in projections")

    if not player_names:
        print("No players found. Run precompute first.")
        return

    # Get games for event matching
    print("\nFetching games from database...")
    db_games = get_games_from_db()
    print(f"  Found {len(db_games)} scheduled games")
    for g in db_games:
        print(f"    {g['away_team']} @ {g['home_team']}")

    if not db_games:
        print("No scheduled games found. Run precompute first.")
        return

    # Fetch events from SGO
    print("\nFetching events from SGO API...")
    events = fetch_sgo_events("NBA")
    print(f"  Received {len(events)} events")

    if not events:
        print("No events found. Check API key and date.")
        return

    # Parse odds
    print("\nParsing odds...")
    odds_records = parse_sgo_odds(events, player_names, db_games)
    print(f"  Parsed {len(odds_records)} odds records")

    # Save to database
    if odds_records:
        print("\nSaving to database...")
        count = save_odds(odds_records)
        print(f"  Saved {count} odds records")
    else:
        print("\nNo odds to save (SGO events may not match tonight's games).")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
