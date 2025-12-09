#!/usr/bin/env python3
"""
Fetch NBA box scores and grade pending bets.

Run daily in the morning after games complete:
    python batch/fetch_results.py

This script:
1. Finds games that should be complete (started > 6 hours ago)
2. Fetches final box scores from nba_api
3. Updates games table with final scores and status
4. Grades pending bets by comparing actual stats to lines
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")

# Map our stat types to nba_api column names
STAT_TO_NBA_COL = {
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TO",
    "fg3m": "FG3M",
}


def get_db():
    """Get database connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = RealDictCursor
    return conn


def get_completed_games() -> list[dict]:
    """
    Find games that should be complete but haven't been graded yet.

    Games are considered complete if:
    - Status is 'scheduled' (not yet marked final)
    - Started more than 4 hours ago (NBA games typically last 2.5-3 hours)

    Returns:
        List of game dicts needing box score fetch and grading
    """
    conn = get_db()
    cursor = conn.cursor()

    # Find scheduled games that started > 4 hours ago
    cursor.execute("""
        SELECT id, home_team, away_team, starts_at
        FROM games
        WHERE status = 'scheduled'
          AND starts_at IS NOT NULL
          AND starts_at < NOW() - INTERVAL '4 hours'
        ORDER BY starts_at
    """)
    games = [dict(row) for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return games


def fetch_box_score(game_id: str) -> Optional[dict]:
    """
    Fetch box score from nba_api for a completed game.

    Args:
        game_id: NBA game ID (e.g., "0022400123")

    Returns:
        Dict with:
        - home_score: int
        - away_score: int
        - player_stats: dict mapping player_name -> stat_type -> value

        Returns None if game not found or not final.
    """
    from nba_api.stats.endpoints import BoxScoreTraditionalV2

    try:
        print(f"    Fetching box score for {game_id}...")
        time.sleep(0.6)  # Rate limiting for nba_api

        box = BoxScoreTraditionalV2(game_id=game_id)

        # Get team stats for final scores
        team_stats = box.team_stats.get_data_frame()
        if team_stats.empty:
            print(f"    No team stats found for {game_id}")
            return None

        # Get player stats
        player_stats_df = box.player_stats.get_data_frame()
        if player_stats_df.empty:
            print(f"    No player stats found for {game_id}")
            return None

        # Extract scores (first row is away, second is home based on typical ordering)
        # But safer to look at specific columns if available
        away_score = int(team_stats.iloc[0]['PTS'])
        home_score = int(team_stats.iloc[1]['PTS'])

        # Build player stats lookup
        player_stats = {}
        for _, row in player_stats_df.iterrows():
            player_name = row['PLAYER_NAME']
            stats = {}
            for our_stat, nba_col in STAT_TO_NBA_COL.items():
                val = row.get(nba_col)
                if val is not None:
                    stats[our_stat] = float(val)
            player_stats[player_name] = stats

        print(f"    Found {len(player_stats)} players with stats")
        return {
            "home_score": home_score,
            "away_score": away_score,
            "player_stats": player_stats
        }

    except Exception as e:
        print(f"    Error fetching box score for {game_id}: {e}")
        return None


def update_game_final(game_id: str, home_score: int, away_score: int) -> None:
    """
    Mark game as final and store scores.

    Args:
        game_id: NBA game ID
        home_score: Final home team score
        away_score: Final away team score
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE games
        SET status = 'final',
            home_score = %s,
            away_score = %s
        WHERE id = %s
    """, (home_score, away_score, game_id))

    conn.commit()
    cursor.close()
    conn.close()


def get_pending_bets_for_game(game_id: str) -> list[dict]:
    """
    Get all pending (ungraded) bets for a specific game.

    Args:
        game_id: NBA game ID

    Returns:
        List of bet dicts needing grading
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, player_name, stat_type, line, direction
        FROM bets
        WHERE game_id = %s AND result IS NULL
    """, (game_id,))
    bets = [dict(row) for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return bets


def grade_bet(direction: str, line: float, actual_value: float) -> str:
    """
    Determine bet result: WIN, LOSS, or PUSH.

    Args:
        direction: OVER or UNDER
        line: The betting line
        actual_value: Player's actual stat

    Returns:
        "WIN", "LOSS", or "PUSH"
    """
    if actual_value == line:
        return "PUSH"
    elif direction == "OVER":
        return "WIN" if actual_value > line else "LOSS"
    else:  # UNDER
        return "WIN" if actual_value < line else "LOSS"


def update_bet_result(bet_id: int, result: str, actual_value: float) -> None:
    """
    Update bet with grading result.

    Args:
        bet_id: Bet ID
        result: WIN, LOSS, or PUSH
        actual_value: Player's actual stat
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE bets
        SET result = %s,
            actual_value = %s,
            graded_at = NOW()
        WHERE id = %s
    """, (result, actual_value, bet_id))

    conn.commit()
    cursor.close()
    conn.close()


def grade_pending_bets(game_id: str, player_stats: dict) -> dict:
    """
    Grade all pending bets for a game using actual player stats.

    Args:
        game_id: NBA game ID
        player_stats: Dict mapping player_name -> stat_type -> actual_value

    Returns:
        Dict with wins, losses, pushes counts
    """
    bets = get_pending_bets_for_game(game_id)

    if not bets:
        return {"wins": 0, "losses": 0, "pushes": 0, "skipped": 0}

    results = {"wins": 0, "losses": 0, "pushes": 0, "skipped": 0}

    for bet in bets:
        player_name = bet['player_name']
        stat_type = bet['stat_type']
        line = bet['line']
        direction = bet['direction']

        # Find player in box score (handle name variations)
        actual_stats = player_stats.get(player_name)

        if actual_stats is None:
            # Try case-insensitive match
            for ps_name, stats in player_stats.items():
                if ps_name.lower() == player_name.lower():
                    actual_stats = stats
                    break

        if actual_stats is None:
            print(f"      Warning: Player '{player_name}' not found in box score")
            results["skipped"] += 1
            continue

        actual_value = actual_stats.get(stat_type)
        if actual_value is None:
            print(f"      Warning: Stat '{stat_type}' not found for {player_name}")
            results["skipped"] += 1
            continue

        result = grade_bet(direction, line, actual_value)
        update_bet_result(bet['id'], result, actual_value)

        results[result.lower() + "s"] += 1
        print(f"      {player_name} {stat_type} {direction} {line}: actual={actual_value} -> {result}")

    return results


def main(dry_run: bool = False):
    """
    Orchestrate results fetching and bet grading.

    Args:
        dry_run: If True, fetch but don't update database
    """
    start = datetime.now()
    print("=" * 60)
    print(f"Fetching results at {start}")
    print("=" * 60)

    # Find games needing grading
    games = get_completed_games()
    print(f"\nFound {len(games)} completed games to process")

    if not games:
        print("No games to grade. Exiting.")
        return

    total_results = {"wins": 0, "losses": 0, "pushes": 0, "skipped": 0}
    games_processed = 0

    for game in games:
        matchup = f"{game['away_team']} @ {game['home_team']}"
        print(f"\n  Processing {matchup} ({game['id']})...")

        # Fetch box score
        box_score = fetch_box_score(game['id'])

        if box_score is None:
            print(f"    Skipping - could not fetch box score")
            continue

        games_processed += 1
        home_score = box_score['home_score']
        away_score = box_score['away_score']
        print(f"    Final score: {game['away_team']} {away_score} - {game['home_team']} {home_score}")

        if not dry_run:
            # Update game status
            update_game_final(game['id'], home_score, away_score)

            # Grade bets
            results = grade_pending_bets(game['id'], box_score['player_stats'])
            print(f"    Graded bets: {results['wins']}W {results['losses']}L {results['pushes']}P ({results['skipped']} skipped)")

            for key in total_results:
                total_results[key] += results[key]
        else:
            print(f"    [DRY RUN] Would update game and grade bets")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Complete. Processed {games_processed}/{len(games)} games in {elapsed:.1f}s")
    if not dry_run:
        print(f"Total: {total_results['wins']}W {total_results['losses']}L {total_results['pushes']}P ({total_results['skipped']} skipped)")
    print("=" * 60)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
