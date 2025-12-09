#!/usr/bin/env python3
"""
Nightly precompute job for Monte Carlo projections.

Run daily at 3pm ET before games start:
    python batch/precompute.py

This populates the projections table with simulation results
for all players in all tonight's games.
"""
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import time
import re
import json
from zoneinfo import ZoneInfo

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")

# Stats to extract from simulations
STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"]


def create_histogram(values: np.ndarray, n_bins: int = 50) -> dict:
    """
    Create histogram from simulation array for empirical probability calculation.

    Args:
        values: Array of simulated values (10K samples)
        n_bins: Number of histogram bins (default 50)

    Returns:
        Dict with 'counts' (list of int) and 'edges' (list of float)
    """
    hist, edges = np.histogram(values, bins=n_bins)
    return {
        "counts": hist.tolist(),
        "edges": edges.tolist()
    }

# Build team ID to abbreviation mapping
TEAM_ID_TO_ABBREV = {t["id"]: t["abbreviation"] for t in teams.get_teams()}


def get_db():
    """Get database connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = RealDictCursor
    return conn


def parse_game_time(game_date: str, status_text: str) -> Optional[datetime]:
    """
    Parse game start time from NBA API fields.

    Args:
        game_date: ISO date string from GAME_DATE_EST (e.g., "2025-12-08T00:00:00")
        status_text: Status text from GAME_STATUS_TEXT (e.g., "7:00 pm ET", "Final")

    Returns:
        datetime object with correct start time in UTC, or None if unable to parse
    """
    try:
        # Extract date from GAME_DATE_EST
        date_str = game_date.split('T')[0]  # "2025-12-08"

        # Try to extract time from GAME_STATUS_TEXT using regex
        # Expected formats: "7:00 pm ET", "7:30 pm ET", etc.
        time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm)\s*ET'
        match = re.search(time_pattern, status_text, re.IGNORECASE)

        if not match:
            # Game might be live or final, not scheduled
            return None

        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).lower()

        # Convert to 24-hour format
        if period == 'pm' and hour != 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0

        # Combine date and time in ET timezone
        et_tz = ZoneInfo("America/New_York")
        dt_et = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
        dt_et = dt_et.replace(tzinfo=et_tz)

        # Convert to UTC for database storage
        dt_utc = dt_et.astimezone(ZoneInfo("UTC"))

        return dt_utc

    except Exception as e:
        print(f"Warning: Could not parse game time from '{status_text}': {e}")
        return None


def fetch_tonights_games() -> list[dict]:
    """
    Fetch today's NBA games from nba_api and insert into games table.

    Returns:
        List of game dicts with:
        - game_id: NBA game ID (e.g., "0022400123")
        - home_team: 3-letter code (e.g., "LAL")
        - away_team: 3-letter code (e.g., "BOS")
        - starts_at: ISO datetime string or None
        - status: "scheduled" | "live" | "final"
    """
    try:
        today = date.today().strftime("%m/%d/%Y")
        print(f"Fetching games for {today}...")

        scoreboard = ScoreboardV2(game_date=today)
        games_df = scoreboard.get_data_frames()[0]  # GameHeader

        if games_df.empty:
            print("No games found in NBA API response")
            return []

        games = []
        for _, row in games_df.iterrows():
            # Convert team IDs to abbreviations
            home_team_id = row["HOME_TEAM_ID"]
            away_team_id = row["VISITOR_TEAM_ID"]

            home_abbrev = TEAM_ID_TO_ABBREV.get(home_team_id, str(home_team_id)[:3])
            away_abbrev = TEAM_ID_TO_ABBREV.get(away_team_id, str(away_team_id)[:3])

            # Parse game start time from GAME_DATE_EST + GAME_STATUS_TEXT
            starts_at = parse_game_time(row["GAME_DATE_EST"], row["GAME_STATUS_TEXT"])

            game = {
                "game_id": row["GAME_ID"],
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "starts_at": starts_at,
                "status": "scheduled"
            }
            games.append(game)

        # Insert games into database (FK constraint requires games exist before projections)
        if games:
            _save_games_to_db(games)

        return games

    except Exception as e:
        print(f"Error fetching games: {e}")
        return []


def _save_games_to_db(games: list[dict]) -> None:
    """Insert/update games in database."""
    conn = get_db()
    cursor = conn.cursor()

    for game in games:
        cursor.execute("""
            INSERT INTO games (id, home_team, away_team, starts_at, status, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                starts_at = EXCLUDED.starts_at
        """, (game["game_id"], game["home_team"], game["away_team"], game["starts_at"], game["status"]))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"  Saved {len(games)} games to database")


def run_simulations(game: dict) -> list[dict]:
    """
    Run Monte Carlo simulations for a single game.

    Args:
        game: Game dict from fetch_tonights_games()

    Returns:
        List of projection dicts with:
        - game_id, player_name, stat_type
        - mean, std, p10, p25, p50, p75, p90
        - computed_at
    """
    from simulation.engine import MonteCarloEngine

    home_team = game["home_team"]
    away_team = game["away_team"]
    game_id = game["game_id"]

    print(f"  Initializing Monte Carlo engine...")
    start_time = time.time()

    try:
        # Initialize engine (loads models)
        models_dir = PROJECT_ROOT / "models"
        engine = MonteCarloEngine(models_dir=str(models_dir))

        # Run simulation
        print(f"  Running 10,000 simulations for {away_team} @ {home_team}...")
        result = engine.simulate_game(home_team=home_team, away_team=away_team)

        elapsed = time.time() - start_time
        print(f"  Simulation completed in {elapsed:.1f}s")

        # Extract projections for all players
        projections = []
        computed_at = datetime.utcnow().isoformat()

        for player_name, prediction in result.players.items():
            # Get raw simulation arrays for this player (for histogram)
            raw_player_sims = result.raw_simulations.get(player_name, {})

            for stat in STATS:
                dist = getattr(prediction, stat, None)
                if dist is None:
                    continue

                # Create histogram from raw simulations if available
                raw_values = raw_player_sims.get(stat)
                histogram = create_histogram(raw_values) if raw_values is not None else None

                projections.append({
                    "game_id": game_id,
                    "player_name": player_name,
                    "stat_type": stat,
                    "mean": float(dist.mean) if dist.mean is not None else 0.0,
                    "std": float(dist.std) if dist.std is not None else 0.0,
                    "p10": float(dist.p10) if dist.p10 is not None else 0.0,
                    "p25": float(dist.p25) if dist.p25 is not None else 0.0,
                    "p50": float(dist.p50) if dist.p50 is not None else 0.0,
                    "p75": float(dist.p75) if dist.p75 is not None else 0.0,
                    "p90": float(dist.p90) if dist.p90 is not None else 0.0,
                    "sim_histogram": histogram,
                    "computed_at": computed_at
                })

        return projections

    except Exception as e:
        print(f"  ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_projections(projections: list[dict]) -> int:
    """
    Save projections to database using PostgreSQL upsert.

    Args:
        projections: List from run_simulations()

    Returns:
        Number of rows inserted/updated
    """
    if not projections:
        return 0

    conn = get_db()
    cursor = conn.cursor()

    # Use executemany with ON CONFLICT for upsert
    sql = """
        INSERT INTO projections
            (game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90, sim_histogram, computed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id, player_name, stat_type)
        DO UPDATE SET
            mean = EXCLUDED.mean,
            std = EXCLUDED.std,
            p10 = EXCLUDED.p10,
            p25 = EXCLUDED.p25,
            p50 = EXCLUDED.p50,
            p75 = EXCLUDED.p75,
            p90 = EXCLUDED.p90,
            sim_histogram = EXCLUDED.sim_histogram,
            computed_at = EXCLUDED.computed_at
    """

    data = [
        (
            p["game_id"],
            p["player_name"],
            p["stat_type"],
            p["mean"],
            p["std"],
            p["p10"],
            p["p25"],
            p["p50"],
            p["p75"],
            p["p90"],
            json.dumps(p["sim_histogram"]) if p.get("sim_histogram") else None,
            p["computed_at"]
        )
        for p in projections
    ]

    cursor.executemany(sql, data)
    conn.commit()

    count = len(data)
    cursor.close()
    conn.close()

    game_id = projections[0]["game_id"] if projections else "unknown"
    print(f"  Saved {count} projections for game {game_id}")

    return count


def main(dry_run: bool = False) -> None:
    """
    Orchestrate nightly precompute.

    Args:
        dry_run: If True, simulate without saving to DB
    """
    start = datetime.now()
    print(f"{'='*60}")
    print(f"Starting precompute at {start}")
    print(f"{'='*60}")

    games = fetch_tonights_games()
    print(f"\nFound {len(games)} games tonight")

    if not games:
        print("No games tonight. Exiting.")
        return

    total_projections = 0
    for i, game in enumerate(games, 1):
        matchup = f"{game.get('away_team', '???')} @ {game.get('home_team', '???')}"
        print(f"\n[{i}/{len(games)}] Simulating {matchup}...")

        projections = run_simulations(game)
        print(f"  Generated {len(projections)} projections")

        if not dry_run and projections:
            count = save_projections(projections)
            total_projections += count
        else:
            total_projections += len(projections)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"Complete. {total_projections} projections in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
