#!/usr/bin/env python3
"""
Bet grading job for completed games.

Run morning after games complete:
    python batch/grade_bets.py

This grades all pending bets and updates house bot bankrolls.
"""
from datetime import datetime
from typing import Optional


def fetch_actuals(game_id: str) -> dict[str, dict[str, float]]:
    """
    Fetch actual player stats from completed game.

    Args:
        game_id: NBA game ID

    Returns:
        Dict mapping player name to stat dict:
        {
            "LeBron James": {"pts": 28, "reb": 8, "ast": 11, ...},
            ...
        }
    """
    # TODO: Implement in Sprint 4 (S4-T1)
    # Use: nba_api.stats.endpoints.boxscoretraditionalv3
    return {}


def grade_bets(game_id: str) -> dict[str, int]:
    """
    Grade all pending bets for a completed game.

    Args:
        game_id: NBA game ID

    Returns:
        Dict with counts: {"wins": 5, "losses": 3, "pushes": 1}
    """
    # TODO: Implement in Sprint 4 (S4-T2)
    # 1. Get actuals for game
    # 2. Query bets WHERE game_id = ? AND result IS NULL
    # 3. For each bet:
    #    - Compare actual to line based on direction
    #    - Set result = WIN, LOSS, or PUSH
    #    - Set actual_value and graded_at
    return {"wins": 0, "losses": 0, "pushes": 0}


def update_bot_bankrolls() -> None:
    """
    Update house bot bankrolls based on today's results.

    Calculates profit/loss from graded bets and updates bankroll.
    Sets is_alive = FALSE if bankroll <= 0.
    """
    # TODO: Implement in Sprint 5 (S5-T3)
    # 1. Get all bots
    # 2. For each bot, calculate today's P/L
    # 3. Update bankroll
    # 4. Mark dead bots
    pass


def main() -> None:
    """Grade all pending bets from completed games."""
    print(f"Starting bet grading at {datetime.now()}")

    # TODO: Get completed games from yesterday/today
    completed_games: list[str] = []

    for game_id in completed_games:
        print(f"Grading bets for game {game_id}...")
        results = grade_bets(game_id)
        print(f"  W: {results['wins']}, L: {results['losses']}, P: {results['pushes']}")

    print("\nUpdating bot bankrolls...")
    update_bot_bankrolls()

    print("Grading complete.")


if __name__ == "__main__":
    main()
