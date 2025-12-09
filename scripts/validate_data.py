#!/usr/bin/env python3
"""
Data validation script for The Brain projections.

Validates:
1. All p50 values in reasonable ranges per stat type
2. No NULL values in critical columns (mean, std, p10-p90)
3. Every game has at least 10 players with projections
4. Histogram data is valid JSON with counts/edges arrays

Usage:
    python scripts/validate_data.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_db_cursor


# Reasonable ranges for each stat type
STAT_RANGES = {
    'pts': (0, 60),      # 0 to 60 points
    'reb': (0, 25),      # 0 to 25 rebounds
    'ast': (0, 20),      # 0 to 20 assists
    'stl': (0, 8),       # 0 to 8 steals
    'blk': (0, 10),      # 0 to 10 blocks
    'tov': (0, 12),      # 0 to 12 turnovers
    'fg3m': (0, 15),     # 0 to 15 three-pointers
}


def validate_percentile_ranges() -> list[str]:
    """Check all p50 values are within reasonable ranges."""
    errors = []

    with get_db_cursor() as cursor:
        for stat_type, (min_val, max_val) in STAT_RANGES.items():
            cursor.execute("""
                SELECT player_name, p50, mean
                FROM projections
                WHERE stat_type = %s
                  AND (p50 < %s OR p50 > %s)
            """, (stat_type, min_val, max_val))

            bad_rows = cursor.fetchall()
            for row in bad_rows:
                errors.append(
                    f"Out of range p50: {row['player_name']} {stat_type} "
                    f"p50={row['p50']:.2f} (expected {min_val}-{max_val})"
                )

    return errors


def validate_no_nulls() -> list[str]:
    """Check for NULL values in critical columns."""
    errors = []
    critical_columns = ['mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90']

    with get_db_cursor() as cursor:
        for col in critical_columns:
            cursor.execute(f"""
                SELECT player_name, stat_type, game_id
                FROM projections
                WHERE {col} IS NULL
                LIMIT 10
            """)

            bad_rows = cursor.fetchall()
            for row in bad_rows:
                errors.append(
                    f"NULL {col}: {row['player_name']} {row['stat_type']} "
                    f"game={row['game_id']}"
                )

    return errors


def validate_players_per_game() -> list[str]:
    """Check every game has at least 10 players with projections."""
    errors = []
    min_players = 10

    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT g.id, g.home_team, g.away_team,
                   COUNT(DISTINCT p.player_name) as player_count
            FROM games g
            LEFT JOIN projections p ON g.id = p.game_id
            WHERE g.status = 'scheduled'
            GROUP BY g.id, g.home_team, g.away_team
            HAVING COUNT(DISTINCT p.player_name) < %s
        """, (min_players,))

        bad_games = cursor.fetchall()
        for game in bad_games:
            errors.append(
                f"Too few players: {game['away_team']}@{game['home_team']} "
                f"has {game['player_count']} players (min {min_players})"
            )

    return errors


def validate_histograms() -> list[str]:
    """Check histogram data is valid JSON with counts/edges arrays."""
    errors = []

    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT player_name, stat_type, sim_histogram
            FROM projections
            WHERE sim_histogram IS NOT NULL
            LIMIT 100
        """)

        rows = cursor.fetchall()
        for row in rows:
            hist = row['sim_histogram']

            # Check structure
            if not isinstance(hist, dict):
                errors.append(
                    f"Invalid histogram type: {row['player_name']} {row['stat_type']} "
                    f"expected dict, got {type(hist).__name__}"
                )
                continue

            if 'counts' not in hist or 'edges' not in hist:
                errors.append(
                    f"Missing histogram keys: {row['player_name']} {row['stat_type']} "
                    f"keys={list(hist.keys())}"
                )
                continue

            # Check arrays are non-empty
            if len(hist['counts']) == 0 or len(hist['edges']) == 0:
                errors.append(
                    f"Empty histogram arrays: {row['player_name']} {row['stat_type']}"
                )
                continue

            # Check edges = counts + 1
            if len(hist['edges']) != len(hist['counts']) + 1:
                errors.append(
                    f"Histogram size mismatch: {row['player_name']} {row['stat_type']} "
                    f"counts={len(hist['counts'])}, edges={len(hist['edges'])}"
                )

    return errors


def validate_percentile_ordering() -> list[str]:
    """Check p10 < p25 < p50 < p75 < p90."""
    errors = []

    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT player_name, stat_type, p10, p25, p50, p75, p90
            FROM projections
            WHERE NOT (p10 <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= p90)
            LIMIT 20
        """)

        bad_rows = cursor.fetchall()
        for row in bad_rows:
            errors.append(
                f"Percentile ordering wrong: {row['player_name']} {row['stat_type']} "
                f"p10={row['p10']:.1f} p25={row['p25']:.1f} p50={row['p50']:.1f} "
                f"p75={row['p75']:.1f} p90={row['p90']:.1f}"
            )

    return errors


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("The Brain - Data Validation")
    print("=" * 60)

    all_errors = []

    # Run each validation
    checks = [
        ("Percentile ranges", validate_percentile_ranges),
        ("NULL values", validate_no_nulls),
        ("Players per game", validate_players_per_game),
        ("Histogram structure", validate_histograms),
        ("Percentile ordering", validate_percentile_ordering),
    ]

    for name, check_func in checks:
        print(f"\nChecking {name}...", end=" ")
        errors = check_func()

        if errors:
            print(f"FAILED ({len(errors)} errors)")
            all_errors.extend(errors)
            # Show first 3 errors for this check
            for err in errors[:3]:
                print(f"  - {err}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more")
        else:
            print("PASSED")

    print("\n" + "=" * 60)

    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} total errors")
        return 1
    else:
        print("VALIDATION PASSED: All checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
