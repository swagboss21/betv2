#!/usr/bin/env python3
"""
Transform SGO historical JSON files to training CSV.

Reads both sgo_historical_202425.json and sgo_historical_202324.json,
consolidates over/under pairs into single rows, and extracts player results.

Output: data/processed/training_data_full.csv
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def american_to_implied(odds_str: str) -> float:
    """Convert American odds to implied probability."""
    odds = float(odds_str.replace('+', ''))
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return abs(odds) / (abs(odds) + 100)


def extract_stat_details(oddid: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract stat_type, player_id, and side from oddID.

    Format: {stat}-{playerID}-{periodID}-{betTypeID}-{sideID}
    Example: assists-ANDREW_WIGGINS_1_NBA-game-ou-over

    Returns: (stat_type, player_id, side) or (None, None, None) if not a game-ou prop
    """
    parts = oddid.split('-')

    # We need at least: stat-playerid-game-ou-side (5 parts minimum)
    # But playerID can contain underscores, so we need to reconstruct it

    if len(parts) < 5:
        return None, None, None

    # Last part is the side (over/under)
    side = parts[-1]
    if side not in ['over', 'under']:
        return None, None, None

    # Second to last should be 'ou'
    if parts[-2] != 'ou':
        return None, None, None

    # Third to last should be 'game'
    if parts[-3] != 'game':
        return None, None, None

    # Everything between stat and 'game' is the playerID
    stat_type = parts[0]
    player_id = '-'.join(parts[1:-3])

    return stat_type, player_id, side


def consolidate_props(odds_dict: Dict) -> Dict:
    """
    Consolidate over/under pairs into single rows.

    Input: {oddID: odd_data, ...}
    Output: {(stat_type, player_id): {'over': odd_data, 'under': odd_data}}
    """
    consolidated = {}

    for oddid, odd_data in odds_dict.items():
        stat_type, player_id, side = extract_stat_details(oddid)

        # Skip if not a valid game-ou prop or no playerID
        if stat_type is None or not player_id or not side:
            continue

        # Skip if playerID is 'all' (team props)
        if player_id == 'all':
            continue

        key = (stat_type, player_id)
        if key not in consolidated:
            consolidated[key] = {}

        consolidated[key][side] = odd_data

    return consolidated


def transform_game(game: Dict, game_idx: int) -> List[Dict]:
    """
    Transform a single game into player prop rows.

    Returns list of rows ready for CSV.
    """
    rows = []

    try:
        # Extract basic game info
        event_id = game.get('eventID', '')
        starts_at = game.get('status', {}).get('startsAt', '')
        date = starts_at.split('T')[0] if starts_at else ''

        # Extract team codes
        away_team = game.get('teams', {}).get('away', {}).get('names', {}).get('short', '')
        home_team = game.get('teams', {}).get('home', {}).get('names', {}).get('short', '')
        game_str = f"{away_team}@{home_team}"

        # Get player results
        results = game.get('results', {})
        player_results = {k: v for k, v in results.items() if k not in ['game', '1q', '2q', '3q', '4q', 'reg']}

        # Get player name mapping
        players = game.get('players', {})

        # Consolidate odds
        odds = game.get('odds', {})
        consolidated = consolidate_props(odds)

        # Process each consolidated prop
        for (stat_type, player_id), sides_data in consolidated.items():
            # Both over and under must be present
            if 'over' not in sides_data or 'under' not in sides_data:
                continue

            over_data = sides_data['over']
            under_data = sides_data['under']

            # Extract odds information (use over data as primary)
            line = float(over_data.get('bookOverUnder', 0))
            over_odds = over_data.get('bookOdds', '')
            under_odds = under_data.get('bookOdds', '')
            fair_odds = over_data.get('fairOdds', '')

            # Calculate implied probabilities
            over_implied = american_to_implied(over_odds) if over_odds else None
            under_implied = american_to_implied(under_odds) if under_odds else None

            # Get player actual result
            player_actual = None
            if player_id in player_results:
                # Map stat_type to result field
                stat_mapping = {
                    'points': 'points',
                    'assists': 'assists',
                    'rebounds': 'rebounds',
                    'blocks': 'blocks',
                    'steals': 'steals',
                    'turnovers': 'turnovers',
                    'threePointersMade': 'threePointersMade',
                }

                field_name = stat_mapping.get(stat_type)
                if field_name:
                    player_actual = player_results[player_id].get(field_name)

            # Also check the score field on the odds (if available)
            if player_actual is None and 'score' in over_data:
                player_actual = over_data.get('score')

            # Get player name
            player_name = players.get(player_id, {}).get('name', '')

            # Calculate hit and margin if we have actual result
            hit = None
            margin = None
            if player_actual is not None:
                hit = 1 if player_actual > line else 0
                margin = player_actual - line

            # Build row
            row = {
                'date': date,
                'player': player_name,
                'player_id': player_id,
                'game': game_str,
                'stat_type': stat_type,
                'line': line,
                'over_odds': over_odds,
                'under_odds': under_odds,
                'over_implied': over_implied,
                'under_implied': under_implied,
                'fair_odds': fair_odds,
                'game_id': event_id,
                'starts_at': starts_at,
                'actual': player_actual,
                'hit': hit,
                'margin': margin,
            }

            rows.append(row)

    except Exception as e:
        print(f"Warning: Error processing game {game_idx}: {str(e)}", file=sys.stderr)

    return rows


def transform_file(json_path: str) -> List[Dict]:
    """Transform a single JSON file into rows."""
    rows = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle both formats: {'data': [...]} and {'events': [...]}
        games = data.get('events', data.get('data', []))
        print(f"Processing {json_path}: {len(games)} games")

        for idx, game in enumerate(games):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(games)} games...")

            game_rows = transform_game(game, idx)
            rows.extend(game_rows)

    except Exception as e:
        print(f"Error reading {json_path}: {str(e)}", file=sys.stderr)

    return rows


def main():
    """Main transformation pipeline."""
    base_dir = Path(__file__).parent.parent

    # Input files
    files_to_process = [
        base_dir / 'data/raw/sgo_historical_202425.json',
        base_dir / 'data/raw/sgo_historical_202324.json',
    ]

    # Output file
    output_path = base_dir / 'data/processed/training_data_full.csv'

    print("=" * 70)
    print("SGO Historical Data Transformation")
    print("=" * 70)

    all_rows = []
    season_rows = {}

    # Process each file
    for json_file in files_to_process:
        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping")
            continue

        season = json_file.stem.split('_')[2]  # Extract season from filename
        rows = transform_file(str(json_file))
        all_rows.extend(rows)
        season_rows[season] = len(rows)
        print(f"  -> {len(rows)} rows extracted\n")

    # Write CSV
    if not all_rows:
        print("Error: No rows extracted!")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'date', 'player', 'player_id', 'game', 'stat_type', 'line',
        'over_odds', 'under_odds', 'over_implied', 'under_implied',
        'fair_odds', 'game_id', 'starts_at', 'actual', 'hit', 'margin'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total rows: {len(all_rows):,}")
    print("\nBreakdown by season:")
    for season, count in sorted(season_rows.items()):
        print(f"  {season}: {count:,} rows")

    # Print verification stats
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    # Check null values
    null_counts = {}
    for field in fieldnames:
        null_count = sum(1 for row in all_rows if row.get(field) is None or row.get(field) == '')
        if null_count > 0:
            null_counts[field] = null_count

    if null_counts:
        print("\nNull value counts:")
        for field, count in sorted(null_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(all_rows)
            print(f"  {field}: {count:,} ({pct:.1f}%)")
    else:
        print("\nNo null values found in critical columns!")

    # Verify hit calculation on first labeled row
    labeled_rows = [r for r in all_rows if r.get('hit') is not None]
    if labeled_rows:
        print(f"\nLabeled rows (with actual results): {len(labeled_rows):,} ({100*len(labeled_rows)/len(all_rows):.1f}%)")

        # Verify hit calculation
        sample = labeled_rows[0]
        actual = sample.get('actual')
        line = sample.get('line')
        hit = sample.get('hit')
        expected_hit = 1 if actual > line else 0

        if hit == expected_hit:
            print(f"Hit calculation verified ✓")
            print(f"  Sample: {sample['player']} {sample['stat_type']} {line} vs actual {actual} -> hit={hit}")
        else:
            print(f"Warning: Hit calculation mismatch!")

    # Print first 5 rows
    print(f"\nFirst 5 rows:")
    for i, row in enumerate(all_rows[:5], 1):
        print(f"\n  Row {i}:")
        for field in fieldnames:
            value = row.get(field)
            if field in ['over_implied', 'under_implied']:
                if value is not None:
                    print(f"    {field}: {value:.4f}")
                else:
                    print(f"    {field}: None")
            else:
                print(f"    {field}: {value}")

    print("\n" + "=" * 70)
    print("Transformation complete!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
