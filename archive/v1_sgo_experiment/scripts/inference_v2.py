"""
Edge-Based Inference for NBA Player Props (v2)

Instead of ML prediction, this uses historical player edge:
    edge = actual_hit_rate - implied_probability

Strategy: Bet UNDER on props where player has negative edge
(their overs are historically overpriced by the market)

Output:
- Console: Filtered UNDER picks ranked by edge magnitude
- File: predictions/YYYY-MM-DD_edge.csv

Usage:
    python scripts/inference_v2.py                # Default -10pp threshold
    python scripts/inference_v2.py --threshold -15  # Stricter threshold
    python scripts/inference_v2.py --threshold -20 --sniper  # High accuracy mode
"""

import os
import sys
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("sgo_api")
BASE_URL = "https://api.sportsgameodds.com/v2/events"
BASE_DIR = Path(__file__).parent.parent
PLAYER_EDGE_PATH = BASE_DIR / "analysis/player_edge_lookup.csv"
STAT_EDGE_PATH = BASE_DIR / "analysis/stat_type_edge_lookup.csv"
PREDICTIONS_DIR = BASE_DIR / "predictions"


def american_to_implied(odds_str: str) -> float:
    """Convert American odds to implied probability."""
    if odds_str is None or odds_str == "":
        return None
    try:
        odds = float(str(odds_str).replace("+", ""))
        if odds > 0:
            return 100 / (100 + odds)
        else:
            return abs(odds) / (abs(odds) + 100)
    except (ValueError, TypeError):
        return None


def format_player_name(player_id: str) -> str:
    """Convert FIRSTNAME_LASTNAME_1_NBA to readable name."""
    if not player_id:
        return "Unknown"
    parts = player_id.split("_")
    if len(parts) >= 2:
        first = parts[0].title()
        last = parts[1].title()
        return f"{first} {last}"
    return player_id


def load_edge_lookups():
    """Load player and stat type edge lookup tables."""
    lookups = {}

    if PLAYER_EDGE_PATH.exists():
        player_df = pd.read_csv(PLAYER_EDGE_PATH)
        lookups["player"] = player_df.set_index("player_id")["edge"].to_dict()
        print(f"Loaded {len(lookups['player'])} player edges")
    else:
        print(f"WARNING: Player edge lookup not found at {PLAYER_EDGE_PATH}")
        print("Run: python analysis/edge_analysis.py to generate it")
        lookups["player"] = {}

    if STAT_EDGE_PATH.exists():
        stat_df = pd.read_csv(STAT_EDGE_PATH)
        lookups["stat"] = stat_df.set_index("stat_type")["edge"].to_dict()
        print(f"Loaded {len(lookups['stat'])} stat type edges")
    else:
        lookups["stat"] = {}

    return lookups


def fetch_todays_props() -> list:
    """Fetch today's NBA props from SportsGameOdds API."""
    if not API_KEY:
        print("ERROR: SGO API key not found in .env (sgo_api)")
        sys.exit(1)

    headers = {"X-Api-Key": API_KEY}

    # Get today and tomorrow in UTC
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "leagueID": "NBA",
        "startsAfter": today,
        "startsBefore": tomorrow,
        "limit": 100,
    }

    print(f"Fetching NBA props for {today}...")

    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"ERROR: API request failed: {e}")
        sys.exit(1)

    if not data.get("success"):
        print(f"ERROR: API returned error: {data}")
        sys.exit(1)

    events = data.get("data", [])

    if not events:
        print("No NBA games found for today.")
        return []

    print(f"Found {len(events)} games")

    # Extract props from each event
    props = []
    for event in events:
        game_id = event.get("eventID", "")
        status = event.get("status", {})
        starts_at = status.get("startsAt", "")

        # Get team names
        teams = event.get("teams", {})
        away = teams.get("away", {}).get("names", {}).get("short", "???")
        home = teams.get("home", {}).get("names", {}).get("short", "???")
        game_str = f"{away} @ {home}"

        # Extract odds
        odds = event.get("odds", {})

        # Group over/under pairs
        prop_pairs = {}
        for odd_id, odd_data in odds.items():
            if "game-ou" not in odd_id:
                continue

            player_id = odd_data.get("playerID")
            stat_type = odd_data.get("statID")
            side = odd_data.get("sideID")
            line = odd_data.get("bookOverUnder")
            book_odds = odd_data.get("bookOdds")

            if not all([player_id, stat_type, side, line]):
                continue

            if player_id in ["home", "away"]:
                continue

            prop_key = f"{player_id}_{stat_type}_{line}"

            if prop_key not in prop_pairs:
                prop_pairs[prop_key] = {
                    "player_id": player_id,
                    "player": format_player_name(player_id),
                    "stat_type": stat_type,
                    "line": float(line),
                    "game": game_str,
                    "game_id": game_id,
                    "starts_at": starts_at,
                }

            if side == "over":
                prop_pairs[prop_key]["over_odds"] = book_odds
            elif side == "under":
                prop_pairs[prop_key]["under_odds"] = book_odds

        # Convert to props with implied probabilities
        for prop_key, prop_data in prop_pairs.items():
            over_odds = prop_data.get("over_odds")
            under_odds = prop_data.get("under_odds")

            if not over_odds or not under_odds:
                continue

            over_implied = american_to_implied(over_odds)
            under_implied = american_to_implied(under_odds)

            if over_implied is None or under_implied is None:
                continue

            prop_data["over_implied"] = over_implied
            prop_data["under_implied"] = under_implied
            props.append(prop_data)

    return props


def apply_edge_filter(props: list, lookups: dict, threshold: float) -> pd.DataFrame:
    """
    Filter props to those where player has edge <= threshold.
    Returns UNDER recommendations only.
    """
    results = []

    for prop in props:
        player_id = prop["player_id"]
        stat_type = prop["stat_type"]

        # Get player edge (None if not in lookup = new player)
        player_edge = lookups["player"].get(player_id)
        stat_edge = lookups["stat"].get(stat_type, 0)

        # Skip if player not in lookup (no historical data)
        if player_edge is None:
            continue

        # Combined edge
        combined_edge = player_edge + stat_edge

        # Filter by threshold
        if player_edge > threshold:
            continue

        # Calculate expected accuracy based on historical edge
        # Under accuracy = 1 - hit_rate = 1 - (implied + edge)
        expected_under_acc = 1 - (prop["over_implied"] + player_edge)

        result = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "player": prop["player"],
            "player_id": player_id,
            "game": prop["game"],
            "stat_type": stat_type,
            "line": prop["line"],
            "over_odds": prop.get("over_odds", ""),
            "under_odds": prop.get("under_odds", ""),
            "over_implied": prop["over_implied"],
            "player_edge": player_edge,
            "player_edge_pp": player_edge * 100,
            "stat_edge": stat_edge,
            "stat_edge_pp": stat_edge * 100,
            "combined_edge": combined_edge,
            "combined_edge_pp": combined_edge * 100,
            "expected_under_acc": expected_under_acc,
            "recommendation": "UNDER",
        }
        results.append(result)

    df = pd.DataFrame(results)

    if not df.empty:
        # Sort by player edge (most negative = strongest signal)
        df = df.sort_values("player_edge")

    return df


def print_results(df: pd.DataFrame, threshold: float):
    """Pretty print results to console."""
    if df.empty:
        print(f"\nNo props found with player edge <= {threshold*100:.0f}pp")
        print("Try a less strict threshold (e.g., --threshold -5)")
        return

    print("\n" + "=" * 100)
    print(f"EDGE-BASED UNDER PICKS (threshold: {threshold*100:.0f}pp)")
    print("=" * 100)

    # Header
    print(f"\n{'Player':<22} {'Stat':<15} {'Line':>6} {'Odds':>8} {'Edge':>8} {'Exp Acc':>8}")
    print("-" * 100)

    for _, row in df.iterrows():
        player = row["player"][:21]
        stat = row["stat_type"][:14]
        line = row["line"]
        under_odds = row["under_odds"]
        edge_pp = row["player_edge_pp"]
        exp_acc = row["expected_under_acc"]

        print(f"{player:<22} {stat:<15} {line:>6.1f} {under_odds:>8} {edge_pp:>+7.1f}pp {exp_acc:>7.1%}")

    # Summary
    print("-" * 100)
    print(f"\nTotal UNDER picks: {len(df)}")
    print(f"Avg player edge: {df['player_edge_pp'].mean():.1f}pp")
    print(f"Avg expected accuracy: {df['expected_under_acc'].mean():.1%}")

    # Top picks
    print(f"\n--- TOP 5 STRONGEST SIGNALS ---")
    for _, row in df.head(5).iterrows():
        print(f"  {row['player']} {row['stat_type']} UNDER {row['line']} "
              f"({row['under_odds']}) | Edge: {row['player_edge_pp']:+.1f}pp")

    # Historical context
    print(f"\n--- HISTORICAL CONTEXT ---")
    print(f"At {threshold*100:.0f}pp threshold, backtest showed:")

    # Reference values from our analysis
    threshold_map = {
        -5: (63.4, 479),
        -10: (67.5, 214),
        -15: (73.4, 67),
        -20: (80.8, 22),
        -25: (89.4, 11),
        -30: (93.6, 7),
    }

    t_int = int(threshold * 100)
    if t_int in threshold_map:
        acc, vol = threshold_map[t_int]
        print(f"  - Historical accuracy: {acc}%")
        print(f"  - Avg props per day: {vol}")


def save_predictions(df: pd.DataFrame):
    """Save predictions to CSV."""
    if df.empty:
        return

    PREDICTIONS_DIR.mkdir(exist_ok=True)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    filepath = PREDICTIONS_DIR / f"{today}_edge.csv"

    df.to_csv(filepath, index=False)
    print(f"\nSaved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Edge-based NBA prop inference")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=-10,
        help="Player edge threshold in percentage points (default: -10)"
    )
    parser.add_argument(
        "--sniper",
        action="store_true",
        help="High accuracy mode (sets threshold to -20)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all props with edge data (no threshold filter)"
    )
    args = parser.parse_args()

    threshold = args.threshold / 100  # Convert pp to decimal
    if args.sniper:
        threshold = -0.20
    if args.all:
        threshold = 1.0  # Include everything

    print("=" * 70)
    print("THE BRAIN - EDGE-BASED INFERENCE (v2)")
    print("=" * 70)

    # Load edge lookups
    print("\nLoading edge lookups...")
    lookups = load_edge_lookups()

    if not lookups["player"]:
        print("\nERROR: Player edge lookup is empty.")
        print("Run: python analysis/edge_analysis.py first")
        sys.exit(1)

    # Fetch today's props
    print("\nFetching today's props...")
    props = fetch_todays_props()

    if not props:
        print("\nNo props available.")
        return

    print(f"Found {len(props)} player props")

    # Apply edge filter
    print(f"\nFiltering to player edge <= {threshold*100:.0f}pp...")
    results = apply_edge_filter(props, lookups, threshold)

    # Display results
    print_results(results, threshold)

    # Save to CSV
    save_predictions(results)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
