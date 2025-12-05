"""
Inference script for NBA player prop predictions.

Pulls tonight's NBA props from SportsGameOdds API and runs predictions
through both v1.0 and v1.1 models.

Output:
- Console: Ranked predictions (agreement first, then confidence)
- File: predictions/YYYY-MM-DD.csv

Usage:
    python scripts/inference.py
"""

import os
import sys
import pickle
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("sgo_api")
BASE_URL = "https://api.sportsgameodds.com/v2/events"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"


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
        # Handle names like LEBRON_JAMES_1_NBA -> LeBron James
        first = parts[0].title()
        last = parts[1].title()
        return f"{first} {last}"
    return player_id


def fetch_todays_props() -> list:
    """Fetch today's NBA props from SportsGameOdds API."""
    if not API_KEY:
        print("ERROR: SGO API key not found in .env (sgo_api)")
        sys.exit(1)

    headers = {"X-Api-Key": API_KEY}

    # Get today and tomorrow in UTC
    # NBA games in evening ET = next day UTC
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

        # Get team names for game description
        teams = event.get("teams", {})
        away = teams.get("away", {}).get("names", {}).get("short", "???")
        home = teams.get("home", {}).get("names", {}).get("short", "???")
        game_str = f"{away} @ {home}"

        # Extract odds
        odds = event.get("odds", {})

        # Group over/under pairs by prop key
        prop_pairs = {}
        for odd_id, odd_data in odds.items():
            # Only process game-level over/under props
            if "game-ou" not in odd_id:
                continue

            player_id = odd_data.get("playerID")
            stat_type = odd_data.get("statID")
            side = odd_data.get("sideID")
            line = odd_data.get("bookOverUnder")
            book_odds = odd_data.get("bookOdds")

            if not all([player_id, stat_type, side, line]):
                continue

            # Skip team totals
            if player_id in ["home", "away"]:
                continue

            # Create unique key for this prop
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

        # Convert pairs to props with implied probabilities
        for prop_key, prop_data in prop_pairs.items():
            over_odds = prop_data.get("over_odds")
            under_odds = prop_data.get("under_odds")

            # Need both sides
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


def load_models():
    """Load both v1.0 and v1.1 models."""
    models = {}

    # Load v1.0
    v1_path = os.path.join(MODELS_DIR, "logistic_v1.pkl")
    if os.path.exists(v1_path):
        with open(v1_path, "rb") as f:
            models["v1.0"] = pickle.load(f)
        print("✓ Loaded model v1.0")
    else:
        print(f"WARNING: v1.0 model not found at {v1_path}")

    # Load v1.1
    v11_path = os.path.join(MODELS_DIR, "logistic_v1.1.pkl")
    if os.path.exists(v11_path):
        with open(v11_path, "rb") as f:
            models["v1.1"] = pickle.load(f)
        print("✓ Loaded model v1.1")
    else:
        print(f"WARNING: v1.1 model not found at {v11_path}")

    if not models:
        print("ERROR: No models found!")
        sys.exit(1)

    return models


def run_inference(props: list, models: dict) -> pd.DataFrame:
    """Run predictions through both models."""
    if not props:
        return pd.DataFrame()

    results = []

    for prop in props:
        result = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "player": prop["player"],
            "player_id": prop["player_id"],
            "game": prop["game"],
            "stat_type": prop["stat_type"],
            "line": prop["line"],
        }

        # v1.0 prediction (uses: line, over_implied, under_implied, stat_type)
        if "v1.0" in models:
            try:
                v1_input = pd.DataFrame([{
                    "line": prop["line"],
                    "over_implied": prop["over_implied"],
                    "under_implied": prop["under_implied"],
                    "stat_type": prop["stat_type"],
                }])
                v1_proba = models["v1.0"].predict_proba(v1_input)[0]
                v1_prob_over = v1_proba[1]
                result["v1_pred"] = "OVER" if v1_prob_over >= 0.5 else "UNDER"
                result["v1_prob"] = v1_prob_over
            except Exception as e:
                result["v1_pred"] = "ERROR"
                result["v1_prob"] = None

        # v1.1 prediction (uses: line, over_implied, under_implied, stat_type, player_id)
        if "v1.1" in models:
            try:
                v11_input = pd.DataFrame([{
                    "line": prop["line"],
                    "over_implied": prop["over_implied"],
                    "under_implied": prop["under_implied"],
                    "stat_type": prop["stat_type"],
                    "player_id": prop["player_id"],
                }])
                v11_proba = models["v1.1"].predict_proba(v11_input)[0]
                v11_prob_over = v11_proba[1]
                result["v1.1_pred"] = "OVER" if v11_prob_over >= 0.5 else "UNDER"
                result["v1.1_prob"] = v11_prob_over
            except Exception as e:
                result["v1.1_pred"] = "ERROR"
                result["v1.1_prob"] = None

        # Check agreement
        if result.get("v1_pred") and result.get("v1.1_pred"):
            if result["v1_pred"] == "ERROR" or result["v1.1_pred"] == "ERROR":
                result["agree"] = False
            else:
                result["agree"] = result["v1_pred"] == result["v1.1_pred"]
        else:
            result["agree"] = False

        # Calculate combined confidence (average of both model probabilities)
        probs = []
        if result.get("v1_prob") is not None:
            probs.append(max(result["v1_prob"], 1 - result["v1_prob"]))
        if result.get("v1.1_prob") is not None:
            probs.append(max(result["v1.1_prob"], 1 - result["v1.1_prob"]))
        result["confidence"] = sum(probs) / len(probs) if probs else 0

        results.append(result)

    df = pd.DataFrame(results)

    # Sort: agreement first, then by confidence
    df = df.sort_values(["agree", "confidence"], ascending=[False, False])

    return df


def print_results(df: pd.DataFrame):
    """Pretty print results to console."""
    if df.empty:
        print("\nNo predictions to display.")
        return

    print("\n" + "=" * 100)
    print("NBA PLAYER PROP PREDICTIONS")
    print("=" * 100)

    # Header
    print(f"\n{'Player':<20} {'Prop':<15} {'Line':>6} {'v1.0':>8} {'v1.1':>8} {'Agree':>7} {'Conf':>6}")
    print("-" * 100)

    for _, row in df.iterrows():
        player = row["player"][:19]
        stat = row["stat_type"][:14]
        line = row["line"]
        v1_pred = row.get("v1_pred", "N/A")
        v11_pred = row.get("v1.1_pred", "N/A")
        agree = "✓ YES" if row["agree"] else "  no"
        conf = f"{row['confidence']*100:.1f}%"

        # Color coding for agreement
        print(f"{player:<20} {stat:<15} {line:>6.1f} {v1_pred:>8} {v11_pred:>8} {agree:>7} {conf:>6}")

    # Summary
    total = len(df)
    agreed = df["agree"].sum()
    print("-" * 100)
    print(f"\nTotal props: {total}")
    print(f"Models agree: {agreed} ({agreed/total*100:.1f}%)" if total > 0 else "")

    # Show high-confidence agreed picks
    high_conf_agreed = df[(df["agree"] == True) & (df["confidence"] >= 0.55)]
    if not high_conf_agreed.empty:
        print(f"\n🎯 HIGH CONFIDENCE PICKS (agree + >55% conf): {len(high_conf_agreed)}")
        for _, row in high_conf_agreed.head(10).iterrows():
            pred = row["v1_pred"]
            print(f"   {row['player']} {row['stat_type']} {pred} {row['line']} ({row['confidence']*100:.1f}%)")


def save_predictions(df: pd.DataFrame):
    """Save predictions to CSV."""
    if df.empty:
        return

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    filepath = os.path.join(PREDICTIONS_DIR, f"{today}.csv")

    df.to_csv(filepath, index=False)
    print(f"\n✓ Predictions saved to {filepath}")


def main():
    print("=" * 70)
    print("THE BRAIN - NBA PROP INFERENCE")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    models = load_models()

    # Fetch today's props
    print("\nFetching today's props...")
    props = fetch_todays_props()

    if not props:
        print("\nNo props available for prediction.")
        return

    print(f"Found {len(props)} player props")

    # Run inference
    print("\nRunning predictions...")
    results = run_inference(props, models)

    # Display results
    print_results(results)

    # Save to CSV
    save_predictions(results)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
