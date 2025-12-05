#!/usr/bin/env python3
"""
Make predictions using the trained model.

Usage:
    python scripts/4_predict.py                    # Interactive mode
    python scripts/4_predict.py --csv input.csv    # Batch mode from CSV

Input CSV format:
    line,over_implied,under_implied,stat_type
    25.5,0.48,0.57,points
    6.5,0.45,0.60,assists
"""

import pickle
import pandas as pd
import argparse
import sys
from pathlib import Path

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "logistic_v1.pkl"
THRESHOLD = 0.40  # Lowered from 0.50 based on analysis (see docs/SHORTCOMINGS_REPORT)


def load_model():
    """Load the trained model."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run 'python scripts/3_train_model.py' first.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_single(model, line, over_implied, under_implied, stat_type):
    """Make a single prediction."""
    prop = pd.DataFrame({
        'line': [line],
        'over_implied': [over_implied],
        'under_implied': [under_implied],
        'stat_type': [stat_type]
    })

    proba = model.predict_proba(prop)[0]
    prob_over = proba[1]
    prob_under = proba[0]

    # Use adjusted threshold
    prediction = "OVER" if prob_over >= THRESHOLD else "UNDER"

    return {
        'prediction': prediction,
        'prob_over': prob_over,
        'prob_under': prob_under,
        'confidence': max(prob_over, prob_under)
    }


def predict_batch(model, csv_path):
    """Make predictions on a CSV file."""
    df = pd.read_csv(csv_path)

    required_cols = ['line', 'over_implied', 'under_implied', 'stat_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    proba = model.predict_proba(df[required_cols])
    df['prob_over'] = proba[:, 1]
    df['prob_under'] = proba[:, 0]
    df['prediction'] = df['prob_over'].apply(lambda x: 'OVER' if x >= THRESHOLD else 'UNDER')

    return df


def interactive_mode(model):
    """Interactive prediction mode."""
    print("\n" + "=" * 50)
    print("NBA Player Prop Predictor")
    print(f"Model: logistic_v1.pkl | Threshold: {THRESHOLD}")
    print("=" * 50)

    stat_types = [
        'points', 'assists', 'rebounds', 'steals', 'blocks',
        'turnovers', 'threePointersMade', 'points+assists',
        'points+rebounds', 'points+rebounds+assists',
        'assists+rebounds', 'blocks+steals'
    ]

    while True:
        print("\n--- New Prediction ---")

        try:
            line = float(input("Line (e.g., 25.5): "))
            over_implied = float(input("Over implied prob (e.g., 0.48): "))
            under_implied = float(input("Under implied prob (e.g., 0.57): "))

            print("\nStat types:", ', '.join(stat_types[:6]))
            print("            ", ', '.join(stat_types[6:]))
            stat_type = input("Stat type: ").strip()

            if stat_type not in stat_types:
                print(f"Warning: '{stat_type}' may not be recognized by model")

            result = predict_single(model, line, over_implied, under_implied, stat_type)

            print("\n" + "-" * 30)
            print(f"PREDICTION: {result['prediction']}")
            print(f"Over prob:  {result['prob_over']:.1%}")
            print(f"Under prob: {result['prob_under']:.1%}")
            print("-" * 30)

        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

        cont = input("\nAnother prediction? (y/n): ").strip().lower()
        if cont != 'y':
            break


def main():
    parser = argparse.ArgumentParser(description='NBA Player Prop Predictor')
    parser.add_argument('--csv', type=str, help='Path to CSV file for batch predictions')
    parser.add_argument('--output', type=str, help='Output CSV path (for batch mode)')
    args = parser.parse_args()

    model = load_model()
    print(f"Model loaded from {MODEL_PATH}")

    if args.csv:
        # Batch mode
        print(f"Processing {args.csv}...")
        results = predict_batch(model, args.csv)

        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print("\nResults:")
            print(results.to_string())

        # Summary
        over_count = (results['prediction'] == 'OVER').sum()
        under_count = (results['prediction'] == 'UNDER').sum()
        print(f"\nSummary: {over_count} OVER, {under_count} UNDER")
    else:
        # Interactive mode
        interactive_mode(model)


if __name__ == '__main__':
    main()
