"""
Production prediction script with adjustable threshold.

Usage:
  python3 scripts/predict_with_threshold.py --threshold 0.40

Based on Branch C analysis:
  - Default threshold=0.50 gives 14.3% over recall
  - Optimal threshold=0.40 gives 65.5% over recall
  - Trade-off: -5.6pp accuracy for +51pp recall (9:1 ROI)
"""

import pandas as pd
import numpy as np
import pickle
import argparse

def load_model():
    """Load trained model pipeline."""
    with open("models/logistic_v1.pkl", "rb") as f:
        return pickle.load(f)

def predict_with_threshold(X, model, threshold=0.40):
    """
    Make predictions with custom threshold.

    Args:
        X: Feature DataFrame with columns [line, over_implied, under_implied, stat_type]
        model: Trained sklearn pipeline
        threshold: Decision threshold (default 0.40 from Branch C analysis)

    Returns:
        DataFrame with predictions and probabilities
    """
    # Get probabilities
    prob_over = model.predict_proba(X)[:, 1]

    # Apply threshold
    pred = (prob_over >= threshold).astype(int)

    # Create results DataFrame
    results = X.copy()
    results['prob_over'] = prob_over
    results['prob_under'] = 1 - prob_over
    results['prediction'] = pred
    results['prediction_label'] = results['prediction'].map({0: 'UNDER', 1: 'OVER'})
    results['confidence'] = np.where(
        pred == 1,
        prob_over,
        1 - prob_over
    )

    return results

def format_recommendation(row):
    """Format a single prediction as a recommendation."""
    if row['prediction'] == 1:
        return f"✓ OVER {row['line']:.1f} ({row['stat_type']}) - {row['confidence']:.1%} confidence (P={row['prob_over']:.3f})"
    else:
        return f"✗ UNDER {row['line']:.1f} ({row['stat_type']}) - {row['confidence']:.1%} confidence (P={row['prob_under']:.3f})"

def main():
    parser = argparse.ArgumentParser(description='Predict with custom threshold')
    parser.add_argument('--threshold', type=float, default=0.40,
                        help='Decision threshold (default: 0.40)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV file (default: test set from training data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (default: print to console)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence to show (default: 0.0)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"NBA PLAYER PROP PREDICTIONS (Threshold = {args.threshold:.2f})")
    print("=" * 80)

    # Load model
    print("\n[1/3] Loading model...")
    model = load_model()
    print("✓ Model loaded")

    # Load data
    print("\n[2/3] Loading data...")
    if args.input:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df):,} rows from {args.input}")
    else:
        # Use test set from training data
        from sklearn.model_selection import train_test_split

        df_full = pd.read_csv("data/processed/training_data_full.csv")
        df_labeled = df_full[df_full["hit"].notna()].copy()
        df_labeled = df_labeled.dropna(subset=["line", "over_implied", "under_implied", "stat_type"])

        X = df_labeled[["line", "over_implied", "under_implied", "stat_type"]].copy()
        y = df_labeled["hit"].astype(int).copy()

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        df = X_test.copy()
        df['actual'] = y_test
        print(f"✓ Using test set: {len(df):,} rows")

    # Make predictions
    print("\n[3/3] Making predictions...")
    X = df[["line", "over_implied", "under_implied", "stat_type"]].copy()
    results = predict_with_threshold(X, model, threshold=args.threshold)

    # Add actual results if available
    if 'actual' in df.columns:
        results['actual'] = df['actual'].values
        results['correct'] = (results['prediction'] == results['actual'])

        accuracy = results['correct'].mean()
        over_predictions = (results['prediction'] == 1).sum()
        under_predictions = (results['prediction'] == 0).sum()

        print(f"✓ Predictions complete")
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Over predictions:  {over_predictions:,} ({over_predictions/len(results)*100:.1f}%)")
        print(f"  Under predictions: {under_predictions:,} ({under_predictions/len(results)*100:.1f}%)")

        if results['actual'].notna().all():
            # Calculate recall by class
            actual_overs = (results['actual'] == 1)
            actual_unders = (results['actual'] == 0)

            over_recall = (results.loc[actual_overs, 'prediction'] == 1).mean()
            under_recall = (results.loc[actual_unders, 'prediction'] == 0).mean()

            print(f"\nRecall by Class:")
            print(f"  Over recall:  {over_recall:.1%} ({(results['prediction'] == 1)[actual_overs].sum():,} / {actual_overs.sum():,})")
            print(f"  Under recall: {under_recall:.1%} ({(results['prediction'] == 0)[actual_unders].sum():,} / {actual_unders.sum():,})")
    else:
        print(f"✓ Predictions complete")

    # Filter by confidence
    if args.min_confidence > 0:
        results = results[results['confidence'] >= args.min_confidence].copy()
        print(f"\n✓ Filtered to {len(results):,} predictions with ≥{args.min_confidence:.0%} confidence")

    # Show sample predictions
    print("\n" + "=" * 80)
    print(f"SAMPLE PREDICTIONS (First 20)")
    print("=" * 80)

    for idx, row in results.head(20).iterrows():
        rec = format_recommendation(row)

        if 'actual' in results.columns:
            actual_label = 'OVER' if row['actual'] == 1 else 'UNDER'
            correct = "✓" if row['correct'] else "✗"
            print(f"{correct} {rec:80s} | Actual: {actual_label}")
        else:
            print(f"  {rec}")

    # Show high-confidence over predictions
    over_preds = results[results['prediction'] == 1].copy()
    if len(over_preds) > 0:
        print("\n" + "=" * 80)
        print(f"HIGH CONFIDENCE OVER PREDICTIONS (Top 10)")
        print("=" * 80)

        top_overs = over_preds.nlargest(10, 'prob_over')
        for idx, row in top_overs.iterrows():
            rec = format_recommendation(row)

            if 'actual' in results.columns:
                actual_label = 'OVER' if row['actual'] == 1 else 'UNDER'
                correct = "✓" if row['correct'] else "✗"
                print(f"{correct} {rec:80s} | Actual: {actual_label}")
            else:
                print(f"  {rec}")

    # Save to file if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✓ Saved {len(results):,} predictions to {args.output}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
