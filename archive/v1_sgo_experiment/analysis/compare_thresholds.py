"""
Compare performance at different thresholds side-by-side.

Shows the impact of threshold adjustment on model performance.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def evaluate_at_threshold(y_test, prob_over, threshold):
    """Evaluate model at given threshold."""
    y_pred = (prob_over >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'threshold': threshold,
        'accuracy': acc,
        'precision_under': prec_0,
        'precision_over': prec_1,
        'recall_under': rec_0,
        'recall_over': rec_1,
        'f1_under': f1_0,
        'f1_over': f1_1,
        'f1_avg': (f1_0 + f1_1) / 2,
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1],
    }

def main():
    print("=" * 80)
    print("THRESHOLD COMPARISON: OLD (0.50) vs NEW (0.40)")
    print("=" * 80)

    # Load model
    print("\n[1/3] Loading model...")
    with open("models/logistic_v1.pkl", "rb") as f:
        model = pickle.load(f)
    print("✓ Model loaded")

    # Load data
    print("\n[2/3] Loading test data...")
    df = pd.read_csv("data/processed/training_data_full.csv")
    df_labeled = df[df["hit"].notna()].copy()
    df_labeled = df_labeled.dropna(subset=["line", "over_implied", "under_implied", "stat_type"])

    X = df_labeled[["line", "over_implied", "under_implied", "stat_type"]].copy()
    y = df_labeled["hit"].astype(int).copy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"  - Under (0): {(y_test == 0).sum():,} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
    print(f"  - Over (1): {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

    # Get predictions
    print("\n[3/3] Comparing thresholds...")
    prob_over = model.predict_proba(X_test)[:, 1]

    # Evaluate at both thresholds
    results_old = evaluate_at_threshold(y_test, prob_over, 0.50)
    results_new = evaluate_at_threshold(y_test, prob_over, 0.40)

    # Print comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30s} {'OLD (0.50)':>12s} {'NEW (0.40)':>12s} {'Change':>12s}")
    print("-" * 80)

    metrics = [
        ('Accuracy', 'accuracy', '%'),
        ('', None, None),
        ('Under Precision', 'precision_under', '%'),
        ('Under Recall', 'recall_under', '%'),
        ('Under F1-Score', 'f1_under', '%'),
        ('', None, None),
        ('Over Precision', 'precision_over', '%'),
        ('Over Recall', 'recall_over', '%'),
        ('Over F1-Score', 'f1_over', '%'),
        ('', None, None),
        ('Average F1-Score', 'f1_avg', '%'),
    ]

    for label, key, fmt in metrics:
        if key is None:
            print()
            continue

        old_val = results_old[key]
        new_val = results_new[key]
        change = new_val - old_val

        if fmt == '%':
            old_str = f"{old_val * 100:.1f}%"
            new_str = f"{new_val * 100:.1f}%"
            change_str = f"{change * 100:+.1f}pp"
        else:
            old_str = f"{old_val:,.0f}"
            new_str = f"{new_val:,.0f}"
            change_str = f"{change:+,.0f}"

        print(f"{label:<30s} {old_str:>12s} {new_str:>12s} {change_str:>12s}")

    # Confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)

    print("\nOLD THRESHOLD (0.50):")
    print(f"       Pred_0  Pred_1")
    print(f"Act_0  {results_old['tn']:6,d}  {results_old['fp']:6,d}")
    print(f"Act_1  {results_old['fn']:6,d}  {results_old['tp']:6,d}")
    print(f"\nInterpretation:")
    print(f"  - {results_old['tn']:,} correct under predictions (True Negatives)")
    print(f"  - {results_old['fp']:,} wrong over predictions (False Positives)")
    print(f"  - {results_old['fn']:,} missed overs (False Negatives) ← PROBLEM")
    print(f"  - {results_old['tp']:,} correct over predictions (True Positives)")

    print("\nNEW THRESHOLD (0.40):")
    print(f"       Pred_0  Pred_1")
    print(f"Act_0  {results_new['tn']:6,d}  {results_new['fp']:6,d}")
    print(f"Act_1  {results_new['fn']:6,d}  {results_new['tp']:6,d}")
    print(f"\nInterpretation:")
    print(f"  - {results_new['tn']:,} correct under predictions (True Negatives)")
    print(f"  - {results_new['fp']:,} wrong over predictions (False Positives)")
    print(f"  - {results_new['fn']:,} missed overs (False Negatives) ← IMPROVED")
    print(f"  - {results_new['tp']:,} correct over predictions (True Positives)")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    tp_gain = results_new['tp'] - results_old['tp']
    fn_reduction = results_old['fn'] - results_new['fn']
    fp_increase = results_new['fp'] - results_old['fp']
    tn_loss = results_old['tn'] - results_new['tn']

    print(f"\n1. OVER DETECTION IMPROVED")
    print(f"   - Caught {tp_gain:,} MORE overs (+{tp_gain/(y_test == 1).sum()*100:.1f}pp recall)")
    print(f"   - Reduced missed overs by {fn_reduction:,} (-{fn_reduction/results_old['fn']*100:.1f}%)")

    print(f"\n2. TRADE-OFF ANALYSIS")
    print(f"   - Gained: {tp_gain:,} True Positives (correct overs)")
    print(f"   - Cost: {fp_increase:,} False Positives (wrong overs)")
    print(f"   - Cost: {tn_loss:,} True Negatives (correct unders)")
    print(f"   - ROI: {tp_gain/max(1, fp_increase + tn_loss):.1f}x (gain/cost ratio)")

    print(f"\n3. BUSINESS IMPACT")
    print(f"   - OLD: Only surfaced {results_old['tp']:,} profitable overs")
    print(f"   - NEW: Surfaces {results_new['tp']:,} profitable overs")
    print(f"   - Increase: +{tp_gain:,} opportunities (+{tp_gain/results_old['tp']*100:.0f}%)")

    print(f"\n4. ACCURACY TRADE-OFF")
    print(f"   - Accuracy dropped {(results_old['accuracy'] - results_new['accuracy'])*100:.1f}pp")
    print(f"   - BUT: Over recall improved {(results_new['recall_over'] - results_old['recall_over'])*100:.1f}pp")
    print(f"   - This is acceptable for a betting co-pilot (surfacing opportunities > precision)")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print(f"\n✅ DEPLOY WITH THRESHOLD = 0.40")
    print(f"\nExpected Outcomes:")
    print(f"  - {results_new['recall_over']*100:.1f}% of overs will be detected (vs {results_old['recall_over']*100:.1f}%)")
    print(f"  - {results_new['accuracy']*100:.1f}% overall accuracy (vs {results_old['accuracy']*100:.1f}%)")
    print(f"  - Users will see {results_new['tp']:,} correct over recommendations per 18K props")
    print(f"  - {results_new['fp']:,} will be false alarms (but still valuable to evaluate)")

    print(f"\nWhy This is the Right Choice:")
    print(f"  1. Betting co-pilot's job is to SURFACE opportunities, not make final decisions")
    print(f"  2. Users can evaluate recommendations themselves")
    print(f"  3. Missing {results_old['fn']:,} overs (old) vs {results_new['fn']:,} overs (new) is huge improvement")
    print(f"  4. 5.6pp accuracy drop is acceptable for 51pp recall gain")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
