"""
Train logistic regression model v1.1 for NBA player prop prediction.

Changes from v1.0:
- Filter out team totals (player_id = "home"/"away")
- Filter to qualified players (50+ props minimum)
- Add player_id as categorical feature
- Add class_weight='balanced' to fix under-bias

Input: data/processed/training_data_full.csv
Output: models/logistic_v1.1.pkl

Target: hit (1 = over hit, 0 = under hit)
Features: line, over_implied, under_implied, stat_type, player_id
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("=" * 70)
print("NBA PLAYER PROP PREDICTION - LOGISTIC REGRESSION v1.1 TRAINING")
print("=" * 70)
print("\nv1.1 Changes:")
print("  - Filter out team totals (player_id = 'home'/'away')")
print("  - Filter to qualified players (50+ props)")
print("  - Add player_id as categorical feature")
print("  - Add class_weight='balanced'")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
df = pd.read_csv("data/processed/training_data_full.csv")
print(f"Total rows: {len(df):,}")

# ============================================================================
# 2. FILTER OUT TEAM TOTALS
# ============================================================================
print("\n[2/7] Filtering out team totals...")
team_totals = df[df["player_id"].isin(["home", "away"])]
print(f"Team total rows found: {len(team_totals):,}")
df = df[~df["player_id"].isin(["home", "away"])]
print(f"Rows after removing team totals: {len(df):,}")

# ============================================================================
# 3. FILTER TO LABELED ROWS
# ============================================================================
print("\n[3/7] Filtering to labeled rows...")
df_labeled = df[df["hit"].notna()].copy()
print(f"Labeled rows: {len(df_labeled):,}")
print(f"Unlabeled rows: {len(df) - len(df_labeled):,}")

# Check for missing values in features
print("\nMissing values in key columns:")
print(f"  line: {df_labeled['line'].isna().sum()}")
print(f"  over_implied: {df_labeled['over_implied'].isna().sum()}")
print(f"  under_implied: {df_labeled['under_implied'].isna().sum()}")
print(f"  stat_type: {df_labeled['stat_type'].isna().sum()}")
print(f"  player_id: {df_labeled['player_id'].isna().sum()}")
print(f"  hit: {df_labeled['hit'].isna().sum()}")

# Remove rows with missing feature values
df_labeled = df_labeled.dropna(subset=["line", "over_implied", "under_implied", "stat_type", "player_id"])
print(f"Rows after dropping missing features: {len(df_labeled):,}")

# ============================================================================
# 4. FILTER TO QUALIFIED PLAYERS (50+ props)
# ============================================================================
print("\n[4/7] Filtering to qualified players (50+ props)...")
player_counts = df_labeled["player_id"].value_counts()
print(f"Total unique players: {len(player_counts):,}")

qualified_players = player_counts[player_counts >= 50].index
print(f"Qualified players (50+ props): {len(qualified_players):,}")

# Show some stats
print(f"\nPlayer prop distribution:")
print(f"  Min props: {player_counts.min()}")
print(f"  Max props: {player_counts.max()}")
print(f"  Median props: {player_counts.median():.0f}")
print(f"  Mean props: {player_counts.mean():.1f}")

# Filter to qualified players
rows_before = len(df_labeled)
df_labeled = df_labeled[df_labeled["player_id"].isin(qualified_players)]
rows_removed = rows_before - len(df_labeled)
print(f"\nRows after filtering to qualified players: {len(df_labeled):,}")
print(f"Rows removed (players with <50 props): {rows_removed:,}")

# ============================================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[5/7] Preparing features and target...")

X = df_labeled[["line", "over_implied", "under_implied", "stat_type", "player_id"]].copy()
y = df_labeled["hit"].astype(int).copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check target distribution
print("\nTarget distribution (hit):")
print(f"  0 (under hit): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  1 (over hit): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")

baseline_accuracy = max((y == 0).sum(), (y == 1).sum()) / len(y)
print(f"\nBaseline accuracy (always predicting majority class): {baseline_accuracy:.1%}")

# Check stat_type distribution
print(f"\nStat types in data:")
stat_types = X["stat_type"].value_counts().sort_values(ascending=False)
for stat, count in stat_types.items():
    print(f"  {stat}: {count:,}")

# ============================================================================
# 6. FEATURE ENGINEERING & PREPROCESSING PIPELINE
# ============================================================================
print("\n[6/7] Setting up preprocessing pipeline...")

# Define numeric and categorical columns
numeric_features = ["line", "over_implied", "under_implied"]
categorical_features = ["stat_type", "player_id"]  # Added player_id

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# Create full pipeline with preprocessing + model
# KEY CHANGE: Added class_weight='balanced'
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced"  # NEW: Fix under-bias
        )),
    ]
)

print("Pipeline steps:")
print("  1. StandardScaler (numeric features)")
print("  2. OneHotEncoder (categorical features: stat_type + player_id)")
print("  3. LogisticRegression (solver=lbfgs, max_iter=1000, class_weight='balanced')")

# ============================================================================
# 7. TRAIN/TEST SPLIT & MODEL TRAINING
# ============================================================================
print("\n[7/7] Training model...")

# Stratified split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} samples")
print(f"  - 0 (under hit): {(y_train == 0).sum():,} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
print(f"  - 1 (over hit): {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

print(f"Test set: {len(X_test):,} samples")
print(f"  - 0 (under hit): {(y_test == 0).sum():,} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
print(f"  - 1 (over hit): {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

# Train the pipeline
pipeline.fit(X_train, y_train)
print("✓ Model training complete")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE - v1.1")
print("=" * 70)

# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Baseline (always 0): {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
print(f"Improvement over baseline: {(accuracy - baseline_accuracy) * 100:+.2f} percentage points")

print("\nClass 0 (under hit):")
print(f"  Precision: {precision_0:.4f}")
print(f"  Recall: {recall_0:.4f}")
print(f"  F1-score: {f1_0:.4f}")

print("\nClass 1 (over hit):")
print(f"  Precision: {precision_1:.4f}")
print(f"  Recall: {recall_1:.4f}")
print(f"  F1-score: {f1_1:.4f}")

print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(f"       Pred 0  Pred 1")
print(f"Act 0  {cm[0, 0]:6d}  {cm[0, 1]:6d}")
print(f"Act 1  {cm[1, 0]:6d}  {cm[1, 1]:6d}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["under_hit", "over_hit"]))

# ============================================================================
# v1.0 vs v1.1 COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("v1.0 vs v1.1 COMPARISON")
print("=" * 70)

# v1.0 metrics (hardcoded from previous training)
v1_0_accuracy = 0.6176
v1_0_recall_0 = 0.9256
v1_0_recall_1 = 0.1427
v1_0_precision_0 = 0.6247
v1_0_precision_1 = 0.5543

print("\n| Metric         | v1.0    | v1.1    | Change   |")
print("|----------------|---------|---------|----------|")
print(f"| Accuracy       | {v1_0_accuracy*100:.2f}%  | {accuracy*100:.2f}%  | {(accuracy-v1_0_accuracy)*100:+.2f}pp  |")
print(f"| Under Recall   | {v1_0_recall_0*100:.2f}%  | {recall_0*100:.2f}%  | {(recall_0-v1_0_recall_0)*100:+.2f}pp  |")
print(f"| Over Recall    | {v1_0_recall_1*100:.2f}%  | {recall_1*100:.2f}%  | {(recall_1-v1_0_recall_1)*100:+.2f}pp  |")
print(f"| Under Precision| {v1_0_precision_0*100:.2f}%  | {precision_0*100:.2f}%  | {(precision_0-v1_0_precision_0)*100:+.2f}pp  |")
print(f"| Over Precision | {v1_0_precision_1*100:.2f}%  | {precision_1*100:.2f}%  | {(precision_1-v1_0_precision_1)*100:+.2f}pp  |")

# ============================================================================
# FEATURE IMPORTANCE (TOP 20)
# ============================================================================
print("\n" + "=" * 70)
print("TOP 20 FEATURE IMPORTANCE (LOGISTIC REGRESSION COEFFICIENTS)")
print("=" * 70)

# Get feature names from preprocessor
preprocessor_fitted = pipeline.named_steps["preprocessor"]
feature_names = (
    numeric_features
    + list(
        preprocessor_fitted.named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
)

# Get coefficients
coefficients = pipeline.named_steps["classifier"].coef_[0]
feature_importance = pd.DataFrame(
    {"feature": feature_names, "coefficient": coefficients}
).sort_values("coefficient", key=abs, ascending=False)

print(f"\nTotal features: {len(feature_names)}")

print("\nTop 10 positive coefficients (favor over prediction):")
top_positive = feature_importance.nlargest(10, "coefficient")
for idx, row in top_positive.iterrows():
    print(f"  {row['feature']:45s} {row['coefficient']:+.6f}")

print("\nTop 10 negative coefficients (favor under prediction):")
top_negative = feature_importance.nsmallest(10, "coefficient")
for idx, row in top_negative.iterrows():
    print(f"  {row['feature']:45s} {row['coefficient']:+.6f}")

# ============================================================================
# SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 70)

sample_indices = np.arange(min(10, len(X_test)))
sample_X = X_test.iloc[sample_indices]
sample_y_true = y_test.iloc[sample_indices].values
sample_y_pred = y_pred[sample_indices]
sample_y_proba = y_pred_proba[sample_indices]

for i, idx in enumerate(sample_indices):
    actual = sample_y_true[i]
    predicted = sample_y_pred[i]
    prob_under = sample_y_proba[i, 0]
    prob_over = sample_y_proba[i, 1]
    correct = "✓" if actual == predicted else "✗"

    print(
        f"\n{i+1}. {correct} Actual: {actual}, Pred: {predicted} | P(under)={prob_under:.3f}, P(over)={prob_over:.3f}"
    )
    print(f"   Line: {sample_X.iloc[i]['line']:.1f}, Stat: {sample_X.iloc[i]['stat_type']}, Player: {sample_X.iloc[i]['player_id']}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_path = "models/logistic_v1.1.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✓ Model saved to {model_path}")

# ============================================================================
# SUMMARY & ASSESSMENT
# ============================================================================
print("\n" + "=" * 70)
print("ASSESSMENT - v1.1")
print("=" * 70)

# Is model better than baseline?
improvement = accuracy - baseline_accuracy

if improvement > 0.05:
    assessment = "USEFUL - Significant improvement over baseline"
elif improvement > 0.01:
    assessment = "MARGINAL - Small improvement, needs more features/data"
else:
    assessment = "NOT USEFUL - No improvement or worse than baseline"

print(f"\nModel accuracy: {accuracy:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Improvement: {improvement:+.4f} ({improvement * 100:+.2f}%)")
print(f"\nAssessment: {assessment}")

# Check prediction distribution
over_predictions = (y_pred == 1).sum()
under_predictions = (y_pred == 0).sum()
print(f"\nPrediction distribution:")
print(f"  Under (0): {under_predictions:,} ({under_predictions/len(y_pred)*100:.1f}%)")
print(f"  Over (1): {over_predictions:,} ({over_predictions/len(y_pred)*100:.1f}%)")

print(f"\nActual distribution:")
print(f"  Under (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Over (1): {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# Key improvement check
if recall_1 > v1_0_recall_1 + 0.10:
    print("\n✓ SUCCESS: Over recall significantly improved (target was >40%)")
else:
    print("\n⚠ Over recall improvement may be below target")

print("\n" + "=" * 70)
