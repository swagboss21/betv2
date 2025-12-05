"""
Train logistic regression model for NBA player prop prediction.

Input: data/processed/training_data_full.csv (95,714 rows, 91,518 labeled)
Output: models/logistic_v1.pkl

Target: hit (1 = over hit, 0 = under hit)
Features: line, over_implied, under_implied, stat_type (one-hot encoded)
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
print("NBA PLAYER PROP PREDICTION - LOGISTIC REGRESSION TRAINING")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv("data/processed/training_data_full.csv")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# 2. FILTER TO LABELED ROWS
# ============================================================================
print("\n[2/6] Filtering to labeled rows...")
df_labeled = df[df["hit"].notna()].copy()
print(f"Labeled rows: {len(df_labeled):,}")
print(f"Unlabeled rows: {len(df) - len(df_labeled):,}")

# Check for missing values in features
print("\nMissing values in key columns:")
print(f"  line: {df_labeled['line'].isna().sum()}")
print(f"  over_implied: {df_labeled['over_implied'].isna().sum()}")
print(f"  under_implied: {df_labeled['under_implied'].isna().sum()}")
print(f"  stat_type: {df_labeled['stat_type'].isna().sum()}")
print(f"  hit: {df_labeled['hit'].isna().sum()}")

# Remove rows with missing feature values
df_labeled = df_labeled.dropna(subset=["line", "over_implied", "under_implied", "stat_type"])
print(f"Rows after dropping missing features: {len(df_labeled):,}")

# ============================================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[3/6] Preparing features and target...")

X = df_labeled[["line", "over_implied", "under_implied", "stat_type"]].copy()
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
# 4. FEATURE ENGINEERING & PREPROCESSING PIPELINE
# ============================================================================
print("\n[4/6] Setting up preprocessing pipeline...")

# Define numeric and categorical columns
numeric_features = ["line", "over_implied", "under_implied"]
categorical_features = ["stat_type"]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# Create full pipeline with preprocessing + model
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")),
    ]
)

print("Pipeline steps:")
print("  1. StandardScaler (numeric features)")
print("  2. OneHotEncoder (categorical features)")
print("  3. LogisticRegression (solver=lbfgs, max_iter=1000)")

# ============================================================================
# 5. TRAIN/TEST SPLIT & MODEL TRAINING
# ============================================================================
print("\n[5/6] Training model...")

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
# 6. EVALUATION
# ============================================================================
print("\n[6/6] Evaluating model...")

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
print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

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
# FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (LOGISTIC REGRESSION COEFFICIENTS)")
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

print("\nAll features (sorted by absolute coefficient):")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:40s} {row['coefficient']:+.6f}")

print("\nTop 5 positive coefficients (favor over prediction):")
top_positive = feature_importance.nlargest(5, "coefficient")
for idx, row in top_positive.iterrows():
    print(f"  {row['feature']:40s} {row['coefficient']:+.6f}")

print("\nTop 5 negative coefficients (favor under prediction):")
top_negative = feature_importance.nsmallest(5, "coefficient")
for idx, row in top_negative.iterrows():
    print(f"  {row['feature']:40s} {row['coefficient']:+.6f}")

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
    print(f"   Line: {sample_X.iloc[i]['line']:.1f}, Stat: {sample_X.iloc[i]['stat_type']}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_path = "models/logistic_v1.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✓ Model saved to {model_path}")

# ============================================================================
# SUMMARY & ASSESSMENT
# ============================================================================
print("\n" + "=" * 70)
print("ASSESSMENT")
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

# Check if model is learning or just predicting majority class
majority_class_predictions = (y_pred == 0).sum()
majority_class_actual = (y_test == 0).sum()
print(
    f"\nMajority class (0) predictions: {majority_class_predictions:,} / {len(y_pred):,}"
)
print(f"Majority class (0) actual: {majority_class_actual:,} / {len(y_test):,}")

if majority_class_predictions > len(y_pred) * 0.95:
    print("WARNING: Model is heavily biased toward predicting the majority class!")
elif abs(majority_class_predictions - majority_class_actual) < len(y_pred) * 0.05:
    print("Model is learning patterns (class distribution shifts with features)")
else:
    print("Model shows some learning (class predictions vary from actual distribution)")

print("\n" + "=" * 70)
