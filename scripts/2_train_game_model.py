"""
Train XGBoost game score prediction model for NBA betting project.

Purpose:
    Train a multi-output regression model to predict both home and away team scores
    for NBA games. The model will be used in Monte Carlo simulations to generate
    game score distributions.

Input:
    - data/processed/game_features.csv (6,885 games across 4 seasons)

Output:
    - models/game_model.pkl (model + variance lookup for simulation)

Model:
    - Multi-output XGBoost regressor
    - Predicts: (home_pts, away_pts)
    - Features: L10 rolling stats for both teams (pts, opp_pts, fg_pct, pace, rest)

Validation:
    - Train: 2022-23, 2023-24, 2024-25 seasons
    - Test: 2025-26 season (out-of-sample temporal validation)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "game_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "game_model.pkl"

# Features and targets
FEATURES = [
    'home_pts_L10', 'home_opp_pts_L10', 'home_fg_pct_L10', 'home_pace_L10', 'home_rest_days',
    'away_pts_L10', 'away_opp_pts_L10', 'away_fg_pct_L10', 'away_pace_L10', 'away_rest_days'
]
TARGETS = ['home_pts', 'away_pts']

# Score buckets for variance calculation (for Monte Carlo simulation)
SCORE_BUCKETS = [
    (90, 100),
    (100, 110),
    (110, 120),
    (120, 130),
    (130, 200)  # 130+
]


def load_data():
    """Load and validate game features dataset."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df):,} games from {DATA_PATH}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"\nSeason distribution:")
    print(df['season'].value_counts().sort_index())

    return df


def prepare_train_test_split(df):
    """Split data into train/test sets by season."""
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)

    # Drop rows with missing features
    before_drop = len(df)
    df_clean = df.dropna(subset=FEATURES + TARGETS).copy()
    dropped = before_drop - len(df_clean)

    print(f"\nDropped {dropped:,} rows with missing values")
    print(f"Remaining: {len(df_clean):,} games")

    # Train: 2022-23, 2023-24, 2024-25
    # Test: 2025-26
    train_df = df_clean[df_clean['season'].isin(['2022-23', '2023-24', '2024-25'])].copy()
    test_df = df_clean[df_clean['season'] == '2025-26'].copy()

    print(f"\nTrain set: {len(train_df):,} games (2022-23, 2023-24, 2024-25)")
    print(f"Test set:  {len(test_df):,} games (2025-26)")

    # Extract features and targets
    X_train = train_df[FEATURES]
    y_train = train_df[TARGETS]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGETS]

    # Print feature stats
    print("\nFeature summary (training set):")
    print(X_train.describe().round(2))

    return X_train, X_test, y_train, y_test, test_df


def train_model(X_train, y_train):
    """Train multi-output XGBoost regressor."""
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ))

    print("\nModel configuration:")
    print("  Estimators: 200")
    print("  Max depth: 6")
    print("  Learning rate: 0.05")
    print("  Random state: 42")

    print("\nTraining...")
    model.fit(X_train, y_train)
    print("Training complete!")

    return model


def calculate_variance_lookup(y_test, y_pred):
    """
    Calculate residual standard deviation by score bucket.

    For Monte Carlo simulation, we need to know the variance of predictions
    at different score levels. Higher-scoring games may have more variance.

    Returns:
        dict: {bucket_label: {'std': float, 'count': int}}
    """
    print("\n" + "=" * 80)
    print("CALCULATING VARIANCE LOOKUP (for Monte Carlo)")
    print("=" * 80)

    variance_lookup = {}

    # Calculate residuals for both home and away predictions
    residuals_home = y_test['home_pts'].values - y_pred[:, 0]
    residuals_away = y_test['away_pts'].values - y_pred[:, 1]

    # Combine all residuals and predictions for bucketing
    all_residuals = np.concatenate([residuals_home, residuals_away])
    all_predictions = np.concatenate([y_pred[:, 0], y_pred[:, 1]])

    print(f"\nTotal residual samples: {len(all_residuals):,}")
    print("\nScore buckets:")

    for low, high in SCORE_BUCKETS:
        bucket_label = f"{low}-{high}"
        mask = (all_predictions >= low) & (all_predictions < high)
        bucket_residuals = all_residuals[mask]

        if len(bucket_residuals) > 0:
            std = np.std(bucket_residuals)
            variance_lookup[bucket_label] = {
                'std': float(std),
                'count': int(len(bucket_residuals))
            }
            print(f"  {bucket_label:>8}: std={std:5.2f}, n={len(bucket_residuals):,}")
        else:
            # Fallback to overall std if bucket is empty
            overall_std = np.std(all_residuals)
            variance_lookup[bucket_label] = {
                'std': float(overall_std),
                'count': 0
            }
            print(f"  {bucket_label:>8}: std={overall_std:5.2f} (fallback), n=0")

    # Add overall std as fallback
    variance_lookup['overall'] = {
        'std': float(np.std(all_residuals)),
        'count': int(len(all_residuals))
    }
    print(f"\n  Overall: std={np.std(all_residuals):5.2f}, n={len(all_residuals):,}")

    return variance_lookup


def evaluate_model(model, X_test, y_test, test_df):
    """Evaluate model performance and print detailed metrics."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    y_pred = model.predict(X_test)

    # Overall metrics
    mae_home = mean_absolute_error(y_test['home_pts'], y_pred[:, 0])
    mae_away = mean_absolute_error(y_test['away_pts'], y_pred[:, 1])
    rmse_home = np.sqrt(mean_squared_error(y_test['home_pts'], y_pred[:, 0]))
    rmse_away = np.sqrt(mean_squared_error(y_test['away_pts'], y_pred[:, 1]))
    r2_home = r2_score(y_test['home_pts'], y_pred[:, 0])
    r2_away = r2_score(y_test['away_pts'], y_pred[:, 1])

    print("\n" + "-" * 80)
    print("HOME TEAM PREDICTIONS")
    print("-" * 80)
    print(f"MAE:  {mae_home:.2f} points")
    print(f"RMSE: {rmse_home:.2f} points")
    print(f"R²:   {r2_home:.4f}")

    print("\n" + "-" * 80)
    print("AWAY TEAM PREDICTIONS")
    print("-" * 80)
    print(f"MAE:  {mae_away:.2f} points")
    print(f"RMSE: {rmse_away:.2f} points")
    print(f"R²:   {r2_away:.4f}")

    print("\n" + "-" * 80)
    print("COMBINED METRICS")
    print("-" * 80)
    mae_combined = (mae_home + mae_away) / 2
    rmse_combined = (rmse_home + rmse_away) / 2
    r2_combined = (r2_home + r2_away) / 2
    print(f"Average MAE:  {mae_combined:.2f} points")
    print(f"Average RMSE: {rmse_combined:.2f} points")
    print(f"Average R²:   {r2_combined:.4f}")

    # Sample predictions
    print("\n" + "-" * 80)
    print("SAMPLE PREDICTIONS (first 10 test games)")
    print("-" * 80)

    sample_df = test_df.head(10).copy()
    sample_df['pred_home_pts'] = y_pred[:10, 0]
    sample_df['pred_away_pts'] = y_pred[:10, 1]
    sample_df['home_error'] = sample_df['home_pts'] - sample_df['pred_home_pts']
    sample_df['away_error'] = sample_df['away_pts'] - sample_df['pred_away_pts']

    display_cols = [
        'game_date', 'home_team_abbr', 'away_team_abbr',
        'home_pts', 'pred_home_pts', 'home_error',
        'away_pts', 'pred_away_pts', 'away_error'
    ]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(sample_df[display_cols].to_string(index=False))

    return y_pred


def print_feature_importance(model, feature_names):
    """Print feature importance for both home and away predictions."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    # XGBoost feature importance from both estimators
    home_model = model.estimators_[0]
    away_model = model.estimators_[1]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'home_importance': home_model.feature_importances_,
        'away_importance': away_model.feature_importances_
    })

    importance_df['avg_importance'] = (
        importance_df['home_importance'] + importance_df['away_importance']
    ) / 2

    importance_df = importance_df.sort_values('avg_importance', ascending=False)

    print("\nTop features (by average importance):")
    print(importance_df.to_string(index=False))


def save_model(model, variance_lookup):
    """Save trained model with variance lookup for Monte Carlo simulation."""
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_artifact = {
        'model': model,
        'variance_lookup': variance_lookup,
        'features': FEATURES,
        'targets': TARGETS,
        'metadata': {
            'model_type': 'MultiOutputRegressor(XGBRegressor)',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'train_seasons': ['2022-23', '2023-24', '2024-25'],
            'test_season': '2025-26'
        }
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_artifact, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Artifact contains:")
    print(f"  - Trained model")
    print(f"  - Variance lookup (for Monte Carlo)")
    print(f"  - Feature list: {FEATURES}")
    print(f"  - Target list: {TARGETS}")
    print(f"  - Metadata")


def main():
    """Main training pipeline."""
    print("\n")
    print("=" * 80)
    print("NBA GAME SCORE PREDICTION MODEL TRAINING")
    print("=" * 80)

    # Load data
    df = load_data()

    # Prepare train/test split
    X_train, X_test, y_train, y_test, test_df = prepare_train_test_split(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test, test_df)

    # Calculate variance lookup for Monte Carlo
    variance_lookup = calculate_variance_lookup(y_test, y_pred)

    # Print feature importance
    print_feature_importance(model, FEATURES)

    # Save model
    save_model(model, variance_lookup)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review metrics above")
    print("  2. If satisfactory, proceed to train minutes model")
    print("  3. Use this model in Monte Carlo game simulations")
    print("\n")


if __name__ == "__main__":
    main()
