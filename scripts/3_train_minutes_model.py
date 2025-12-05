#!/usr/bin/env python3
"""
Train XGBoost model to predict player minutes played.

Purpose: Minutes Model for Monte Carlo simulation (Game→Minutes→Stats hierarchy)

Input: data/processed/player_features_enriched.csv
Output: models/minutes_model.pkl (model + variance lookup)

Validation Strategy:
- Train: 2022-23, 2023-24, 2024-25 seasons
- Test: 2025-26 season

Target Metrics:
- MAE < 5 minutes
- RMSE, R-squared
- Performance by player tier
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_features_enriched.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "minutes_model.pkl"

# Features for minutes prediction
FEATURES = [
    'min_L5_avg',      # Recent minutes average
    'min_szn_avg',     # Season minutes average
    'is_starter',      # Binary: 1 if starter
    'games_played_szn',# Games played this season
    'rest_days',       # Days since last game
    'is_home',         # Home game indicator
]

TARGET = 'actual_min'

# Player tiers for variance calculation (defined as dict with ranges for pickling)
TIER_DEFINITIONS = {
    'Stars': (30, float('inf')),      # min_szn_avg >= 30
    'Starters': (20, 30),              # 20 <= min_szn_avg < 30
    'Rotation': (10, 20),              # 10 <= min_szn_avg < 20
    'Bench': (0, 10),                  # min_szn_avg < 10
}

def get_tier_mask(df, tier_name):
    """Get boolean mask for a player tier."""
    min_val, max_val = TIER_DEFINITIONS[tier_name]
    return (df['min_szn_avg'] >= min_val) & (df['min_szn_avg'] < max_val)


def load_and_prepare_data():
    """Load data and prepare train/test splits."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # Filter to players who actually played (exclude DNPs)
    df = df[df[TARGET] > 0].copy()
    print(f"After filtering DNPs: {len(df):,} rows")

    # Drop rows with missing features or target
    required_cols = FEATURES + [TARGET, 'season']
    df = df.dropna(subset=required_cols)
    print(f"After dropping NaN: {len(df):,} rows")

    # Train/test split by season
    train_seasons = ['2022-23', '2023-24', '2024-25']
    test_season = '2025-26'

    train_df = df[df['season'].isin(train_seasons)].copy()
    test_df = df[df['season'] == test_season].copy()

    print(f"\nTrain: {len(train_df):,} rows ({train_seasons})")
    print(f"Test: {len(test_df):,} rows ({test_season})")

    if len(test_df) == 0:
        print(f"WARNING: No test data for season {test_season}")
        print("Available seasons:", df['season'].unique())
        # Fallback: use latest 20% as test
        print("\nFalling back to 80/20 split by date...")
        df = df.sort_values('game_date')
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"Train: {len(train_df):,} rows")
        print(f"Test: {len(test_df):,} rows")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_model(X_train, y_train):
    """Train XGBoost model."""
    print("\nTraining XGBoost model...")

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Training complete!")

    return model


def calculate_variance_lookup(df, predictions):
    """Calculate residual standard deviation by player tier."""
    print("\nCalculating variance by player tier...")

    df = df.copy()
    df['prediction'] = predictions
    df['residual'] = df[TARGET] - df['prediction']

    variance_lookup = {}

    for tier_name in TIER_DEFINITIONS.keys():
        tier_df = df[get_tier_mask(df, tier_name)]
        if len(tier_df) > 0:
            std = tier_df['residual'].std()
            variance_lookup[tier_name] = {
                'std': std,
                'count': len(tier_df),
                'mean_actual': tier_df[TARGET].mean(),
                'mean_pred': tier_df['prediction'].mean()
            }
            print(f"  {tier_name:12s}: std={std:.2f}, n={len(tier_df):,}, "
                  f"mean_actual={tier_df[TARGET].mean():.1f}")
        else:
            print(f"  {tier_name:12s}: No data")

    return variance_lookup


def evaluate_model(model, X_test, y_test, test_df):
    """Evaluate model performance."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    predictions = model.predict(X_test)

    # Overall metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\nOverall Performance:")
    print(f"  MAE:  {mae:.2f} minutes {'✓' if mae < 5 else '✗ (target: <5)'}")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²:   {r2:.3f}")

    # Performance by player tier
    print(f"\nPerformance by Player Tier:")
    test_df = test_df.copy()
    test_df['prediction'] = predictions
    test_df['error'] = np.abs(test_df[TARGET] - test_df['prediction'])

    for tier_name in TIER_DEFINITIONS.keys():
        tier_df = test_df[get_tier_mask(test_df, tier_name)]
        if len(tier_df) > 0:
            tier_mae = tier_df['error'].mean()
            tier_rmse = np.sqrt(((tier_df[TARGET] - tier_df['prediction']) ** 2).mean())
            print(f"  {tier_name:12s}: MAE={tier_mae:.2f}, RMSE={tier_rmse:.2f}, "
                  f"n={len(tier_df):,}")

    # Feature importance
    print(f"\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.3f}")

    # Sample predictions
    print(f"\nSample Predictions (first 10 test cases):")
    print(f"{'Actual':>8s} {'Pred':>8s} {'Error':>8s}")
    print("-" * 26)
    for i in range(min(10, len(predictions))):
        actual = y_test.iloc[i]
        pred = predictions[i]
        error = actual - pred
        print(f"{actual:8.1f} {pred:8.1f} {error:+8.1f}")

    return predictions, mae, rmse, r2


def save_model(model, variance_lookup):
    """Save model and variance lookup."""
    print(f"\nSaving model to {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'variance_lookup': variance_lookup,
        'features': FEATURES,
        'target': TARGET,
        'tier_definitions': TIER_DEFINITIONS
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved successfully!")
    print(f"  Size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


def main():
    """Main execution."""
    print("="*70)
    print("MINUTES MODEL TRAINING")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test, train_df, test_df = load_and_prepare_data()

    # Train model
    model = train_model(X_train, y_train)

    # Calculate variance on training data
    train_predictions = model.predict(X_train)
    variance_lookup = calculate_variance_lookup(train_df, train_predictions)

    # Evaluate on test data
    test_predictions, mae, rmse, r2 = evaluate_model(model, X_test, y_test, test_df)

    # Save model
    save_model(model, variance_lookup)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel Performance Summary:")
    print(f"  MAE:  {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²:   {r2:.3f}")
    print(f"\nNext Step: Train stats model (4_train_stats_model.py)")


if __name__ == '__main__':
    main()
