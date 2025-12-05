"""
Train 7 XGBoost models to predict player stats: PTS, REB, AST, STL, BLK, TOV, FG3M

Input: data/processed/player_features_enriched.csv
Output: models/stats_models.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'player_features_enriched.csv'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = MODELS_DIR / 'stats_models.pkl'

# Stat configurations
STAT_CONFIGS = {
    'pts': {
        'target': 'actual_pts',
        'l5_feature': 'pts_L5_avg',
        'variance_buckets': [(0, 10), (10, 20), (20, 30), (30, 100)],
        'display_name': 'Points'
    },
    'reb': {
        'target': 'actual_reb',
        'l5_feature': 'reb_L5_avg',
        'variance_buckets': [(0, 5), (5, 10), (10, 100)],
        'display_name': 'Rebounds'
    },
    'ast': {
        'target': 'actual_ast',
        'l5_feature': 'ast_L5_avg',
        'variance_buckets': [(0, 5), (5, 10), (10, 100)],
        'display_name': 'Assists'
    },
    'stl': {
        'target': 'actual_stl',
        'l5_feature': 'stl_L5_avg',
        'variance_buckets': [(0, 2), (2, 5), (5, 100)],
        'display_name': 'Steals'
    },
    'blk': {
        'target': 'actual_blk',
        'l5_feature': 'blk_L5_avg',
        'variance_buckets': [(0, 2), (2, 5), (5, 100)],
        'display_name': 'Blocks'
    },
    'tov': {
        'target': 'actual_tov',
        'l5_feature': 'tov_L5_avg',
        'variance_buckets': [(0, 2), (2, 5), (5, 100)],
        'display_name': 'Turnovers'
    },
    'fg3m': {
        'target': 'actual_fg3m',
        'l5_feature': 'fg3m_L5_avg',
        'variance_buckets': [(0, 2), (2, 5), (5, 100)],
        'display_name': '3-Pointers Made'
    }
}

# Base features (used by all models)
BASE_FEATURES = [
    'min_L5_avg',              # Recent minutes
    'player_usage_proxy',      # Usage rate approximation
    'player_team_pts_share',   # Scoring role
    'opp_pts_allowed_L10',     # Opponent defense
    'is_home'                  # Home game indicator
]


def load_and_prepare_data():
    """Load data and split into train/test sets"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"Total rows: {len(df):,}")
    print(f"\nSeason distribution:")
    print(df['season'].value_counts().sort_index())

    # Filter to players who actually played
    df = df[df['actual_min'] > 0].copy()
    print(f"\nRows with actual_min > 0: {len(df):,}")

    # Split: 2022-23, 2023-24, 2024-25 = train; 2025-26 = test
    train_df = df[df['season'].isin(['2022-23', '2023-24', '2024-25'])].copy()
    test_df = df[df['season'] == '2025-26'].copy()

    print(f"\nTrain set: {len(train_df):,} rows (seasons 2022-23 to 2024-25)")
    print(f"Test set: {len(test_df):,} rows (season 2025-26)")

    return train_df, test_df


def calculate_variance_lookup(y_true, y_pred, variance_buckets):
    """
    Calculate residual std by prediction bucket for Monte Carlo simulation

    Returns dict mapping bucket (low, high) to residual std
    """
    residuals = y_true - y_pred
    variance_lookup = {}

    for low, high in variance_buckets:
        mask = (y_pred >= low) & (y_pred < high)
        if mask.sum() > 10:  # Need reasonable sample size
            bucket_std = residuals[mask].std()
            variance_lookup[(low, high)] = float(bucket_std)
        else:
            # Fallback to overall std if not enough samples
            variance_lookup[(low, high)] = float(residuals.std())

    return variance_lookup


def train_stat_model(stat_key, config, train_df, test_df):
    """Train a single stat model with variance capture"""

    print(f"\n{'='*80}")
    print(f"Training {config['display_name']} ({stat_key.upper()}) Model")
    print(f"{'='*80}")

    # Features for this model
    features = BASE_FEATURES + [config['l5_feature']]
    target = config['target']

    print(f"\nFeatures: {features}")
    print(f"Target: {target}")

    # Prepare train data (drop NaN in features or target)
    train_clean = train_df[features + [target]].dropna()
    X_train = train_clean[features]
    y_train = train_clean[target]

    print(f"\nTrain samples after dropping NaN: {len(X_train):,}")
    print(f"Target distribution (train):")
    print(y_train.describe())

    # Prepare test data
    test_clean = test_df[features + [target]].dropna()
    X_test = test_clean[features]
    y_test = test_clean[target]

    print(f"\nTest samples after dropping NaN: {len(X_test):,}")

    # Train model
    print(f"\nTraining XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"\n{'TRAIN METRICS':^40}")
    print(f"{'-'*40}")
    print(f"MAE:  {train_mae:.3f}")
    print(f"RMSE: {train_rmse:.3f}")
    print(f"R²:   {train_r2:.3f}")

    print(f"\n{'TEST METRICS':^40}")
    print(f"{'-'*40}")
    print(f"MAE:  {test_mae:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"R²:   {test_r2:.3f}")

    # Feature importance
    print(f"\n{'FEATURE IMPORTANCE':^40}")
    print(f"{'-'*40}")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")

    # Variance lookup (based on train predictions)
    variance_lookup = calculate_variance_lookup(
        y_train.values,
        train_pred,
        config['variance_buckets']
    )

    print(f"\n{'VARIANCE LOOKUP (for Monte Carlo)':^40}")
    print(f"{'-'*40}")
    for bucket, std in variance_lookup.items():
        print(f"Predictions {bucket[0]:2d}-{bucket[1]:2d}: std = {std:.3f}")

    # Sample predictions
    print(f"\n{'SAMPLE PREDICTIONS (Test Set)':^60}")
    print(f"{'-'*60}")
    sample_indices = np.random.choice(len(test_clean), min(10, len(test_clean)), replace=False)
    sample_df = test_clean.iloc[sample_indices].copy()
    sample_df['predicted'] = test_pred[sample_indices]
    sample_df['actual'] = y_test.iloc[sample_indices].values
    sample_df['error'] = sample_df['predicted'] - sample_df['actual']

    print(sample_df[['predicted', 'actual', 'error']].to_string(index=False))

    return {
        'model': model,
        'features': features,
        'variance_lookup': variance_lookup,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        },
        'feature_importance': feature_importance.to_dict('records')
    }


def main():
    """Train all 7 stat models"""

    print("="*80)
    print("TRAIN STATS MODELS - 7 PLAYER STAT PREDICTIONS")
    print("="*80)

    # Load data
    train_df, test_df = load_and_prepare_data()

    # Train all models
    all_models = {}
    summary_metrics = []

    for stat_key, config in STAT_CONFIGS.items():
        result = train_stat_model(stat_key, config, train_df, test_df)
        all_models[stat_key] = result

        summary_metrics.append({
            'stat': stat_key.upper(),
            'display_name': config['display_name'],
            'test_mae': result['metrics']['test']['mae'],
            'test_rmse': result['metrics']['test']['rmse'],
            'test_r2': result['metrics']['test']['r2']
        })

    # Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(all_models, f)

    print(f"\nSaved all 7 models to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY - ALL MODELS")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(summary_metrics)
    print(f"\n{summary_df.to_string(index=False)}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll 7 models trained and saved successfully!")
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"\nModels included:")
    for stat_key, config in STAT_CONFIGS.items():
        print(f"  - {config['display_name']} ({stat_key})")

    print(f"\nEach model includes:")
    print(f"  - Trained XGBoost model")
    print(f"  - Feature list")
    print(f"  - Variance lookup table (for Monte Carlo simulation)")
    print(f"  - Performance metrics")
    print(f"  - Feature importance")


if __name__ == '__main__':
    main()
