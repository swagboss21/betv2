"""
Example script showing how to use the trained stats models

This demonstrates how to:
1. Load the models
2. Make predictions
3. Get variance estimates for Monte Carlo simulation
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load models
MODELS_PATH = Path(__file__).parent.parent / 'models' / 'stats_models.pkl'

with open(MODELS_PATH, 'rb') as f:
    stats_models = pickle.load(f)

print("Loaded 7 stat models:", list(stats_models.keys()))


# Example: Predict stats for a player
def predict_player_stats(player_features):
    """
    Predict all 7 stats for a player

    Args:
        player_features: dict with required features

    Returns:
        dict with predictions and variance estimates
    """
    predictions = {}

    for stat_name, model_data in stats_models.items():
        model = model_data['model']
        features = model_data['features']
        variance_lookup = model_data['variance_lookup']

        # Create feature array in correct order
        X = np.array([[player_features[f] for f in features]])

        # Make prediction
        pred = model.predict(X)[0]

        # Get variance estimate based on prediction bucket
        variance = None
        for (low, high), std in variance_lookup.items():
            if low <= pred < high:
                variance = std
                break

        predictions[stat_name] = {
            'mean': round(pred, 2),
            'std': round(variance, 2) if variance else None
        }

    return predictions


# Example usage
example_player = {
    # Base features
    'min_L5_avg': 32.5,           # Playing 32.5 min/game recently
    'player_usage_proxy': 0.25,   # 25% usage rate
    'player_team_pts_share': 0.22, # 22% of team scoring
    'opp_pts_allowed_L10': 112.5, # Opponent allows 112.5 pts/game
    'is_home': 1,                 # Home game

    # Recent stat averages
    'pts_L5_avg': 24.8,
    'reb_L5_avg': 7.2,
    'ast_L5_avg': 8.1,
    'stl_L5_avg': 1.4,
    'blk_L5_avg': 0.8,
    'tov_L5_avg': 3.2,
    'fg3m_L5_avg': 2.6
}

print("\n" + "="*60)
print("EXAMPLE PREDICTION: High-usage wing player at home")
print("="*60)

predictions = predict_player_stats(example_player)

print("\nPredicted Stats (with variance for Monte Carlo):")
print("-" * 60)
for stat_name, pred in predictions.items():
    stat_display = stats_models[stat_name]['model'].__class__.__name__
    print(f"{stat_name.upper():5s}: {pred['mean']:6.2f} (std: {pred['std']})")

print("\n" + "="*60)
print("MONTE CARLO SIMULATION EXAMPLE")
print("="*60)

# Simulate 1000 games for this player
n_simulations = 1000
simulated_stats = {stat: [] for stat in predictions.keys()}

np.random.seed(42)
for _ in range(n_simulations):
    for stat_name, pred in predictions.items():
        # Sample from normal distribution
        simulated_value = np.random.normal(pred['mean'], pred['std'])
        simulated_value = max(0, simulated_value)  # Can't be negative
        simulated_stats[stat_name].append(simulated_value)

print(f"\nSimulated {n_simulations} games:")
print("-" * 60)
for stat_name in simulated_stats.keys():
    values = simulated_stats[stat_name]
    print(f"{stat_name.upper():5s}:")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  10th percentile: {np.percentile(values, 10):.2f}")
    print(f"  50th percentile: {np.percentile(values, 50):.2f}")
    print(f"  90th percentile: {np.percentile(values, 90):.2f}")
    print(f"  Max: {np.max(values):.2f}")

print("\n" + "="*60)
print("OVER/UNDER PROBABILITY EXAMPLE")
print("="*60)

# Example prop bets
prop_bets = [
    ('pts', 25.5, 'over'),
    ('reb', 7.5, 'over'),
    ('ast', 7.5, 'over'),
]

print("\nProp bet probabilities (based on simulation):")
print("-" * 60)
for stat, line, direction in prop_bets:
    if direction == 'over':
        hit_prob = np.mean(np.array(simulated_stats[stat]) > line)
    else:
        hit_prob = np.mean(np.array(simulated_stats[stat]) < line)

    print(f"{stat.upper()} {direction} {line}: {hit_prob*100:.1f}% probability")

print("\n" + "="*60)
