#!/usr/bin/env python3
"""
Test the trained minutes model with sample predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load model
MODEL_PATH = Path(__file__).parent.parent / "models" / "minutes_model.pkl"

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
variance_lookup = model_data['variance_lookup']
tier_definitions = model_data['tier_definitions']

# Sample player scenarios
test_cases = [
    {
        'name': 'LeBron James (Star)',
        'min_L5_avg': 35.5,
        'min_szn_avg': 34.8,
        'is_starter': 1,
        'games_played_szn': 50,
        'rest_days': 1,
        'is_home': 1
    },
    {
        'name': 'Rotation Player',
        'min_L5_avg': 18.2,
        'min_szn_avg': 17.5,
        'is_starter': 0,
        'games_played_szn': 45,
        'rest_days': 2,
        'is_home': 0
    },
    {
        'name': 'Bench Player',
        'min_L5_avg': 8.5,
        'min_szn_avg': 9.1,
        'is_starter': 0,
        'games_played_szn': 38,
        'rest_days': 1,
        'is_home': 1
    }
]

print("MINUTES MODEL PREDICTIONS")
print("=" * 70)

for case in test_cases:
    name = case.pop('name')

    # Create DataFrame for prediction
    X = pd.DataFrame([case])

    # Get prediction
    pred = model.predict(X)[0]

    # Determine tier for variance
    min_szn = case['min_szn_avg']
    tier = None
    for tier_name, (min_val, max_val) in tier_definitions.items():
        if min_val <= min_szn < max_val:
            tier = tier_name
            break

    # Get variance
    std = variance_lookup[tier]['std'] if tier else 6.0

    print(f"\n{name}")
    print(f"  Prediction: {pred:.1f} minutes")
    print(f"  Tier: {tier}")
    print(f"  Std Dev: {std:.1f} minutes")
    print(f"  90% range: {pred - 1.645*std:.1f} - {pred + 1.645*std:.1f} minutes")
    print(f"  Input features: min_L5={case['min_L5_avg']:.1f}, min_szn={case['min_szn_avg']:.1f}, "
          f"starter={case['is_starter']}, rest={case['rest_days']}")

print("\n" + "=" * 70)
