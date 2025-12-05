# Minutes Model Training Summary

## Model Performance

**Overall Metrics:**
- MAE: 4.71 minutes ✓ (Target: < 5 minutes)
- RMSE: 6.14 minutes
- R²: 0.656

**Training Data:**
- Total rows: 83,621 player-games (after filtering DNPs and NaN)
- Train: 76,914 rows (2022-23, 2023-24, 2024-25 seasons)
- Test: 6,707 rows (2025-26 season)

## Performance by Player Tier

| Tier | MAE | RMSE | Test Count |
|------|-----|------|------------|
| Stars (30+ min/game) | 3.91 | 5.46 | 1,586 |
| Starters (20-30 min) | 4.58 | 5.92 | 2,306 |
| Rotation (10-20 min) | 5.36 | 6.78 | 1,823 |
| Bench (<10 min) | 5.15 | 6.47 | 992 |

## Variance by Tier (for Monte Carlo)

| Tier | Std Dev | Training Count | Mean Actual |
|------|---------|----------------|-------------|
| Stars | 5.27 | 19,121 | 33.3 min |
| Starters | 6.22 | 25,521 | 25.6 min |
| Rotation | 7.15 | 23,174 | 16.6 min |
| Bench | 6.95 | 9,098 | 8.9 min |

## Feature Importance

| Feature | Importance |
|---------|------------|
| min_L5_avg | 0.905 |
| min_szn_avg | 0.064 |
| rest_days | 0.019 |
| games_played_szn | 0.008 |
| is_home | 0.004 |
| is_starter | 0.000 |

**Key Insight:** Recent minutes average (L5) dominates prediction (90.5% importance).

## Model Files

- **Model:** `/models/minutes_model.pkl` (502 KB)
- **Training Script:** `/scripts/3_train_minutes_model.py`
- **Test Script:** `/scripts/test_minutes_model.py`

## Usage Example

```python
import pickle
import pandas as pd

# Load model
with open('models/minutes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
variance_lookup = model_data['variance_lookup']

# Predict
X = pd.DataFrame([{
    'min_L5_avg': 35.5,
    'min_szn_avg': 34.8,
    'is_starter': 1,
    'games_played_szn': 50,
    'rest_days': 1,
    'is_home': 1
}])

prediction = model.predict(X)[0]  # 34.8 minutes
std = variance_lookup['Stars']['std']  # 5.27 minutes
```

## Next Steps

1. Train stats model (points, rebounds, assists per minute)
2. Build Monte Carlo simulation engine
3. Integrate injury adjustments
