# The Brain - NBA Player Prop Prediction

AI-powered NBA player prop betting analysis tool.

## What It Does

Predicts whether player props will hit OVER or UNDER based on historical data.

**Example:** "Will LeBron score over 25.5 points?" → Model predicts OVER or UNDER

## Current Status

- **Model:** Logistic Regression v1
- **Accuracy:** 61.76% (baseline: 60.66%)
- **Data:** ~96,000 historical player props (2023-24 & 2024-25 seasons)

## Quick Start

### Make a Prediction

```python
import pickle
import pandas as pd

# Load model
with open('models/logistic_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
prop = pd.DataFrame({
    'line': [25.5],
    'over_implied': [0.48],
    'under_implied': [0.57],
    'stat_type': ['points']
})

prob = model.predict_proba(prop)[0]
print(f"Over: {prob[1]:.1%}, Under: {prob[0]:.1%}")
```

### Run the Pipeline

```bash
# 1. Fetch data from SGO API
python scripts/1_fetch_data.py

# 2. Transform JSON to CSV
python scripts/2_transform_data.py

# 3. Train the model
python scripts/3_train_model.py

# 4. Make predictions
python scripts/4_predict.py
```

## Project Structure

```
the-brain/
├── scripts/           # Core pipeline (numbered 1-4)
├── analysis/          # Analysis tools
├── data/raw/          # Source JSON from SGO API
├── data/processed/    # Training CSV
├── models/            # Trained model (.pkl)
└── docs/              # Documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Full project context and decisions |
| `docs/SHORTCOMINGS_REPORT_VERIFIED.md` | Known issues and improvement roadmap |
| `models/logistic_v1.pkl` | Trained prediction model |

## Next Steps

See `docs/SHORTCOMINGS_REPORT_VERIFIED.md` for improvement roadmap:
1. Lower prediction threshold (0.50 → 0.40) for better over detection
2. Add player_id as feature (+3-5% accuracy)
3. Address class imbalance with balanced weighting
