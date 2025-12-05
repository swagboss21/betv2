# Training Report - Logistic Regression Model v1

**Status:** COMPLETE
**Date:** Dec 3, 2025
**Subagent:** #4 (Model Training)

---

## Summary for Orchestrator

Trained logistic regression model on 90,955 labeled player prop records. Model shows statistical learning (better than baseline) but marginal improvement suggests need for feature engineering before production use.

---

## Key Results

### Model Performance
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **61.76%** |
| **Baseline** | 60.66% (always under) |
| **Improvement** | +1.10 percentage points |
| **Status** | Learning, but marginal |

### Class-Specific Metrics
```
Class 0 (Under hit):
  - Precision: 62.47%
  - Recall: 92.56%
  - F1-Score: 0.7460

Class 1 (Over hit):
  - Precision: 55.43%
  - Recall: 14.27%
  - F1-Score: 0.2269
```

### Confusion Matrix
```
              Predicted
             Under    Over
Actual  Under 10,214   821
        Over   6,135  1,021
```

**Problem:** Model biased toward predicting "under" (majority class). Misses 85.73% of actual "over" hits.

---

## Feature Importance (Top Signals)

### Strongest Signals for "Over" Prediction
1. **over_implied** (+0.3789) - Books' implied over probability (strongest signal)
2. **blocks+steals props** (+0.1605) - This stat type tends to hit over
3. **turnovers props** (+0.1339) - This stat type tends to hit over

### Strongest Signals for "Under" Prediction
1. **points+assists props** (-0.1602) - This combo tends to hit under
2. **points+rebounds props** (-0.1480) - This combo tends to hit under
3. **plain points props** (-0.0910) - Straight point props hit under more

**Interpretation:**
- Main signal is **over_implied** (books' pricing)
- Stat type matters (combos trade differently than singles)
- Model is learning real patterns, but with low confidence

---

## Training Configuration

### Data
- **Input:** `data/processed/training_data_full.csv` (95,714 rows)
- **Labeled rows:** 91,518
- **Used for training:** 90,955 (after dropping NaNs)
- **Train/test split:** 80/20 with stratification

### Features
1. `line` - Numerical (e.g., 15.5 points)
2. `over_implied` - Numerical (0-1 probability)
3. `under_implied` - Numerical (0-1 probability)
4. `stat_type` - Categorical, 13 categories (one-hot encoded)

### Pipeline
```python
Pipeline:
├── StandardScaler (numeric features)
├── OneHotEncoder (categorical features)
└── LogisticRegression (solver=lbfgs, max_iter=1000)
```

---

## Model Quality Assessment

### Is It Useful?

**VERDICT: NOT YET - MARGINAL IMPROVEMENT**

| Check | Result | Status |
|-------|--------|--------|
| Beats baseline? | Yes (+1.1%) | ✓ PASS |
| Significant improvement? | No (need 5%+) | ✗ FAIL |
| Learning real patterns? | Yes (uneven class preds) | ✓ PASS |
| Minority class recall? | No (14.27%) | ✗ FAIL |
| Production ready? | No | ✗ FAIL |

### Why Marginal?

1. **Limited features** - Only 4 features, 1 is categorical
   - Missing: player form, injuries, matchups, game context

2. **Highly correlated features** - over_implied, under_implied, line are all derived from same odds
   - Adds noise, reduces signal

3. **Class imbalance not handled** - Model defaults to predicting "under"
   - 92.56% recall for under, only 14.27% for over

4. **Books may be efficient** - Sportsbooks have sophisticated models
   - Simple betting odds may already reflect true probability

---

## Files Generated

| File | Purpose |
|------|---------|
| `scripts/train_model.py` | Training script (runnable, reproducible) |
| `models/logistic_v1.pkl` | Serialized model (2.7 KB) |
| `docs/model_v1_analysis.md` | Detailed analysis report |
| `TRAINING_REPORT_V1.md` | This file (executive summary) |

---

## How to Use the Model

```python
import pickle

# Load model
with open('models/logistic_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
X = pd.DataFrame({
    'line': [15.5],
    'over_implied': [0.45],
    'under_implied': [0.60],
    'stat_type': ['points']
})

prediction = model.predict(X)  # Returns [0] or [1]
probability = model.predict_proba(X)  # Returns [[P(under), P(over)]]
```

---

## Next Steps (For Orchestrator)

### IMMEDIATE (Session 5)
1. **Build v1.1 with class weighting** - LogisticRegression(class_weight='balanced')
   - Should improve "over" recall without losing much accuracy
   - Low effort, high potential improvement

2. **Implement threshold tuning** - Lower decision threshold from 0.5 to 0.35-0.40
   - Predict "over" more often
   - Measure Sharpe ratio on test set

### SHORT TERM (Session 6+)
3. **Feature engineering** - Create behavioral features:
   - Recent hit rate (last 10 games)
   - Player minutes
   - Line movement (check if open != close)
   - Opponent strength metrics

4. **Transition to XGBoost** - Once feature set expands
   - Handles non-linear patterns better
   - More powerful than logistic regression

### VALIDATION
5. **Walk-forward test** - Use recent games (never seen by model)
   - Simulate real trading
   - Measure expected value per prop

---

## Reproducibility

### To retrain:
```bash
cd /Users/noahcantu/Desktop/the-brain-organized\ 2
python3 scripts/train_model.py
```

### To run from clean slate:
```bash
# 1. Ensure training_data_full.csv exists
# 2. Run script (handles feature engineering + training)
# 3. Model saved to models/logistic_v1.pkl
```

---

## Data Quality Notes

- **Missing values in training data:** 4,196 rows (4.4% of labeled data)
  - over_implied: 10 NaNs
  - under_implied: 563 NaNs
  - Action: Dropped these rows before training

- **No data leakage** - Test set never seen during training

- **Stratified split** - Preserves 60.7/39.3 class distribution in both train and test

---

## Conclusion

Model v1 is a baseline proof-of-concept. It learns real patterns (better than random/baseline) but improvements are marginal. Ready for v1.1 iteration with class weighting and feature engineering.

Next iteration should target:
- +5%+ accuracy improvement (to 66%+)
- Better "over" detection (>30% recall minimum)
- Multiple model comparison (LR vs XGBoost)

**Status: HANDOFF TO ORCHESTRATOR FOR NEXT PHASE**
