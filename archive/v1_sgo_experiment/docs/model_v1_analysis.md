# Logistic Regression Model v1 - Analysis Report

**Date:** Dec 3, 2025
**Model:** `models/logistic_v1.pkl`
**Training Script:** `scripts/train_model.py`
**Status:** MARGINAL - Model shows learning but needs enhancement

---

## Executive Summary

**The model is learning, but barely outperforming baseline.**

- **Test Accuracy:** 61.76%
- **Baseline Accuracy:** 60.66% (always predicting "under")
- **Improvement:** +1.10 percentage points
- **Verdict:** Model shows statistical learning (not just memorizing majority class) but needs more features/data for practical utility

---

## Training Data Summary

| Metric | Value |
|--------|-------|
| Total rows | 95,714 |
| Labeled rows | 91,518 |
| Used for training | 90,955 (after dropping NaNs) |
| Training set | 72,764 (80%) |
| Test set | 18,191 (20%) |
| Stratified split | Yes (preserves 60.7/39.3 class distribution) |

### Target Distribution
- **Class 0 (under hit):** 60.7% (55,173 samples)
- **Class 1 (over hit):** 39.3% (35,782 samples)

**Key Insight:** Overs systematically underperform (60.7% of props lose). This is the baseline we must beat.

---

## Model Architecture

```
Pipeline:
├── StandardScaler (numeric features: line, over_implied, under_implied)
├── OneHotEncoder (categorical: 13 stat types)
└── LogisticRegression (solver=lbfgs, max_iter=1000)
```

### Features Used
1. **line** - Numerical bet line (e.g., 15.5 points)
2. **over_implied** - Implied probability of over (0-1)
3. **under_implied** - Implied probability of under (0-1)
4. **stat_type** - One-hot encoded (13 categories: points, assists, rebounds, etc.)

---

## Performance Metrics

### Overall Accuracy
```
Test Accuracy:     61.76%
Baseline (class 0): 60.66%
Improvement:       +1.10 percentage points
```

### Class-Specific Performance

| Metric | Class 0 (Under) | Class 1 (Over) |
|--------|-----------------|----------------|
| Precision | 62.47% | 55.43% |
| Recall | 92.56% | 14.27% |
| F1-Score | 0.7460 | 0.2269 |
| Support | 11,035 | 7,156 |

**Interpretation:**
- **Class 0 bias:** Model is heavily biased toward predicting "under" (92.56% recall, but catches only 14.27% of "overs")
- **Precision:** When model predicts "under," it's right 62.47% of the time
- **Precision for overs:** Only 55.43% - many false positives when predicting "over"

### Confusion Matrix
```
              Predicted
             Under    Over
Actual  Under 10,214   821
        Over   6,135  1,021
```

**What this means:**
- Out of 7,156 actual "over" hits, the model:
  - Correctly identifies 1,021 (14.27% recall)
  - Misses 6,135 (85.73%)
- Out of 11,035 actual "under" hits, the model:
  - Correctly identifies 10,214 (92.56% recall)
  - Mispredicts only 821 (7.44%)

---

## Feature Importance (Logistic Regression Coefficients)

### Top Features Favoring "Over" Prediction
| Feature | Coefficient | Interpretation |
|---------|-------------|-----------------|
| over_implied | +0.3789 | Higher implied probability for over = more likely to predict over ✓ |
| stat_type_blocks+steals | +0.1605 | Blocks+steals props tend to hit over more |
| stat_type_turnovers | +0.1339 | Turnovers tend to hit over more |
| line | +0.0367 | Higher lines slightly favor over (weaker signal) |
| stat_type_steals | +0.0327 | Steals props favor over |

### Top Features Favoring "Under" Prediction
| Feature | Coefficient | Interpretation |
|---------|-------------|-----------------|
| stat_type_points+assists | -0.1602 | Points+assists props strongly favor under |
| stat_type_points+rebounds | -0.1480 | Points+rebounds props favor under |
| stat_type_rebounds+assists | -0.0983 | Rebounds+assists props favor under |
| stat_type_points | -0.0910 | Plain points props favor under |
| under_implied | -0.0889 | Higher implied probability for under = more likely to predict under ✓ |

### Key Insight
The strongest signal is **over_implied** (+0.3789), meaning:
- When sportsbooks give a low implied probability to the over, the model learns that overs are less likely to hit
- This is sensible: if books price over at 55%, it means they expect under to win (even accounting for vig)

---

## Model Quality Assessment

### Is the Model Useful?

**SHORT ANSWER:** Not yet. It's learning real patterns (+1.1% above baseline), but the improvement is too marginal.

**Verdict Matrix:**

| Criterion | Result | Status |
|-----------|--------|--------|
| Better than baseline? | Yes (+1.1%) | ✓ PASS (barely) |
| Significant improvement? | No (need +5%+) | ✗ FAIL |
| Class 1 recall acceptable? | No (14.27%) | ✗ FAIL |
| Learning real patterns? | Yes (uneven predictions) | ✓ PASS |

### Why So Marginal?

1. **Limited features:** Only 4 features (1 is categorical with limited variance)
   - Missing: recent player form, injury status, game context, opponent strength

2. **Signal obscured by noise:** Raw betting lines may already be well-calibrated
   - Books have sophisticated models; our simple features can't easily beat them

3. **Class imbalance:** 60/40 split, but model struggles with minority class (over)
   - Model learns "underperform is default" and doesn't learn when to predict over

4. **Potential data quality issues:**
   - implied probabilities are derived from odds (redundant with other features)
   - line + implied odds are highly correlated

---

## Recommendations for Improvement

### Priority 1: Add Behavioral Features
```
- Recent hit rate (last 10 games) per player/stat_type
- Line movement from open to close
- Consensus prediction volume
- Player minutes played (if available)
```

### Priority 2: Fix Class Imbalance
```
Options:
a) Use class_weight='balanced' in LogisticRegression
b) Apply SMOTE (Synthetic Minority Oversampling)
c) Adjust decision threshold (currently 0.5)
   - Lower to 0.35 to predict more "overs"
```

### Priority 3: Engineer Better Features
```
- Interaction terms (line × stat_type)
- Polynomial features (line²)
- Odds ratio features (over_odds / under_odds)
- Time-based features (day of week, season stage)
```

### Priority 4: Try Better Models
```
Current: Logistic Regression (linear model)
Next:   - XGBoost (captures non-linear patterns)
        - Random Forest (ensemble learning)
        - Neural Networks (if enough data)
```

---

## Reproducibility

### To retrain:
```bash
cd /Users/noahcantu/Desktop/the-brain-organized\ 2
python3 scripts/train_model.py
```

### To use the model:
```python
import pickle

with open('models/logistic_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)  # 0 or 1
probabilities = model.predict_proba(X_test)  # [[P(under), P(over)], ...]
```

---

## Data Quality Notes

### Missing Values Handled
| Column | Missing | Action |
|--------|---------|--------|
| over_implied | 10 | Dropped (0.01% data loss) |
| under_implied | 563 | Dropped (0.6% data loss) |
| Other features | 0 | None |

Total data loss: ~4,196 rows (0.6% of training set)

---

## Next Steps

1. **Implement baseline v2 with class weighting** - Should improve "over" recall without much accuracy loss
2. **Extract behavioral features** - Create aggregations per player/stat_type
3. **Investigate line movement** - Check if opening line ≠ closing line (SGO data showed none, but verify)
4. **A/B test threshold adjustment** - Lower decision threshold to 0.35-0.40, measure Sharpe ratio
5. **Transition to XGBoost** - Once we have rich feature set

---

## Files Generated

- **Model:** `/Users/noahcantu/Desktop/the-brain-organized 2/models/logistic_v1.pkl` (2.7 KB)
- **Script:** `/Users/noahcantu/Desktop/the-brain-organized 2/scripts/train_model.py`
- **Report:** This file

---

**Model trained and validated on 90,955 labeled samples. Ready for next iteration.**
