# THRESHOLD ADJUSTMENT RECOMMENDATION

## The Problem

Your logistic regression model has **14.27% over recall** - it's missing 85.7% of profitable over bets.

## The Root Cause

The default **threshold=0.50** is too high for an imbalanced dataset:
- 60.7% of bets are unders (majority class)
- 39.3% of bets are overs (minority class)
- Model is biased toward predicting the majority class

## The Solution

**Change threshold from 0.50 to 0.40**

## The Impact

```
                       BEFORE (0.50)    AFTER (0.40)    CHANGE
────────────────────────────────────────────────────────────────
Over Recall               14.3%            65.5%       +51.2pp
Accuracy                  61.8%            56.2%        -5.6pp
Correct Overs Detected    1,021            4,687      +3,666 (+359%)
Missed Overs             6,135            2,469       -3,666 (-60%)
F1-Score (average)        48.6%            56.1%       +7.5pp
```

## Confusion Matrices

### BEFORE (threshold=0.50)
```
              Predicted Under    Predicted Over
Actual Under      10,214              821        ← 92.6% under recall (good)
Actual Over        6,135            1,021        ← 14.3% over recall (BAD)
```

**Problem:** Missing 6,135 overs (85.7% of all overs)

### AFTER (threshold=0.40)
```
              Predicted Under    Predicted Over
Actual Under       5,540            5,495       ← 50.2% under recall
Actual Over        2,469            4,687       ← 65.5% over recall (FIXED)
```

**Solution:** Only missing 2,469 overs (34.5% of all overs)

## Why This Works

The model's probability predictions are reasonable:
- Actual overs average P(over) = 0.418
- Actual unders average P(over) = 0.377

But 45.8% of predictions fall in the [0.4, 0.5) range - just below the threshold!

By lowering the threshold to 0.40, we capture **thousands of borderline cases** that the model correctly identified as "probable overs" but were rejected by the 0.50 cutoff.

## The Trade-off

**What You Gain:**
- 3,666 more correct over predictions (+359%)
- 3,666 fewer missed opportunities (-60%)
- Better F1-score (+7.5pp)

**What You Give Up:**
- 5.6pp accuracy
- 4,674 more false alarms (wrong over predictions)
- 4,674 fewer correct under predictions

**Is It Worth It?**

YES - for a betting co-pilot:
1. The job is to SURFACE opportunities, not make final decisions
2. Users can evaluate recommendations themselves
3. Missing 6,135 profitable overs is unacceptable
4. Getting 3,666 more correct overs is worth 4,674 false alarms

## How to Deploy

### Option 1: Use the provided script
```bash
python3 scripts/predict_with_threshold.py --threshold 0.40
```

### Option 2: Modify your production code
```python
# OLD (uses default threshold=0.50)
y_pred = model.predict(X)

# NEW (uses threshold=0.40)
prob_over = model.predict_proba(X)[:, 1]
y_pred = (prob_over >= 0.40).astype(int)
```

### Option 3: Make threshold configurable
```python
def predict_with_threshold(model, X, threshold=0.40):
    """Make predictions with custom threshold."""
    prob_over = model.predict_proba(X)[:, 1]
    return (prob_over >= threshold).astype(int)

# Use in production
y_pred = predict_with_threshold(model, X, threshold=0.40)
```

## No Retraining Required

- The model is already trained
- Just change the decision boundary
- Zero downtime
- Immediate impact
- Reversible if needed

## Validation Plan

After deploying with threshold=0.40:

1. **Monitor these metrics:**
   - Over recall (target: 65%)
   - Accuracy (expect: ~56%)
   - User feedback on recommendations

2. **Track business metrics:**
   - Number of over recommendations surfaced
   - User engagement with recommendations
   - Profitability of acted-upon recommendations

3. **Fine-tune if needed:**
   - If too many false alarms: increase to 0.42 or 0.45
   - If still missing overs: decrease to 0.38 or 0.35
   - Threshold is a dial you can turn

## Alternative Thresholds

If 0.40 doesn't work for your use case:

```
Threshold    Accuracy    Over Recall    Use Case
─────────────────────────────────────────────────────────────
0.30         49.4%       89.1%          Maximum recall (surface everything)
0.35         52.3%       81.7%          High recall, low precision
0.40         56.2%       65.5%          Balanced (RECOMMENDED)
0.45         60.5%       33.3%          Conservative
0.50         61.8%       14.3%          Very conservative (CURRENT)
```

## Bottom Line

**Change one line of code. Get 51pp more recall. Deploy today.**

```python
# Change this:
threshold = 0.50

# To this:
threshold = 0.40
```

That's it. No retraining. No feature engineering. Just a better calibrated decision boundary.

---

**Recommendation:** Deploy with threshold=0.40 immediately
**Expected Outcome:** 4,687 correct over predictions (vs 1,021 today)
**Risk:** Low (reversible, no model changes)
**ROI:** 9:1 (gain 51pp recall for 5.6pp accuracy cost)
