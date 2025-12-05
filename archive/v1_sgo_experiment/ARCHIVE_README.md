# v1 SGO Experiment Archive

**Archived:** December 5, 2025
**Period:** December 2024 - December 2025

## What This Was

First iteration of NBA player prop betting model using SportsGameOdds API.

## Key Results

| Model | Accuracy | Over Recall |
|-------|----------|-------------|
| v1.0 | 61.76% | 14.27% |
| v1.1 | 57.65% | 65.39% |

## Key Insight

Overs are systematically overpriced:
- Actual over hit rate: **39.14%**
- Book implied probability: **47.65%**
- Market edge: **-8.51%**

Edge-based selection (bet UNDER on worst-edge players) outperformed ML prediction:
- At -10pp threshold: 67.5% accuracy
- At -20pp threshold: 80.8% accuracy

## Contents

```
v1_sgo_experiment/
├── models/           # logistic_v1.pkl, logistic_v1.1.pkl
├── scripts/          # Pipeline (1_fetch, 2_transform, 3_train, etc)
├── data/
│   ├── raw/          # ~210MB SGO API JSON (2023-24, 2024-25 seasons)
│   └── processed/    # training_data_full.csv (87K rows)
├── analysis/         # Edge analysis, threshold comparison scripts
├── predictions/      # Daily prediction outputs
└── docs/             # Training reports, schema docs, analysis reports
```

## Why Archived

Starting fresh with new data collection approach. This work was valuable for:
1. Validating SGO API as data source
2. Understanding market inefficiencies
3. Learning that simple edge-based approach beats ML complexity
