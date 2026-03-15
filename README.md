# NCAA Tournament Seed Prediction — Don Devils

**Final Four Analytics Challenge 2026, Round 1**

Predict the Overall Seed (1–68) for NCAA tournament teams using historical performance data and advanced basketball metrics. Non-tournament teams are predicted as 0. Submissions evaluated on RMSE.

---

## Our Approach

### Problem Framing

The target is the **S-curve seed (1–68)**, not the traditional 1–16 bracket seed. Seed 1 is the best team; seed 68 is the last qualifier. Non-tournament teams must be predicted as exactly 0, which dilutes leaderboard RMSE (only 91 of 451 test rows are tournament teams).

Effective leaderboard RMSE = `tournament_RMSE × sqrt(91/451)`

### Key Insight: WAB as Anchor

Wins Above Bubble (WAB) is the single strongest predictor (Pearson r ≈ −0.92 with Overall Seed). Most of our feature engineering builds on top of this signal.

### Feature Engineering

We use 12 features (empirically optimal for 249 training samples — more causes overfitting):

| Feature             | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `wab`               | Wins Above Bubble — most predictive single feature       |
| `net_pct_season`    | Season-normalized NET rank percentile                    |
| `conf_avg_net`      | Conference average NET rank (strength of schedule proxy) |
| `barthag`           | Barttorvik power rating                                  |
| `net_rank`          | Raw NET ranking                                          |
| `prevnet`           | Previous year NET rank                                   |
| `netsos`            | NET strength of schedule                                 |
| `q1_win_pct`        | Win % in Quadrant 1 games                                |
| `is_at_large`       | Binary: at-large bid vs. auto-qualifier                  |
| `total_q1q2_wins`   | Combined Q1+Q2 wins (quality win volume)                 |
| `total_q3q4_losses` | Combined Q3+Q4 losses (bad loss penalty)                 |
| `conf_wins`         | Conference win count                                     |

External data from Barttorvik (ADJOE, ADJDE, BARTHAG, WAB, SEED) is merged per season.

### Model Architecture

**3-Model Stacking Ensemble** (GBR + XGBoost + LightGBM):

- Each model trained only on tournament teams (249 rows)
- Predictions are a weighted blend: `weights = [0.55 GBR, 0.35 XGB, 0.10 LGBM]`
- Each model's raw prediction is mixed with rank-scaled prediction via per-model `blend_alpha`
- Final seeds are clipped to [1, 68]

### Constrained Seed Assignment

Barttorvik provides traditional seeds (1–16) for tournament teams. We use these to assign S-curve slots deterministically where possible:

- Each traditional seed maps to known S-curve slot ranges (e.g., trad seed 1 → slots 1, 2, 3, 4)
- For traditional seeds with exactly one unoccupied slot, assignment is deterministic (26/91 test teams)
- For ambiguous cases, we rank teams by ensemble score within each traditional-seed group

This hybrid approach — constrained assignment where certain, ML ranking where uncertain — yielded our best scores.

### Cross-Validation Strategy

Primary evaluation uses **leave-one-season-out temporal CV** (5 folds, one per season 2021–2025) to prevent leakage from teams appearing across seasons.

---

## Results

| Model                       | Leaderboard RMSE |
| --------------------------- | ---------------- |
| Random Forest (baseline)    | ~2.1             |
| XGBoost + feature selection | ~1.8             |
| 3-model stacking ensemble   | 1.54             |
| Constrained seed assignment | 0.365            |

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Setup

**Competition data** (required): Download from the [BARK portal](http://analyticschallenge.butler.edu) and place CSVs in `data/raw/`.

**External Barttorvik data** (required for best performance):

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d andrewsundberg/college-basketball-dataset -p data/external/ --unzip
```

Expected files in `data/external/`: `cbb21.csv`, `cbb22.csv`, `cbb23.csv`, `cbb24.csv`, `cbb25.csv`

### 3. Run the Ensemble Model (Best)

```bash
# Train and evaluate with cross-validation
python run_experiment.py --model ensemble --cv 10

# Generate submission CSV
python run_experiment.py --model ensemble --submit
```

Submission will be saved to `submissions/submission.csv`.

### 4. Other Commands

```bash
# Run individual models
python run_experiment.py --model random_forest
python run_experiment.py --model xgboost
python run_experiment.py --model lightgbm

# Temporal CV (leave-one-season-out) — primary evaluation
python run_experiment.py --model ensemble --temporal-cv

# Compare all models
python run_experiment.py --compare

# Hyperparameter tuning (Optuna, ~100 trials)
python run_experiment.py --model xgboost --tune --n-trials 100
```

---

## Project Structure

```
├── run_experiment.py          # CLI entrypoint — orchestrates full pipeline
├── requirements.txt
│
├── src/
│   ├── base_model.py          # Abstract BaseModel class
│   └── core/
│       ├── data_loader.py     # CSV loading with validation
│       ├── preprocessing.py   # W-L date fix, cleaning, imputation, encoding
│       ├── feature_engineering.py  # Season percentiles, quality win composites
│       ├── external_data.py   # Barttorvik data merge
│       ├── seed_assignment.py # Constrained slot assignment logic
│       ├── evaluate.py        # RMSE, k-fold CV, temporal CV
│       ├── submission.py      # Submission CSV generation
│       └── tuning.py          # Optuna Bayesian hyperparameter search
│
├── models/
│   ├── ensemble/              # Weighted blend: GBR + XGBoost + LightGBM
│   ├── xgboost/
│   ├── lightgbm/
│   ├── random_forest/
│   └── _template/             # Starter template for new algorithms
│
├── dashboard/
│   └── index.html             # Standalone data explorer (open in browser)
│
├── submissions/
│   └── submission.csv         # Latest submission
│
└── docs/
    ├── model_architecture.md
    └── march_madness_research_report.md
```

---

## Adding a New Model

1. Copy `models/_template/` → `models/your_algo/`
2. Implement `train()` and `predict()` from `BaseModel` in `model.py`
3. Configure hyperparameters in `config.yaml`
4. Run: `python run_experiment.py --model your_algo`

---

## Team

**Don Devils** — Amit Jadhav, Shaurya Beriwala, Abhisar Anand - competing in the Final Four Analytics Challenge 2026.
