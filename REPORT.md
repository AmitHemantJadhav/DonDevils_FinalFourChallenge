# FFAC 2026 - NCAA Seed Prediction Report

**Team**: Don Devils
**Competition**: Final Four Analytics Challenge 2026, Round 1
**Objective**: Predict Overall Seed (1-68) for NCAA tournament teams, evaluated by RMSE

---

## Results Summary

| Submission | Kaggle RMSE | Notes |
|------------|-------------|-------|
| Baseline (naive) | 2.41 | Simple guessing approach |
| **Enhanced ensemble** | **1.796** | GBR+XGB+LGBM, external data, feature engineering |
| Leaderboard top | ~0.00-0.66 | Top competitors |

**Current tournament-only CV RMSE**: 4.95 (translates to ~2.22 leaderboard via `RMSE * sqrt(91/451)`)

---

## Dataset Overview

### Competition Data
- **Training**: 1,353 rows across 5 seasons (2020-21 to 2024-25)
  - 249 tournament teams (have Overall Seed 1-68)
  - 1,104 non-tournament teams (Overall Seed = NaN)
- **Test**: 451 rows (same 5 seasons, no team overlap with training)
  - ~91 tournament teams
  - ~360 non-tournament teams
- **Columns (20)**: RecordID, Season, Team, Conference, Overall Seed (target), Bid Type, NET Rank, PrevNET, AvgOppNETRank, AvgOppNET, WL, Conf.Record, Non-ConferenceRecord, RoadWL, NETSOS, NETNonConfSOS, Quadrant1-4

### External Data (Barttorvik via Kaggle)
- **Source**: Andrew Sundberg's College Basketball Dataset
- **Coverage**: Per-year files cbb21.csv through cbb25.csv
- **Key columns**: ADJOE, ADJDE, BARTHAG, WAB, EFG_O, EFG_D, TOR, TORD, ORB, DRB, FTR, FTRD, 2P_O, 2P_D, 3P_O, 3P_D, ADJ_T
- **Match rate**: 79.2% training, 78.7% test (fuzzy name matching)
- **13 tournament teams unmatched** in test set (improvement opportunity)

---

## Data Issues & Fixes

### 1. Excel Date Corruption (Critical)
W-L columns were corrupted by Excel auto-formatting dates:
- `"8-Sep"` should be `"8-9"` (8 wins, 9 losses)
- `"1-Jan"` should be `"1-1"` (1 win, 1 loss)
- **Fix**: Map month abbreviations back to numbers in `preprocessing.py`
- **Affected columns**: WL, Conf.Record, Non-ConferenceRecord, RoadWL, Quadrant1-4

### 2. Target Variable Interpretation (Critical)
- "Overall Seed" is a 1-68 S-curve ranking, NOT traditional 1-16 seeds
- Seed 1 = #1 overall pick, Seed 68 = last team in
- Non-tournament teams must be predicted as **0**
- Clipping in `submission.py`: `np.clip(predictions, 1, 68)`

### 3. RMSE Dilution Effect
- 360/451 test rows are non-tournament (predicted as 0, always correct)
- Leaderboard RMSE = `tournament_RMSE * sqrt(91/451) ≈ tournament_RMSE * 0.449`
- A tournament RMSE of 4.95 → leaderboard ~2.22

---

## Feature Engineering

### From W-L Columns (8 columns → 24 features)
Each W-L column produces `{prefix}_wins`, `{prefix}_losses`, `{prefix}_win_pct`:
- Total record, Conference record, Non-conference record, Road record
- Quadrant 1-4 records (Q1 wins heavily weighted by selection committee)

### Derived Features
| Feature | Description | Importance |
|---------|-------------|------------|
| `net_rank_change` | PrevNET - NET Rank (improvement) | Medium |
| `sos_diff` | NETSOS - NETNonConfSOS (SOS gap) | Medium |
| `total_q1q2_wins` | Q1 + Q2 wins combined | High |
| `total_q1q2_losses` | Q1 + Q2 losses combined | Medium |
| `total_q3q4_losses` | Q3 + Q4 losses (bad losses) | High |
| `q1q2_win_pct` | Combined Q1+Q2 win% | High |
| `conf_avg_net` | Mean NET rank of conference | Medium |
| `road_win_ratio` | Road wins / total wins | Low |
| `conf_nonconf_gap` | Conf win% - Non-conf win% | Low |
| `net_vs_opp` | NET rank - AvgOppNETRank | Medium |

### Season Percentile Features (Key Improvement)
Ranking within each season normalizes across years:
| Feature | Correlation with Seed |
|---------|----------------------|
| `wab_pct_season` | r = -0.92 (strongest) |
| `net_pct_season` | r = 0.88 |
| `barthag_pct_season` | r = -0.87 |
| `efficiency_margin_pct` | r = -0.86 |

### Interaction Features
- `adjoe_minus_adjde` — Offensive - defensive efficiency
- `wab_x_barthag` — WAB * BARTHAG interaction

---

## Model Architecture

### Ensemble (Best Model)
Weighted blend of three Optuna-tuned models with raw/ranked post-processing:

**Sub-models & Weights**:
| Model | Weight | Tuned RMSE (temporal CV) |
|-------|--------|--------------------------|
| GradientBoostingRegressor | 0.35 | 5.04 |
| XGBoost | 0.45 | 5.10 |
| LightGBM | 0.20 | 5.04 |

**Ensemble tournament CV RMSE**: 4.95

**Post-processing**: Each model's raw predictions are blended with rank-scaled predictions using per-model blend alphas, then combined with ensemble weights.

### Hyperparameter Tuning
- **Method**: Optuna Bayesian optimization (TPE sampler)
- **Trials**: 100-150 per model
- **CV**: Temporal leave-one-season-out (5 folds, one per season)

### Cross-Validation Strategy
- **Primary**: Temporal CV (leave-one-season-out) — prevents temporal leakage
- **5 folds**: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25
- **Only train on tournament teams** (249 total, ~200 train / ~50 val per fold)

---

## Pipeline Flow

```
1. Load CSVs (data_loader.py)
2. Fix W-L date corruption (preprocessing.py)
3. Clean & standardize column names (preprocessing.py)
4. Create derived features (feature_engineering.py)
5. Merge barttorvik external data (external_data.py)
6. Filter to tournament teams only for training
7. Drop non-feature columns (RecordID, Team, Season, Conference, Bid Type)
8. Handle missing values (median imputation)
9. One-hot encode remaining categoricals
10. Train ensemble on full tournament data
11. Predict test set, set non-tournament = 0
12. Clip predictions to [1, 68], round to int
13. Generate submission.csv
```

---

## Submission History & Lessons Learned

1. **Score 45.14** — Changed clipping from [1,16] to [1,68] but model was still predicting on wrong scale. Lesson: Always verify end-to-end pipeline before submitting.

2. **Score 14.00** — Tried converting to traditional 1-16 seeds via `ceil(overall_seed/4)`. Wrong format. Lesson: The competition uses Overall Seed (1-68), not traditional seeds.

3. **Score 2.41** — Baseline from original naive approach. Confirmed correct format.

4. **Score 1.796** — Full pipeline with feature engineering, external data, Optuna tuning, and ensemble. Correct format with Overall Seed (1-68) for tournament teams, 0 for non-tournament.

---

## File Structure

```
FinalFourDonDevils/
├── run_experiment.py          # CLI entrypoint
├── requirements.txt           # Python dependencies
├── CLAUDE.md                  # AI assistant instructions
├── REPORT.md                  # This file
├── PLAN.md                    # Improvement roadmap
├── src/
│   ├── base_model.py          # Abstract model base class
│   └── core/
│       ├── data_loader.py     # CSV loading
│       ├── preprocessing.py   # W-L fix, cleaning, encoding
│       ├── feature_engineering.py  # Derived features
│       ├── external_data.py   # Barttorvik data merge
│       ├── evaluate.py        # RMSE, CV utilities
│       ├── submission.py      # Submission generation
│       └── tuning.py          # Optuna tuning
├── models/
│   ├── _template/             # Model template
│   ├── random_forest/         # RF model + config
│   ├── xgboost/               # XGB model + config
│   ├── lightgbm/              # LGBM model + config
│   └── ensemble/              # Ensemble model + config
├── data/
│   ├── raw/                   # Competition CSVs
│   ├── external/              # Barttorvik per-year CSVs
│   └── processed/             # Cached data
├── submissions/               # Generated submission CSVs
└── dashboard/                 # HTML/JS data explorer
```
