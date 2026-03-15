# Don Devils - NCAA Tournament Seed Prediction

## Model Architecture Documentation

**Competition**: Final Four Analytics Challenge 2026 (Round 1)
**Team**: Don Devils
**Task**: Predict Overall Seed (1-68) for NCAA tournament teams
**Best Kaggle Score**: 0.365 RMSE

---

## 1. Problem Definition

Given a set of NCAA Division I basketball teams across 5 seasons (2020-21 through 2024-25), predict the **Overall Seed** (1-68 S-curve ranking) for tournament teams and **0** for non-tournament teams.

- **Overall Seed** is NOT the traditional 1-16 bracket seed. It is a 1-68 S-curve ranking where Seed 1 = best team overall and Seed 68 = the last team selected (last at-large bid or worst auto-qualifier).
- The leaderboard RMSE is diluted because 360 of 451 test rows are non-tournament teams predicted as 0 (perfect predictions). The effective formula is: `leaderboard_RMSE = tournament_RMSE * sqrt(91/451)`.

---

## 2. Datasets

### 2.1 Competition Data (BARK Portal)

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `NCAA_Seed_Training_Set2.0.csv` | 1,353 | 20 | Training set (249 tournament + 1,104 non-tournament) |
| `NCAA_Seed_Test_Set2.0.csv` | 451 | 19 | Test set (91 tournament + 360 non-tournament) |

**Raw columns** (before preprocessing):

| Column | Description |
|--------|-------------|
| RecordID | `{Season}-{Team}` identifier |
| Season | Academic year (e.g., `2020-21`) |
| Team | School name |
| Conference | Conference affiliation |
| Overall Seed | Target variable (1-68), training only |
| Bid Type | `AL` (at-large) or `AQ` (auto-qualifier), tournament teams only |
| NET Rank | NCAA Evaluation Tool ranking |
| PrevNET | Previous season's NET rank |
| AvgOppNETRank | Average opponent NET rank |
| AvgOppNET | Average opponent NET rating |
| WL | Overall win-loss record |
| Conf.Record | Conference win-loss record |
| Non-ConferenceRecord | Non-conference win-loss record |
| RoadWL | Road win-loss record |
| NETSOS | NET strength of schedule |
| NETNonConfSOS | NET non-conference strength of schedule |
| Quadrant1-4 | Win-loss records vs each quadrant of opponents |

**Data quirk**: W-L columns like `8-Sep` are Excel-corrupted dates that actually mean `8-9` (8 wins, 9 losses). The preprocessing pipeline detects and corrects this.

### 2.2 External Data (Barttorvik / Kaggle)

Source: [Andrew Sundberg's College Basketball Dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset)

Files: `cbb21.csv` through `cbb25.csv` (one per season, ~364 teams each)

| Column | Description | Used For |
|--------|-------------|----------|
| ADJOE | Adjusted Offensive Efficiency | Feature + disambiguation |
| ADJDE | Adjusted Defensive Efficiency | Feature + disambiguation |
| BARTHAG | Power rating (win probability vs avg team) | Feature |
| WAB | Wins Above Bubble | Feature (strongest single predictor, r = -0.92) |
| EFG_O / EFG_D | Effective field goal % (offense/defense) | Feature |
| TOR / TORD | Turnover rate (offense/defense) | Feature |
| ORB / DRB | Offensive/Defensive rebound rate | Feature |
| FTR / FTRD | Free throw rate (offense/defense) | Feature |
| 2P_O / 2P_D | Two-point shooting % | Feature |
| 3P_O / 3P_D | Three-point shooting % | Feature |
| ADJ_T | Adjusted tempo | Feature |
| SEED | Traditional NCAA bracket seed (1-16) | Constrained assignment |

A comprehensive name-matching system maps ~160+ competition team names to barttorvik names, including year-specific overrides (e.g., "LIU" maps to "LIU Brooklyn" in 2021-24 but "LIU" in 2025). Match rate: 99.1% of training teams, 98.4% of test teams.

---

## 3. Data Pipeline

```
Raw CSVs
  |
  v
[1. Preprocessing] -- Fix W-L date corruption, standardize column names
  |
  v
[2. Feature Engineering] -- Create 20+ domain features, select top 12
  |
  v
[3. External Data Merge] -- Add barttorvik stats (ADJOE, ADJDE, BARTHAG, WAB, etc.)
  |
  v
[4. Missing Value Imputation] -- Median fill for numeric columns
  |
  v
[5. Categorical Encoding] -- One-hot encode remaining string columns
  |
  v
[6. Feature Selection] -- Keep only 12 curated features
  |
  v
[7. Model Training] -- Stacking ensemble (GBR + XGB + LGBM)
  |
  v
[8. Multi-Seed Averaging] -- 5 ensemble models with different random seeds
  |
  v
[9. Constrained Seed Assignment] -- Map predictions to exact 1-68 slots
  |
  v
Submission CSV
```

### 3.1 Preprocessing (`src/core/preprocessing.py`)

1. **Strip whitespace** from all string columns
2. **Fix W-L date corruption**: Parse columns like `WL`, `Conf.Record`, `Quadrant1-4` from corrupted date strings (e.g., `8-Sep` -> 8 wins, 9 losses)
3. For each W-L column, extract: `{prefix}_wins`, `{prefix}_losses`, `{prefix}_win_pct`
4. **Standardize column names**: lowercase, replace spaces with underscores
5. **Missing value imputation**: Median fill for all numeric columns
6. **Categorical encoding**: One-hot encode remaining string columns (test data aligned to training columns)

### 3.2 Feature Engineering (`src/core/feature_engineering.py`)

Features are created from the parsed W-L columns and external data:

| Feature | Formula / Source | Rationale |
|---------|-----------------|-----------|
| `wab` | Barttorvik WAB | Wins Above Bubble; strongest predictor (r = -0.92) |
| `net_pct_season` | Within-season percentile of NET rank | Normalizes across seasons |
| `conf_avg_net` | Mean NET rank of conference peers | Conference strength proxy |
| `barthag` | Barttorvik BARTHAG | Power rating |
| `net_rank` | Raw NET rank | Direct selection committee input |
| `prevnet` | Previous season NET rank | Historical strength |
| `netsos` | NET strength of schedule | Schedule difficulty |
| `q1_win_pct` | Q1 wins / (Q1 wins + Q1 losses) | Quality win rate |
| `is_at_large` | 1 if bid_type = AL, else 0 | At-large vs auto-qualifier |
| `total_q1q2_wins` | Q1 wins + Q2 wins | Quality win volume |
| `total_q3q4_losses` | Q3 losses + Q4 losses | Bad loss count |
| `conf_wins` | Conference record wins | Conference performance |

**Why 12 features?** With only 249 tournament training samples, more features cause overfitting. Empirically tested: 8-12 features is optimal; 15+ degrades performance.

Additional features are computed but not selected (available for future use): `net_rank_change`, `sos_diff`, `road_win_ratio`, `conf_nonconf_gap`, `net_vs_opp`, `wab_pct_season`, `barthag_pct_season`, `adjoe_minus_adjde`, `efficiency_margin_pct`, `wab_x_barthag`, `q1q2_win_pct`.

---

## 4. Model Architecture

### 4.1 Stacking Ensemble

The core model is a **two-level stacking ensemble** with 3 gradient boosting base learners and a Ridge regression meta-learner.

```
                    Input Features (12)
                   /        |        \
                  v         v         v
              +------+  +------+  +------+
              |  GBR |  |  XGB |  | LGBM |
              +------+  +------+  +------+
                  \        |        /
                   v       v       v
              +-------------------------+
              |  Ridge Meta-Learner     |
              |  (trained on OOF preds) |
              +-------------------------+
                         |
                         v
                  Raw Prediction
```

**Base Learners** (all Optuna-tuned with 150 trials each):

| Model | Library | Key Hyperparameters |
|-------|---------|-------------------|
| **GBR** | sklearn GradientBoostingRegressor | 103 estimators, depth 5, lr 0.055, subsample 0.877 |
| **XGB** | XGBRegressor | 479 estimators, depth 5, lr 0.026, subsample 0.768 |
| **LGBM** | LGBMRegressor | 232 estimators, depth 5, lr 0.015, subsample 0.879, 22 leaves |

**Meta-Learner**: `RidgeCV` with alpha search over [0.01, 0.1, 1.0, 10.0, 100.0]. Typical learned weights: GBR ~0.24, XGB ~0.46, LGBM ~0.32.

**Out-of-Fold (OOF) Training**: The meta-learner is trained on 5-fold out-of-fold predictions from the base models, not on in-sample predictions. This prevents the stacker from overfitting to the base models' training-set outputs.

### 4.2 Multi-Seed Ensemble Averaging

To reduce prediction variance (which directly impacts disambiguation accuracy), 5 copies of the stacking ensemble are trained with different random seeds (42, 49, 56, 63, 70). The base model `random_state` parameters change, producing different tree structures and slightly different meta-learner weights. Final raw predictions are the **element-wise mean** across all 5 models.

### 4.3 Individual Models (Available but Not Used in Final Submission)

| Model | File | Description |
|-------|------|-------------|
| Random Forest | `models/random_forest/model.py` | sklearn RandomForestRegressor |
| XGBoost | `models/xgboost/model.py` | Standalone XGBRegressor |
| LightGBM | `models/lightgbm/model.py` | Standalone LGBMRegressor |

These can be run individually via `python run_experiment.py --model <name>` but the stacking ensemble consistently outperforms them.

---

## 5. Constrained Seed Assignment

The key insight that brought our score from 1.54 to 0.365: **we don't just predict a number, we assign teams to specific slots using constraints from the NCAA bracket structure.**

### 5.1 How It Works

1. **Extract traditional seeds (1-16)** from barttorvik's `SEED` column for all tournament teams
2. **Map traditional seeds to overall seed ranges**: Traditional seed 1 maps to overall seeds 1-4, seed 2 maps to 5-8, etc. (with play-in adjustments for seeds 11/16 which can have 5-6 teams)
3. **Identify occupied slots**: Training teams already fill some overall seed slots. The remaining slots are available for test teams.
4. **Group test teams by traditional seed line**: e.g., all test teams with trad seed 3 compete for available slots in overall seeds 9-12
5. **Rank within each group** using composite disambiguation scoring (see below)
6. **Assign**: Best-ranked team in each group gets the lowest available slot

### 5.2 Deterministic vs Ambiguous Teams

| Category | Count | Error |
|----------|-------|-------|
| Deterministic (1 team, 1 slot) | 26 / 91 | 0 (always correct) |
| 2-option groups | 46 | 0-1 per wrong pick |
| 3-option groups | 13 | 0-4 per wrong pick |
| 4-option groups | 6 | 0-9 per wrong pick |

### 5.3 Composite Disambiguation Scoring

For groups with multiple teams competing for multiple slots, a composite score determines the ranking:

```
composite = 0.69 * z_pred + 0.31 * z_netm + 0.27 * AQ_penalty
```

Where:
- **z_pred**: Z-score normalized ensemble raw prediction (within the group). Lower prediction = better team.
- **z_netm**: Z-score normalized NET_MARGIN (ADJOE - ADJDE from barttorvik), negated so higher margin = better. Provides complementary signal to the ensemble.
- **AQ_penalty**: 0.27 added for auto-qualifier teams (0 for at-large). Within a seed line, at-large teams typically get lower (better) overall seeds than auto-qualifiers.

The weights (0.69, 0.31, 0.27) were determined empirically via grid search over pairwise accuracy on training data.

**Edge cases handled**:
- Groups of size 1: Skip scoring entirely (deterministic)
- Zero standard deviation: Fall back to raw prediction only
- Missing barttorvik stats: Use ensemble prediction + AL penalty only
- Deficit groups (fewer slots than teams): Overflow teams assigned to nearest available slots

---

## 6. Evaluation

### 6.1 Cross-Validation Strategies

| Method | Description | Use Case |
|--------|-------------|----------|
| **K-Fold CV** | 5-fold random split | Quick iteration |
| **Temporal CV** | Leave-one-season-out | Primary evaluation (prevents temporal leakage) |

Training is always on **tournament teams only** (249 rows). Non-tournament teams are predicted as 0 without going through the model.

### 6.2 Hyperparameter Tuning

All base learner hyperparameters were tuned using **Optuna** (Bayesian optimization, 150 trials each) with temporal cross-validation as the objective function. This ensures hyperparameters generalize across seasons rather than overfitting to a single train/test split.

### 6.3 Score Progression

| Version | Approach | Kaggle RMSE |
|---------|----------|-------------|
| v1 | Single model, basic features | 1.796 |
| v2 | Stacking ensemble + feature selection | 1.539 |
| v3 | + Constrained seed assignment | 0.365 |
| v4 | + Composite disambiguation + multi-seed averaging | Pending |

---

## 7. Project Structure

```
FinalFourDonDevils/
|-- run_experiment.py              # CLI entrypoint
|-- src/
|   |-- base_model.py              # Abstract model base class
|   |-- core/
|       |-- data_loader.py         # CSV loading
|       |-- preprocessing.py       # W-L fix, cleaning, imputation, encoding
|       |-- feature_engineering.py # Feature creation + selection
|       |-- external_data.py       # Barttorvik merge + traditional seeds + stats
|       |-- seed_assignment.py     # Constrained assignment + disambiguation
|       |-- evaluate.py            # RMSE, K-fold CV, temporal CV
|       |-- submission.py          # Submission CSV generation
|       |-- tuning.py              # Optuna hyperparameter optimization
|-- models/
|   |-- ensemble/                  # Stacking ensemble (primary)
|   |-- random_forest/             # Individual RF model
|   |-- xgboost/                   # Individual XGB model
|   |-- lightgbm/                  # Individual LGBM model
|-- data/
|   |-- raw/                       # Competition CSVs
|   |-- external/                  # Barttorvik per-year files
|-- submissions/                   # Generated submission CSVs
|-- dashboard/                     # HTML/JS data explorer
```

---

## 8. How to Run

```bash
# Setup
pip install -r requirements.txt
# Download barttorvik data (one-time)
kaggle datasets download -d andrewsundberg/college-basketball-dataset -p data/external/ --unzip

# Train + generate submission
python run_experiment.py --model ensemble --submit

# Cross-validation
python run_experiment.py --model ensemble --temporal-cv

# Hyperparameter tuning
python run_experiment.py --model xgboost --tune --n-trials 150

# Compare all models
python run_experiment.py --compare
```
