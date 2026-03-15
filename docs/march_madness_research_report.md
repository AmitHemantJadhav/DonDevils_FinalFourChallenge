# 🏀 March Madness Prediction — Open Source Landscape Report

> Research compiled for **Team Don Devils** — Final Four Analytics Challenge 2026 (Seed Prediction / RMSE)

---

## Executive Summary

There is a **rich open-source ecosystem** for NCAA tournament prediction on GitHub and Kaggle. The approaches range from simple logistic regression baselines to deep learning with Transformers. Your competition's focus on **seed prediction (RMSE)** is somewhat unique — most projects predict **game-winner probabilities (log loss)** — but the feature engineering, data sources, and modeling techniques are directly transferable.

---

## Notable Open-Source Repositories

| Repository | Stars | Focus | Key Models | Language |
|---|---|---|---|---|
| [adeshpande3/March-Madness-ML](https://github.com/adeshpande3/March-Madness-ML) | ~1k | Game-winner prediction, extensible year-to-year | Logistic Regression, SVM, Random Forest | Python |
| [pjmartinkus/College_Basketball](https://github.com/pjmartinkus/College_Basketball) | — | Full pipeline: scraping → features → bracket | Random Forest, favorites vs. underdogs framing | Python |
| [sfirke/predicting-march-madness](https://github.com/sfirke/predicting-march-madness) | — | ML tutorial for Kaggle March Mania | End-to-end walkthrough | R |
| [kjaisingh/Forecasting-March-Madness](https://github.com/kjaisingh/Forecasting-March-Madness) | — | Google Cloud & NCAA ML Competition 2019 | Uses seedings + slots explicitly | Python |
| [bsrinath9/March-Madness-Prediction](https://github.com/bsrinath9/March-Madness-Prediction) | — | Winner prediction with seed features | Logistic Regression, Random Forest | Python |
| [pastormt/March-Madness-Predictions](https://github.com/pastormt/March-Madness-Predictions) | — | Regular season → tournament results | Uses tournament seeding, SOS, BPI | Python |
| [dlm1223/march-madness](https://github.com/dlm1223/march-madness) | — | Optimal bracket selection system | Customizable projection models | Python |
| [markjrieke/2025-march-madness](https://github.com/markjrieke/2025-march-madness) | — | Dynamic Bayesian model for 2025 | Bayesian inference, simulations | R |

---

## Machine Learning Models Used (by popularity)

### 🏆 Tier 1 — Most Common & Most Successful

| Model | Why It's Popular | Kaggle Track Record |
|---|---|---|
| **XGBoost** | Best single-model performer in many competitions | 1st place in multiple Kaggle March Mania editions |
| **LightGBM** | Fast training, handles categorical features natively | 1st and 3rd place solutions; often ensembled |
| **Logistic Regression** | Strong baseline, interpretable, fast | Competitive standalone; often part of ensembles |
| **Random Forest** | Robust to overfitting, handles noisy features well | Widely used as a reliable baseline |

### 🥈 Tier 2 — Advanced / Niche

| Model | Use Case |
|---|---|
| **Neural Networks (Dense)** | ~80% accuracy reported; some repos train on 16+ years of data |
| **LSTM** | Captures temporal patterns across a season; top 7.3% on ESPN Challenge (2025) |
| **Transformers** | Emerging — arXiv paper (2025) shows promise for complex team-feature relationships |
| **Bayesian Models** | Dynamic team strength modeling with uncertainty quantification |
| **KNN Regression** | Predicting spreads; sometimes combined with other models |
| **SVM** | Classification; generally outperformed by boosting methods |
| **AdaBoost** | Ensemble learning; less common than XGBoost/LightGBM |

### 🏅 Tier 3 — Specialty Approaches

| Approach | Description |
|---|---|
| **Elo Ratings** | FiveThirtyEight-inspired; season-long cumulative rating with margin-of-victory adjustments |
| **Monte Carlo Simulation** | Simulate entire bracket thousands of times to find optimal picks |
| **Ensemble / Stacking** | Combining 3-4 diverse models (XGB + LGBM + LR) is the most common winning strategy |

---

## Data Sources Used Across Projects

> [!IMPORTANT]
> The biggest differentiator in winning solutions is **data quality and feature engineering**, not model choice.

| Source | What It Provides | Used By |
|---|---|---|
| **Kaggle** (March Machine Learning Mania datasets) | Historical game results, seeds, team IDs, ordinals | Nearly all projects |
| **KenPom** (kenpom.com) | Adjusted efficiency (offense/defense), tempo, luck, SOS | Top-performing models |
| **Bart Torvik** (barttorvik.com / T-Rank) | Similar to KenPom with additional metrics | `pjmartinkus/College_Basketball`, others |
| **Sports-Reference.com** | Box-score stats, season summaries, advanced stats | `adeshpande3/March-Madness-ML`, many others |
| **FiveThirtyEight / 538** | Elo ratings, team strength projections | Kaggle ensembles (3rd place 2021) |
| **ESPN BPI** | Basketball Power Index | `pastormt/March-Madness-Predictions` |
| **HeatCheck CBB** | Advanced stats for college basketball | Neural network projects (2024 model) |

---

## Feature Engineering Patterns

The most impactful features across projects, **ordered by importance**:

### Core Features (used by nearly everyone)
1. **Seed / Seed Difference** — Single strongest predictor of tournament outcomes
2. **Adjusted Offensive/Defensive Efficiency** (KenPom AdjO, AdjD)
3. **Win Percentage / Win Count**
4. **Strength of Schedule (SOS)**
5. **Average Point Differential (Margin of Victory)**

### Advanced Features (used by winning solutions)
6. **Elo Ratings** — cumulative season rating
7. **Tempo-adjusted statistics** (KenPom AdjT)
8. **True Shooting Percentage**
9. **Turnover Rate** (offensive and defensive)
10. **Three-point shooting percentage** (made and attempted)
11. **Rebound Rate** (offensive and defensive)
12. **Assist-to-turnover ratio**

### Matchup Engineering (key differentiator)
- **Difference features**: For each stat, compute `Team_A_stat - Team_B_stat` to capture relative strengths
- **Favorite vs. Underdog framing**: Each game is structured with the higher-seeded team as "favorite"
- **Interaction features**: Cross-terms between offensive and defensive metrics

---

## Evaluation & Optimization Strategies

| Strategy | Details |
|---|---|
| **Log Loss** | Primary Kaggle metric — rewards calibrated probabilities |
| **RMSE** | ⬅️ **Your competition's metric** — less common but used in seed prediction tasks |
| **Cross-validation by year** | Leave-one-year-out CV prevents temporal leakage |
| **Post-processing** | Clip probabilities, adjust based on seed matchup history |
| **"Passive" vs "Aggressive" submissions** | Some Kaggle winners submit multiple brackets with different risk profiles |
| **Meta-modeling competitors** | One past winner modeled what other competitors would submit to differentiate |

---

## Relevance to Your Project (FinalFourDonDevils)

Your project already has **XGBoost, LightGBM, and Random Forest** set up — all three are Tier 1 models that dominate Kaggle. Here's what the research suggests you should focus on:

> [!TIP]
> ### High-Impact Actions Based on This Research
> 1. **Feature engineering > model tuning** — Winning solutions consistently cite features as the #1 differentiator
> 2. **Incorporate KenPom / Bart Torvik efficiency metrics** if your data allows — these are the most powerful predictive features after seed
> 3. **Seed difference is king** — Make sure your model has access to historical seed data and seed-related features
> 4. **Try an ensemble** — Blend your XGBoost + LightGBM + Random Forest predictions (weighted average or stacking)
> 5. **Use Elo ratings** as an additional feature — several repos provide Python implementations you can adapt
> 6. **Cross-validate by year** to avoid temporal leakage in your RMSE evaluation

> [!WARNING]
> ### Common Pitfalls to Avoid
> - Deep learning (LSTM/Transformers) generally **underperforms** gradient boosting for tabular NCAA data — don't over-invest here
> - Player-level features are powerful but **extremely difficult** to engineer correctly at scale
> - Most open-source projects predict game **winners**, not **seeds** — you'll need to adapt their feature ideas to your regression target

---

## Key Takeaway

The open-source community overwhelmingly favors **gradient boosting (XGBoost/LightGBM) + rich feature engineering** as the winning formula. Your project is already well-positioned with the right model infrastructure. The gap to close is in **feature depth** — especially incorporating advanced efficiency metrics (KenPom-style) and engineered matchup features.
