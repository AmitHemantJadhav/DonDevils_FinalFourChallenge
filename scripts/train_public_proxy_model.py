#!/usr/bin/env python3
"""Train a live-season proxy model against public bracket projections."""

from __future__ import annotations

import math
import os
import re

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_BASE = os.path.join(ROOT, "data", "benchmarks", "public_projections")
CURRENT_SEASON_PATH = os.path.join(
    ROOT,
    "data",
    "external",
    "current_season",
    "2026-03-15",
    "current_season_external_competition_aligned_2026.csv",
)


def _normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _latest_projection_dir() -> str:
    entries = [
        os.path.join(BENCHMARK_BASE, name)
        for name in os.listdir(BENCHMARK_BASE)
        if os.path.isdir(os.path.join(BENCHMARK_BASE, name))
    ]
    if not entries:
        raise FileNotFoundError("No public projection snapshots found")
    return sorted(entries)[-1]


def _load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_dir = _latest_projection_dir()
    frame = pd.read_csv(CURRENT_SEASON_PATH).copy()
    frame["Team"] = frame["competition_team_comp"].fillna(frame.get("competition_team"))
    frame["team_key"] = frame["Team"].map(_normalize_team_name)

    tr = pd.read_csv(os.path.join(latest_dir, "teamrankings_bracketology.csv"))
    tr = tr[tr["projected_tournament"]].copy()
    cbs = pd.read_csv(os.path.join(latest_dir, "cbs_bracketology.csv"))
    bm = pd.read_csv(os.path.join(latest_dir, "bracketmatrix_consensus.csv"))
    for df in [tr, cbs, bm]:
        df["team_key"] = df["team_key"].map(_normalize_team_name)

    frame = frame.merge(
        tr[["team_key", "projected_seed_line", "projected_s_curve"]].rename(
            columns={"projected_seed_line": "tr_line"}
        ),
        on="team_key",
        how="left",
    )
    frame = frame.merge(
        cbs[["team_key", "projected_seed_line"]].rename(columns={"projected_seed_line": "cbs_line"}),
        on="team_key",
        how="left",
    )
    frame = frame.merge(
        bm[["team_key", "projected_seed_line"]].rename(columns={"projected_seed_line": "bm_line"}),
        on="team_key",
        how="left",
    )
    frame["source_count"] = frame[["tr_line", "cbs_line", "bm_line"]].notna().sum(axis=1)
    frame["field_target"] = (frame["source_count"] >= 1).astype(int)
    frame["seed_target_avg"] = frame[["tr_line", "cbs_line", "bm_line"]].mean(axis=1)
    frame["seed_target_wtd"] = (
        0.25 * frame["tr_line"].fillna(0)
        + 0.375 * frame["cbs_line"].fillna(0)
        + 0.375 * frame["bm_line"].fillna(0)
    ) / (
        0.25 * frame["tr_line"].notna()
        + 0.375 * frame["cbs_line"].notna()
        + 0.375 * frame["bm_line"].notna()
    ).replace(0, pd.NA)
    return frame, tr, cbs, bm


def _build_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    feature_cols = [
        "Team",
        "Conference",
        "NET Rank",
        "PrevNET",
        "AvgOppNETRank",
        "AvgOppNET",
        "NETSOS",
        "NETNonConfSOS",
        "wn_wab_rank",
        "wn_t_rank",
        "wn_kenpom_rank",
        "wn_bpi_rank",
        "wn_sor_rank",
        "wn_kpi_rank",
        "srs",
        "sos",
        "adjoe",
        "adjde",
        "barthag",
        "wab",
        "source_count",
    ]
    feature_cols = [c for c in feature_cols if c in frame.columns]
    categorical_cols = [c for c in ["Team", "Conference"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return preprocessor, feature_cols


def _fit_models(frame: pd.DataFrame, feature_cols: list[str], preprocessor: ColumnTransformer):
    X = frame[feature_cols]

    field_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=42,
                    n_estimators=800,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    field_model.fit(X, frame["field_target"])

    seed_train = frame[frame["seed_target_avg"].notna()].copy()
    seed_train["seed_target_avg"] = seed_train["seed_target_avg"].copy()
    seed_train.loc[seed_train["source_count"] == 1, "seed_target_avg"] += 0.5

    seed_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                ),
            ),
        ]
    )
    seed_model.fit(seed_train[feature_cols], seed_train["seed_target_avg"])
    return field_model, seed_model


def _build_submission(
    frame: pd.DataFrame,
    feature_cols: list[str],
    field_model: Pipeline,
    seed_model: Pipeline,
) -> pd.DataFrame:
    X = frame[feature_cols]
    field_prob = field_model.predict_proba(X)[:, 1]
    seed_pred = seed_model.predict(X)

    submission = frame[["RecordID", "Team", "team_key"]].copy()
    submission["score"] = seed_pred - 8.0 * field_prob
    field = submission.sort_values(["score", "Team"]).head(68).copy()
    field["Overall Seed"] = range(1, 69)
    submission = submission.merge(field[["team_key", "Overall Seed"]], on="team_key", how="left")
    submission["Overall Seed"] = submission["Overall Seed"].fillna(0).astype(int)
    submission["seed_line"] = submission["Overall Seed"].apply(lambda x: math.ceil(x / 4) if x > 0 else 0)
    return submission


def _print_source_rmse(submission: pd.DataFrame, tr: pd.DataFrame, cbs: pd.DataFrame, bm: pd.DataFrame) -> None:
    for source_name, df in [("TeamRankings", tr), ("CBS", cbs), ("Bracket Matrix", bm)]:
        cols = ["team_key", "projected_seed_line"]
        if "projected_s_curve" in df.columns:
            cols.append("projected_s_curve")
        merged = submission.merge(df[cols], on="team_key", how="inner")
        seed_line_rmse = mean_squared_error(merged["projected_seed_line"], merged["seed_line"]) ** 0.5
        print(
            f"{source_name}: seed_line_rmse={seed_line_rmse:.4f}, "
            f"mean_abs={(merged['projected_seed_line'] - merged['seed_line']).abs().mean():.4f}, "
            f"exact={(merged['projected_seed_line'] == merged['seed_line']).sum()}"
        )
        if "projected_s_curve" in merged.columns:
            s_curve_rmse = mean_squared_error(merged["projected_s_curve"], merged["Overall Seed"]) ** 0.5
            print(f"{source_name}: s_curve_rmse={s_curve_rmse:.4f}")


def main() -> None:
    frame, tr, cbs, bm = _load_training_frame()
    preprocessor, feature_cols = _build_preprocessor(frame)
    field_model, seed_model = _fit_models(frame, feature_cols, preprocessor)
    submission = _build_submission(frame, feature_cols, field_model, seed_model)

    out_path = os.path.join(ROOT, "submissions", "submission_public_proxy_model.csv")
    submission[["RecordID", "Overall Seed"]].to_csv(out_path, index=False)
    print(out_path)
    print(f"tournament_teams={(submission['Overall Seed'] > 0).sum()}")
    _print_source_rmse(submission, tr, cbs, bm)


if __name__ == "__main__":
    main()
