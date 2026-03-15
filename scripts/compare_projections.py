#!/usr/bin/env python3
"""Compare the current submission against stored public projections."""

from __future__ import annotations

import math
import os
import re

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_BASE = os.path.join(ROOT, "data", "benchmarks", "public_projections")


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


def _load_our_predictions() -> pd.DataFrame:
    submission = pd.read_csv(os.path.join(ROOT, "submissions", "submission.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "data", "raw", "NCAA_Seed_Test_Set2.0.csv"))
    merged = submission.merge(test_df[["RecordID", "Team", "Season"]], on="RecordID", how="left")
    merged["team_key"] = merged["Team"].map(_normalize_team_name)
    merged["our_seed_line"] = merged["Overall Seed"].apply(
        lambda x: math.ceil(x / 4) if pd.notna(x) and x > 0 else 0
    )
    return merged


def _compare_one(ours: pd.DataFrame, source_name: str, path: str) -> tuple[pd.DataFrame, dict]:
    ext = pd.read_csv(path)
    if "projected_tournament" in ext.columns:
        ext_field = ext[ext["projected_tournament"]].copy()
    else:
        ext_field = ext[ext["projected_seed_line"].notna()].copy()
    merged = ours.merge(
        ext_field[["team_key", "team", "projected_seed_line"]],
        on="team_key",
        how="inner",
        suffixes=("_ours", "_source"),
    )
    merged["seed_line_delta"] = merged["our_seed_line"] - merged["projected_seed_line"]
    seed_line_rmse = None
    if len(merged):
        seed_line_rmse = float(((merged["seed_line_delta"] ** 2).mean()) ** 0.5)

    s_curve_rmse = None
    if "projected_s_curve" in ext_field.columns:
        merged = merged.merge(
            ext_field[["team_key", "projected_s_curve"]],
            on="team_key",
            how="left",
        )
        merged["s_curve_delta"] = merged["Overall Seed"] - merged["projected_s_curve"]
        if len(merged):
            s_curve_rmse = float(((merged["s_curve_delta"] ** 2).mean()) ** 0.5)

    summary = {
        "source": source_name,
        "matches": int(len(merged)),
        "mean_abs_seed_line_delta": float(merged["seed_line_delta"].abs().mean()) if len(merged) else None,
        "seed_line_rmse": seed_line_rmse,
        "s_curve_rmse": s_curve_rmse,
        "exact_seed_line_matches": int((merged["seed_line_delta"] == 0).sum()) if len(merged) else 0,
        "our_tournament_teams": int((ours["Overall Seed"] > 0).sum()),
        "source_projected_teams": int(len(ext_field)),
    }
    return merged, summary


def _write_outputs(latest_dir: str, comparisons: list[tuple[str, pd.DataFrame]], summaries: list[dict]) -> None:
    comparison_dir = os.path.join(latest_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    for source_name, df in comparisons:
        out_name = f"{source_name.lower().replace(' ', '_')}_comparison.csv"
        df.to_csv(os.path.join(comparison_dir, out_name), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(comparison_dir, "summary.csv"), index=False)


def main() -> None:
    latest_dir = _latest_projection_dir()
    ours = _load_our_predictions()

    comparisons = []
    summaries = []
    for source_name, filename in [
        ("Bracket Matrix", "bracketmatrix_consensus.csv"),
        ("CBS", "cbs_bracketology.csv"),
        ("TeamRankings", "teamrankings_bracketology.csv"),
    ]:
        path = os.path.join(latest_dir, filename)
        if not os.path.exists(path):
            continue
        comparison_df, summary = _compare_one(ours, source_name, path)
        comparisons.append((source_name, comparison_df))
        summaries.append(summary)

    _write_outputs(latest_dir, comparisons, summaries)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
