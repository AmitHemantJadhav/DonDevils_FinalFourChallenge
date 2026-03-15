#!/usr/bin/env python3
"""Generate a live 2025-26 submission blended with public consensus."""

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


def _load_base_predictions() -> pd.DataFrame:
    submission = pd.read_csv(os.path.join(ROOT, "submissions", "submission.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "data", "raw", "NCAA_Seed_Test_Set2.0.csv"))
    merged = submission.merge(test_df[["RecordID", "Team", "Season"]], on="RecordID", how="left")
    merged["team_key"] = merged["Team"].map(_normalize_team_name)
    merged["our_seed_line"] = merged["Overall Seed"].apply(
        lambda x: math.ceil(x / 4) if pd.notna(x) and x > 0 else 0
    )
    return merged


def _load_sources(latest_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(os.path.join(latest_dir, "teamrankings_bracketology.csv"))
    tr = tr[tr["projected_tournament"]].copy()
    cbs = pd.read_csv(os.path.join(latest_dir, "cbs_bracketology.csv"))
    bm = pd.read_csv(os.path.join(latest_dir, "bracketmatrix_consensus.csv"))
    return tr, cbs, bm


def build_consensus_submission() -> str:
    latest_dir = _latest_projection_dir()
    ours = _load_base_predictions()
    tr, cbs, bm = _load_sources(latest_dir)

    merged = ours.copy()
    for name, df in [("tr", tr), ("cbs", cbs), ("bm", bm)]:
        keep = ["team_key", "projected_seed_line"]
        if name == "tr" and "projected_s_curve" in df.columns:
            keep.append("projected_s_curve")
        merged = merged.merge(
            df[keep].rename(
                columns={
                    "projected_seed_line": f"{name}_seed_line",
                    "projected_s_curve": "tr_s_curve",
                }
            ),
            on="team_key",
            how="left",
        )

    source_cols = ["tr_seed_line", "cbs_seed_line", "bm_seed_line"]
    merged["source_count"] = merged[source_cols].notna().sum(axis=1)
    merged["consensus_seed_line"] = merged[source_cols].mean(axis=1)
    merged["public_in_field"] = merged["source_count"] >= 1

    # Use consensus field membership first, then fill to 68 with the strongest
    # remaining teams from our model if the public snapshots don't cover 68 teams.
    field = merged[merged["public_in_field"]].copy()
    if len(field) < 68:
        extras = merged[~merged["public_in_field"]].copy()
        extras = extras.sort_values(["Overall Seed", "our_seed_line", "Team"])
        field = pd.concat([field, extras.head(68 - len(field))], ignore_index=True)
    elif len(field) > 68:
        field = field.sort_values(
            ["source_count", "consensus_seed_line", "tr_s_curve", "Overall Seed"],
            ascending=[False, True, True, True],
            na_position="last",
        ).head(68)

    # Final ordering: prioritize TeamRankings S-curve when present, otherwise
    # consensus seed line and then our exact model seed.
    field = field.sort_values(
        ["tr_s_curve", "consensus_seed_line", "Overall Seed", "source_count"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    field["Consensus Overall Seed"] = range(1, len(field) + 1)

    final = ours[["RecordID", "Team", "team_key"]].copy()
    final = final.merge(
        field[["team_key", "Consensus Overall Seed"]],
        on="team_key",
        how="left",
    )
    final["Overall Seed"] = final["Consensus Overall Seed"].fillna(0).astype(int)
    final = final.drop(columns=["Consensus Overall Seed"])

    out_path = os.path.join(ROOT, "submissions", "submission_consensus.csv")
    final[["RecordID", "Overall Seed"]].to_csv(out_path, index=False)
    print(out_path)
    print(f"tournament_teams={int((final['Overall Seed'] > 0).sum())}")
    return out_path


if __name__ == "__main__":
    build_consensus_submission()
