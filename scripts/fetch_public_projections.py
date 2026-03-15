#!/usr/bin/env python3
"""Fetch public NCAA tournament projections into versioned CSV files."""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import urllib3


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_BASE = os.path.join(ROOT, "data", "benchmarks", "public_projections")
RUN_DATE = datetime.now(timezone.utc).astimezone().date().isoformat()
OUTPUT_DIR = os.path.join(OUTPUT_BASE, RUN_DATE)

TEAM_RECORD_RE = re.compile(r"^(?P<team>.+?)\((?P<record>\d+-\d+)\)$")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _split_team_record(value: object) -> tuple[str | None, str | None]:
    text = str(value).strip()
    if not text or text == "nan":
        return None, None
    match = TEAM_RECORD_RE.match(text)
    if match:
        return match.group("team").strip(), match.group("record")
    return text, None


def fetch_teamrankings() -> str:
    url = "https://www.teamrankings.com/ncaa-tournament/bracketology/"
    tables = pd.read_html(url)
    df = tables[0].copy()
    df.columns = [
        " ".join(str(part).strip() for part in col if str(part) != "nan").strip()
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]
    df = df.rename(
        columns={
            "Projected Seed Seed": "projected_seed_line",
            "Projected Seed S-Curve": "projected_s_curve",
            "Current & Projected Record Team": "team_with_record",
            "Current & Projected Record Proj W/L": "projected_record",
            "Bracketology Odds Bid": "ncaa_bid_odds",
            "Bracketology Odds Auto Bid": "auto_bid_odds",
            "Bracketology Odds 1 Seed": "one_seed_odds",
        }
    )
    df["team"], extracted_record = zip(*df["team_with_record"].map(_split_team_record))
    df["display_record"] = extracted_record
    df["team_key"] = df["team"].map(_normalize_team_name)
    df["projected_tournament"] = df["projected_s_curve"] <= 68
    out_path = os.path.join(OUTPUT_DIR, "teamrankings_bracketology.csv")
    df.to_csv(out_path, index=False)
    return out_path


def fetch_cbs() -> str:
    url = "https://new.cbssports.com/college-basketball/bracketology/"
    tables = pd.read_html(url)
    df = tables[2].copy()
    df = df[df["Seed"].notna() & df["Team"].notna()].copy()
    df["projected_seed_line"] = df["Seed"].astype(int)
    df = df.rename(columns={"Seed.1": "region"})
    df["team"], df["display_record"] = zip(*df["Team"].map(_split_team_record))
    df["team_key"] = df["team"].map(_normalize_team_name)
    keep_cols = [
        "team",
        "team_key",
        "display_record",
        "projected_seed_line",
        "region",
        "Conf",
        "Net*",
        "Quad 1*",
        "Quad 2*",
        "Quad 3*",
        "Quad 4*",
        "Sos*",
        "Next Game",
    ]
    out_path = os.path.join(OUTPUT_DIR, "cbs_bracketology.csv")
    df[keep_cols].to_csv(out_path, index=False)
    return out_path


def fetch_bracketmatrix() -> str:
    url = "https://www.bracketmatrix.com/"
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0].copy()
    df = df[df[0].notna()].copy()
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].copy()
    df["projected_seed_line"] = df[0].astype(int)
    df["team"] = df[1].astype(str).str.strip()
    df["conference"] = df[2].astype(str).str.strip()
    df["average_seed"] = pd.to_numeric(df[3], errors="coerce")
    df["bracket_count"] = pd.to_numeric(df[4], errors="coerce")
    df["team_key"] = df["team"].map(_normalize_team_name)
    out_path = os.path.join(OUTPUT_DIR, "bracketmatrix_consensus.csv")
    df[["team", "team_key", "conference", "projected_seed_line", "average_seed", "bracket_count"]].to_csv(
        out_path, index=False
    )
    return out_path


def write_source_manifest(paths: list[str]) -> str:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_date": RUN_DATE,
        "files": [os.path.relpath(path, ROOT) for path in paths],
        "sources": [
            {
                "name": "TeamRankings",
                "url": "https://www.teamrankings.com/ncaa-tournament/bracketology/",
                "status": "fetched",
            },
            {
                "name": "CBS Sports Bracketology",
                "url": "https://new.cbssports.com/college-basketball/bracketology/",
                "status": "fetched",
            },
            {
                "name": "Bracket Matrix",
                "url": "https://www.bracketmatrix.com/",
                "status": "fetched",
            },
            {
                "name": "ESPN Bracketology",
                "url": "https://www.espn.com/espn/feature/story/_/id/26156763/bracketology-march-madness-projections",
                "status": "reference_only",
                "note": "Not exported automatically here because the page is not exposed as a structured public table.",
            },
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, "sources.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def main() -> None:
    _ensure_dir(OUTPUT_DIR)
    outputs = [
        fetch_teamrankings(),
        fetch_cbs(),
        fetch_bracketmatrix(),
    ]
    outputs.append(write_source_manifest(outputs))
    for path in outputs:
        print(os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
