#!/usr/bin/env python3
"""Fetch current-season external team data for 2025-26."""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEASON = "2025-26"
SEASON_YEAR = 2026
RUN_DATE = datetime.now(timezone.utc).astimezone().date().isoformat()
OUTPUT_DIR = os.path.join(ROOT, "data", "external", "current_season", RUN_DATE)


def _ensure_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _team_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _match_key(name: str) -> str:
    text = str(name).lower()
    replacements = {
        "&": "and",
        "saint": "st",
        "mount": "mt",
        "state": "st",
        "university": "",
        "collegeof": "charleston" if "charleston" in text else "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return re.sub(r"[^a-z0-9]", "", text)


def _competition_alias_map(competition_names: list[str]) -> dict[str, str]:
    aliases = {}
    for name in competition_names:
        aliases[_team_key(name)] = name
        aliases[_match_key(name)] = name
    aliases.update({
        "connecticut": "UConn",
        "brighamyoung": "BYU",
        "collegeofcharleston": "Col. of Charleston",
        "charleston": "Col. of Charleston",
        "appalachianst": "App State",
        "appalachianstate": "App State",
        "boisestate": "Boise St.",
        "coloradostate": "Colorado St.",
        "fresnostate": "Fresno St.",
        "illinoisstate": "Illinois St.",
        "indianastate": "Indiana St.",
        "iowastate": "Iowa St.",
        "kansasstate": "Kansas St.",
        "kentstate": "Kent St.",
        "michiganstate": "Michigan St.",
        "mississippistate": "Mississippi St.",
        "ncstate": "NC State",
        "northcarolinast": "NC State",
        "ohiostate": "Ohio St.",
        "oklahomastate": "Oklahoma St.",
        "oregonstate": "Oregon St.",
        "pennstate": "Penn St.",
        "sanjosestate": "San Jose St.",
        "sandiegostate": "San Diego St.",
        "southcarolinaupstate": "USC Upstate",
        "smu": "SMU",
        "southflorida": "South Fla.",
        "ucf": "UCF",
        "ucirvine": "UC Irvine",
        "ucsantabarbara": "UC Santa Barbara",
        "ucsandiego": "UC San Diego",
        "uab": "UAB",
        "uconn": "UConn",
        "utahstate": "Utah St.",
        "vcu": "VCU",
        "washingtonst": "Washington St.",
        "washingtonstate": "Washington St.",
        "wichitastate": "Wichita St.",
    })
    return aliases


def _resolve_source_team(name: str, competition_names: list[str], alias_map: dict[str, str]) -> str:
    raw_key = _team_key(name)
    if raw_key in alias_map:
        return alias_map[raw_key]

    alt_key = _match_key(name)
    if alt_key in alias_map:
        return alias_map[alt_key]

    scored = []
    for comp_name in competition_names:
        score = SequenceMatcher(None, alt_key, _match_key(comp_name)).ratio()
        if score >= 0.92:
            scored.append((score, comp_name))
    scored.sort(reverse=True)
    if len(scored) == 1:
        return scored[0][1]
    if len(scored) >= 2 and scored[0][0] - scored[1][0] >= 0.03:
        return scored[0][1]
    return str(name)


def fetch_ncaa_net() -> str:
    url = "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings"
    df = pd.read_html(url)[0].copy()
    df = df.rename(
        columns={
            "Rank": "ncaa_net_rank",
            "School": "team",
            "Record": "record",
            "Conf": "conference",
            "Prev": "ncaa_prev_net_rank",
            "Quad 1": "quad1_record",
            "Quad 2": "quad2_record",
            "Quad 3": "quad3_record",
            "Quad 4": "quad4_record",
        }
    )
    df["season"] = SEASON
    df["team_key"] = df["team"].map(_team_key)
    path = os.path.join(OUTPUT_DIR, "ncaa_net_rankings_2026.csv")
    df.to_csv(path, index=False)
    return path


def fetch_warren_nolan() -> str:
    url = "https://www.warrennolan.com/basketball/2026/compare-rankings"
    df = pd.read_html(url)[0].copy()
    df = df.rename(
        columns={
            "Team": "team",
            "Record": "record",
            "NET": "wn_net_rank",
            "ELO": "wn_elo_rank",
            "KPI": "wn_kpi_rank",
            "SOR": "wn_sor_rank",
            "WAB": "wn_wab_rank",
            "Average Rank": "wn_average_rank",
            "BPI": "wn_bpi_rank",
            "POM": "wn_kenpom_rank",
            "T-Rank": "wn_t_rank",
            "Avg. Pred. Rank": "wn_avg_predictive_rank",
            "RPI": "wn_rpi_rank",
            "Pred. RPI": "wn_predictive_rpi_rank",
        }
    )
    df["season"] = SEASON
    df["team_key"] = df["team"].map(_team_key)
    path = os.path.join(OUTPUT_DIR, "warren_nolan_compare_rankings_2026.csv")
    df.to_csv(path, index=False)
    return path


def fetch_sports_reference() -> str:
    url = "https://www.sports-reference.com/cbb/seasons/men/2026-school-stats.html"
    df = pd.read_html(url)[0].copy()
    flat_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat = "_".join(str(part).strip() for part in col if str(part) != "nan").strip("_")
        else:
            flat = str(col)
        flat_columns.append(flat.lower().replace("%", "pct").replace(".", "").replace(" ", "_"))
    df.columns = flat_columns
    df = df.rename(
        columns={
            "unnamed:_1_level_0_school": "team",
            "unnamed:_0_level_0_rk": "sportsref_rank",
            "overall_g": "games",
            "overall_w": "wins",
            "overall_l": "losses",
            "overall_w-lpct": "win_pct",
            "overall_srs": "srs",
            "overall_sos": "sos",
            "points_tm": "points_for",
            "points_opp": "points_against",
            "totals_fg": "fg",
            "totals_fga": "fga",
            "totals_fgpct": "fg_pct",
            "totals_3p": "three_p",
            "totals_3pa": "three_pa",
            "totals_3ppct": "three_p_pct",
            "totals_ft": "ft",
            "totals_fta": "fta",
            "totals_ftpct": "ft_pct",
            "totals_orb": "orb",
            "totals_trb": "trb",
            "totals_ast": "ast",
            "totals_stl": "stl",
            "totals_blk": "blk",
            "totals_tov": "tov",
            "totals_pf": "pf",
            "conf_w": "conf_wins",
            "conf_l": "conf_losses",
            "home_w": "home_wins",
            "home_l": "home_losses",
            "away_w": "away_wins",
            "away_l": "away_losses",
        }
    )
    df = df[df["team"].notna()].copy()
    df = df[df["team"] != "School"].copy()
    df["season"] = SEASON
    df["team_key"] = df["team"].map(_team_key)
    path = os.path.join(OUTPUT_DIR, "sports_reference_school_stats_2026.csv")
    df.to_csv(path, index=False)
    return path


def _extract_numeric_prefix(value: object) -> float:
    text = str(value).strip()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else float("nan")


def _clean_barttorvik_team(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+vs\..*$", "", text)
    text = re.sub(r"\s{2,}.*$", "", text)
    return text.strip()


def fetch_barttorvik_local() -> str | None:
    html_path = Path(ROOT) / "T-Rank.html"
    if not html_path.exists():
        return None

    tables = pd.read_html(str(html_path))
    if not tables:
        return None

    df = tables[0].copy()
    flat_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat = "_".join(str(part).strip() for part in col if str(part) != "nan").strip("_")
        else:
            flat = str(col)
        flat_columns.append(flat.lower().replace("%", "pct").replace(".", "").replace(" ", "_"))
    df.columns = flat_columns
    df = df.rename(
        columns={
            "unnamed:_0_level_0_rk": "bart_rank",
            "unnamed:_1_level_0_team": "team",
            "unnamed:_2_level_0_conf": "conference",
            "unnamed:_3_level_0_g": "games",
            "d-i_avg:_rec": "record",
            "109_adjoe": "adjoe_raw",
            "109_adjde": "adjde_raw",
            "04893_barthag": "barthag_raw",
            "unnamed:_23_level_0_wab": "wab_raw",
            "674_adj_t": "adj_t_raw",
            "eff_fgpct_514_efgpct": "efg_o_raw",
            "eff_fgpct_514_efgdpct": "efg_d_raw",
            "turnoverpct_168_tor": "tor_raw",
            "turnoverpct_168_tord": "tord_raw",
            "reboundpct_306_orb": "orb_raw",
            "reboundpct_306_drb": "drb_raw",
            "ft_rate_351_ftr": "ftr_raw",
            "ft_rate_351_ftrd": "ftrd_raw",
            "2-pt_pct_517_2ppct": "two_p_o_raw",
            "2-pt_pct_517_2pdpct": "two_p_d_raw",
            "3-pt_pct_339_3ppct": "three_p_o_raw",
            "3-pt_pct_339_3pdpct": "three_p_d_raw",
        }
    )
    df["team"] = df["team"].map(_clean_barttorvik_team)
    for raw_col, clean_col in [
        ("bart_rank", "bart_rank"),
        ("adjoe_raw", "adjoe"),
        ("adjde_raw", "adjde"),
        ("barthag_raw", "barthag"),
        ("wab_raw", "wab"),
        ("adj_t_raw", "adj_t"),
        ("efg_o_raw", "efg_o"),
        ("efg_d_raw", "efg_d"),
        ("tor_raw", "tor"),
        ("tord_raw", "tord"),
        ("orb_raw", "orb"),
        ("drb_raw", "drb"),
        ("ftr_raw", "ftr"),
        ("ftrd_raw", "ftrd"),
        ("two_p_o_raw", "2p_o"),
        ("two_p_d_raw", "2p_d"),
        ("three_p_o_raw", "3p_o"),
        ("three_p_d_raw", "3p_d"),
    ]:
        if raw_col in df.columns:
            df[clean_col] = df[raw_col].map(_extract_numeric_prefix)
    df["season"] = SEASON
    df["team_key"] = df["team"].map(_team_key)
    keep_cols = [
        "team", "team_key", "season", "bart_rank", "conference", "games", "record",
        "adjoe", "adjde", "barthag", "wab", "adj_t", "efg_o", "efg_d", "tor",
        "tord", "orb", "drb", "ftr", "ftrd", "2p_o", "2p_d", "3p_o", "3p_d",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    path = os.path.join(OUTPUT_DIR, "barttorvik_2026_from_saved_html.csv")
    df.to_csv(path, index=False)
    return path


def build_combined_snapshot(ncaa_path: str, warren_path: str, sportsref_path: str, bart_path: str | None) -> str:
    ncaa = pd.read_csv(ncaa_path)
    warren = pd.read_csv(warren_path)
    sportsref = pd.read_csv(sportsref_path)

    merged = ncaa.merge(warren, on=["team_key", "season"], how="outer", suffixes=("_ncaa", "_warren"))
    merged = merged.merge(sportsref, on=["team_key", "season"], how="outer", suffixes=("", "_sportsref"))
    if bart_path:
        bart = pd.read_csv(bart_path)
        merged = merged.merge(bart, on=["team_key", "season"], how="outer", suffixes=("", "_bart"))

    if "team_ncaa" in merged.columns:
        merged["team"] = merged["team_ncaa"]
    if "team" in merged.columns and "team_warren" in merged.columns:
        merged["team"] = merged["team"].fillna(merged["team_warren"])
    if "team" in merged.columns and "team_sportsref" in merged.columns:
        merged["team"] = merged["team"].fillna(merged["team_sportsref"])

    path = os.path.join(OUTPUT_DIR, "current_season_external_snapshot_2026.csv")
    merged.to_csv(path, index=False)
    return path


def build_competition_aligned_snapshot(
    ncaa_path: str, warren_path: str, sportsref_path: str, bart_path: str | None
) -> str:
    competition = pd.read_csv(os.path.join(ROOT, "data", "raw", "NCAA_Seed_Test_Set2.0.csv"))
    competition = competition.rename(columns={"Team": "competition_team", "Season": "season"})
    competition["team_key"] = competition["competition_team"].map(_team_key)
    competition_names = competition["competition_team"].dropna().unique().tolist()
    alias_map = _competition_alias_map(competition_names)

    ncaa = pd.read_csv(ncaa_path)
    warren = pd.read_csv(warren_path)
    sportsref = pd.read_csv(sportsref_path)
    bart = pd.read_csv(bart_path) if bart_path else None

    source_dfs = [ncaa, warren, sportsref]
    if bart is not None:
        source_dfs.append(bart)

    for df in source_dfs:
        df["competition_team"] = df["team"].map(
            lambda name: _resolve_source_team(name, competition_names, alias_map)
        )
        df["team_key"] = df["competition_team"].map(_team_key)
        df.drop_duplicates(subset=["team_key", "season"], inplace=True)

    merged = competition.merge(ncaa, on=["team_key", "season"], how="left", suffixes=("_comp", "_ncaa"))
    merged = merged.merge(warren, on=["team_key", "season"], how="left", suffixes=("", "_warren"))
    merged = merged.merge(sportsref, on=["team_key", "season"], how="left", suffixes=("", "_sportsref"))
    if bart is not None:
        merged = merged.merge(bart, on=["team_key", "season"], how="left", suffixes=("", "_bart"))

    path = os.path.join(OUTPUT_DIR, "current_season_external_competition_aligned_2026.csv")
    merged.to_csv(path, index=False)
    return path


def main() -> None:
    _ensure_dir()
    ncaa_path = fetch_ncaa_net()
    warren_path = fetch_warren_nolan()
    sportsref_path = fetch_sports_reference()
    bart_path = fetch_barttorvik_local()
    combined_path = build_combined_snapshot(ncaa_path, warren_path, sportsref_path, bart_path)
    aligned_path = build_competition_aligned_snapshot(ncaa_path, warren_path, sportsref_path, bart_path)

    paths = [ncaa_path, warren_path, sportsref_path]
    if bart_path:
        paths.append(bart_path)
    paths.extend([combined_path, aligned_path])

    for path in paths:
        print(os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
