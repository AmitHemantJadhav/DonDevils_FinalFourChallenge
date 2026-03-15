"""
seed_assignment.py - Constrained overall seed assignment using traditional seeds.

Uses barttorvik traditional seeds (1-16) to map test tournament teams to
overall seed slots (1-68), with composite disambiguation scoring.
"""

import numpy as np
import pandas as pd
from collections import Counter


def _build_seed_line_ranges(traditional_seeds: dict, season: str) -> dict:
    """Compute the expected overall seed range for each traditional seed line.

    The overall seed 1-68 is the S-curve ranking. Each traditional seed line
    occupies a consecutive block, with size determined by play-in structure.

    Args:
        traditional_seeds: {(season, team): trad_seed} for all teams.
        season: Season string like '2020-21'.

    Returns:
        Dict of {trad_seed: [list of overall seed positions]}.
    """
    # Count teams per traditional seed for this season
    season_seeds = {t: s for (sz, t), s in traditional_seeds.items() if sz == season}
    cnt = Counter(season_seeds.values())

    ranges = {}
    pos = 1
    for trad in range(1, 17):
        n = cnt.get(trad, 4)
        ranges[trad] = list(range(pos, pos + n))
        pos += n

    return ranges


def _compute_disambiguation_scores(team_list, raw_predictions, test_df,
                                   barttorvik_stats, season):
    """Compute composite scores for ranking teams within a seed line.

    Lower score = better team = lower overall seed slot.

    Args:
        team_list: List of (test_index, raw_pred) tuples.
        raw_predictions: Full raw predictions array.
        test_df: Test DataFrame with bid_type column.
        barttorvik_stats: {(season, team): {'ADJOE': x, 'ADJDE': y}} or None.
        season: Current season string.

    Returns:
        List of (test_index, composite_score) tuples, sorted by score ascending.
    """
    if len(team_list) <= 1 or barttorvik_stats is None:
        # Deterministic or no barttorvik data — use raw prediction only
        return sorted(team_list, key=lambda x: x[1])

    # Gather raw predictions and NET_MARGIN for the group
    indices = [idx for idx, _ in team_list]
    preds = np.array([pred for _, pred in team_list])

    net_margins = []
    is_at_large = []
    has_all_stats = True
    for idx, _ in team_list:
        team = test_df.loc[idx, 'team']
        stats = barttorvik_stats.get((season, team))
        if stats and 'ADJOE' in stats and 'ADJDE' in stats:
            net_margins.append(stats['ADJOE'] - stats['ADJDE'])
        else:
            has_all_stats = False
            net_margins.append(0.0)

        bid = test_df.loc[idx, 'bid_type'] if 'bid_type' in test_df.columns else None
        is_at_large.append(1 if bid == 'at_large' else 0)

    net_margins = np.array(net_margins)
    is_at_large = np.array(is_at_large)

    # Z-score normalize within the group
    pred_std = np.std(preds)
    netm_std = np.std(net_margins)

    if pred_std < 1e-9 and netm_std < 1e-9:
        # All identical — fall back to raw prediction
        return sorted(team_list, key=lambda x: x[1])

    if pred_std < 1e-9:
        z_pred = np.zeros_like(preds)
    else:
        z_pred = (preds - np.mean(preds)) / pred_std

    if netm_std < 1e-9 or not has_all_stats:
        z_netm = np.zeros_like(net_margins)
        # Without NET_MARGIN, use only ensemble + AL penalty
        w_pred, w_netm = 1.0, 0.0
    else:
        z_netm = -(net_margins - np.mean(net_margins)) / netm_std  # negate: higher NETM = better
        w_pred, w_netm = 0.69, 0.31

    al_penalty = 0.27 * (1 - is_at_large)  # AQ teams get pushed down (higher score)

    composite = w_pred * z_pred + w_netm * z_netm + al_penalty

    # Build scored list and sort
    scored = [(indices[i], composite[i]) for i in range(len(indices))]
    return sorted(scored, key=lambda x: x[1])


def constrained_seed_assignment(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    traditional_seeds: dict,
    barttorvik_stats: dict = None,
) -> np.ndarray:
    """Assign overall seeds to test teams using traditional seed constraints.

    For tournament test teams: uses known traditional seeds to narrow possible
    overall seed slots, then uses composite disambiguation scoring to rank
    within each group. For non-tournament test teams: assigns 0.

    Args:
        test_df: Full test DataFrame with team, season, bid_type columns.
        train_df: Training DataFrame with overall_seed column.
        raw_predictions: Model's raw predictions for ALL test rows (451).
        traditional_seeds: {(season, team): trad_seed} from barttorvik.
        barttorvik_stats: {(season, team): {'ADJOE': x, 'ADJDE': y}} or None.

    Returns:
        Array of length len(test_df) with final overall seed predictions.
    """
    predictions = np.zeros(len(test_df))

    # Identify tournament vs non-tournament test teams
    is_tournament = test_df['bid_type'].notna()
    non_tourn_count = (~is_tournament).sum()
    tourn_count = is_tournament.sum()
    print(f"  Constrained assignment: {tourn_count} tournament, {non_tourn_count} non-tournament")

    if barttorvik_stats is not None:
        print(f"  Disambiguation: composite scoring (0.69*ens + 0.31*netm + AL penalty)")
    else:
        print(f"  Disambiguation: raw prediction only (no barttorvik stats)")

    # Process each season independently
    for season in sorted(test_df['season'].unique()):
        season_mask = test_df['season'] == season

        # Get occupied overall seed slots from training data
        season_train = train_df[
            (train_df['season'] == season) & (train_df['overall_seed'].notna())
        ]
        occupied = set(int(x) for x in season_train['overall_seed'].tolist())

        # Build expected ranges for this season
        ranges = _build_seed_line_ranges(traditional_seeds, season)

        # Get test tournament teams for this season
        season_tourn_mask = season_mask & is_tournament
        test_indices = test_df.index[season_tourn_mask].tolist()

        if not test_indices:
            continue

        # Group test teams by traditional seed
        groups = {}  # trad_seed -> [(test_index, raw_pred)]
        unmatched = []  # teams without traditional seed match
        for idx in test_indices:
            team = test_df.loc[idx, 'team']
            trad = traditional_seeds.get((season, team))
            if trad is None:
                unmatched.append(idx)
                continue
            groups.setdefault(trad, []).append((idx, raw_predictions[idx]))

        # Available slots per seed line
        available = {}
        for trad in range(1, 17):
            available[trad] = [s for s in ranges[trad] if s not in occupied]

        # Detect mismatches (surplus/deficit in slot counts vs team counts)
        deficit_teams = []  # test teams that don't have enough slots in their range

        balanced_groups = {}
        for trad, team_list in groups.items():
            n_teams = len(team_list)
            n_avail = len(available[trad])

            if n_teams == n_avail:
                balanced_groups[trad] = (team_list, available[trad])
            elif n_avail > n_teams:
                # More slots than teams: assign teams, extras go to surplus
                balanced_groups[trad] = (team_list, available[trad])
            else:
                # Fewer slots than teams: some teams need slots from neighbors
                deficit_teams.extend(team_list[n_avail:])
                if n_avail > 0:
                    balanced_groups[trad] = (team_list[:n_avail], available[trad][:n_avail])

        # Assign balanced groups: rank teams by composite score, assign to sorted slots
        assigned_slots = set()
        for trad in sorted(balanced_groups.keys()):
            team_list, slot_list = balanced_groups[trad]
            n_teams = len(team_list)
            n_slots = len(slot_list)

            # Use composite disambiguation scoring
            team_list_sorted = _compute_disambiguation_scores(
                team_list, raw_predictions, test_df, barttorvik_stats, season
            )
            slot_list_sorted = sorted(slot_list)

            # Assign: best scored team gets lowest slot
            for i, (idx, _) in enumerate(team_list_sorted):
                if i < n_slots:
                    predictions[idx] = slot_list_sorted[i]
                    assigned_slots.add(slot_list_sorted[i])

        # Handle surplus slots: collect all unassigned slots
        all_unoccupied = set(range(1, 69)) - occupied - assigned_slots
        remaining_slots = sorted(all_unoccupied)

        # Handle deficit teams + unmatched teams
        remaining_teams = deficit_teams + [(idx, raw_predictions[idx]) for idx in unmatched]
        if remaining_teams:
            # Use composite scoring for remaining teams too
            remaining_sorted = _compute_disambiguation_scores(
                remaining_teams, raw_predictions, test_df, barttorvik_stats, season
            )
            for i, (idx, _) in enumerate(remaining_sorted):
                if i < len(remaining_slots):
                    predictions[idx] = remaining_slots[i]
                else:
                    # Fallback: use raw prediction rounded
                    predictions[idx] = max(1, min(68, round(raw_predictions[idx])))

        # Summary for this season
        season_test_teams = len(test_indices)
        deterministic = sum(
            1 for trad, (tl, sl) in balanced_groups.items()
            if len(tl) == 1 and len(sl) == 1
        )
        print(f"    {season}: {season_test_teams} test teams, "
              f"{deterministic} deterministic, "
              f"{len(deficit_teams)} deficit, {len(unmatched)} unmatched")

    return predictions.astype(int)
