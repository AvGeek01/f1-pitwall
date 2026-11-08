
"""Build leakage-safe features and pit-stop targets from cleaned lap data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RACE_COLUMNS = ["season", "round", "race"]

DRIVER_RACE_COLUMNS = [
    "season",
    "round",
    "race",
    "driver",
]

REQUIRED_LAP_COLUMNS = [
    "season",
    "round",
    "race",
    "circuit",
    "driver",
    "lap_number",
    "lap_time_s",
    "fuel_corrected_time_s",
    "compound",
    "tyre_life",
    "stint",
    "position",
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
]


def _validate_inputs(
    laps: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> None:
    """Validate cleaned lap data and circuit-level pit-loss data."""

    missing_laps = set(REQUIRED_LAP_COLUMNS) - set(laps.columns)

    if missing_laps:
        raise ValueError(
            f"Missing required lap columns: {sorted(missing_laps)}"
        )

    if "circuit" not in pit_loss.columns:
        raise ValueError(
            "Pit-loss data must contain a circuit column."
        )

    valid_pit_loss_columns = {
        "pit_loss_median_s",
        "median_pit_loss_s",
    }

    if not valid_pit_loss_columns.intersection(pit_loss.columns):
        raise ValueError(
            "Pit-loss data must contain either "
            "'pit_loss_median_s' or 'median_pit_loss_s'."
        )

    if laps.empty:
        raise ValueError("The cleaned lap dataset is empty.")

    duplicate_columns = DRIVER_RACE_COLUMNS + ["lap_number"]

    if laps.duplicated(duplicate_columns).any():
        raise ValueError(
            "Duplicate driver-lap records found. "
            "Expected one row per driver per race lap."
        )


def _add_tyre_features(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Add tyre age, stint and compound-use features."""

    group = group.copy()

    group = group.sort_values("lap_number")

    group["tyre_age"] = group["tyre_life"]

    group["stint_number"] = group["stint"]

    dry_compounds = group["compound"].where(
        ~group["compound"]
        .astype("string")
        .str.upper()
        .isin(["INTERMEDIATE", "WET"])
    )

    compounds_seen: set[str] = set()
    compounds_count: list[int] = []

    for compound in dry_compounds:
        if pd.notna(compound):
            compounds_seen.add(str(compound).upper())

        compounds_count.append(len(compounds_seen))

    group["compounds_used"] = compounds_count

    group["two_compound_rule_satisfied"] = (
        group["compounds_used"] >= 2
    )

    return group


def _add_race_progress(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Add race progress and remaining-lap features."""

    group = group.copy()

    total_laps = group["lap_number"].max()

    if pd.isna(total_laps) or total_laps <= 0:
        group["laps_remaining"] = np.nan
        group["race_progress"] = np.nan
        return group

    group["laps_remaining"] = (
        total_laps - group["lap_number"]
    ).clip(lower=0)

    group["race_progress"] = (
        group["lap_number"] / total_laps
    )

    return group


def _add_pace_features(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Add trailing three-lap pace and tyre degradation slope."""

    group = group.copy()

    group = group.sort_values("lap_number")

    clean_lap = (
        (~group["is_inlap"].fillna(False).astype(bool))
        & (~group["is_outlap"].fillna(False).astype(bool))
        & (~group["is_sc"].fillna(False).astype(bool))
    )

    clean_pace = group["fuel_corrected_time_s"].where(
        clean_lap
    )

    group["rolling_3_lap_pace"] = (
        clean_pace.rolling(
            window=3,
            min_periods=1,
        ).mean()
    )

    slopes: list[float] = []

    for index in range(len(group)):
        current = group.iloc[: index + 1].copy()

        current_stint = current["stint"].iloc[-1]

        current = current[
            current["stint"] == current_stint
        ]

        current = current.loc[
            clean_pace.loc[current.index].notna()
        ].tail(5)

        if len(current) < 2:
            slopes.append(np.nan)
            continue

        x = current["tyre_life"].to_numpy(dtype=float)

        y = current[
            "fuel_corrected_time_s"
        ].to_numpy(dtype=float)

        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or len(np.unique(x)) < 2
        ):
            slopes.append(np.nan)
            continue

        slope = float(np.polyfit(x, y, 1)[0])

        slopes.append(slope)

    group["degradation_slope"] = slopes

    return group


def _add_gap_features(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate gaps to the leader, car ahead and car behind.

    Cumulative time is calculated separately for each driver.
    Drivers are compared only at the same lap number.

    These are approximate gaps, not official FIA timing intervals.
    """

    group = group.copy()

    group = group.sort_values(
        ["driver", "lap_number"]
    )

    group["cumulative_race_time"] = (
        group.groupby("driver")["lap_time_s"].cumsum()
    )

    group["gap_ahead"] = np.nan
    group["gap_behind"] = np.nan
    group["gap_leader"] = np.nan

    lap_numbers = sorted(
        group["lap_number"].dropna().unique()
    )

    for lap_number in lap_numbers:

        current = group.loc[
            group["lap_number"] == lap_number
        ].copy()

        current = current.dropna(
            subset=["cumulative_race_time"]
        )

        current = current.sort_values(
            ["cumulative_race_time", "driver"]
        )

        times = current[
            "cumulative_race_time"
        ].to_numpy(dtype=float)

        ahead_gaps = np.full(len(current), np.nan)
        behind_gaps = np.full(len(current), np.nan)

        if len(current) > 1:
            ahead_gaps[1:] = times[1:] - times[:-1]

            behind_gaps[:-1] = times[1:] - times[:-1]

        leader_gaps = times - times.min()

        group.loc[
            current.index,
            "gap_ahead",
        ] = ahead_gaps

        group.loc[
            current.index,
            "gap_behind",
        ] = behind_gaps

        group.loc[
            current.index,
            "gap_leader",
        ] = leader_gaps

    return group


def _add_undercut_threat(
    group: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> pd.DataFrame:
    """Flag whether the car behind is within estimated pit loss."""

    group = group.copy()

    circuit = group["circuit"].iloc[0]

    circuit_data = pit_loss.loc[
        pit_loss["circuit"] == circuit,
        "pit_loss_median_s",
    ]

    if circuit_data.empty or pd.isna(circuit_data.iloc[0]):
        group["undercut_threat"] = False
        return group

    threshold = float(circuit_data.iloc[0])

    group["undercut_threat"] = (
        group["gap_behind"].notna()
        & (group["gap_behind"] <= threshold)
        & (group["gap_behind"] >= 0)
    )

    return group


def _add_pit_targets(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Create future pit-stop prediction targets for one driver-race.

    Future information is intentionally used for targets.
    These target columns must not be used as model input features.
    """

    group = group.copy()

    group = group.sort_values("lap_number")

    pit_laps = (
        group.loc[
            group["is_inlap"].fillna(False).astype(bool),
            "lap_number",
        ]
        .dropna()
        .astype(int)
        .to_numpy()
    )

    current_laps = group[
        "lap_number"
    ].to_numpy(dtype=float)

    laps_to_next: list[float] = []

    pits_next_lap: list[bool] = []

    pits_within_3: list[bool] = []

    for current_lap in current_laps:

        future_stops = pit_laps[
            pit_laps > current_lap
        ]

        if len(future_stops) == 0:
            laps_to_next.append(np.nan)
            pits_next_lap.append(False)
            pits_within_3.append(False)
            continue

        distance = int(
            future_stops[0] - current_lap
        )

        laps_to_next.append(float(distance))

        pits_next_lap.append(
            distance == 1
        )

        pits_within_3.append(
            distance <= 3
        )

    group["laps_to_next_pit"] = laps_to_next

    group["pits_next_lap"] = pits_next_lap

    group["pits_within_3"] = pits_within_3

    return group


def build_features(
    laps: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> pd.DataFrame:
    """Build model-ready features from cleaned driver-lap data.

    Parameters
    ----------
    laps:
        Cleaned driver-lap dataset produced by A2.
    pit_loss:
        Circuit-level pit-loss estimates produced by A2.
        Supports both real and synthetic A2 column naming.

    Returns
    -------
    pd.DataFrame
        One row per driver per lap, containing features and targets.
    """

    _validate_inputs(laps, pit_loss)

    pit_loss = pit_loss.copy()

    if "pit_loss_median_s" not in pit_loss.columns:
        pit_loss = pit_loss.rename(
            columns={
                "median_pit_loss_s": "pit_loss_median_s"
            }
        )

    features = laps.copy()

    optional_defaults: dict[str, Any] = {
        "team": "UNKNOWN",
        "grid_position": np.nan,
        "air_temp": np.nan,
        "track_temp": np.nan,
        "rainfall": 0.0,
    }

    for column, default in optional_defaults.items():
        if column not in features.columns:
            features[column] = default

    features = features.sort_values(
        DRIVER_RACE_COLUMNS + ["lap_number"]
    ).reset_index(drop=True)

    driver_groups: list[pd.DataFrame] = []

    for _, group in features.groupby(
        DRIVER_RACE_COLUMNS,
        sort=False,
        dropna=False,
    ):

        group = _add_tyre_features(group)

        group = _add_pace_features(group)

        group = _add_pit_targets(group)

        driver_groups.append(group)

    features = pd.concat(
        driver_groups,
        ignore_index=True,
    )

    race_groups: list[pd.DataFrame] = []

    for _, group in features.groupby(
        RACE_COLUMNS,
        sort=False,
        dropna=False,
    ):

        group = _add_race_progress(group)

        group = _add_gap_features(group)

        driver_results: list[pd.DataFrame] = []

        for _, driver_group in group.groupby(
            "driver",
            sort=False,
            dropna=False,
        ):

            driver_results.append(
                _add_undercut_threat(
                    driver_group,
                    pit_loss,
                )
            )

        race_groups.append(
            pd.concat(
                driver_results,
                ignore_index=True,
            )
        )

    features = pd.concat(
        race_groups,
        ignore_index=True,
    )

    features = features.sort_values(
        DRIVER_RACE_COLUMNS + ["lap_number"]
    ).reset_index(drop=True)

    features["pits_next_lap"] = (
        features["pits_next_lap"].astype(bool)
    )

    features["pits_within_3"] = (
        features["pits_within_3"].astype(bool)
    )

    logger.info(
        "Built %s feature rows across %s races.",
        len(features),
        features.groupby(
            RACE_COLUMNS,
            dropna=False,
        ).ngroups,
    )

    return features
