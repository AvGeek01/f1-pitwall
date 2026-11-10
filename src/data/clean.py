
"""Clean OpenF1 race data and estimate circuit pit loss."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.schema import (
    MAX_PLAUSIBLE_LAP_DURATION_S,
    MIN_PLAUSIBLE_LAP_DURATION_S,
    MIN_PLAUSIBLE_TIMESTAMP_FALLBACK_S,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PARQUET_DIRECTORY = (
    PROJECT_ROOT / "data" / "raw" / "openf1_parquet"
)
DEFAULT_FUEL_EFFECT = 0.03
DEFAULT_RAW_ENDPOINTS = (
    "sessions",
    "meetings",
    "laps",
    "stints",
    "pit",
    "weather",
    "race_control",
    "position",
    "starting_grid",
    "drivers",
    "intervals",
    "session_result",
)
CLI_RAW_ENDPOINTS = (
    "laps",
    "stints",
    "pit",
    "weather",
    "race_control",
    "position",
    "starting_grid",
    "drivers",
    "sessions",
    "meetings",
    "session_result",
)

LAPS_CLEAN_COLUMNS = [
    "season",
    "round",
    "race",
    "circuit",
    "driver",
    "team",
    "lap_number",
    "lap_time_s",
    "fuel_corrected_time_s",
    "compound",
    "tyre_life",
    "stint",
    "position",
    "grid_position",
    "grid_source",
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
    "air_temp",
    "track_temp",
    "rainfall",
]


def load_raw_data(
    years: Sequence[int],
    data_directory: str | Path = DEFAULT_RAW_PARQUET_DIRECTORY,
    endpoints: Sequence[str] = DEFAULT_RAW_ENDPOINTS,
    required_endpoints: Sequence[str] = ("laps", "stints", "drivers"),
) -> dict[str, pd.DataFrame]:
    """Load full or partial endpoint parquet files, preferring full files.

    For each endpoint and year, ``{endpoint}_{year}.parquet`` is selected
    before ``{endpoint}_{year}_partial.parquet`` and
    ``{endpoint}_partial_{year}.parquet``. Missing optional endpoint files are
    logged and represented by empty DataFrames. Required endpoints must have a
    file for every requested year.
    """
    selected_years = tuple(dict.fromkeys(int(year) for year in years))
    if not selected_years:
        raise ValueError("years must contain at least one season")

    directory = Path(data_directory)
    loaded: dict[str, list[pd.DataFrame]] = {
        endpoint: [] for endpoint in endpoints
    }
    missing_required: list[str] = []
    for year in selected_years:
        for endpoint in endpoints:
            full_path = directory / f"{endpoint}_{year}.parquet"
            partial_paths = (
                directory / f"{endpoint}_{year}_partial.parquet",
                directory / f"{endpoint}_partial_{year}.parquet",
            )
            selected_path = next(
                (
                    candidate
                    for candidate in (full_path, *partial_paths)
                    if candidate.is_file()
                ),
                None,
            )
            if selected_path is None:
                logger.warning(
                    "No full or partial parquet found for endpoint %s, year %s",
                    endpoint,
                    year,
                )
                if endpoint in required_endpoints:
                    missing_required.append(f"{endpoint}_{year}")
                continue

            logger.info("Loading OpenF1 parquet file: %s", selected_path)
            frame = pd.read_parquet(selected_path)
            if frame.empty and endpoint in required_endpoints:
                missing_required.append(f"{endpoint}_{year} (empty)")
            loaded[endpoint].append(frame)

    if missing_required:
        raise FileNotFoundError(
            "Missing required full or partial OpenF1 parquet files: "
            + ", ".join(missing_required)
        )

    return {
        endpoint: (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
        for endpoint, frames in loaded.items()
    }


def main() -> int:
    """Load raw endpoint parquets, clean laps, estimate pit loss, and save."""
    parser = argparse.ArgumentParser(
        description="Clean cached OpenF1 parquet data and estimate pit loss."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/openf1_parquet"),
        help="directory containing per-endpoint OpenF1 parquet files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="directory for laps_clean.parquet and pit_loss.parquet",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="seasons to load (default: 2023 2024 2025)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info(
        "Starting OpenF1 cleaning for seasons %s from %s",
        args.years,
        args.raw_dir,
    )
    if not args.raw_dir.is_dir() or not any(
        args.raw_dir.glob("*.parquet")
    ):
        logger.error(
            "No raw parquet files found in %s; expected endpoint/year parquet "
            "files for seasons %s",
            args.raw_dir,
            args.years,
        )
        return 1

    try:
        raw = load_raw_data(
            years=args.years,
            data_directory=args.raw_dir,
            endpoints=CLI_RAW_ENDPOINTS,
            required_endpoints=("laps",),
        )
        logger.info("Cleaning concatenated laps from seasons %s", args.years)
        laps_clean = clean_laps(raw)
        logger.info("Estimating circuit pit loss")
        pit_loss = estimate_pit_loss(laps_clean)
    except (FileNotFoundError, ValueError, OSError) as error:
        logger.error("OpenF1 cleaning failed: %s", error)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    laps_path = args.out_dir / "laps_clean.parquet"
    pit_loss_path = args.out_dir / "pit_loss.parquet"
    laps_clean.to_parquet(laps_path, index=False)
    pit_loss.to_parquet(pit_loss_path, index=False)

    if laps_clean.empty:
        logger.error("Cleaning produced no lap rows; output files were written empty")
        return 1

    race_counts = (
        laps_clean[["season", "round", "race"]]
        .drop_duplicates()
        .groupby("season", dropna=False)
        .size()
        .to_dict()
    )
    season_counts = laps_clean.groupby("season", dropna=False).size().to_dict()
    logger.info(
        "Wrote %s (%d rows) and %s (%d rows)",
        laps_path,
        len(laps_clean),
        pit_loss_path,
        len(pit_loss),
    )
    logger.info("Lap rows per season: %s", season_counts)
    logger.info("Races per season: %s", race_counts)
    return 0


def _get_frame(raw: dict[str, pd.DataFrame], endpoint: str) -> pd.DataFrame:
    """Return an endpoint DataFrame, or an empty DataFrame if unavailable."""
    frame = raw.get(endpoint)

    if frame is None:
        return pd.DataFrame()

    return frame.copy()


def _safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
) -> pd.DataFrame:
    """Merge two DataFrames safely when the right-hand data is available."""
    if right.empty:
        return left

    if not all(column in right.columns for column in on):
        logger.warning("Skipping merge: missing merge columns %s", on)
        return left

    return left.merge(right, on=on, how="left")


def _numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values, replacing invalid values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def _grid_position_lookup(
    frame: pd.DataFrame,
    key_columns: tuple[str, str],
    sort_column: str | None = None,
) -> dict[tuple[str, str], float]:
    """Build a driver lookup for valid grid-position fallback records."""
    required = {*key_columns, "position"}
    if frame.empty or not required.issubset(frame.columns):
        return {}

    columns = [*key_columns, "position"]
    if sort_column is not None and sort_column in frame.columns:
        columns.append(sort_column)
    positions = frame.loc[:, columns].copy()
    positions["position"] = _numeric(positions["position"])
    positions = positions.dropna(subset=[*key_columns, "position"])
    positions = positions.loc[
        np.isfinite(positions["position"]) & positions["position"].gt(0)
    ]
    if sort_column is not None and sort_column in positions.columns:
        if sort_column == "date":
            positions[sort_column] = pd.to_datetime(
                positions[sort_column],
                errors="coerce",
                utc=True,
            )
            positions = positions.dropna(subset=[sort_column])
        positions = positions.sort_values(sort_column, kind="stable")

    positions["_key_0"] = positions[key_columns[0]].astype("string")
    positions["_key_1"] = positions[key_columns[1]].astype("string")
    positions = positions.drop_duplicates(["_key_0", "_key_1"], keep="first")
    return {
        (str(key_0), str(key_1)): float(position)
        for key_0, key_1, position in positions[
            ["_key_0", "_key_1", "position"]
        ].itertuples(index=False, name=None)
    }


def _apply_grid_position_fallbacks(
    laps: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply grid-position sources in priority order, retaining the source."""
    result = laps.copy()
    result["grid_position"] = np.nan
    result["grid_source"] = "unavailable"

    starting_grid = _get_frame(raw, "starting_grid")
    qualifying_results = _get_frame(raw, "session_result")
    race_positions = _get_frame(raw, "position")
    lookups = (
        (
            "starting_grid",
            ("session_key", "driver_number"),
            _grid_position_lookup(
                starting_grid,
                ("session_key", "driver_number"),
            ),
        ),
        (
            "qualifying_session_result",
            ("meeting_key", "driver_number"),
            _grid_position_lookup(
                qualifying_results,
                ("meeting_key", "driver_number"),
            ),
        ),
        (
            "race_position",
            ("session_key", "driver_number"),
            _grid_position_lookup(
                race_positions,
                ("session_key", "driver_number"),
                sort_column="date",
            ),
        ),
    )

    for source, keys, lookup in lookups:
        if not lookup or not set(keys).issubset(result.columns):
            continue
        normalized_keys = [
            result[key].astype("string").fillna("") for key in keys
        ]
        row_keys = zip(*normalized_keys)
        values = pd.Series(
            [
                lookup.get((str(key_0), str(key_1)), np.nan)
                for key_0, key_1 in row_keys
            ],
            index=result.index,
            dtype=float,
        )
        unresolved = result["grid_position"].isna() & values.notna()
        result.loc[unresolved, "grid_position"] = values.loc[unresolved]
        result.loc[unresolved, "grid_source"] = source

    driver_keys = ["session_key", "driver_number"]
    if set(driver_keys).issubset(result.columns):
        driver_sources = result.drop_duplicates(driver_keys).groupby(
            "grid_source",
            dropna=False,
        ).size()
        source_counts = driver_sources.to_dict()
        for source in (
            "starting_grid",
            "qualifying_session_result",
            "race_position",
            "unavailable",
        ):
            source_counts.setdefault(source, 0)
        logger.info(
            "Grid position sources by driver: %s",
            source_counts,
        )
    return result



def clean_laps(
    raw: dict[str, pd.DataFrame],
    fuel_effect: float = DEFAULT_FUEL_EFFECT,
) -> pd.DataFrame:
    """
    Clean OpenF1 endpoint data into one row per driver per lap.

    Args:
        raw: Dictionary containing DataFrames for OpenF1 endpoints.
        fuel_effect: Seconds removed per lap remaining.

    Returns:
        Cleaned lap DataFrame following LAPS_CLEAN_COLUMNS.
    """
    if fuel_effect < 0:
        raise ValueError("fuel_effect cannot be negative")

    laps = _get_frame(raw, "laps")

    if laps.empty:
        return pd.DataFrame(columns=LAPS_CLEAN_COLUMNS)

    required = {"session_key", "driver_number", "lap_number"}
    missing = required.difference(laps.columns)

    if missing:
        raise ValueError(
            f"Lap data is missing required columns: {sorted(missing)}"
        )

    # Convert numeric columns.
    for column in [
        "season",
        "round",
        "lap_number",
        "lap_duration",
        "driver_number",
    ]:
        if column in laps.columns:
            laps[column] = _numeric(laps[column])

    # Convert timestamps.
    if "date_start" in laps.columns:
        laps["date_start"] = pd.to_datetime(
            laps["date_start"], errors="coerce", utc=True
        )

    # Remove duplicate driver-lap records.
    laps = (
        laps.sort_values(
            ["session_key", "driver_number", "lap_number"]
        )
        .drop_duplicates(
            subset=["session_key", "driver_number", "lap_number"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Keep only plausible raw durations for pace-model inputs. Durations above
    # 300 seconds can include stoppages and are not representative lap pace.
    raw_duration = laps.get(
        "lap_duration",
        pd.Series(np.nan, index=laps.index),
    )
    raw_lap_time = _numeric(raw_duration)
    valid_raw_duration = (
        np.isfinite(raw_lap_time)
        & raw_lap_time.ge(MIN_PLAUSIBLE_LAP_DURATION_S)
        & raw_lap_time.le(MAX_PLAUSIBLE_LAP_DURATION_S)
    )
    laps["lap_time_s"] = raw_lap_time.where(valid_raw_duration)

    # Use timestamp differences only when OpenF1 supplied no raw duration.
    # A fallback is valid only for a finite 30-300 second interval.
    if "date_start" in laps.columns:
        previous_start = laps.groupby(
            ["session_key", "driver_number"]
        )["date_start"].shift(1)

        timestamp_difference = (
            laps["date_start"] - previous_start
        ).dt.total_seconds()
        valid_fallback = (
            np.isfinite(timestamp_difference)
            & timestamp_difference.ge(MIN_PLAUSIBLE_TIMESTAMP_FALLBACK_S)
            & timestamp_difference.le(MAX_PLAUSIBLE_LAP_DURATION_S)
        )
        missing_raw_duration = raw_lap_time.isna()
        use_fallback = missing_raw_duration & valid_fallback
        laps.loc[use_fallback, "lap_time_s"] = timestamp_difference[use_fallback]

    laps["lap_time_s"] = _numeric(laps["lap_time_s"])

    logger.info("Prepared %d driver-lap records", len(laps))

    # Attach driver names and team names.
    drivers = _get_frame(raw, "drivers")

    if not drivers.empty:
        driver_columns = [
            column
            for column in [
                "session_key",
                "driver_number",
                "name_acronym",
                "team_name",
            ]
            if column in drivers.columns
        ]

        drivers = drivers[driver_columns].drop_duplicates(
            subset=["session_key", "driver_number"]
        )

        laps = _safe_merge(
            laps,
            drivers,
            on=["session_key", "driver_number"],
        )

    if "name_acronym" in laps.columns:
        laps["driver"] = laps["name_acronym"]
    else:
        laps["driver"] = (
            laps["driver_number"].astype("Int64").astype(str)
        )

    if "team_name" in laps.columns:
        laps["team"] = laps["team_name"]
    else:
        laps["team"] = np.nan

    # Attach race and circuit information.
    meetings = _get_frame(raw, "meetings")

    if not meetings.empty and "meeting_key" in laps.columns:
        meeting_columns = [
            column
            for column in [
                "meeting_key",
                "meeting_name",
                "circuit_short_name",
                "year",
                "season",
                "round",
            ]
            if column in meetings.columns
        ]

        meetings = meetings[meeting_columns].drop_duplicates(
            subset=["meeting_key"]
        )

        laps = _safe_merge(laps, meetings, on=["meeting_key"])

    if "meeting_name" in laps.columns:
        laps["race"] = laps["meeting_name"]
    else:
        laps["race"] = np.nan

    if "circuit_short_name" in laps.columns:
        laps["circuit"] = laps["circuit_short_name"]
    else:
        laps["circuit"] = np.nan

    if "season_x" in laps.columns:
        laps["season"] = laps["season_x"].combine_first(
            laps.get("season_y", pd.Series(np.nan, index=laps.index))
        )
    elif "season" not in laps.columns:
        laps["season"] = laps.get("year", np.nan)

    if "round_x" in laps.columns:
        laps["round"] = laps["round_x"].combine_first(
            laps.get("round_y", pd.Series(np.nan, index=laps.index))
        )
    elif "round" not in laps.columns:
        laps["round"] = np.nan

    # Attach tyre stint information.
    stints = _get_frame(raw, "stints")

    laps["compound"] = np.nan
    laps["stint"] = np.nan
    laps["tyre_life"] = np.nan

    if not stints.empty:
        required_stint_columns = {
            "session_key",
            "driver_number",
            "lap_start",
            "lap_end",
        }

        if required_stint_columns.issubset(stints.columns):
            stints["lap_start"] = _numeric(stints["lap_start"])
            stints["lap_end"] = _numeric(stints["lap_end"])

            stint_columns = [
                column
                for column in [
                    "session_key",
                    "driver_number",
                    "lap_start",
                    "lap_end",
                    "compound",
                    "stint_number",
                    "tyre_age_at_start",
                ]
                if column in stints.columns
            ]

            stint_matches = laps[
                ["session_key", "driver_number", "lap_number"]
            ].merge(
                stints[stint_columns],
                on=["session_key", "driver_number"],
                how="left",
            )

            valid_matches = stint_matches[
                stint_matches["lap_number"].between(
                    stint_matches["lap_start"],
                    stint_matches["lap_end"],
                    inclusive="both",
                )
            ].copy()

            valid_matches = valid_matches.drop_duplicates(
                subset=[
                    "session_key",
                    "driver_number",
                    "lap_number",
                ],
                keep="last",
            )

            valid_matches["tyre_life"] = (
                _numeric(valid_matches["tyre_age_at_start"])
                + valid_matches["lap_number"]
                - valid_matches["lap_start"]
                + 1
            )

            stint_values = valid_matches[
                [
                    "session_key",
                    "driver_number",
                    "lap_number",
                    "compound",
                    "stint_number",
                    "tyre_life",
                ]
            ]

            laps = laps.drop(
                columns=["compound", "stint", "tyre_life"]
            ).merge(
                stint_values,
                on=[
                    "session_key",
                    "driver_number",
                    "lap_number",
                ],
                how="left",
            )

            laps = laps.rename(
                columns={"stint_number": "stint"}
            )

        else:
            logger.warning(
                "Stint data is missing required columns"
            )

    # Identify in-laps and out-laps.
    laps["is_inlap"] = False
    laps["is_outlap"] = laps["lap_number"].eq(1)

    pit = _get_frame(raw, "pit")

    if not pit.empty:
        required_pit_columns = {
            "session_key",
            "driver_number",
            "lap_number",
        }

        if required_pit_columns.issubset(pit.columns):
            pit_laps = pit[
                ["session_key", "driver_number", "lap_number"]
            ].copy()

            pit_laps["lap_number"] = _numeric(
                pit_laps["lap_number"]
            )

            pit_laps = pit_laps.dropna(
                subset=[
                    "session_key",
                    "driver_number",
                    "lap_number",
                ]
            )

            pit_laps["is_inlap"] = True

            laps = laps.merge(
                pit_laps.drop_duplicates(
                    subset=[
                        "session_key",
                        "driver_number",
                        "lap_number",
                    ]
                ),
                on=[
                    "session_key",
                    "driver_number",
                    "lap_number",
                ],
                how="left",
                suffixes=("", "_pit"),
            )

            laps["is_inlap"] = (
                laps["is_inlap"]
                | laps["is_inlap_pit"].eq(True)
            )

            laps = laps.drop(columns=["is_inlap_pit"])

    if "is_pit_out_lap" in laps.columns:
        laps["is_outlap"] = (
            laps["is_outlap"]
            | laps["is_pit_out_lap"].fillna(False).astype(bool)
        )

    laps = laps.sort_values(
        ["session_key", "driver_number", "lap_number"]
    )

    previous_was_pit = (
        laps.groupby(
            ["session_key", "driver_number"]
        )["is_inlap"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    laps["is_outlap"] = (
        laps["is_outlap"] | previous_was_pit
    )

    # Identify Safety Car, VSC and red-flag periods.
    laps["is_sc"] = False

    race_control = _get_frame(raw, "race_control")

    if not race_control.empty:
        if {"session_key", "lap_number"}.issubset(
            race_control.columns
        ):
            race_control["lap_number"] = _numeric(
                race_control["lap_number"]
            )

            race_control["event_text"] = (
                race_control.get(
                    "category",
                    pd.Series("", index=race_control.index),
                )
                .fillna("")
                .astype(str)
                + " "
                + race_control.get(
                    "message",
                    pd.Series("", index=race_control.index),
                )
                .fillna("")
                .astype(str)
            ).str.upper()

            for session_key, events in race_control.groupby(
                "session_key"
            ):
                events = events.dropna(
                    subset=["lap_number"]
                ).sort_values("lap_number")

                active_start = None

                for _, event in events.iterrows():
                    message = event["event_text"]
                    lap = int(event["lap_number"])

                    is_deployment = (
                        "DEPLOYED" in message
                        or (
                            "RED FLAG" in message
                            and "GREEN" not in message
                        )
                    )

                    is_ending = any(
                        phrase in message
                        for phrase in [
                            "ENDING",
                            "IN THIS LAP",
                            "GREEN FLAG",
                            "GREEN LIGHT",
                            "RESUMED",
                            "WITHDRAWN",
                        ]
                    )

                    if is_deployment and active_start is None:
                        active_start = lap

                    elif is_ending and active_start is not None:
                        mask = (
                            laps["session_key"].eq(session_key)
                            & laps["lap_number"].between(
                                active_start, lap
                            )
                        )

                        laps.loc[mask, "is_sc"] = True
                        active_start = None

                if active_start is not None:
                    last_lap = laps.loc[
                        laps["session_key"].eq(session_key),
                        "lap_number",
                    ].max()

                    mask = (
                        laps["session_key"].eq(session_key)
                        & laps["lap_number"].between(
                            active_start, last_lap
                        )
                    )

                    laps.loc[mask, "is_sc"] = True

    # Merge weather observations.
    laps["air_temp"] = np.nan
    laps["track_temp"] = np.nan
    laps["rainfall"] = np.nan

    weather = _get_frame(raw, "weather")

    if not weather.empty and "date" in weather.columns:
        weather["date"] = pd.to_datetime(
            weather["date"], errors="coerce", utc=True
        )

        weather_columns = [
            column
            for column in [
                "session_key",
                "date",
                "air_temperature",
                "track_temperature",
                "rainfall",
            ]
            if column in weather.columns
        ]

        weather = weather[weather_columns].dropna(
            subset=["date"]
        )

        # Rename weather columns to avoid collisions.
        weather = weather.rename(
            columns={
                "air_temperature": "weather_air_temperature",
                "track_temperature": "weather_track_temperature",
                "rainfall": "weather_rainfall",
            }
        )

        if (
            "date_start" in laps.columns
            and not weather.empty
        ):
            left = laps.copy()
            left["_clean_row_id"] = np.arange(len(left))

            # Separate rows with valid and missing timestamps.
            valid_left = left.dropna(
                subset=["date_start"]
            ).sort_values("date_start")

            invalid_left = left[
                left["date_start"].isna()
            ].copy()

            right = weather.sort_values("date")

            if not valid_left.empty:
                merged_valid = pd.merge_asof(
                    valid_left,
                    right,
                    left_on="date_start",
                    right_on="date",
                    by="session_key",
                    direction="backward",
                    tolerance=pd.Timedelta("5min"),
                )
            else:
                merged_valid = valid_left

            # Restore rows that had missing timestamps.
            merged = pd.concat(
                [merged_valid, invalid_left],
                ignore_index=True,
                sort=False,
            )

            merged = merged.sort_values("_clean_row_id")
            merged = merged.drop(columns="_clean_row_id")

            # Copy the weather values into the required columns.
            for source, target in [
                ("weather_air_temperature", "air_temp"),
                ("weather_track_temperature", "track_temp"),
                ("weather_rainfall", "rainfall"),
            ]:
                if source in merged.columns:
                    merged[target] = _numeric(merged[source])

            laps = merged

    # Identify wet laps.
    compound_text = (
        laps["compound"].fillna("").astype(str).str.upper()
    )

    wet_compound = compound_text.isin(
        ["INTERMEDIATE", "WET"]
    )

    wet_weather = (
        _numeric(laps["rainfall"]).fillna(0).gt(0)
    )

    laps["is_wet"] = wet_compound | wet_weather

    # Merge driver positions at the end of each lap.
    laps["position"] = np.nan

    position_data = _get_frame(raw, "position")

    if not position_data.empty:
        required_position = {
            "session_key",
            "driver_number",
            "position",
            "date",
        }

        if required_position.issubset(
            position_data.columns
        ):
            position_data["date"] = pd.to_datetime(
                position_data["date"],
                errors="coerce",
                utc=True,
            )

            position_data["position"] = _numeric(
                position_data["position"]
            )

            if "date_start" in laps.columns:
                laps["lap_end"] = (
                    laps["date_start"]
                    + pd.to_timedelta(
                        laps["lap_time_s"], unit="s"
                    )
                )

                position_data = position_data.dropna(
                    subset=["date", "position"]
                )

                if not position_data.empty:
                    left = laps.copy()
                    left["_row_id"] = np.arange(len(left))

                    valid_left = left.dropna(
                        subset=["lap_end"]
                    ).sort_values("lap_end")

                    right = position_data[
                        [
                            "session_key",
                            "driver_number",
                            "date",
                            "position",
                        ]
                    ].sort_values("date")

                    if not valid_left.empty:
                        merged = pd.merge_asof(
                            valid_left,
                            right,
                            left_on="lap_end",
                            right_on="date",
                            by=[
                                "session_key",
                                "driver_number",
                            ],
                            direction="backward",
                            suffixes=("", "_position"),
                        )

                        laps.loc[
                            merged["_row_id"], "position"
                        ] = merged[
                            "position_position"
                        ].to_numpy()

                laps = laps.drop(
                    columns=["lap_end"],
                    errors="ignore",
                )

    # Apply starting grid, qualifying result, then earliest race position.
    laps = _apply_grid_position_fallbacks(laps, raw)

    # Calculate fuel-corrected lap times.
    maximum_lap = laps.groupby(
        "session_key"
    )["lap_number"].transform("max")

    laps_remaining = (
        maximum_lap - laps["lap_number"]
    ).clip(lower=0)

    laps["fuel_corrected_time_s"] = (
        laps["lap_time_s"]
        - fuel_effect * laps_remaining
    )
    valid_pace_time = (
        np.isfinite(laps["lap_time_s"])
        & laps["lap_time_s"].gt(0)
        & laps["lap_time_s"].le(MAX_PLAUSIBLE_LAP_DURATION_S)
    )
    laps.loc[~valid_pace_time, "fuel_corrected_time_s"] = np.nan

    # Ensure the exact required output columns exist.
    for column in LAPS_CLEAN_COLUMNS:
        if column not in laps.columns:
            laps[column] = np.nan

    boolean_columns = [
        "is_inlap",
        "is_outlap",
        "is_sc",
        "is_wet",
    ]

    for column in boolean_columns:
        laps[column] = (
            laps[column].fillna(False).astype(bool)
        )

    numeric_columns = [
        "season",
        "round",
        "lap_number",
        "lap_time_s",
        "fuel_corrected_time_s",
        "tyre_life",
        "stint",
        "position",
        "grid_position",
        "air_temp",
        "track_temp",
        "rainfall",
    ]

    for column in numeric_columns:
        laps[column] = _numeric(laps[column])

    result = laps[LAPS_CLEAN_COLUMNS].copy()

    result = result.sort_values(
        [
            "season",
            "round",
            "race",
            "driver",
            "lap_number",
        ],
        na_position="last",
    ).reset_index(drop=True)

    logger.info(
        "Cleaned dataset contains %d rows", len(result)
    )

    return result

def estimate_pit_loss(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate additional time lost during pit stops for each circuit.

    Calculates pit loss using the in-lap and out-lap times compared
    with the driver's normal pace on nearby clean laps.

    Only green-flag pit stops are included. Extreme estimates are
    filtered using the 3-MAD rule.

    Args:
        laps: Cleaned driver-lap DataFrame from clean_laps().

    Returns:
        One row per circuit containing median pit loss, standard
        deviation, and number of pit stops used.
    """
    output_columns = [
        "circuit",
        "median_pit_loss_s",
        "std_pit_loss_s",
        "n_stops",
    ]

    if laps.empty:
        return pd.DataFrame(columns=output_columns)

    required_columns = {
        "season",
        "round",
        "race",
        "driver",
        "lap_number",
        "lap_time_s",
        "circuit",
        "is_inlap",
        "is_outlap",
        "is_sc",
    }

    missing = required_columns.difference(laps.columns)

    if missing:
        raise ValueError(
            f"Lap data is missing required columns: {sorted(missing)}"
        )

    data = laps.copy()

    data["lap_number"] = _numeric(data["lap_number"])
    data["lap_time_s"] = _numeric(data["lap_time_s"])

    for column in ["is_inlap", "is_outlap", "is_sc"]:
        data[column] = (
            data[column].fillna(False).astype(bool)
        )

    data = data.dropna(
        subset=[
            "season",
            "round",
            "race",
            "driver",
            "lap_number",
            "lap_time_s",
            "circuit",
        ]
    )

    group_columns = [
        "season",
        "round",
        "race",
        "driver",
    ]

    estimates = []

    for _, driver_laps in data.groupby(
        group_columns,
        dropna=False,
    ):
        driver_laps = driver_laps.sort_values(
            "lap_number"
        ).reset_index(drop=True)

        inlaps = driver_laps[
            driver_laps["is_inlap"]
        ]

        for _, inlap in inlaps.iterrows():
            inlap_number = inlap["lap_number"]

            possible_outlaps = driver_laps[
                (driver_laps["lap_number"] > inlap_number)
                & driver_laps["is_outlap"]
            ]

            if possible_outlaps.empty:
                continue

            outlap = possible_outlaps.iloc[0]
            outlap_number = outlap["lap_number"]

            if bool(inlap["is_sc"]) or bool(outlap["is_sc"]):
                continue

            before_pit = driver_laps[
                (
                    driver_laps["lap_number"]
                    >= inlap_number - 2
                )
                & (
                    driver_laps["lap_number"]
                    < inlap_number
                )
            ]

            after_pit = driver_laps[
                (
                    driver_laps["lap_number"]
                    > outlap_number
                )
                & (
                    driver_laps["lap_number"]
                    <= outlap_number + 2
                )
            ]

            clean_neighbours = pd.concat(
                [before_pit, after_pit],
                ignore_index=True,
            )

            clean_neighbours = clean_neighbours[
                (~clean_neighbours["is_inlap"])
                & (~clean_neighbours["is_outlap"])
                & (~clean_neighbours["is_sc"])
                & (clean_neighbours["lap_time_s"] > 0)
            ]

            if len(clean_neighbours) < 2:
                continue

            normal_lap_time = clean_neighbours[
                "lap_time_s"
            ].median()

            pit_loss = (
                inlap["lap_time_s"]
                + outlap["lap_time_s"]
                - 2 * normal_lap_time
            )

            if not np.isfinite(pit_loss):
                continue

            estimates.append(
                {
                    "circuit": inlap["circuit"],
                    "pit_loss_s": float(pit_loss),
                }
            )

    if not estimates:
        logger.warning(
            "No valid green-flag pit stops found"
        )
        return pd.DataFrame(columns=output_columns)

    stop_data = pd.DataFrame(estimates)

    filtered_groups = []

    for circuit, circuit_stops in stop_data.groupby("circuit"):
        median_loss = circuit_stops["pit_loss_s"].median()

        absolute_deviation = (
            circuit_stops["pit_loss_s"] - median_loss
        ).abs()

        mad = absolute_deviation.median()

        if mad > 0:
            circuit_stops = circuit_stops[
                absolute_deviation <= 3 * mad
            ]

        filtered_groups.append(circuit_stops)

    filtered_stops = pd.concat(
        filtered_groups,
        ignore_index=True,
    )

    result = (
        filtered_stops.groupby("circuit")["pit_loss_s"]
        .agg(
            median_pit_loss_s="median",
            std_pit_loss_s="std",
            n_stops="count",
        )
        .reset_index()
    )

    result["n_stops"] = result["n_stops"].astype(int)

    logger.info(
        "Estimated pit loss for %d circuits using %d stops",
        len(result),
        len(filtered_stops),
    )

    return result[output_columns]


if __name__ == "__main__":
    raise SystemExit(main())
