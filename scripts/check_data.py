"""Validate the processed OpenF1 datasets without changing them."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import PIT_LOSS_COLUMNS, normalize_pit_loss_schema

DEFAULT_DATA_DIR = ROOT / "data" / "processed"
REQUIRED_SEASONS = (2023, 2024, 2025)

LAPS_COLUMNS = (
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
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
    "air_temp",
    "track_temp",
    "rainfall",
    "grid_source",
)
IMPORTANT_LAPS_COLUMNS = (
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
)
IMPORTANT_PIT_LOSS_COLUMNS = (
    "circuit",
    "median_pit_loss_s",
    "std_pit_loss_s",
    "n_stops",
)
SYNTHETIC_LAPS_ROWS = 49_500
SYNTHETIC_RACES_PER_SEASON = 15
SYNTHETIC_CIRCUITS = 3
SYNTHETIC_CIRCUIT_NAMES = {"Bahrain", "Monaco", "Silverstone"}
SYNTHETIC_DRIVERS = 20
SYNTHETIC_LAPS_PER_RACE = 55
SYNTHETIC_MIN_STOPS_PER_CIRCUIT = 350


def _report_missing_percentages(
    frame: pd.DataFrame,
    columns: Sequence[str],
    dataset_name: str,
    warnings: list[str],
) -> None:
    """Print missing-value percentages and record any affected columns."""
    for column in columns:
        if column not in frame.columns:
            print(f"  Missing {column}: unavailable (column missing)")
            continue

        missing_count = int(frame[column].isna().sum())
        missing_percent = 100 * missing_count / len(frame) if len(frame) else 0.0
        print(
            f"  Missing {column}: {missing_percent:.2f}% "
            f"({missing_count:,}/{len(frame):,})"
        )
        if missing_count:
            warnings.append(
                f"{dataset_name}.{column} has {missing_percent:.2f}% missing values"
            )


def _is_copy_of_synthetic(
    actual: pd.DataFrame,
    synthetic_path: Path,
    columns: Sequence[str],
) -> bool:
    """Compare selected columns against a generated synthetic dataset."""
    if not synthetic_path.is_file():
        return False

    synthetic = pd.read_parquet(synthetic_path)
    shared_columns = [column for column in columns if column in synthetic.columns]
    if not shared_columns or not set(shared_columns).issubset(actual.columns):
        return False
    if len(actual) != len(synthetic):
        return False

    actual_hashes = pd.util.hash_pandas_object(
        actual[shared_columns], index=False
    ).sort_values(ignore_index=True)
    synthetic_hashes = pd.util.hash_pandas_object(
        synthetic[shared_columns], index=False
    ).sort_values(ignore_index=True)
    return actual_hashes.equals(synthetic_hashes)


def _has_synthetic_laps_structure(laps: pd.DataFrame) -> bool:
    """Identify the generator's distinctive fixed race and lap dimensions."""
    required = {"season", "round", "race", "circuit", "driver", "lap_number"}
    if not required.issubset(laps.columns) or len(laps) != SYNTHETIC_LAPS_ROWS:
        return False

    race_keys = ["season", "round", "race"]
    races_by_season = laps[race_keys].drop_duplicates().groupby("season").size()
    drivers_by_race = laps.groupby(race_keys)["driver"].nunique()
    laps_by_driver_race = laps.groupby(race_keys + ["driver"])["lap_number"].agg(
        ["count", "min", "max", "nunique"]
    )
    return bool(
        len(races_by_season) == len(REQUIRED_SEASONS)
        and races_by_season.eq(SYNTHETIC_RACES_PER_SEASON).all()
        and laps["circuit"].nunique() == SYNTHETIC_CIRCUITS
        and drivers_by_race.eq(SYNTHETIC_DRIVERS).all()
        and laps_by_driver_race["count"].eq(SYNTHETIC_LAPS_PER_RACE).all()
        and laps_by_driver_race["min"].eq(1).all()
        and laps_by_driver_race["max"].eq(SYNTHETIC_LAPS_PER_RACE).all()
        and laps_by_driver_race["nunique"].eq(SYNTHETIC_LAPS_PER_RACE).all()
    )


def _has_synthetic_pit_loss_structure(pit_loss: pd.DataFrame) -> bool:
    """Identify the generator's synthetic circuits and unusually large stop counts."""
    required = {"circuit", "n_stops"}
    if not required.issubset(pit_loss.columns) or len(pit_loss) != len(
        SYNTHETIC_CIRCUIT_NAMES
    ):
        return False

    stops = pd.to_numeric(pit_loss["n_stops"], errors="coerce")
    return bool(
        set(pit_loss["circuit"].dropna()) == SYNTHETIC_CIRCUIT_NAMES
        and stops.notna().all()
        and stops.ge(SYNTHETIC_MIN_STOPS_PER_CIRCUIT).all()
    )


def _check_laps(
    laps: pd.DataFrame,
    warnings: list[str],
    failures: list[str],
    synthetic_path: Path,
) -> None:
    """Validate lap data quality and synthetic-data fingerprints."""
    if "season" in laps.columns:
        seasons_text = sorted(laps["season"].dropna().unique().tolist())
    else:
        seasons_text = "unavailable"
    print(
        f"Laps: {len(laps):,} rows, {len(laps.columns):,} columns; "
        f"seasons={seasons_text}"
    )
    missing_columns = sorted(set(LAPS_COLUMNS).difference(laps.columns))
    if missing_columns:
        failures.append(f"laps file is missing required columns: {missing_columns}")
    _report_missing_percentages(
        laps, IMPORTANT_LAPS_COLUMNS, "laps", warnings
    )

    if {"season", "round", "driver", "lap_number"}.issubset(laps.columns):
        duplicate_count = int(
            laps.duplicated(
                subset=["season", "round", "driver", "lap_number"],
                keep=False,
            ).sum()
        )
        print(f"  Duplicate driver-lap records: {duplicate_count:,}")
        if duplicate_count:
            failures.append(
                f"laps file has {duplicate_count:,} duplicate driver-lap records"
            )

    if "season" in laps.columns:
        available_seasons = set(pd.to_numeric(laps["season"], errors="coerce").dropna())
        missing_seasons = sorted(set(REQUIRED_SEASONS).difference(available_seasons))
        if missing_seasons:
            failures.append(f"laps file is missing seasons: {missing_seasons}")
    else:
        failures.append("laps file cannot be checked for required seasons")

    invalid_counts: dict[str, int] = {}
    for column in ("lap_time_s", "fuel_corrected_time_s"):
        if column in laps.columns:
            lap_times = pd.to_numeric(laps[column], errors="coerce")
            invalid_counts[column] = int(
                (lap_times.notna() & ~lap_times.between(30, 300)).sum()
                + (laps[column].notna() & lap_times.isna()).sum()
            )
    if "tyre_life" in laps.columns:
        tyre_life = pd.to_numeric(laps["tyre_life"], errors="coerce")
        invalid_counts["tyre_life"] = int(
            (tyre_life.notna() & (tyre_life <= 0)).sum()
            + (laps["tyre_life"].notna() & tyre_life.isna()).sum()
        )
    if "lap_number" in laps.columns:
        lap_numbers = pd.to_numeric(laps["lap_number"], errors="coerce")
        invalid_counts["lap_number"] = int(
            (
                lap_numbers.notna()
                & ((lap_numbers <= 0) | (lap_numbers % 1 != 0))
            ).sum()
            + (laps["lap_number"].notna() & lap_numbers.isna()).sum()
        )

    for column, count in invalid_counts.items():
        print(f"  Invalid {column} values: {count:,}")
        if count:
            failures.append(f"laps file has {count:,} invalid {column} values")

    if _is_copy_of_synthetic(laps, synthetic_path, LAPS_COLUMNS):
        failures.append(
            f"laps file matches the synthetic dataset {synthetic_path.name}"
        )
    elif _has_synthetic_laps_structure(laps):
        failures.append(
            "laps file matches the synthetic generator's fixed structure "
            "(49,500 rows; 3 circuits; 15 races per season; "
            "20 drivers; 55 laps)"
        )


def _check_pit_loss(
    pit_loss: pd.DataFrame,
    warnings: list[str],
    failures: list[str],
    synthetic_path: Path,
) -> None:
    """Validate pit-loss values and identify the generated synthetic output."""
    print(
        f"Pit loss: {len(pit_loss):,} rows, {len(pit_loss.columns):,} columns"
    )
    missing_columns = sorted(set(PIT_LOSS_COLUMNS).difference(pit_loss.columns))
    if missing_columns:
        failures.append(
            f"pit-loss file is missing required columns: {missing_columns}"
        )
    _report_missing_percentages(
        pit_loss, IMPORTANT_PIT_LOSS_COLUMNS, "pit_loss", warnings
    )

    if "median_pit_loss_s" in pit_loss.columns:
        median = pd.to_numeric(pit_loss["median_pit_loss_s"], errors="coerce")
        outside_range = median.notna() & ~median.between(15, 35)
        if outside_range.any():
            warning = (
                f"{int(outside_range.sum())} pit-loss estimates are outside "
                "the plausible 15-35 second range"
            )
            warnings.append(warning)
            print(f"  WARNING: {warning}")

    if _is_copy_of_synthetic(pit_loss, synthetic_path, PIT_LOSS_COLUMNS):
        failures.append(
            f"pit-loss file matches the synthetic dataset {synthetic_path.name}"
        )
    elif _has_synthetic_pit_loss_structure(pit_loss):
        failures.append(
            "pit-loss file matches the synthetic generator's three-circuit "
            "structure and unusually large per-circuit stop counts"
        )


def validate_data(data_directory: Path) -> int:
    """Validate processed datasets and return a process status code."""
    laps_path = data_directory / "laps_clean.parquet"
    pit_loss_path = data_directory / "pit_loss.parquet"
    synthetic_laps_path = data_directory / "synthetic_laps_clean.parquet"
    synthetic_pit_loss_path = data_directory / "synthetic_pit_loss.parquet"
    warnings: list[str] = []
    failures: list[str] = []

    print(f"Checking processed data in {data_directory}")
    datasets: dict[str, pd.DataFrame] = {}
    for name, path in (("laps", laps_path), ("pit-loss", pit_loss_path)):
        if not path.is_file():
            failures.append(f"required {name} file does not exist: {path}")
            print(f"FAIL: missing {name} file: {path}")
            continue
        try:
            frame = pd.read_parquet(path)
            datasets[name] = (
                normalize_pit_loss_schema(frame)
                if name == "pit-loss"
                else frame
            )
        except (OSError, ValueError, ImportError) as error:
            failures.append(f"could not read {name} file {path}: {error}")
            print(f"FAIL: could not read {name} file: {error}")

    if "laps" in datasets:
        _check_laps(
            datasets["laps"], warnings, failures, synthetic_laps_path
        )
    if "pit-loss" in datasets:
        _check_pit_loss(
            datasets["pit-loss"],
            warnings,
            failures,
            synthetic_pit_loss_path,
        )

    print("\nValidation summary:")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        print(f"RESULT: FAIL ({len(failures)} critical issue(s))")
        return 1
    if warnings:
        print(f"RESULT: WARNING ({len(warnings)} warning(s))")
        return 0
    print("RESULT: PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, validate the datasets, and return a process status."""
    parser = argparse.ArgumentParser(
        description="Check the processed Pit Wall F1 datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing laps_clean.parquet and pit_loss.parquet",
    )
    arguments = parser.parse_args(argv)
    return validate_data(arguments.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
