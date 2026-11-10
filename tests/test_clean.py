
"""Tests for OpenF1 lap cleaning and pit-loss estimation."""

from pathlib import Path
import sys

import pandas as pd
import pytest

from src.data.clean import (
    CLI_RAW_ENDPOINTS,
    LAPS_CLEAN_COLUMNS,
    clean_laps,
    estimate_pit_loss,
    load_raw_data,
    main,
)
from src.data.schema import PIT_LOSS_COLUMNS


def make_test_laps(
    inlap_time: float = 100.0,
    outlap_time: float = 100.0,
    safety_car: bool = False,
) -> pd.DataFrame:
    """Create a small race dataset with one known pit stop."""
    return pd.DataFrame(
        {
            "season": [2024] * 8,
            "round": [3] * 8,
            "race": ["Australian GP"] * 8,
            "driver": ["VER"] * 8,
            "circuit": ["Melbourne"] * 8,
            "lap_number": list(range(1, 9)),
            "lap_time_s": [
                90.0,
                90.0,
                90.0,
                inlap_time,
                outlap_time,
                90.0,
                90.0,
                90.0,
            ],
            "is_inlap": [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            "is_outlap": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            "is_sc": [
                False,
                False,
                False,
                False,
                safety_car,
                False,
                False,
                False,
            ],
        }
    )


def make_raw_laps(
    durations: list[float | None],
    timestamp_offsets_s: list[float | None],
) -> pd.DataFrame:
    """Build raw lap rows with optional start-time offsets for duration tests."""
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "session_key": [100] * len(durations),
            "driver_number": [1] * len(durations),
            "lap_number": list(range(1, len(durations) + 1)),
            "lap_duration": durations,
            "date_start": [
                start + pd.Timedelta(seconds=offset)
                if offset is not None
                else pd.NaT
                for offset in timestamp_offsets_s
            ],
        }
    )


def test_clean_laps_retains_normal_raw_duration() -> None:
    """Keep normal OpenF1 durations as measured lap times."""
    cleaned = clean_laps({"laps": make_raw_laps([92.5], [0.0])})

    assert cleaned.loc[0, "lap_time_s"] == pytest.approx(92.5)
    assert cleaned.loc[0, "fuel_corrected_time_s"] == pytest.approx(92.5)


def test_clean_laps_rejects_raw_duration_above_threshold() -> None:
    """Represent an implausibly long raw duration as missing pace."""
    raw = {"laps": make_raw_laps([90.0, 400.0], [0.0, 100.0])}

    cleaned = clean_laps(raw)

    assert len(cleaned) == len(raw["laps"])
    assert cleaned.loc[0, "lap_time_s"] == pytest.approx(90.0)
    assert pd.isna(cleaned.loc[1, "lap_time_s"])
    assert pd.isna(cleaned.loc[1, "fuel_corrected_time_s"])


def test_clean_laps_rejects_timestamp_fallback_above_threshold() -> None:
    """Do not accept a timestamp difference longer than 300 seconds."""
    raw = {
        "laps": make_raw_laps(
            [90.0, None],
            [0.0, 400.0],
        )
    }

    cleaned = clean_laps(raw)

    assert len(cleaned) == 2
    assert pd.isna(cleaned.loc[1, "lap_time_s"])
    assert pd.isna(cleaned.loc[1, "fuel_corrected_time_s"])


def test_clean_laps_accepts_valid_timestamp_fallback() -> None:
    """Use a finite 30-300 second interval when the raw duration is missing."""
    cleaned = clean_laps(
        {"laps": make_raw_laps([90.0, None], [0.0, 100.0])}
    )

    assert cleaned.loc[1, "lap_time_s"] == pytest.approx(100.0)
    assert cleaned.loc[1, "fuel_corrected_time_s"] == pytest.approx(100.0)


def test_clean_laps_keeps_missing_duration_and_driver_lap_row() -> None:
    """Keep the driver-lap row and leave pace missing without a valid source."""
    raw_laps = make_raw_laps([None, 400.0, 90.0], [None, None, 100.0])

    cleaned = clean_laps({"laps": raw_laps})

    assert len(cleaned) == len(raw_laps)
    assert cleaned["lap_number"].tolist() == [1, 2, 3]
    assert pd.isna(cleaned.loc[0, "lap_time_s"])
    assert pd.isna(cleaned.loc[0, "fuel_corrected_time_s"])
    assert pd.isna(cleaned.loc[1, "lap_time_s"])
    assert pd.isna(cleaned.loc[1, "fuel_corrected_time_s"])


def test_estimate_pit_loss_calculates_correct_value():
    """A 100-second in-lap and out-lap against 90-second pace means 20 seconds."""
    laps = make_test_laps()

    result = estimate_pit_loss(laps)

    assert len(result) == 1
    assert result.iloc[0]["circuit"] == "Melbourne"
    assert result.iloc[0]["median_pit_loss_s"] == pytest.approx(20.0)
    assert result.iloc[0]["n_stops"] == 1
    assert result.columns.tolist() == list(PIT_LOSS_COLUMNS)


def test_estimate_pit_loss_excludes_safety_car_stop():
    """Stops involving a Safety Car lap must not be included."""
    laps = make_test_laps(safety_car=True)

    result = estimate_pit_loss(laps)

    assert result.empty


def test_estimate_pit_loss_returns_empty_for_empty_input():
    """An empty input should return an empty result with the expected columns."""
    laps = pd.DataFrame()

    result = estimate_pit_loss(laps)

    assert result.empty
    assert list(result.columns) == [
        "circuit",
        "median_pit_loss_s",
        "std_pit_loss_s",
        "n_stops",
    ]


def test_estimate_pit_loss_rejects_missing_columns():
    """Missing required columns should produce a clear error."""
    laps = pd.DataFrame({"driver": ["VER"]})

    with pytest.raises(ValueError, match="missing required columns"):
        estimate_pit_loss(laps)


def test_empty_starting_grid_falls_back_to_qualifying_then_race_position(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use qualifying results first, then earliest race position per driver."""
    caplog.set_level("INFO", logger="src.data.clean")
    lap_rows = [
        {
            "session_key": 100,
            "meeting_key": 200,
            "driver_number": driver_number,
            "lap_number": lap_number,
            "lap_duration": 90.0,
            "date_start": f"2024-03-02T15:0{lap_number}:00+00:00",
        }
        for driver_number in (1, 2, 3)
        for lap_number in (1, 2)
    ]
    raw = {
        "laps": pd.DataFrame(lap_rows),
        "drivers": pd.DataFrame(
            {
                "session_key": [100, 100, 100],
                "driver_number": [1, 2, 3],
                "name_acronym": ["AAA", "BBB", "CCC"],
                "team_name": ["Team A", "Team B", "Team C"],
            }
        ),
        "meetings": pd.DataFrame(
            {
                "meeting_key": [200],
                "meeting_name": ["Test Grand Prix"],
                "circuit_short_name": ["Test Circuit"],
                "year": [2024],
                "round": [1],
            }
        ),
        "starting_grid": pd.DataFrame(),
        "session_result": pd.DataFrame(
            {
                "meeting_key": [200, 200],
                "driver_number": [1, 3],
                "position": [2, 3],
            }
        ),
        "position": pd.DataFrame(
            {
                "session_key": [100, 100, 100, 100],
                "driver_number": [1, 2, 3, 2],
                "date": [
                    "2024-03-02T15:00:00+00:00",
                    "2024-03-02T15:00:03+00:00",
                    "2024-03-02T15:00:02+00:00",
                    "2024-03-02T15:00:04+00:00",
                ],
                "position": [1, 4, 2, 1],
            }
        ),
    }

    cleaned = clean_laps(raw)

    assert cleaned.columns.tolist() == LAPS_CLEAN_COLUMNS
    by_driver = cleaned.drop_duplicates("driver").set_index("driver")
    assert by_driver.loc["AAA", "grid_position"] == 2
    assert by_driver.loc["AAA", "grid_source"] == "qualifying_session_result"
    assert by_driver.loc["BBB", "grid_position"] == 4
    assert by_driver.loc["BBB", "grid_source"] == "race_position"
    assert by_driver.loc["CCC", "grid_position"] == 3
    assert by_driver.loc["CCC", "grid_source"] == "qualifying_session_result"
    assert "Grid position sources by driver" in caplog.text
    assert "'qualifying_session_result': 2" in caplog.text
    assert "'race_position': 1" in caplog.text

    raw["starting_grid"] = pd.DataFrame(
        {
            "session_key": [100],
            "driver_number": [1],
            "position": [1],
        }
    )
    with_starting_grid = clean_laps(raw).drop_duplicates("driver").set_index("driver")
    assert with_starting_grid.loc["AAA", "grid_position"] == 1
    assert with_starting_grid.loc["AAA", "grid_source"] == "starting_grid"


def test_load_raw_data_prefers_full_and_falls_back_to_partial(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Choose one file per endpoint/year, preferring full data when available."""
    caplog.set_level("INFO", logger="src.data.clean")
    pd.DataFrame({"value": ["full"]}).to_parquet(
        tmp_path / "laps_2023.parquet",
        index=False,
    )
    pd.DataFrame({"value": ["ignored partial"]}).to_parquet(
        tmp_path / "laps_partial_2023.parquet",
        index=False,
    )
    pd.DataFrame({"value": ["partial"]}).to_parquet(
        tmp_path / "laps_partial_2024.parquet",
        index=False,
    )
    pd.DataFrame({"value": ["positions"]}).to_parquet(
        tmp_path / "position_2023.parquet",
        index=False,
    )

    raw = load_raw_data(
        years=(2023, 2024),
        data_directory=tmp_path,
        endpoints=("laps", "position"),
        required_endpoints=("laps",),
    )

    assert raw["laps"]["value"].tolist() == ["full", "partial"]
    assert raw["position"]["value"].tolist() == ["positions"]
    assert "laps_2023.parquet" in caplog.text
    assert "laps_partial_2024.parquet" in caplog.text


def test_load_raw_data_supports_alternate_partial_filename(
    tmp_path: Path,
) -> None:
    """Load endpoint_year_partial files when full and legacy names are absent."""
    pd.DataFrame({"value": ["alternate"]}).to_parquet(
        tmp_path / "laps_2023_partial.parquet",
        index=False,
    )

    raw = load_raw_data(
        years=(2023,),
        data_directory=tmp_path,
        endpoints=("laps",),
        required_endpoints=("laps",),
    )

    assert raw["laps"]["value"].tolist() == ["alternate"]


def test_clean_main_writes_processed_outputs_from_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI reads partial per-endpoint files and writes both outputs."""
    raw_directory = tmp_path / "raw"
    output_directory = tmp_path / "processed"
    raw_directory.mkdir()
    pd.DataFrame(
        {
            "session_key": [100],
            "meeting_key": [200],
            "driver_number": [1],
            "lap_number": [1],
            "lap_duration": [90.0],
        }
    ).to_parquet(raw_directory / "laps_2023_partial.parquet", index=False)
    pd.DataFrame(
        {
            "session_key": [100],
            "meeting_key": [200],
            "driver_number": [1],
            "name_acronym": ["AAA"],
            "team_name": ["Team A"],
        }
    ).to_parquet(raw_directory / "drivers_partial_2023.parquet", index=False)
    pd.DataFrame(
        {
            "meeting_key": [200],
            "meeting_name": ["Test Grand Prix"],
            "circuit_short_name": ["Test Circuit"],
            "year": [2023],
            "round": [1],
        }
    ).to_parquet(raw_directory / "meetings_2023.parquet", index=False)
    pd.DataFrame(
        {
            "session_key": [100],
            "meeting_key": [200],
            "driver_number": [1],
            "stint_number": [1],
            "compound": ["MEDIUM"],
            "lap_start": [1],
            "lap_end": [1],
            "tyre_age_at_start": [0],
        }
    ).to_parquet(raw_directory / "stints_2023.parquet", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean.py",
            "--raw-dir",
            str(raw_directory),
            "--out-dir",
            str(output_directory),
            "--years",
            "2023",
        ],
    )

    assert main() == 0
    clean_output = pd.read_parquet(output_directory / "laps_clean.parquet")
    pit_loss_output = pd.read_parquet(output_directory / "pit_loss.parquet")
    assert clean_output.columns.tolist() == LAPS_CLEAN_COLUMNS
    assert len(clean_output) == 1
    assert clean_output.loc[0, "grid_source"] == "unavailable"
    assert pit_loss_output.empty
    assert set(CLI_RAW_ENDPOINTS).issuperset({"laps", "starting_grid"})


def test_clean_main_fails_clearly_when_no_raw_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return a non-zero status and log clearly when raw files are absent."""
    raw_directory = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        ["clean.py", "--raw-dir", str(raw_directory)],
    )

    assert main() != 0
    assert "No raw parquet files found" in caplog.text
