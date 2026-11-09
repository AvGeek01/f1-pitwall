"""Tests for race-level pit-stop count modelling."""

import pandas as pd
import pytest

from src.models.stop_count import (
    add_historical_features,
    prepare_race_level_data,
)


def make_laps(
    stop_counts: list[int],
) -> pd.DataFrame:
    """Create a small lap dataset with known pit-stop counts."""
    rows = []

    for driver_number, stops in enumerate(stop_counts, start=1):
        for lap_number in range(1, 11):
            rows.append(
                {
                    "season": 2024,
                    "round": 1,
                    "race": "Test Grand Prix",
                    "circuit": "Test Circuit",
                    "driver": f"DRV{driver_number:02d}",
                    "team": "Test Team",
                    "grid_position": driver_number,
                    "lap_number": lap_number,
                    "is_inlap": lap_number in [
                        3 + 2 * stop_index
                        for stop_index in range(stops)
                    ],
                    "is_outlap": False,
                }
            )

    return pd.DataFrame(rows)


def test_prepare_race_level_data_counts_pit_stops():
    laps = make_laps([1, 2, 3])

    result = prepare_race_level_data(laps)

    counts = result.set_index("driver")["pit_stops"]

    assert counts["DRV01"] == 1
    assert counts["DRV02"] == 2
    assert counts["DRV03"] == 3


def test_three_or_more_stops_share_class_three():
    laps = make_laps([3, 4])

    result = prepare_race_level_data(laps)

    assert result["stop_count_class"].tolist() == [3, 3]


def test_zero_stop_records_are_excluded():
    laps = make_laps([0, 1])

    result = prepare_race_level_data(laps)

    assert len(result) == 1
    assert result.iloc[0]["pit_stops"] == 1


def test_historical_features_use_only_earlier_seasons():
    data = pd.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "circuit": ["Circuit A"] * 3,
            "pit_stops": [2, 1, 3],
        }
    )

    result = add_historical_features(data)

    assert result.loc[0, "historical_circuit_median_stops"] == 2.0

    assert result.loc[1, "historical_circuit_median_stops"] == 2.0

    assert result.loc[2, "historical_circuit_median_stops"] == 1.5


def test_future_season_changes_do_not_affect_past_features():
    data = pd.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "circuit": ["Circuit A"] * 3,
            "pit_stops": [2, 1, 3],
        }
    )

    original = add_historical_features(data)

    changed_data = data.copy()
    changed_data.loc[
        changed_data["season"] == 2025,
        "pit_stops",
    ] = 10

    changed = add_historical_features(changed_data)

    original_2024 = original.loc[
        original["season"] == 2024,
        "historical_circuit_median_stops",
    ].iloc[0]

    changed_2024 = changed.loc[
        changed["season"] == 2024,
        "historical_circuit_median_stops",
    ].iloc[0]

    assert original_2024 == changed_2024


def test_unseen_circuit_uses_global_historical_median():
    data = pd.DataFrame(
        {
            "season": [2023, 2023, 2024],
            "circuit": [
                "Circuit A",
                "Circuit B",
                "Circuit C",
            ],
            "pit_stops": [1, 3, 2],
        }
    )

    result = add_historical_features(data)

    unseen_circuit_value = result.loc[
        result["season"] == 2024,
        "historical_circuit_median_stops",
    ].iloc[0]

    assert unseen_circuit_value == 2.0