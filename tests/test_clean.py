
"""Tests for OpenF1 lap cleaning and pit-loss estimation."""

import pandas as pd
import pytest

from src.data.clean import clean_laps, estimate_pit_loss


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


def test_estimate_pit_loss_calculates_correct_value():
    """A 100-second in-lap and out-lap against 90-second pace means 20 seconds."""
    laps = make_test_laps()

    result = estimate_pit_loss(laps)

    assert len(result) == 1
    assert result.iloc[0]["circuit"] == "Melbourne"
    assert result.iloc[0]["pit_loss_median_s"] == pytest.approx(20.0)
    assert result.iloc[0]["n_stops"] == 1


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
        "pit_loss_median_s",
        "pit_loss_std_s",
        "n_stops",
    ]


def test_estimate_pit_loss_rejects_missing_columns():
    """Missing required columns should produce a clear error."""
    laps = pd.DataFrame({"driver": ["VER"]})

    with pytest.raises(ValueError, match="missing required columns"):
        estimate_pit_loss(laps)
