"""Tests for processed-data schema compatibility helpers."""

import pandas as pd

from src.data.schema import normalize_pit_loss_schema


def test_normalize_pit_loss_schema_renames_legacy_columns() -> None:
    """Translate legacy pit-loss names into the canonical contract."""
    legacy = pd.DataFrame(
        {
            "circuit": ["Test Circuit"],
            "pit_loss_median_s": [22.0],
            "pit_loss_std_s": [1.5],
            "n_stops": [8],
        }
    )

    normalized = normalize_pit_loss_schema(legacy)

    assert normalized.columns.tolist() == [
        "circuit",
        "median_pit_loss_s",
        "std_pit_loss_s",
        "n_stops",
    ]
    assert normalized.loc[0, "median_pit_loss_s"] == 22.0
    assert normalized.loc[0, "std_pit_loss_s"] == 1.5
    assert "pit_loss_median_s" in legacy.columns


def test_normalize_pit_loss_schema_coalesces_both_conventions() -> None:
    """Prefer canonical values, fill their nulls, and drop legacy duplicates."""
    pit_loss = pd.DataFrame(
        {
            "circuit": ["A", "B"],
            "median_pit_loss_s": [21.0, None],
            "pit_loss_median_s": [99.0, 22.0],
            "std_pit_loss_s": [1.0, None],
            "pit_loss_std_s": [9.0, 2.0],
            "n_stops": [5, 6],
        }
    )

    normalized = normalize_pit_loss_schema(pit_loss)

    assert normalized.columns.tolist() == [
        "circuit",
        "median_pit_loss_s",
        "std_pit_loss_s",
        "n_stops",
    ]
    assert normalized["median_pit_loss_s"].tolist() == [21.0, 22.0]
    assert normalized["std_pit_loss_s"].tolist() == [1.0, 2.0]
