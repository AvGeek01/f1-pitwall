
"""Tests for A3 feature engineering."""

import pandas as pd

from src.features.build import build_features


def test_build_features_on_synthetic_data() -> None:
    """Check that A3 builds all required features and targets."""

    laps = pd.read_parquet(
        "data/processed/synthetic_laps_clean.parquet"
    )

    pit_loss = pd.read_parquet(
        "data/processed/synthetic_pit_loss.parquet"
    )

    features = build_features(laps, pit_loss)

    required_columns = [
        "tyre_age",
        "stint_number",
        "compounds_used",
        "two_compound_rule_satisfied",
        "laps_remaining",
        "race_progress",
        "gap_ahead",
        "gap_behind",
        "gap_leader",
        "rolling_3_lap_pace",
        "degradation_slope",
        "undercut_threat",
        "pits_next_lap",
        "pits_within_3",
        "laps_to_next_pit",
    ]

    for column in required_columns:
        assert column in features.columns

    assert len(features) == len(laps)

    assert features["pits_next_lap"].dtype == bool
    assert features["pits_within_3"].dtype == bool

    assert features["race_progress"].between(0, 1).all()

    assert (features["laps_remaining"] >= 0).all()

    assert features["gap_leader"].dropna().ge(0).all()

    assert features["tyre_age"].notna().any()

    assert features["rolling_3_lap_pace"].notna().any()


def test_features_do_not_use_future_laps():
    """Verify that future laps do not change features at an earlier lap."""

    laps = pd.read_parquet(
        "data/processed/synthetic_laps_clean.parquet"
    )

    pit_loss = pd.read_parquet(
        "data/processed/synthetic_pit_loss.parquet"
    )

    race_key = laps[
        ["season", "round", "race"]
    ].drop_duplicates().iloc[0]

    race_laps = laps[
        (laps["season"] == race_key["season"])
        & (laps["round"] == race_key["round"])
        & (laps["race"] == race_key["race"])
    ].copy()

    selected_driver = race_laps["driver"].iloc[0]
    target_lap = 20

    full_features = build_features(
        race_laps,
        pit_loss,
    )

    truncated_laps = race_laps[
        ~(
            (race_laps["driver"] == selected_driver)
            & (race_laps["lap_number"] > target_lap)
        )
    ].copy()

    truncated_features = build_features(
        truncated_laps,
        pit_loss,
    )

    full_row = full_features[
        (full_features["driver"] == selected_driver)
        & (full_features["lap_number"] == target_lap)
    ]

    truncated_row = truncated_features[
        (truncated_features["driver"] == selected_driver)
        & (truncated_features["lap_number"] == target_lap)
    ]

    assert len(full_row) == 1
    assert len(truncated_row) == 1

    target_columns = [
        "pits_next_lap",
        "pits_within_3",
        "laps_to_next_pit",
    ]

    feature_columns = [
        column
        for column in full_features.columns
        if column not in target_columns
    ]

    pd.testing.assert_frame_equal(
        full_row[feature_columns].reset_index(drop=True),
        truncated_row[feature_columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )
