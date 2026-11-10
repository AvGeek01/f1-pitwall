
"""Tests for A3 feature engineering."""

import pandas as pd

from src.features.build import _add_gap_features, build_features


def test_gap_features_leave_gaps_missing_when_all_cumulative_times_are_missing() -> None:
    """Retain rows and leave gaps missing when no car has a valid race time."""
    laps = pd.DataFrame(
        {
            "driver": ["AAA", "BBB", "AAA", "BBB"],
            "lap_number": [1, 1, 2, 2],
            "lap_time_s": [90.0, 91.0, float("nan"), float("nan")],
        },
        index=[10, 11, 12, 13],
    )

    result = _add_gap_features(laps)

    assert len(result) == len(laps)
    assert sorted(result.index.tolist()) == sorted(laps.index.tolist())
    affected_lap = result.loc[result["lap_number"].eq(2)]
    assert affected_lap[
        ["gap_ahead", "gap_behind", "gap_leader"]
    ].isna().all().all()
    valid_lap = result.loc[result["lap_number"].eq(1)].set_index("driver")
    assert valid_lap.loc["AAA", "gap_leader"] == 0.0
    assert valid_lap.loc["BBB", "gap_leader"] == 1.0
    assert pd.isna(valid_lap.loc["AAA", "gap_ahead"])
    assert valid_lap.loc["AAA", "gap_behind"] == 1.0
    assert valid_lap.loc["BBB", "gap_ahead"] == 1.0
    assert pd.isna(valid_lap.loc["BBB", "gap_behind"])


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


def test_features_accept_legacy_pit_loss_columns() -> None:
    """Normalize legacy pit-loss names before feature construction."""
    laps = pd.read_parquet("data/processed/synthetic_laps_clean.parquet")
    pit_loss = pd.read_parquet("data/processed/synthetic_pit_loss.parquet")
    legacy_pit_loss = pit_loss.rename(
        columns={
            "median_pit_loss_s": "pit_loss_median_s",
            "std_pit_loss_s": "pit_loss_std_s",
        }
    )

    features = build_features(laps.iloc[:1000], legacy_pit_loss)

    assert len(features) == 1000
    assert "undercut_threat" in features.columns


def test_missing_lap_time_does_not_create_cumulative_time() -> None:
    """Keep unknown cumulative timing missing after a driver has a missing lap."""
    laps = pd.read_parquet("data/processed/synthetic_laps_clean.parquet")
    pit_loss = pd.read_parquet("data/processed/synthetic_pit_loss.parquet")
    race_key = laps[["season", "round", "race"]].drop_duplicates().iloc[0]
    race_laps = laps.loc[
        laps["season"].eq(race_key["season"])
        & laps["round"].eq(race_key["round"])
        & laps["race"].eq(race_key["race"])
    ].copy()
    driver = str(race_laps["driver"].iloc[0])
    target_lap = 20
    missing_mask = (
        race_laps["driver"].eq(driver)
        & race_laps["lap_number"].eq(target_lap)
    )
    race_laps.loc[
        missing_mask,
        ["lap_time_s", "fuel_corrected_time_s"],
    ] = float("nan")

    features = build_features(race_laps, pit_loss)
    driver_features = features.loc[features["driver"].eq(driver)].set_index(
        "lap_number"
    )

    assert len(features) == len(race_laps)
    assert pd.isna(driver_features.loc[target_lap, "cumulative_race_time"])
    assert pd.isna(
        driver_features.loc[target_lap + 1, "cumulative_race_time"]
    )
