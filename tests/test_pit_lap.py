
"""Tests for A4 grouped validation and baseline pit-lap prediction."""

import pandas as pd
import pytest

from src.models.pit_lap import (
    split_by_season,
    predict_baseline,
    evaluate_baseline_cv,
)


def make_small_dataset():
    """Create a small dataset with five independent races."""

    rows = []

    for race_number in range(1, 6):
        rows.append({
            "season": 2023,
            "round": race_number,
            "race": f"Race {race_number}",
            "circuit": "Bahrain",
            "driver": "DRV01",
            "compound": "MEDIUM",
            "stint": 1,
            "tyre_age": 2,
            "is_inlap": True,
            "laps_to_next_pit": 1.0,
        })

    return pd.DataFrame(rows)


def test_split_by_season():
    """Training must contain 2023-24 and testing only 2025."""

    data = pd.DataFrame({
        "season": [2023, 2024, 2025, 2025],
        "value": [10, 20, 30, 40],
    })

    train, test = split_by_season(data)

    assert set(train["season"]) == {2023, 2024}
    assert set(test["season"]) == {2025}

    assert len(train) == 2
    assert len(test) == 2


def test_split_rejects_missing_holdout():
    """The split must fail if the requested holdout season is absent."""

    data = pd.DataFrame({
        "season": [2023, 2024],
        "value": [10, 20],
    })

    with pytest.raises(ValueError):
        split_by_season(data)


def test_baseline_uses_training_median():
    """Predictions must use median stint length learned from training."""

    training = pd.DataFrame({
        "season": [2023, 2023, 2024],
        "round": [1, 2, 1],
        "race": ["A", "B", "C"],
        "circuit": ["Bahrain"] * 3,
        "driver": ["DRV01", "DRV01", "DRV01"],
        "compound": ["MEDIUM"] * 3,
        "stint": [1, 1, 1],
        "tyre_age": [20, 24, 28],
        "is_inlap": [True, True, True],
        "laps_to_next_pit": [1.0, 1.0, 1.0],
    })

    prediction_data = pd.DataFrame({
        "circuit": ["Bahrain"],
        "compound": ["MEDIUM"],
        "tyre_age": [18],
    })

    result = predict_baseline(
        training,
        prediction_data,
    )

    assert result["baseline_predicted_laps"].iloc[0] == 6


def test_baseline_uses_global_fallback():
    """Unseen circuit-compound combinations use training global median."""

    training = pd.DataFrame({
        "season": [2023, 2023],
        "round": [1, 2],
        "race": ["A", "B"],
        "circuit": ["Bahrain", "Bahrain"],
        "driver": ["DRV01", "DRV01"],
        "compound": ["MEDIUM", "MEDIUM"],
        "stint": [1, 1],
        "tyre_age": [20, 30],
        "is_inlap": [True, True],
        "laps_to_next_pit": [1.0, 1.0],
    })

    prediction_data = pd.DataFrame({
        "circuit": ["Monaco"],
        "compound": ["SOFT"],
        "tyre_age": [10],
    })

    result = predict_baseline(
        training,
        prediction_data,
    )

    assert result["baseline_predicted_laps"].iloc[0] == 15


def test_grouped_cv_runs_without_race_overlap():
    """Grouped cross-validation must work with independent races."""

    data = make_small_dataset()

    results = evaluate_baseline_cv(
        data,
        n_splits=5,
    )

    assert len(results) == 5

    assert results["mae"].notna().all()

    assert (results["validation_races"] == 1).all()

    assert (results["validation_samples"] == 1).all()
