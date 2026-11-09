
"""Tests for A4 grouped validation and baseline pit-lap prediction."""

import pandas as pd
import pytest

from src.models import pit_lap

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


def make_a5_validation_dataset():
    """Create the minimum dataset required by A5 feature validation."""

    data = {
        column: [0, 1, 2, 3]
        for column in pit_lap.A5_FEATURES
    }

    data.update({
        pit_lap.A5_TARGET_REGRESSION: [5.0, 4.0, 3.0, 2.0],
        pit_lap.A5_TARGET_CLASSIFICATION: [0, 0, 1, 0],
        "season": [2023, 2023, 2024, 2024],
        "round": [1, 1, 2, 2],
        "race": ["Race A", "Race A", "Race B", "Race B"],
    })

    return pd.DataFrame(data)


def test_validate_a5_features_accepts_valid_data():
    """Valid A5 predictors, targets and race identifiers are accepted."""

    data = make_a5_validation_dataset()

    pit_lap.validate_a5_features(data)


def test_validate_a5_features_rejects_missing_feature():
    """A5 validation must identify missing predictor columns."""

    data = make_a5_validation_dataset()
    data = data.drop(columns=[pit_lap.A5_FEATURES[0]])

    with pytest.raises(ValueError, match="Missing columns"):
        pit_lap.validate_a5_features(data)


def test_calculate_scale_pos_weight():
    """Class imbalance weight equals negatives divided by positives."""

    target = pd.Series([0, 0, 0, 1])

    result = pit_lap.calculate_scale_pos_weight(target)

    assert result == 3.0


def test_scale_pos_weight_rejects_no_positive_samples():
    """The class weight cannot be calculated without positive samples."""

    target = pd.Series([0, 0, 0, 0])

    with pytest.raises(ValueError, match="no positive"):
        pit_lap.calculate_scale_pos_weight(target)


def test_valid_training_rows_removes_missing_targets():
    """Rows without known target values must be excluded."""

    data = pd.DataFrame({
        "feature": [10, 20, 30],
        "target": [1.0, None, 3.0],
    })

    result = pit_lap._valid_training_rows(data, "target")

    assert len(result) == 2
    assert result["target"].notna().all()


def test_group_cv_keeps_races_separate():
    """A race must never appear in both sides of a CV fold."""

    data = pd.DataFrame({
        "season": [2023, 2023, 2023, 2023],
        "round": [1, 1, 2, 2],
        "race": ["A", "A", "B", "B"],
    })

    splitter, groups = pit_lap._check_group_cv(data, n_splits=2)

    for train_indices, validation_indices in splitter.split(
        data,
        groups=groups,
    ):
        train_races = set(groups.iloc[train_indices])
        validation_races = set(groups.iloc[validation_indices])

        assert train_races.isdisjoint(validation_races)


def test_f1_threshold_selection():
    """Threshold selection returns a valid threshold and F1 score."""

    actual = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.30, 0.70, 0.90])

    threshold, f1 = pit_lap.select_f1_threshold(
        actual,
        probabilities,
    )

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.0


def test_f1_threshold_rejects_single_class():
    """Threshold selection requires both positive and negative labels."""

    actual = pd.Series([0, 0, 0])
    probabilities = pd.Series([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="both classes"):
        pit_lap.select_f1_threshold(actual, probabilities)


def test_build_a5_pipeline_supports_both_tasks():
    """A5 must build separate regression and classification pipelines."""

    regression = pit_lap.build_a5_pipeline(
        task="regression",
        parameters={"n_estimators": 5, "max_depth": 2},
    )

    classification = pit_lap.build_a5_pipeline(
        task="classification",
        parameters={"n_estimators": 5, "max_depth": 2},
        scale_pos_weight=3.0,
    )

    assert "preprocessor" in regression.named_steps
    assert "model" in regression.named_steps

    assert "preprocessor" in classification.named_steps
    assert "model" in classification.named_steps


def test_build_a5_pipeline_rejects_invalid_task():
    """Unknown model tasks must raise a clear error."""

    with pytest.raises(ValueError, match="task must be"):
        pit_lap.build_a5_pipeline(
            task="invalid",
            parameters={},
        )
