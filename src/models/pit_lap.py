
"""A4: Race-grouped validation and baseline pit-lap prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.features.build import build_features

logger = logging.getLogger(__name__)

RACE_COLUMNS = ["season", "round", "race"]

STINT_GROUP_COLUMNS = [
    "season",
    "round",
    "race",
    "driver",
    "stint",
    "circuit",
    "compound",
]


def prepare_model_data(
    laps: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> pd.DataFrame:
    """Generate model-ready features and targets using A3."""

    features = build_features(laps, pit_loss)

    required_columns = [
        "season",
        "round",
        "race",
        "circuit",
        "driver",
        "compound",
        "tyre_age",
        "stint",
        "is_inlap",
        "laps_to_next_pit",
    ]

    missing = set(required_columns) - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for A4: {sorted(missing)}"
        )

    return features


def split_by_season(
    features: pd.DataFrame,
    train_seasons: tuple[int, ...] = (2023, 2024),
    test_season: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by season, keeping the final test season separate."""

    if "season" not in features.columns:
        raise ValueError("Features must contain a season column.")

    train_data = features[
        features["season"].isin(train_seasons)
    ].copy()

    test_data = features[
        features["season"] == test_season
    ].copy()

    if train_data.empty:
        raise ValueError("Training dataset is empty.")

    if test_data.empty:
        raise ValueError(
            f"No data available for holdout season {test_season}."
        )

    if set(train_data["season"]) & set(test_data["season"]):
        raise ValueError("Training and test seasons overlap.")

    logger.info(
        "Training: %s rows from seasons %s",
        len(train_data),
        train_seasons,
    )

    logger.info(
        "Holdout: %s rows from season %s",
        len(test_data),
        test_season,
    )

    return train_data, test_data


def _calculate_stint_medians(
    training_data: pd.DataFrame,
) -> tuple[pd.Series, float]:
    """Calculate median completed stint lengths using training data only.

    A stint is considered completed when its driver has an in-lap.
    The maximum tyre age in that stint is used as its length.
    """

    required = set(STINT_GROUP_COLUMNS + ["tyre_age", "is_inlap"])

    missing = required - set(training_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns for stint calculation: {sorted(missing)}"
        )

    data = training_data.copy()

    data["is_inlap"] = (
        data["is_inlap"].fillna(False).astype(bool)
    )

    data = data.dropna(
        subset=["circuit", "compound", "stint", "tyre_age"]
    )

    stints = (
        data.groupby(
            STINT_GROUP_COLUMNS,
            dropna=True,
        )
        .agg(
            stint_length=("tyre_age", "max"),
            has_pit_stop=("is_inlap", "max"),
        )
        .reset_index()
    )

    completed_stints = stints[
        stints["has_pit_stop"]
    ].copy()

    if completed_stints.empty:
        raise ValueError(
            "No completed stints found in the training data."
        )

    medians = completed_stints.groupby(
        ["circuit", "compound"]
    )["stint_length"].median()

    global_median = float(
        completed_stints["stint_length"].median()
    )

    return medians, global_median


def predict_baseline(
    training_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    """Predict laps until the next pit using historical median stint length.

    Circuit and compound medians are learned exclusively from
    the supplied training data.
    """

    required = {
        "circuit",
        "compound",
        "tyre_age",
    }

    for name, frame in [
        ("training_data", training_data),
        ("prediction_data", prediction_data),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    medians, global_median = _calculate_stint_medians(
        training_data
    )

    prediction = prediction_data.copy()

    lookup_index = pd.MultiIndex.from_frame(
        prediction[["circuit", "compound"]]
    )

    predicted_stint_length = medians.reindex(
        lookup_index
    ).to_numpy(dtype=float)

    predicted_stint_length = np.where(
        np.isnan(predicted_stint_length),
        global_median,
        predicted_stint_length,
    )

    prediction["baseline_predicted_laps"] = np.maximum(
        predicted_stint_length
        - prediction["tyre_age"].to_numpy(dtype=float),
        0,
    )

    return prediction


def _race_group_ids(
    data: pd.DataFrame,
) -> pd.Series:
    """Create unique race identifiers for grouped cross-validation."""

    return (
        data[RACE_COLUMNS]
        .astype(str)
        .agg("_".join, axis=1)
    )


def evaluate_baseline_cv(
    training_data: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate the baseline with GroupKFold, grouping by race.

    Every fold learns its stint medians from its training races only.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    data = training_data.reset_index(drop=True).copy()

    groups = _race_group_ids(data)

    unique_races = groups.nunique()

    if unique_races < n_splits:
        raise ValueError(
            f"Need at least {n_splits} races, found {unique_races}."
        )

    splitter = GroupKFold(n_splits=n_splits)

    results: list[dict[str, float | int]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(data, groups=groups),
        start=1,
    ):
        fold_train = data.iloc[train_indices].copy()

        fold_validation = data.iloc[validation_indices].copy()

        train_races = set(
            _race_group_ids(fold_train)
        )

        validation_races = set(
            _race_group_ids(fold_validation)
        )

        if train_races & validation_races:
            raise RuntimeError(
                "Race leakage detected between CV training "
                "and validation folds."
            )

        predictions = predict_baseline(
            fold_train,
            fold_validation,
        )

        valid = predictions.dropna(
            subset=["laps_to_next_pit"]
        )

        if valid.empty:
            raise ValueError(
                f"Fold {fold} has no valid pit-stop targets."
            )

        mae = mean_absolute_error(
            valid["laps_to_next_pit"],
            valid["baseline_predicted_laps"],
        )

        results.append(
            {
                "fold": fold,
                "mae": float(mae),
                "validation_races": len(validation_races),
                "validation_samples": len(valid),
            }
        )

        logger.info(
            "Fold %s: MAE %.3f across %s races",
            fold,
            mae,
            len(validation_races),
        )

    return pd.DataFrame(results)


def evaluate_baseline_holdout(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Evaluate the baseline on the completely separate test season."""

    predictions = predict_baseline(
        training_data,
        test_data,
    )

    valid = predictions.dropna(
        subset=["laps_to_next_pit"]
    ).copy()

    if valid.empty:
        raise ValueError(
            "The holdout dataset has no valid pit-stop targets."
        )

    mae = float(
        mean_absolute_error(
            valid["laps_to_next_pit"],
            valid["baseline_predicted_laps"],
        )
    )

    logger.info(
        "2025 holdout MAE: %.3f across %s samples",
        mae,
        len(valid),
    )

    return mae, valid


def run_a4(
    laps: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> dict[str, object]:
    """Run the complete A4 baseline modelling workflow."""

    features = prepare_model_data(
        laps,
        pit_loss,
    )

    training_data, test_data = split_by_season(
        features
    )

    cv_results = evaluate_baseline_cv(
        training_data,
        n_splits=5,
    )

    holdout_mae, holdout_predictions = (
        evaluate_baseline_holdout(
            training_data,
            test_data,
        )
    )

    return {
        "features": features,
        "training_data": training_data,
        "test_data": test_data,
        "cv_results": cv_results,
        "cv_mean_mae": float(cv_results["mae"].mean()),
        "cv_std_mae": float(cv_results["mae"].std()),
        "holdout_mae": holdout_mae,
        "holdout_predictions": holdout_predictions,
    }


def main() -> None:
    """Run A4 using the synthetic development datasets."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    laps = pd.read_parquet(
        "data/processed/synthetic_laps_clean.parquet"
    )

    pit_loss = pd.read_parquet(
        "data/processed/synthetic_pit_loss.parquet"
    )

    results = run_a4(
        laps,
        pit_loss,
    )

    print("\nA4 DATASET SUMMARY")
    print("------------------")

    print(
        "Training rows:",
        len(results["training_data"]),
    )

    print(
        "2025 holdout rows:",
        len(results["test_data"]),
    )

    print("\nGROUPED CROSS-VALIDATION RESULTS")
    print("--------------------------------")

    print(results["cv_results"].to_string(index=False))

    print(
        f"\nMean CV MAE: {results['cv_mean_mae']:.3f} laps"
    )

    print(
        f"CV MAE standard deviation: "
        f"{results['cv_std_mae']:.3f} laps"
    )

    print("\n2025 HOLDOUT RESULT")
    print("-------------------")

    print(
        f"2025 Holdout MAE: {results['holdout_mae']:.3f} laps"
    )


if __name__ == "__main__":
    main()
