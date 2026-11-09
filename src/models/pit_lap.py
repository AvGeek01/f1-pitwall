"""A4/A5: Race-grouped baseline and XGBoost pit-lap prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier, XGBRegressor

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

NUMERIC_FEATURES = [
    "lap_number",
    "lap_time_s",
    "fuel_corrected_time_s",
    "position",
    "grid_position",
    "is_sc",
    "is_wet",
    "air_temp",
    "track_temp",
    "rainfall",
    "tyre_age",
    "stint_number",
    "compounds_used",
    "two_compound_rule_satisfied",
    "rolling_3_lap_pace",
    "degradation_slope",
    "cumulative_race_time",
    "gap_ahead",
    "gap_behind",
    "gap_leader",
    "undercut_threat",
]

CATEGORICAL_FEATURES = [
    "circuit",
    "team",
    "compound",
]

A5_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
A5_TARGET_REGRESSION = "laps_to_next_pit"
A5_TARGET_CLASSIFICATION = "pits_within_3"

MODEL_DIRECTORY = Path("models")
FIGURE_DIRECTORY = Path("reports/figures")
METRICS_PATH = Path("reports/a5_metrics.csv")

PARAMETER_GRID = {
    "n_estimators": [150, 250],
    "max_depth": [2, 3],
    "learning_rate": [0.05],
}


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
    data["is_inlap"] = data["is_inlap"].fillna(False).astype(bool)
    data = data.dropna(
        subset=["circuit", "compound", "stint", "tyre_age"]
    )

    stints = (
        data.groupby(STINT_GROUP_COLUMNS, dropna=True)
        .agg(
            stint_length=("tyre_age", "max"),
            has_pit_stop=("is_inlap", "max"),
        )
        .reset_index()
    )

    completed_stints = stints[stints["has_pit_stop"]].copy()

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

    required = {"circuit", "compound", "tyre_age"}

    for name, frame in [
        ("training_data", training_data),
        ("prediction_data", prediction_data),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    medians, global_median = _calculate_stint_medians(training_data)
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


def _race_group_ids(data: pd.DataFrame) -> pd.Series:
    """Create unique race identifiers for grouped cross-validation."""

    return data[RACE_COLUMNS].astype(str).agg("_".join, axis=1)


def evaluate_baseline_cv(
    training_data: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate the baseline with GroupKFold, grouping by race."""

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

        train_races = set(_race_group_ids(fold_train))
        validation_races = set(_race_group_ids(fold_validation))

        if train_races & validation_races:
            raise RuntimeError(
                "Race leakage detected between CV training "
                "and validation folds."
            )

        predictions = predict_baseline(fold_train, fold_validation)
        valid = predictions.dropna(subset=["laps_to_next_pit"])

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
            "Baseline fold %s: MAE %.3f across %s races",
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

    predictions = predict_baseline(training_data, test_data)
    valid = predictions.dropna(subset=["laps_to_next_pit"]).copy()

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
        "2025 baseline holdout MAE: %.3f across %s samples",
        mae,
        len(valid),
    )

    return mae, valid


def run_a4(
    laps: pd.DataFrame,
    pit_loss: pd.DataFrame,
) -> dict[str, Any]:
    """Run the complete A4 baseline modelling workflow."""

    features = prepare_model_data(laps, pit_loss)
    training_data, test_data = split_by_season(features)

    cv_results = evaluate_baseline_cv(training_data, n_splits=5)
    holdout_mae, holdout_predictions = evaluate_baseline_holdout(
        training_data,
        test_data,
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


def validate_a5_features(data: pd.DataFrame) -> None:
    """Raise a clear error if an A5 predictor or target is unavailable."""

    required = set(A5_FEATURES) | {
        A5_TARGET_REGRESSION,
        A5_TARGET_CLASSIFICATION,
    } | set(RACE_COLUMNS)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns required for A5: {sorted(missing)}"
        )


def build_preprocessor() -> ColumnTransformer:
    """Create fold-fitted numeric and categorical preprocessing."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_a5_pipeline(
    task: str,
    parameters: dict[str, Any],
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """Build an XGBoost pipeline with preprocessing fitted per fold."""

    if task == "regression":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **parameters,
        )
    elif task == "classification":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **parameters,
        )
    else:
        raise ValueError(
            "task must be either 'regression' or 'classification'."
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def calculate_scale_pos_weight(target: pd.Series) -> float:
    """Calculate negative-to-positive class ratio from training labels."""

    labels = pd.Series(target).dropna().astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if positives == 0:
        raise ValueError(
            "Classification training fold contains no positive samples."
        )

    return float(negatives / positives)


def _valid_training_rows(
    data: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Return rows with a known target and all required predictors."""

    valid = data.dropna(subset=[target]).copy()

    if valid.empty:
        raise ValueError(f"No valid training rows for target '{target}'.")

    return valid


def _check_group_cv(
    data: pd.DataFrame,
    n_splits: int,
) -> tuple[GroupKFold, pd.Series]:
    """Create race-grouped folds after validating the race count."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    groups = _race_group_ids(data)
    race_count = groups.nunique()

    if race_count < n_splits:
        raise ValueError(
            f"Need at least {n_splits} races for CV; found {race_count}."
        )

    return GroupKFold(n_splits=n_splits), groups


def tune_regressor(
    training_data: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select regression parameters using race-grouped CV MAE."""

    data = _valid_training_rows(
        training_data,
        A5_TARGET_REGRESSION,
    ).reset_index(drop=True)

    splitter, groups = _check_group_cv(data, n_splits)
    results: list[dict[str, Any]] = []

    for parameters in ParameterGrid(PARAMETER_GRID):
        fold_maes: list[float] = []

        for train_indices, validation_indices in splitter.split(
            data,
            groups=groups,
        ):
            fold_train = data.iloc[train_indices]
            fold_validation = data.iloc[validation_indices]

            model = build_a5_pipeline("regression", parameters)
            model.fit(
                fold_train[A5_FEATURES],
                fold_train[A5_TARGET_REGRESSION],
            )

            predictions = model.predict(
                fold_validation[A5_FEATURES]
            )

            fold_maes.append(
                float(
                    mean_absolute_error(
                        fold_validation[A5_TARGET_REGRESSION],
                        predictions,
                    )
                )
            )

        results.append(
            {
                **parameters,
                "mean_cv_mae": float(np.mean(fold_maes)),
                "std_cv_mae": float(np.std(fold_maes)),
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        "mean_cv_mae"
    ).reset_index(drop=True)

    best_parameters = {
        key: results_frame.loc[0, key]
        for key in PARAMETER_GRID
    }
    best_parameters["n_estimators"] = int(
        best_parameters["n_estimators"]
    )
    best_parameters["max_depth"] = int(
        best_parameters["max_depth"]
    )
    best_parameters["learning_rate"] = float(
        best_parameters["learning_rate"]
    )

    return best_parameters, results_frame


def tune_classifier(
    training_data: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select classifier parameters using race-grouped CV average precision."""

    data = _valid_training_rows(
        training_data,
        A5_TARGET_CLASSIFICATION,
    ).reset_index(drop=True)

    data[A5_TARGET_CLASSIFICATION] = (
        data[A5_TARGET_CLASSIFICATION].astype(int)
    )

    splitter, groups = _check_group_cv(data, n_splits)
    results: list[dict[str, Any]] = []

    for parameters in ParameterGrid(PARAMETER_GRID):
        fold_scores: list[float] = []

        for train_indices, validation_indices in splitter.split(
            data,
            groups=groups,
        ):
            fold_train = data.iloc[train_indices]
            fold_validation = data.iloc[validation_indices]

            class_weight = calculate_scale_pos_weight(
                fold_train[A5_TARGET_CLASSIFICATION]
            )

            model = build_a5_pipeline(
                "classification",
                parameters,
                scale_pos_weight=class_weight,
            )
            model.fit(
                fold_train[A5_FEATURES],
                fold_train[A5_TARGET_CLASSIFICATION],
            )

            probabilities = model.predict_proba(
                fold_validation[A5_FEATURES]
            )[:, 1]

            fold_scores.append(
                float(
                    average_precision_score(
                        fold_validation[A5_TARGET_CLASSIFICATION],
                        probabilities,
                    )
                )
            )

        results.append(
            {
                **parameters,
                "mean_cv_pr_auc": float(np.mean(fold_scores)),
                "std_cv_pr_auc": float(np.std(fold_scores)),
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        "mean_cv_pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    best_parameters = {
        key: results_frame.loc[0, key]
        for key in PARAMETER_GRID
    }
    best_parameters["n_estimators"] = int(
        best_parameters["n_estimators"]
    )
    best_parameters["max_depth"] = int(
        best_parameters["max_depth"]
    )
    best_parameters["learning_rate"] = float(
        best_parameters["learning_rate"]
    )

    return best_parameters, results_frame


def generate_oof_probabilities(
    training_data: pd.DataFrame,
    parameters: dict[str, Any],
    n_splits: int = 5,
) -> pd.Series:
    """Generate out-of-fold probabilities for training-only threshold tuning."""

    data = _valid_training_rows(
        training_data,
        A5_TARGET_CLASSIFICATION,
    ).reset_index(drop=True)

    data[A5_TARGET_CLASSIFICATION] = (
        data[A5_TARGET_CLASSIFICATION].astype(int)
    )

    splitter, groups = _check_group_cv(data, n_splits)
    oof_probabilities = np.full(len(data), np.nan, dtype=float)

    for train_indices, validation_indices in splitter.split(
        data,
        groups=groups,
    ):
        fold_train = data.iloc[train_indices]
        fold_validation = data.iloc[validation_indices]

        class_weight = calculate_scale_pos_weight(
            fold_train[A5_TARGET_CLASSIFICATION]
        )

        model = build_a5_pipeline(
            "classification",
            parameters,
            scale_pos_weight=class_weight,
        )
        model.fit(
            fold_train[A5_FEATURES],
            fold_train[A5_TARGET_CLASSIFICATION],
        )

        oof_probabilities[validation_indices] = model.predict_proba(
            fold_validation[A5_FEATURES]
        )[:, 1]

    return pd.Series(oof_probabilities, index=data.index)


def select_f1_threshold(
    actual: pd.Series,
    probabilities: pd.Series,
) -> tuple[float, float]:
    """Choose the probability threshold with the best training OOF F1."""

    y_true = pd.Series(actual).astype(int).to_numpy()
    y_prob = pd.Series(probabilities).to_numpy(dtype=float)

    valid = np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Threshold selection requires both classes in OOF labels."
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob,
    )

    if len(thresholds) == 0:
        return 0.5, float(f1_score(y_true, y_prob >= 0.5))

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.nanargmax(f1_values))
    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
    )


def _save_calibration_plot(
    actual: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save a reliability diagram for the 2025 holdout classifier."""

    y_true = pd.Series(actual).astype(int)

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="XGBoost classifier",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Pit within 3 laps: 2025 holdout calibration",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_shap_summary(
    pipeline: Pipeline,
    data: pd.DataFrame,
    output_path: Path,
    sample_size: int = 500,
) -> None:
    """Save a SHAP summary plot using a sample of holdout predictors."""

    sample = data[A5_FEATURES].sample(
        n=min(sample_size, len(data)),
        random_state=42,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def run_a5(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train, tune, evaluate and save the A5 XGBoost models.

    The 2025 holdout is used only for final evaluation and plots.
    Hyperparameters and the classifier threshold are selected using
    race-grouped cross-validation on 2023-2024 training data.
    """

    validate_a5_features(training_data)
    validate_a5_features(test_data)

    if set(training_data["season"].unique()) & set(
        test_data["season"].unique()
    ):
        raise ValueError("A5 training and holdout seasons overlap.")

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_regression_parameters, regression_grid_results = tune_regressor(
        training_data,
        n_splits=n_splits,
    )

    regression_train = _valid_training_rows(
        training_data,
        A5_TARGET_REGRESSION,
    )
    regression_test = _valid_training_rows(
        test_data,
        A5_TARGET_REGRESSION,
    )

    regression_model = build_a5_pipeline(
        "regression",
        best_regression_parameters,
    )
    regression_model.fit(
        regression_train[A5_FEATURES],
        regression_train[A5_TARGET_REGRESSION],
    )

    regression_predictions = regression_model.predict(
        regression_test[A5_FEATURES]
    )
    regression_mae = float(
        mean_absolute_error(
            regression_test[A5_TARGET_REGRESSION],
            regression_predictions,
        )
    )

    best_classifier_parameters, classifier_grid_results = tune_classifier(
        training_data,
        n_splits=n_splits,
    )

    oof_probabilities = generate_oof_probabilities(
        training_data,
        best_classifier_parameters,
        n_splits=n_splits,
    )
    classification_train = _valid_training_rows(
        training_data,
        A5_TARGET_CLASSIFICATION,
    )
    threshold, oof_f1 = select_f1_threshold(
        classification_train[A5_TARGET_CLASSIFICATION].reset_index(
            drop=True
        ),
        oof_probabilities,
    )

    classification_train = classification_train.copy()
    classification_train[A5_TARGET_CLASSIFICATION] = (
        classification_train[A5_TARGET_CLASSIFICATION].astype(int)
    )

    class_weight = calculate_scale_pos_weight(
        classification_train[A5_TARGET_CLASSIFICATION]
    )
    classifier_model = build_a5_pipeline(
        "classification",
        best_classifier_parameters,
        scale_pos_weight=class_weight,
    )
    classifier_model.fit(
        classification_train[A5_FEATURES],
        classification_train[A5_TARGET_CLASSIFICATION],
    )

    classification_test = _valid_training_rows(
        test_data,
        A5_TARGET_CLASSIFICATION,
    ).copy()
    classification_test[A5_TARGET_CLASSIFICATION] = (
        classification_test[A5_TARGET_CLASSIFICATION].astype(int)
    )

    holdout_probabilities = classifier_model.predict_proba(
        classification_test[A5_FEATURES]
    )[:, 1]
    holdout_predictions = (
        holdout_probabilities >= threshold
    ).astype(int)

    holdout_pr_auc = float(
        average_precision_score(
            classification_test[A5_TARGET_CLASSIFICATION],
            holdout_probabilities,
        )
    )
    holdout_f1 = float(
        f1_score(
            classification_test[A5_TARGET_CLASSIFICATION],
            holdout_predictions,
            zero_division=0,
        )
    )

    baseline_predictions = predict_baseline(
        training_data,
        regression_test,
    )
    baseline_mae = float(
        mean_absolute_error(
            baseline_predictions[A5_TARGET_REGRESSION],
            baseline_predictions["baseline_predicted_laps"],
        )
    )

    regression_model_path = MODEL_DIRECTORY / "pit_lap_regressor.joblib"
    classifier_model_path = MODEL_DIRECTORY / "pit_lap_classifier.joblib"

    joblib.dump(regression_model, regression_model_path)
    joblib.dump(classifier_model, classifier_model_path)

    calibration_path = (
        FIGURE_DIRECTORY / "pit_within_3_calibration.png"
    )
    _save_calibration_plot(
        classification_test[A5_TARGET_CLASSIFICATION],
        holdout_probabilities,
        calibration_path,
    )

    regression_shap_path = (
        FIGURE_DIRECTORY / "pit_lap_regressor_shap.png"
    )
    classifier_shap_path = (
        FIGURE_DIRECTORY / "pit_lap_classifier_shap.png"
    )
    _save_shap_summary(
        regression_model,
        regression_test,
        regression_shap_path,
    )
    _save_shap_summary(
        classifier_model,
        classification_test,
        classifier_shap_path,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "A4 baseline",
                "metric": "MAE",
                "value": baseline_mae,
            },
            {
                "model": "A5 XGBoost regressor",
                "metric": "MAE",
                "value": regression_mae,
            },
            {
                "model": "A5 XGBoost classifier",
                "metric": "PR-AUC",
                "value": holdout_pr_auc,
            },
            {
                "model": "A5 XGBoost classifier",
                "metric": "F1",
                "value": holdout_f1,
            },
            {
                "model": "A5 XGBoost classifier",
                "metric": "OOF training F1",
                "value": oof_f1,
            },
            {
                "model": "A5 XGBoost classifier",
                "metric": "Selected threshold",
                "value": threshold,
            },
        ]
    )
    metrics.to_csv(METRICS_PATH, index=False)
    regression_grid_results.to_csv(
        Path("reports/a5_regression_grid_search.csv"),
        index=False,
    )
    classifier_grid_results.to_csv(
        Path("reports/a5_classification_grid_search.csv"),
        index=False,
    )

    return {
        "regression_model": regression_model,
        "classifier_model": classifier_model,
        "regression_parameters": best_regression_parameters,
        "classifier_parameters": best_classifier_parameters,
        "regression_grid_results": regression_grid_results,
        "classifier_grid_results": classifier_grid_results,
        "regression_mae": regression_mae,
        "baseline_mae": baseline_mae,
        "classifier_pr_auc": holdout_pr_auc,
        "classifier_f1": holdout_f1,
        "classifier_threshold": threshold,
        "oof_f1": oof_f1,
        "metrics": metrics,
        "regression_model_path": regression_model_path,
        "classifier_model_path": classifier_model_path,
        "calibration_plot": calibration_path,
        "regression_shap_plot": regression_shap_path,
        "classifier_shap_plot": classifier_shap_path,
    }


def main() -> None:
    """Run A4 and A5 using the synthetic development datasets."""

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

    a4_results = run_a4(laps, pit_loss)

    print("\nA4 DATASET SUMMARY")
    print("------------------")
    print("Training rows:", len(a4_results["training_data"]))
    print("2025 holdout rows:", len(a4_results["test_data"]))
    print("\nGROUPED CROSS-VALIDATION RESULTS")
    print("--------------------------------")
    print(a4_results["cv_results"].to_string(index=False))
    print(
        f"\nMean CV MAE: {a4_results['cv_mean_mae']:.3f} laps"
    )
    print(
        f"CV MAE standard deviation: "
        f"{a4_results['cv_std_mae']:.3f} laps"
    )
    print(
        f"2025 A4 baseline MAE: "
        f"{a4_results['holdout_mae']:.3f} laps"
    )

    print("\nRUNNING A5 XGBOOST MODELS")
    print("-------------------------")
    a5_results = run_a5(
        a4_results["training_data"],
        a4_results["test_data"],
        n_splits=5,
    )

    print("\nA5 RESULTS")
    print("----------")
    print(
        f"Baseline regression MAE: "
        f"{a5_results['baseline_mae']:.3f} laps"
    )
    print(
        f"XGBoost regression MAE: "
        f"{a5_results['regression_mae']:.3f} laps"
    )
    print(
        f"Classifier PR-AUC: "
        f"{a5_results['classifier_pr_auc']:.4f}"
    )
    print(
        f"Classifier F1: "
        f"{a5_results['classifier_f1']:.4f}"
    )
    print(
        f"Selected classifier threshold: "
        f"{a5_results['classifier_threshold']:.4f}"
    )
    print("\nSaved model and diagnostic files:")
    print(a5_results["regression_model_path"])
    print(a5_results["classifier_model_path"])
    print(a5_results["calibration_plot"])
    print(a5_results["regression_shap_plot"])
    print(a5_results["classifier_shap_plot"])


if __name__ == "__main__":
    main()
