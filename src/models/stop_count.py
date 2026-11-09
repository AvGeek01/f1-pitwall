"""Build race-level pit-stop count data and historical features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RACE_COLUMNS = ["season", "round", "race"]
DRIVER_RACE_COLUMNS = RACE_COLUMNS + ["driver"]

REQUIRED_COLUMNS = [
    "season",
    "round",
    "race",
    "circuit",
    "driver",
    "is_inlap",
]

CATEGORICAL_FEATURES = [
    "circuit",
    "team",
]

NUMERIC_FEATURES = [
    "grid_position",
    "historical_circuit_median_stops",
    "historical_global_median_stops",
]

TARGET_COLUMN = "stop_count_class"

DEFAULT_COLD_START_STOPS = 2.0


def prepare_race_level_data(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """Convert driver-lap records into one row per driver per race.

    Pit stops are counted using is_inlap. Out-laps are not counted,
    because they represent the lap after a pit stop.

    Returns:
        One row per driver-race with the actual number of pit stops.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(laps.columns))

    if missing:
        raise ValueError(
            f"Lap data is missing required columns: {missing}"
        )

    if laps.empty:
        raise ValueError("Lap dataset is empty.")

    data = laps.copy()

    data["season"] = pd.to_numeric(
        data["season"], errors="coerce"
    )

    data["round"] = pd.to_numeric(
        data["round"], errors="coerce"
    )

    data["grid_position"] = pd.to_numeric(
        data.get(
            "grid_position",
            pd.Series(np.nan, index=data.index),
        ),
        errors="coerce",
    )

    data = data.dropna(
        subset=DRIVER_RACE_COLUMNS
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid driver-race records remain after cleaning."
        )

    data["season"] = data["season"].astype(int)
    data["is_inlap"] = (
        data["is_inlap"].fillna(False).astype(bool)
    )

    if "team" not in data.columns:
        data["team"] = "UNKNOWN"

    data["team"] = (
        data["team"].fillna("UNKNOWN").astype(str)
    )

    data["circuit"] = (
        data["circuit"].fillna("UNKNOWN").astype(str)
    )

    race_data = (
        data.groupby(
            DRIVER_RACE_COLUMNS,
            as_index=False,
            dropna=False,
        )
        .agg(
            circuit=("circuit", "first"),
            team=("team", "first"),
            grid_position=("grid_position", "first"),
            pit_stops=("is_inlap", "sum"),
        )
    )

    race_data["pit_stops"] = (
        race_data["pit_stops"].astype(int)
    )

    zero_stop_count = int(
        race_data["pit_stops"].eq(0).sum()
    )

    if zero_stop_count:
        logger.warning(
            "Excluding %d driver-race records with zero pit stops.",
            zero_stop_count,
        )

        race_data = race_data.loc[
            race_data["pit_stops"] > 0
        ].copy()

    if race_data.empty:
        raise ValueError(
            "No driver-race records with at least one pit stop."
        )

    race_data[TARGET_COLUMN] = np.select(
        [
            race_data["pit_stops"].eq(1),
            race_data["pit_stops"].eq(2),
            race_data["pit_stops"].ge(3),
        ],
        [1, 2, 3],
        default=0,
    ).astype(int)

    return race_data.sort_values(
        DRIVER_RACE_COLUMNS
    ).reset_index(drop=True)


def add_historical_features(
    race_data: pd.DataFrame,
    cold_start_stops: float = DEFAULT_COLD_START_STOPS,
) -> pd.DataFrame:
    """Add circuit and global pit-stop statistics from earlier seasons.

    For a particular season, only seasons strictly earlier than that
    season are used. Results from the current or future season are
    never included.

    When no earlier season exists, cold_start_stops is used as the
    documented fallback.
    """
    required = {
        "season",
        "circuit",
        "pit_stops",
    }

    missing = sorted(required - set(race_data.columns))

    if missing:
        raise ValueError(
            f"Race-level data is missing columns: {missing}"
        )

    if not np.isfinite(cold_start_stops) or cold_start_stops < 0:
        raise ValueError(
            "cold_start_stops must be finite and non-negative."
        )

    result = race_data.copy()

    result["historical_circuit_median_stops"] = np.nan
    result["historical_global_median_stops"] = np.nan

    seasons = sorted(result["season"].dropna().unique())

    for season in seasons:
        current_mask = result["season"].eq(season)

        history = result.loc[
            result["season"] < season
        ]

        if history.empty:
            global_median = float(cold_start_stops)

            circuit_medians = pd.Series(dtype=float)

        else:
            global_median = float(
                history["pit_stops"].median()
            )

            circuit_medians = (
                history.groupby("circuit")["pit_stops"]
                .median()
            )

        current_circuits = result.loc[
            current_mask, "circuit"
        ]

        circuit_values = current_circuits.map(
            circuit_medians
        )

        circuit_values = circuit_values.fillna(
            global_median
        )

        result.loc[
            current_mask,
            "historical_circuit_median_stops",
        ] = circuit_values.to_numpy()

        result.loc[
            current_mask,
            "historical_global_median_stops",
        ] = global_median

    return result


def build_stop_count_dataset(
    laps: pd.DataFrame,
    cold_start_stops: float = DEFAULT_COLD_START_STOPS,
) -> pd.DataFrame:
    """Build the complete race-level dataset for A6.

    The returned dataset contains one row per driver per race,
    actual pit-stop counts, classification targets, and historical
    features calculated without using future-season information.
    """
    race_data = prepare_race_level_data(laps)

    dataset = add_historical_features(
        race_data,
        cold_start_stops=cold_start_stops,
    )

    logger.info(
        "Built %d driver-race records across %d races.",
        len(dataset),
        dataset.groupby(
            RACE_COLUMNS,
            dropna=False,
        ).ngroups,
    )

    return dataset


def add_historical_features_from_reference(
    data: pd.DataFrame,
    history_data: pd.DataFrame,
    cold_start_stops: float = DEFAULT_COLD_START_STOPS,
) -> pd.DataFrame:
    """Calculate historical features using a separate reference dataset.

    Only reference records from strictly earlier seasons are used.
    This allows cross-validation to avoid historical-feature leakage.
    """
    result = data.copy()

    result["historical_circuit_median_stops"] = np.nan
    result["historical_global_median_stops"] = np.nan

    for season in sorted(result["season"].unique()):
        mask = result["season"].eq(season)

        history = history_data.loc[
            history_data["season"] < season
        ]

        if history.empty:
            global_median = cold_start_stops
            circuit_medians = pd.Series(dtype=float)
        else:
            global_median = float(
                history["pit_stops"].median()
            )
            circuit_medians = (
                history.groupby("circuit")["pit_stops"].median()
            )

        circuit_values = (
            result.loc[mask, "circuit"]
            .map(circuit_medians)
            .fillna(global_median)
        )

        result.loc[
            mask, "historical_circuit_median_stops"
        ] = circuit_values.to_numpy()

        result.loc[
            mask, "historical_global_median_stops"
        ] = global_median

    return result


def split_stop_count_by_season(
    data: pd.DataFrame,
    train_seasons: tuple[int, ...] = (2023, 2024),
    test_season: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split A6 data into historical training and future holdout seasons."""
    training = data.loc[
        data["season"].isin(train_seasons)
    ].copy()

    holdout = data.loc[
        data["season"].eq(test_season)
    ].copy()

    if training.empty or holdout.empty:
        raise ValueError("Training or holdout dataset is empty.")

    if set(training["season"]) & set(holdout["season"]):
        raise ValueError("Training and holdout seasons overlap.")

    return training, holdout


def get_race_groups(data: pd.DataFrame) -> pd.Series:
    """Create unique race identifiers for grouped validation."""
    return (
        data["season"].astype(str)
        + "_"
        + data["round"].astype(str)
        + "_"
        + data["race"].astype(str)
    )


def build_stop_count_pipeline(
    model_type: str,
) -> Pipeline:
    """Create a preprocessing and classification pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(
            "model_type must be 'random_forest' or 'xgboost'."
        )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def prepare_model_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Select the approved A6 predictor columns."""
    missing = sorted(
        set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing A6 predictor columns: {missing}"
        )

    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def fit_stop_count_model(
    model_type: str,
    training_data: pd.DataFrame,
) -> Pipeline:
    """Fit a stop-count classifier using the three target classes."""
    features = prepare_model_features(training_data)

    labels = training_data[TARGET_COLUMN].astype(int)

    if set(labels.unique()) != {1, 2, 3}:
        raise ValueError(
            "Training data must contain all three classes: 1, 2 and 3+."
        )

    encoded_labels = labels - 1

    model = build_stop_count_pipeline(model_type)
    model.fit(features, encoded_labels)

    return model


def predict_stop_count(
    model: Pipeline,
    data: pd.DataFrame,
) -> np.ndarray:
    """Predict the original stop-count classes 1, 2 and 3+."""
    features = prepare_model_features(data)

    encoded_predictions = model.predict(features)

    return np.asarray(encoded_predictions, dtype=int) + 1


def circuit_majority_baseline(
    training_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> np.ndarray:
    """Predict the most common training class for each circuit."""
    circuit_modes = (
        training_data.groupby("circuit")[TARGET_COLUMN]
        .agg(lambda values: values.mode().iloc[0])
    )

    global_mode = int(
        training_data[TARGET_COLUMN].mode().iloc[0]
    )

    return (
        prediction_data["circuit"]
        .map(circuit_modes)
        .fillna(global_mode)
        .to_numpy(dtype=int)
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate accuracy and macro-F1."""
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=[1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_grouped_cv(
    training_data: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate both models using stratified race-grouped CV.

    Historical features for each fold are recomputed using only
    that fold's training races.
    """
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    groups = get_race_groups(training_data)
    labels = training_data[TARGET_COLUMN].to_numpy()

    results = []

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(training_data, labels, groups),
        start=1,
    ):
        fold_train_raw = training_data.iloc[
            train_indices
        ].copy()

        fold_val_raw = training_data.iloc[
            val_indices
        ].copy()

        fold_train = add_historical_features_from_reference(
            fold_train_raw,
            fold_train_raw,
        )

        fold_val = add_historical_features_from_reference(
            fold_val_raw,
            fold_train_raw,
        )

        train_groups = set(
            get_race_groups(fold_train_raw)
        )
        val_groups = set(
            get_race_groups(fold_val_raw)
        )

        if train_groups & val_groups:
            raise RuntimeError(
                "Race leakage detected in cross-validation."
            )

        for model_type in [
            "random_forest",
            "xgboost",
        ]:
            model = fit_stop_count_model(
                model_type,
                fold_train,
            )

            predictions = predict_stop_count(
                model,
                fold_val,
            )

            metrics = calculate_metrics(
                fold_val[TARGET_COLUMN],
                predictions,
            )

            results.append(
                {
                    "fold": fold,
                    "model": model_type,
                    **metrics,
                }
            )

        baseline_predictions = circuit_majority_baseline(
            fold_train,
            fold_val,
        )

        baseline_metrics = calculate_metrics(
            fold_val[TARGET_COLUMN],
            baseline_predictions,
        )

        results.append(
            {
                "fold": fold,
                "model": "circuit_majority",
                **baseline_metrics,
            }
        )

    return pd.DataFrame(results)


def bootstrap_macro_f1_ci(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Calculate a 95% bootstrap confidence interval for macro-F1."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    rng = np.random.default_rng(random_state)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(actual_array),
            size=len(actual_array),
        )

        score = f1_score(
            actual_array[indices],
            predicted_array[indices],
            labels=[1, 2, 3],
            average="macro",
            zero_division=0,
        )

        scores.append(score)

    lower, upper = np.percentile(
        scores,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def run_a6(
    laps: pd.DataFrame,
    output_dir: str | Path = ".",
) -> dict:
    """Run the complete A6 stop-count modelling workflow."""
    output_dir = Path(output_dir)

    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    figures_dir = reports_dir / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    race_data = build_stop_count_dataset(laps)

    training_raw, holdout_raw = split_stop_count_by_season(
        race_data
    )

    if set(training_raw[TARGET_COLUMN].unique()) != {1, 2, 3}:
        raise ValueError(
            "Training seasons must contain all three stop-count classes."
        )

    if set(holdout_raw[TARGET_COLUMN].unique()) != {1, 2, 3}:
        raise ValueError(
            "Holdout season must contain all three stop-count classes."
        )

    logger.info(
        "A6 training: %d records; holdout: %d records.",
        len(training_raw),
        len(holdout_raw),
    )

    cv_results = evaluate_grouped_cv(training_raw)

    training_data = add_historical_features_from_reference(
        training_raw,
        training_raw,
    )

    holdout_data = add_historical_features_from_reference(
        holdout_raw,
        training_raw,
    )

    actual = holdout_data[TARGET_COLUMN].to_numpy()

    final_models = {}
    predictions_by_model = {}
    metric_rows = []

    for model_type in [
        "random_forest",
        "xgboost",
    ]:
        model = fit_stop_count_model(
            model_type,
            training_data,
        )

        predictions = predict_stop_count(
            model,
            holdout_data,
        )

        metrics = calculate_metrics(
            actual,
            predictions,
        )

        ci_lower, ci_upper = bootstrap_macro_f1_ci(
            actual,
            predictions,
        )

        metrics["macro_f1_ci_lower"] = ci_lower
        metrics["macro_f1_ci_upper"] = ci_upper

        final_models[model_type] = model
        predictions_by_model[model_type] = predictions

        metric_rows.append(
            {
                "model": model_type,
                **metrics,
            }
        )

        joblib.dump(
            model,
            models_dir / f"stop_count_{model_type}.joblib",
        )

    baseline_predictions = circuit_majority_baseline(
        training_data,
        holdout_data,
    )

    baseline_metrics = calculate_metrics(
        actual,
        baseline_predictions,
    )

    baseline_lower, baseline_upper = bootstrap_macro_f1_ci(
        actual,
        baseline_predictions,
    )

    metric_rows.append(
        {
            "model": "circuit_majority",
            **baseline_metrics,
            "macro_f1_ci_lower": baseline_lower,
            "macro_f1_ci_upper": baseline_upper,
        }
    )

    predictions_by_model["circuit_majority"] = (
        baseline_predictions
    )

    metrics_table = pd.DataFrame(metric_rows)

    cv_results.to_csv(
        reports_dir / "a6_cross_validation.csv",
        index=False,
    )

    metrics_table.to_csv(
        reports_dir / "a6_holdout_metrics.csv",
        index=False,
    )

    for model_name, predictions in predictions_by_model.items():
        matrix = confusion_matrix(
            actual,
            predictions,
            labels=[1, 2, 3],
        )

        figure, axis = plt.subplots(figsize=(6, 5))

        display = axis.imshow(matrix)

        axis.set(
            xticks=[0, 1, 2],
            yticks=[0, 1, 2],
            xticklabels=["1 stop", "2 stops", "3+ stops"],
            yticklabels=["1 stop", "2 stops", "3+ stops"],
            xlabel="Predicted class",
            ylabel="Actual class",
            title=f"A6 Confusion Matrix: {model_name}",
        )

        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                )

        figure.colorbar(display, ax=axis)
        figure.tight_layout()

        figure.savefig(
            figures_dir / f"a6_confusion_{model_name}.png",
            dpi=150,
        )

        plt.close(figure)

    logger.info("A6 modelling completed.")

    return {
        "dataset": race_data,
        "training_data": training_data,
        "holdout_data": holdout_data,
        "cv_results": cv_results,
        "holdout_metrics": metrics_table,
        "models": final_models,
        "predictions": predictions_by_model,
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    data_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "synthetic_stop_count_laps.parquet"
    )

    laps = pd.read_parquet(data_path)

    results = run_a6(
        laps,
        output_dir=Path(__file__).resolve().parents[2],
    )

    print("\nA6 CROSS-VALIDATION RESULTS")
    print(
        results["cv_results"]
        .groupby("model")[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\nA6 2025 HOLDOUT RESULTS")
    print(
        results["holdout_metrics"].round(4).to_string(index=False)
    )