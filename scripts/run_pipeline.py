"""Run the OpenF1-to-simulator project pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.clean import (
    LAPS_CLEAN_COLUMNS,
    clean_laps,
    estimate_pit_loss,
    load_raw_data,
)
from src.data.schema import normalize_pit_loss_schema
from src.data.fetch import (
    CORE_ENDPOINTS,
    RAW_DIR as DEFAULT_RAW_DIR,
    SESSION_ENDPOINTS,
    OpenF1Error,
    SESSION_RESULT_ENDPOINT,
    assign_rounds,
    fetch_season,
    finished_sessions,
    tag_records,
    to_frame,
)
from src.features.build import build_features
from src.models.degradation import (
    DegradationModel,
    fit_degradation,
)
from src.sim.validation import validate_simulator

LOGGER = logging.getLogger("pitwall.pipeline")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASONS = (2023, 2024, 2025)
ENDPOINTS = (
    "sessions",
    "meetings",
    *SESSION_ENDPOINTS,
    SESSION_RESULT_ENDPOINT,
)
RACE_KEYS = ("season", "round", "race")
SMOKE_NUMERIC_FEATURES = (
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
)
SMOKE_CATEGORICAL_FEATURES = ("circuit", "team", "compound")
SMOKE_FEATURES = SMOKE_NUMERIC_FEATURES + SMOKE_CATEGORICAL_FEATURES
SMOKE_TARGET_REGRESSION = "laps_to_next_pit"
SMOKE_TARGET_CLASSIFICATION = "pits_within_3"
T = TypeVar("T")


def _run_step(name: str, action: Callable[[], T]) -> T:
    """Run a pipeline action with explicit progress and failure logging."""
    LOGGER.info("Starting step: %s", name)
    try:
        result = action()
    except Exception as error:
        LOGGER.exception("Pipeline step failed: %s", name)
        raise RuntimeError(f"Pipeline step {name!r} failed: {error}") from error
    LOGGER.info("Completed step: %s", name)
    return result


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    description: str,
) -> None:
    """Raise a clear error when a pipeline DataFrame lacks required fields."""
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{description} is missing required columns: {missing}")


def _read_json_cache(path: Path, endpoint: str) -> list[dict[str, object]]:
    """Read one cached OpenF1 endpoint response and validate its shape."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached OpenF1 {endpoint} file: {path}. "
            "Run without --skip-fetch to retrieve it."
        )
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read cached {endpoint} JSON at {path}: {error}") from error
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Cached {endpoint} file must contain a JSON list of objects: {path}")
    return records


def _write_endpoint_parquets(
    year: int,
    endpoint_records: dict[str, list[dict[str, object]]],
    output_directory: Path,
) -> dict[str, Path]:
    """Write cached endpoint records to the fetcher's standard parquet layout."""
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for endpoint, records in endpoint_records.items():
        if not records and endpoint in CORE_ENDPOINTS:
            raise ValueError(
                f"Cached OpenF1 {endpoint} data is empty for season {year}"
            )
        frame = to_frame(endpoint, records)
        path = output_directory / f"{endpoint}_{year}.parquet"
        frame.to_parquet(path, index=False)
        outputs[endpoint] = path
        LOGGER.info("Cached %s rows=%d at %s", endpoint, len(frame), path)
    return outputs


def _rebuild_cached_season(
    year: int,
    raw_directory: Path,
    parquet_directory: Path,
    now: datetime,
) -> dict[str, Path]:
    """Reconstruct per-season parquet files solely from OpenF1 JSON caches."""
    sessions = _read_json_cache(raw_directory / "sessions" / f"{year}.json", "sessions")
    meetings = _read_json_cache(raw_directory / "meetings" / f"{year}.json", "meetings")
    calendar = assign_rounds(sessions)
    finished = finished_sessions(calendar, now)
    if not finished:
        raise OpenF1Error(f"No finished cached race sessions found for season {year}")

    meetings_by_key = {
        meeting["meeting_key"]: meeting
        for meeting in meetings
        if "meeting_key" in meeting
    }
    round_by_meeting: dict[object, int] = {
        session.get("meeting_key"): int(session["round"])
        for session in calendar
    }
    records_by_endpoint: dict[str, list[dict[str, object]]] = {
        endpoint: [] for endpoint in ENDPOINTS
    }
    records_by_endpoint["sessions"] = [
        {
            **session,
            "season": year,
            "meeting_name": meetings_by_key.get(
                session.get("meeting_key"),
                {},
            ).get("meeting_name"),
            "meeting_official_name": meetings_by_key.get(
                session.get("meeting_key"),
                {},
            ).get("meeting_official_name"),
            "circuit_type": meetings_by_key.get(
                session.get("meeting_key"),
                {},
            ).get("circuit_type"),
        }
        for session in calendar
    ]
    missing_grid_meeting_rounds: dict[object, int] = {}
    for session in finished:
        round_number = int(session["round"])
        session_key = session.get("session_key")
        if session_key is None:
            raise ValueError(
                f"Cached season {year} session is missing session_key: {session}"
            )
        for endpoint in SESSION_ENDPOINTS:
            cache_path = raw_directory / endpoint / f"{session_key}.json"
            if not cache_path.is_file():
                if endpoint in CORE_ENDPOINTS:
                    raise FileNotFoundError(
                        f"Missing required cached OpenF1 {endpoint} response: "
                        f"{cache_path}. Run without --skip-fetch to fetch it."
                    )
                LOGGER.warning("Optional cached %s file missing: %s", endpoint, cache_path)
                continue
            records = _read_json_cache(cache_path, endpoint)
            records_by_endpoint[endpoint].extend(
                tag_records(records, year, round_number, int(session_key))
            )
        driver_numbers = {
            record.get("driver_number")
            for record in _read_json_cache(
                raw_directory / "drivers" / f"{session_key}.json",
                "drivers",
            )
            if record.get("driver_number") is not None
        }
        grid_records_path = raw_directory / "starting_grid" / f"{session_key}.json"
        grid_records = (
            _read_json_cache(grid_records_path, "starting_grid")
            if grid_records_path.is_file()
            else []
        )
        grid_numbers = {
            record.get("driver_number")
            for record in grid_records
            if record.get("driver_number") is not None
            and pd.notna(pd.to_numeric(record.get("position"), errors="coerce"))
        }
        meeting_key = session.get("meeting_key")
        if (
            meeting_key is not None
            and (not grid_numbers or not driver_numbers.issubset(grid_numbers))
        ):
            missing_grid_meeting_rounds[meeting_key] = round_number

    qualifying_sessions_path = (
        raw_directory / "sessions" / f"{year}_qualifying.json"
    )
    if qualifying_sessions_path.is_file() and missing_grid_meeting_rounds:
        qualifying_sessions = _read_json_cache(
            qualifying_sessions_path,
            "qualifying sessions",
        )
        for qualifying_session in qualifying_sessions:
            meeting_key = qualifying_session.get("meeting_key")
            session_key = qualifying_session.get("session_key")
            if (
                qualifying_session.get("session_name") != "Qualifying"
                or qualifying_session.get("is_cancelled")
                or meeting_key not in missing_grid_meeting_rounds
                or session_key is None
                or not finished_sessions([qualifying_session], now)
            ):
                continue
            result_path = (
                raw_directory
                / SESSION_RESULT_ENDPOINT
                / f"{session_key}.json"
            )
            if not result_path.is_file():
                LOGGER.warning("Optional cached qualifying results missing: %s", result_path)
                continue
            result_records = _read_json_cache(
                result_path,
                SESSION_RESULT_ENDPOINT,
            )
            records_by_endpoint[SESSION_RESULT_ENDPOINT].extend(
                {
                    **record,
                    "season": year,
                    "round": missing_grid_meeting_rounds[meeting_key],
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                }
                for record in result_records
            )

    meeting_records = []
    for meeting in meetings:
        meeting_records.append(
            {
                **meeting,
                "season": year,
                "round": round_by_meeting.get(meeting.get("meeting_key")),
            }
        )
    records_by_endpoint["meetings"] = meeting_records
    return _write_endpoint_parquets(year, records_by_endpoint, parquet_directory)


def _load_raw_seasons(
    seasons: Sequence[int],
    parquet_directory: Path,
) -> dict[str, pd.DataFrame]:
    """Read and concatenate endpoint parquet files for selected seasons."""
    return load_raw_data(
        years=seasons,
        data_directory=parquet_directory,
        endpoints=ENDPOINTS,
        required_endpoints=CORE_ENDPOINTS,
    )


def _normalize_pit_loss(pit_loss: pd.DataFrame) -> pd.DataFrame:
    """Add simulator-compatible pit-loss names and validate statistics."""
    result = normalize_pit_loss_schema(pit_loss)
    _require_columns(
        result,
        ("circuit", "median_pit_loss_s", "std_pit_loss_s", "n_stops"),
        "Pit-loss estimates",
    )
    result["median_pit_loss_s"] = pd.to_numeric(
        result["median_pit_loss_s"],
        errors="coerce",
    )
    result["std_pit_loss_s"] = pd.to_numeric(
        result["std_pit_loss_s"],
        errors="coerce",
    ).fillna(0.0)
    if result.empty or result["circuit"].isna().any():
        raise ValueError("Pit-loss estimates must contain circuits")
    if (
        not np.isfinite(result["median_pit_loss_s"]).all()
        or not np.isfinite(result["std_pit_loss_s"]).all()
        or result["std_pit_loss_s"].lt(0).any()
    ):
        raise ValueError("Pit-loss estimates contain invalid mean or standard deviation")
    return result


def _fit_smoke_models(
    features: pd.DataFrame,
    training_seasons: Sequence[int],
    holdout_season: int,
    models_directory: Path,
) -> dict[str, object]:
    """Fit lightweight sklearn models for the network-free smoke pipeline."""
    _require_columns(
        features,
        (*RACE_KEYS, *SMOKE_FEATURES, SMOKE_TARGET_CLASSIFICATION, SMOKE_TARGET_REGRESSION),
        "Feature data",
    )
    training = features.loc[features["season"].isin(training_seasons)].copy()
    holdout = features.loc[features["season"].eq(holdout_season)].copy()
    if training.empty or holdout.empty:
        raise ValueError(
            f"Smoke training requires seasons {tuple(training_seasons)} "
            f"and holdout season {holdout_season}"
        )
    if set(training["season"]) & set(holdout["season"]):
        raise ValueError("Smoke training and holdout seasons overlap")

    numeric_preprocessor = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, list(SMOKE_NUMERIC_FEATURES)),
            (
                "categorical",
                categorical_preprocessor,
                list(SMOKE_CATEGORICAL_FEATURES),
            ),
        ]
    )
    models_directory.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, object] = {}
    for target, estimator, filename in (
        (
            SMOKE_TARGET_CLASSIFICATION,
            RandomForestClassifier(
                n_estimators=25,
                max_depth=6,
                random_state=42,
                n_jobs=1,
            ),
            "pit_lap_classifier_smoke.joblib",
        ),
        (
            SMOKE_TARGET_REGRESSION,
            RandomForestRegressor(
                n_estimators=25,
                max_depth=6,
                random_state=42,
                n_jobs=1,
            ),
            "pit_lap_regressor_smoke.joblib",
        ),
    ):
        training_rows = training.dropna(subset=[target])
        if training_rows.empty:
            raise ValueError(f"No training rows with target {target!r}")
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessing),
                ("model", estimator),
            ]
        )
        model.fit(training_rows.loc[:, list(SMOKE_FEATURES)], training_rows[target])
        model_path = models_directory / filename
        joblib.dump(model, model_path)
        fitted[target] = {"model": model, "path": model_path}
        LOGGER.info("Fitted smoke model for %s at %s", target, model_path)
    return fitted


def _train_models(
    features: pd.DataFrame,
    training_seasons: Sequence[int],
    holdout_season: int,
    root: Path,
    smoke_test: bool,
) -> dict[str, object]:
    """Train either the production pit-lap models or fast smoke models."""
    if smoke_test:
        return _fit_smoke_models(
            features,
            training_seasons,
            holdout_season,
            root / "models",
        )

    from src.models import pit_lap

    training, holdout = pit_lap.split_by_season(
        features,
        train_seasons=tuple(training_seasons),
        test_season=holdout_season,
    )
    original_model_directory = pit_lap.MODEL_DIRECTORY
    original_figure_directory = pit_lap.FIGURE_DIRECTORY
    original_metrics_path = pit_lap.METRICS_PATH
    try:
        pit_lap.MODEL_DIRECTORY = root / "models"
        pit_lap.FIGURE_DIRECTORY = root / "reports" / "figures"
        pit_lap.METRICS_PATH = root / "reports" / "a5_metrics.csv"
        return pit_lap.run_a5(training, holdout)
    finally:
        pit_lap.MODEL_DIRECTORY = original_model_directory
        pit_lap.FIGURE_DIRECTORY = original_figure_directory
        pit_lap.METRICS_PATH = original_metrics_path


def _validate_processed_laps(laps: pd.DataFrame) -> None:
    """Require the clean-lap contract and usable race data."""
    _require_columns(laps, LAPS_CLEAN_COLUMNS, "Clean laps")
    if laps.empty:
        raise ValueError("Clean laps contain no rows")
    if laps.loc[:, ["season", "round", "race", "circuit", "driver"]].isna().any().any():
        raise ValueError("Clean laps contain missing race or driver identifiers")
    if pd.to_numeric(laps["lap_number"], errors="coerce").isna().any():
        raise ValueError("Clean laps contain missing or invalid lap_number values")
    if laps.duplicated([*RACE_KEYS, "driver", "lap_number"]).any():
        raise ValueError("Clean laps contain duplicate driver-lap rows")


def _load_training_pit_loss(
    training_laps: pd.DataFrame,
    supplied_pit_loss: pd.DataFrame | None,
) -> pd.DataFrame:
    """Estimate pit loss from training seasons or normalize injected estimates."""
    if supplied_pit_loss is not None:
        return _normalize_pit_loss(supplied_pit_loss)
    estimates = estimate_pit_loss(training_laps)
    return _normalize_pit_loss(estimates)


def run_pipeline(
    seasons: Sequence[int] = DEFAULT_SEASONS,
    *,
    skip_fetch: bool = False,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    smoke_test: bool = False,
    root: str | Path = ROOT,
    training_seasons: Sequence[int] = (2023, 2024),
    validation_season: int = 2025,
    n_validation_sims: int = 1000,
    max_validation_strategies: int = 50,
) -> dict[str, object]:
    """Fetch or reuse data, clean, train, fit degradation, and validate.

    ``synthetic_data`` may provide an already-clean lap DataFrame and pit-loss
    estimates for an offline smoke test. In that mode network fetching,
    raw-data cleaning, and pit-loss estimation are deliberately bypassed;
    feature building, model fitting, degradation fitting, and validation are
    still executed end-to-end.
    """
    selected_seasons = tuple(int(year) for year in seasons)
    train_years = tuple(int(year) for year in training_seasons)
    project_root = Path(root).resolve()
    processed_directory = project_root / "data" / "processed"
    raw_directory = project_root / "data" / "raw" / "openf1"
    parquet_directory = project_root / "data" / "raw" / "openf1_parquet"
    if not selected_seasons or not train_years:
        raise ValueError("seasons and training_seasons must not be empty")
    if validation_season in train_years:
        raise ValueError("validation season cannot be included in training_seasons")
    if any(year >= validation_season for year in train_years):
        raise ValueError("all training seasons must precede validation_season")
    if not set(train_years).issubset(selected_seasons):
        raise ValueError("selected seasons must include all training seasons")
    if validation_season not in selected_seasons:
        raise ValueError("selected seasons must include validation_season")
    if smoke_test and synthetic_data is None:
        raise ValueError("smoke_test requires synthetic_data=(laps, pit_loss)")

    processed_directory.mkdir(parents=True, exist_ok=True)
    root_raw_directory = (
        DEFAULT_RAW_DIR if project_root == ROOT else raw_directory
    )

    if synthetic_data is None:
        def fetch_step() -> None:
            if skip_fetch:
                LOGGER.info(
                    "Skipping network fetch; reconstructing seasons from cached "
                    "JSON in %s",
                    root_raw_directory,
                )
                for year in selected_seasons:
                    _rebuild_cached_season(
                        year,
                        root_raw_directory,
                        parquet_directory,
                        datetime.now(timezone.utc),
                    )
            else:
                for year in selected_seasons:
                    written = fetch_season(
                        year,
                        parquet_dir=parquet_directory,
                    )
                    missing_core = [
                        endpoint
                        for endpoint in CORE_ENDPOINTS
                        if endpoint not in written
                        or not written[endpoint].is_file()
                    ]
                    if missing_core:
                        raise FileNotFoundError(
                            f"OpenF1 fetch did not produce required data for "
                            f"{year}: {missing_core}"
                        )

        _run_step("fetch OpenF1 data", fetch_step)
        raw = _run_step(
            "load fetched endpoint data",
            lambda: _load_raw_seasons(selected_seasons, parquet_directory),
        )
        laps = _run_step("clean laps", lambda: clean_laps(raw))
        _validate_processed_laps(laps)
        laps_path = processed_directory / "laps_clean.parquet"
        _run_step(
            "save cleaned laps",
            lambda: laps.to_parquet(laps_path, index=False),
        )

        training_laps = laps.loc[laps["season"].isin(train_years)]
        if training_laps.empty:
            raise ValueError(f"No clean laps for training seasons {train_years}")
        pit_loss = _run_step(
            "estimate training-season pit loss",
            lambda: _load_training_pit_loss(training_laps, None),
        )
    else:
        LOGGER.info(
            "Synthetic offline mode: no network requests will be made"
        )
        laps, supplied_pit_loss = synthetic_data
        laps = laps.copy()
        _run_step(
            "use injected synthetic fetch data",
            lambda: LOGGER.info("Using %d synthetic clean lap rows", len(laps)),
        )
        _run_step(
            "validate injected clean laps",
            lambda: _validate_processed_laps(laps),
        )
        pit_loss = _run_step(
            "use injected synthetic pit-loss estimates",
            lambda: _normalize_pit_loss(supplied_pit_loss),
        )
        laps_path = processed_directory / "laps_clean.parquet"
        _run_step(
            "save synthetic cleaned laps",
            lambda: laps.to_parquet(laps_path, index=False),
        )

    pit_loss_path = processed_directory / "pit_loss.parquet"
    _run_step(
        "save pit-loss estimates",
        lambda: pit_loss.to_parquet(pit_loss_path, index=False),
    )
    required_seasons = set(train_years) | {validation_season}
    available_seasons = set(
        pd.to_numeric(laps["season"], errors="coerce").dropna().astype(int)
    )
    missing_seasons = sorted(required_seasons.difference(available_seasons))
    if missing_seasons:
        raise ValueError(f"Clean lap data is missing required seasons: {missing_seasons}")

    features = _run_step(
        "build race and driver features",
        lambda: build_features(laps, pit_loss),
    )
    features_path = processed_directory / "features.parquet"
    _run_step(
        "save features",
        lambda: features.to_parquet(features_path, index=False),
    )
    model_results = _run_step(
        "train pit-lap models",
        lambda: _train_models(
            features,
            train_years,
            validation_season,
            project_root,
            smoke_test,
        ),
    )

    training_laps = laps.loc[laps["season"].isin(train_years)].copy()
    degradation_model: DegradationModel = _run_step(
        "fit training-season degradation curves",
        lambda: fit_degradation(
            training_laps,
            per_circuit=True,
            n_bootstrap=300 if smoke_test else 1000,
            seed=42,
        ),
    )
    degradation_model_path = project_root / "models" / "degradation_model.joblib"
    _run_step(
        "save degradation model",
        lambda: joblib.dump(degradation_model, degradation_model_path),
    )

    validation_summary = _run_step(
        f"validate simulator on {validation_season}",
        lambda: validate_simulator(
            validation_season,
            laps_data=laps,
            pit_loss_data=pit_loss,
            training_seasons=train_years,
            n_sims=min(20, n_validation_sims) if smoke_test else n_validation_sims,
            seed=42,
            max_strategies_per_stop_count=(
                min(2, max_validation_strategies)
                if smoke_test
                else max_validation_strategies
            ),
            n_bootstrap=30 if smoke_test else 300,
            output_path=(
                project_root
                / "reports"
                / "figures"
                / f"simulator_validation_{validation_season}.png"
            ),
        ),
    )
    return {
        "laps": laps,
        "pit_loss": pit_loss,
        "features": features,
        "models": model_results,
        "degradation_model": degradation_model,
        "validation_summary": validation_summary,
        "outputs": {
            "laps": laps_path,
            "pit_loss": pit_loss_path,
            "features": features_path,
            "degradation_model": degradation_model_path,
            "validation_plot": validation_summary.attrs["plot_path"],
        },
    }


def main() -> None:
    """Parse command-line options and run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Fetch, clean, model, and validate F1 pitwall data."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEASONS),
        help="seasons to include (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--training-seasons",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="training seasons, all strictly earlier than the validation season",
    )
    parser.add_argument(
        "--validation-season",
        type=int,
        default=2025,
        help="held-out season to validate (default: 2025)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="rebuild fetched parquet files from cached data/raw/openf1 JSON",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.smoke_test:
        from tests.make_synthetic import generate_synthetic_data

        synthetic_laps, synthetic_pit_loss = generate_synthetic_data(
            output_dir=ROOT / "data" / "processed",
            seed=42,
        )
        results = run_pipeline(
            args.seasons,
            synthetic_data=(synthetic_laps, synthetic_pit_loss),
            smoke_test=True,
            training_seasons=args.training_seasons,
            validation_season=args.validation_season,
        )
    else:
        results = run_pipeline(
            args.seasons,
            skip_fetch=args.skip_fetch,
            training_seasons=args.training_seasons,
            validation_season=args.validation_season,
        )
    validation = results["validation_summary"]
    if isinstance(validation, pd.DataFrame):
        print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
