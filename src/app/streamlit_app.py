"""Streamlit dashboard for race state, pit predictions, and strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError

from src.data.schema import (
    MAX_PLAUSIBLE_LAP_DURATION_S,
    normalize_pit_loss_schema,
)
from src.features.build import build_features
from src.models.degradation import (
    DEFAULT_COMPOUNDS,
    DegradationModel,
    fit_degradation,
)
from src.sim.montecarlo import (
    DEFAULT_COMPOUND_OFFSETS_S,
    simulate_race,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
LAPS_PATH = PROCESSED_DIRECTORY / "laps_clean.parquet"
SYNTHETIC_LAPS_PATH = PROCESSED_DIRECTORY / "synthetic_laps_clean.parquet"
PIT_LOSS_PATH = PROCESSED_DIRECTORY / "pit_loss.parquet"
SYNTHETIC_PIT_LOSS_PATH = PROCESSED_DIRECTORY / "synthetic_pit_loss.parquet"
CLASSIFIER_PATH = MODEL_DIRECTORY / "pit_lap_classifier.joblib"
REGRESSOR_PATH = MODEL_DIRECTORY / "pit_lap_regressor.joblib"
N_SIMULATIONS = 1000
RANDOM_SEED = 42

MODEL_FEATURE_COLUMNS = (
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
    "circuit",
    "team",
    "compound",
)

DashboardScenario: TypeAlias = tuple[tuple[tuple[str, int], ...], int, bool]
DegradationFunction: TypeAlias = Callable[[str, float, str], float]


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file and cache its DataFrame for the Streamlit session."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet data file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


@st.cache_data(show_spinner="Building race state features...")
def load_model_features(laps: pd.DataFrame, pit_loss: pd.DataFrame) -> pd.DataFrame:
    """Build and cache the leakage-safe model features used by the dashboard."""
    return build_features(laps, pit_loss)


@st.cache_data(show_spinner="Fitting degradation curves...")
def fit_dashboard_degradation(laps: pd.DataFrame) -> DegradationModel:
    """Fit and cache per-circuit tyre degradation curves for available data."""
    return fit_degradation(laps, per_circuit=True, n_bootstrap=300, seed=42)


@st.cache_resource(show_spinner=False)
def load_joblib_model(path: str) -> tuple[object | None, str | None]:
    """Load one joblib model, returning a visible status for unavailable files."""
    model_path = Path(path)
    if not model_path.is_file():
        return None, f"Model file not found: {model_path}"
    try:
        return joblib.load(model_path), None
    except (
        AttributeError,
        EOFError,
        ImportError,
        OSError,
        ValueError,
        XGBoostError,
    ) as error:
        return None, f"Could not load {model_path.name}: {error}"


def build_remaining_scenarios(
    current_compound: str,
    remaining_laps: int,
    tyre_age: int,
    compounds: Sequence[str] = DEFAULT_COMPOUNDS,
    compounds_used: int = 1,
) -> dict[str, DashboardScenario]:
    """Build valid pit-now and planned-stop scenarios for remaining laps."""
    if remaining_laps <= 0:
        return {}
    if tyre_age < 0:
        raise ValueError("tyre_age cannot be negative")

    current = current_compound.upper()
    available = tuple(dict.fromkeys(compound.upper() for compound in compounds))
    alternatives = tuple(compound for compound in available if compound != current)
    scenarios: dict[str, DashboardScenario] = {}

    if compounds_used >= 2:
        scenarios["Stay out"] = (((current, remaining_laps),), tyre_age, False)

    for compound in alternatives:
        scenarios[f"Pit now · {compound}"] = (
            ((compound, remaining_laps),),
            0,
            True,
        )
        if remaining_laps > 1:
            hold_laps = min(5, max(1, remaining_laps // 2))
            if hold_laps < remaining_laps:
                scenarios[f"Pit in {hold_laps} laps · {compound}"] = (
                    (
                        (current, hold_laps),
                        (compound, remaining_laps - hold_laps),
                    ),
                    tyre_age,
                    False,
                )
    return scenarios


def summarize_simulations(
    distributions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Summarize scenario time samples as mean, P10, and P90."""
    if not distributions:
        return pd.DataFrame(columns=("mean", "P10", "P90"))
    return pd.DataFrame(
        {
            "mean": {name: float(np.mean(times)) for name, times in distributions.items()},
            "P10": {name: float(np.quantile(times, 0.10)) for name, times in distributions.items()},
            "P90": {name: float(np.quantile(times, 0.90)) for name, times in distributions.items()},
        }
    )


def _zero_degradation(compound: str, tyre_age: float, circuit: str) -> float:
    """Return no degradation when a fitted curve is unavailable."""
    if not compound.strip() or not circuit.strip():
        raise ValueError("compound and circuit must be non-empty")
    if not np.isfinite(tyre_age) or tyre_age < 0:
        raise ValueError("tyre_age must be finite and non-negative")
    return 0.0


def _dashboard_degradation_function(
    model: DegradationModel | None,
) -> DegradationFunction:
    """Return a degradation lookup with an explicit zero for missing curves."""
    if model is None:
        return _zero_degradation

    def degradation_loss(compound: str, tyre_age: float, circuit: str) -> float:
        """Return a fitted loss, or zero when that compound curve is absent."""
        key = (compound.upper(), circuit)
        if key not in model.curves and (compound.upper(), None) not in model.curves:
            return 0.0
        return model.degradation(compound, tyre_age, circuit).expected_loss_s

    return degradation_loss


def _predict_pit_state(
    features: pd.DataFrame,
    classifier: object | None,
    regressor: object | None,
) -> tuple[float | None, float | None]:
    """Predict near-term pit probability and laps to pit when models exist."""
    missing_features = set(MODEL_FEATURE_COLUMNS).difference(features.columns)
    if missing_features:
        raise ValueError(f"Model features are missing: {sorted(missing_features)}")
    model_input = features.loc[:, list(MODEL_FEATURE_COLUMNS)]
    pit_probability: float | None = None
    laps_to_pit: float | None = None

    if classifier is not None:
        probabilities = classifier.predict_proba(model_input)
        classes = np.asarray(classifier.classes_)
        positive_class = np.flatnonzero(classes == 1)
        if len(positive_class) != 1:
            raise ValueError("Pit classifier does not contain positive class 1")
        pit_probability = float(probabilities[0, int(positive_class[0])])

    if regressor is not None:
        prediction = float(np.asarray(regressor.predict(model_input)).reshape(-1)[0])
        if np.isfinite(prediction):
            laps_to_pit = max(0.0, prediction)

    return pit_probability, laps_to_pit


def _make_degradation_figure(
    model: DegradationModel,
    circuit: str,
    compounds: Sequence[str],
) -> plt.Figure:
    """Create the selected circuit's compound degradation curves."""
    curves = [
        (compound, model.curves[(compound, circuit)])
        for compound in compounds
        if (compound, circuit) in model.curves
    ]
    if not curves:
        raise ValueError(f"No degradation curves are available for {circuit}")

    maximum_age = max(curve.max_tyre_life for _, curve in curves)
    tyre_ages = np.linspace(0.0, maximum_age, 100)
    figure, axis = plt.subplots(figsize=(8, 4))
    for compound, _ in curves:
        estimates = [
            model.degradation(compound, float(age), circuit)
            for age in tyre_ages
        ]
        axis.plot(
            tyre_ages,
            [estimate.expected_loss_s for estimate in estimates],
            label=compound,
        )
        axis.fill_between(
            tyre_ages,
            [estimate.ci_lower_s for estimate in estimates],
            [estimate.ci_upper_s for estimate in estimates],
            alpha=0.18,
        )
    axis.set_title(f"Tyre degradation · {circuit}")
    axis.set_xlabel("Tyre age (laps)")
    axis.set_ylabel("Time loss vs fresh tyre (s)")
    axis.grid(alpha=0.25)
    axis.legend(title="Compound")
    figure.tight_layout()
    return figure


def _make_strategy_figure(summary: pd.DataFrame) -> plt.Figure:
    """Create a mean race-time bar chart with P10–P90 error bars."""
    figure, axis = plt.subplots(figsize=(10, 4.5))
    means = summary["mean"].to_numpy(dtype=float)
    errors = np.vstack(
        (
            means - summary["P10"].to_numpy(dtype=float),
            summary["P90"].to_numpy(dtype=float) - means,
        )
    )
    errors = np.maximum(errors, 0.0)
    axis.bar(
        summary.index.astype(str),
        means,
        yerr=errors,
        capsize=5,
        color="#e10600",
        alpha=0.82,
    )
    axis.set_ylabel("Remaining race time (s)")
    axis.set_title("Monte Carlo strategy comparison (mean with P10–P90)")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _simulate_scenarios(
    scenarios: Mapping[str, DashboardScenario],
    circuit: str,
    remaining_laps: int,
    base_pace_s: float,
    pit_loss: pd.DataFrame,
    degradation_model: DegradationModel | None,
    n_sims: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """Simulate each remaining-race scenario and return percentile summaries."""
    if not scenarios:
        return pd.DataFrame(columns=("mean", "P10", "P90"))

    degradation_fn = _dashboard_degradation_function(degradation_model)
    distributions = {
        name: simulate_race(
            strategy=strategy,
            circuit=circuit,
            n_laps=remaining_laps,
            n_sims=n_sims,
            seed=RANDOM_SEED,
            pit_loss_data=pit_loss,
            degradation_model=None,
            degradation_fn=degradation_fn,
            base_pace_s=base_pace_s,
            initial_tyre_age=starting_age,
            initial_pit_stop=pit_now,
        )
        for name, (strategy, starting_age, pit_now) in scenarios.items()
    }
    return summarize_simulations(distributions)


def _format_metric(value: object, suffix: str = "") -> str:
    """Format a finite numeric value for a Streamlit metric."""
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.2f}{suffix}"


def main() -> None:
    """Render the interactive pitwall strategy dashboard."""
    st.set_page_config(page_title="F1 Pitwall", page_icon="🏁", layout="wide")
    st.title("F1 Pitwall · Race strategy")
    st.caption("Live-state views and Monte Carlo estimates from processed race laps.")

    laps_path = LAPS_PATH if LAPS_PATH.is_file() else SYNTHETIC_LAPS_PATH
    pit_loss_path = PIT_LOSS_PATH if PIT_LOSS_PATH.is_file() else SYNTHETIC_PIT_LOSS_PATH
    if laps_path == SYNTHETIC_LAPS_PATH:
        st.warning(
            "Real processed laps were not found. The dashboard is using "
            "synthetic development data."
        )
    if pit_loss_path == SYNTHETIC_PIT_LOSS_PATH:
        st.warning("Using synthetic circuit pit-loss estimates.")

    try:
        laps = load_parquet(str(laps_path))
        pit_loss = normalize_pit_loss_schema(load_parquet(str(pit_loss_path)))
        features = load_model_features(laps, pit_loss)
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"Could not prepare race data: {error}")
        st.stop()

    model_status: dict[str, tuple[object | None, str | None]] = {
        "pit classifier": load_joblib_model(str(CLASSIFIER_PATH)),
        "laps-to-pit regressor": load_joblib_model(str(REGRESSOR_PATH)),
    }
    for label, (_, error) in model_status.items():
        if error is not None:
            st.warning(f"{label.title()} unavailable: {error}")

    seasons = sorted(pd.to_numeric(laps["season"], errors="coerce").dropna().astype(int).unique())
    if not seasons:
        st.error("No valid seasons were found in the lap data.")
        st.stop()
    with st.sidebar:
        st.header("Race selection")
        season = st.selectbox("Season", seasons, index=len(seasons) - 1)

    season_laps = features.loc[features["season"].eq(season)].copy()
    race_options = (
        season_laps[["round", "race", "circuit"]]
        .drop_duplicates()
        .sort_values(["round", "race", "circuit"])
    )
    race_options["label"] = race_options.apply(
        lambda row: f"R{int(row['round']):02d} · {row['race']} ({row['circuit']})",
        axis=1,
    )
    with st.sidebar:
        selected_race_label = st.selectbox("Race", race_options["label"].tolist())
        selected_race = race_options.loc[
            race_options["label"].eq(selected_race_label)
        ].iloc[0]
        race_laps = season_laps.loc[
            season_laps["round"].eq(selected_race["round"])
            & season_laps["race"].eq(selected_race["race"])
            & season_laps["circuit"].eq(selected_race["circuit"])
        ]
        drivers = sorted(race_laps["driver"].dropna().astype(str).unique())
        driver = st.selectbox("Driver", drivers)
        driver_laps = race_laps.loc[race_laps["driver"].astype(str).eq(driver)]
        lap_numbers = sorted(
            pd.to_numeric(driver_laps["lap_number"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        default_lap_index = max(0, len(lap_numbers) - 6)
        lap_number = st.selectbox(
            "Lap",
            lap_numbers,
            index=default_lap_index,
        )

    selected_rows = driver_laps.loc[driver_laps["lap_number"].eq(lap_number)]
    if selected_rows.empty:
        st.error("No selected driver-lap row is available.")
        st.stop()
    current = selected_rows.iloc[0]
    circuit = str(current["circuit"])

    st.subheader(f"{current['race']} · {driver} · Lap {lap_number}")
    state_columns = st.columns(4)
    state_columns[0].metric("Compound", str(current["compound"]))
    state_columns[1].metric("Tyre age", _format_metric(current["tyre_age"], " laps"))
    state_columns[2].metric("Gap ahead", _format_metric(current["gap_ahead"], " s"))
    state_columns[3].metric("Gap to leader", _format_metric(current["gap_leader"], " s"))

    weather_columns = st.columns(4)
    weather_columns[0].metric("Air temperature", _format_metric(current["air_temp"], " °C"))
    weather_columns[1].metric("Track temperature", _format_metric(current["track_temp"], " °C"))
    weather_columns[2].metric("Rainfall", _format_metric(current["rainfall"], " mm"))
    weather_columns[3].metric(
        "Conditions",
        "Safety car" if bool(current["is_sc"]) else (
            "Wet" if bool(current["is_wet"]) else "Dry"
        ),
    )

    classifier, _ = model_status["pit classifier"]
    regressor, _ = model_status["laps-to-pit regressor"]
    if classifier is not None or regressor is not None:
        try:
            probability, laps_to_pit = _predict_pit_state(
                selected_rows,
                classifier,
                regressor,
            )
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, XGBoostError) as error:
            st.error(f"Pit-stop model prediction failed: {error}")
        else:
            prediction_columns = st.columns(2)
            prediction_columns[0].metric(
                "Pit within 3 laps",
                "Unavailable" if probability is None else f"{probability:.1%}",
            )
            prediction_columns[1].metric(
                "Predicted laps to pit",
                _format_metric(laps_to_pit, " laps"),
            )
    else:
        st.info(
            "Pit probability and laps-to-pit predictions are unavailable until "
            "the pit-lap classifier and regressor can be loaded."
        )

    degradation_model: DegradationModel | None = None
    try:
        degradation_model = fit_dashboard_degradation(laps)
        available_compounds = tuple(
            compound
            for compound in DEFAULT_COMPOUNDS
            if (compound, circuit) in degradation_model.curves
        )
        if available_compounds:
            degradation_figure = _make_degradation_figure(
                degradation_model,
                circuit,
                available_compounds,
            )
            st.subheader("Tyre degradation")
            st.pyplot(degradation_figure, clear_figure=True)
            plt.close(degradation_figure)
        else:
            st.warning(f"No dry-compound degradation curves available for {circuit}.")
    except (ValueError, KeyError) as error:
        st.warning(f"Degradation curves unavailable: {error}")

    maximum_lap = int(race_laps["lap_number"].max())
    remaining_laps = maximum_lap - int(lap_number)
    if remaining_laps <= 0:
        st.info("The selected lap is the final lap; no remaining-lap simulation is available.")
        return

    current_compound = str(current["compound"]).upper()
    compounds_used = int(current.get("compounds_used", 1))
    tyre_age = int(current["tyre_age"])
    scenarios = build_remaining_scenarios(
        current_compound,
        remaining_laps,
        tyre_age,
        compounds_used=compounds_used,
    )
    if not scenarios:
        st.warning("No valid remaining-race strategies can be constructed.")
        return

    pit_loss = normalize_pit_loss_schema(pit_loss)
    pit_rows = pit_loss.loc[pit_loss["circuit"].eq(circuit)]
    if len(pit_rows) != 1:
        st.error(f"Expected one pit-loss estimate for {circuit}, found {len(pit_rows)}.")
        return

    try:
        current_curve_available = (
            degradation_model is not None
            and (
                (current_compound, circuit) in degradation_model.curves
                or (current_compound, None) in degradation_model.curves
            )
        )
        degradation_loss = (
            degradation_model.degradation(current_compound, tyre_age, circuit).expected_loss_s
            if current_curve_available and degradation_model is not None
            else 0.0
        )
        current_pace = pd.to_numeric(
            pd.Series([current["fuel_corrected_time_s"]]),
            errors="coerce",
        ).iloc[0]
        if (
            pd.isna(current_pace)
            or not np.isfinite(current_pace)
            or current_pace <= 0
            or current_pace > MAX_PLAUSIBLE_LAP_DURATION_S
        ):
            st.warning(
                "The selected lap has no plausible measured pace; "
                "remaining-race simulation is unavailable."
            )
            return
        base_pace_s = (
            float(current_pace)
            - DEFAULT_COMPOUND_OFFSETS_S.get(current_compound, 0.0)
            - degradation_loss
        )
        scenario_compounds = {
            compound
            for strategy, _, _ in scenarios.values()
            for compound, _ in strategy
        }
        missing_curves = {
            compound
            for compound in scenario_compounds
            if degradation_model is None
            or (
                (compound, circuit) not in degradation_model.curves
                and (compound, None) not in degradation_model.curves
            )
        }
        if missing_curves:
            st.warning(
                "Zero degradation is used where curves are missing for: "
                + ", ".join(sorted(missing_curves))
            )

        st.subheader("Remaining-race strategy comparison")
        summary = _simulate_scenarios(
            scenarios,
            circuit,
            remaining_laps,
            base_pace_s,
            pit_loss,
            degradation_model,
        )
        strategy_figure = _make_strategy_figure(summary)
        st.pyplot(strategy_figure, clear_figure=True)
        plt.close(strategy_figure)
        st.dataframe(summary.style.format({"mean": "{:.1f}", "P10": "{:.1f}", "P90": "{:.1f}"}))

        if st.button("What if I pit now?"):
            pit_now_scenarios = {
                name: scenario
                for name, scenario in scenarios.items()
                if name.startswith("Pit now")
            }
            if not pit_now_scenarios:
                st.warning("No alternative dry compound is available for a pit-now scenario.")
            else:
                pit_now_summary = _simulate_scenarios(
                    pit_now_scenarios,
                    circuit,
                    remaining_laps,
                    base_pace_s,
                    pit_loss,
                    degradation_model,
                )
                st.subheader("Pit-now re-simulation")
                pit_now_figure = _make_strategy_figure(pit_now_summary)
                st.pyplot(pit_now_figure, clear_figure=True)
                plt.close(pit_now_figure)
                st.dataframe(
                    pit_now_summary.style.format(
                        {"mean": "{:.1f}", "P10": "{:.1f}", "P90": "{:.1f}"}
                    )
                )
    except (ValueError, KeyError, FileNotFoundError) as error:
        st.error(f"Could not simulate remaining strategies: {error}")


if __name__ == "__main__":
    main()
