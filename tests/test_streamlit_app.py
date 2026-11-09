"""Tests for dashboard strategy and prediction helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.app.streamlit_app import (
    _predict_pit_state,
    build_remaining_scenarios,
    load_joblib_model,
    summarize_simulations,
)


class FakeClassifier:
    """Minimal classifier matching the saved model prediction interface."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return a fixed probability distribution for one selected row."""
        assert len(features) == 1
        return np.array([[0.7, 0.3]])


class FakeRegressor:
    """Minimal regressor matching the saved model prediction interface."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a fixed pit-horizon estimate for one selected row."""
        assert len(features) == 1
        return np.array([4.5])


def test_build_remaining_scenarios_resets_age_when_pitting_now() -> None:
    """Pit-now scenarios use a fresh tyre while deferred stops keep current age."""
    scenarios = build_remaining_scenarios(
        "MEDIUM",
        remaining_laps=12,
        tyre_age=9,
        compounds_used=1,
    )

    assert "Pit now · SOFT" in scenarios
    assert scenarios["Pit now · SOFT"] == ((("SOFT", 12),), 0, True)
    assert scenarios["Pit in 5 laps · SOFT"] == (
        (("MEDIUM", 5), ("SOFT", 7)),
        9,
        False,
    )
    assert "Stay out" not in scenarios


def test_build_remaining_scenarios_allows_staying_out_after_rule_satisfied() -> None:
    """A no-stop option is offered once the two-compound rule is met."""
    scenarios = build_remaining_scenarios(
        "HARD",
        remaining_laps=8,
        tyre_age=4,
        compounds_used=2,
    )

    assert scenarios["Stay out"] == ((("HARD", 8),), 4, False)


def test_predict_pit_state_uses_saved_model_interfaces() -> None:
    """Classifier probability and regressor output are surfaced independently."""
    from src.app.streamlit_app import MODEL_FEATURE_COLUMNS

    features = pd.DataFrame(
        [{column: 0 for column in MODEL_FEATURE_COLUMNS}]
    )
    probability, laps_to_pit = _predict_pit_state(
        features,
        FakeClassifier(),
        FakeRegressor(),
    )

    assert probability == 0.3
    assert laps_to_pit == 4.5


def test_summarize_simulations_calculates_requested_quantiles() -> None:
    """Scenario samples are summarized using the mean and decile bounds."""
    summary = summarize_simulations(
        {"candidate": np.array([10.0, 20.0, 30.0])}
    )

    assert summary.loc["candidate", "mean"] == 20.0
    assert summary.loc["candidate", "P10"] == 12.0
    assert summary.loc["candidate", "P90"] == 28.0


def test_missing_joblib_model_returns_visible_status(tmp_path: Path) -> None:
    """Missing optional models do not prevent dashboard helpers from running."""
    model, error = load_joblib_model(str(tmp_path / "missing.joblib"))

    assert model is None
    assert error is not None
    assert "not found" in error
