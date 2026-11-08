"""Tests for clean-lap degradation fitting and plotting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.degradation import (
    fit_degradation,
    plot_degradation_curves,
)


def make_synthetic_degradation_laps() -> pd.DataFrame:
    """Create reproducible clean laps with known compound/circuit slopes."""
    rng = np.random.default_rng(42)
    rows = []
    slopes = {
        ("SOFT", "Bahrain"): 0.08,
        ("MEDIUM", "Bahrain"): 0.06,
        ("HARD", "Bahrain"): 0.04,
        ("SOFT", "Monaco"): 0.09,
        ("MEDIUM", "Monaco"): 0.07,
        ("HARD", "Monaco"): 0.05,
        ("SOFT", "Silverstone"): 0.07,
        ("MEDIUM", "Silverstone"): 0.05,
        ("HARD", "Silverstone"): 0.03,
    }
    for (compound, circuit), slope in slopes.items():
        baseline = 70.0 + len(circuit) + len(compound)
        for tyre_life in range(1, 21):
            for _ in range(12):
                rows.append(
                    {
                        "circuit": circuit,
                        "compound": compound,
                        "tyre_life": tyre_life,
                        "fuel_corrected_time_s": (
                            baseline + slope * tyre_life + rng.normal(0.0, 0.01)
                        ),
                        "is_inlap": False,
                        "is_outlap": False,
                        "is_sc": False,
                        "is_wet": False,
                    }
                )
    return pd.DataFrame(rows)


def test_fit_degradation_recovers_slopes_and_is_reproducible() -> None:
    """Verify synthetic curves, bootstrap intervals, and fixed-seed behavior."""
    laps = make_synthetic_degradation_laps()
    model = fit_degradation(laps, n_bootstrap=100, seed=42)
    repeated = fit_degradation(laps, n_bootstrap=100, seed=42)

    estimate = model.degradation("SOFT", tyre_age=10, circuit="Bahrain")
    repeated_estimate = repeated.degradation("SOFT", tyre_age=10, circuit="Bahrain")

    assert estimate.expected_loss_s == pytest.approx(0.8, abs=0.01)
    assert estimate.ci_lower_s <= estimate.expected_loss_s
    assert estimate.expected_loss_s <= estimate.ci_upper_s
    assert estimate == repeated_estimate


def test_fit_degradation_excludes_flagged_laps() -> None:
    """Ensure inlaps and other excluded conditions do not affect fitted curves."""
    laps = make_synthetic_degradation_laps()
    expected = fit_degradation(laps, n_bootstrap=50, seed=42)
    flagged_lap = laps.iloc[[0]].copy()
    flagged_lap["is_sc"] = True
    flagged_lap["fuel_corrected_time_s"] = 1000.0
    actual = fit_degradation(
        pd.concat((laps, flagged_lap), ignore_index=True),
        n_bootstrap=50,
        seed=42,
    )

    assert (
        actual.degradation("SOFT", 10, "Bahrain")
        == expected.degradation("SOFT", 10, "Bahrain")
    )


def test_fit_degradation_can_pool_circuits() -> None:
    """Verify optional compound-only fits are available across circuits."""
    model = fit_degradation(
        make_synthetic_degradation_laps(),
        per_circuit=False,
        n_bootstrap=50,
    )

    assert model.circuits == ()
    estimate = model.degradation("HARD", 10, "Bahrain")
    assert 0.3 < estimate.expected_loss_s < 0.5


def test_plot_degradation_curves_saves_figure(tmp_path: Path) -> None:
    """Check that the requested compound and circuit comparison is saved."""
    model = fit_degradation(make_synthetic_degradation_laps(), n_bootstrap=50)
    output_path = plot_degradation_curves(
        model,
        output_path=tmp_path / "figures" / "degradation.png",
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_fit_degradation_rejects_missing_columns() -> None:
    """Report a useful error when required lap fields are absent."""
    with pytest.raises(ValueError, match="missing columns"):
        fit_degradation(pd.DataFrame({"compound": ["SOFT"]}))
