"""Tests for vectorized race strategy simulation."""

import numpy as np
import pandas as pd
import pytest

from src.sim.montecarlo import compare_strategies, simulate_race


def make_pit_loss_data(mean: float = 20.0, std: float = 0.0) -> pd.DataFrame:
    """Build a single-circuit pit-loss table for simulator tests."""
    return pd.DataFrame(
        {
            "circuit": ["Test Circuit"],
            "median_pit_loss_s": [mean],
            "std_pit_loss_s": [std],
        }
    )


def zero_degradation(compound: str, tyre_age: float, circuit: str) -> float:
    """Return no degradation for controlled pit-cost tests."""
    return 0.0


def linear_degradation(compound: str, tyre_age: float, circuit: str) -> float:
    """Return a simple deterministic loss for seeded simulation tests."""
    return 0.05 * tyre_age


def test_simulate_race_is_deterministic_for_fixed_seed() -> None:
    """The same seed and inputs must produce identical race-time samples."""
    options = {
        "pit_loss_data": make_pit_loss_data(std=1.5),
        "degradation_fn": linear_degradation,
        "n_sims": 100,
        "seed": 42,
    }
    first = simulate_race(
        [("MEDIUM", 10), ("HARD", 10)],
        "Test Circuit",
        20,
        **options,
    )
    second = simulate_race(
        [("MEDIUM", 10), ("HARD", 10)],
        "Test Circuit",
        20,
        **options,
    )

    np.testing.assert_array_equal(first, second)


def test_adding_a_stop_increases_race_time_without_degradation() -> None:
    """A positive pit loss increases total time if tyre degradation is zero."""
    common_options = {
        "pit_loss_data": make_pit_loss_data(mean=20.0),
        "degradation_fn": zero_degradation,
        "n_sims": 100,
        "lap_time_noise_s": 0.0,
        "safety_car_probability": 0.0,
    }
    no_stop = simulate_race(
        [("MEDIUM", 20)],
        "Test Circuit",
        20,
        **common_options,
    )
    one_stop = simulate_race(
        [("MEDIUM", 10), ("MEDIUM", 10)],
        "Test Circuit",
        20,
        **common_options,
    )

    assert np.all(one_stop > no_stop)
    np.testing.assert_allclose(one_stop - no_stop, 20.0)


def test_compare_strategies_returns_required_summary() -> None:
    """Comparison results include race-time percentiles and fastest chance."""
    summaries = compare_strategies(
        {
            "one_stop": [("MEDIUM", 10), ("HARD", 10)],
            "two_stop": [("MEDIUM", 7), ("HARD", 7), ("SOFT", 6)],
        },
        circuit="Test Circuit",
        n_laps=20,
        n_sims=100,
        pit_loss_data=make_pit_loss_data(),
        degradation_fn=zero_degradation,
        lap_time_noise_s=0.0,
        safety_car_probability=0.0,
    )

    assert summaries.columns.tolist() == [
        "mean",
        "P10",
        "P90",
        "probability_fastest",
    ]
    assert summaries.loc["one_stop", "mean"] < summaries.loc["two_stop", "mean"]
    assert summaries["probability_fastest"].sum() == pytest.approx(1.0)


def test_compare_strategies_splits_probability_for_ties() -> None:
    """Identical strategies share the probability of being fastest."""
    strategy = [("MEDIUM", 10), ("HARD", 10)]
    summaries = compare_strategies(
        {"first": strategy, "second": strategy},
        circuit="Test Circuit",
        n_laps=20,
        n_sims=10,
        pit_loss_data=make_pit_loss_data(),
        degradation_fn=zero_degradation,
        lap_time_noise_s=0.0,
        safety_car_probability=0.0,
    )

    assert summaries["probability_fastest"].tolist() == [0.5, 0.5]


def test_simulate_race_rejects_stints_that_do_not_cover_race() -> None:
    """Reject strategies whose total stint lengths differ from race distance."""
    with pytest.raises(ValueError, match="sum exactly"):
        simulate_race(
            [("SOFT", 9)],
            "Test Circuit",
            10,
            pit_loss_data=make_pit_loss_data(),
            degradation_fn=zero_degradation,
        )
