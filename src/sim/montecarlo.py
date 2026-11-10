"""Vectorized Monte Carlo simulation for race strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

from src.models.degradation import (
    DEFAULT_DATA_PATH as DEFAULT_LAPS_PATH,
    DegradationModel,
    load_degradation_model,
)
from src.data.schema import normalize_pit_loss_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIT_LOSS_PATH = PROJECT_ROOT / "data" / "processed" / "pit_loss.parquet"
DEFAULT_COMPOUND_OFFSETS_S = {"SOFT": 0.0, "MEDIUM": 0.35, "HARD": 0.75}
PIT_LOSS_COLUMNS = ("circuit", "median_pit_loss_s", "std_pit_loss_s")
Strategy: TypeAlias = Sequence[tuple[str, int]]
DegradationFunction: TypeAlias = Callable[[str, float, str], float]


def _validate_strategy(strategy: Strategy, n_laps: int) -> tuple[tuple[str, int], ...]:
    """Validate a strategy and return normalized compound/stint pairs."""
    if not strategy:
        raise ValueError("strategy must contain at least one stint")
    normalized: list[tuple[str, int]] = []
    for compound, stint_length in strategy:
        if not isinstance(compound, str) or not compound.strip():
            raise ValueError("each stint compound must be a non-empty string")
        if isinstance(stint_length, bool) or not isinstance(
            stint_length, (int, np.integer)
        ):
            raise ValueError("each stint length must be a positive integer")
        if stint_length <= 0:
            raise ValueError("each stint length must be a positive integer")
        normalized.append((compound.upper(), int(stint_length)))
    if sum(length for _, length in normalized) != n_laps:
        raise ValueError("stint lengths must sum exactly to n_laps")
    return tuple(normalized)


def _get_pit_loss(
    circuit: str,
    pit_loss_data: pd.DataFrame | None,
    pit_loss_path: str | Path,
) -> tuple[float, float]:
    """Return the circuit pit-loss mean and standard deviation."""
    if pit_loss_data is None:
        path = Path(pit_loss_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pit-loss parquet not found: {path}")
        pit_loss_data = pd.read_parquet(path)
    pit_loss_data = normalize_pit_loss_schema(pit_loss_data)

    missing_columns = sorted(set(PIT_LOSS_COLUMNS).difference(pit_loss_data.columns))
    if missing_columns:
        raise ValueError(f"Pit-loss data is missing columns: {missing_columns}")
    circuit_rows = pit_loss_data.loc[pit_loss_data["circuit"].eq(circuit)]
    if len(circuit_rows) != 1:
        raise ValueError(
            f"Expected exactly one pit-loss row for circuit {circuit!r}, "
            f"found {len(circuit_rows)}"
        )

    mean = float(circuit_rows.iloc[0]["median_pit_loss_s"])
    standard_deviation = float(circuit_rows.iloc[0]["std_pit_loss_s"])
    if (
        not np.isfinite(mean)
        or not np.isfinite(standard_deviation)
        or standard_deviation < 0
    ):
        raise ValueError(f"Invalid pit-loss statistics for circuit {circuit!r}")
    return mean, standard_deviation


def _get_degradation_function(
    degradation_model: DegradationModel | None,
    degradation_fn: DegradationFunction | None,
    laps_path: str | Path,
) -> DegradationFunction:
    """Resolve the supplied degradation function or load the fitted model."""
    if degradation_fn is not None:
        return degradation_fn
    model = degradation_model or load_degradation_model(data_path=laps_path)
    return lambda compound, tyre_age, circuit: model.degradation(
        compound,
        tyre_age,
        circuit,
    ).expected_loss_s


def simulate_race(
    strategy: Strategy,
    circuit: str,
    n_laps: int,
    n_sims: int = 5000,
    seed: int = 42,
    *,
    pit_loss_data: pd.DataFrame | None = None,
    pit_loss_path: str | Path = DEFAULT_PIT_LOSS_PATH,
    degradation_model: DegradationModel | None = None,
    degradation_fn: DegradationFunction | None = None,
    laps_path: str | Path = DEFAULT_LAPS_PATH,
    base_pace_s: float = 90.0,
    fuel_effect_s_per_lap: float = 0.03,
    compound_offsets_s: Mapping[str, float] | None = None,
    lap_time_noise_s: float = 0.2,
    safety_car_probability: float = 0.05,
    safety_car_pit_loss_factor: float = 0.5,
    safety_car_lap_slowdown_s: float = 8.0,
    initial_tyre_age: int = 0,
    initial_pit_stop: bool = False,
) -> np.ndarray:
    """Simulate total race times using vectorized Monte Carlo draws.

    Stint lengths must sum to ``n_laps``. Pit losses come from
    ``pit_loss_path`` by default; the data frame override supports tests and
    in-memory workflows. Degradation is evaluated deterministically from the
    supplied model or function, while lap noise, safety cars, and pit losses
    are sampled independently for each simulation.
    """
    if not isinstance(n_laps, int) or isinstance(n_laps, bool) or n_laps <= 0:
        raise ValueError("n_laps must be a positive integer")
    if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims <= 0:
        raise ValueError("n_sims must be a positive integer")
    if not 0 <= safety_car_probability <= 1:
        raise ValueError("safety_car_probability must be between 0 and 1")
    if not 0 <= safety_car_pit_loss_factor <= 1:
        raise ValueError("safety_car_pit_loss_factor must be between 0 and 1")
    scalar_parameters = (
        base_pace_s,
        fuel_effect_s_per_lap,
        lap_time_noise_s,
        safety_car_lap_slowdown_s,
    )
    if not all(np.isfinite(value) for value in scalar_parameters):
        raise ValueError("pace, fuel, noise, and safety-car slowdown must be finite")
    if lap_time_noise_s < 0:
        raise ValueError("lap_time_noise_s cannot be negative")
    if (
        not isinstance(initial_tyre_age, (int, np.integer))
        or isinstance(initial_tyre_age, bool)
        or initial_tyre_age < 0
    ):
        raise ValueError("initial_tyre_age must be a non-negative integer")

    normalized_strategy = _validate_strategy(strategy, n_laps)
    pit_loss_mean, pit_loss_std = _get_pit_loss(
        circuit,
        pit_loss_data,
        pit_loss_path,
    )
    get_degradation = _get_degradation_function(
        degradation_model,
        degradation_fn,
        laps_path,
    )
    offsets = {
        **DEFAULT_COMPOUND_OFFSETS_S,
        **{
            compound.upper(): float(offset)
            for compound, offset in (compound_offsets_s or {}).items()
        },
    }
    if not all(np.isfinite(offset) for offset in offsets.values()):
        raise ValueError("compound offsets must be finite")

    rng = np.random.default_rng(seed)
    safety_car_active = rng.random((n_sims, n_laps)) < safety_car_probability
    lap_times = np.full((n_sims, n_laps), base_pace_s, dtype=float)
    lap_times += (
        (n_laps - np.arange(1, n_laps + 1, dtype=float))
        * fuel_effect_s_per_lap
    )[None, :]
    lap_times += rng.normal(0.0, lap_time_noise_s, size=(n_sims, n_laps))
    lap_times += safety_car_active * safety_car_lap_slowdown_s

    lap_index = 0
    pit_stop_ends: list[int] = []
    for stint_index, (compound, stint_length) in enumerate(normalized_strategy):
        first_lap_age = initial_tyre_age + 1 if stint_index == 0 else 1
        lap_ages = np.arange(
            first_lap_age,
            first_lap_age + stint_length,
            dtype=float,
        )
        losses = np.array(
            [
                get_degradation(compound, tyre_age, circuit)
                for tyre_age in lap_ages
            ],
            dtype=float,
        )
        if not np.isfinite(losses).all():
            raise ValueError(
                f"Degradation returned non-finite values for {compound!r}"
            )
        lap_times[:, lap_index : lap_index + stint_length] += (
            offsets.get(compound, 0.0) + losses
        )[None, :]
        lap_index += stint_length
        if stint_index < len(normalized_strategy) - 1:
            pit_stop_ends.append(lap_index - 1)

    total_times = lap_times.sum(axis=1)
    if initial_pit_stop:
        sampled_pit_loss = rng.normal(pit_loss_mean, pit_loss_std, size=n_sims)
        total_times += np.where(
            safety_car_active[:, 0],
            sampled_pit_loss * safety_car_pit_loss_factor,
            sampled_pit_loss,
        )

    for pit_lap_index in pit_stop_ends:
        sampled_pit_loss = rng.normal(pit_loss_mean, pit_loss_std, size=n_sims)
        reduced_pit_loss = np.where(
            safety_car_active[:, pit_lap_index],
            sampled_pit_loss * safety_car_pit_loss_factor,
            sampled_pit_loss,
        )
        total_times += reduced_pit_loss
    return total_times


def compare_strategies(
    strategies: Mapping[str, Strategy],
    circuit: str,
    n_laps: int,
    n_sims: int = 5000,
    seed: int = 42,
    *,
    pit_loss_data: pd.DataFrame | None = None,
    pit_loss_path: str | Path = DEFAULT_PIT_LOSS_PATH,
    degradation_model: DegradationModel | None = None,
    degradation_fn: DegradationFunction | None = None,
    laps_path: str | Path = DEFAULT_LAPS_PATH,
    base_pace_s: float = 90.0,
    fuel_effect_s_per_lap: float = 0.03,
    compound_offsets_s: Mapping[str, float] | None = None,
    lap_time_noise_s: float = 0.2,
    safety_car_probability: float = 0.05,
    safety_car_pit_loss_factor: float = 0.5,
    safety_car_lap_slowdown_s: float = 8.0,
    initial_tyre_age: int = 0,
    initial_pit_stop: bool = False,
) -> pd.DataFrame:
    """Compare strategy time distributions and probability of being fastest.

    Returns a DataFrame indexed by strategy name, with ``mean``, ``P10``,
    ``P90``, and ``probability_fastest`` columns. Every strategy uses the same
    seed to make comparisons reproducible.
    """
    if not strategies:
        raise ValueError("strategies must contain at least one strategy")

    time_distributions = {
        name: simulate_race(
            strategy=strategy,
            circuit=circuit,
            n_laps=n_laps,
            n_sims=n_sims,
            seed=seed,
            pit_loss_data=pit_loss_data,
            pit_loss_path=pit_loss_path,
            degradation_model=degradation_model,
            degradation_fn=degradation_fn,
            laps_path=laps_path,
            base_pace_s=base_pace_s,
            fuel_effect_s_per_lap=fuel_effect_s_per_lap,
            compound_offsets_s=compound_offsets_s,
            lap_time_noise_s=lap_time_noise_s,
            safety_car_probability=safety_car_probability,
            safety_car_pit_loss_factor=safety_car_pit_loss_factor,
            safety_car_lap_slowdown_s=safety_car_lap_slowdown_s,
            initial_tyre_age=initial_tyre_age,
            initial_pit_stop=initial_pit_stop,
        )
        for name, strategy in strategies.items()
    }
    names = list(time_distributions)
    times = np.vstack([time_distributions[name] for name in names])
    minimum_times = times.min(axis=0)
    winners = np.isclose(times, minimum_times[None, :])
    winner_counts = winners.sum(axis=0)
    return pd.DataFrame(
        {
            "mean": [float(np.mean(time_distributions[name])) for name in names],
            "P10": [float(np.quantile(time_distributions[name], 0.10)) for name in names],
            "P90": [float(np.quantile(time_distributions[name], 0.90)) for name in names],
            "probability_fastest": [
                float(np.mean(winners[index] / winner_counts))
                for index in range(len(names))
            ],
        },
        index=pd.Index(names, name="strategy"),
    )
