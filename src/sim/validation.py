"""Out-of-season validation for simulated race strategies."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.degradation import (
    DEFAULT_DATA_PATH as DEFAULT_LAPS_PATH,
    DEFAULT_COMPOUNDS,
    DegradationModel,
    fit_degradation,
)
from src.sim.montecarlo import (
    DEFAULT_COMPOUND_OFFSETS_S,
    DEFAULT_PIT_LOSS_PATH,
    Strategy,
    simulate_race,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_PATH_TEMPLATE = "simulator_validation_{season}.png"
RACE_KEY_COLUMNS = ("season", "round", "race", "circuit")
REQUIRED_RACE_COLUMNS = (
    "season",
    "round",
    "race",
    "circuit",
    "driver",
    "lap_number",
    "lap_time_s",
    "compound",
    "tyre_life",
    "position",
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
    "fuel_corrected_time_s",
)


def _read_frame(
    frame: pd.DataFrame | None,
    path: str | Path,
    description: str,
) -> pd.DataFrame:
    """Return a copy of an in-memory frame or read its parquet path."""
    if frame is not None:
        return frame.copy()
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"{description} parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _candidate_strategies(
    n_laps: int,
    n_stops: int,
    compounds: Sequence[str],
    min_stint_laps: int,
    timing_step: int,
) -> Iterator[Strategy]:
    """Enumerate compound and pit-lap combinations satisfying race rules."""
    n_stints = n_stops + 1
    if n_laps < n_stints * min_stint_laps:
        return

    possible_cuts = range(
        min_stint_laps,
        n_laps - min_stint_laps + 1,
        timing_step,
    )
    for cuts in combinations(possible_cuts, n_stops):
        boundaries = (0, *cuts, n_laps)
        stint_lengths = tuple(
            boundaries[index + 1] - boundaries[index]
            for index in range(n_stints)
        )
        if min(stint_lengths) < min_stint_laps:
            continue
        for compound_sequence in product(compounds, repeat=n_stints):
            if len(set(compound_sequence)) < 2:
                continue
            yield tuple(zip(compound_sequence, stint_lengths))


def _sample_strategies(
    candidates: Iterator[Strategy],
    maximum: int,
    seed: int,
) -> tuple[list[Strategy], int]:
    """Use seeded reservoir sampling to cap strategy simulation cost."""
    rng = np.random.default_rng(seed)
    sample: list[Strategy] = []
    total_candidates = 0
    for candidate in candidates:
        total_candidates += 1
        if len(sample) < maximum:
            sample.append(candidate)
            continue
        replacement_index = int(rng.integers(0, total_candidates))
        if replacement_index < maximum:
            sample[replacement_index] = candidate
    return sample, total_candidates


def _clean_training_laps(laps: pd.DataFrame, seasons: Sequence[int]) -> pd.DataFrame:
    """Select the specified seasons' valid, clean training laps."""
    training = laps.loc[laps["season"].isin(seasons)].copy()
    clean_mask = pd.Series(True, index=training.index)
    for flag in ("is_inlap", "is_outlap", "is_sc", "is_wet"):
        clean_mask &= training[flag].eq(False).fillna(False)

    training = training.loc[clean_mask].copy()
    training["tyre_life"] = pd.to_numeric(training["tyre_life"], errors="coerce")
    training["fuel_corrected_time_s"] = pd.to_numeric(
        training["fuel_corrected_time_s"],
        errors="coerce",
    )
    training = training.dropna(
        subset=("circuit", "compound", "tyre_life", "fuel_corrected_time_s")
    )
    training = training[
        np.isfinite(training["fuel_corrected_time_s"])
        & np.isfinite(pd.to_numeric(training["tyre_life"], errors="coerce"))
    ]
    if training.empty:
        raise ValueError(f"No clean training laps found for seasons {tuple(seasons)}")
    return training


def _estimate_training_pace(
    training_laps: pd.DataFrame,
    model: DegradationModel,
) -> dict[str, float]:
    """Estimate circuit baseline pace using training laps only."""
    adjusted_paces: dict[str, list[float]] = {}
    for row in training_laps.itertuples(index=False):
        compound = str(row.compound).upper()
        circuit = str(row.circuit)
        degradation = model.degradation(
            compound,
            float(row.tyre_life),
            circuit,
        ).expected_loss_s
        compound_offset = DEFAULT_COMPOUND_OFFSETS_S.get(compound, 0.0)
        adjusted_paces.setdefault(circuit, []).append(
            float(row.fuel_corrected_time_s) - degradation - compound_offset
        )
    return {
        circuit: float(np.median(paces))
        for circuit, paces in adjusted_paces.items()
    }


def _sum_complete_laps(lap_times: pd.Series) -> float:
    """Sum a driver's lap times only when every lap time is present."""
    return float(lap_times.sum(min_count=len(lap_times)))


def _winner_result(race_laps: pd.DataFrame) -> tuple[str, int, float, int]:
    """Return winner driver, stop count, elapsed time, and lap distance."""
    race_laps = race_laps.copy()
    race_laps["lap_number"] = pd.to_numeric(
        race_laps["lap_number"],
        errors="coerce",
    )
    race_laps["lap_time_s"] = pd.to_numeric(
        race_laps["lap_time_s"],
        errors="coerce",
    )
    numeric_positions = pd.to_numeric(race_laps["position"], errors="coerce")
    final_lap = race_laps["lap_number"].max()
    final_lap_rows = race_laps.loc[race_laps["lap_number"].eq(final_lap)].copy()
    final_lap_rows["_position"] = numeric_positions.loc[final_lap_rows.index]
    classified = final_lap_rows.dropna(subset=["_position"])
    if classified.empty:
        raise ValueError("Race has no classified driver on its final lap")

    winning_position = classified["_position"].min()
    winner_candidates = classified.loc[
        classified["_position"].eq(winning_position),
        "driver",
    ].unique()
    candidate_laps = race_laps.loc[
        race_laps["driver"].isin(winner_candidates)
    ]
    driver_totals = (
        candidate_laps.groupby("driver", sort=True)["lap_time_s"]
        .agg(_sum_complete_laps)
    )
    driver_totals = driver_totals.dropna()
    if driver_totals.empty:
        raise ValueError("Classified winner candidates have incomplete lap times")
    driver = str(driver_totals.idxmin())
    winner_laps = race_laps.loc[race_laps["driver"].eq(driver)]
    actual_time = float(winner_laps["lap_time_s"].sum(min_count=len(winner_laps)))
    if not np.isfinite(actual_time):
        raise ValueError("Winner's actual race time is not finite")
    actual_stops = int(winner_laps["is_inlap"].fillna(False).astype(bool).sum())
    return driver, actual_stops, actual_time, int(final_lap)


def _save_validation_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save stop-count accuracy and time-error plots for the summary."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    race_labels = summary["race"].astype(str)
    x_positions = np.arange(len(summary))
    axes[0].bar(
        x_positions - 0.18,
        summary["actual_stop_count"],
        width=0.36,
        label="Actual winner",
    )
    axes[0].bar(
        x_positions + 0.18,
        summary["simulated_stop_count"],
        width=0.36,
        label="Simulated best",
    )
    axes[0].set_ylabel("Pit stops")
    axes[0].set_title("Stop-count comparison")
    axes[0].set_xticks(x_positions, race_labels, rotation=75, ha="right")
    axes[0].legend()

    axes[1].bar(x_positions, summary["time_error_s"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Simulated mean - actual winner (s)")
    axes[1].set_title("Race-time error")
    axes[1].set_xticks(x_positions, race_labels, rotation=75, ha="right")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def validate_simulator(
    season: int = 2025,
    *,
    laps_data: pd.DataFrame | None = None,
    pit_loss_data: pd.DataFrame | None = None,
    laps_path: str | Path = DEFAULT_LAPS_PATH,
    pit_loss_path: str | Path = DEFAULT_PIT_LOSS_PATH,
    training_seasons: Sequence[int] = (2023, 2024),
    n_sims: int = 1000,
    seed: int = 42,
    min_stint_laps: int = 8,
    timing_step: int = 5,
    max_strategies_per_stop_count: int = 50,
    n_bootstrap: int = 300,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate simulated strategies against every race in a held-out season.

    Degradation curves and circuit baseline pace are fitted using only
    ``training_seasons``. Candidate strategies cover 1, 2, and 3 stops, use
    at least two compounds, and have stints of at least ``min_stint_laps``.
    Pit timing candidates are placed on a ``timing_step`` lap grid; when the
    candidate space exceeds the cap, seeded reservoir sampling selects which
    strategies to simulate for each stop count.

    Returns one summary row per race and saves a two-panel plot. In a notebook,
    the returned DataFrame is directly displayable.
    """
    if season in training_seasons:
        raise ValueError("validation season cannot be included in training_seasons")
    if any(training_season >= season for training_season in training_seasons):
        raise ValueError("all training seasons must precede the validation season")
    if not training_seasons:
        raise ValueError("training_seasons must not be empty")
    if min_stint_laps <= 0 or timing_step <= 0:
        raise ValueError("min_stint_laps and timing_step must be positive")
    if max_strategies_per_stop_count <= 0:
        raise ValueError("max_strategies_per_stop_count must be positive")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    laps = _read_frame(laps_data, laps_path, "Laps")
    missing_columns = sorted(set(REQUIRED_RACE_COLUMNS).difference(laps.columns))
    if missing_columns:
        raise ValueError(f"Lap data is missing columns: {missing_columns}")
    if not laps["season"].eq(season).any():
        raise ValueError(f"No lap data found for validation season {season}")

    training_laps = _clean_training_laps(laps, training_seasons)
    degradation_model = fit_degradation(
        training_laps,
        per_circuit=True,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    baseline_paces = _estimate_training_pace(training_laps, degradation_model)
    pit_data = _read_frame(pit_loss_data, pit_loss_path, "Pit-loss")
    compounds = tuple(
        compound
        for compound in DEFAULT_COMPOUNDS
        if compound in degradation_model.compounds
    )
    if len(compounds) < 2:
        raise ValueError(
            "At least two of SOFT, MEDIUM, and HARD must exist in training data"
        )

    target_laps = laps.loc[laps["season"].eq(season)]
    summary_rows: list[dict[str, object]] = []
    for race_key, race_laps in target_laps.groupby(
        list(RACE_KEY_COLUMNS),
        sort=True,
        dropna=False,
    ):
        season_number, round_number, race_name, circuit_name = race_key
        circuit = str(circuit_name)
        if circuit not in baseline_paces:
            raise ValueError(
                f"No training baseline pace for circuit {circuit!r}"
            )
        winner, actual_stops, actual_time, n_laps = _winner_result(race_laps)

        candidates_by_stops: dict[int, list[Strategy]] = {}
        candidate_counts: dict[int, int] = {}
        for n_stops in (1, 2, 3):
            candidates_by_stops[n_stops], candidate_counts[n_stops] = (
                _sample_strategies(
                    _candidate_strategies(
                        n_laps=n_laps,
                        n_stops=n_stops,
                        compounds=compounds,
                        min_stint_laps=min_stint_laps,
                        timing_step=timing_step,
                    ),
                    maximum=max_strategies_per_stop_count,
                    seed=seed + int(round_number) * 10 + n_stops,
                )
            )

        evaluated_strategies = [
            (n_stops, strategy)
            for n_stops, strategies in candidates_by_stops.items()
            for strategy in strategies
        ]
        if not evaluated_strategies:
            raise ValueError(
                f"No valid candidate strategies for race {race_name!r}"
            )

        race_seed = seed + int(round_number) * 100
        simulated_means: list[float] = []
        for _, strategy in evaluated_strategies:
            distribution = simulate_race(
                strategy=strategy,
                circuit=circuit,
                n_laps=n_laps,
                n_sims=n_sims,
                seed=race_seed,
                pit_loss_data=pit_data,
                degradation_model=degradation_model,
                base_pace_s=baseline_paces[circuit],
            )
            simulated_means.append(float(distribution.mean()))

        best_index = int(np.argmin(simulated_means))
        best_stop_count, best_strategy = evaluated_strategies[best_index]
        simulated_mean = simulated_means[best_index]
        summary_rows.append(
            {
                "season": int(season_number),
                "round": int(round_number),
                "race": str(race_name),
                "circuit": circuit,
                "winner": winner,
                "actual_stop_count": actual_stops,
                "simulated_best_strategy": best_strategy,
                "simulated_stop_count": best_stop_count,
                "stop_count_matches": best_stop_count == actual_stops,
                "simulated_mean_time_s": simulated_mean,
                "actual_race_time_s": actual_time,
                "time_error_s": simulated_mean - actual_time,
                "absolute_time_error_s": abs(simulated_mean - actual_time),
                "candidate_strategies": sum(candidate_counts.values()),
                "strategies_evaluated": len(evaluated_strategies),
            }
        )

    summary = pd.DataFrame(summary_rows)
    plot_path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT
        / "reports"
        / "figures"
        / DEFAULT_PLOT_PATH_TEMPLATE.format(season=season)
    )
    _save_validation_plot(summary, plot_path)
    summary.attrs["training_seasons"] = tuple(training_seasons)
    summary.attrs["plot_path"] = str(plot_path)
    return summary
