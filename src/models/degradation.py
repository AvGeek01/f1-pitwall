"""Fit tyre degradation curves from clean, fuel-corrected laps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "laps_clean.parquet"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "reports" / "figures" / "degradation_curves.png"

REQUIRED_COLUMNS = (
    "circuit",
    "compound",
    "tyre_life",
    "fuel_corrected_time_s",
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
)
EXCLUSION_FLAGS = ("is_inlap", "is_outlap", "is_sc", "is_wet")
DEFAULT_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class DegradationEstimate:
    """Expected time loss and percentile confidence interval, in seconds."""

    expected_loss_s: float
    ci_lower_s: float
    ci_upper_s: float


@dataclass(frozen=True)
class _Curve:
    """Linear degradation fit and its bootstrapped slope distribution."""

    slope_s_per_lap: float
    bootstrap_slopes: np.ndarray
    max_tyre_life: float
    observations: int


@dataclass(frozen=True)
class DegradationModel:
    """Fitted per-compound, optionally per-circuit degradation curves."""

    curves: dict[tuple[str, str | None], _Curve]
    per_circuit: bool

    @property
    def compounds(self) -> tuple[str, ...]:
        """Return the compounds represented by fitted curves."""
        return tuple(sorted({compound for compound, _ in self.curves}))

    @property
    def circuits(self) -> tuple[str, ...]:
        """Return circuits represented in the fitted data."""
        return tuple(
            sorted(
                circuit
                for _, circuit in self.curves
                if circuit is not None
            )
        )

    def degradation(
        self,
        compound: str,
        tyre_age: float,
        circuit: str | None,
    ) -> DegradationEstimate:
        """Estimate loss from a fresh tyre and its bootstrapped 95% interval.

        A circuit-specific fit is preferred when available. If no such fit
        exists, the compound-wide curve is used. The return value is ordered
        as expected loss, lower confidence bound, and upper confidence bound.
        """
        age = float(tyre_age)
        if not np.isfinite(age) or age < 0:
            raise ValueError("tyre_age must be a finite, non-negative number")

        compound_name = str(compound).upper()
        curve = self.curves.get((compound_name, circuit))
        if curve is None:
            curve = self.curves.get((compound_name, None))
        if curve is None:
            raise ValueError(
                f"No degradation curve for compound {compound_name!r}"
                + (f" at circuit {circuit!r}" if circuit is not None else "")
            )

        bootstrapped_losses = curve.bootstrap_slopes * age
        lower, upper = np.quantile(bootstrapped_losses, (0.025, 0.975))
        return DegradationEstimate(
            expected_loss_s=float(curve.slope_s_per_lap * age),
            ci_lower_s=float(lower),
            ci_upper_s=float(upper),
        )


def _fit_curve(
    tyre_life: np.ndarray,
    lap_times: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> _Curve:
    """Fit a linear curve and bootstrap its slope by resampling lap rows."""
    if np.unique(tyre_life).size < 2:
        raise ValueError("Each degradation curve needs at least two tyre ages")

    slope, _ = np.polyfit(tyre_life, lap_times, deg=1)
    sample_count = tyre_life.size
    bootstrap_slopes = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        sampled_ages = tyre_life[sample_indices]
        if np.unique(sampled_ages).size < 2:
            bootstrap_slopes[bootstrap_index] = slope
        else:
            bootstrap_slopes[bootstrap_index] = np.polyfit(
                sampled_ages,
                lap_times[sample_indices],
                deg=1,
            )[0]

    return _Curve(
        slope_s_per_lap=float(slope),
        bootstrap_slopes=bootstrap_slopes,
        max_tyre_life=float(tyre_life.max()),
        observations=sample_count,
    )


def fit_degradation(
    laps: pd.DataFrame,
    per_circuit: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DegradationModel:
    """Fit clean-lap linear degradation curves from a laps DataFrame.

    Rows with any exclusion flag set are removed before fitting. With
    ``per_circuit=False``, one curve is fitted per compound across all circuits.
    """
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(laps.columns))
    if missing_columns:
        raise ValueError(f"Lap data is missing columns: {missing_columns}")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    clean_mask = pd.Series(True, index=laps.index)
    for flag in EXCLUSION_FLAGS:
        clean_mask &= laps[flag].eq(False).fillna(False)

    clean_laps = laps.loc[clean_mask, list(REQUIRED_COLUMNS)].copy()
    clean_laps["tyre_life"] = pd.to_numeric(
        clean_laps["tyre_life"],
        errors="coerce",
    )
    clean_laps["fuel_corrected_time_s"] = pd.to_numeric(
        clean_laps["fuel_corrected_time_s"],
        errors="coerce",
    )
    clean_laps = clean_laps.dropna(
        subset=("circuit", "compound", "tyre_life", "fuel_corrected_time_s")
    )
    clean_laps = clean_laps[
        np.isfinite(clean_laps["tyre_life"])
        & np.isfinite(clean_laps["fuel_corrected_time_s"])
        & clean_laps["tyre_life"].ge(0)
    ]
    if clean_laps.empty:
        raise ValueError("No valid clean laps are available for fitting")

    clean_laps["compound"] = clean_laps["compound"].astype(str).str.upper()
    rng = np.random.default_rng(seed)
    curves: dict[tuple[str, str | None], _Curve] = {}
    group_columns = ["compound", "circuit"] if per_circuit else ["compound"]
    for group_key, group in clean_laps.groupby(group_columns, sort=True):
        compound = str(group_key[0] if isinstance(group_key, tuple) else group_key)
        circuit = (
            str(group_key[1])
            if per_circuit and isinstance(group_key, tuple)
            else None
        )
        curves[(compound, circuit)] = _fit_curve(
            group["tyre_life"].to_numpy(dtype=float),
            group["fuel_corrected_time_s"].to_numpy(dtype=float),
            rng,
            n_bootstrap,
        )

    return DegradationModel(curves=curves, per_circuit=per_circuit)


def load_degradation_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    per_circuit: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DegradationModel:
    """Read the processed laps parquet and fit degradation curves."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed laps parquet not found: {path}")
    laps = pd.read_parquet(path)
    return fit_degradation(
        laps,
        per_circuit=per_circuit,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def plot_degradation_curves(
    model: DegradationModel,
    output_path: str | Path = DEFAULT_PLOT_PATH,
    compounds: Sequence[str] = DEFAULT_COMPOUNDS,
    circuits: Sequence[str] | None = None,
) -> Path:
    """Plot fitted curves and confidence bands, then return the saved path."""
    selected_circuits = tuple(circuits) if circuits is not None else model.circuits
    if not selected_circuits:
        raise ValueError("No circuits are available to plot")

    for circuit in selected_circuits:
        for compound in compounds:
            if (compound.upper(), circuit) not in model.curves and (
                compound.upper(),
                None,
            ) not in model.curves:
                raise ValueError(
                    f"No curve available for compound {compound!r} "
                    f"at circuit {circuit!r}"
                )

    figure, axes = plt.subplots(
        len(selected_circuits),
        1,
        figsize=(9, max(4, 3.5 * len(selected_circuits))),
        sharex=True,
        squeeze=False,
    )
    try:
        for axis, circuit in zip(axes[:, 0], selected_circuits):
            max_age = max(
                model.curves.get((compound.upper(), circuit),
                                 model.curves.get((compound.upper(), None))).max_tyre_life
                for compound in compounds
            )
            tyre_ages = np.linspace(0.0, max_age, 100)
            for compound in compounds:
                estimates = [
                    model.degradation(compound, age, circuit)
                    for age in tyre_ages
                ]
                axis.plot(
                    tyre_ages,
                    [estimate.expected_loss_s for estimate in estimates],
                    label=compound.upper(),
                )
                axis.fill_between(
                    tyre_ages,
                    [estimate.ci_lower_s for estimate in estimates],
                    [estimate.ci_upper_s for estimate in estimates],
                    alpha=0.18,
                )
            axis.set_title(circuit)
            axis.set_ylabel("Time loss vs fresh tyre (s)")
            axis.grid(alpha=0.25)
            axis.legend(title="Compound")

        axes[-1, 0].set_xlabel("Tyre life (laps)")
        figure.tight_layout()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)

    return path


def main() -> None:
    """Fit curves from the processed parquet and save the comparison plot."""
    model = load_degradation_model()
    plot_path = plot_degradation_curves(model)
    print(f"Saved degradation curves to {plot_path}")


if __name__ == "__main__":
    main()
