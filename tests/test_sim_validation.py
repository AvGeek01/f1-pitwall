"""Tests for held-out-season simulator validation."""

from pathlib import Path

import pandas as pd

from src.sim.validation import validate_simulator


def make_validation_laps() -> pd.DataFrame:
    """Create three seasons of two-driver, twelve-lap races."""
    rows: list[dict[str, object]] = []
    for season in (2023, 2024, 2025):
        for driver_number, driver in enumerate(("DRV01", "DRV02"), start=1):
            elapsed_time = 0.0
            for lap_number in range(1, 13):
                compound = "SOFT" if lap_number <= 6 else "MEDIUM"
                tyre_life = lap_number if lap_number <= 6 else lap_number - 6
                is_inlap = lap_number == 6
                lap_time = (
                    80.0
                    + 0.05 * tyre_life
                    + (0.25 if compound == "MEDIUM" else 0.0)
                    + (0.4 if driver_number == 2 else 0.0)
                )
                elapsed_time += lap_time
                rows.append(
                    {
                        "season": season,
                        "round": 1,
                        "race": f"Test Grand Prix {season}",
                        "circuit": "Test Circuit",
                        "driver": driver,
                        "lap_number": lap_number,
                        "lap_time_s": lap_time,
                        "fuel_corrected_time_s": lap_time - (12 - lap_number) * 0.03,
                        "compound": compound,
                        "tyre_life": tyre_life,
                        "position": driver_number,
                        "is_inlap": is_inlap,
                        "is_outlap": lap_number == 7,
                        "is_sc": False,
                        "is_wet": False,
                    }
                )
    return pd.DataFrame(rows)


def make_validation_pit_loss() -> pd.DataFrame:
    """Return one deterministic pit-loss estimate for the test circuit."""
    return pd.DataFrame(
        {
            "circuit": ["Test Circuit"],
            "median_pit_loss_s": [20.0],
            "std_pit_loss_s": [0.0],
        }
    )


def test_validate_simulator_returns_race_summary_and_plot(tmp_path: Path) -> None:
    """Validate all races, restrict training seasons, and save the plot."""
    output_path = tmp_path / "figures" / "validation.png"
    summary = validate_simulator(
        2025,
        laps_data=make_validation_laps(),
        pit_loss_data=make_validation_pit_loss(),
        n_sims=20,
        min_stint_laps=2,
        timing_step=1,
        max_strategies_per_stop_count=5,
        n_bootstrap=20,
        output_path=output_path,
    )

    assert len(summary) == 1
    assert summary.loc[0, "season"] == 2025
    assert summary.loc[0, "strategies_evaluated"] == 15
    assert summary.loc[0, "simulated_stop_count"] in (1, 2, 3)
    assert summary.loc[0, "actual_stop_count"] == 1
    assert summary.attrs["training_seasons"] == (2023, 2024)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_validation_does_not_use_holdout_lap_times_for_simulation(
    tmp_path: Path,
) -> None:
    """Changing test-season pace must not change fitted simulated predictions."""
    laps = make_validation_laps()
    baseline = validate_simulator(
        2025,
        laps_data=laps,
        pit_loss_data=make_validation_pit_loss(),
        n_sims=20,
        min_stint_laps=2,
        timing_step=1,
        max_strategies_per_stop_count=5,
        n_bootstrap=20,
        output_path=tmp_path / "baseline.png",
    )
    changed_laps = laps.copy()
    holdout_rows = changed_laps["season"].eq(2025)
    changed_laps.loc[holdout_rows, "lap_time_s"] += 50.0
    changed_laps.loc[holdout_rows, "fuel_corrected_time_s"] += 50.0

    changed = validate_simulator(
        2025,
        laps_data=changed_laps,
        pit_loss_data=make_validation_pit_loss(),
        n_sims=20,
        min_stint_laps=2,
        timing_step=1,
        max_strategies_per_stop_count=5,
        n_bootstrap=20,
        output_path=tmp_path / "changed.png",
    )

    assert (
        baseline.loc[0, "simulated_mean_time_s"]
        == changed.loc[0, "simulated_mean_time_s"]
    )
    assert changed.loc[0, "actual_race_time_s"] > baseline.loc[0, "actual_race_time_s"]
