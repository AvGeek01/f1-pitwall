"""Generate deterministic synthetic lap and pit-loss data for development."""

from pathlib import Path

import numpy as np
import pandas as pd

LAPS_COLUMNS = [
    "season",
    "round",
    "race",
    "circuit",
    "driver",
    "team",
    "lap_number",
    "lap_time_s",
    "fuel_corrected_time_s",
    "compound",
    "tyre_life",
    "stint",
    "position",
    "grid_position",
    "is_inlap",
    "is_outlap",
    "is_sc",
    "is_wet",
    "air_temp",
    "track_temp",
    "rainfall",
]
PIT_LOSS_COLUMNS = [
    "circuit",
    "median_pit_loss_s",
    "std_pit_loss_s",
    "n_stops",
]

CIRCUITS = {
    "Bahrain": {"base_pace_s": 94.0, "pit_loss_s": 22.0},
    "Monaco": {"base_pace_s": 75.5, "pit_loss_s": 20.5},
    "Silverstone": {"base_pace_s": 89.5, "pit_loss_s": 24.0},
}
COMPOUND_OFFSETS_S = {"SOFT": 0.0, "MEDIUM": 0.35, "HARD": 0.75}
COMPOUND_DEGRADATION_S = {"SOFT": 0.075, "MEDIUM": 0.055, "HARD": 0.04}


def _stint_lengths(rng: np.random.Generator) -> list[int]:
    """Split 55 laps into two or three plausible tyre stints."""
    if rng.random() < 0.65:
        first_stint = int(rng.integers(22, 34))
        return [first_stint, 55 - first_stint]

    first_stint = int(rng.integers(14, 23))
    second_stint = int(rng.integers(14, 56 - first_stint - 9))
    return [first_stint, second_stint, 55 - first_stint - second_stint]


def generate_synthetic_data(
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and save synthetic lap and per-circuit pit-loss DataFrames.

    Each of three circuits has five races in each of the 2023, 2024, and 2025
    seasons. Every race has 20 drivers completing 55 laps. By default, Parquet
    files are written to the repository's ``data/processed`` directory.
    """
    rng = np.random.default_rng(seed)
    seasons = (2023, 2024, 2025)
    drivers = [f"DRV{number:02d}" for number in range(1, 21)]
    teams = {driver: f"Team {(index // 2) + 1:02d}" for index, driver in enumerate(drivers)}
    driver_pace = dict(zip(drivers, rng.normal(0.0, 0.38, len(drivers))))
    team_pace = {
        team: float(rng.normal(0.0, 0.16))
        for team in sorted(set(teams.values()))
    }

    lap_rows: list[dict[str, object]] = []
    pit_losses: dict[str, list[float]] = {circuit: [] for circuit in CIRCUITS}

    for season in seasons:
        for circuit_index, (circuit, circuit_data) in enumerate(CIRCUITS.items()):
            for race_index in range(1, 6):
                round_number = circuit_index * 5 + race_index
                race = f"{circuit} Grand Prix {race_index}"
                grid_order = rng.permutation(drivers)
                grid_positions = {
                    driver: int(index + 1) for index, driver in enumerate(grid_order)
                }

                is_sc = np.zeros(55, dtype=bool)
                if rng.random() < 0.75:
                    sc_start = int(rng.integers(8, 48))
                    sc_length = int(rng.integers(2, 5))
                    is_sc[sc_start : min(sc_start + sc_length, 55)] = True

                rainfall = np.zeros(55)
                if rng.random() < 0.25:
                    rain_start = int(rng.integers(5, 43))
                    rain_length = int(rng.integers(4, 13))
                    rain_end = min(rain_start + rain_length, 55)
                    rainfall[rain_start:rain_end] = rng.uniform(
                        0.1, 1.0, rain_end - rain_start
                    )

                air_temps = rng.normal(24.0, 2.5, 55)
                track_temps = air_temps + rng.normal(10.0, 3.0, 55)
                track_temps[rainfall > 0] -= rng.uniform(2.0, 6.0)

                for driver in drivers:
                    stint_lengths = _stint_lengths(rng)
                    compounds = rng.choice(
                        ("SOFT", "MEDIUM", "HARD"),
                        size=len(stint_lengths),
                        p=(0.35, 0.4, 0.25),
                    )
                    lap_number = 0
                    elapsed_race_time = 0.0
                    base_pace = (
                        float(circuit_data["base_pace_s"])
                        - 0.08 * (season - seasons[0])
                        + driver_pace[driver]
                        + team_pace[teams[driver]]
                    )

                    for stint_number, (stint_length, compound) in enumerate(
                        zip(stint_lengths, compounds), start=1
                    ):
                        for tyre_life in range(1, stint_length + 1):
                            lap_number += 1
                            lap_index = lap_number - 1
                            is_inlap = tyre_life == stint_length and stint_number < len(
                                stint_lengths
                            )
                            is_outlap = tyre_life == 1 and stint_number > 1
                            wet = rainfall[lap_index] > 0
                            compound_name = str(compound)
                            degradation = (
                                COMPOUND_DEGRADATION_S[compound_name] * tyre_life
                            )
                            lap_time = (
                                base_pace
                                + COMPOUND_OFFSETS_S[compound_name]
                                + degradation
                                + rng.normal(0.0, 0.22)
                                + (1.8 if wet else 0.0)
                                + (12.0 if is_sc[lap_index] else 0.0)
                                + (1.2 if is_outlap else 0.0)
                                + (0.8 if is_inlap else 0.0)
                            )
                            lap_time = round(float(lap_time), 3)
                            elapsed_race_time += lap_time
                            fuel_effect_s = (55 - lap_number) * 0.03

                            lap_rows.append(
                                {
                                    "season": season,
                                    "round": round_number,
                                    "race": race,
                                    "circuit": circuit,
                                    "driver": driver,
                                    "team": teams[driver],
                                    "lap_number": lap_number,
                                    "lap_time_s": lap_time,
                                    "fuel_corrected_time_s": round(
                                        lap_time - fuel_effect_s, 3
                                    ),
                                    "compound": compound_name,
                                    "tyre_life": tyre_life,
                                    "stint": stint_number,
                                    "position": 0,
                                    "grid_position": grid_positions[driver],
                                    "is_inlap": is_inlap,
                                    "is_outlap": is_outlap,
                                    "is_sc": bool(is_sc[lap_index]),
                                    "is_wet": bool(wet),
                                    "air_temp": round(float(air_temps[lap_index]), 1),
                                    "track_temp": round(
                                        float(track_temps[lap_index]), 1
                                    ),
                                    "rainfall": round(float(rainfall[lap_index]), 2),
                                    "_elapsed_race_time_s": elapsed_race_time,
                                }
                            )

                        if stint_number < len(stint_lengths):
                            pit_losses[circuit].append(
                                float(
                                    rng.normal(
                                        float(circuit_data["pit_loss_s"]), 1.8
                                    )
                                )
                            )

    laps_clean = pd.DataFrame(lap_rows)
    position_groups = ["season", "round", "lap_number"]
    laps_clean["position"] = (
        laps_clean.groupby(position_groups)["_elapsed_race_time_s"]
        .rank(method="first")
        .astype(int)
    )
    laps_clean = laps_clean[LAPS_COLUMNS]

    pit_loss = pd.DataFrame(
        [
            {
                "circuit": circuit,
                "median_pit_loss_s": round(float(np.median(losses)), 3),
                "std_pit_loss_s": round(float(np.std(losses, ddof=1)), 3),
                "n_stops": len(losses),
            }
            for circuit, losses in pit_losses.items()
        ],
        columns=PIT_LOSS_COLUMNS,
    )

    if output_dir is None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "processed"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    laps_clean.to_parquet(output_path / "synthetic_laps_clean.parquet", index=False)
    pit_loss.to_parquet(output_path / "synthetic_pit_loss.parquet", index=False)

    return laps_clean, pit_loss


if __name__ == "__main__":
    laps, pit_losses = generate_synthetic_data()
    print(f"Generated {len(laps):,} laps and pit-loss data for {len(pit_losses)} circuits.")
