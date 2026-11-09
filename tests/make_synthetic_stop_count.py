"""Generate a separate synthetic dataset for A6 stop-count modelling."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_stop_count_laps.parquet"
)

CIRCUITS = [
    "Bahrain",
    "Monaco",
    "Silverstone",
    "Monza",
    "Singapore",
]

SEASONS = [2023, 2024, 2025]
DRIVERS = 20
LAPS_PER_RACE = 55


def generate_synthetic_stop_count_laps() -> pd.DataFrame:
    """Generate lap records containing all three pit-stop classes."""
    rows = []

    for season in SEASONS:
        for circuit_index, circuit in enumerate(CIRCUITS):
            for race_number in range(1, 4):
                round_number = (
                    circuit_index * 3 + race_number
                )

                race_name = (
                    f"{circuit} Grand Prix {season}"
                )

                for driver_index in range(DRIVERS):
                    driver = f"DRV{driver_index + 1:02d}"

                    team_number = driver_index // 2 + 1
                    team = f"Team {team_number:02d}"

                    grid_position = driver_index + 1

                    pattern = (
                        driver_index + circuit_index + race_number
                    ) % 10

                    if pattern <= 2:
                        stop_count = 3
                    elif pattern <= 6:
                        stop_count = 2
                    else:
                        stop_count = 1

                    pit_laps = [
                        10 + stop_index * (
                            40 // (stop_count + 1)
                        )
                        for stop_index in range(stop_count)
                    ]

                    for lap_number in range(
                        1, LAPS_PER_RACE + 1
                    ):
                        rows.append(
                            {
                                "season": season,
                                "round": round_number,
                                "race": race_name,
                                "circuit": circuit,
                                "driver": driver,
                                "team": team,
                                "lap_number": lap_number,
                                "lap_time_s": 90.0
                                + driver_index * 0.05,
                                "fuel_corrected_time_s": 89.0
                                + driver_index * 0.05,
                                "compound": "MEDIUM",
                                "tyre_life": lap_number,
                                "stint": 1,
                                "position": grid_position,
                                "grid_position": grid_position,
                                "is_inlap": lap_number in pit_laps,
                                "is_outlap": False,
                                "is_sc": False,
                                "is_wet": False,
                                "air_temp": 25.0,
                                "track_temp": 35.0,
                                "rainfall": 0.0,
                            }
                        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save the A6 synthetic lap dataset."""
    data = generate_synthetic_stop_count_laps()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Lap records: {len(data):,}")

    race_level = (
        data.groupby(
            ["season", "round", "race", "driver"]
        )["is_inlap"]
        .sum()
        .reset_index(name="pit_stops")
    )

    race_level["stop_count_class"] = (
        race_level["pit_stops"].clip(upper=3)
    )

    print("\nPit-stop class distribution:")
    print(
        race_level["stop_count_class"]
        .value_counts()
        .sort_index()
    )

    print("\nSeason distribution:")
    print(
        race_level.groupby("season")["driver"]
        .count()
    )


if __name__ == "__main__":
    main()