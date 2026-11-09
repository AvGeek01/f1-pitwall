"""Network-free smoke test for the end-to-end project pipeline."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from scripts.run_pipeline import _rebuild_cached_season, run_pipeline
from tests.make_synthetic import generate_synthetic_data


def test_pipeline_runs_end_to_end_on_synthetic_data_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run feature building, model fitting, degradation and validation offline."""
    def block_network(*args: Any, **kwargs: Any) -> None:
        """Fail the test if any network request is attempted."""
        raise AssertionError("Synthetic pipeline attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "send", block_network)
    laps, pit_loss = generate_synthetic_data(output_dir=tmp_path / "fixtures")

    results = run_pipeline(
        synthetic_data=(laps, pit_loss),
        smoke_test=True,
        root=tmp_path / "pipeline",
        n_validation_sims=10,
        max_validation_strategies=2,
    )

    outputs = results["outputs"]
    assert isinstance(outputs, dict)
    assert Path(outputs["laps"]).is_file()
    assert Path(outputs["pit_loss"]).is_file()
    assert Path(outputs["features"]).is_file()
    assert Path(outputs["degradation_model"]).is_file()
    assert Path(outputs["validation_plot"]).is_file()
    assert len(results["validation_summary"]) == 15
    assert (tmp_path / "pipeline" / "models" / "pit_lap_classifier_smoke.joblib").is_file()
    assert (tmp_path / "pipeline" / "models" / "pit_lap_regressor_smoke.joblib").is_file()
    assert not (tmp_path / "pipeline" / "data" / "raw").exists()


def test_skip_fetch_rebuilds_parquets_from_cached_openf1_json(
    tmp_path: Path,
) -> None:
    """Reconstruct a season's endpoint files without contacting OpenF1."""
    raw_directory = tmp_path / "data" / "raw" / "openf1"
    parquet_directory = tmp_path / "data" / "raw" / "openf1_parquet"

    def write_cache(endpoint: str, key: int, records: list[dict[str, object]]) -> None:
        """Write one cached endpoint response in the fetcher's layout."""
        cache_path = raw_directory / endpoint / f"{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(records), encoding="utf-8")

    write_cache(
        "sessions",
        2025,
        [
            {
                "session_key": 100,
                "meeting_key": 200,
                "session_name": "Race",
                "date_start": "2025-03-01T10:00:00+00:00",
                "date_end": "2025-03-01T12:00:00+00:00",
            },
            {
                "session_key": 101,
                "meeting_key": 201,
                "session_name": "Race",
                "date_start": "2027-03-01T10:00:00+00:00",
                "date_end": "2027-03-01T12:00:00+00:00",
            },
        ],
    )
    write_cache(
        "meetings",
        2025,
        [
            {"meeting_key": 200, "meeting_name": "Test Grand Prix"},
            {"meeting_key": 201, "meeting_name": "Future Grand Prix"},
        ],
    )
    for endpoint in ("laps", "stints", "drivers"):
        write_cache(
            endpoint,
            100,
            [{"session_key": 100, "driver_number": 1, "fixture": endpoint}],
        )

    outputs = _rebuild_cached_season(
        2025,
        raw_directory,
        parquet_directory,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert set(("sessions", "meetings", "laps", "stints", "drivers")).issubset(outputs)
    laps = pd.read_parquet(outputs["laps"])
    assert laps.loc[0, "season"] == 2025
    assert laps.loc[0, "round"] == 1
    assert laps.loc[0, "session_key"] == 100
