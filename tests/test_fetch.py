"""Tests for src/data/fetch.py. Every test runs with the network blocked."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from src.data.fetch import (
    SESSION_ENDPOINTS,
    OpenF1Client,
    OpenF1Error,
    assign_rounds,
    fetch_season,
    finished_sessions,
    to_frame,
)

NOW = datetime(2025, 1, 15, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to send a real HTTP request."""

    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted in a test")

    monkeypatch.setattr(requests.sessions.Session, "send", _blocked)


class FakeClock:
    """A clock whose ``sleep`` advances time instantly and records the wait."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status: int = 200, payload: Any = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Serves queued responses per (endpoint, key); records every call with its time."""

    def __init__(self, routes: dict[tuple[str, Any], list[Any]], clock: FakeClock) -> None:
        self.routes = routes
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None):
        params = params or {}
        endpoint = url.rsplit("/", 1)[-1]
        key = params.get("session_key", params.get("year"))
        self.calls.append((endpoint, dict(params), self.clock()))
        queue = self.routes.get((endpoint, key), [FakeResponse(404, {"detail": "No results"})])
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_client(tmp_path: Path, routes: dict, **kwargs: Any) -> tuple[OpenF1Client, FakeHttp, FakeClock]:
    """Build a client wired to fakes, with its cache under tmp_path."""
    clock = FakeClock()
    http = FakeHttp(routes, clock)
    client = OpenF1Client(
        raw_dir=tmp_path / "raw", http=http, clock=clock, sleep=clock.sleep, **kwargs
    )
    return client, http, clock


def test_requests_are_spaced_at_least_2_5_seconds(tmp_path: Path) -> None:
    routes = {("laps", 1): [FakeResponse(200, [{"a": 1}])], ("laps", 2): [FakeResponse(200, [{"a": 2}])]}
    client, http, _ = make_client(tmp_path, routes)
    client.get_json("laps", {"session_key": 1})
    client.get_json("laps", {"session_key": 2})
    times = [t for _, _, t in http.calls]
    assert times[1] - times[0] >= 2.5 - 1e-9


def test_retries_429_and_5xx_with_exponential_backoff(tmp_path: Path) -> None:
    routes = {("laps", 1): [FakeResponse(500), FakeResponse(429), FakeResponse(200, [{"ok": 1}])]}
    client, http, clock = make_client(tmp_path, routes)
    assert client.get_json("laps", {"session_key": 1}) == [{"ok": 1}]
    assert len(http.calls) == 3
    backoffs = [s for s in clock.sleeps if s >= 2.0]
    assert backoffs[:2] == [2.0, 4.0]


def test_retry_after_header_is_respected(tmp_path: Path) -> None:
    routes = {("laps", 1): [FakeResponse(429, headers={"Retry-After": "9"}), FakeResponse(200, [{"ok": 1}])]}
    client, _, clock = make_client(tmp_path, routes)
    client.get_json("laps", {"session_key": 1})
    assert 9.0 in clock.sleeps


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    client, http, _ = make_client(tmp_path, {("laps", 1): [FakeResponse(503)]}, max_retries=3)
    with pytest.raises(OpenF1Error):
        client.get_json("laps", {"session_key": 1})
    assert len(http.calls) == 4


def test_network_errors_are_retried(tmp_path: Path) -> None:
    routes = {("laps", 1): [requests.ConnectionError("boom"), FakeResponse(200, [{"ok": 1}])]}
    client, http, _ = make_client(tmp_path, routes)
    assert client.get_json("laps", {"session_key": 1}) == [{"ok": 1}]
    assert len(http.calls) == 2


def test_404_means_no_results_and_other_4xx_fail_fast(tmp_path: Path) -> None:
    routes = {("laps", 1): [FakeResponse(404, {"detail": "No results"})], ("laps", 2): [FakeResponse(422)]}
    client, http, _ = make_client(tmp_path, routes)
    assert client.get_json("laps", {"session_key": 1}) == []
    with pytest.raises(OpenF1Error):
        client.get_json("laps", {"session_key": 2})
    assert len(http.calls) == 2


def test_network_guard_blocks_real_requests() -> None:
    with pytest.raises(AssertionError):
        requests.Session().get("https://api.openf1.org/v1/sessions")


def test_cache_is_reused_and_force_refreshes(tmp_path: Path) -> None:
    routes = {("laps", 7): [FakeResponse(200, [{"x": 1}])]}
    client, http, _ = make_client(tmp_path, routes)
    assert client.fetch_cached("laps", 7, {"session_key": 7}) == [{"x": 1}]
    assert client.cache_path("laps", 7) == tmp_path / "raw" / "laps" / "7.json"
    assert client.cache_path("laps", 7).exists()
    client.fetch_cached("laps", 7, {"session_key": 7})
    assert len(http.calls) == 1
    client.fetch_cached("laps", 7, {"session_key": 7}, force=True)
    assert len(http.calls) == 2


def test_empty_responses_are_not_cached(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, {("laps", 7): [FakeResponse(200, [])]})
    assert client.fetch_cached("laps", 7, {"session_key": 7}) == []
    assert not client.cache_path("laps", 7).exists()


def test_corrupt_cache_file_is_requested_again(tmp_path: Path) -> None:
    client, http, _ = make_client(tmp_path, {("laps", 7): [FakeResponse(200, [{"x": 1}])]})
    path = client.cache_path("laps", 7)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert client.fetch_cached("laps", 7, {"session_key": 7}) == [{"x": 1}]
    assert len(http.calls) == 1


def _session(key: int, meeting: int, start: str, end: str, name: str = "Race", cancelled: bool = False):
    return {
        "session_key": key, "meeting_key": meeting, "session_name": name,
        "date_start": start, "date_end": end, "is_cancelled": cancelled,
        "circuit_short_name": f"C{meeting}", "country_name": f"Country{meeting}",
    }


CALENDAR = [
    _session(102, 2, "2024-03-09T15:00:00+00:00", "2024-03-09T17:00:00+00:00"),
    _session(101, 1, "2024-03-02T15:00:00+00:00", "2024-03-02T17:00:00+00:00"),
    _session(103, 3, "2024-03-16T15:00:00+00:00", "2024-03-16T17:00:00+00:00", cancelled=True),
    _session(104, 4, "2030-03-23T15:00:00+00:00", "2030-03-23T17:00:00+00:00"),
    _session(199, 1, "2024-03-01T15:00:00+00:00", "2024-03-01T16:00:00+00:00", name="Sprint"),
]


def test_assign_rounds_sorts_and_drops_sprints_and_cancelled() -> None:
    rounds = {s["session_key"]: s["round"] for s in assign_rounds(CALENDAR)}
    assert rounds == {101: 1, 102: 2, 104: 3}


def test_finished_sessions_excludes_future_races() -> None:
    keys = [s["session_key"] for s in finished_sessions(assign_rounds(CALENDAR), NOW)]
    assert keys == [101, 102]


def test_to_frame_stores_mixed_type_columns_as_strings() -> None:
    frame = to_frame("intervals", [
        {"interval": 0.5, "gap_to_leader": 1.2},
        {"interval": None, "gap_to_leader": "+1 LAP"},
    ])
    assert frame["interval_s"].iloc[0] == 0.5
    assert pd.isna(frame["gap_to_leader_s"].iloc[1])
    assert frame["gap_to_leader"].tolist() == ["1.2", "+1 LAP"]
    frame.to_parquet(None)


def _rows(endpoint: str, key: int) -> list[dict[str, Any]]:
    base = {"session_key": key, "meeting_key": key, "driver_number": 1}
    extra = {
        "laps": {"lap_number": 1, "lap_duration": 91.2, "is_pit_out_lap": False,
                 "date_start": "2024-03-02T15:01:00+00:00", "segments_sector_1": [2049, 2051]},
        "stints": {"stint_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": 20,
                   "tyre_age_at_start": 0},
        "pit": {"lap_number": 20, "lane_duration": 22.1},
        "weather": {"air_temperature": 25.0, "track_temperature": 40.0, "rainfall": 0},
        "race_control": {"category": "Flag", "flag": "GREEN", "lap_number": 1},
        "position": {"date": "2024-03-02T15:00:00+00:00", "position": 3},
        "starting_grid": {"position": 3},
        "drivers": {"name_acronym": "VER", "team_name": "Red Bull Racing"},
        "intervals": {"interval": 0.4, "gap_to_leader": "+1 LAP"},
    }[endpoint]
    return [{**base, **extra}]


def _routes(bad_laps: tuple[int, ...] = ()) -> dict[tuple[str, Any], list[Any]]:
    routes: dict[tuple[str, Any], list[Any]] = {
        ("sessions", 2024): [FakeResponse(200, CALENDAR)],
        ("meetings", 2024): [FakeResponse(200, [
            {"meeting_key": k, "meeting_name": f"GP {k}", "meeting_official_name": f"Official {k}",
             "circuit_type": "Permanent", "circuit_short_name": f"C{k}", "country_name": f"Country{k}"}
            for k in (1, 2, 3, 4)
        ])],
    }
    for key in (101, 102, 104):
        for endpoint in (*SESSION_ENDPOINTS, "intervals"):
            routes[(endpoint, key)] = [FakeResponse(200, _rows(endpoint, key))]
    for key in bad_laps:
        routes[("laps", key)] = [FakeResponse(500)]
    return routes


def test_fetch_season_writes_one_parquet_per_endpoint(tmp_path: Path) -> None:
    client, http, _ = make_client(tmp_path, _routes())
    written = fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=NOW)

    assert set(written) == {"sessions", "meetings", *SESSION_ENDPOINTS}
    assert all(p.exists() and p.name.endswith("_2024.parquet") for p in written.values())

    laps = pd.read_parquet(written["laps"])
    assert sorted(laps["round"].unique()) == [1, 2] 
    assert (laps["season"] == 2024).all()
    assert {"session_key", "lap_duration", "date_start"} <= set(laps.columns)

    sessions = pd.read_parquet(written["sessions"]).sort_values("round")
    assert sessions["round"].tolist() == [1, 2, 3]
    assert sessions["meeting_name"].tolist() == ["GP 1", "GP 2", "GP 4"]

    meetings = pd.read_parquet(written["meetings"]).set_index("meeting_key")
    assert pd.isna(meetings.loc[3, "round"])

    assert len(http.calls) == 2 + 2 * len(SESSION_ENDPOINTS)
    gaps = [b[2] - a[2] for a, b in zip(http.calls, http.calls[1:])]
    assert min(gaps) >= 2.5 - 1e-9


def test_failing_race_is_skipped_without_crashing(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, _routes(bad_laps=(102,)))
    written = fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=NOW)
    laps = pd.read_parquet(written["laps"])
    assert laps["round"].unique().tolist() == [1]
    assert client.cache_path("laps", 101).exists()
    assert not client.cache_path("laps", 102).exists()


def test_all_races_failing_still_returns_calendar_only(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, _routes(bad_laps=(101, 102)))
    written = fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=NOW)
    assert set(written) == {"sessions", "meetings"}


def test_second_run_uses_cache_and_force_refetches(tmp_path: Path) -> None:
    client, http, _ = make_client(tmp_path, _routes())
    fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=NOW)
    first = len(http.calls)
    fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=NOW)
    assert len(http.calls) == first
    fetch_season(2024, force=True, client=client, parquet_dir=tmp_path / "pq", now=NOW)
    assert len(http.calls) > first


def test_current_year_calendar_is_always_refreshed(tmp_path: Path) -> None:
    client, http, _ = make_client(tmp_path, _routes())
    same_year = datetime(2024, 12, 31, tzinfo=timezone.utc)
    fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=same_year)
    before = sum(1 for c in http.calls if c[0] == "sessions")
    fetch_season(2024, client=client, parquet_dir=tmp_path / "pq", now=same_year)
    assert sum(1 for c in http.calls if c[0] == "sessions") == before + 1


def test_intervals_flag_adds_numeric_columns_and_never_skips_a_race(tmp_path: Path) -> None:
    routes = _routes()
    routes[("intervals", 102)] = [FakeResponse(500)]
    client, _, _ = make_client(tmp_path, routes, max_retries=1)
    written = fetch_season(
        2024, include_intervals=True, client=client, parquet_dir=tmp_path / "pq", now=NOW
    )
    assert pd.read_parquet(written["laps"])["round"].nunique() == 2
    intervals = pd.read_parquet(written["intervals"])
    assert pd.isna(intervals["gap_to_leader_s"]).all()


def test_rounds_option_writes_partial_files(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, _routes())
    written = fetch_season(2024, rounds=[2], client=client, parquet_dir=tmp_path / "pq", now=NOW)
    assert written["laps_partial"].name == "laps_partial_2024.parquet"
    assert pd.read_parquet(written["laps_partial"])["round"].unique().tolist() == [2]


def test_unsupported_year_is_rejected(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, {})
    for year in (2022, 2027):
        with pytest.raises(ValueError):
            fetch_season(year, client=client, parquet_dir=tmp_path / "pq")