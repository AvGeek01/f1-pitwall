"""OpenF1 client and season fetcher (task A1).

Downloads race data from the OpenF1 REST API (historical data, 2023 onwards,
no authentication), caches every raw JSON response on disk, and builds one
parquet file per endpoint per season.

"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "openf1"
PARQUET_DIR = ROOT / "data" / "raw" / "openf1_parquet"

SUPPORTED_YEARS = range(2023, 2027)
MIN_INTERVAL_S = 2.5  
MAX_RETRIES = 5  
BACKOFF_BASE_S = 2.0 
REQUEST_TIMEOUT_S = 60

SESSION_ENDPOINTS: tuple[str, ...] = (
    "laps",
    "stints",
    "pit",
    "weather",
    "race_control",
    "position",
    "starting_grid",
    "drivers",
)

CORE_ENDPOINTS: tuple[str, ...] = ("laps", "stints", "drivers")
OPTIONAL_ENDPOINT = "intervals"


class OpenF1Error(RuntimeError):
    """Raised when an OpenF1 request cannot be completed."""


class OpenF1Client:
    """Polite OpenF1 client: throttled, retrying, and cached on disk.

    The clock, sleep function and HTTP session can be injected so tests run
    instantly and never touch the network.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        http: Any | None = None,
        min_interval_s: float = MIN_INTERVAL_S,
        max_retries: int = MAX_RETRIES,
        backoff_base_s: float = BACKOFF_BASE_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client. ``http`` needs a ``get(url, params=, timeout=)`` method."""
        self.raw_dir = Path(raw_dir)
        self.http = http if http is not None else requests.Session()
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        """Sleep until at least ``min_interval_s`` has passed since the last request."""
        if self._last_request_at is None:
            return
        wait = self.min_interval_s - (self._clock() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        """Return the exponential backoff wait (seconds) for a 0-based attempt."""
        return self.backoff_base_s * (2**attempt)

    @staticmethod
    def _retry_after(response: Any) -> float:
        """Return the server's ``Retry-After`` seconds, or 0 if absent or invalid."""
        try:
            return max(0.0, float(response.headers.get("Retry-After", 0)))
        except (TypeError, ValueError):
            return 0.0

    def get_json(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET ``/{endpoint}`` and return the decoded JSON list.

        Retries with exponential backoff on HTTP 429, 5xx and network errors.
        A 404 is treated as "no results" and returns an empty list. Any other
        4xx, or running out of retries, raises :class:`OpenF1Error`. Every
        request is logged.
        """
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            started = self._clock()
            self._last_request_at = started
            try:
                response = self.http.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                log.warning("GET %s %s -> network error: %s", url, params, exc)
                if attempt == self.max_retries:
                    raise OpenF1Error(f"{endpoint} {params}: network error: {exc}") from exc
                self._sleep(self._backoff(attempt))
                continue

            status = response.status_code
            log.info(
                "GET %s %s -> HTTP %s (%.1fs, attempt %d)",
                url, params, status, self._clock() - started, attempt + 1,
            )
            if status == 429 or status >= 500:
                if attempt == self.max_retries:
                    raise OpenF1Error(
                        f"{endpoint} {params}: HTTP {status} after {attempt + 1} attempts"
                    )
                wait = max(self._backoff(attempt), self._retry_after(response))
                log.warning("HTTP %s from %s, retrying in %.0fs", status, endpoint, wait)
                self._sleep(wait)
                continue
            if status == 404:
                log.warning("No results for %s %s", endpoint, params)
                return []
            if status >= 400:
                raise OpenF1Error(f"{endpoint} {params}: HTTP {status}: {response.text[:200]}")
            try:
                data = response.json()
            except ValueError as exc:
                raise OpenF1Error(f"{endpoint} {params}: response is not JSON") from exc
            if not isinstance(data, list):
                raise OpenF1Error(f"{endpoint} {params}: unexpected payload {str(data)[:200]}")
            return data
        raise OpenF1Error(f"{endpoint} {params}: retries exhausted")

    def cache_path(self, endpoint: str, key: str | int) -> Path:
        """Return ``raw_dir/{endpoint}/{key}.json``."""
        return self.raw_dir / endpoint / f"{key}.json"

    def fetch_cached(
        self,
        endpoint: str,
        key: str | int,
        params: dict[str, Any],
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the cached JSON for ``endpoint/key``, requesting it only if needed.

        An existing cache file is reused unless ``force`` is True. Empty
        responses are never cached, so a data-ingestion delay on OpenF1's side
        is not remembered forever.
        """
        path = self.cache_path(endpoint, key)
        if path.exists() and not force:
            try:
                data = json.loads(path.read_text())
                log.info("cache hit %s", path)
                return data
            except json.JSONDecodeError:
                log.warning("corrupt cache file %s, requesting it again", path)
        data = self.get_json(endpoint, params)
        if data:
            self._write_cache(path, data)
        else:
            log.warning("empty response for %s/%s, not cached", endpoint, key)
        return data

    @staticmethod
    def _write_cache(path: Path, data: list[dict[str, Any]]) -> None:
        """Write JSON through a temp file so a crash never leaves a partial cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)


def assign_rounds(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep non-cancelled ``Race`` sessions, sort by start date, and number them 1..N.

    OpenF1 has no round field, so the round is the race's position in the
    season calendar.
    """
    races = [
        s for s in sessions if s.get("session_name") == "Race" and not s.get("is_cancelled")
    ]
    races.sort(key=lambda s: pd.Timestamp(s["date_start"]))
    return [{**s, "round": i} for i, s in enumerate(races, start=1)]


def finished_sessions(sessions: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Return the sessions whose ``date_end`` is not in the future."""
    cutoff = pd.Timestamp(now)
    return [s for s in sessions if pd.Timestamp(s["date_end"]) <= cutoff]


def tag_records(
    records: list[dict[str, Any]], season: int, round_number: int, session_key: int
) -> list[dict[str, Any]]:
    """Add ``season``, ``round`` and ``session_key`` to every record."""
    return [
        {**r, "season": season, "round": round_number, "session_key": session_key}
        for r in records
    ]


def to_frame(endpoint: str, records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn JSON records into a DataFrame that parquet can store.

    Columns that mix Python types (for example the intervals endpoint returns
    numbers and the text ``"+1 LAP"``) are stored as strings; for intervals,
    numeric ``interval_s`` and ``gap_to_leader_s`` columns are added.
    """
    df = pd.DataFrame.from_records(records)
    if endpoint == OPTIONAL_ENDPOINT:
        for col in ("interval", "gap_to_leader"):
            if col in df.columns:
                df[f"{col}_s"] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if df[col].dtype == object:
            kinds = {type(v) for v in df[col].dropna()}
            if len(kinds) > 1:
                df[col] = df[col].map(lambda v: None if v is None or v != v else str(v))
    return df


def _fetch_session(
    client: OpenF1Client,
    session_key: int,
    endpoints: list[str],
    force: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch every endpoint for one race session; raise if a core endpoint fails or is empty."""
    out: dict[str, list[dict[str, Any]]] = {}
    for endpoint in endpoints:
        try:
            records = client.fetch_cached(
                endpoint, session_key, {"session_key": session_key}, force=force
            )
        except OpenF1Error:
            if endpoint == OPTIONAL_ENDPOINT:
                log.warning("optional endpoint %s failed for session %s", endpoint, session_key)
                records = []
            else:
                raise
        if not records:
            if endpoint in CORE_ENDPOINTS:
                raise OpenF1Error(f"{endpoint} returned no data for session {session_key}")
            log.warning("%s is empty for session %s", endpoint, session_key)
        out[endpoint] = records
    return out


def fetch_season(
    year: int,
    force: bool = False,
    include_intervals: bool = False,
    rounds: list[int] | None = None,
    client: OpenF1Client | None = None,
    parquet_dir: Path = PARQUET_DIR,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Fetch one season and write one parquet per endpoint.

    Gets the race sessions and meetings, then one request per endpoint for each
    finished race. A race that fails is logged and skipped. Existing cache
    files are reused unless ``force`` is True. For the current year the
    calendar is always refreshed, because OpenF1 updates it daily.

    ``rounds`` restricts the run to specific rounds (for testing); per-race
    parquet files then get a ``_partial`` suffix so they are never mistaken
    for a full season. Returns a mapping of name to the parquet path written.
    """
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"year must be between {SUPPORTED_YEARS[0]} and {SUPPORTED_YEARS[-1]}")
    client = client or OpenF1Client()
    now = now or datetime.now(timezone.utc)
    refresh_calendar = force or year >= now.year

    sessions_raw = client.fetch_cached(
        "sessions", year, {"year": year, "session_name": "Race"}, force=refresh_calendar
    )
    meetings_raw = client.fetch_cached("meetings", year, {"year": year}, force=refresh_calendar)
    calendar = assign_rounds(sessions_raw)
    if not calendar:
        raise OpenF1Error(f"no race sessions found for {year}")

    meetings = {m["meeting_key"]: m for m in meetings_raw}
    calendar = [
        {
            **s,
            "season": year,
            "meeting_name": meetings.get(s["meeting_key"], {}).get("meeting_name"),
            "meeting_official_name": meetings.get(s["meeting_key"], {}).get(
                "meeting_official_name"
            ),
            "circuit_type": meetings.get(s["meeting_key"], {}).get("circuit_type"),
        }
        for s in calendar
    ]

    todo = finished_sessions(calendar, now)
    if rounds is not None:
        todo = [s for s in todo if s["round"] in set(rounds)]
    endpoints = list(SESSION_ENDPOINTS) + ([OPTIONAL_ENDPOINT] if include_intervals else [])
    log.info("Season %d: %d races in calendar, %d to fetch", year, len(calendar), len(todo))

    collected: dict[str, list[dict[str, Any]]] = {ep: [] for ep in endpoints}
    done: list[int] = []
    failed: list[int] = []
    for session in tqdm(todo, desc=f"OpenF1 {year}", unit="race"):
        key, round_number = session["session_key"], session["round"]
        try:
            data = _fetch_session(client, key, endpoints, force)
        except Exception as exc:
            failed.append(round_number)
            log.error("Skipping round %s (session_key=%s): %s", round_number, key, exc)
            continue
        for endpoint, records in data.items():
            collected[endpoint].extend(tag_records(records, year, round_number, key))
        done.append(round_number)
        log.info("Round %s done (%d/%d)", round_number, len(done) + len(failed), len(todo))

    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _save(name: str, frame: pd.DataFrame) -> None:
        path = parquet_dir / f"{name}_{year}.parquet"
        frame.to_parquet(path, index=False)
        written[name] = path
        log.info("Wrote %s (%d rows)", path, len(frame))

    _save("sessions", to_frame("sessions", calendar))
    round_by_meeting = {s["meeting_key"]: s["round"] for s in calendar}
    meetings_frame = to_frame("meetings", meetings_raw)
    meetings_frame["season"] = year
    meetings_frame["round"] = pd.array(
        [round_by_meeting.get(k) for k in meetings_frame["meeting_key"]], dtype="Int64"
    )
    _save("meetings", meetings_frame)

    suffix = "_partial" if rounds is not None else ""
    if done:
        for endpoint in endpoints:
            _save(f"{endpoint}{suffix}", to_frame(endpoint, collected[endpoint]))
    else:
        log.error("Season %d: no races could be fetched, no per-race parquet written", year)

    if failed:
        log.warning("Season %d: skipped rounds %s. Re-run to retry them.", year, failed)
    log.info("Season %d: %d races fetched, %d skipped", year, len(done), len(failed))
    return written


def main() -> None:
    """Parse command-line arguments and fetch the requested seasons."""
    parser = argparse.ArgumentParser(description="Download OpenF1 race data.")
    parser.add_argument("years", nargs="+", type=int, help="e.g. 2023 2024 2025")
    parser.add_argument("--round", type=int, nargs="+", dest="rounds",
                        help="only these round numbers (test mode)")
    parser.add_argument("--force", action="store_true", help="ignore the on-disk cache")
    parser.add_argument("--intervals", action="store_true", help="also fetch /intervals")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for year in args.years:
        fetch_season(
            year, force=args.force, include_intervals=args.intervals, rounds=args.rounds
        )


if __name__ == "__main__":
    main()