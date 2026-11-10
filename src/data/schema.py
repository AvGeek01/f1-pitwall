"""Shared schema and quality rules for processed race data."""

from __future__ import annotations

import pandas as pd

MAX_PLAUSIBLE_LAP_DURATION_S = 300.0
MIN_PLAUSIBLE_LAP_DURATION_S = 30.0
MIN_PLAUSIBLE_TIMESTAMP_FALLBACK_S = 30.0
PIT_LOSS_COLUMNS = (
    "circuit",
    "median_pit_loss_s",
    "std_pit_loss_s",
    "n_stops",
)

PIT_LOSS_LEGACY_COLUMNS = {
    "pit_loss_median_s": "median_pit_loss_s",
    "pit_loss_std_s": "std_pit_loss_s",
}


def normalize_pit_loss_schema(pit_loss: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy pit-loss names normalized to canonical names.

    If both a canonical and legacy column are present, non-missing canonical
    values take precedence and missing values are filled from the legacy
    column. Legacy columns are then removed.
    """
    normalized = pit_loss.copy()
    for legacy_name, canonical_name in PIT_LOSS_LEGACY_COLUMNS.items():
        if legacy_name not in normalized.columns:
            continue
        if canonical_name in normalized.columns:
            normalized[canonical_name] = normalized[canonical_name].combine_first(
                normalized[legacy_name]
            )
            normalized = normalized.drop(columns=legacy_name)
        else:
            normalized = normalized.rename(
                columns={legacy_name: canonical_name}
            )
    return normalized
