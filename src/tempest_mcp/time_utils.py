"""Shared timezone helpers for business-window handling.

Tempest keeps canonical timestamps in UTC internally, but interprets business
window inputs in America/New_York by default unless the caller provides an
explicit timezone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

BUSINESS_TZ_NAME = "America/New_York"
BUSINESS_TZ = ZoneInfo(BUSINESS_TZ_NAME)
UTC_TZ = timezone.utc


def coerce_window_datetime_to_utc(dt: datetime | None) -> datetime | None:
    """Convert a window datetime to UTC.

    Naive datetimes are interpreted in the default business timezone
    (America/New_York), then converted to UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BUSINESS_TZ)
    return dt.astimezone(UTC_TZ)


def ensure_utc_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return a UTC-aware pandas timestamp."""
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def business_day_bounds_utc(
    reference_timestamp: pd.Timestamp,
    *,
    day_offset: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get UTC bounds for the ET business day around a reference timestamp.

    Args:
        reference_timestamp: Timestamp used to identify the target business day.
        day_offset: Relative day offset in business timezone. ``0`` returns the
            ET day containing ``reference_timestamp``; ``-1`` returns the
            preceding ET day.
    """
    reference_utc = ensure_utc_timestamp(reference_timestamp)
    reference_business = reference_utc.tz_convert(BUSINESS_TZ)
    day_start_business = reference_business.normalize() + pd.Timedelta(days=day_offset)
    day_end_business = day_start_business + pd.Timedelta(days=1)
    return day_start_business.tz_convert("UTC"), day_end_business.tz_convert("UTC")
