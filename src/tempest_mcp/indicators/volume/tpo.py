"""TPO (Time-Price Opportunity) Indicator Engine.

TPO charts organize time spent at each price level, revealing where the market
spent the most time during a trading session. This implementation produces a
deterministic market-profile row surface with POC, value-area, and Initial
Balance metadata.

The engine operates on already-windowed single-session OHLCV data and does not
resolve sessions, timezones, or market-data windows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal

import pandas as pd

# Column names for output DataFrame (exact order as per contract)
COL_ROW_LOW = "row_low"
COL_ROW_HIGH = "row_high"
COL_ROW_MID = "row_mid"
COL_TPO_COUNT = "tpo_count"
COL_PERIOD_MARKERS = "period_markers"
COL_PERIOD_COUNT = "period_count"
COL_IN_VALUE_AREA = "in_value_area"

# Default marker sequence: A-Z, a-z, 0-9
_DEFAULT_MARKERS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(c) for c in range(10)]
)

MAX_TPO_PERIODS = 1000
MAX_TPO_ROWS = 10_000
MAX_TPO_TOUCH_OPERATIONS = 1_000_000


def _coerce_finite_numeric_series(series: pd.Series, column_name: str) -> pd.Series:
    """Coerce a Series to numeric and require all values to be finite."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError(f"OHLCV {column_name} values must be finite numbers")
    return numeric


def _validate_ohlcv(ohlcv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Validate OHLCV DataFrame input.

    Args:
        ohlcv: DataFrame with columns open, high, low, close, volume and UTC-aware index.

    Returns:
        Tuple of validated (high, low) numeric Series.

    Raises:
        ValueError: If validation fails.
    """
    if ohlcv.empty:
        raise ValueError("OHLCV DataFrame must not be empty")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV must contain columns: {required_columns}. Missing: {missing}")

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a DatetimeIndex")

    if ohlcv.index.tz is None:
        raise ValueError("OHLCV DatetimeIndex must be UTC-aware (tz_localize or tz_convert)")

    if ohlcv.index.has_duplicates:
        raise ValueError("OHLCV DatetimeIndex must not contain duplicate values")

    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("OHLCV DatetimeIndex must be monotonic increasing")

    high = _coerce_finite_numeric_series(ohlcv["high"], "high")
    low = _coerce_finite_numeric_series(ohlcv["low"], "low")

    if (high < low).any():
        raise ValueError("OHLCV high values must be greater than or equal to low values")

    return high, low


def _validate_row_size(row_size: float) -> float:
    """Validate row_size parameter.

    Args:
        row_size: The price increment for each row bucket.

    Returns:
        Validated row size as a float.

    Raises:
        ValueError: If validation fails.
    """
    try:
        row_size_value = float(row_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_size must be a positive finite number") from exc

    if not math.isfinite(row_size_value) or row_size_value <= 0:
        raise ValueError("row_size must be a positive finite number")

    return row_size_value


def _validate_value_area_pct(value_area_pct: float) -> None:
    """Validate value_area_pct parameter.

    Args:
        value_area_pct: The target percentage of total TPOs in the value area.

    Raises:
        ValueError: If validation fails.
    """
    if not (0 < value_area_pct <= 1):
        raise ValueError("value_area_pct must be in range (0, 1]")


def _validate_markers(markers: Sequence[str] | None, required_count: int) -> list[str]:
    """Validate and prepare marker sequence.

    Args:
        markers: Optional custom marker sequence.
        required_count: Number of periods that need markers.

    Returns:
        List of marker strings to use.

    Raises:
        ValueError: If validation fails.
    """
    if markers is None:
        # Use default markers, checking capacity
        if required_count > len(_DEFAULT_MARKERS):
            raise ValueError(
                f"Session has {required_count} periods but only {len(_DEFAULT_MARKERS)} "
                "default markers available (A-Z, a-z, 0-9). Provide custom markers to extend capacity."
            )
        return list(_DEFAULT_MARKERS[:required_count])

    # Validate custom markers
    if len(markers) == 0:
        raise ValueError("Custom markers sequence must not be empty")

    marker_list = [str(marker) for marker in markers]

    if any(marker == "" for marker in marker_list):
        raise ValueError("Custom markers must not contain empty strings")

    # Check for duplicates after coercion to the final marker type.
    if len(set(marker_list)) != len(marker_list):
        raise ValueError("Custom markers must be unique")

    # Check capacity
    if len(marker_list) < required_count:
        raise ValueError(
            f"Session has {required_count} periods but only {len(marker_list)} "
            "custom markers provided. Marker count must be >= period count."
        )

    return marker_list[:required_count]


def _validate_tpo_dimensions(n_periods: int, n_rows: int) -> None:
    """Guard against pathological row/period combinations."""
    if n_periods > MAX_TPO_PERIODS:
        raise ValueError(
            f"TPO session has {n_periods} periods, exceeding safety limit of {MAX_TPO_PERIODS}"
        )

    if n_rows > MAX_TPO_ROWS:
        raise ValueError(
            f"TPO row lattice expands to {n_rows} rows, exceeding safety limit of {MAX_TPO_ROWS}; "
            "increase row_size or reduce price span"
        )

    estimated_operations = n_periods * n_rows
    if estimated_operations > MAX_TPO_TOUCH_OPERATIONS:
        raise ValueError(
            "TPO allocation would require "
            f"{estimated_operations} row-period checks, exceeding safety limit of "
            f"{MAX_TPO_TOUCH_OPERATIONS}"
        )


def _build_row_lattice(
    price_min: float,
    price_max: float,
    row_size: float,
) -> pd.Index:
    """Build the row lattice (price bucket edges) for the session.

    Args:
        price_min: Minimum price in the session.
        price_max: Maximum price in the session.
        row_size: The price increment for each row bucket.

    Returns:
        Index of row edge prices (length = n_rows + 1).
    """
    if price_min == price_max:
        return pd.Index([price_min, price_max])

    min_decimal = Decimal(str(price_min))
    max_decimal = Decimal(str(price_max))
    width_decimal = Decimal(str(row_size))
    span = max_decimal - min_decimal

    # Calculate number of full steps needed
    full_steps = int((span / width_decimal).to_integral_value(rounding=ROUND_FLOOR))

    # Build edges from floor to ceiling
    edges = [float(min_decimal + (width_decimal * step)) for step in range(full_steps + 1)]

    # Ensure the last edge is at least price_max
    if edges[-1] < price_max:
        edges.append(float(Decimal(str(edges[-1])) + width_decimal))

    return pd.Index(edges)


def _allocate_tpo_to_rows(
    low_vals: pd.Series,
    high_vals: pd.Series,
    row_edges: pd.Index,
    markers: list[str],
) -> list[dict]:
    """Allocate each period's TPO to the rows it touches.

    Args:
        low_vals: Series of period low prices.
        high_vals: Series of period high prices.
        row_edges: Sorted array of row boundary prices.
        markers: List of markers in chronological order.

    Returns:
        List of dicts, one per row, containing:
        - row_low, row_high, row_mid
        - period_markers: list of markers touching this row
        - tpo_count: count of unique periods touching this row
    """
    n_rows = len(row_edges) - 1
    n_periods = len(low_vals)

    # Initialize storage for each row
    row_markers: list[list[str]] = [[] for _ in range(n_rows)]

    for period_idx in range(n_periods):
        period_low = float(low_vals.iloc[period_idx])
        period_high = float(high_vals.iloc[period_idx])
        marker = markers[period_idx]

        if period_low == period_high:
            for row_idx in range(n_rows):
                row_low = float(row_edges[row_idx])
                row_high = float(row_edges[row_idx + 1])

                if row_low <= period_low <= row_high:
                    row_markers[row_idx].append(marker)
                    break
            continue

        # Find all rows that this period touches
        for row_idx in range(n_rows):
            row_low = float(row_edges[row_idx])
            row_high = float(row_edges[row_idx + 1])

            # Check for intersection: period_low < row_high AND period_high > row_low
            # Using inclusive boundaries as per design contract
            if period_low < row_high and period_high > row_low:
                row_markers[row_idx].append(marker)

    # Build result rows
    result = []
    for row_idx in range(n_rows):
        row_low = float(row_edges[row_idx])
        row_high = float(row_edges[row_idx + 1])
        row_mid = (row_low + row_high) / 2
        period_markers = row_markers[row_idx]
        tpo_count = len(set(period_markers))  # Unique period touches

        result.append(
            {
                COL_ROW_LOW: row_low,
                COL_ROW_HIGH: row_high,
                COL_ROW_MID: row_mid,
                COL_TPO_COUNT: tpo_count,
                COL_PERIOD_MARKERS: period_markers,
                COL_PERIOD_COUNT: len(period_markers),
                COL_IN_VALUE_AREA: False,  # Will be set after VA calculation
            }
        )

    return result


def _calculate_poc_and_value_area(
    rows: list[dict],
    value_area_pct: float,
) -> tuple[float, int, float, float]:
    """Calculate POC and Value Area bounds.

    Args:
        rows: List of row dictionaries from _allocate_tpo_to_rows.
        value_area_pct: Target percentage of TPOs in value area.

    Returns:
        Tuple of (poc_price, poc_row_idx, vah_price, val_price).
    """
    if not rows:
        return 0.0, 0, 0.0, 0.0

    # Find POC: row with highest tpo_count; tie-breaker is lower row_idx (lower price)
    max_tpo = 0
    poc_row_idx = 0
    for idx, row in enumerate(rows):
        tpo = row[COL_TPO_COUNT]
        if tpo > max_tpo or (tpo == max_tpo and idx < poc_row_idx):
            max_tpo = tpo
            poc_row_idx = idx

    poc_price = rows[poc_row_idx][COL_ROW_MID]

    # Calculate value area by expanding from POC
    # Note: target_tpo not needed here as _find_va_bounds computes it internally

    # Find VA bounds
    va_low_idx, va_high_idx = _find_va_bounds(rows, poc_row_idx, value_area_pct)

    val_price = rows[va_low_idx][COL_ROW_LOW]
    vah_price = rows[va_high_idx][COL_ROW_HIGH]

    return poc_price, poc_row_idx, vah_price, val_price


def _calculate_initial_balance_and_range_expansion(
    high_vals: pd.Series,
    low_vals: pd.Series,
) -> tuple[float, float, bool, bool]:
    """Calculate Initial Balance and range expansion status.

    Initial Balance is computed from the first period only.
    Range expansion is true if any later period extends beyond IB.

    Args:
        high_vals: Series of period high prices.
        low_vals: Series of period low prices.

    Returns:
        Tuple of (ib_low, ib_high, range_expanded_up, range_expanded_down).
    """
    if len(high_vals) == 0 or len(low_vals) == 0:
        return 0.0, 0.0, False, False

    # Initial Balance from first period only
    ib_low = float(low_vals.iloc[0])
    ib_high = float(high_vals.iloc[0])

    # Check for range expansion in subsequent periods
    range_expanded_up = False
    range_expanded_down = False

    for i in range(1, len(high_vals)):
        period_high = float(high_vals.iloc[i])
        period_low = float(low_vals.iloc[i])

        if period_high > ib_high:
            range_expanded_up = True
        if period_low < ib_low:
            range_expanded_down = True

        # Early exit if both expansions are detected
        if range_expanded_up and range_expanded_down:
            break

    return ib_low, ib_high, range_expanded_up, range_expanded_down


def _find_va_bounds(
    rows: list[dict],
    poc_row_idx: int,
    value_area_pct: float,
) -> tuple[int, int]:
    """Find both Value Area bounds.

    Args:
        rows: List of row dictionaries.
        poc_row_idx: Index of the POC row.
        value_area_pct: Target percentage of TPOs in value area.

    Returns:
        Tuple of (va_low_idx, va_high_idx).
    """
    total_tpo = sum(row[COL_TPO_COUNT] for row in rows)
    target_tpo = total_tpo * value_area_pct

    va_low_idx = poc_row_idx
    va_high_idx = poc_row_idx
    current_tpo = rows[poc_row_idx][COL_TPO_COUNT]

    while current_tpo < target_tpo and (va_low_idx > 0 or va_high_idx < len(rows) - 1):
        left_add = rows[va_low_idx - 1][COL_TPO_COUNT] if va_low_idx > 0 else float("inf")
        right_add = (
            rows[va_high_idx + 1][COL_TPO_COUNT] if va_high_idx < len(rows) - 1 else float("inf")
        )

        if left_add >= float("inf") and right_add >= float("inf"):
            break

        # Prefer the larger adjacent count; tie-breaker favors lower-price side.
        if left_add >= right_add and va_low_idx > 0:
            va_low_idx -= 1
            current_tpo += left_add
        elif right_add > left_add and va_high_idx < len(rows) - 1:
            va_high_idx += 1
            current_tpo += right_add
        elif va_low_idx > 0:
            va_low_idx -= 1
            current_tpo += left_add
        elif va_high_idx < len(rows) - 1:
            va_high_idx += 1
            current_tpo += right_add

    return va_low_idx, va_high_idx


def calculate_tpo_chart(
    ohlcv: pd.DataFrame,
    row_size: float,
    value_area_pct: float = 0.70,
    markers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate TPO (Time-Price Opportunity) chart for single-session OHLCV.

    TPO charts organize time spent at each price level, revealing where the market
    spent the most time during a trading session. Key levels identified:
    - POC (Point of Control): Row with highest TPO count
    - VAH/VAL (Value Area High/Low): Boundaries covering value_area_pct of TPOs

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume] and
            UTC-aware DatetimeIndex. Must be non-empty, monotonic, non-duplicated.
            Already windowed to a single session by the caller.
        row_size: Required positive finite price increment for row buckets.
            Caller/owner determines appropriate tick size.
        value_area_pct: Percentage of total TPOs for Value Area (default 0.70).
        markers: Optional deterministic marker sequence override.
            Must be unique and contain at least as many markers as input periods.

    Returns:
        pd.DataFrame with exact columns in order:
            - row_low: Lower boundary of price row
            - row_high: Upper boundary of price row
            - row_mid: Midpoint of row (price level)
            - tpo_count: Count of unique periods touching this row
            - period_markers: List of marker strings for this row (chronological)
            - period_count: Total period touches for this row
            - in_value_area: True if row is within Value Area

        DataFrame.attrs with required metadata:
            - row_size: The configured row size
            - marker_count: Number of markers used
            - poc_price: Point of Control price
            - poc_row_idx: Index of POC row
            - vah_price: Value Area High price
            - val_price: Value Area Low price
            - initial_balance_low: First period low
            - initial_balance_high: First period high
            - range_expanded_up: True if any period extends above IB high
            - range_expanded_down: True if any period extends below IB low

    Raises:
        ValueError: If ohlcv is empty or missing required columns.
        ValueError: If row_size is not positive and finite.
        ValueError: If value_area_pct is not in (0, 1].
        ValueError: If markers are invalid (empty, duplicates, insufficient).
        TypeError: If ohlcv index is not UTC-aware DatetimeIndex.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> n = 20
        >>> idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        >>> data = pd.DataFrame({
        ...     'open': np.random.uniform(100, 110, n),
        ...     'high': np.random.uniform(105, 115, n),
        ...     'low': np.random.uniform(95, 105, n),
        ...     'close': np.random.uniform(100, 110, n),
        ...     'volume': np.random.uniform(1000, 5000, n)
        ... }, index=idx)
        >>> tpo = calculate_tpo_chart(data, row_size=1.0)
        >>> print(tpo[tpo['in_value_area']])
    """
    # Validate inputs
    high, low = _validate_ohlcv(ohlcv)
    row_size = _validate_row_size(row_size)
    _validate_value_area_pct(value_area_pct)

    n_periods = len(ohlcv)
    marker_list = _validate_markers(markers, n_periods)

    # Calculate price range
    price_min = float(low.min())
    price_max = float(high.max())

    # Build row lattice
    row_edges = _build_row_lattice(price_min, price_max, row_size)
    _validate_tpo_dimensions(n_periods=n_periods, n_rows=len(row_edges) - 1)

    # Allocate TPOs to rows
    rows = _allocate_tpo_to_rows(low, high, row_edges, marker_list)

    # Calculate POC and Value Area
    poc_price, poc_row_idx, vah_price, val_price = _calculate_poc_and_value_area(
        rows, value_area_pct
    )

    # Find VA bounds for marking
    va_low_idx, va_high_idx = _find_va_bounds(rows, poc_row_idx, value_area_pct)

    # Build result DataFrame
    result = pd.DataFrame(rows)

    # Ensure exact column order
    result = result[
        [
            COL_ROW_LOW,
            COL_ROW_HIGH,
            COL_ROW_MID,
            COL_TPO_COUNT,
            COL_PERIOD_MARKERS,
            COL_PERIOD_COUNT,
            COL_IN_VALUE_AREA,
        ]
    ]

    # Mark value area
    result[COL_IN_VALUE_AREA] = (result.index >= va_low_idx) & (result.index <= va_high_idx)

    # Calculate Initial Balance and range expansion
    ib_low, ib_high, range_expanded_up, range_expanded_down = (
        _calculate_initial_balance_and_range_expansion(high, low)
    )

    # Attach metadata to DataFrame.attrs
    result.attrs["row_size"] = row_size
    result.attrs["marker_count"] = len(marker_list)
    result.attrs["poc_price"] = poc_price
    result.attrs["poc_row_idx"] = poc_row_idx
    result.attrs["vah_price"] = vah_price
    result.attrs["val_price"] = val_price
    result.attrs["initial_balance_low"] = ib_low
    result.attrs["initial_balance_high"] = ib_high
    result.attrs["range_expanded_up"] = range_expanded_up
    result.attrs["range_expanded_down"] = range_expanded_down

    return result


__all__ = [
    "calculate_tpo_chart",
    "COL_ROW_LOW",
    "COL_ROW_HIGH",
    "COL_ROW_MID",
    "COL_TPO_COUNT",
    "COL_PERIOD_MARKERS",
    "COL_PERIOD_COUNT",
    "COL_IN_VALUE_AREA",
]
