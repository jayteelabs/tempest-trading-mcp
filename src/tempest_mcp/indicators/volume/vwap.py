"""VWAP Indicator Engine - Volume Weighted Average Price calculations.

Implements VWAP calculations with session anchoring for different trading sessions
(Asia, London, NY, Daily). VWAP resets at the start of each session based on UTC time.

VWAP Formula:
    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3

Session Anchors (UTC times - NO DST):
    - 'asia': 00:00 UTC
    - 'london': 08:00 UTC
    - 'ny': 13:30 UTC (NYSE open, DEFAULT)
    - 'daily': 00:00 UTC (calendar day)

Pre-first-anchor behavior: accumulates from bar 0, resets at first anchor boundary.
"""

import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Session anchor times in UTC (NO DST adjustments)
SESSION_ANCHORS = {
    "asia": 0,  # 00:00 UTC
    "london": 8,  # 08:00 UTC
    "ny": 13.5,  # 13:30 UTC (NYSE open)
    "daily": 0,  # 00:00 UTC (calendar day)
}


def _get_session_anchor_hour(anchor: str) -> float:
    """Get the UTC hour for a session anchor.

    Args:
        anchor: Session anchor type ('asia', 'london', 'ny', 'daily')

    Returns:
        UTC hour as float (e.g., 13.5 for 13:30)

    Raises:
        ValueError: If anchor type is not recognized
    """
    if anchor not in SESSION_ANCHORS:
        raise ValueError(
            f"Invalid anchor '{anchor}'. Must be one of: {list(SESSION_ANCHORS.keys())}"
        )
    return SESSION_ANCHORS[anchor]


def _ensure_utc_index(series: pd.Series) -> pd.Series:
    """Ensure the series has a UTC-aware DatetimeIndex.

    Args:
        series: pd.Series with potential tz-naive DatetimeIndex

    Returns:
        pd.Series with UTC-aware DatetimeIndex
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex")

    if series.index.tz is None:
        # Treat as UTC if tz-naive
        series = series.copy()
        series.index = series.index.tz_localize("UTC")

    return series


def _calculate_session_groups(dates: pd.DatetimeIndex, anchor_hour: float) -> pd.Series:
    """Calculate session group identifiers for VWAP accumulation.

    Each session is identified by a unique group ID. VWAP resets at each
    new session group.

    A session runs from anchor_hour on day D to just before anchor_hour on day D+1.
    For example, with 'ny' (13:30 UTC), the session runs from 13:30 UTC to the
    next day's 13:30 UTC.

    Args:
        dates: DatetimeIndex of timestamps (UTC-aware)
        anchor_hour: UTC hour for session anchor

    Returns:
        pd.Series of integers representing session groups
    """
    # Convert to pandas Series for easier manipulation
    dates_series = pd.Series(dates)

    # Ensure UTC
    if dates.tz is None:
        dates_series = dates_series.dt.tz_localize("UTC")
    else:
        dates_series = dates_series.dt.tz_convert("UTC")

    # Extract hour as decimal
    hours = dates_series.dt.hour + dates_series.dt.minute / 60.0

    # Get calendar date
    date_only = dates_series.dt.date

    # Adjust date based on whether we're before or after anchor hour
    # If before anchor hour, we're in the previous session
    # If at or after anchor hour, we're in the current session
    adjusted_date = date_only.copy()

    # Subtract 1 day from dates that are before the anchor hour
    # This puts them in the previous session
    mask_before_anchor = hours < anchor_hour
    adjusted_date[mask_before_anchor] = date_only[mask_before_anchor] - pd.Timedelta(days=1)

    # Convert adjusted_date to group IDs
    # Use year*100000 + month*1000 + day*10 for unique grouping
    adjusted_series = pd.Series(adjusted_date)
    adjusted_datetime = pd.to_datetime(adjusted_series)

    # Create session ID from adjusted date
    session_id = (
        adjusted_datetime.dt.year * 100000
        + adjusted_datetime.dt.month * 1000
        + adjusted_datetime.dt.day * 10
    )

    return session_id


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    anchor: str = "ny",
) -> pd.Series:
    """Calculate Volume Weighted Average Price with session anchoring.

    VWAP = cumulative(typical_price × volume) / cumulative(volume), reset at session anchor.

    Session anchors (UTC times - NO DST):
        - 'asia': 00:00 UTC
        - 'london': 08:00 UTC
        - 'ny': 13:30 UTC (DEFAULT)
        - 'daily': 00:00 UTC (calendar day)

    Pre-first-anchor: accumulates from bar 0, resets at first anchor boundary encountered.
    After first reset: resets at defined UTC times each trading day.

    Args:
        high: Series of high prices with DatetimeIndex (UTC-aware or treated as UTC)
        low: Series of low prices with DatetimeIndex
        close: Series of close prices with DatetimeIndex
        volume: Series of volume values with DatetimeIndex
        anchor: Session anchor type ('asia', 'london', 'ny', 'daily'). Default 'ny'.

    Returns:
        pd.Series with UTC-aware index (aligned with input), containing VWAP values.
        Returns empty Series if input length is 0.

    Raises:
        ValueError: If inputs have mismatched lengths or invalid anchor.

    Example:
        >>> high = pd.Series([105, 106, 107], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 101, 102], index=high.index)
        >>> close = pd.Series([103, 104, 105], index=high.index)
        >>> volume = pd.Series([1000, 1100, 1200], index=high.index)
        >>> vwap = calculate_vwap(high, low, close, volume, anchor='ny')
    """
    # Handle empty input
    if len(high) == 0:
        return pd.Series(dtype=float)

    # Validate lengths
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("All input Series must have the same length")

    # Get anchor hour
    anchor_hour = _get_session_anchor_hour(anchor)

    # Ensure UTC-aware index
    high = _ensure_utc_index(high)
    low = _ensure_utc_index(low)
    close = _ensure_utc_index(close)
    volume = _ensure_utc_index(volume)

    # Align indices (should be aligned already, but safety check)
    # align returns tuple of (aligned_series1, aligned_series2)
    high, low = high.align(low, join="inner")
    close = close.reindex(high.index)
    volume = volume.reindex(high.index)

    if len(high) == 0:
        return pd.Series(dtype=float, index=high.index[:0])

    # Calculate typical price
    typical_price = (high + low + close) / 3.0

    # Calculate TP × Volume
    tp_volume = typical_price * volume

    # Get session groups
    session_groups = _calculate_session_groups(high.index, anchor_hour)

    # Create a DataFrame for groupby operations
    df = pd.DataFrame(
        {
            "tp_volume": tp_volume.values,
            "volume": volume.values,
            "session": session_groups.values,
        },
        index=high.index,
    )

    # Calculate cumulative sums per session
    df["cum_tp_volume"] = df.groupby("session")["tp_volume"].cumsum()
    df["cum_volume"] = df.groupby("session")["volume"].cumsum()

    # Calculate VWAP (handle division by zero)
    vwap = pd.Series(dtype=float, index=high.index)
    vwap = df["cum_tp_volume"] / df["cum_volume"]

    # Handle potential NaN from zero volume
    # If volume is 0, use typical price at that bar
    zero_volume_mask = df["cum_volume"] == 0
    vwap[zero_volume_mask] = typical_price[zero_volume_mask]

    return vwap


def calculate_vwap_bands(
    vwap: pd.Series,
    close: pd.Series,
    std_dev: tuple[float, float] = (1.0, 2.0),
) -> pd.DataFrame:
    """Calculate VWAP bands (standard deviation bands around VWAP).

    Deviation = close - vwap
    Standard deviation uses population std dev (ddof=0) - TradingView/StockCharts convention.

    Args:
        vwap: Series of VWAP values with UTC-aware DatetimeIndex
        close: Series of close prices with UTC-aware DatetimeIndex
        std_dev: Tuple of standard deviation multipliers. Default (1.0, 2.0) for 1σ and 2σ bands.

    Returns:
        pd.DataFrame with UTC-aware pd.Timestamp index and columns:
            - vwap: The VWAP series
            - upper_band_1std, lower_band_1std: 1σ bands
            - upper_band_2std, lower_band_2std: 2σ bands

        Empty DataFrame with columns if inputs are empty.

    Example:
        >>> vwap = calculate_vwap(high, low, close, volume, anchor='ny')
        >>> bands = calculate_vwap_bands(vwap, close, std_dev=(1.0, 2.0))
    """
    # Define expected columns
    columns = [
        "vwap",
        "upper_band_1std",
        "lower_band_1std",
        "upper_band_2std",
        "lower_band_2std",
    ]

    # Handle empty input
    if vwap.empty or close.empty:
        return pd.DataFrame(columns=columns)

    # Align series
    vwap, close = vwap.align(close, join="inner")

    if len(vwap) == 0:
        return pd.DataFrame(columns=columns)

    # Filter out NaN values for calculation
    valid_mask = vwap.notna() & close.notna()
    vwap_valid = vwap[valid_mask]
    close_valid = close[valid_mask]

    if len(vwap_valid) == 0:
        return pd.DataFrame(columns=columns)

    # Calculate deviation
    deviation = close_valid - vwap_valid

    # Calculate population standard deviation (ddof=0)
    # This is the TradingView/StockCharts convention
    std_dev_value = deviation.std(ddof=0)

    # Get std multipliers (default: 1.0 and 2.0)
    std1, std2 = std_dev

    # Build result DataFrame
    result = pd.DataFrame(index=vwap_valid.index)
    result["vwap"] = vwap_valid

    # Calculate bands
    result["upper_band_1std"] = vwap_valid + std1 * std_dev_value
    result["lower_band_1std"] = vwap_valid - std1 * std_dev_value
    result["upper_band_2std"] = vwap_valid + std2 * std_dev_value
    result["lower_band_2std"] = vwap_valid - std2 * std_dev_value

    return result


def detect_vwap_cross(price: pd.Series, vwap: pd.Series) -> pd.DataFrame:
    """Detect price crossing above or below VWAP.

    Bullish cross: price moves from below VWAP to above VWAP
    Bearish cross: price moves from above VWAP to below VWAP

    One event per crossing - no repeated signals while price stays on one side.

    Args:
        price: Series of price values (typically close) with UTC-aware DatetimeIndex
        vwap: Series of VWAP values with UTC-aware DatetimeIndex

    Returns:
        pd.DataFrame with columns:
            - date: pd.Timestamp of crossover (UTC-aware)
            - direction: 'bullish' or 'bearish'
            - price: Price value at crossover point
            - vwap_value: VWAP value at crossover point

        Empty DataFrame if no crossovers detected or insufficient data.

    Example:
        >>> vwap = calculate_vwap(high, low, close, volume)
        >>> crosses = detect_vwap_cross(close, vwap)
        >>> bullish = crosses[crosses['direction'] == 'bullish']
    """
    # Define expected columns
    columns = ["date", "direction", "price", "vwap_value"]

    # Handle empty input
    if price.empty or vwap.empty:
        return pd.DataFrame(columns=columns)

    # Align series
    price, vwap = price.align(vwap, join="inner")

    if len(price) == 0:
        return pd.DataFrame(columns=columns)

    # Filter out NaN values
    valid_mask = price.notna() & vwap.notna()
    price_valid = price[valid_mask]
    vwap_valid = vwap[valid_mask]

    if len(price_valid) < 2:
        return pd.DataFrame(columns=columns)

    # Calculate position relative to VWAP
    # above_vwap = True when price >= vwap, False when price < vwap
    above_vwap = price_valid >= vwap_valid

    # Detect state changes (crossover points)
    # diff = 1: crossed from below to above (bullish)
    # diff = -1: crossed from above to below (bearish)
    cross_changes = above_vwap.astype(int).diff()

    # Get indices where cross occurred (diff is non-zero and not NaN)
    cross_indices = cross_changes[cross_changes.notna() & (cross_changes != 0)].index

    if len(cross_indices) == 0:
        return pd.DataFrame(columns=columns)

    # Build result DataFrame
    records = []
    for idx in cross_indices:
        diff_val = int(cross_changes.loc[idx])

        # Determine direction based on diff sign
        # diff = 1 means False->True (crossed from below to above = bullish)
        # diff = -1 means True->False (crossed from above to below = bearish)
        direction = "bullish" if diff_val == 1 else "bearish"

        # Convert index to timestamp if needed
        if isinstance(idx, pd.Timestamp):
            date_val = idx
        elif isinstance(idx, (int, float)):
            date_val = idx
        else:
            date_val = pd.to_datetime(idx, errors="coerce")
            if pd.isna(date_val):
                date_val = idx

        records.append(
            {
                "date": date_val,
                "direction": direction,
                "price": float(price_valid.loc[idx]),
                "vwap_value": float(vwap_valid.loc[idx]),
            }
        )

    return pd.DataFrame(records)


__all__ = [
    "calculate_vwap",
    "calculate_vwap_bands",
    "detect_vwap_cross",
    "SESSION_ANCHORS",
]
