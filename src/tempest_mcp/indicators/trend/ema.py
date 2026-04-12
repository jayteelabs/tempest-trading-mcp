"""EMA Indicator Engine - Exponential Moving Average calculations.

Implements EMA calculations for periods 7, 25, 50, and 200 with crossover detection
and stack confirmation for trend analysis.
"""

import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Standard EMA periods for stack analysis
EMA_PERIODS = [7, 25, 50, 200]


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average using standard smoothing factor.

    Uses the exponential smoothing factor: α = 2 / (period + 1)

    Args:
        prices: Series of price values (typically close prices) with datetime index
        period: Number of periods for EMA calculation (must be positive)

    Returns:
        Series containing EMA values, aligned with input prices index.
        Returns empty Series if insufficient data (len < period).

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> prices = pd.Series([100, 101, 102, 103, 104], index=pd.date_range('2024-01-01', periods=5))
        >>> ema = calculate_ema(prices, period=3)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    if len(prices) < period:
        logger.debug("Insufficient data for EMA(%d): %d < %d", period, len(prices), period)
        return pd.Series(dtype=float)

    # Use pandas native ewm for efficiency
    # span parameter automatically applies: alpha = 2/(span+1)
    ema = prices.ewm(span=period, adjust=False).mean()

    return ema


def calculate_ema_stack(prices: pd.Series, periods: list[int] | None = None) -> dict[str, pd.Series]:
    """Calculate EMA periods in a single call.

    Returns EMAs for the specified periods - the standard stack
    used for trend confirmation in the EMA Stack strategy (TVMCP-019).

    Args:
        prices: Series of price values (typically close prices) with datetime index
        periods: List of EMA periods to calculate. Defaults to [7, 25, 50, 200].

    Returns:
        Dictionary with keys 'ema{N}' for each period,
        each containing a pd.Series aligned with the input prices index.
        Series will be empty if insufficient data for that period.

    Raises:
        ValueError: If len(prices) < max(periods) - insufficient data for the
            longest period.

    Example:
        >>> prices = pd.Series(...)  # 300+ data points
        >>> stack = calculate_ema_stack(prices, [7, 25, 50, 200])
        >>> print(stack.keys())
        dict_keys(['ema7', 'ema25', 'ema50', 'ema200'])
    """
    if periods is None:
        periods = EMA_PERIODS

    if prices.empty:
        return {}

    if len(prices) < max(periods):
        raise ValueError(
            f"Insufficient data for EMA calculation: {len(prices)} values required for "
            f"period {max(periods)}. Need at least {max(periods)} data points."
        )

    result = {}

    for period in periods:
        ema = calculate_ema(prices, period)
        result[f"ema{period}"] = ema

    return result


def _get_last_valid_ema(ema_stack: dict[str, pd.Series], name: str) -> float | None:
    """Extract last non-NaN value from EMA series, logging missing keys.

    Args:
        ema_stack: Dictionary containing EMA Series
        name: Key name to extract (e.g., 'ema7')

    Returns:
        Last non-NaN value as float, or None if not found or if the
        latest value is NaN.
    """
    if name not in ema_stack:
        logger.warning("EMA key missing from stack", ema_key=name)
        return None
    ema_series = ema_stack[name]
    if ema_series.empty:
        logger.debug("EMA series is empty", ema_key=name)
        return None
    # Check if the latest value is NaN
    if ema_series.iloc[-1] != ema_series.iloc[-1]:  # NaN != NaN is True
        logger.debug("Latest EMA value is NaN", ema_key=name)
        return None
    return float(ema_series.iloc[-1])


def detect_ema_cross(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.DataFrame:
    """Detect crossover points between two EMA series.

    Identifies points where the faster EMA crosses above (bullish) or
    below (bearish) the slower EMA. Returns one signal per crossover
    event - no repeated signals.

    Args:
        ema_fast: Faster EMA series (e.g., EMA 7)
        ema_slow: Slower EMA series (e.g., EMA 25)

    Returns:
        DataFrame with columns:
            - date: pd.Timestamp of crossover
            - fast_above: True if fast is above slow at this point
            - direction: 'cross_up' or 'cross_down'

        Empty DataFrame if no crossovers detected or insufficient data.

    Example:
        >>> ema7 = calculate_ema(prices, 7)
        >>> ema25 = calculate_ema(prices, 25)
        >>> crosses = detect_ema_cross(ema7, ema25)
    """
    # Handle empty series
    if ema_fast.empty or ema_slow.empty:
        return pd.DataFrame(columns=["date", "fast_above", "direction"])

    # Align series by index to ensure proper timestamp matching
    ema_fast, ema_slow = ema_fast.align(ema_slow, join="inner")

    # Filter out NaN values before computing crossovers
    valid_mask = ema_fast.notna() & ema_slow.notna()
    fast_valid = ema_fast[valid_mask]
    slow_valid = ema_slow[valid_mask]

    if len(fast_valid) == 0:
        return pd.DataFrame(columns=["date", "fast_above", "direction"])

    # Calculate cross state at each point
    fast_above = fast_valid > slow_valid

    # Detect state changes (actual crossover points)
    # A cross occurs when the relationship changes from previous bar
    cross_changes = fast_above.astype(int).diff()

    # Get indices where cross occurred (diff is non-zero and not NaN)
    cross_indices = cross_changes[cross_changes.notna() & (cross_changes != 0)].index

    if len(cross_indices) == 0:
        return pd.DataFrame(columns=["date", "fast_above", "direction"])

    # Build result DataFrame
    records = []
    for idx in cross_indices:
        idx_pos = fast_above.index.get_loc(idx)
        curr_fast_above = bool(fast_above.loc[idx])

        # If fast_above is False (equality at this point), verify it's a true cross.
        # A true cross_up must have fast below before (diff < 0).
        # A true cross_down must have fast above before (diff > 0).
        # Equality touching (fast going from below to equal, or above to equal) is not a cross.
        if not curr_fast_above:
            if idx_pos > 0:
                prev_diff = float(fast_valid.iloc[idx_pos - 1] - slow_valid.iloc[idx_pos - 1])
                if prev_diff <= 0:
                    # Was below or equal before; now equal — not a cross_down
                    continue
            else:
                continue

        direction = "cross_up" if curr_fast_above else "cross_down"

        # Convert index to timestamp if it's datetime-like, otherwise preserve original
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
                "fast_above": curr_fast_above,
                "direction": direction,
            }
        )

    return pd.DataFrame(records)


def golden_cross(ema_stack: dict[str, pd.Series]) -> bool:
    """Check for bullish EMA stack (golden cross pattern).

    Returns True when all EMAs are stacked in bullish order:
    EMA7 > EMA25 > EMA50 > EMA200

    This indicates a strong bullish trend with proper momentum alignment.

    Args:
        ema_stack: Dictionary containing 'ema7', 'ema25', 'ema50', 'ema200' Series

    Returns:
        True if bullish stack confirmed, False otherwise.
        Returns False if any EMA series is empty.

    Example:
        >>> stack = calculate_ema_stack(prices)
        >>> if golden_cross(stack):
        ...     print("Bullish trend confirmed!")
    """
    required_keys = ["ema7", "ema25", "ema50", "ema200"]

    # Validate stack has all required EMAs
    for key in required_keys:
        if key not in ema_stack:
            logger.warning("Missing EMA key in stack", ema_key=key)
            return False
        if ema_stack[key].empty:
            return False

    # Get latest non-NaN values using shared helper
    ema7 = _get_last_valid_ema(ema_stack, "ema7")
    ema25 = _get_last_valid_ema(ema_stack, "ema25")
    ema50 = _get_last_valid_ema(ema_stack, "ema50")
    ema200 = _get_last_valid_ema(ema_stack, "ema200")

    # Return False if any value is missing/NaN
    if None in (ema7, ema25, ema50, ema200):
        return False

    # Explicitly convert to Python bool to avoid np.True_/np.False_ issues
    return bool(ema7 > ema25) and bool(ema25 > ema50) and bool(ema50 > ema200)


def death_cross(ema_stack: dict[str, pd.Series]) -> bool:
    """Check for bearish EMA stack (death cross pattern).

    Returns True when all EMAs are stacked in bearish order:
    EMA7 < EMA25 < EMA50 < EMA200

    This indicates a strong bearish trend with proper momentum alignment.

    Args:
        ema_stack: Dictionary containing 'ema7', 'ema25', 'ema50', 'ema200' Series

    Returns:
        True if bearish stack confirmed, False otherwise.
        Returns False if any EMA series is empty.

    Example:
        >>> stack = calculate_ema_stack(prices)
        >>> if death_cross(stack):
        ...     print("Bearish trend confirmed!")
    """
    required_keys = ["ema7", "ema25", "ema50", "ema200"]

    # Validate stack has all required EMAs
    for key in required_keys:
        if key not in ema_stack:
            logger.warning("Missing EMA key in stack", ema_key=key)
            return False
        if ema_stack[key].empty:
            return False

    # Get latest non-NaN values using shared helper
    ema7 = _get_last_valid_ema(ema_stack, "ema7")
    ema25 = _get_last_valid_ema(ema_stack, "ema25")
    ema50 = _get_last_valid_ema(ema_stack, "ema50")
    ema200 = _get_last_valid_ema(ema_stack, "ema200")

    # Return False if any value is missing/NaN
    if None in (ema7, ema25, ema50, ema200):
        return False

    # Explicitly convert to Python bool to avoid np.True_/np.False_ issues
    return bool(ema7 < ema25) and bool(ema25 < ema50) and bool(ema50 < ema200)
