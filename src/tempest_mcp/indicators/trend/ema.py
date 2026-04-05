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
    if period <= 0:
        raise ValueError("Period must be a positive integer")

    if len(prices) < period:
        logger.debug("Insufficient data for EMA(%d): %d < %d", period, len(prices), period)
        return pd.Series(dtype=float)

    # Use pandas native ewm for efficiency
    # span parameter automatically applies: alpha = 2/(span+1)
    ema = prices.ewm(span=period, adjust=False).mean()

    return ema


def calculate_ema_stack(prices: pd.Series) -> dict[str, pd.Series]:
    """Calculate all four standard EMA periods in a single call.

    Returns EMAs for periods 7, 25, 50, and 200 - the standard stack
    used for trend confirmation in the EMA Stack strategy (TVMCP-019).

    Args:
        prices: Series of price values (typically close prices) with datetime index

    Returns:
        Dictionary with keys 'ema7', 'ema25', 'ema50', 'ema200',
        each containing a pd.Series aligned with the input prices index.
        Series will be empty if insufficient data for that period.

    Example:
        >>> prices = pd.Series(...)  # 300+ data points
        >>> stack = calculate_ema_stack(prices)
        >>> print(stack.keys())
        dict_keys(['ema7', 'ema25', 'ema50', 'ema200'])
    """
    result = {}

    for period in EMA_PERIODS:
        ema = calculate_ema(prices, period)
        result[f"ema{period}"] = ema

    return result


def detect_ema_cross(
    ema_fast: pd.Series, ema_slow: pd.Series
) -> pd.DataFrame:
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

    # Ensure same length and alignment
    if len(ema_fast) != len(ema_slow):
        logger.warning("EMA series length mismatch in cross detection")
        min_len = min(len(ema_fast), len(ema_slow))
        ema_fast = ema_fast.iloc[:min_len]
        ema_slow = ema_slow.iloc[:min_len]

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

    # Get indices where cross occurred (diff is non-zero)
    cross_indices = cross_changes[cross_changes != 0].index

    if len(cross_indices) == 0:
        return pd.DataFrame(columns=["date", "fast_above", "direction"])

    # Build result DataFrame
    records = []
    for idx in cross_indices:
        fast_above_at_cross = fast_above.loc[idx]
        direction = "cross_up" if fast_above_at_cross else "cross_down"

        # Convert index to timestamp if it's datetime-like
        if isinstance(idx, pd.Timestamp):
            date_val = idx
        else:
            date_val = pd.Timestamp(idx)

        records.append({
            "date": date_val,
            "fast_above": fast_above_at_cross,
            "direction": direction,
        })

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
            logger.warning(f"Missing {key} in EMA stack")
            return False
        if ema_stack[key].empty:
            return False

    # Get latest non-NaN values
    try:
        ema7_series = ema_stack["ema7"].dropna()
        ema25_series = ema_stack["ema25"].dropna()
        ema50_series = ema_stack["ema50"].dropna()
        ema200_series = ema_stack["ema200"].dropna()

        if ema7_series.empty or ema25_series.empty or ema50_series.empty or ema200_series.empty:
            return False

        ema7 = ema7_series.iloc[-1]
        ema25 = ema25_series.iloc[-1]
        ema50 = ema50_series.iloc[-1]
        ema200 = ema200_series.iloc[-1]

        return (ema7 > ema25) and (ema25 > ema50) and (ema50 > ema200)
    except (IndexError, KeyError) as e:
        logger.debug("Error accessing EMA values: %s", e)
        return False


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
            logger.warning(f"Missing {key} in EMA stack")
            return False
        if ema_stack[key].empty:
            return False

    # Get latest non-NaN values
    try:
        ema7_series = ema_stack["ema7"].dropna()
        ema25_series = ema_stack["ema25"].dropna()
        ema50_series = ema_stack["ema50"].dropna()
        ema200_series = ema_stack["ema200"].dropna()

        if ema7_series.empty or ema25_series.empty or ema50_series.empty or ema200_series.empty:
            return False

        ema7 = ema7_series.iloc[-1]
        ema25 = ema25_series.iloc[-1]
        ema50 = ema50_series.iloc[-1]
        ema200 = ema200_series.iloc[-1]

        return (ema7 < ema25) and (ema25 < ema50) and (ema50 < ema200)
    except (IndexError, KeyError) as e:
        logger.debug("Error accessing EMA values: %s", e)
        return False
