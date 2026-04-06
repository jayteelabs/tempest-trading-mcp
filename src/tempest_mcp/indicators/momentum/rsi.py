"""RSI Indicator Engine - Relative Strength Index calculations.

Implements RSI calculations with configurable smoothing (SMMA default per Josh's decision),
extreme zone detection, divergence detection, and threshold crossover detection.

RSI Formula: 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over period

Smoothing types:
- SMMA (default): Wilder's smoothed moving average (alpha = 1/period)
- EMA: Exponential moving average (alpha = 2/(period+1))
"""

import numpy as np
import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Default RSI period
RSI_DEFAULT_PERIOD = 14

# RSI zone thresholds
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70
CENTERLINE = 50.0


def _calculate_smma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Smoothed Moving Average (Wilder's smoothing).

    SMMA uses alpha = 1/period, which is different from EMA's alpha = 2/(period+1).
    This is the standard RSI smoothing method used by Welles Wilder.

    Args:
        series: Series of values to smooth
        period: Smoothing period

    Returns:
        SMMA smoothed series
    """
    if len(series) < period:
        return pd.Series(dtype=float, index=series.index)

    # First value is simple average of first 'period' values
    smma = pd.Series(dtype=float, index=series.index)
    smma.iloc[:period] = np.nan

    # Calculate first SMMA as SMA
    first_smma = series.iloc[:period].mean()
    smma.iloc[period - 1] = first_smma

    # Apply Wilder's smoothing: SMMA = (prev_SMMA * (period-1) + current) / period
    alpha = 1.0 / period
    for i in range(period, len(series)):
        if pd.isna(series.iloc[i]):
            smma.iloc[i] = smma.iloc[i - 1]
        else:
            smma.iloc[i] = smma.iloc[i - 1] * (1 - alpha) + series.iloc[i] * alpha

    return smma


def _calculate_ema_custom(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA with standard smoothing factor.

    Uses alpha = 2/(period+1) which is standard EMA formula.

    Args:
        series: Series of values to smooth
        period: Smoothing period

    Returns:
        EMA smoothed series
    """
    # Use pandas native ewm for efficiency
    # span parameter automatically applies: alpha = 2/(span+1)
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14, smooth_type: str = "smma") -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price movements to identify
    overbought or oversold conditions.

    Args:
        prices: Series of price values (typically close prices) with datetime index.
                Index should be UTC-aware pd.Timestamp.
        period: Number of periods for RSI calculation (default 14).
        smooth_type: Smoothing method - 'smma' (Wilder's, default) or 'ema'.

    Returns:
        pd.Series containing RSI values (0-100), aligned with input prices index.
        Returns empty Series if insufficient data (len < period + 1).

    Raises:
        ValueError: If period is not a positive integer or smooth_type is invalid.

    Example:
        >>> prices = pd.Series([100, 101, 102, ...], index=pd.date_range('2024-01-01', periods=100, tz='UTC'))
        >>> rsi = calculate_rsi(prices, period=14)  # Uses SMMA (default)
        >>> rsi_ema = calculate_rsi(prices, period=14, smooth_type='ema')
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    if smooth_type not in ("smma", "ema"):
        raise ValueError(f"smooth_type must be 'smma' or 'ema', got '{smooth_type}'")

    # Need at least period + 1 values to calculate changes
    if len(prices) < period + 1:
        logger.debug(
            "Insufficient data for RSI(%d): %d < %d",
            period,
            len(prices),
            period + 1,
        )
        return pd.Series(dtype=float, index=prices.index[:0])

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using selected smoothing
    if smooth_type == "smma":
        avg_gains = _calculate_smma(gains, period)
        avg_losses = _calculate_smma(losses, period)
    else:  # ema
        avg_gains = _calculate_ema_custom(gains, period)
        avg_losses = _calculate_ema_custom(losses, period)

    # Calculate RS and RSI
    # RS = Average Gain / Average Loss
    # RSI = 100 - (100 / (1 + RS))
    # Handle division by zero: if avg_loss is 0, RS is inf, RSI is 100
    rs = avg_gains / avg_losses

    # When avg_loss is 0 but avg_gain > 0, RSI should be 100
    # When both are 0 (no movement), RSI should be 50 (neutral)
    # When avg_gain is 0 and avg_loss > 0, RSI should be 0
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases:
    # - When avg_loss = 0 and avg_gain > 0: RS = inf, RSI = 100 - 0 = 100 (correct)
    # - When both = 0: RS = NaN, RSI = NaN -> should be 50 (no movement)
    # - When avg_gain = 0 and avg_loss > 0: RS = 0, RSI = 0 (correct)

    # For flat prices (no movement), set RSI to 50 (neutral)
    no_movement_mask = (avg_gains == 0) & (avg_losses == 0)
    rsi = rsi.where(~no_movement_mask, 50.0)

    # Clamp to valid range [0, 100]
    rsi = rsi.clip(0, 100)

    return rsi


def detect_rsi_extremes(rsi: pd.Series, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """Detect RSI zone transitions at threshold crossings.

    Identifies points where RSI enters or exits oversold/overbought zones.

    Args:
        rsi: Series of RSI values with datetime index.
        oversold: Oversold threshold (default 30). Values <= this are oversold.
        overbought: Overbought threshold (default 70). Values >= this are overbought.

    Returns:
        pd.DataFrame with columns:
            - date: pd.Timestamp of zone transition (UTC-aware)
            - zone: 'oversold', 'overbought', or 'neutral'

        Empty DataFrame if no zone transitions detected or insufficient data.

    Example:
        >>> rsi = calculate_rsi(prices, period=14)
        >>> extremes = detect_rsi_extremes(rsi)
        >>> oversold_events = extremes[extremes['zone'] == 'oversold']
    """
    if rsi.empty:
        return pd.DataFrame(columns=["date", "zone", "value"])

    # Handle NaN values
    valid_mask = rsi.notna()
    rsi_valid = rsi[valid_mask]

    if len(rsi_valid) < 1:
        return pd.DataFrame(columns=["date", "zone", "value"])

    # Determine zone for each RSI value
    def get_zone(value: float) -> str:
        if value <= oversold:
            return "oversold"
        elif value >= overbought:
            return "overbought"
        else:
            return "neutral"

    zones = rsi_valid.apply(get_zone)

    # Detect zone transitions (where zone changes from previous value)
    # First valid point is a transition
    zone_changes = zones != zones.shift(1)
    zone_changes.iloc[0] = True

    # Get transition points
    transition_indices = zone_changes[zone_changes].index

    if len(transition_indices) == 0:
        return pd.DataFrame(columns=["date", "zone", "value"])

    # Build result DataFrame
    records = []
    for idx in transition_indices:
        zone_val = zones.loc[idx]
        rsi_val = float(rsi_valid.loc[idx])

        # Convert index to timestamp if needed
        if isinstance(idx, pd.Timestamp):
            date_val = idx
        else:
            date_val = pd.to_datetime(idx, errors="coerce")
            if pd.isna(date_val):
                date_val = idx

        records.append(
            {
                "date": date_val,
                "zone": zone_val,
                "value": rsi_val,
            }
        )

    return pd.DataFrame(records)


def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, window: int = 20) -> pd.DataFrame:
    """Detect RSI divergence patterns.

    Bullish divergence: Price makes Lower Low (LL) while RSI makes Higher Low (HL)
    Bearish divergence: Price makes Higher High (HH) while RSI makes Lower High (LH)

    This is a simplified divergence detection using local extrema within rolling windows.

    Args:
        prices: Series of price values with datetime index.
        rsi: Series of RSI values with datetime index.
        window: Window size for detecting local extrema (default 20).

    Returns:
        pd.DataFrame with columns:
            - date: pd.Timestamp of divergence detection (UTC-aware)
            - type: 'bullish' or 'bearish'
            - price: Price value at detection point
            - rsi_value: RSI value at detection point

        Empty DataFrame if no divergence detected or insufficient data.

    Example:
        >>> rsi = calculate_rsi(prices, period=14)
        >>> divergence = detect_rsi_divergence(prices, rsi, window=20)
    """
    if prices.empty or rsi.empty:
        return pd.DataFrame(columns=["date", "type", "price", "rsi_value"])

    # Align series by index
    prices, rsi = prices.align(rsi, join="inner")

    # Filter out NaN values
    valid_mask = prices.notna() & rsi.notna()
    prices = prices[valid_mask]
    rsi = rsi[valid_mask]

    if len(prices) < window:
        return pd.DataFrame(columns=["date", "type", "price", "rsi_value"])

    records = []
    half_window = window // 2

    for i in range(half_window, len(prices) - half_window):
        start_idx = i - half_window
        end_idx = i + half_window + 1

        local_prices = prices.iloc[start_idx:end_idx]

        current_price = prices.iloc[i]
        current_rsi = rsi.iloc[i]

        # Check if current point is a local extremum in price
        is_price_low = current_price <= local_prices.min()
        is_price_high = current_price >= local_prices.max()

        if not (is_price_low or is_price_high):
            continue

        # Look for previous extremum of the same type in earlier data
        lookback_start = max(0, i - window * 2)
        lookback_prices = prices.iloc[lookback_start:i]
        lookback_rsi = rsi.iloc[lookback_start:i]

        if len(lookback_prices) < half_window:
            continue

        if is_price_low:
            # Price making lower low - check for bullish divergence
            prev_lows = lookback_prices[
                lookback_prices <= lookback_prices.rolling(window, min_periods=1).min()
            ]
            if len(prev_lows) > 0:
                prev_low_idx = prev_lows.index[-1]
                prev_low_price = lookback_prices.loc[prev_low_idx]
                prev_low_rsi = lookback_rsi.loc[prev_low_idx]

                # Bullish divergence: Lower Low in price, Higher Low in RSI
                if current_price < prev_low_price and current_rsi > prev_low_rsi:
                    idx_timestamp = prices.index[i]
                    if isinstance(idx_timestamp, pd.Timestamp):
                        date_val = idx_timestamp
                    else:
                        date_val = pd.to_datetime(idx_timestamp, errors="coerce")

                    records.append(
                        {
                            "date": date_val,
                            "type": "bullish",
                            "price": float(current_price),
                            "rsi_value": float(current_rsi),
                        }
                    )

        elif is_price_high:
            # Price making higher high - check for bearish divergence
            prev_highs = lookback_prices[
                lookback_prices >= lookback_prices.rolling(window, min_periods=1).max()
            ]
            if len(prev_highs) > 0:
                prev_high_idx = prev_highs.index[-1]
                prev_high_price = lookback_prices.loc[prev_high_idx]
                prev_high_rsi = lookback_rsi.loc[prev_high_idx]

                # Bearish divergence: Higher High in price, Lower High in RSI
                if current_price > prev_high_price and current_rsi < prev_high_rsi:
                    idx_timestamp = prices.index[i]
                    if isinstance(idx_timestamp, pd.Timestamp):
                        date_val = idx_timestamp
                    else:
                        date_val = pd.to_datetime(idx_timestamp, errors="coerce")

                    records.append(
                        {
                            "date": date_val,
                            "type": "bearish",
                            "price": float(current_price),
                            "rsi_value": float(current_rsi),
                        }
                    )

    return pd.DataFrame(records)


def detect_rsi_cross(rsi: pd.Series, threshold: float = 50.0) -> pd.DataFrame:
    """Detect RSI crosses through a threshold level.

    Identifies points where RSI crosses above (bullish) or below (bearish)
    a threshold level. Returns one signal per crossover event - no repeated signals.

    Args:
        rsi: Series of RSI values with datetime index.
        threshold: Threshold level to detect crosses (default 50.0 - centerline).

    Returns:
        pd.DataFrame with columns:
            - date: pd.Timestamp of crossover (UTC-aware)
            - direction: 'bullish' (cross up) or 'bearish' (cross down)
            - value: RSI value at crossover point

        Empty DataFrame if no crossovers detected or insufficient data.

    Example:
        >>> rsi = calculate_rsi(prices, period=14)
        >>> crosses = detect_rsi_cross(rsi, threshold=50.0)
        >>> bullish_crosses = crosses[crosses['direction'] == 'bullish']
    """
    if rsi.empty:
        return pd.DataFrame(columns=["date", "direction", "value"])

    # Handle NaN values
    valid_mask = rsi.notna()
    rsi_valid = rsi[valid_mask]

    if len(rsi_valid) < 2:
        return pd.DataFrame(columns=["date", "direction", "value"])

    # Calculate position relative to threshold
    above_threshold = rsi_valid > threshold

    # Detect state changes (actual crossover points)
    cross_changes = above_threshold.astype(int).diff()

    # Get indices where cross occurred (diff is non-zero and not NaN)
    cross_indices = cross_changes[cross_changes.notna() & (cross_changes != 0)].index

    if len(cross_indices) == 0:
        return pd.DataFrame(columns=["date", "direction", "value"])

    # Build result DataFrame
    records = []
    for idx in cross_indices:
        idx_pos = rsi_valid.index.get_loc(idx)
        curr_above = bool(above_threshold.loc[idx])

        # If exactly at threshold (not above), verify it's a true cross
        if not curr_above and abs(float(rsi_valid.loc[idx]) - threshold) < 1e-9:
            if idx_pos > 0:
                prev_val = float(rsi_valid.iloc[idx_pos - 1])
                if prev_val <= threshold:
                    # Was below or equal before, now equal - not a true cross_down
                    continue
            else:
                continue

        direction = "bullish" if curr_above else "bearish"

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
                "value": float(rsi_valid.loc[idx]),
            }
        )

    return pd.DataFrame(records)


__all__ = [
    "calculate_rsi",
    "detect_rsi_extremes",
    "detect_rsi_divergence",
    "detect_rsi_cross",
    "RSI_DEFAULT_PERIOD",
    "OVERSOLD_THRESHOLD",
    "OVERBOUGHT_THRESHOLD",
    "CENTERLINE",
]
