"""Volume Profile Indicator Engine - Fixed and Dynamic Range.

Volume Profile organizes volume by price level, revealing where trading activity
concentrates. This implementation supports two modes:
- Fixed: Equal-width bins across the full observed price range
- Dynamic: ATR-based or percentage-of-price bin sizing

The profile identifies key levels (POC, VAH, VAL) and classifies the profile shape.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tempest_mcp.indicators.volatility.atr import calculate_atr

# Profile shape classifications
ProfileShape = Literal["bell", "bimodal", "directional", "flat", "single"]

# Column names for output DataFrame
COL_BIN_LOW = "bin_low"
COL_BIN_HIGH = "bin_high"
COL_BIN_MID = "bin_mid"
COL_BIN_VOLUME = "bin_volume"
COL_BIN_CANDLE_COUNT = "bin_candle_count"
COL_IS_HVN = "is_hvn"
COL_IS_LVN = "is_lvn"

VALID_PROFILE_TYPES = ("fixed", "dynamic")
VALID_DYNAMIC_MODES = ("atr", "pct")
HVN_QUANTILE = 0.80
LVN_QUANTILE = 0.20


def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame input.

    Args:
        ohlcv: DataFrame with columns open, high, low, close, volume and UTC-aware index.

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

    if (ohlcv["high"] < ohlcv["low"]).any():
        raise ValueError("OHLCV high values must be greater than or equal to low values")

    if (ohlcv["volume"] < 0).any():
        raise ValueError("OHLCV volume values must be non-negative")


def _allocate_volume_to_bins(
    low_vals: pd.Series,
    high_vals: pd.Series,
    volume_vals: pd.Series,
    bin_edges: pd.Index,
) -> pd.Series:
    """Allocate candle volume to bins based on price range.

    Each candle's volume is distributed proportionally across bins that
    intersect its [low, high] range.

    Args:
        low_vals: Series of candle low prices.
        high_vals: Series of candle high prices.
        volume_vals: Series of candle volumes.
        bin_edges: Sorted array of bin boundary prices (length = n_bins + 1).

    Returns:
        Series with volume per bin (indexed by bin center values).
    """
    n_bins = len(bin_edges) - 1
    bin_volumes = pd.Series(0.0, index=range(n_bins))

    for i in range(len(low_vals)):
        candle_low = low_vals.iloc[i]
        candle_high = high_vals.iloc[i]
        candle_vol = volume_vals.iloc[i]

        if candle_vol <= 0:
            continue

        # Calculate candle range width
        candle_width = candle_high - candle_low

        if candle_width == 0:
            for j in range(n_bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]
                if bin_low <= candle_low <= bin_high:
                    bin_volumes.iloc[j] += candle_vol
                    break
            continue

        # Find bins that intersect with [candle_low, candle_high]
        for j in range(n_bins):
            bin_low = bin_edges[j]
            bin_high = bin_edges[j + 1]

            if bin_low == bin_high:
                continue

            # Check for intersection: bin_low < candle_high AND bin_high > candle_low
            if bin_low < candle_high and bin_high > candle_low:
                # Calculate intersection width
                intersect_low = max(bin_low, candle_low)
                intersect_high = min(bin_high, candle_high)
                intersect_width = intersect_high - intersect_low

                # Allocate proportional volume
                proportion = intersect_width / candle_width
                bin_volumes.iloc[j] += candle_vol * proportion

    return bin_volumes


def _calculate_atr_based_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int,
    atr_mult: float,
) -> float:
    """Calculate ATR-based range for dynamic bin sizing.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        atr_period: ATR period.
        atr_mult: ATR multiplier.

    Returns:
        ATR-derived range width.
    """
    atr = calculate_atr(high, low, close, period=atr_period)
    if atr.empty or atr.dropna().empty:
        # Fallback: use price range
        return float(close.max() - close.min())

    atr_value = float(atr.dropna().iloc[-1])
    return atr_value * atr_mult


def _calculate_pct_based_range(close: pd.Series, range_pct: float) -> float:
    """Calculate percentage-of-price range for dynamic bin sizing.

    Args:
        close: Series of close prices.
        range_pct: Percentage of last close price (e.g., 0.02 for 2%).

    Returns:
        Percentage-derived range width.
    """
    last_close = float(close.iloc[-1])
    return last_close * range_pct


def _build_dynamic_bin_edges(price_min: float, price_max: float, bin_width: float) -> pd.Index:
    """Build dynamic profile bin edges using the requested step size."""
    if price_min == price_max:
        return pd.Index([price_min, price_max])

    edges = np.arange(price_min, price_max, bin_width, dtype=float)
    if edges.size == 0 or not np.isclose(edges[0], price_min):
        edges = np.insert(edges, 0, price_min)

    if not np.isclose(edges[-1], price_max):
        edges = np.append(edges, price_max)

    return pd.Index(edges)


def _identify_hvn_lvn(
    bin_volumes: pd.Series,
    bin_centers: pd.Series,
    hvn_threshold: float,
    lvn_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Identify High-Volume Nodes (HVN) and Low-Volume Nodes (LVN).

    HVN: bins with volume >= the configured upper quantile cutoff.
    LVN: bins with volume <= the configured lower quantile cutoff.

    Args:
        bin_volumes: Series of bin volumes.
        bin_centers: Series of bin center prices.
        hvn_threshold: Quantile threshold for HVN (e.g., 0.8).
        lvn_threshold: Quantile threshold for LVN (e.g., 0.2).

    Returns:
        Tuple of (is_hvn Series, is_lvn Series) indexed by bin_centers.
    """
    max_vol = bin_volumes.max()
    if max_vol == 0:
        return pd.Series(False, index=bin_centers), pd.Series(False, index=bin_centers)

    hvn_cutoff = float(bin_volumes.quantile(hvn_threshold))
    lvn_cutoff = float(bin_volumes.quantile(lvn_threshold))

    is_hvn = bin_volumes >= hvn_cutoff
    is_lvn = bin_volumes <= lvn_cutoff

    return is_hvn, is_lvn


def _classify_profile_shape(
    bin_volumes: pd.Series,
    poc_idx: int,
) -> ProfileShape:
    """Classify the shape of the volume profile.

    Classification rules:
    - bell: POC near center, symmetric distribution
    - bimodal: Two distinct peaks away from center
    - directional: Strong skew toward one side
    - flat: Uniform distribution across bins
    - single: One dominant bin with minimal others

    Args:
        bin_volumes: Series of bin volumes.
        poc_idx: Index of the POC (peak) bin.

    Returns:
        ProfileShape classification.
    """
    n_bins = len(bin_volumes)
    if n_bins == 0:
        return "flat"

    if n_bins == 1:
        return "single"

    max_vol = bin_volumes.max()
    total_volume = bin_volumes.sum()

    if total_volume == 0:
        return "flat"

    # Calculate normalized position of POC (0 to 1)
    poc_position = poc_idx / (n_bins - 1) if n_bins > 1 else 0.5

    # Check for single dominant bin
    dominant_ratio = max_vol / total_volume
    if dominant_ratio > 0.8:
        return "single"

    # Check for flat distribution (low variance)
    normalized_vols = bin_volumes / max_vol
    variance = normalized_vols.var()
    if variance < 0.05:
        return "flat"

    # Check for bimodal (two local maxima well separated)
    if n_bins >= 4:
        # Simple bimodal detection: look for two peaks
        peaks = []
        for i in range(1, n_bins - 1):
            if (
                bin_volumes.iloc[i] > bin_volumes.iloc[i - 1]
                and bin_volumes.iloc[i] > bin_volumes.iloc[i + 1]
            ):
                peaks.append(i)
        if len(peaks) >= 2:
            # Check if peaks are well separated (at least 30% of range apart)
            peak_distance = max(peaks) - min(peaks)
            if peak_distance >= n_bins * 0.3:
                return "bimodal"

    # Check for directional (strong skew)
    left_vol = bin_volumes.iloc[:poc_idx].sum() if poc_idx > 0 else 0
    right_vol = bin_volumes.iloc[poc_idx + 1 :].sum() if poc_idx < n_bins - 1 else 0

    if poc_position < 0.3 and left_vol > right_vol * 2:
        return "directional"
    if poc_position > 0.7 and right_vol > left_vol * 2:
        return "directional"

    # Default to bell shape (normal-ish distribution centered on POC)
    return "bell"


def _calculate_value_area(
    bin_volumes: pd.Series,
    poc_idx: int,
    value_area_pct: float,
) -> tuple[int, int]:
    """Calculate Value Area High and Low.

    VAH/VAL are the boundaries of the narrowest contiguous set of bins
    centered on POC that covers at least value_area_pct of total volume.

    Args:
        bin_volumes: Series of bin volumes indexed by position.
        poc_idx: Index of the POC bin.
        value_area_pct: Target percentage (e.g., 0.70 for 70%).

    Returns:
        Tuple of (value_area_low_idx, value_area_high_idx) as bin indices.
    """
    n_bins = len(bin_volumes)
    if n_bins == 0:
        return 0, 0

    total_volume = bin_volumes.sum()
    if total_volume == 0:
        return 0, n_bins - 1

    target_volume = total_volume * value_area_pct

    # Expand outward from POC symmetrically
    left_idx = poc_idx
    right_idx = poc_idx
    current_volume = bin_volumes.iloc[poc_idx]

    # Expand contiguously from the POC until target coverage is reached.
    while current_volume < target_volume and (left_idx > 0 or right_idx < n_bins - 1):
        left_add = bin_volumes.iloc[left_idx - 1] if left_idx > 0 else 0
        right_add = bin_volumes.iloc[right_idx + 1] if right_idx < n_bins - 1 else 0

        if left_idx <= 0:
            right_idx += 1
            current_volume += right_add
        elif right_idx >= n_bins - 1:
            left_idx -= 1
            current_volume += left_add
        elif left_add >= right_add:
            left_idx -= 1
            current_volume += left_add
        else:
            right_idx += 1
            current_volume += right_add

    return left_idx, right_idx


def calculate_volume_profile(
    ohlcv: pd.DataFrame,
    *,
    bin_count: int = 100,
    profile_type: Literal["fixed", "dynamic"] = "fixed",
    dynamic_mode: Literal["atr", "pct"] | None = None,
    atr_period: int = 14,
    atr_mult: float = 1.0,
    range_pct: float | None = None,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """Calculate Volume Profile for OHLCV data.

    Volume Profile organizes trading volume by price level, revealing where
    significant buying/selling occurred. Key levels identified:
    - POC (Point of Control): Bin with highest volume
    - VAH/VAL (Value Area High/Low): Boundaries covering value_area_pct of volume

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume] and
            UTC-aware DatetimeIndex. Must be non-empty, monotonic, non-duplicated.
        bin_count: Number of bins for fixed mode (default 100). Ignored in dynamic mode.
        profile_type: 'fixed' for equal-width bins, 'dynamic' for ATR/pct-based bins.
        dynamic_mode: 'atr' for ATR-based bin width, 'pct' for percentage-of-price.
            Required when profile_type='dynamic'.
        atr_period: Period for ATR calculation in dynamic ATR mode (default 14).
        atr_mult: Multiplier for ATR in dynamic ATR mode (default 1.0).
        range_pct: Percentage of last close for bin width in dynamic pct mode.
            Required when dynamic_mode='pct'.
        value_area_pct: Percentage of total volume for Value Area (default 0.70).

    Returns:
        pd.DataFrame with columns:
            - bin_low: Lower boundary of price bin
            - bin_high: Upper boundary of price bin
            - bin_mid: Midpoint of bin (price level)
            - bin_volume: Total volume in this bin
            - bin_candle_count: Number of candles contributing to this bin
            - is_hvn: True if High-Volume Node (volume >= q80 quantile)
            - is_lvn: True if Low-Volume Node (volume <= q20 quantile)
            - in_value_area: True if bin is within Value Area
            - profile_shape: Shape classification (bell/bimodal/directional/flat/single)

        Returns empty DataFrame with all numeric columns set to float
        if input data is insufficient.

    Raises:
        ValueError: If ohlcv is empty or missing required columns.
        ValueError: If profile_type is invalid.
        ValueError: If dynamic_mode is required but not provided.
        ValueError: If value_area_pct is not in (0, 1] range.
        ValueError: If bin_count <= 0.
        TypeError: If ohlcv index is not DatetimeIndex or not UTC-aware.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> n = 100
        >>> idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        >>> data = pd.DataFrame({
        ...     'open': np.random.uniform(100, 110, n),
        ...     'high': np.random.uniform(105, 115, n),
        ...     'low': np.random.uniform(95, 105, n),
        ...     'close': np.random.uniform(100, 110, n),
        ...     'volume': np.random.uniform(1000, 5000, n)
        ... }, index=idx)
        >>> profile = calculate_volume_profile(data, bin_count=50)
        >>> print(profile[profile['is_hvn']])
    """
    # Validate input
    _validate_ohlcv(ohlcv)

    if bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")

    if not (0 < value_area_pct <= 1):
        raise ValueError("value_area_pct must be in range (0, 1]")

    if profile_type not in VALID_PROFILE_TYPES:
        raise ValueError(
            f"Invalid profile_type '{profile_type}'. Must be one of {VALID_PROFILE_TYPES}."
        )

    if profile_type == "dynamic":
        if dynamic_mode is None:
            raise ValueError("dynamic_mode is required when profile_type='dynamic'")

        if dynamic_mode not in VALID_DYNAMIC_MODES:
            raise ValueError(
                f"Invalid dynamic_mode '{dynamic_mode}'. Must be one of {VALID_DYNAMIC_MODES}."
            )

        if dynamic_mode == "atr":
            if atr_period <= 0:
                raise ValueError("atr_period must be a positive integer")
            if atr_mult <= 0:
                raise ValueError("atr_mult must be positive")

        if dynamic_mode == "pct":
            if range_pct is None:
                raise ValueError("range_pct is required when dynamic_mode='pct'")
            if range_pct <= 0:
                raise ValueError("range_pct must be positive")

    # Extract series
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    # Calculate price range and bin edges
    price_min = float(low.min())
    price_max = float(high.max())

    if profile_type == "fixed":
        # Fixed mode: equal-width bins across full range
        if price_min == price_max:
            # Single price - create single bin
            bin_edges = pd.Index([price_min, price_max])
        else:
            bin_edges = pd.Index(np.linspace(price_min, price_max, bin_count + 1))
    else:
        # Dynamic mode
        if dynamic_mode == "atr":
            range_width = _calculate_atr_based_range(high, low, close, atr_period, atr_mult)
        else:  # pct
            range_width = _calculate_pct_based_range(close, range_pct)

        if range_width <= 0:
            range_width = price_max - price_min if price_max > price_min else 1.0

        bin_edges = _build_dynamic_bin_edges(price_min, price_max, range_width)

    # Calculate bin centers
    n_bins = len(bin_edges) - 1
    bin_centers = pd.Series(
        [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)],
        index=range(n_bins),
    )

    # Allocate volume to bins
    bin_volumes = _allocate_volume_to_bins(low, high, volume, bin_edges)

    # Find POC (Point of Control - highest volume bin)
    if bin_volumes.sum() == 0:
        poc_idx = 0
    else:
        poc_idx = int(bin_volumes.idxmax())

    poc_price = float(bin_centers.iloc[poc_idx])

    # Calculate Value Area
    va_low_idx, va_high_idx = _calculate_value_area(bin_volumes, poc_idx, value_area_pct)
    va_low_price = float(bin_edges[va_low_idx])
    va_high_price = float(bin_edges[va_high_idx + 1])

    # Identify HVN and LVN
    is_hvn, is_lvn = _identify_hvn_lvn(
        bin_volumes,
        bin_centers,
        hvn_threshold=HVN_QUANTILE,
        lvn_threshold=LVN_QUANTILE,
    )

    # Classify profile shape
    profile_shape = _classify_profile_shape(bin_volumes, poc_idx)

    # Count candles per bin
    bin_candle_counts = pd.Series(0, index=range(n_bins), dtype=float)
    for i in range(len(low)):
        candle_low = low.iloc[i]
        candle_high = high.iloc[i]
        candle_width = candle_high - candle_low

        if candle_width == 0:
            for j in range(n_bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]
                if bin_low <= candle_low <= bin_high:
                    bin_candle_counts.iloc[j] += 1
                    break
            continue

        for j in range(n_bins):
            bin_low = bin_edges[j]
            bin_high = bin_edges[j + 1]
            if bin_low < candle_high and bin_high > candle_low:
                bin_candle_counts.iloc[j] += 1

    # Build result DataFrame
    result = pd.DataFrame(
        {
            COL_BIN_LOW: bin_edges[:-1].values,
            COL_BIN_HIGH: bin_edges[1:].values,
            COL_BIN_MID: bin_centers.values,
            COL_BIN_VOLUME: bin_volumes.values,
            COL_BIN_CANDLE_COUNT: bin_candle_counts.values,
            COL_IS_HVN: is_hvn.values,
            COL_IS_LVN: is_lvn.values,
        },
        index=bin_centers.index,
    )

    # Add in_value_area column
    result["in_value_area"] = (result.index >= va_low_idx) & (result.index <= va_high_idx)

    # Add profile metadata as index attributes (stored in index name)
    result.index.name = "bin_index"

    # Attach scalar metadata as DataFrame attributes
    result.attrs["poc_price"] = poc_price
    result.attrs["poc_bin_idx"] = poc_idx
    result.attrs["vah_price"] = va_high_price
    result.attrs["val_price"] = va_low_price
    result.attrs["value_area_pct"] = value_area_pct
    result.attrs["profile_shape"] = profile_shape
    result.attrs["profile_type"] = profile_type
    result.attrs["bin_count"] = n_bins
    result.attrs["total_volume"] = float(bin_volumes.sum())

    return result


__all__ = [
    "calculate_volume_profile",
    "ProfileShape",
    "COL_BIN_LOW",
    "COL_BIN_HIGH",
    "COL_BIN_MID",
    "COL_BIN_VOLUME",
    "COL_BIN_CANDLE_COUNT",
    "COL_IS_HVN",
    "COL_IS_LVN",
]
