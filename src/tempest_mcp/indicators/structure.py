"""Structure indicators: Fibonacci, Pivot Points."""

from numbers import Real

import numpy as np
import pandas as pd

# Default Fibonacci levels
DEFAULT_FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
DEFAULT_FIB_EXTENSION_LEVELS = [1.272, 1.618, 2.0, 2.618]


def _validate_swing_high_low(swing_high: float, swing_low: float) -> None:
    """Validate swing_high and swing_low are finite numerics with swing_high > swing_low."""
    if not isinstance(swing_high, Real) or not isinstance(swing_low, Real):
        raise ValueError("swing_high and swing_low must be numeric")
    if not np.isfinite(swing_high) or not np.isfinite(swing_low):
        raise ValueError("swing_high and swing_low must be finite")
    if swing_high <= swing_low:
        raise ValueError("swing_high must be greater than swing_low")


def _validate_levels(levels: list, name: str, min_val: float = None, max_val: float = None) -> None:
    """Validate levels is a sorted list of unique finite numerics within optional bounds."""
    if not isinstance(levels, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if len(levels) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not all(isinstance(lv, Real) and np.isfinite(lv) for lv in levels):
        raise ValueError(f"{name} must contain only finite numeric values")
    if len(levels) != len(set(levels)):
        raise ValueError(f"{name} must contain unique values")
    sorted_levels = sorted(levels)
    if sorted_levels != list(levels):
        raise ValueError(f"{name} must be sorted in ascending order")
    if min_val is not None and any(lv < min_val for lv in levels):
        raise ValueError(f"{name} values must be >= {min_val}")
    if max_val is not None and any(lv > max_val for lv in levels):
        raise ValueError(f"{name} values must be <= {max_val}")


def _validate_tolerance(tolerance: float) -> None:
    """Validate tolerance is a positive finite number."""
    if not isinstance(tolerance, Real) or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite number")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")


def calculate_fib_retracements(
    swing_high: float, swing_low: float, levels: list[float] | None = None
) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels between swing_high and swing_low.

    Parameters
    ----------
    swing_high : float
        The swing high price (must be greater than swing_low)
    swing_low : float
        The swing low price (must be less than swing_high)
    levels : list[float] | None
        Custom retracement levels. Defaults to [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        Each level must be in [0.0, 1.0], strictly increasing, and unique.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - level_type: "retracement"
        - level_ratio: float
        - price: float
        - swing_high: float
        - swing_low: float
        - trend_direction: None

    Raises
    ------
    ValueError
        If swing_high <= swing_low, levels are invalid, or inputs are non-finite.
    """
    _validate_swing_high_low(swing_high, swing_low)

    if levels is None:
        levels = DEFAULT_FIB_RETRACEMENT_LEVELS.copy()
    else:
        _validate_levels(levels, "levels", min_val=0.0, max_val=1.0)
        levels = list(levels)

    diff = swing_high - swing_low

    rows = []
    for ratio in levels:
        price = swing_low + diff * ratio
        rows.append(
            {
                "level_type": "retracement",
                "level_ratio": ratio,
                "price": price,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "trend_direction": None,
            }
        )

    return pd.DataFrame(rows)


def calculate_fib_extensions(
    swing_high: float, swing_low: float, trend_direction: str, levels: list[float] | None = None
) -> pd.DataFrame:
    """
    Calculate Fibonacci extension levels from a swing high/low with explicit trend direction.

    Parameters
    ----------
    swing_high : float
        The swing high price (must be greater than swing_low)
    swing_low : float
        The swing low price (must be less than swing_high)
    trend_direction : str
        Must be "bullish" or "bearish"
    levels : list[float] | None
        Custom extension levels. Defaults to [1.272, 1.618, 2.0, 2.618]
        Each level must be > 1.0, strictly increasing, and unique.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - level_type: "extension"
        - level_ratio: float
        - price: float
        - swing_high: float
        - swing_low: float
        - trend_direction: "bullish" or "bearish"

    Raises
    ------
    ValueError
        If swing_high <= swing_low, trend_direction is invalid, levels are invalid,
        or inputs are non-finite.
    """
    _validate_swing_high_low(swing_high, swing_low)

    if trend_direction not in ("bullish", "bearish"):
        raise ValueError("trend_direction must be 'bullish' or 'bearish'")

    if levels is None:
        levels = DEFAULT_FIB_EXTENSION_LEVELS.copy()
    else:
        _validate_levels(levels, "levels")
        if any(level <= 1.0 for level in levels):
            raise ValueError("levels values must be > 1.0")
        levels = list(levels)

    diff = swing_high - swing_low

    rows = []
    if trend_direction == "bullish":
        # Projects above swing high
        for ratio in levels:
            price = swing_low + diff * ratio
            rows.append(
                {
                    "level_type": "extension",
                    "level_ratio": ratio,
                    "price": price,
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "trend_direction": "bullish",
                }
            )
    else:  # bearish
        # Projects below swing low
        for ratio in levels:
            price = swing_high - diff * ratio
            rows.append(
                {
                    "level_type": "extension",
                    "level_ratio": ratio,
                    "price": price,
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "trend_direction": "bearish",
                }
            )

    return pd.DataFrame(rows)


def detect_fib_confluence(level_sets: list[pd.DataFrame], tolerance: float = 0.5) -> pd.DataFrame:
    """
    Detect confluence zones where precomputed Fibonacci levels cluster within tolerance.

    Parameters
    ----------
    level_sets : list[pd.DataFrame]
        List of precomputed Fibonacci DataFrames from calculate_fib_retracements
        or calculate_fib_extensions. Each must contain columns:
        level_type, level_ratio, price, swing_high, swing_low, trend_direction
    tolerance : float
        Absolute price tolerance in quote units. Must be > 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - cluster_id: int (ascending by cluster_price)
        - cluster_price: float (centroid)
        - tolerance_low: float
        - tolerance_high: float
        - contributor_count: int
        - contributors: list of dicts with source_set_index, level_type, level_ratio,
          price, swing_high, swing_low, trend_direction

        Only clusters with contributor_count >= 2 are returned.
        Empty input returns empty DataFrame with pinned columns.

    Raises
    ------
    ValueError
        If tolerance is invalid or input DataFrames lack required columns.
    """
    _validate_tolerance(tolerance)

    required_cols = {
        "level_type",
        "level_ratio",
        "price",
        "swing_high",
        "swing_low",
        "trend_direction",
    }

    if len(level_sets) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_price",
                "tolerance_low",
                "tolerance_high",
                "contributor_count",
                "contributors",
            ]
        )

    for i, df in enumerate(level_sets):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"level_sets must contain DataFrames, got {type(df)} at index {i}")
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame at index {i} missing required columns: {required_cols - set(df.columns)}"
            )

    # Concatenate inputs with source_set_index
    combined = []
    for set_idx, df in enumerate(level_sets):
        df_copy = df.copy()
        df_copy["source_set_index"] = set_idx
        combined.append(df_copy)

    result_df = pd.concat(combined, ignore_index=True)

    # Sort: price, then source_set_index, level_type, level_ratio
    result_df = result_df.sort_values(
        ["price", "source_set_index", "level_type", "level_ratio"]
    ).reset_index(drop=True)

    if len(result_df) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_price",
                "tolerance_low",
                "tolerance_high",
                "contributor_count",
                "contributors",
            ]
        )

    # Greedy cluster sweep
    clusters = []
    current_cluster = []
    current_centroid = None

    for _, row in result_df.iterrows():
        if current_centroid is None:
            current_centroid = row["price"]
            current_cluster = [row]
        elif abs(row["price"] - current_centroid) <= tolerance:
            current_cluster.append(row)
            # Update centroid
            current_centroid = sum(r["price"] for r in current_cluster) / len(current_cluster)
        else:
            # Finalize current cluster if >= 2 contributors
            if len(current_cluster) >= 2:
                clusters.append(
                    _build_cluster(clusters, current_cluster, current_centroid, tolerance)
                )
            # Start new cluster
            current_centroid = row["price"]
            current_cluster = [row]

    # Don't forget the last cluster
    if len(current_cluster) >= 2:
        clusters.append(_build_cluster(clusters, current_cluster, current_centroid, tolerance))

    if len(clusters) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_price",
                "tolerance_low",
                "tolerance_high",
                "contributor_count",
                "contributors",
            ]
        )

    return pd.DataFrame(clusters)


def _build_cluster(existing_clusters: list, rows: list, centroid: float, tolerance: float) -> dict:
    """Build a cluster dict from a list of rows."""
    contributors = [
        {
            "source_set_index": int(row["source_set_index"]),
            "level_type": row["level_type"],
            "level_ratio": float(row["level_ratio"]),
            "price": float(row["price"]),
            "swing_high": float(row["swing_high"]),
            "swing_low": float(row["swing_low"]),
            "trend_direction": row["trend_direction"],
        }
        for row in rows
    ]

    cluster_id = len(existing_clusters) + 1
    return {
        "cluster_id": cluster_id,
        "cluster_price": round(centroid, 10),
        "tolerance_low": round(centroid - tolerance, 10),
        "tolerance_high": round(centroid + tolerance, 10),
        "contributor_count": len(rows),
        "contributors": contributors,
    }


def calculate_fibonacci_levels(high, low, trend: str = "up"):
    """
    Legacy compatibility wrapper for Fibonacci levels.

    .. deprecated::
        This function is kept for backward compatibility.
        Use :func:`calculate_fib_retracements` for new code.

    Parameters
    ----------
    high : array-like
        High prices (used to find swing_high)
    low : array-like
        Low prices (used to find swing_low)
    trend : str
        "up" or "down" (legacy parameter, retained for compatibility)

    Returns
    -------
    dict
        Dict with swing_high, swing_low, fib_382, fib_500, fib_618
    """
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    swing_high = float(np.max(high_arr))
    swing_low = float(np.min(low_arr))

    diff = swing_high - swing_low
    if trend.lower() == "up":
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "fib_382": swing_low + diff * 0.382,
            "fib_500": swing_low + diff * 0.500,
            "fib_618": swing_low + diff * 0.618,
        }
    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_382": swing_high - diff * 0.382,
        "fib_500": swing_high - diff * 0.500,
        "fib_618": swing_high - diff * 0.618,
    }


def calculate_pivot_points(high, low, close, method: str = "standard"):
    h, lo, c = float(high[-1]), float(low[-1]), float(close[-1])
    pp = (h + lo + c) / 3
    return {
        "pivot": pp,
        "r1": 2 * pp - lo,
        "s1": 2 * pp - h,
        "r2": pp + (h - lo),
        "s2": pp - (h - lo),
    }
