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


# =============================================================================
# Elliott Wave Engine
# =============================================================================

# Output column order for detect_elliott_waves (pinned schema)
_ELLIOTT_OUTPUT_COLUMNS = [
    "sequence_id",
    "sequence_type",
    "wave_label",
    "segment_order",
    "direction",
    "degree",
    "start_ts",
    "end_ts",
    "start_price",
    "end_price",
    "price_delta",
    "retrace_ratio",
    "extension_ratio",
    "overlap_violation",
    "invalidation_violation",
    "is_rule_compliant",
    "is_accepted_sequence",
    "rejection_reason",
]


def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame meets requirements."""
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")
    if ohlcv.empty:
        raise ValueError("ohlcv must not be empty")
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {missing}")
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv index must be a DatetimeIndex")
    if ohlcv.index.tz is None:
        raise ValueError("ohlcv index must be UTC-aware (tz-aware)")
    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("ohlcv index must be monotonically increasing")
    if ohlcv.index.has_duplicates:
        raise ValueError("ohlcv index must not have duplicates")
    for col in ["high", "low", "open", "close", "volume"]:
        if not np.isfinite(ohlcv[col]).all():
            raise ValueError(f"{col} must contain only finite values")
    if (ohlcv["high"] < ohlcv["low"]).any():
        raise ValueError("high must be >= low for all rows")
    if (ohlcv["volume"] < 0).any():
        raise ValueError("volume must be non-negative")


def _validate_elliott_params(
    swing_window: int,
    min_swing_pct: float,
    wave2_retrace_band: tuple,
    wave3_extension_min: float,
    wave4_retrace_max: float,
    waveb_retrace_band: tuple,
    wavec_extension_min: float,
    degree_thresholds: tuple,
) -> None:
    """Validate Elliott Wave detector parameters."""
    if not isinstance(swing_window, int) or swing_window < 1:
        raise ValueError("swing_window must be an integer >= 1")
    if not isinstance(min_swing_pct, Real) or not (0.0 < min_swing_pct < 1.0):
        raise ValueError("min_swing_pct must be a float in (0.0, 1.0)")
    if not isinstance(wave2_retrace_band, tuple) or len(wave2_retrace_band) != 2:
        raise ValueError("wave2_retrace_band must be a tuple of (min, max)")
    if not (0.0 < wave2_retrace_band[0] < wave2_retrace_band[1] < 1.0):
        raise ValueError("wave2_retrace_band values must satisfy 0 < min < max < 1")
    if not isinstance(wave3_extension_min, Real) or wave3_extension_min < 0.0:
        raise ValueError("wave3_extension_min must be a non-negative number")
    if not isinstance(wave4_retrace_max, Real) or not (0.0 < wave4_retrace_max < 1.0):
        raise ValueError("wave4_retrace_max must be a float in (0.0, 1.0)")
    if not isinstance(waveb_retrace_band, tuple) or len(waveb_retrace_band) != 2:
        raise ValueError("waveb_retrace_band must be a tuple of (min, max)")
    if not (0.0 < waveb_retrace_band[0] < waveb_retrace_band[1] < 1.0):
        raise ValueError("waveb_retrace_band values must satisfy 0 < min < max < 1")
    if not isinstance(wavec_extension_min, Real) or wavec_extension_min < 0.0:
        raise ValueError("wavec_extension_min must be a non-negative number")
    if not isinstance(degree_thresholds, tuple) or len(degree_thresholds) != 2:
        raise ValueError("degree_thresholds must be a tuple of (micro_max, minor_max)")
    if not (0.0 < degree_thresholds[0] < degree_thresholds[1]):
        raise ValueError("degree_thresholds must satisfy 0 < micro_max < minor_max")


# TODO(ENG-32): if this helper continues serving multiple detectors, split it into a neutral
# shared helper/module with its own pinned schema and tests.
def _extract_swings(ohlcv: pd.DataFrame, swing_window: int, min_swing_pct: float) -> list[dict]:
    """
    Extract deterministic alternating swing pivots from OHLCV data.

    Returns
    -------
    list[dict]
        Chronologically ordered swing pivots with keys:
        index, kind, price, start_ts, start_price, end_ts, end_price
    """
    n = len(ohlcv)
    swing_candidates: list[dict] = []

    for i in range(swing_window, n - swing_window):
        window_highs = ohlcv["high"].iloc[i - swing_window : i + swing_window + 1].values
        window_lows = ohlcv["low"].iloc[i - swing_window : i + swing_window + 1].values
        current_high = ohlcv["high"].iloc[i]
        current_low = ohlcv["low"].iloc[i]

        if current_high == window_highs.max() and current_high > window_highs.min():
            swing_candidates.append(
                {
                    "index": i,
                    "kind": "high",
                    "price": float(current_high),
                    "end_ts": ohlcv.index[i],
                }
            )

        if current_low == window_lows.min() and current_low < window_lows.max():
            swing_candidates.append(
                {
                    "index": i,
                    "kind": "low",
                    "price": float(current_low),
                    "end_ts": ohlcv.index[i],
                }
            )

    if not swing_candidates:
        return []

    swing_candidates = sorted(
        swing_candidates,
        key=lambda swing: (swing["index"], 0 if swing["kind"] == "low" else 1),
    )

    confirmed_swings: list[dict] = []

    for candidate in swing_candidates:
        if not confirmed_swings:
            start_idx = max(0, candidate["index"] - swing_window)
            confirmed_swings.append(
                {
                    **candidate,
                    "start_ts": ohlcv.index[start_idx],
                    "start_price": float(ohlcv.iloc[start_idx]["close"]),
                    "end_price": candidate["price"],
                }
            )
            continue

        previous_swing = confirmed_swings[-1]

        if candidate["kind"] == previous_swing["kind"]:
            is_more_extreme = (
                candidate["price"] >= previous_swing["price"]
                if candidate["kind"] == "high"
                else candidate["price"] <= previous_swing["price"]
            )
            if is_more_extreme:
                confirmed_swings[-1] = {
                    **candidate,
                    "start_ts": previous_swing["start_ts"],
                    "start_price": previous_swing["start_price"],
                    "end_price": candidate["price"],
                }
            continue

        previous_price = previous_swing["price"]
        pct_move = (
            abs(candidate["price"] - previous_price) / abs(previous_price)
            if previous_price != 0
            else abs(candidate["price"] - previous_price)
        )
        if pct_move < min_swing_pct:
            continue

        confirmed_swings.append(
            {
                **candidate,
                "start_ts": previous_swing["end_ts"],
                "start_price": previous_swing["price"],
                "end_price": candidate["price"],
            }
        )

    return confirmed_swings


def _classify_degree(price_delta: float, prev_legs: list, thresholds: tuple) -> str:
    """Classify wave degree based on normalized move and thresholds."""
    if not prev_legs:
        normalized = abs(price_delta)
    else:
        avg_leg = sum(abs(leg) for leg in prev_legs) / len(prev_legs)
        if avg_leg > 0:
            normalized = abs(price_delta) / avg_leg
        else:
            normalized = abs(price_delta)

    micro_max, minor_max = thresholds
    if normalized < micro_max:
        return "micro"
    elif normalized < minor_max:
        return "minor"
    else:
        return "intermediate"


def _build_impulse_candidates(
    swings: list,
    direction: str,
    wave2_band: tuple,
    wave3_ext_min: float,
    wave4_max: float,
    thresholds: tuple,
    include_rejected: bool,
) -> list:
    """Build impulse wave (5-wave) candidates from alternating swing highs/lows."""
    candidates = []

    if len(swings) < 5:
        return candidates

    expected_kinds = ["high", "low", "high", "low", "high"]
    if direction == "bearish":
        expected_kinds = ["low", "high", "low", "high", "low"]

    for start_idx in range(len(swings) - 4):
        seq_swings = swings[start_idx : start_idx + 5]
        if len(seq_swings) != 5:
            continue
        if [swing["kind"] for swing in seq_swings] != expected_kinds:
            continue

        waves = []
        prev_legs = []
        sequence_id = f"impulse_{direction}_{seq_swings[0]['index']}"

        for i, swing in enumerate(seq_swings):
            wave_num = i + 1
            wave_label = str(wave_num)
            retrace_ratio_out = np.nan
            extension_ratio_out = np.nan

            if i == 0:
                is_accepted = True
                rejection_reason = None
                overlap_violation = False
                invalidation_violation = False
            else:
                prev_leg = abs(waves[i - 1]["price_delta"])
                prev_legs.append(prev_leg)

                if wave_num == 2:
                    wave1 = waves[0]
                    retraced = abs(swing["price"] - wave1["end_price"])
                    wave1_range = abs(wave1["end_price"] - wave1["start_price"])
                    if wave1_range > 0:
                        retrace_ratio = retraced / wave1_range
                    else:
                        retrace_ratio = 0.0

                    if not (wave2_band[0] <= retrace_ratio <= wave2_band[1]):
                        is_accepted = False
                        rejection_reason = f"wave2_retrace_{retrace_ratio:.3f}_not_in_band"
                    else:
                        is_accepted = True
                        rejection_reason = None

                    retrace_ratio_out = retrace_ratio
                    extension_ratio_out = np.nan

                    overlap_violation = False
                    invalidation_violation = False

                elif wave_num == 3:
                    wave1 = waves[0]
                    wave3_price_diff = (
                        swing["price"] - wave1["end_price"]
                        if direction == "bullish"
                        else wave1["end_price"] - swing["price"]
                    )
                    wave1_range = abs(wave1["end_price"] - wave1["start_price"])
                    ext_ratio = wave3_price_diff / wave1_range if wave1_range > 0 else 0.0

                    if ext_ratio < wave3_ext_min:
                        is_accepted = False
                        rejection_reason = f"wave3_extension_{ext_ratio:.3f}_below_min"
                    else:
                        is_accepted = True
                        rejection_reason = None

                    retrace_ratio_out = np.nan
                    extension_ratio_out = ext_ratio

                    overlap_violation = False
                    invalidation_violation = False

                elif wave_num == 4:
                    wave3 = waves[2]

                    retraced = abs(swing["price"] - wave3["end_price"])
                    wave3_range = abs(wave3["end_price"] - wave3["start_price"])
                    if wave3_range > 0:
                        retrace_ratio = retraced / wave3_range
                    else:
                        retrace_ratio = 0.0

                    if retrace_ratio > wave4_max:
                        is_accepted = False
                        rejection_reason = f"wave4_retrace_{retrace_ratio:.3f}_exceeds_max"
                    else:
                        is_accepted = True
                        rejection_reason = None

                    retrace_ratio_out = retrace_ratio
                    extension_ratio_out = np.nan

                    wave1 = waves[0]
                    overlap_violation = (
                        swing["price"] <= wave1["end_price"]
                        if direction == "bullish"
                        else swing["price"] >= wave1["end_price"]
                    )
                    if overlap_violation:
                        rejection_reason = "wave4_overlap_violation"
                    invalidation_violation = False

                elif wave_num == 5:
                    wave3 = waves[2]
                    wave4 = waves[3]

                    wave4_range = abs(wave4["end_price"] - wave4["start_price"])
                    wave3_range = abs(wave3["end_price"] - wave3["start_price"])
                    ext_ratio = (
                        abs(swing["price"] - wave4["end_price"]) / wave4_range
                        if wave4_range > 0
                        else 0.0
                    )

                    is_accepted = True
                    rejection_reason = None
                    retrace_ratio_out = np.nan
                    extension_ratio_out = ext_ratio

                    overlap_violation = False
                    invalidation_violation = (
                        swing["price"] <= wave4["end_price"]
                        if direction == "bullish"
                        else swing["price"] >= wave4["end_price"]
                    )
                    if invalidation_violation:
                        rejection_reason = "wave5_invalidation_violation"

                else:
                    is_accepted = True
                    rejection_reason = None
                    retrace_ratio_out = np.nan
                    extension_ratio_out = np.nan
                    overlap_violation = False
                    invalidation_violation = False

            price_delta = swing["price"] - swing["start_price"]

            degree = _classify_degree(price_delta, prev_legs, thresholds)

            is_rule_compliant = is_accepted and not overlap_violation and not invalidation_violation

            waves.append(
                {
                    "sequence_id": sequence_id,
                    "sequence_type": "impulse",
                    "wave_label": wave_label,
                    "segment_order": wave_num,
                    "direction": direction,
                    "degree": degree,
                    "start_ts": swing["start_ts"],
                    "end_ts": swing["end_ts"],
                    "start_price": swing["start_price"],
                    "end_price": swing["price"],
                    "price_delta": price_delta,
                    "retrace_ratio": retrace_ratio_out if wave_num in (2, 4) else np.nan,
                    "extension_ratio": extension_ratio_out if wave_num in (3, 5) else np.nan,
                    "overlap_violation": overlap_violation if wave_num in (2, 4, 5) else False,
                    "invalidation_violation": invalidation_violation if wave_num == 5 else False,
                    "is_rule_compliant": is_rule_compliant,
                    "is_accepted_sequence": False,
                    "rejection_reason": rejection_reason,
                }
            )

        sequence_accepted = all(w["is_rule_compliant"] for w in waves)
        for wave in waves:
            wave["is_accepted_sequence"] = sequence_accepted

        if include_rejected or sequence_accepted:
            candidates.extend(waves)

    return candidates


def _build_corrective_candidates(
    swings: list,
    direction: str,
    waveb_band: tuple,
    wavec_ext_min: float,
    thresholds: tuple,
    include_rejected: bool,
) -> list:
    """Build corrective wave (A-B-C) candidates from alternating swing highs/lows."""
    candidates = []

    if len(swings) < 3:
        return candidates

    expected_kinds = ["low", "high", "low"]
    if direction == "bearish":
        expected_kinds = ["high", "low", "high"]

    for start_idx in range(len(swings) - 2):
        seq_swings = swings[start_idx : start_idx + 3]
        if len(seq_swings) != 3:
            continue
        if [swing["kind"] for swing in seq_swings] != expected_kinds:
            continue

        waves = []
        prev_legs = []
        sequence_id = f"corrective_{direction}_{seq_swings[0]['index']}"

        for i, swing in enumerate(seq_swings):
            wave_label = ["A", "B", "C"][i]

            if i == 0:
                is_accepted = True
                rejection_reason = None
                retrace_ratio_out = np.nan
                extension_ratio_out = np.nan
                overlap_violation = False
                invalidation_violation = False
            elif i == 1:
                wave_a = waves[0]
                retraced = abs(swing["price"] - wave_a["end_price"])
                wave_a_range = abs(wave_a["end_price"] - wave_a["start_price"])
                if wave_a_range > 0:
                    retrace_ratio = retraced / wave_a_range
                else:
                    retrace_ratio = 0.0

                if not (waveb_band[0] <= retrace_ratio <= waveb_band[1]):
                    is_accepted = False
                    rejection_reason = f"waveb_retrace_{retrace_ratio:.3f}_not_in_band"
                else:
                    is_accepted = True
                    rejection_reason = None

                retrace_ratio_out = retrace_ratio
                extension_ratio_out = np.nan
                overlap_violation = False
                invalidation_violation = False

            else:
                wave_a = waves[0]
                wave_b = waves[1]

                wave_a_range = abs(wave_a["end_price"] - wave_a["start_price"])
                ext_ratio = (
                    abs(swing["price"] - wave_b["end_price"]) / wave_a_range
                    if wave_a_range > 0
                    else 0.0
                )

                if ext_ratio < wavec_ext_min:
                    is_accepted = False
                    rejection_reason = f"wavec_extension_{ext_ratio:.3f}_below_min"
                else:
                    is_accepted = True
                    rejection_reason = None

                retrace_ratio_out = np.nan
                extension_ratio_out = ext_ratio

                overlap_violation = (
                    swing["price"] >= wave_a["start_price"]
                    if direction == "bullish"
                    else swing["price"] <= wave_a["start_price"]
                )
                invalidation_violation = False

            if i > 0:
                prev_leg = abs(waves[i - 1]["price_delta"])
                prev_legs.append(prev_leg)

            if i == 2 and overlap_violation:
                rejection_reason = "wavec_overlap_violation"

            price_delta = swing["price"] - swing["start_price"]

            degree = _classify_degree(price_delta, prev_legs, thresholds)

            is_rule_compliant = is_accepted and not overlap_violation

            waves.append(
                {
                    "sequence_id": sequence_id,
                    "sequence_type": "corrective",
                    "wave_label": wave_label,
                    "segment_order": i + 1,
                    "direction": direction,
                    "degree": degree,
                    "start_ts": swing["start_ts"],
                    "end_ts": swing["end_ts"],
                    "start_price": swing["start_price"],
                    "end_price": swing["price"],
                    "price_delta": price_delta,
                    "retrace_ratio": retrace_ratio_out if i > 0 else np.nan,
                    "extension_ratio": extension_ratio_out if i == 2 else np.nan,
                    "overlap_violation": overlap_violation if i == 2 else False,
                    "invalidation_violation": invalidation_violation if i == 2 else False,
                    "is_rule_compliant": is_rule_compliant,
                    "is_accepted_sequence": False,
                    "rejection_reason": rejection_reason,
                }
            )

        sequence_accepted = all(w["is_rule_compliant"] for w in waves)
        for wave in waves:
            wave["is_accepted_sequence"] = sequence_accepted

        if include_rejected or sequence_accepted:
            candidates.extend(waves)

    return candidates


def detect_elliott_waves(
    ohlcv: pd.DataFrame,
    *,
    swing_window: int = 2,
    min_swing_pct: float = 0.05,
    wave2_retrace_band: tuple[float, float] = (0.382, 0.786),
    wave3_extension_min: float = 1.0,
    wave4_retrace_max: float = 0.618,
    waveb_retrace_band: tuple[float, float] = (0.382, 0.886),
    wavec_extension_min: float = 1.0,
    degree_thresholds: tuple[float, float] = (0.02, 0.08),
    include_rejected: bool = True,
) -> pd.DataFrame:
    """
    Detect Elliott Wave patterns from OHLCV data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Windowed OHLCV data with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume. Must be monotonically increasing
        with no duplicates.
    swing_window : int, optional
        Window size for swing detection. Default is 2.
    min_swing_pct : float, optional
        Minimum percentage move to qualify as a swing. Default is 0.05 (5%).
    wave2_retrace_band : tuple[float, float], optional
        Acceptable retracement range for wave 2 as (min, max).
        Default is (0.382, 0.786).
    wave3_extension_min : float, optional
        Minimum extension ratio for wave 3 relative to wave 1.
        Default is 1.0.
    wave4_retrace_max : float, optional
        Maximum retracement for wave 4 relative to wave 3.
        Default is 0.618.
    waveb_retrace_band : tuple[float, float], optional
        Acceptable retracement range for wave B as (min, max).
        Default is (0.382, 0.886).
    wavec_extension_min : float, optional
        Minimum extension ratio for wave C relative to wave A.
        Default is 1.0.
    degree_thresholds : tuple[float, float], optional
        Thresholds for degree classification as (micro_max, minor_max).
        Default is (0.02, 0.08).
    include_rejected : bool, optional
        If True, include rejected candidates in output. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - sequence_id: str
        - sequence_type: 'impulse' or 'corrective'
        - wave_label: '1'-'5' or 'A'-'C'
        - segment_order: int
        - direction: 'bullish' or 'bearish'
        - degree: 'micro', 'minor', or 'intermediate'
        - start_ts: pd.Timestamp
        - end_ts: pd.Timestamp
        - start_price: float
        - end_price: float
        - price_delta: float
        - retrace_ratio: float or NaN (impulse waves 2/4, corrective wave B)
        - extension_ratio: float or NaN (impulse waves 3/5, corrective wave C)
        - overlap_violation: bool
        - invalidation_violation: bool
        - is_rule_compliant: bool
        - is_accepted_sequence: bool
        - rejection_reason: str or None

        Rows are ordered by sequence_id ASC, segment_order ASC.

    Raises
    ------
    ValueError
        If OHLCV or parameters fail validation.
    """
    _validate_ohlcv(ohlcv)
    _validate_elliott_params(
        swing_window,
        min_swing_pct,
        wave2_retrace_band,
        wave3_extension_min,
        wave4_retrace_max,
        waveb_retrace_band,
        wavec_extension_min,
        degree_thresholds,
    )

    swings = _extract_swings(ohlcv, swing_window, min_swing_pct)

    all_waves = []

    if swings:
        all_waves.extend(
            _build_impulse_candidates(
                swings,
                "bullish",
                wave2_retrace_band,
                wave3_extension_min,
                wave4_retrace_max,
                degree_thresholds,
                include_rejected,
            )
        )
        all_waves.extend(
            _build_corrective_candidates(
                swings,
                "bullish",
                waveb_retrace_band,
                wavec_extension_min,
                degree_thresholds,
                include_rejected,
            )
        )
        all_waves.extend(
            _build_impulse_candidates(
                swings,
                "bearish",
                wave2_retrace_band,
                wave3_extension_min,
                wave4_retrace_max,
                degree_thresholds,
                include_rejected,
            )
        )
        all_waves.extend(
            _build_corrective_candidates(
                swings,
                "bearish",
                waveb_retrace_band,
                wavec_extension_min,
                degree_thresholds,
                include_rejected,
            )
        )

    if not all_waves:
        return pd.DataFrame(columns=_ELLIOTT_OUTPUT_COLUMNS)

    result_df = pd.DataFrame(all_waves)
    result_df = result_df.sort_values(["sequence_id", "segment_order"]).reset_index(drop=True)

    for col in _ELLIOTT_OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = np.nan

    return result_df[_ELLIOTT_OUTPUT_COLUMNS]


# =============================================================================
# Price Pattern Engine — Swing Points, Market Structure, Range & Breakout
# =============================================================================

# Output column order for detect_swing_points (pinned schema)
_SWING_OUTPUT_COLUMNS = [
    "swing_id",
    "pivot_index",
    "swing_type",
    "pivot_ts",
    "pivot_price",
    "leg_start_ts",
    "leg_start_price",
    "leg_end_ts",
    "leg_end_price",
    "price_delta",
    "pct_delta",
]

# Output column order for classify_market_structure (pinned schema)
_STRUCTURE_OUTPUT_COLUMNS = [
    "event_id",
    "event_ts",
    "swing_id",
    "swing_type",
    "classification",
    "reference_swing_id",
    "reference_price",
    "current_price",
    "price_delta",
    "pct_delta",
    "trend_state",
]

# Output column order for detect_price_ranges (pinned schema)
_RANGE_OUTPUT_COLUMNS = [
    "range_id",
    "start_ts",
    "end_ts",
    "range_high",
    "range_low",
    "range_mid",
    "range_width",
    "range_width_pct",
    "bars_evaluated",
    "containment_ratio",
    "status",
]

# Output column order for detect_range_breakouts (pinned schema)
_BREAKOUT_OUTPUT_COLUMNS = [
    "breakout_id",
    "range_id",
    "breakout_ts",
    "direction",
    "breakout_price",
    "boundary_price",
    "distance",
    "distance_pct",
    "confirm_bars",
]


def _validate_swing_params(swing_window: int, min_swing_pct: float) -> None:
    """Validate swing detection parameters."""
    if not isinstance(swing_window, int) or swing_window < 1:
        raise ValueError("swing_window must be an integer >= 1")
    if not isinstance(min_swing_pct, Real) or not (0.0 <= min_swing_pct < 1.0):
        raise ValueError("min_swing_pct must be a float in [0.0, 1.0)")


def _validate_structure_df(swings: pd.DataFrame) -> None:
    """Validate swings DataFrame from detect_swing_points."""
    if not isinstance(swings, pd.DataFrame):
        raise ValueError("swings must be a pandas DataFrame")
    required_cols = set(_SWING_OUTPUT_COLUMNS)
    if not required_cols.issubset(set(swings.columns)):
        missing = required_cols - set(swings.columns)
        raise ValueError(f"swings missing required columns: {missing}")


def _validate_equal_epsilon(equal_epsilon: float) -> None:
    """Validate market-structure equality tolerance."""
    if not isinstance(equal_epsilon, Real) or not np.isfinite(equal_epsilon) or equal_epsilon < 0.0:
        raise ValueError("equal_epsilon must be a finite float >= 0.0")


def _classify_trend_state(
    latest_high_classification: str | None, latest_low_classification: str | None
) -> str:
    """Infer trend state from the most recent classified high/low pair."""
    if latest_high_classification is None or latest_low_classification is None:
        return "transition"

    if latest_high_classification in ("HH", "EH") and latest_low_classification == "HL":
        return "bullish"

    if latest_high_classification == "LH" and latest_low_classification in ("LL", "EL"):
        return "bearish"

    if latest_high_classification == "LH" and latest_low_classification == "HL":
        return "transition"

    return "range"


def _validate_range_params(
    range_lookback: int,
    max_range_pct: float,
    containment_ratio: float,
    boundary_buffer_pct: float,
) -> None:
    """Validate range detection parameters."""
    if not isinstance(range_lookback, int) or range_lookback < 2:
        raise ValueError("range_lookback must be an integer >= 2")
    if not isinstance(max_range_pct, Real) or not (0.0 < max_range_pct < 1.0):
        raise ValueError("max_range_pct must be a float in (0.0, 1.0)")
    if not isinstance(containment_ratio, Real) or not (0.0 < containment_ratio <= 1.0):
        raise ValueError("containment_ratio must be a float in (0.0, 1.0]")
    if not isinstance(boundary_buffer_pct, Real) or not (0.0 <= boundary_buffer_pct < 1.0):
        raise ValueError("boundary_buffer_pct must be a float in [0.0, 1.0)")


def _validate_breakout_params(breakout_confirm_bars: int, breakout_buffer_pct: float) -> None:
    """Validate breakout detection parameters."""
    if not isinstance(breakout_confirm_bars, int) or breakout_confirm_bars < 1:
        raise ValueError("breakout_confirm_bars must be an integer >= 1")
    if not isinstance(breakout_buffer_pct, Real) or not (0.0 <= breakout_buffer_pct < 1.0):
        raise ValueError("breakout_buffer_pct must be a float in [0.0, 1.0)")


def detect_swing_points(
    ohlcv: pd.DataFrame,
    *,
    swing_window: int = 2,
    min_swing_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Detect deterministic swing pivot points from already-windowed OHLCV input.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Windowed OHLCV data with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume. Must be monotonically increasing
        with no duplicates.
    swing_window : int, optional
        Window size for swing detection. Default is 2.
    min_swing_pct : float, optional
        Minimum percentage move to qualify as a swing. Default is 0.02 (2%).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - swing_id: int (ascending, 1-based)
        - pivot_index: int
        - swing_type: 'high' or 'low'
        - pivot_ts: pd.Timestamp
        - pivot_price: float
        - leg_start_ts: pd.Timestamp
        - leg_start_price: float
        - leg_end_ts: pd.Timestamp
        - leg_end_price: float
        - price_delta: float
        - pct_delta: float

        Rows are ordered by pivot_index ASC, then swing_type stable tie-break
        (lows before highs at the same index).

    Raises
    ------
    ValueError
        If OHLCV fails validation or parameters are invalid.
    """
    _validate_ohlcv(ohlcv)
    _validate_swing_params(swing_window, min_swing_pct)

    # Reuse Elliott internal swing extraction (doesn't change Elliott public contract)
    swings = _extract_swings(ohlcv, swing_window, min_swing_pct)

    if not swings:
        return pd.DataFrame(columns=_SWING_OUTPUT_COLUMNS)

    rows = []
    for idx, swing in enumerate(swings, start=1):
        price_delta = swing["price"] - swing["start_price"]
        prev_price = swing["start_price"]
        pct_delta = abs(price_delta) / abs(prev_price) if prev_price != 0 else 0.0

        rows.append(
            {
                "swing_id": idx,
                "pivot_index": swing["index"],
                "swing_type": swing["kind"],
                "pivot_ts": swing["end_ts"],
                "pivot_price": swing["price"],
                "leg_start_ts": swing["start_ts"],
                "leg_start_price": swing["start_price"],
                "leg_end_ts": swing["end_ts"],
                "leg_end_price": swing["end_price"],
                "price_delta": price_delta,
                "pct_delta": pct_delta,
            }
        )

    result_df = pd.DataFrame(rows)
    # Deterministic ordering: pivot_index ASC, swing_type tie-break (low before high)
    result_df["_swing_order"] = result_df["swing_type"].map({"low": 0, "high": 1})
    result_df = result_df.sort_values(["pivot_index", "_swing_order"]).reset_index(drop=True)
    result_df["swing_id"] = range(1, len(result_df) + 1)
    result_df = result_df.drop(columns=["_swing_order"])

    return result_df[_SWING_OUTPUT_COLUMNS]


def classify_market_structure(
    swings: pd.DataFrame,
    *,
    equal_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """
    Classify deterministic market structure transitions from swing points.

    Classifies each swing as HH (Higher High), HL (Higher Low), LH (Lower High),
    LL (Lower Low), EH (Equal High), or EL (Equal Low) based on comparison to the
    prior swing of the same type.

    Parameters
    ----------
    swings : pd.DataFrame
        DataFrame from detect_swing_points with columns:
        swing_id, pivot_index, swing_type, pivot_ts, pivot_price,
        leg_start_ts, leg_start_price, leg_end_ts, leg_end_price,
        price_delta, pct_delta.
    equal_epsilon : float, optional
        Tolerance for considering prices equal. Default is 1e-9.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - event_id: int (ascending, 1-based)
        - event_ts: pd.Timestamp
        - swing_id: int
        - swing_type: 'high' or 'low'
        - classification: 'HH', 'HL', 'LH', 'LL', 'EH', or 'EL'
        - reference_swing_id: int
        - reference_price: float
        - current_price: float
        - price_delta: float
        - pct_delta: float
        - trend_state: 'bullish', 'bearish', 'transition', or 'range'

        Rows are ordered by event_id ASC.

    Raises
    ------
    ValueError
        If swings DataFrame fails validation or parameters are invalid.
    """
    _validate_structure_df(swings)
    _validate_equal_epsilon(equal_epsilon)

    if len(swings) == 0:
        return pd.DataFrame(columns=_STRUCTURE_OUTPUT_COLUMNS)

    # Separate highs and lows, preserving order
    highs = swings[swings["swing_type"] == "high"].sort_values("swing_id").reset_index(drop=True)
    lows = swings[swings["swing_type"] == "low"].sort_values("swing_id").reset_index(drop=True)

    rows = []
    event_id = 1

    # Process all swings in chronological order by pivot_index.
    # When a bar is both a high and a low pivot, keep the low-before-high
    # convention used by detect_swing_points.
    all_swings_sorted = swings.copy()
    all_swings_sorted["_swing_order"] = all_swings_sorted["swing_type"].map({"low": 0, "high": 1})
    all_swings_sorted = all_swings_sorted.sort_values(["pivot_index", "_swing_order", "swing_id"])
    all_swings_sorted = all_swings_sorted.drop(columns=["_swing_order"]).reset_index(drop=True)

    for _, swing in all_swings_sorted.iterrows():
        if swing["swing_type"] == "high":
            if len(highs) < 2:
                # First high has no prior to compare
                classification = None
                reference_swing_id = None
                reference_price = None
            else:
                # Find prior high
                prior_high_idx = highs[highs["swing_id"] < swing["swing_id"]].index
                if len(prior_high_idx) == 0:
                    classification = None
                    reference_swing_id = None
                    reference_price = None
                else:
                    prior_high = highs.loc[prior_high_idx[-1]]
                    reference_swing_id = int(prior_high["swing_id"])
                    reference_price = prior_high["pivot_price"]

                    price_diff = swing["pivot_price"] - reference_price
                    if abs(price_diff) <= equal_epsilon:
                        classification = "EH"
                    elif price_diff > 0:
                        classification = "HH"
                    else:
                        classification = "LH"

                    current_price = swing["pivot_price"]
                    price_delta = current_price - reference_price
                    pct_delta = (
                        abs(price_delta) / abs(reference_price) if reference_price != 0 else 0.0
                    )
        else:  # low
            if len(lows) < 2:
                classification = None
                reference_swing_id = None
                reference_price = None
            else:
                prior_low_idx = lows[lows["swing_id"] < swing["swing_id"]].index
                if len(prior_low_idx) == 0:
                    classification = None
                    reference_swing_id = None
                    reference_price = None
                else:
                    prior_low = lows.loc[prior_low_idx[-1]]
                    reference_swing_id = int(prior_low["swing_id"])
                    reference_price = prior_low["pivot_price"]

                    price_diff = swing["pivot_price"] - reference_price
                    if abs(price_diff) <= equal_epsilon:
                        classification = "EL"
                    elif price_diff < 0:
                        classification = "LL"
                    else:
                        classification = "HL"

                    current_price = swing["pivot_price"]
                    price_delta = current_price - reference_price
                    pct_delta = (
                        abs(price_delta) / abs(reference_price) if reference_price != 0 else 0.0
                    )

        if classification is not None:
            rows.append(
                {
                    "event_id": event_id,
                    "event_ts": swing["pivot_ts"],
                    "swing_id": int(swing["swing_id"]),
                    "swing_type": swing["swing_type"],
                    "classification": classification,
                    "reference_swing_id": int(reference_swing_id),
                    "reference_price": reference_price,
                    "current_price": swing["pivot_price"],
                    "price_delta": price_delta,
                    "pct_delta": pct_delta,
                    "trend_state": None,  # Will be filled after all rows processed
                }
            )
            event_id += 1

    if not rows:
        return pd.DataFrame(columns=_STRUCTURE_OUTPUT_COLUMNS)

    result_df = pd.DataFrame(rows)

    latest_high_classification = None
    latest_low_classification = None
    trend_states = []

    for row in result_df.itertuples(index=False):
        if row.swing_type == "high":
            latest_high_classification = row.classification
        else:
            latest_low_classification = row.classification

        trend_states.append(
            _classify_trend_state(latest_high_classification, latest_low_classification)
        )

    result_df["trend_state"] = trend_states

    return result_df[_STRUCTURE_OUTPUT_COLUMNS]


def detect_price_ranges(
    ohlcv: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    range_lookback: int = 20,
    max_range_pct: float = 0.03,
    containment_ratio: float = 0.8,
    boundary_buffer_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Detect deterministic range-bound periods from OHLCV and swing data.

    A range is detected when price consolidates within a tight band over
    the lookback window, with highs and lows respecting boundaries.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Windowed OHLCV data with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume. Must be monotonically increasing
        with no duplicates.
    swings : pd.DataFrame
        DataFrame from detect_swing_points.
    range_lookback : int, optional
        Number of bars to evaluate for range detection. Default is 20.
    max_range_pct : float, optional
        Maximum range width as percentage of range midpoint. Default is 0.03 (3%).
    containment_ratio : float, optional
        Minimum ratio of bars that must be within range boundaries. Default is 0.8.
    boundary_buffer_pct : float, optional
        Buffer around boundaries as percentage. Default is 0.001 (0.1%).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - range_id: int (ascending, 1-based)
        - start_ts: pd.Timestamp
        - end_ts: pd.Timestamp
        - range_high: float
        - range_low: float
        - range_mid: float
        - range_width: float
        - range_width_pct: float
        - bars_evaluated: int
        - containment_ratio: float
        - status: 'active', 'broken_up', or 'broken_down'

    Raises
    ------
    ValueError
        If inputs fail validation or parameters are invalid.
    """
    _validate_ohlcv(ohlcv)
    _validate_structure_df(swings)
    _validate_range_params(range_lookback, max_range_pct, containment_ratio, boundary_buffer_pct)

    if len(swings) < 4:
        return pd.DataFrame(columns=_RANGE_OUTPUT_COLUMNS)

    rows = []
    range_id = 1

    # Get swing highs and lows with their timestamps
    swing_highs = (
        swings[swings["swing_type"] == "high"].sort_values("pivot_index").reset_index(drop=True)
    )
    swing_lows = (
        swings[swings["swing_type"] == "low"].sort_values("pivot_index").reset_index(drop=True)
    )

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return pd.DataFrame(columns=_RANGE_OUTPUT_COLUMNS)

    # Evaluate rolling windows of swings
    for start_idx in range(len(swing_highs) - 1):
        for end_idx in range(start_idx + 1, min(start_idx + range_lookback, len(swing_highs))):
            # Get swing subset for this window
            window_highs = swing_highs.iloc[start_idx : end_idx + 1]
            window_lows_subset = swing_lows[
                (swing_lows["pivot_index"] >= window_highs["pivot_index"].iloc[0])
                & (swing_lows["pivot_index"] <= window_highs["pivot_index"].iloc[-1])
            ]

            if len(window_highs) < 2 or len(window_lows_subset) < 2:
                continue

            range_high = window_highs["pivot_price"].max()
            range_low = window_lows_subset["pivot_price"].min()
            range_mid = (range_high + range_low) / 2.0
            range_width = range_high - range_low
            range_width_pct = range_width / range_mid if range_mid != 0 else 0.0

            # Check width threshold
            if range_width_pct > max_range_pct:
                continue

            # Get OHLCV slice for this range
            start_ts = window_highs["pivot_ts"].iloc[0]
            end_ts = window_highs["pivot_ts"].iloc[-1]

            # Align to actual bar times in ohlcv
            ohlcv_slice = ohlcv[(ohlcv.index >= start_ts) & (ohlcv.index <= end_ts)]

            if len(ohlcv_slice) == 0:
                continue

            bars_evaluated = len(ohlcv_slice)

            # Calculate containment: bars within range boundaries with buffer
            buffer = range_width * boundary_buffer_pct
            high_bound = range_high - buffer
            low_bound = range_low + buffer

            contained_bars = (
                (ohlcv_slice["close"] >= low_bound) & (ohlcv_slice["close"] <= high_bound)
            ).sum()
            actual_containment = contained_bars / bars_evaluated if bars_evaluated > 0 else 0.0

            if actual_containment + 1e-9 < containment_ratio:
                continue

            # Determine status based on current price position
            last_close = ohlcv_slice["close"].iloc[-1]
            if last_close > high_bound:
                status = "broken_up"
            elif last_close < low_bound:
                status = "broken_down"
            else:
                status = "active"

            rows.append(
                {
                    "range_id": range_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_mid": range_mid,
                    "range_width": range_width,
                    "range_width_pct": range_width_pct,
                    "bars_evaluated": bars_evaluated,
                    "containment_ratio": round(actual_containment, 4),
                    "status": status,
                }
            )
            range_id += 1

    if not rows:
        return pd.DataFrame(columns=_RANGE_OUTPUT_COLUMNS)

    result_df = pd.DataFrame(rows)
    # Deduplicate only exact temporal boundary repeats; distinct ranges may legitimately
    # share the same price boundaries while ending at different timestamps.
    result_df = result_df.drop_duplicates(
        subset=["start_ts", "end_ts", "range_high", "range_low"], keep="first"
    )
    result_df = result_df.sort_values(
        ["start_ts", "end_ts", "range_high", "range_low"]
    ).reset_index(drop=True)
    result_df["range_id"] = range(1, len(result_df) + 1)

    return result_df[_RANGE_OUTPUT_COLUMNS]


def detect_range_breakouts(
    ohlcv: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    breakout_confirm_bars: int = 1,
    breakout_buffer_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Detect deterministic breakout events from range boundaries.

    Identifies upward or downward structural breakouts relative to
    pinned range definitions from detect_price_ranges.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Windowed OHLCV data with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume. Must be monotonically increasing
        with no duplicates.
    ranges : pd.DataFrame
        DataFrame from detect_price_ranges.
    breakout_confirm_bars : int, optional
        Number of consecutive bars required to confirm breakout. Default is 1.
    breakout_buffer_pct : float, optional
        Buffer above/below range boundaries for breakout detection.
        Default is 0.001 (0.1%).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (in order):
        - breakout_id: int (ascending, 1-based)
        - range_id: int
        - breakout_ts: pd.Timestamp
        - direction: 'up' or 'down'
        - breakout_price: float
        - boundary_price: float
        - distance: float
        - distance_pct: float
        - confirm_bars: int

    Raises
    ------
    ValueError
        If inputs fail validation or parameters are invalid.
    """
    _validate_ohlcv(ohlcv)
    _validate_breakout_params(breakout_confirm_bars, breakout_buffer_pct)

    if not isinstance(ranges, pd.DataFrame):
        raise ValueError("ranges must be a pandas DataFrame")

    required_range_cols = set(_RANGE_OUTPUT_COLUMNS)
    if not required_range_cols.issubset(set(ranges.columns)):
        missing = required_range_cols - set(ranges.columns)
        raise ValueError(f"ranges missing required columns: {missing}")

    if len(ranges) == 0:
        return pd.DataFrame(columns=_BREAKOUT_OUTPUT_COLUMNS)

    rows = []
    breakout_id = 1

    for _, range_row in ranges.iterrows():
        range_id = int(range_row["range_id"])
        range_high = range_row["range_high"]
        range_low = range_row["range_low"]
        end_ts = range_row["end_ts"]

        # Include the end_ts bar so ranges already marked broken at the
        # boundary bar can emit a matching breakout event.
        post_range_ohlcv = ohlcv[ohlcv.index >= end_ts]

        if len(post_range_ohlcv) < breakout_confirm_bars:
            continue

        buffer = (range_high - range_low) * breakout_buffer_pct

        # Check for upward breakout: close > range_high + buffer for confirm_bars
        up_breakout_idx = None
        for i in range(len(post_range_ohlcv) - breakout_confirm_bars + 1):
            window = post_range_ohlcv.iloc[i : i + breakout_confirm_bars]
            if all(window["close"] > range_high + buffer):
                up_breakout_idx = i
                break

        # Check for downward breakout: close < range_low - buffer for confirm_bars
        down_breakout_idx = None
        for i in range(len(post_range_ohlcv) - breakout_confirm_bars + 1):
            window = post_range_ohlcv.iloc[i : i + breakout_confirm_bars]
            if all(window["close"] < range_low - buffer):
                down_breakout_idx = i
                break

        # Emit first qualifying breakout per range (deterministic no-dup policy)
        if up_breakout_idx is not None and down_breakout_idx is not None:
            # Both detected - take the earlier one
            if up_breakout_idx <= down_breakout_idx:
                breakout_idx = up_breakout_idx
                direction = "up"
                boundary_price = range_high
                breakout_price = post_range_ohlcv.iloc[breakout_idx]["close"]
                distance = breakout_price - range_high
            else:
                breakout_idx = down_breakout_idx
                direction = "down"
                boundary_price = range_low
                breakout_price = post_range_ohlcv.iloc[breakout_idx]["close"]
                distance = range_low - breakout_price
        elif up_breakout_idx is not None:
            breakout_idx = up_breakout_idx
            direction = "up"
            boundary_price = range_high
            breakout_price = post_range_ohlcv.iloc[breakout_idx]["close"]
            distance = breakout_price - range_high
        elif down_breakout_idx is not None:
            breakout_idx = down_breakout_idx
            direction = "down"
            boundary_price = range_low
            breakout_price = post_range_ohlcv.iloc[breakout_idx]["close"]
            distance = range_low - breakout_price
        else:
            continue

        distance_pct = distance / boundary_price if boundary_price != 0 else 0.0

        rows.append(
            {
                "breakout_id": breakout_id,
                "range_id": range_id,
                "breakout_ts": post_range_ohlcv.index[breakout_idx],
                "direction": direction,
                "breakout_price": breakout_price,
                "boundary_price": boundary_price,
                "distance": distance,
                "distance_pct": distance_pct,
                "confirm_bars": breakout_confirm_bars,
            }
        )
        breakout_id += 1

    if not rows:
        return pd.DataFrame(columns=_BREAKOUT_OUTPUT_COLUMNS)

    result_df = pd.DataFrame(rows)
    return result_df[_BREAKOUT_OUTPUT_COLUMNS]


# =============================================================================
# Market Structure Summary Engine — Composition Layer
# =============================================================================

# Output column order for summarize_market_structure (pinned schema)
_SUMMARY_OUTPUT_COLUMNS = [
    "analysis_ts",
    "window_start_ts",
    "window_end_ts",
    "summary_label",
    "decision_rule",
    "structure_event_ts",
    "structure_classification",
    "structure_trend_state",
    "adx",
    "plus_di",
    "minus_di",
    "di_spread",
    "range_id",
    "range_status",
    "range_high",
    "range_low",
    "breakout_id",
    "breakout_ts",
    "breakout_direction",
    "breakout_distance_pct",
    "regime_strength",
    "confidence",
]


def _validate_market_summary_params(
    swing_window: int,
    min_swing_pct: float,
    equal_epsilon: float,
    range_lookback: int,
    max_range_pct: float,
    range_containment_ratio: float,
    range_boundary_buffer_pct: float,
    breakout_confirm_bars: int,
    breakout_buffer_pct: float,
    adx_period: int,
    adx_trend_threshold: float,
    adx_range_ceiling: float,
    di_spread_min: float,
    breakout_recency_bars: int,
) -> None:
    """Validate all market summary parameters."""
    _validate_swing_params(swing_window, min_swing_pct)
    _validate_equal_epsilon(equal_epsilon)
    _validate_range_params(
        range_lookback, max_range_pct, range_containment_ratio, range_boundary_buffer_pct
    )
    _validate_breakout_params(breakout_confirm_bars, breakout_buffer_pct)

    if not isinstance(adx_period, int) or adx_period < 1:
        raise ValueError("adx_period must be an integer >= 1")
    if not isinstance(adx_trend_threshold, Real) or not (0.0 <= adx_trend_threshold <= 100.0):
        raise ValueError("adx_trend_threshold must be a float in [0.0, 100.0]")
    if not isinstance(adx_range_ceiling, Real) or not (0.0 <= adx_range_ceiling <= 100.0):
        raise ValueError("adx_range_ceiling must be a float in [0.0, 100.0]")
    if not isinstance(di_spread_min, Real) or not (0.0 <= di_spread_min <= 100.0):
        raise ValueError("di_spread_min must be a float in [0.0, 100.0]")
    if not isinstance(breakout_recency_bars, int) or breakout_recency_bars < 1:
        raise ValueError("breakout_recency_bars must be an integer >= 1")


def _latest_finite(series: pd.Series, default: float = float("nan")) -> float:
    """Return the last finite value in a series, or default if none found."""
    finite_vals = series.dropna()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if len(finite_vals) == 0:
        return default
    return float(finite_vals.iloc[-1])


def _select_latest_range(ranges: pd.DataFrame) -> dict | None:
    """Select the most recent range by end_ts."""
    if len(ranges) == 0:
        return None
    ranges_sorted = ranges.sort_values("end_ts", ascending=False)
    row = ranges_sorted.iloc[0]
    return {
        "range_id": int(row["range_id"]),
        "range_status": row["status"],
        "range_high": float(row["range_high"]),
        "range_low": float(row["range_low"]),
    }


def _select_latest_breakout_for_range(
    breakouts: pd.DataFrame, range_id: int, ohlcv_index: pd.DatetimeIndex
) -> dict | None:
    """Select the most recent breakout for a given range, or None if recency exceeded."""
    if len(breakouts) == 0:
        return None
    range_breakouts = breakouts[breakouts["range_id"] == range_id]
    if len(range_breakouts) == 0:
        return None
    latest = range_breakouts.sort_values("breakout_ts", ascending=False).iloc[0]
    return {
        "breakout_id": int(latest["breakout_id"]),
        "breakout_ts": latest["breakout_ts"],
        "breakout_direction": latest["direction"],
        "breakout_distance_pct": float(latest["distance_pct"]),
    }


def summarize_market_structure(
    ohlcv: pd.DataFrame,
    *,
    swing_window: int = 2,
    min_swing_pct: float = 0.02,
    equal_epsilon: float = 1e-9,
    range_lookback: int = 20,
    max_range_pct: float = 0.03,
    range_containment_ratio: float = 0.8,
    range_boundary_buffer_pct: float = 0.001,
    breakout_confirm_bars: int = 1,
    breakout_buffer_pct: float = 0.001,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_range_ceiling: float = 20.0,
    di_spread_min: float = 2.0,
    breakout_recency_bars: int = 3,
) -> pd.DataFrame:
    """
    Compose a deterministic market-structure summary from existing indicator primitives.

    Consumes already-windowed OHLCV and composes:
    - detect_swing_points(...)
    - classify_market_structure(...)
    - detect_price_ranges(...)
    - detect_range_breakouts(...)
    - calculate_adx(...) from the momentum engine

    Returns a single-row DataFrame with the latest-regime summary covering
    trend, range, and breakout states.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Windowed OHLCV data with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume. Must be monotonically increasing
        with no duplicates.
    swing_window : int, optional
        Window size for swing detection. Default is 2.
    min_swing_pct : float, optional
        Minimum percentage move to qualify as a swing. Default is 0.02 (2%).
    equal_epsilon : float, optional
        Tolerance for considering prices equal. Default is 1e-9.
    range_lookback : int, optional
        Number of bars to evaluate for range detection. Default is 20.
    max_range_pct : float, optional
        Maximum range width as percentage of range midpoint. Default is 0.03 (3%).
    range_containment_ratio : float, optional
        Minimum ratio of bars that must be within range boundaries. Default is 0.8.
    range_boundary_buffer_pct : float, optional
        Buffer around boundaries as percentage. Default is 0.001 (0.1%).
    breakout_confirm_bars : int, optional
        Number of consecutive bars required to confirm breakout. Default is 1.
    breakout_buffer_pct : float, optional
        Buffer above/below range boundaries for breakout detection. Default is 0.001 (0.1%).
    adx_period : int, optional
        ADX smoothing period. Default is 14.
    adx_trend_threshold : float, optional
        Minimum ADX level to confirm a trending regime. Default is 25.0.
    adx_range_ceiling : float, optional
        Maximum ADX level below which a ranging regime is confirmed. Default is 20.0.
    di_spread_min : float, optional
        Minimum DI+ / DI- spread to confirm directional trend. Default is 2.0.
    breakout_recency_bars : int, optional
        Maximum bars since breakout to still count as a recent breakout. Default is 3.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns (in order):
        1.  analysis_ts (pd.Timestamp): analysis timestamp (last bar in window)
        2.  window_start_ts (pd.Timestamp): first bar timestamp in window
        3.  window_end_ts (pd.Timestamp): last bar timestamp in window
        4.  summary_label (str): trending_up|trending_down|ranging|breakout_up|
           breakout_down|transition|insufficient_data
        5.  decision_rule (str): breakout_up_rule|breakout_down_rule|ranging_rule|
           trending_up_rule|trending_down_rule|transition_rule|insufficient_data_rule
        6.  structure_event_ts (pd.Timestamp or pd.NaT)
        7.  structure_classification (str): HH|HL|LH|LL|EH|EL|None
        8.  structure_trend_state (str): bullish|bearish|range|transition|None
        9.  adx (float)
        10. plus_di (float)
        11. minus_di (float)
        12. di_spread (float = plus_di - minus_di)
        13. range_id (int or pd.NA)
        14. range_status (str): active|broken_up|broken_down|None
        15. range_high (float)
        16. range_low (float)
        17. breakout_id (int or pd.NA)
        18. breakout_ts (pd.Timestamp or pd.NaT)
        19. breakout_direction (str): up|down|None
        20. breakout_distance_pct (float)
        21. regime_strength (float in [0, 100], derived from ADX)
        22. confidence (float in [0.0, 1.0])

    Raises
    ------
    ValueError
        If OHLCV fails validation or any parameter is out of allowed range.
    """
    # Import ADX locally to avoid circular imports
    from .momentum import calculate_adx

    # Validate OHLCV first
    _validate_ohlcv(ohlcv)

    # Validate all parameters
    _validate_market_summary_params(
        swing_window,
        min_swing_pct,
        equal_epsilon,
        range_lookback,
        max_range_pct,
        range_containment_ratio,
        range_boundary_buffer_pct,
        breakout_confirm_bars,
        breakout_buffer_pct,
        adx_period,
        adx_trend_threshold,
        adx_range_ceiling,
        di_spread_min,
        breakout_recency_bars,
    )

    # Insufficient data check
    min_required = max(adx_period * 2, range_lookback, 2 * swing_window + 3)
    if len(ohlcv) < min_required:
        row = {col: float("nan") for col in _SUMMARY_OUTPUT_COLUMNS}
        row["analysis_ts"] = ohlcv.index[-1]
        row["window_start_ts"] = ohlcv.index[0]
        row["window_end_ts"] = ohlcv.index[-1]
        row["summary_label"] = "insufficient_data"
        row["decision_rule"] = "insufficient_data_rule"
        row["structure_classification"] = None
        row["structure_trend_state"] = None
        row["range_status"] = None
        row["breakout_direction"] = None
        row["confidence"] = 0.0
        row["range_id"] = pd.NA
        row["breakout_id"] = pd.NA
        row["breakout_ts"] = pd.NaT
        row["structure_event_ts"] = pd.NaT
        return pd.DataFrame([row])[_SUMMARY_OUTPUT_COLUMNS]

    # Compose primitives
    swings = detect_swing_points(
        ohlcv, swing_window=swing_window, min_swing_pct=min_swing_pct
    )
    structure = classify_market_structure(swings, equal_epsilon=equal_epsilon)
    ranges = detect_price_ranges(
        ohlcv,
        swings,
        range_lookback=range_lookback,
        max_range_pct=max_range_pct,
        containment_ratio=range_containment_ratio,
        boundary_buffer_pct=range_boundary_buffer_pct,
    )
    breakouts = detect_range_breakouts(
        ohlcv,
        ranges,
        breakout_confirm_bars=breakout_confirm_bars,
        breakout_buffer_pct=breakout_buffer_pct,
    )

    # ADX calculation
    adx_result = calculate_adx(
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        period=adx_period,
    )
    adx_series = adx_result["adx"]
    plus_di_series = adx_result["plus_di"]
    minus_di_series = adx_result["minus_di"]

    # Extract latest ADX values
    adx_val = _latest_finite(adx_series, float("nan"))
    plus_di_val = _latest_finite(plus_di_series, float("nan"))
    minus_di_val = _latest_finite(minus_di_series, float("nan"))
    di_spread_val = plus_di_val - minus_di_val

    # Extract latest structure classification
    structure_event_ts = pd.NaT
    structure_classification = None
    structure_trend_state = None
    if len(structure) > 0:
        latest_structure = structure.sort_values("event_ts", ascending=False).iloc[0]
        structure_event_ts = latest_structure["event_ts"]
        structure_classification = latest_structure["classification"]
        structure_trend_state = latest_structure["trend_state"]

    # Extract latest range
    latest_range = _select_latest_range(ranges)

    # Extract latest breakout
    latest_breakout = None
    if latest_range is not None:
        latest_breakout = _select_latest_breakout_for_range(
            breakouts, latest_range["range_id"], ohlcv.index
        )

    # Compute derived evidence
    trend_up_confirmed = (
        structure_trend_state == "bullish"
        and not np.isnan(adx_val)
        and adx_val >= adx_trend_threshold
        and not np.isnan(di_spread_val)
        and di_spread_val >= di_spread_min
    )
    trend_down_confirmed = (
        structure_trend_state == "bearish"
        and not np.isnan(adx_val)
        and adx_val >= adx_trend_threshold
        and not np.isnan(di_spread_val)
        and (-di_spread_val) >= di_spread_min
    )
    range_confirmed = (
        latest_range is not None
        and latest_range["range_status"] == "active"
        and not np.isnan(adx_val)
        and adx_val <= adx_range_ceiling
    )

    # Check recent breakout
    recent_breakout_up = False
    recent_breakout_down = False
    if latest_breakout is not None:
        breakout_ts = latest_breakout["breakout_ts"]
        bars_since_breakout = len(ohlcv) - ohlcv.index.get_loc(breakout_ts) - 1
        if (latest_breakout["breakout_direction"] == "up" and
            bars_since_breakout <= breakout_recency_bars):
            recent_breakout_up = True
        elif (latest_breakout["breakout_direction"] == "down" and
            bars_since_breakout <= breakout_recency_bars):
            recent_breakout_down = True

    # Decision priority (strict order from design)
    if recent_breakout_up:
        summary_label = "breakout_up"
        decision_rule = "breakout_up_rule"
        confidence = 0.85 if abs(latest_breakout["breakout_distance_pct"]) >= 0.003 else 0.75
    elif recent_breakout_down:
        summary_label = "breakout_down"
        decision_rule = "breakout_down_rule"
        confidence = 0.85 if abs(latest_breakout["breakout_distance_pct"]) >= 0.003 else 0.75
    elif range_confirmed:
        summary_label = "ranging"
        decision_rule = "ranging_rule"
        confidence = 0.70
    elif trend_up_confirmed:
        summary_label = "trending_up"
        decision_rule = "trending_up_rule"
        confidence = 0.80 if abs(di_spread_val) >= 5.0 else 0.70
    elif trend_down_confirmed:
        summary_label = "trending_down"
        decision_rule = "trending_down_rule"
        confidence = 0.80 if abs(di_spread_val) >= 5.0 else 0.70
    else:
        summary_label = "transition"
        decision_rule = "transition_rule"
        confidence = 0.40

    # Build row
    row = {
        "analysis_ts": ohlcv.index[-1],
        "window_start_ts": ohlcv.index[0],
        "window_end_ts": ohlcv.index[-1],
        "summary_label": summary_label,
        "decision_rule": decision_rule,
        "structure_event_ts": structure_event_ts,
        "structure_classification": structure_classification,
        "structure_trend_state": structure_trend_state,
        "adx": adx_val,
        "plus_di": plus_di_val,
        "minus_di": minus_di_val,
        "di_spread": di_spread_val,
        "range_id": latest_range["range_id"] if latest_range else pd.NA,
        "range_status": latest_range["range_status"] if latest_range else None,
        "range_high": latest_range["range_high"] if latest_range else float("nan"),
        "range_low": latest_range["range_low"] if latest_range else float("nan"),
        "breakout_id": latest_breakout["breakout_id"] if latest_breakout else pd.NA,
        "breakout_ts": latest_breakout["breakout_ts"] if latest_breakout else pd.NaT,
        "breakout_direction": latest_breakout["breakout_direction"] if latest_breakout else None,
        "breakout_distance_pct": (
            latest_breakout["breakout_distance_pct"] if latest_breakout else float("nan")
        ),
        "regime_strength": adx_val if not np.isnan(adx_val) else float("nan"),
        "confidence": confidence,
    }

    return pd.DataFrame([row])[_SUMMARY_OUTPUT_COLUMNS]
