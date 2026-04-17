"""Unit tests for TPO (Time-Price Opportunity) indicator."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tempest_mcp.indicators.volume.tpo as tpo_module
from tempest_mcp.indicators.volume.tpo import (
    COL_IN_VALUE_AREA,
    COL_PERIOD_COUNT,
    COL_PERIOD_MARKERS,
    COL_ROW_HIGH,
    COL_ROW_LOW,
    COL_ROW_MID,
    COL_TPO_COUNT,
    _build_row_lattice,
    _find_va_bounds,
    calculate_tpo_chart,
)


def _create_ohlcv(
    n: int = 20,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing.

    Args:
        n: Number of candles.
        start_price: Starting price.
        volatility: Price volatility factor.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV columns and UTC-aware DatetimeIndex.
    """
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Generate price series with some trend and noise
    returns = np.random.normal(0.0005, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    # Generate high/low with random spreads
    high_spread = np.random.uniform(0.001, 0.015, n) * close
    low_spread = np.random.uniform(0.001, 0.015, n) * close

    high = close + high_spread
    low = close - low_spread
    open_prices = np.random.uniform(low, high)

    # Ensure OHLC relationships hold
    high = np.maximum.reduce([open_prices, high, low, close])
    low = np.minimum.reduce([open_prices, high, low, close])

    # Generate volume with some pattern
    volume = np.random.uniform(1000, 5000, n) * (1 + np.abs(returns) * 10)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class TestCalculateTPOChart:
    """Tests for calculate_tpo_chart function."""

    def test_basic_tpo_generation(self):
        """Test basic TPO row generation."""
        ohlcv = _create_ohlcv(n=20)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        # Should return a DataFrame
        assert isinstance(tpo, pd.DataFrame)
        assert len(tpo) > 0

        # Check required columns exist in exact order
        expected_columns = [
            COL_ROW_LOW,
            COL_ROW_HIGH,
            COL_ROW_MID,
            COL_TPO_COUNT,
            COL_PERIOD_MARKERS,
            COL_PERIOD_COUNT,
            COL_IN_VALUE_AREA,
        ]
        assert list(tpo.columns) == expected_columns

        # Check row boundaries are valid
        assert (tpo[COL_ROW_HIGH] > tpo[COL_ROW_LOW]).all()

        # Check row mids are between low and high
        assert (tpo[COL_ROW_MID] >= tpo[COL_ROW_LOW]).all()
        assert (tpo[COL_ROW_MID] <= tpo[COL_ROW_HIGH]).all()

        # Check TPO counts are non-negative
        assert (tpo[COL_TPO_COUNT] >= 0).all()

        # Check period counts are non-negative
        assert (tpo[COL_PERIOD_COUNT] >= 0).all()

    def test_deterministic_reproducibility(self):
        """Test that same inputs produce same outputs (determinism)."""
        ohlcv = _create_ohlcv(n=20, seed=42)

        tpo1 = calculate_tpo_chart(ohlcv, row_size=1.0)
        tpo2 = calculate_tpo_chart(ohlcv, row_size=1.0)

        # All values should be identical
        pd.testing.assert_frame_equal(tpo1, tpo2)

        # Metadata should be identical
        assert tpo1.attrs["poc_price"] == tpo2.attrs["poc_price"]
        assert tpo1.attrs["poc_row_idx"] == tpo2.attrs["poc_row_idx"]

    def test_poc_within_price_range(self):
        """Test that POC price is within observed price range."""
        ohlcv = _create_ohlcv(n=20)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        min_price = float(ohlcv["low"].min())
        max_price = float(ohlcv["high"].max())

        assert tpo.attrs["poc_price"] >= min_price
        assert tpo.attrs["poc_price"] <= max_price

    def test_value_area_contains_poc(self):
        """Test that Value Area contains the POC."""
        ohlcv = _create_ohlcv(n=20)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0, value_area_pct=0.70)

        poc_price = tpo.attrs["poc_price"]
        val_price = tpo.attrs["val_price"]
        vah_price = tpo.attrs["vah_price"]

        assert val_price <= poc_price <= vah_price

    def test_val_less_than_vah(self):
        """Test that VAL <= VAH."""
        ohlcv = _create_ohlcv(n=20)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["val_price"] <= tpo.attrs["vah_price"]

    def test_required_attrs_exist(self):
        """Test that all required attrs are present."""
        ohlcv = _create_ohlcv(n=20)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        required_attrs = [
            "row_size",
            "marker_count",
            "poc_price",
            "poc_row_idx",
            "vah_price",
            "val_price",
            "initial_balance_low",
            "initial_balance_high",
            "range_expanded_up",
            "range_expanded_down",
        ]
        for attr in required_attrs:
            assert attr in tpo.attrs, f"Missing required attr: {attr}"

    def test_marker_assignment(self):
        """Test deterministic marker assignment."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        # Check that markers are assigned in chronological order
        # First period should have marker 'A', second 'B', etc.
        all_markers = []
        for markers in tpo[COL_PERIOD_MARKERS]:
            all_markers.extend(markers)

        # Should have markers A, B, C, D, E (at minimum)
        assert "A" in all_markers
        assert "B" in all_markers

    def test_custom_markers(self):
        """Test custom marker sequence."""
        dates = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0] * 3,
            },
            index=dates,
        )

        custom_markers = ["X", "Y", "Z"]
        tpo = calculate_tpo_chart(ohlcv, row_size=1.0, markers=custom_markers)

        # Check custom markers are used
        all_markers = []
        for markers in tpo[COL_PERIOD_MARKERS]:
            all_markers.extend(markers)

        assert "X" in all_markers
        assert "Y" in all_markers
        assert "Z" in all_markers

    def test_marker_overflow_raises(self):
        """Test that marker overflow raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
            index=dates,
        )

        # Default markers only go to 62 (A-Z, a-z, 0-9)
        with pytest.raises(ValueError, match="default markers available"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_insufficient_custom_markers_raises(self):
        """Test that insufficient custom markers raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        # Only 3 markers for 10 periods
        with pytest.raises(ValueError, match="Marker count must be >= period count"):
            calculate_tpo_chart(ohlcv, row_size=1.0, markers=["X", "Y", "Z"])

    def test_duplicate_custom_markers_raises(self):
        """Test that duplicate custom markers raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="must be unique"):
            calculate_tpo_chart(ohlcv, row_size=1.0, markers=["A", "B", "A", "C", "D"])

    def test_stringified_duplicate_custom_markers_raise(self):
        """Test duplicate markers are rejected after string coercion."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="must be unique"):
            calculate_tpo_chart(ohlcv, row_size=1.0, markers=[1, "1", "2", "3", "4"])

    def test_flat_periods_allocate_to_single_row(self):
        """Test zero-width periods allocate deterministically to one row."""
        dates = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 3,
                "high": [100.0] * 3,
                "low": [100.0] * 3,
                "close": [100.0] * 3,
                "volume": [1000.0] * 3,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert len(tpo) == 1
        assert tpo.loc[0, COL_TPO_COUNT] == 3
        assert tpo.loc[0, COL_PERIOD_MARKERS] == ["A", "B", "C"]
        assert tpo.loc[0, COL_PERIOD_COUNT] == 3

    def test_point_touches_allocate_to_lowest_matching_row(self):
        """Test point-touch periods on row edges allocate deterministically."""
        dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [100.0, 101.0],
                "low": [100.0, 101.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert len(tpo) == 1
        assert tpo.loc[0, COL_ROW_LOW] == 100.0
        assert tpo.loc[0, COL_ROW_HIGH] == 101.0
        assert tpo.loc[0, COL_TPO_COUNT] == 2
        assert tpo.loc[0, COL_PERIOD_MARKERS] == ["A", "B"]
        assert tpo.loc[0, COL_PERIOD_COUNT] == 2


class TestTPOValidation:
    """Tests for TPO input validation."""

    def test_empty_ohlcv_raises(self):
        """Test that empty OHLCV raises ValueError."""
        ohlcv = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.date_range("2024-01-01", periods=0, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_missing_columns_raises(self):
        """Test that missing OHLCV columns raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "close": [105.0] * 10,
                # missing 'low' and 'volume'
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="Missing:"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_tz_naive_index_raises(self):
        """Test that UTC-naive DatetimeIndex raises TypeError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")  # No tz
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="UTC-aware"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_duplicate_index_raises(self):
        """Test that duplicate DatetimeIndex raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        dates = dates.append(dates[:3])  # Add duplicates

        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 13,
                "high": [110.0] * 13,
                "low": [90.0] * 13,
                "close": [105.0] * 13,
                "volume": [1000.0] * 13,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="duplicate"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_non_monotonic_index_raises(self):
        """Test that non-monotonic DatetimeIndex raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        dates = dates[::-1]  # Reverse

        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="monotonic"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_invalid_row_size_zero_raises(self):
        """Test that row_size=0 raises ValueError."""
        ohlcv = _create_ohlcv(n=10)

        with pytest.raises(ValueError, match="positive"):
            calculate_tpo_chart(ohlcv, row_size=0)

    def test_invalid_row_size_negative_raises(self):
        """Test that negative row_size raises ValueError."""
        ohlcv = _create_ohlcv(n=10)

        with pytest.raises(ValueError, match="positive"):
            calculate_tpo_chart(ohlcv, row_size=-1.0)

    def test_invalid_row_size_infinite_raises(self):
        """Test that infinite row_size raises ValueError."""
        ohlcv = _create_ohlcv(n=10)

        with pytest.raises(ValueError, match="positive finite"):
            calculate_tpo_chart(ohlcv, row_size=float("inf"))

    @pytest.mark.parametrize(
        "column_name, bad_value",
        [("high", np.inf), ("high", np.nan), ("low", -np.inf), ("low", np.nan)],
    )
    def test_non_finite_price_values_raise(self, column_name: str, bad_value: float):
        """Test that NaN/inf high-low values fail fast with ValueError."""
        ohlcv = _create_ohlcv(n=10)
        ohlcv.loc[ohlcv.index[0], column_name] = bad_value

        with pytest.raises(ValueError, match=rf"OHLCV {column_name} values must be finite numbers"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_excessive_row_count_raises(self):
        """Test that pathological row lattice size is rejected."""
        dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [0.0, 0.0],
                "high": [20_000.0, 20_000.0],
                "low": [0.0, 0.0],
                "close": [10_000.0, 10_000.0],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="row lattice expands"):
            calculate_tpo_chart(ohlcv, row_size=1.0)

    def test_excessive_period_count_raises(self):
        """Test that pathological period counts are rejected."""
        n_periods = 1001
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * n_periods,
                "high": [101.0] * n_periods,
                "low": [99.0] * n_periods,
                "close": [100.5] * n_periods,
                "volume": [1000.0] * n_periods,
            },
            index=dates,
        )
        markers = [f"M{i}" for i in range(n_periods)]

        with pytest.raises(ValueError, match="periods, exceeding safety limit"):
            calculate_tpo_chart(ohlcv, row_size=1.0, markers=markers)

    def test_excessive_row_period_work_factor_raises(self):
        """Test that combined row-period work is capped before allocation."""
        n_periods = 1000
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * n_periods,
                "high": [2_100.0] * n_periods,
                "low": [100.0] * n_periods,
                "close": [1_100.0] * n_periods,
                "volume": [1000.0] * n_periods,
            },
            index=dates,
        )
        markers = [f"M{i}" for i in range(n_periods)]

        with pytest.raises(ValueError, match="row-period checks"):
            calculate_tpo_chart(ohlcv, row_size=1.0, markers=markers)

    def test_invalid_value_area_pct_zero_raises(self):
        """Test that value_area_pct=0 raises ValueError."""
        ohlcv = _create_ohlcv(n=10)

        with pytest.raises(ValueError, match="value_area_pct"):
            calculate_tpo_chart(ohlcv, row_size=1.0, value_area_pct=0)

    def test_valid_value_area_pct_one(self):
        """Test that value_area_pct=1.0 is valid (boundary case)."""
        ohlcv = _create_ohlcv(n=10)

        # value_area_pct=1.0 should be valid (design range is (0, 1])
        tpo = calculate_tpo_chart(ohlcv, row_size=1.0, value_area_pct=1.0)
        assert isinstance(tpo, pd.DataFrame)
        assert len(tpo) > 0

    def test_invalid_value_area_pct_negative_raises(self):
        """Test that negative value_area_pct raises ValueError."""
        ohlcv = _create_ohlcv(n=10)

        with pytest.raises(ValueError, match="value_area_pct"):
            calculate_tpo_chart(ohlcv, row_size=1.0, value_area_pct=-0.5)


class TestTPOMetadata:
    """Tests for TPO metadata and attributes."""

    def test_initial_balance_from_first_period(self):
        """Test that Initial Balance is computed from first period."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [105.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0],
                "low": [95.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0, 84.0, 83.0, 82.0],
                "close": [102.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        # IB should be from first period: low=95, high=105
        assert tpo.attrs["initial_balance_low"] == 95.0
        assert tpo.attrs["initial_balance_high"] == 105.0

    def test_range_expanded_up(self):
        """Test range expansion detection up."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0, 105.0, 105.0, 105.0, 110.0],  # Last extends above IB
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["range_expanded_up"] is True
        assert tpo.attrs["range_expanded_down"] is False

    def test_range_expanded_down(self):
        """Test range expansion detection down."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0, 90.0, 90.0, 90.0, 90.0],  # Last extends below IB
                "close": [97.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["range_expanded_up"] is False
        assert tpo.attrs["range_expanded_down"] is True

    def test_range_expanded_both_directions(self):
        """Test range expansion detection in both directions."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0, 105.0, 105.0, 105.0, 110.0],  # Extends up
                "low": [95.0, 90.0, 90.0, 90.0, 90.0],  # Extends down
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["range_expanded_up"] is True
        assert tpo.attrs["range_expanded_down"] is True

    def test_no_range_expansion(self):
        """Test no range expansion when all within IB."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,  # IB high = 105
                "low": [95.0] * 5,  # IB low = 95
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["range_expanded_up"] is False
        assert tpo.attrs["range_expanded_down"] is False

    def test_marker_count_attr(self):
        """Test that marker_count attr reflects actual markers used."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        assert tpo.attrs["marker_count"] == 10
        assert tpo.attrs["row_size"] == 1.0


class TestTPORowLattice:
    """Tests for row lattice building."""

    def test_single_price_value(self):
        """Test lattice with single price value (flat market)."""
        edges = _build_row_lattice(100.0, 100.0, 1.0)

        assert len(edges) == 2
        assert edges[0] == 100.0
        assert edges[1] == 100.0

    def test_exact_row_size_fit(self):
        """Test lattice when price range is exact multiple of row_size."""
        edges = _build_row_lattice(100.0, 105.0, 1.0)

        # Should have edges at 100, 101, 102, 103, 104, 105
        expected = pd.Index([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        pd.testing.assert_index_equal(edges, expected)

    def test_lattice_decimal_precision(self):
        """Test lattice with decimal price values."""
        edges = _build_row_lattice(100.1, 100.6, 0.1)

        # Should have proper decimal edges
        assert edges[0] == 100.1
        assert edges[-1] >= 100.6


class TestTPOValueArea:
    """Tests for Value Area calculation."""

    def test_find_va_bounds_basic(self):
        """Test basic Value Area bounds finding."""
        rows = [
            {COL_TPO_COUNT: 1, COL_ROW_LOW: 100.0, COL_ROW_HIGH: 101.0},
            {COL_TPO_COUNT: 5, COL_ROW_LOW: 101.0, COL_ROW_HIGH: 102.0},  # POC
            {COL_TPO_COUNT: 3, COL_ROW_LOW: 102.0, COL_ROW_HIGH: 103.0},
            {COL_TPO_COUNT: 2, COL_ROW_LOW: 103.0, COL_ROW_HIGH: 104.0},
        ]

        low_idx, high_idx = _find_va_bounds(rows, poc_row_idx=1, value_area_pct=0.70)

        assert low_idx <= 1
        assert high_idx >= 1

    def test_va_expands_toward_target(self):
        """Test that VA expands until target percentage is reached."""
        rows = [
            {COL_TPO_COUNT: 1, COL_ROW_LOW: 100.0, COL_ROW_HIGH: 101.0},
            {COL_TPO_COUNT: 5, COL_ROW_LOW: 101.0, COL_ROW_HIGH: 102.0},  # POC = 5
            {COL_TPO_COUNT: 3, COL_ROW_LOW: 102.0, COL_ROW_HIGH: 103.0},
            {COL_TPO_COUNT: 2, COL_ROW_LOW: 103.0, COL_ROW_HIGH: 104.0},
        ]

        total_tpo = sum(r[COL_TPO_COUNT] for r in rows)  # = 11
        target_70 = total_tpo * 0.70  # = 7.7

        low_idx, high_idx = _find_va_bounds(rows, poc_row_idx=1, value_area_pct=0.70)

        # VA should include enough rows to reach ~70% of TPOs
        va_tpo = sum(rows[i][COL_TPO_COUNT] for i in range(low_idx, high_idx + 1))
        assert va_tpo >= target_70

    def test_va_prefers_higher_adjacent_count(self):
        """Test VA expands toward the higher adjacent count first."""
        rows = [
            {COL_TPO_COUNT: 1, COL_ROW_LOW: 100.0, COL_ROW_HIGH: 101.0},
            {COL_TPO_COUNT: 5, COL_ROW_LOW: 101.0, COL_ROW_HIGH: 102.0},
            {COL_TPO_COUNT: 3, COL_ROW_LOW: 102.0, COL_ROW_HIGH: 103.0},
            {COL_TPO_COUNT: 2, COL_ROW_LOW: 103.0, COL_ROW_HIGH: 104.0},
        ]

        low_idx, high_idx = _find_va_bounds(rows, poc_row_idx=1, value_area_pct=0.70)

        assert (low_idx, high_idx) == (1, 2)


class TestTPODeterminism:
    """Tests for TPO determinism guarantees."""

    def test_identical_runs_produce_identical_output(self):
        """Test that repeated runs on same data produce identical output."""
        ohlcv = _create_ohlcv(n=30, seed=123)

        results = []
        for _ in range(3):
            tpo = calculate_tpo_chart(ohlcv, row_size=1.0, value_area_pct=0.70)
            results.append(tpo.copy())

        # All runs should be identical
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])
            assert results[0].attrs == results[i].attrs

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different output (sanity check)."""
        ohlcv1 = _create_ohlcv(n=20, seed=100)
        ohlcv2 = _create_ohlcv(n=20, seed=200)

        tpo1 = calculate_tpo_chart(ohlcv1, row_size=1.0)
        tpo2 = calculate_tpo_chart(ohlcv2, row_size=1.0)

        # They should be different (price ranges differ due to different seeds)
        assert not tpo1.attrs["poc_price"] == tpo2.attrs["poc_price"]


class TestTPOSchema:
    """Tests for exact output schema compliance."""

    def test_exact_column_order(self):
        """Test that columns are in the exact required order."""
        ohlcv = _create_ohlcv(n=10)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        expected_columns = [
            "row_low",
            "row_high",
            "row_mid",
            "tpo_count",
            "period_markers",
            "period_count",
            "in_value_area",
        ]
        assert list(tpo.columns) == expected_columns

    def test_period_count_equals_marker_list_length(self):
        """Test that period_count equals len(period_markers) for each row."""
        ohlcv = _create_ohlcv(n=10)

        tpo = calculate_tpo_chart(ohlcv, row_size=1.0)

        for _idx, row in tpo.iterrows():
            assert row[COL_PERIOD_COUNT] == len(row[COL_PERIOD_MARKERS])


class TestTPOBoundary:
    """Tests for indicator-layer boundary enforcement."""

    @staticmethod
    def _module_ast() -> ast.AST:
        source = Path(tpo_module.__file__).read_text(encoding="utf-8")
        return ast.parse(source)

    def test_no_tools_imports(self):
        """Test that tpo module does not import from tools package."""
        module = self._module_ast()

        forbidden_imports = []
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tempest_mcp.tools"):
                        forbidden_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("tempest_mcp.tools"):
                    forbidden_imports.append(node.module)

        assert len(forbidden_imports) == 0, (
            f"Found forbidden imports from tools: {forbidden_imports}"
        )

    def test_no_server_imports(self):
        """Test that tpo module does not import from server package."""
        module = self._module_ast()

        forbidden_imports = []
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tempest_mcp.server"):
                        forbidden_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("tempest_mcp.server"):
                    forbidden_imports.append(node.module)

        assert len(forbidden_imports) == 0, (
            f"Found forbidden imports from server: {forbidden_imports}"
        )
