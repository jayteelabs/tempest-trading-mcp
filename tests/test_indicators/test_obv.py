"""Unit tests for OBV indicator engine."""

import pandas as pd
import pytest

from tempest_mcp.indicators.volume import calculate_obv


class TestCalculateOBV:
    """Tests for calculate_obv function."""

    def test_normal_case(self):
        """Test OBV calculation with sufficient data."""
        close = pd.Series(
            [100, 102, 101, 103, 105, 104, 106],
            index=pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC"),
        )
        volume = pd.Series(
            [1000, 1100, 1050, 1200, 1300, 1250, 1400],
            index=close.index,
        )

        obv = calculate_obv(close, volume)

        assert len(obv) == len(close)
        assert obv.index.equals(close.index)
        # First bar: OBV[0] = volume[0]
        assert obv.iloc[0] == volume.iloc[0]
        # OBV should be positive
        assert (obv >= 0).all()

    def test_calculation_logic_up_close(self):
        """Test OBV adds volume on up closes."""
        close = pd.Series(
            [100, 102, 104],  # All up closes
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1100, 1200], index=close.index)

        obv = calculate_obv(close, volume)

        # OBV[0] = 1000
        # OBV[1] = 1000 + 1100 = 2100 (up close)
        # OBV[2] = 2100 + 1200 = 3300 (up close)
        assert obv.iloc[0] == 1000
        assert obv.iloc[1] == 2100
        assert obv.iloc[2] == 3300

    def test_calculation_logic_down_close(self):
        """Test OBV subtracts volume on down closes."""
        close = pd.Series(
            [100, 98, 96],  # All down closes
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1100, 1200], index=close.index)

        obv = calculate_obv(close, volume)

        # OBV[0] = 1000
        # OBV[1] = 1000 - 1100 = -100 (down close)
        # OBV[2] = -100 - 1200 = -1300 (down close)
        assert obv.iloc[0] == 1000
        assert obv.iloc[1] == -100
        assert obv.iloc[2] == -1300

    def test_calculation_logic_flat_close(self):
        """Test OBV doesn't change on flat closes."""
        close = pd.Series(
            [100, 102, 102, 102, 100],  # Flat in middle
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1100, 1200, 1300, 1400], index=close.index)

        obv = calculate_obv(close, volume)

        # OBV[0] = 1000
        # OBV[1] = 1000 + 1100 = 2100 (up close)
        # OBV[2] = 2100 (flat close, no change)
        # OBV[3] = 2100 (flat close, no change)
        # OBV[4] = 2100 - 1400 = 700 (down close)
        assert obv.iloc[0] == 1000
        assert obv.iloc[1] == 2100
        assert obv.iloc[2] == 2100
        assert obv.iloc[3] == 2100
        assert obv.iloc[4] == 700

    def test_insufficient_data(self):
        """Test returns empty Series when input is empty."""
        close = pd.Series(dtype=float)
        volume = pd.Series(dtype=float)

        obv = calculate_obv(close, volume)

        assert len(obv) == 0
        assert isinstance(obv, pd.Series)

    def test_single_bar(self):
        """Test OBV with single bar."""
        close = pd.Series(
            [100],
            index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000], index=close.index)

        obv = calculate_obv(close, volume)

        assert len(obv) == 1
        assert obv.iloc[0] == 1000

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raises ValueError."""
        close = pd.Series(
            [100, 102, 101],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        volume = pd.Series(
            [1000, 1100],  # Different length
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="same length"):
            calculate_obv(close, volume)

    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError."""
        close = pd.Series(
            [100, 102, 101, 103],
            index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
        )
        volume = pd.Series(
            [1000, 1100, -500, 1200],  # Negative volume at index 2
            index=close.index,
        )

        with pytest.raises(ValueError, match="non-negative"):
            calculate_obv(close, volume)

    def test_empty_series(self):
        """Test returns empty Series for empty input."""
        close = pd.Series([], dtype=float)
        volume = pd.Series([], dtype=float)

        obv = calculate_obv(close, volume)

        assert len(obv) == 0
        assert isinstance(obv, pd.Series)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        close = pd.Series(
            [100, 102, 101, 103],
            index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1100, 1050, 1200], index=close.index)

        obv = calculate_obv(close, volume)

        assert obv.index.tz is not None
        assert str(obv.index.tz) == "UTC"

    def test_price_gaps(self):
        """Test OBV handles price gaps as single directional move."""
        # Gap up: t1 closes above t0
        # Gap down: t2 closes below t1
        close = pd.Series(
            [100, 105, 100],  # Gap up then gap down
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1100, 1200], index=close.index)

        obv = calculate_obv(close, volume)

        # OBV[0] = 1000
        # OBV[1] = 1000 + 1100 = 2100 (up close: 105 > 100)
        # OBV[2] = 2100 - 1200 = 900 (down close: 100 < 105)
        assert obv.iloc[0] == 1000
        assert obv.iloc[1] == 2100
        assert obv.iloc[2] == 900

    def test_extreme_volume_spike(self):
        """Test OBV with extreme volume spike."""
        close = pd.Series(
            [100, 102, 101],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        volume = pd.Series([1000, 1000000, 1050], index=close.index)  # Huge spike in middle

        obv = calculate_obv(close, volume)

        # OBV[0] = 1000
        # OBV[1] = 1000 + 1000000 = 1001000 (up close)
        # OBV[2] = 1001000 - 1050 = 999950 (down close)
        assert obv.iloc[0] == 1000
        assert obv.iloc[1] == 1001000
        assert obv.iloc[2] == 999950


class TestIntegration:
    """Integration tests for OBV indicator workflow."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete OBV analysis workflow."""
        close = pd.Series(
            ohlcv_data["close"],
            index=pd.date_range("2024-01-01", periods=len(ohlcv_data["close"]), freq="h", tz="UTC"),
        )
        volume = pd.Series(
            ohlcv_data["volume"],
            index=pd.date_range(
                "2024-01-01", periods=len(ohlcv_data["volume"]), freq="h", tz="UTC"
            ),
        )

        obv = calculate_obv(close, volume)

        assert len(obv) == len(close)
        assert obv.index.tz is not None
