"""Unit tests for Elliott Wave detector (detect_elliott_waves)."""

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.indicators.structure import detect_elliott_waves

# =============================================================================
# Fixtures
# =============================================================================


def _make_ohlcv(dates, opens, highs, lows, closes, volumes):
    """Helper to create a UTC-aware OHLCV DataFrame."""
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates, tz="UTC"),
    )
    return df


def _make_zigzag_ohlcv(closes):
    """Create a deterministic OHLCV fixture from a zig-zag close path."""
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    opens = [closes[0], *closes[:-1]]
    highs = [max(o, c) + 0.4 for o, c in zip(opens, closes, strict=True)]
    lows = [min(o, c) - 0.4 for o, c in zip(opens, closes, strict=True)]
    volumes = [1000.0] * len(closes)
    return _make_ohlcv(dates, opens, highs, lows, closes, volumes)


def _make_bullish_impulse_ohlcv():
    """Create a fixture with at least one accepted deterministic bullish impulse."""
    closes = [
        100,
        103,
        106,
        110,
        107,
        105.2,
        108,
        114,
        120,
        127,
        124,
        121,
        124,
        130,
        136,
        132,
        128,
        125,
        129,
    ]
    return _make_zigzag_ohlcv(closes)


def _make_threshold_sensitive_ohlcv():
    """Create a fixture where min_swing_pct changes extracted swings."""
    closes = [100, 103, 106, 110, 109, 104, 108, 113, 119, 126, 123, 120, 124, 130, 136, 133, 129]
    return _make_zigzag_ohlcv(closes)


# =============================================================================
# Tests: Output Schema
# =============================================================================


class TestElliottOutputSchema:
    """Tests for detect_elliott_waves output schema."""

    def test_returns_dataframe(self):
        """Test returns a DataFrame."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_pinned_column_order(self):
        """Test output columns are in exact pinned order."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        expected = [
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
        assert list(result.columns) == expected

    def test_deterministic_row_ordering(self):
        """Test rows are ordered by sequence_id ASC, segment_order ASC."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        if len(result) > 1:
            seq_ids = result["sequence_id"].tolist()
            # Check sequence_id is non-decreasing
            assert seq_ids == sorted(seq_ids)
            # For same sequence_id, segment_order should be increasing
            for i in range(len(result) - 1):
                if result.iloc[i]["sequence_id"] == result.iloc[i + 1]["sequence_id"]:
                    assert result.iloc[i]["segment_order"] < result.iloc[i + 1]["segment_order"]

    def test_impulse_ratio_metadata_is_wave_specific(self):
        """Test impulse ratio metadata is populated only on the documented waves."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv, include_rejected=False)

        accepted_impulses = result[
            (result["sequence_type"] == "impulse")
            & (result["direction"] == "bullish")
            & (result["is_accepted_sequence"])
        ]

        assert not accepted_impulses.empty

        first_sequence = accepted_impulses.iloc[0]["sequence_id"]
        sequence_rows = (
            accepted_impulses[accepted_impulses["sequence_id"] == first_sequence]
            .set_index("wave_label")
            .sort_index()
        )

        assert pd.notna(sequence_rows.loc["2", "retrace_ratio"])
        assert pd.notna(sequence_rows.loc["4", "retrace_ratio"])
        assert pd.isna(sequence_rows.loc["1", "retrace_ratio"])
        assert pd.isna(sequence_rows.loc["3", "retrace_ratio"])
        assert pd.isna(sequence_rows.loc["5", "retrace_ratio"])

        assert pd.notna(sequence_rows.loc["3", "extension_ratio"])
        assert pd.notna(sequence_rows.loc["5", "extension_ratio"])
        assert pd.isna(sequence_rows.loc["1", "extension_ratio"])
        assert pd.isna(sequence_rows.loc["2", "extension_ratio"])
        assert pd.isna(sequence_rows.loc["4", "extension_ratio"])


# =============================================================================
# Tests: Input Validation
# =============================================================================


class TestElliottInputValidation:
    """Tests for detect_elliott_waves input validation."""

    def test_empty_dataframe_raises(self):
        """Test ValueError when OHLCV is empty."""
        ohlcv = _make_ohlcv([], [], [], [], [], [])
        with pytest.raises(ValueError, match="ohlcv must not be empty"):
            detect_elliott_waves(ohlcv)

    def test_missing_columns_raises(self):
        """Test ValueError when required columns are missing."""
        ohlcv = pd.DataFrame(
            {"open": [1.0], "high": [2.0]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="ohlcv missing required columns"):
            detect_elliott_waves(ohlcv)

    def test_naive_datetime_index_raises(self):
        """Test ValueError when index is not UTC-aware."""
        ohlcv = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),  # No tz
        )
        with pytest.raises(ValueError, match="ohlcv index must be UTC-aware"):
            detect_elliott_waves(ohlcv)

    def test_non_monotonic_index_raises(self):
        """Test ValueError when index is not monotonically increasing."""
        ohlcv = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="ohlcv index must be monotonically increasing"):
            detect_elliott_waves(ohlcv)

    def test_duplicate_index_raises(self):
        """Test ValueError when index has duplicates."""
        ohlcv = pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [2.0, 2.1],
                "low": [0.5, 0.6],
                "close": [1.5, 1.6],
                "volume": [100.0, 100.0],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="ohlcv index must not have duplicates"):
            detect_elliott_waves(ohlcv)

    def test_high_less_than_low_raises(self):
        """Test ValueError when high < low for any row."""
        ohlcv = pd.DataFrame(
            {"open": [1.0], "high": [0.5], "low": [2.0], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="high must be >= low"):
            detect_elliott_waves(ohlcv)

    def test_negative_volume_raises(self):
        """Test ValueError when volume is negative."""
        ohlcv = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [-100.0]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="volume must be non-negative"):
            detect_elliott_waves(ohlcv)


# =============================================================================
# Tests: Parameter Validation
# =============================================================================


class TestElliottParameterValidation:
    """Tests for detect_elliott_waves parameter validation."""

    def test_invalid_swing_window_raises(self):
        """Test ValueError for invalid swing_window."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="swing_window must be an integer"):
            detect_elliott_waves(ohlcv, swing_window=0)
        with pytest.raises(ValueError, match="swing_window must be an integer"):
            detect_elliott_waves(ohlcv, swing_window=-1)

    def test_invalid_min_swing_pct_raises(self):
        """Test ValueError for invalid min_swing_pct."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="min_swing_pct must be a float in"):
            detect_elliott_waves(ohlcv, min_swing_pct=0.0)
        with pytest.raises(ValueError, match="min_swing_pct must be a float in"):
            detect_elliott_waves(ohlcv, min_swing_pct=1.0)

    def test_invalid_wave2_retrace_band_raises(self):
        """Test ValueError for invalid wave2_retrace_band."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="wave2_retrace_band must be a tuple"):
            detect_elliott_waves(ohlcv, wave2_retrace_band="invalid")
        with pytest.raises(ValueError, match="wave2_retrace_band values must satisfy"):
            detect_elliott_waves(ohlcv, wave2_retrace_band=(0.5, 0.3))
        with pytest.raises(ValueError, match="wave2_retrace_band values must satisfy"):
            detect_elliott_waves(ohlcv, wave2_retrace_band=(0.0, 0.786))

    def test_invalid_wave3_extension_min_raises(self):
        """Test ValueError for invalid wave3_extension_min."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="wave3_extension_min must be a non-negative"):
            detect_elliott_waves(ohlcv, wave3_extension_min=-0.5)

    def test_invalid_wave4_retrace_max_raises(self):
        """Test ValueError for invalid wave4_retrace_max."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="wave4_retrace_max must be a float"):
            detect_elliott_waves(ohlcv, wave4_retrace_max=1.5)

    def test_invalid_degree_thresholds_raises(self):
        """Test ValueError for invalid degree_thresholds."""
        ohlcv = _make_bullish_impulse_ohlcv()
        with pytest.raises(ValueError, match="degree_thresholds must be a tuple"):
            detect_elliott_waves(ohlcv, degree_thresholds=[0.02, 0.08])
        with pytest.raises(ValueError, match="degree_thresholds must satisfy"):
            detect_elliott_waves(ohlcv, degree_thresholds=(0.08, 0.02))


# =============================================================================
# Tests: Degree Classification
# =============================================================================


class TestElliottDegreeClassification:
    """Tests for Elliott Wave degree classification."""

    def test_degree_labels_are_micro_minor_intermediate(self):
        """Test degree labels are exactly micro/minor/intermediate."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        if len(result) > 0:
            assert set(result["degree"].unique()).issubset({"micro", "minor", "intermediate"})

    def test_degree_thresholds_respected(self):
        """Test custom degree thresholds are respected."""
        ohlcv = _make_bullish_impulse_ohlcv()
        # With very tight thresholds, should get intermediate only
        result = detect_elliott_waves(ohlcv, degree_thresholds=(0.001, 0.002))
        if len(result) > 0:
            # All should be intermediate since thresholds are so tight
            assert all(result["degree"] == "intermediate")

    def test_timestamp_fields_are_pandas_native(self):
        """Test timestamp fields remain pd.Timestamp-compatible, not strings."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        if len(result) > 0:
            assert isinstance(result["start_ts"].iloc[0], (pd.Timestamp, np.datetime64))
            assert isinstance(result["end_ts"].iloc[0], (pd.Timestamp, np.datetime64))

    def test_baseline_fixture_contains_accepted_bullish_impulse(self):
        """Test the baseline fixture proves at least one accepted bullish impulse."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv, include_rejected=False)

        accepted_impulses = result[
            (result["sequence_type"] == "impulse")
            & (result["direction"] == "bullish")
            & (result["is_accepted_sequence"])
        ]

        assert not accepted_impulses.empty
        first_sequence = accepted_impulses.iloc[0]["sequence_id"]
        sequence_rows = accepted_impulses[accepted_impulses["sequence_id"] == first_sequence]

        assert sequence_rows["wave_label"].tolist() == ["1", "2", "3", "4", "5"]
        assert all(sequence_rows["is_rule_compliant"])
        assert all(sequence_rows["is_accepted_sequence"])


# =============================================================================
# Tests: Empty/Edge Cases
# =============================================================================


class TestElliottEdgeCases:
    """Tests for empty and edge case inputs."""

    def test_insufficient_data_returns_empty_dataframe_with_schema(self):
        """Test that insufficient data returns empty DataFrame with correct columns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        ohlcv = _make_ohlcv(dates, [1.0] * 5, [2.0] * 5, [0.5] * 5, [1.5] * 5, [100.0] * 5)
        result = detect_elliott_waves(ohlcv)
        expected_columns = [
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
        assert list(result.columns) == expected_columns
        assert len(result) == 0

    def test_include_rejected_false(self):
        """Test include_rejected=False filters to only accepted sequences."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result_all = detect_elliott_waves(ohlcv, include_rejected=True)
        result_accepted = detect_elliott_waves(ohlcv, include_rejected=False)
        if len(result_all) > 0:
            # accepted-only should have fewer or equal rows
            assert len(result_accepted) <= len(result_all)
            # All rows in accepted-only should be accepted
            assert all(result_accepted["is_accepted_sequence"])


# =============================================================================
# Tests: Determinism
# =============================================================================


class TestElliottDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_produces_same_output(self):
        """Test running twice on same data produces identical output."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result1 = detect_elliott_waves(ohlcv)
        result2 = detect_elliott_waves(ohlcv)
        pd.testing.assert_frame_equal(result1, result2)

    def test_parameter_changes_affect_output(self):
        """Test min_swing_pct changes swing extraction and downstream output."""
        ohlcv = _make_threshold_sensitive_ohlcv()
        result1 = detect_elliott_waves(ohlcv, min_swing_pct=0.05)
        result2 = detect_elliott_waves(ohlcv, min_swing_pct=0.08)

        assert len(result1) > len(result2)

    def test_sequence_ids_do_not_span_multiple_directions(self):
        """Test sequence IDs remain unique across bullish and bearish directions."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)

        if len(result) > 0:
            assert result.groupby("sequence_id")["direction"].nunique().max() == 1

    def test_is_accepted_sequence_is_consistent_per_sequence(self):
        """Test sequence-level acceptance is the same for every row in a sequence."""
        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)

        if len(result) > 0:
            assert result.groupby("sequence_id")["is_accepted_sequence"].nunique().max() == 1


# =============================================================================
# Tests: Indicator Layer Boundary
# =============================================================================


class TestElliottIndicatorLayerBoundary:
    """Tests confirming ENG-31 remains indicator-layer only."""

    def test_structure_module_has_no_server_imports(self):
        """Test that structure.py does not import server or tools modules."""
        from tempest_mcp.indicators.structure import __file__ as module_file

        with open(module_file) as f:
            content = f.read()
        assert "from tempest_mcp.tools" not in content
        assert "from tempest_mcp.server" not in content
        assert "import tempest_mcp.tools" not in content
        assert "import tempest_mcp.server" not in content

    def test_structure_module_has_no_backtest_imports(self):
        """Test that structure.py does not import strategy/backtest modules."""
        from tempest_mcp.indicators.structure import __file__ as module_file

        with open(module_file) as f:
            content = f.read()
        assert "from tempest_mcp.strategies" not in content
        assert "import tempest_mcp.strategies" not in content
        assert "backtest_elliot_wave" not in content


# =============================================================================
# Tests: Integration with Existing Structure Functions
# =============================================================================


class TestElliottIntegration:
    """Integration tests for Elliott Wave with existing structure indicators."""

    def test_fib_and_elliott_can_coexist(self):
        """Test that Fibonacci and Elliott Wave functions work in same module."""
        from tempest_mcp.indicators.structure import (
            calculate_fib_retracements,
            detect_elliott_waves,
        )

        ohlcv = _make_bullish_impulse_ohlcv()
        fib_result = calculate_fib_retracements(110.0, 100.0)
        wave_result = detect_elliott_waves(ohlcv)
        assert isinstance(fib_result, pd.DataFrame)
        assert isinstance(wave_result, pd.DataFrame)

    def test_import_via_indicators_package(self):
        """Test detect_elliott_waves can be imported from indicators package."""
        from tempest_mcp.indicators import detect_elliott_waves

        ohlcv = _make_bullish_impulse_ohlcv()
        result = detect_elliott_waves(ohlcv)
        assert isinstance(result, pd.DataFrame)
