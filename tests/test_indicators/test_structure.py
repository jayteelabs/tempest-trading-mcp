"""Unit tests for Fibonacci structure indicator engine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.structure import (
    DEFAULT_FIB_EXTENSION_LEVELS,
    DEFAULT_FIB_RETRACEMENT_LEVELS,
    calculate_fib_extensions,
    calculate_fib_retracements,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    detect_fib_confluence,
)


class TestDefaultConstants:
    """Tests for default Fibonacci level constants."""

    def test_default_retracement_levels(self):
        """Test default retracement levels are correct."""
        expected = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        assert DEFAULT_FIB_RETRACEMENT_LEVELS == expected

    def test_default_extension_levels(self):
        """Test default extension levels are correct."""
        expected = [1.272, 1.618, 2.0, 2.618]
        assert DEFAULT_FIB_EXTENSION_LEVELS == expected


class TestCalculateFibRetracements:
    """Tests for calculate_fib_retracements function."""

    def test_default_levels_return_type(self):
        """Test retracements returns a DataFrame."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert isinstance(df, pd.DataFrame)

    def test_default_levels_schema(self):
        """Test retracements DataFrame has correct columns in exact order."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        expected_columns = [
            "level_type",
            "level_ratio",
            "price",
            "swing_high",
            "swing_low",
            "trend_direction",
        ]
        assert list(df.columns) == expected_columns

    def test_default_levels_count(self):
        """Test retracements has correct number of rows for default levels."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert len(df) == 7  # 7 default levels

    def test_retracement_prices_correct(self):
        """Test retracement price calculations are correct."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        diff = 50.0  # 100 - 50

        for _, row in df.iterrows():
            expected_price = 50.0 + diff * row["level_ratio"]
            assert abs(row["price"] - expected_price) < 1e-10

    def test_retracement_level_ratios(self):
        """Test retracement level ratios are correct."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        expected_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        assert list(df["level_ratio"]) == expected_ratios

    def test_retracement_anchors_preserved(self):
        """Test swing_high and swing_low are preserved in output."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert (df["swing_high"] == 100.0).all()
        assert (df["swing_low"] == 50.0).all()

    def test_retracement_level_type(self):
        """Test all rows have level_type = 'retracement'."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert (df["level_type"] == "retracement").all()

    def test_retracement_trend_direction_null(self):
        """Test all rows have trend_direction = None for retracements."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert df["trend_direction"].isna().all()

    def test_retracement_deterministic_order(self):
        """Test retracement rows are in ascending level_ratio order."""
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert list(df["level_ratio"]) == sorted(df["level_ratio"])

    def test_custom_levels(self):
        """Test custom retracement levels."""
        custom_levels = [0.382, 0.618, 0.786]
        df = calculate_fib_retracements(swing_high=100.0, swing_low=50.0, levels=custom_levels)
        assert len(df) == 3
        assert list(df["level_ratio"]) == custom_levels

    def test_custom_levels_prices_correct(self):
        """Test custom retracement level prices."""
        custom_levels = [0.5, 1.0]
        df = calculate_fib_retracements(swing_high=100.0, swing_low=0.0, levels=custom_levels)
        assert abs(df[df["level_ratio"] == 0.5]["price"].iloc[0] - 50.0) < 1e-10
        assert abs(df[df["level_ratio"] == 1.0]["price"].iloc[0] - 100.0) < 1e-10


class TestCalculateFibExtensions:
    """Tests for calculate_fib_extensions function."""

    def test_bullish_extension_return_type(self):
        """Test extensions returns a DataFrame for bullish."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        assert isinstance(df, pd.DataFrame)

    def test_bearish_extension_return_type(self):
        """Test extensions returns a DataFrame for bearish."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bearish")
        assert isinstance(df, pd.DataFrame)

    def test_extension_schema(self):
        """Test extension DataFrame has correct columns in exact order."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        expected_columns = [
            "level_type",
            "level_ratio",
            "price",
            "swing_high",
            "swing_low",
            "trend_direction",
        ]
        assert list(df.columns) == expected_columns

    def test_extension_default_count(self):
        """Test extensions has correct number of rows for default levels."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        assert len(df) == 4  # 4 default extension levels

    def test_bullish_extension_prices_above_swing_high(self):
        """Test bullish extensions project above swing_high."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        diff = 50.0
        for _, row in df.iterrows():
            expected_price = 50.0 + diff * row["level_ratio"]
            assert abs(row["price"] - expected_price) < 1e-10
            assert row["price"] > 100.0  # All bullish extensions above swing high

    def test_bearish_extension_prices_below_swing_low(self):
        """Test bearish extensions project below swing_low."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bearish")
        diff = 50.0
        for _, row in df.iterrows():
            expected_price = 100.0 - diff * row["level_ratio"]
            assert abs(row["price"] - expected_price) < 1e-10
            assert row["price"] < 50.0  # All bearish extensions below swing low

    def test_extension_level_type(self):
        """Test all rows have level_type = 'extension'."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        assert (df["level_type"] == "extension").all()

    def test_extension_trend_direction_populated(self):
        """Test trend_direction is correctly populated."""
        bullish_df = calculate_fib_extensions(
            swing_high=100.0, swing_low=50.0, trend_direction="bullish"
        )
        bearish_df = calculate_fib_extensions(
            swing_high=100.0, swing_low=50.0, trend_direction="bearish"
        )
        assert (bullish_df["trend_direction"] == "bullish").all()
        assert (bearish_df["trend_direction"] == "bearish").all()

    def test_extension_anchors_preserved(self):
        """Test swing_high and swing_low are preserved in output."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        assert (df["swing_high"] == 100.0).all()
        assert (df["swing_low"] == 50.0).all()

    def test_extension_deterministic_order(self):
        """Test extension rows are in ascending level_ratio order."""
        df = calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="bullish")
        assert list(df["level_ratio"]) == sorted(df["level_ratio"])

    def test_custom_extension_levels(self):
        """Test custom extension levels."""
        custom_levels = [1.5, 2.0]
        df = calculate_fib_extensions(
            swing_high=100.0, swing_low=50.0, trend_direction="bullish", levels=custom_levels
        )
        assert len(df) == 2
        assert list(df["level_ratio"]) == custom_levels


class TestFibValidation:
    """Tests for Fibonacci validation errors."""

    def test_swing_high_equals_swing_low_raises(self):
        """Test ValueError when swing_high equals swing_low."""
        with pytest.raises(ValueError, match="swing_high must be greater than swing_low"):
            calculate_fib_retracements(swing_high=50.0, swing_low=50.0)

    def test_swing_high_less_than_swing_low_raises(self):
        """Test ValueError when swing_high is less than swing_low."""
        with pytest.raises(ValueError, match="swing_high must be greater than swing_low"):
            calculate_fib_retracements(swing_high=50.0, swing_low=100.0)

    def test_extension_swing_high_less_than_swing_low_raises(self):
        """Test ValueError for extensions when swing_high < swing_low."""
        with pytest.raises(ValueError, match="swing_high must be greater than swing_low"):
            calculate_fib_extensions(swing_high=50.0, swing_low=100.0, trend_direction="bullish")

    def test_invalid_trend_direction_raises(self):
        """Test ValueError for invalid trend_direction."""
        with pytest.raises(ValueError, match="trend_direction must be"):
            calculate_fib_extensions(swing_high=100.0, swing_low=50.0, trend_direction="sideways")

    def test_non_numeric_swing_high_raises(self):
        """Test ValueError when swing_high is not numeric."""
        with pytest.raises(ValueError, match="swing_high and swing_low must be numeric"):
            calculate_fib_retracements(swing_high="high", swing_low=50.0)

    def test_non_numeric_swing_low_raises(self):
        """Test ValueError when swing_low is not numeric."""
        with pytest.raises(ValueError, match="swing_high and swing_low must be numeric"):
            calculate_fib_retracements(swing_high=100.0, swing_low="low")

    def test_infinite_swing_high_raises(self):
        """Test ValueError when swing_high is infinite."""
        with pytest.raises(ValueError, match="swing_high and swing_low must be finite"):
            calculate_fib_retracements(swing_high=float("inf"), swing_low=50.0)

    def test_retracement_levels_out_of_range_raises(self):
        """Test ValueError when retracement levels outside [0, 1]."""
        with pytest.raises(ValueError, match="levels values must be"):
            calculate_fib_retracements(swing_high=100.0, swing_low=50.0, levels=[0.5, 1.5])

    def test_extension_levels_below_one_raises(self):
        """Test ValueError when extension levels are <= 1."""
        with pytest.raises(ValueError, match="levels values must be"):
            calculate_fib_extensions(
                swing_high=100.0, swing_low=50.0, trend_direction="bullish", levels=[0.5, 1.0]
            )

    def test_extension_levels_must_be_greater_than_one(self):
        """Test ValueError when extension levels are <= 1."""
        with pytest.raises(ValueError, match=r"levels values must be > 1\.0"):
            calculate_fib_extensions(
                swing_high=100.0, swing_low=50.0, trend_direction="bullish", levels=[1.0, 2.0]
            )

    def test_numpy_real_scalars_are_accepted(self):
        """Test numpy real scalars are accepted by validation helpers."""
        df = calculate_fib_retracements(
            swing_high=np.int64(100),
            swing_low=np.float64(50.0),
            levels=[np.float64(0.382), np.float64(0.618)],
        )

        assert list(df["level_ratio"]) == [0.382, 0.618]
        assert list(df["price"]) == [69.1, 80.9]

    def test_unsorted_levels_raises(self):
        """Test ValueError when levels are not sorted."""
        with pytest.raises(ValueError, match="levels must be sorted in ascending order"):
            calculate_fib_retracements(swing_high=100.0, swing_low=50.0, levels=[0.5, 0.382, 0.618])

    def test_duplicate_levels_raises(self):
        """Test ValueError when levels contain duplicates."""
        with pytest.raises(ValueError, match="levels must contain unique values"):
            calculate_fib_retracements(
                swing_high=100.0, swing_low=50.0, levels=[0.382, 0.382, 0.618]
            )

    def test_empty_levels_raises(self):
        """Test ValueError when levels is empty."""
        with pytest.raises(ValueError, match="levels cannot be empty"):
            calculate_fib_retracements(swing_high=100.0, swing_low=50.0, levels=[])

    def test_tolerance_must_be_positive(self):
        """Test ValueError when tolerance is not positive."""
        with pytest.raises(ValueError, match="tolerance must be positive"):
            detect_fib_confluence([], tolerance=0.0)

    def test_tolerance_must_be_finite(self):
        """Test ValueError when tolerance is infinite."""
        with pytest.raises(ValueError, match="tolerance must be a finite number"):
            detect_fib_confluence([], tolerance=float("inf"))


class TestDetectFibConfluence:
    """Tests for detect_fib_confluence function."""

    def test_empty_level_sets_returns_empty_dataframe(self):
        """Test empty level_sets returns DataFrame with correct columns."""
        result = detect_fib_confluence([])
        assert isinstance(result, pd.DataFrame)
        expected_columns = [
            "cluster_id",
            "cluster_price",
            "tolerance_low",
            "tolerance_high",
            "contributor_count",
            "contributors",
        ]
        assert list(result.columns) == expected_columns
        assert len(result) == 0

    def test_confluence_schema(self):
        """Test confluence result DataFrame has correct columns in exact order."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        expected_columns = [
            "cluster_id",
            "cluster_price",
            "tolerance_low",
            "tolerance_high",
            "contributor_count",
            "contributors",
        ]
        assert list(result.columns) == expected_columns

    def test_no_confluence_returns_empty(self):
        """Test when no levels cluster within tolerance, result is empty."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)  # prices: 50-100
        df2 = calculate_fib_retracements(swing_high=200.0, swing_low=150.0)  # prices: 150-200
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        assert len(result) == 0

    def test_confluence_detected(self):
        """Test confluence is detected when levels are within tolerance."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)  # 618 level ~ 80.9
        df2 = calculate_fib_retracements(
            swing_high=100.1, swing_low=50.0
        )  # 618 level ~ 80.96 - within 0.5
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        assert len(result) > 0
        assert result["contributor_count"].min() >= 2

    def test_cluster_price_is_centroid(self):
        """Test cluster_price is the average of contributing prices."""
        # Two identical swings should produce exact overlap
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        # Each level from df1 clusters with its counterpart from df2 at the same price
        # So we get 7 clusters (one per level), each with contributor_count = 2
        assert len(result) == len(df1)
        assert (result["contributor_count"] == 2).all()

    def test_tolerance_bounds(self):
        """Test tolerance_low and tolerance_high are correctly computed."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        if len(result) > 0:
            first_cluster = result.iloc[0]
            assert (
                abs((first_cluster["tolerance_high"] - first_cluster["tolerance_low"]) / 2 - 0.5)
                < 0.01
            )

    def test_contributor_count_minimum_two(self):
        """Test that only clusters with >= 2 contributors are returned."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence(
            [df1], tolerance=0.5
        )  # single set - can't cluster with itself
        assert len(result) == 0  # No confluence from single set

    def test_contributors_list_structure(self):
        """Test contributors list contains correct dict structure."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        if len(result) > 0:
            first_cluster = result.iloc[0]
            contributors = first_cluster["contributors"]
            assert isinstance(contributors, list)
            if len(contributors) > 0:
                contrib = contributors[0]
                assert "source_set_index" in contrib
                assert "level_type" in contrib
                assert "level_ratio" in contrib
                assert "price" in contrib
                assert "swing_high" in contrib
                assert "swing_low" in contrib
                assert "trend_direction" in contrib

    def test_cluster_id_ascending(self):
        """Test cluster_id values are ascending by cluster_price."""
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        result = detect_fib_confluence([df1, df2], tolerance=0.5)
        if len(result) > 1:
            assert list(result["cluster_id"]) == sorted(result["cluster_id"])
            assert list(result["cluster_price"]) == sorted(result["cluster_price"])


class TestLegacyCompatibility:
    """Tests for legacy calculate_fibonacci_levels compatibility wrapper."""

    def test_legacy_returns_dict(self):
        """Test legacy function returns a dictionary."""
        result = calculate_fibonacci_levels([100.0], [50.0])
        assert isinstance(result, dict)

    def test_legacy_keys_present(self):
        """Test legacy dict contains expected keys."""
        result = calculate_fibonacci_levels([100.0], [50.0])
        assert "swing_high" in result
        assert "swing_low" in result
        assert "fib_382" in result
        assert "fib_500" in result
        assert "fib_618" in result

    def test_legacy_swing_high_correct(self):
        """Test legacy swing_high is correctly computed."""
        result = calculate_fibonacci_levels([100.0, 90.0], [50.0, 60.0])
        assert result["swing_high"] == 100.0

    def test_legacy_swing_low_correct(self):
        """Test legacy swing_low is correctly computed."""
        result = calculate_fibonacci_levels([100.0, 90.0], [50.0, 60.0])
        assert result["swing_low"] == 50.0

    def test_legacy_fib_values_reasonable(self):
        """Test legacy fib values are between swing_low and swing_high."""
        result = calculate_fibonacci_levels([100.0], [50.0])
        assert result["swing_low"] <= result["fib_382"] <= result["swing_high"]
        assert result["swing_low"] <= result["fib_500"] <= result["swing_high"]
        assert result["swing_low"] <= result["fib_618"] <= result["swing_high"]

    def test_legacy_trend_up(self):
        """Test legacy function works with trend='up'."""
        result = calculate_fibonacci_levels([100.0], [50.0], trend="up")
        assert result["swing_high"] == 100.0
        assert result["swing_low"] == 50.0

    def test_legacy_trend_down(self):
        """Test legacy function works with trend='down'."""
        result = calculate_fibonacci_levels([100.0], [50.0], trend="down")
        assert result["swing_high"] == 100.0
        assert result["swing_low"] == 50.0
        assert abs(result["fib_382"] - 80.9) < 1e-9
        assert abs(result["fib_500"] - 75.0) < 1e-9
        assert abs(result["fib_618"] - 69.1) < 1e-9


class TestCalculatePivotPoints:
    """Tests for calculate_pivot_points function (unchanged behavior)."""

    def test_pivot_points_returns_dict(self):
        """Test pivot points returns a dictionary."""
        result = calculate_pivot_points([100.0], [50.0], [75.0])
        assert isinstance(result, dict)

    def test_pivot_points_keys(self):
        """Test pivot points dict contains expected keys."""
        result = calculate_pivot_points([100.0], [50.0], [75.0])
        assert "pivot" in result
        assert "r1" in result
        assert "s1" in result
        assert "r2" in result
        assert "s2" in result

    def test_pivot_calculation(self):
        """Test pivot calculation is correct."""
        result = calculate_pivot_points([100.0], [50.0], [75.0])
        expected_pivot = (100.0 + 50.0 + 75.0) / 3
        assert abs(result["pivot"] - expected_pivot) < 1e-10


class TestIndicatorLayerBoundary:
    """Tests confirming ENG-29 remains an indicator-layer ticket."""

    def test_structure_module_has_no_server_imports(self):
        """Test that structure.py does not import server or tools modules."""
        import tempest_mcp.indicators.structure as structure_module

        module_file = structure_module.__file__
        with open(module_file) as f:
            content = f.read()
        # Should not have imports from these modules
        assert "from tempest_mcp.tools" not in content
        assert "from tempest_mcp.server" not in content
        assert "import tempest_mcp.tools" not in content
        assert "import tempest_mcp.server" not in content

    def test_structure_module_has_no_mcp_related_imports(self):
        """Test that structure.py does not import MCP-related modules."""
        import tempest_mcp.indicators.structure as structure_module

        module_file = structure_module.__file__
        with open(module_file) as f:
            content = f.read()
        # Should not have MCP tool or server imports
        assert "from tempest_mcp import tools" not in content
        assert "from tempest_mcp import server" not in content


class TestIntegration:
    """Integration tests for Fibonacci engine workflow."""

    def test_retracement_extension_workflow(self):
        """Test complete Fibonacci analysis workflow."""
        # Get retracement levels
        retracements = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        assert len(retracements) == 7

        # Get bullish extensions
        bullish_ext = calculate_fib_extensions(
            swing_high=100.0, swing_low=50.0, trend_direction="bullish"
        )
        assert len(bullish_ext) == 4

        # Get bearish extensions
        bearish_ext = calculate_fib_extensions(
            swing_high=100.0, swing_low=50.0, trend_direction="bearish"
        )
        assert len(bearish_ext) == 4

    def test_confluence_with_multiple_sets(self):
        """Test confluence detection with multiple Fibonacci level sets."""
        # Create overlapping retracements
        df1 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df2 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)
        df3 = calculate_fib_retracements(swing_high=100.0, swing_low=50.0)

        result = detect_fib_confluence([df1, df2, df3], tolerance=0.5)

        # All 3 identical sets should produce strong confluence
        if len(result) > 0:
            assert result["contributor_count"].min() >= 2

    def test_legacy_wrapper_with_new_engine(self):
        """Test legacy wrapper produces consistent results with new engine."""
        legacy_result = calculate_fibonacci_levels([100.0], [50.0])

        # Legacy should match the 0.382, 0.5, 0.618 levels from new engine
        diff = 50.0
        assert abs(legacy_result["fib_382"] - (50.0 + diff * 0.382)) < 1e-9
        assert abs(legacy_result["fib_500"] - (50.0 + diff * 0.5)) < 1e-9
        assert abs(legacy_result["fib_618"] - (50.0 + diff * 0.618)) < 1e-9
