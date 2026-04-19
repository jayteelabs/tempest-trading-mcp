"""Tests for server.py backtest tool registration — ENG-17."""

import math

from tempest_mcp.server import (
    TOOL_SCHEMAS,
    TOOLS,
    validate_tool_arguments,
)
from tempest_mcp.tools.backtest_window import SUPPORTED_TIMEFRAMES
from tempest_mcp.tools.screener_tools import MAX_SCAN_SYMBOLS


class TestServerToolRegistration:
    """Tests for server tool registration."""

    def test_six_backtest_tools_in_tools_registry(self):
        """All 6 Phase 2 backtest tools are registered in TOOLS."""
        expected = {
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        }
        for tool_name in expected:
            assert tool_name in TOOLS, f"{tool_name} not found in TOOLS"

    def test_legacy_backtest_strategy_in_tools(self):
        """Legacy backtest_strategy is still in TOOLS (for deprecation response)."""
        assert "backtest_strategy" in TOOLS

    def test_legacy_backtest_strategy_not_in_schemas(self):
        """Legacy backtest_strategy is NOT in TOOL_SCHEMAS (not publicly listed)."""
        schema_names = {tool.name for tool in TOOL_SCHEMAS}
        assert "backtest_strategy" not in schema_names

    def test_all_six_tools_in_schemas(self):
        """All 6 Phase 2 backtest tools are listed in TOOL_SCHEMAS."""
        schema_names = {tool.name for tool in TOOL_SCHEMAS}
        expected = {
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        }
        assert schema_names >= expected

    def test_backtest_tools_have_trade_style_param(self):
        """All 6 backtest tools have trade_style in their schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "trade_style" in props, f"{tool.name} missing trade_style param"

    def test_backtest_tools_have_start_at_param(self):
        """All 6 backtest tools have start_at in their schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "start_at" in props, f"{tool.name} missing start_at param"

    def test_backtest_tools_have_end_at_param(self):
        """All 6 backtest tools have end_at in their schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "end_at" in props, f"{tool.name} missing end_at param"

    def test_backtest_tools_have_timeframe_param(self):
        """All 6 backtest tools have timeframe in their schema (optional)."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "timeframe" in props, f"{tool.name} missing timeframe param"

    def test_backtest_tools_timeframe_param_is_strict_enum(self):
        """All 6 backtest tools expose the supported timeframe enum in schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                timeframe = tool.inputSchema.get("properties", {}).get("timeframe", {})
                assert timeframe.get("enum") == list(SUPPORTED_TIMEFRAMES)

    def test_backtest_tools_have_exchange_param(self):
        """All 6 backtest tools have exchange in their schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "exchange" in props, f"{tool.name} missing exchange param"

    def test_backtest_tools_have_initial_capital_param(self):
        """All 6 backtest tools have initial_capital in their schema."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "initial_capital" in props, f"{tool.name} missing initial_capital param"

    def test_custom_window_datetime_description_matches_runtime_behavior(self):
        """Custom window schema documents naive datetime coercion semantics."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                for field in ("start_at", "end_at"):
                    description = props.get(field, {}).get("description", "")
                    assert "America/New_York" in description


class TestLegacyInputsExcluded:
    """Tests that legacy period/source inputs are NOT in Phase 2 schemas."""

    def test_backtest_pdh_no_period(self):
        """backtest_pdh_session does not expose legacy 'period' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_pdh_session":
                props = tool.inputSchema.get("properties", {})
                assert "period" not in props

    def test_backtest_pdh_no_source(self):
        """backtest_pdh_session does not expose legacy 'source' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_pdh_session":
                props = tool.inputSchema.get("properties", {})
                assert "source" not in props

    def test_backtest_rsi_no_period(self):
        """backtest_rsi does not expose legacy 'period' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "period" not in props

    def test_backtest_rsi_no_source(self):
        """backtest_rsi does not expose legacy 'source' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "source" not in props

    def test_all_phase2_tools_no_period(self):
        """No Phase 2 backtest tool exposes legacy 'period' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "period" not in props, f"{tool.name} should not have 'period'"

    def test_all_phase2_tools_no_source(self):
        """No Phase 2 backtest tool exposes legacy 'source' param."""
        for tool in TOOL_SCHEMAS:
            if tool.name.startswith("backtest_") and tool.name != "backtest_strategy":
                props = tool.inputSchema.get("properties", {})
                assert "source" not in props, f"{tool.name} should not have 'source'"


class TestValidateToolArguments:
    """Tests for validate_tool_arguments with backtest tools."""

    def test_valid_symbol_passes(self):
        """Valid symbol returns None (valid)."""
        for tool_name in (
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        ):
            result = validate_tool_arguments(tool_name, {"symbol": "BTC/USDT"})
            assert result is None, f"{tool_name} should accept valid symbol BTC/USDT"

    def test_empty_symbol_fails(self):
        """Empty symbol returns error message."""
        for tool_name in (
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        ):
            result = validate_tool_arguments(tool_name, {"symbol": ""})
            assert result is not None, f"{tool_name} should reject empty symbol"
            assert "empty" in result.lower()

    def test_invalid_symbol_format_fails(self):
        """Invalid symbol format returns error message."""
        for tool_name in (
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        ):
            result = validate_tool_arguments(tool_name, {"symbol": "INVALID@SYMBOL"})
            assert result is not None, f"{tool_name} should reject invalid symbol format"

    def test_legacy_backtest_strategy_validates_symbol(self):
        """Legacy backtest_strategy still validates symbol."""
        result = validate_tool_arguments("backtest_strategy", {"symbol": "BTC/USDT"})
        assert result is None
        result = validate_tool_arguments("backtest_strategy", {"symbol": ""})
        assert result is not None


class TestScreenerValidateToolArguments:
    """Tests for screener-specific validation in validate_tool_arguments."""

    def test_screener_symbols_are_capped(self):
        result = validate_tool_arguments(
            "screener_scan",
            {"symbols": ["BTC/USDT"] * (MAX_SCAN_SYMBOLS + 1)},
        )

        assert result == f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"

    def test_screener_invalid_min_score_range_fails(self):
        result = validate_tool_arguments("screener_scan", {"min_score": 101.0})

        assert result == "min_score must be between 0 and 100"

    def test_screener_non_finite_min_score_fails(self):
        result = validate_tool_arguments("screener_scan", {"min_score": math.inf})

        assert result == "min_score must be finite"

    def test_screener_invalid_exchange_fails(self):
        result = validate_tool_arguments("screener_scan", {"exchange": "okx"})

        assert result == "exchange must be one of: binance, bybit, coinbase, kraken"

    def test_screener_empty_symbols_remain_valid(self):
        result = validate_tool_arguments("screener_scan", {"symbols": []})

        assert result is None


class TestPDHSchemaParams:
    """Tests for backtest_pdh_session specific schema params."""

    def test_has_atr_period(self):
        """backtest_pdh_session has atr_period param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_pdh_session":
                props = tool.inputSchema.get("properties", {})
                assert "atr_period" in props

    def test_has_atr_multiplier(self):
        """backtest_pdh_session has atr_multiplier param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_pdh_session":
                props = tool.inputSchema.get("properties", {})
                assert "atr_multiplier" in props

    def test_has_session_types(self):
        """backtest_pdh_session has session_types param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_pdh_session":
                props = tool.inputSchema.get("properties", {})
                assert "session_types" in props


class TestRSISchemaParams:
    """Tests for backtest_rsi specific schema params."""

    def test_has_rsi_period(self):
        """backtest_rsi has rsi_period param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "rsi_period" in props

    def test_has_divergence_window(self):
        """backtest_rsi has divergence_window param (required by AC)."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "divergence_window" in props

    def test_has_oversold_threshold(self):
        """backtest_rsi has oversold_threshold param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "oversold_threshold" in props

    def test_has_overbought_threshold(self):
        """backtest_rsi has overbought_threshold param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_rsi":
                props = tool.inputSchema.get("properties", {})
                assert "overbought_threshold" in props


class TestVWAPSchemaParams:
    """Tests for backtest_vwap specific schema params."""

    def test_has_vwap_anchor(self):
        """backtest_vwap has vwap_anchor param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_vwap":
                props = tool.inputSchema.get("properties", {})
                assert "vwap_anchor" in props

    def test_has_trend_fast_period(self):
        """backtest_vwap has trend_fast_period param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_vwap":
                props = tool.inputSchema.get("properties", {})
                assert "trend_fast_period" in props

    def test_has_trend_slow_period(self):
        """backtest_vwap has trend_slow_period param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_vwap":
                props = tool.inputSchema.get("properties", {})
                assert "trend_slow_period" in props

    def test_vwap_defaults_match_strategy_contract(self):
        """VWAP schema defaults match the direct-runner strategy defaults."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_vwap":
                props = tool.inputSchema.get("properties", {})
                assert props["vwap_anchor"]["enum"] == ["asia", "london", "ny", "daily"]
                assert props["vwap_anchor"]["default"] == "ny"
                assert props["trend_fast_period"]["default"] == 7
                assert props["trend_slow_period"]["default"] == 25
                assert props["volume_multiplier"]["default"] == 1.2


class TestEMASchemaParams:
    """Tests for backtest_ema_stack specific schema params."""

    def test_has_ema_periods(self):
        """backtest_ema_stack has ema_periods param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_ema_stack":
                props = tool.inputSchema.get("properties", {})
                assert "ema_periods" in props

    def test_has_rr_multiple(self):
        """backtest_ema_stack has rr_multiple param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_ema_stack":
                props = tool.inputSchema.get("properties", {})
                assert "rr_multiple" in props

    def test_ema_defaults_match_strategy_contract(self):
        """EMA schema defaults match the direct-runner strategy defaults."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_ema_stack":
                props = tool.inputSchema.get("properties", {})
                assert props["ema_periods"]["default"] == [7, 25, 50, 200]
                assert props["trend_confirmation_bars"]["default"] == 1
                assert props["stop_buffer_pct"]["default"] == 0.0


class TestOrderBlocksSchemaParams:
    """Tests for backtest_order_blocks specific schema params."""

    def test_has_confirmation_enabled(self):
        """backtest_order_blocks has confirmation_enabled param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "confirmation_enabled" in props

    def test_has_atr_period(self):
        """backtest_order_blocks has atr_period param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "atr_period" in props

    def test_has_risk_reward_ratio(self):
        """backtest_order_blocks has risk_reward_ratio param."""
        for tool in TOOL_SCHEMAS:
            if tool.name == "backtest_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "risk_reward_ratio" in props


# ── ENG-28 analysis tools registration tests ────────────────────────────────────


class TestEng28ToolRegistration:
    """Tests for ENG-28 analysis tool server registration."""

    def test_calculate_volume_profile_in_tools(self):
        """calculate_volume_profile is registered in TOOLS."""
        from tempest_mcp.server import TOOLS

        assert "calculate_volume_profile" in TOOLS

    def test_detect_order_blocks_in_tools(self):
        """detect_order_blocks is registered in TOOLS."""
        from tempest_mcp.server import TOOLS

        assert "detect_order_blocks" in TOOLS

    def test_both_tools_in_schemas(self):
        """Both ENG-28 tools are listed in TOOL_SCHEMAS."""
        from tempest_mcp.server import TOOL_SCHEMAS

        schema_names = {tool.name for tool in TOOL_SCHEMAS}
        assert "calculate_volume_profile" in schema_names
        assert "detect_order_blocks" in schema_names


class TestEng28SharedWindowContract:
    """Tests that ENG-28 tools follow the shared window naming contract."""

    def test_calculate_volume_profile_has_symbol(self):
        """calculate_volume_profile has symbol param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "symbol" in props

    def test_calculate_volume_profile_has_timeframe(self):
        """calculate_volume_profile has timeframe param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "timeframe" in props

    def test_calculate_volume_profile_has_start_at(self):
        """calculate_volume_profile has start_at param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "start_at" in props

    def test_calculate_volume_profile_has_end_at(self):
        """calculate_volume_profile has end_at param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "end_at" in props

    def test_detect_order_blocks_has_symbol(self):
        """detect_order_blocks has symbol param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "symbol" in props

    def test_detect_order_blocks_has_timeframe(self):
        """detect_order_blocks has timeframe param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "timeframe" in props

    def test_detect_order_blocks_has_start_at(self):
        """detect_order_blocks has start_at param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "start_at" in props

    def test_detect_order_blocks_has_end_at(self):
        """detect_order_blocks has end_at param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "end_at" in props

    def test_both_use_timeframe_enum(self):
        """Both ENG-28 tools use the shared SUPPORTED_TIMEFRAMES enum."""
        from tempest_mcp.server import TOOL_SCHEMAS
        from tempest_mcp.tools.backtest_window import SUPPORTED_TIMEFRAMES

        for tool_name in ("calculate_volume_profile", "detect_order_blocks"):
            for tool in TOOL_SCHEMAS:
                if tool.name == tool_name:
                    tf = tool.inputSchema.get("properties", {}).get("timeframe", {})
                    assert tf.get("enum") == list(SUPPORTED_TIMEFRAMES)


class TestEng28ValidateToolArguments:
    """Tests for validate_tool_arguments with ENG-28 tools."""

    def test_valid_symbol_passes_volume_profile(self):
        """Valid symbol returns None for calculate_volume_profile."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("calculate_volume_profile", {"symbol": "BTC/USDT"})
        assert result is None

    def test_valid_symbol_passes_order_blocks(self):
        """Valid symbol returns None for detect_order_blocks."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("detect_order_blocks", {"symbol": "ETH/USDT"})
        assert result is None

    def test_empty_symbol_fails_volume_profile(self):
        """Empty symbol returns error for calculate_volume_profile."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("calculate_volume_profile", {"symbol": ""})
        assert result is not None
        assert "empty" in result.lower()

    def test_empty_symbol_fails_order_blocks(self):
        """Empty symbol returns error for detect_order_blocks."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("detect_order_blocks", {"symbol": ""})
        assert result is not None
        assert "empty" in result.lower()

    def test_invalid_symbol_format_fails_volume_profile(self):
        """Invalid symbol format returns error for calculate_volume_profile."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("calculate_volume_profile", {"symbol": "INVALID@"})
        assert result is not None

    def test_invalid_symbol_format_fails_order_blocks(self):
        """Invalid symbol format returns error for detect_order_blocks."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("detect_order_blocks", {"symbol": "INVALID/"})
        assert result is not None


class TestVolumeProfileSchemaParams:
    """Tests for calculate_volume_profile specific schema params."""

    def test_has_bin_count(self):
        """calculate_volume_profile has bin_count param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "bin_count" in props

    def test_has_profile_type(self):
        """calculate_volume_profile has profile_type param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "profile_type" in props

    def test_profile_type_enum_fixed_dynamic(self):
        """profile_type enum is ['fixed', 'dynamic']."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert props["profile_type"]["enum"] == ["fixed", "dynamic"]

    def test_has_dynamic_mode(self):
        """calculate_volume_profile has dynamic_mode param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "dynamic_mode" in props

    def test_has_value_area_pct(self):
        """calculate_volume_profile has value_area_pct param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                assert "value_area_pct" in props

    def test_no_legacy_backtest_params(self):
        """calculate_volume_profile does NOT expose backtest-only params."""
        from tempest_mcp.server import TOOL_SCHEMAS

        backtest_params = {
            "trade_style",
            "initial_capital",
            "confirmation_enabled",
            "retest_atr_tolerance",
            "risk_reward_ratio",
            "min_bars_before_entry",
        }
        for tool in TOOL_SCHEMAS:
            if tool.name == "calculate_volume_profile":
                props = tool.inputSchema.get("properties", {})
                for param in backtest_params:
                    assert param not in props, f"calculate_volume_profile should not have {param}"


class TestOrderBlocksAnalyticalBoundary:
    """Tests that detect_order_blocks only exposes detection-stage params.

    Per ENG-28 design: the standalone tool must NOT expose backtest-only
    params (confirmation_enabled, retest_atr_tolerance, min_bars_before_entry,
    risk_reward_ratio, exit/risk-management parameters).
    """

    def test_has_atr_period(self):
        """detect_order_blocks has atr_period param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "atr_period" in props

    def test_has_impulse_atr_mult(self):
        """detect_order_blocks has impulse_atr_mult param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "impulse_atr_mult" in props

    def test_has_max_zone_age_bars(self):
        """detect_order_blocks has max_zone_age_bars param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "max_zone_age_bars" in props

    def test_no_confirmation_enabled(self):
        """detect_order_blocks does NOT have confirmation_enabled param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "confirmation_enabled" not in props

    def test_no_retest_atr_tolerance(self):
        """detect_order_blocks does NOT have retest_atr_tolerance param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "retest_atr_tolerance" not in props

    def test_no_min_bars_before_entry(self):
        """detect_order_blocks does NOT have min_bars_before_entry param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "min_bars_before_entry" not in props

    def test_no_risk_reward_ratio(self):
        """detect_order_blocks does NOT have risk_reward_ratio param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "risk_reward_ratio" not in props

    def test_no_trade_style(self):
        """detect_order_blocks does NOT have trade_style param (uses custom internally)."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "trade_style" not in props

    def test_no_initial_capital(self):
        """detect_order_blocks does NOT have initial_capital param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "detect_order_blocks":
                props = tool.inputSchema.get("properties", {})
                assert "initial_capital" not in props


class TestSessionBreakoutScanRegistration:
    """Tests for session_breakout_scan server registration — ENG-35."""

    def test_session_breakout_scan_in_tools(self):
        """session_breakout_scan is registered in TOOLS."""
        from tempest_mcp.server import TOOLS

        assert "session_breakout_scan" in TOOLS

    def test_session_breakout_scan_in_schemas(self):
        """session_breakout_scan is listed in TOOL_SCHEMAS."""
        from tempest_mcp.server import TOOL_SCHEMAS

        schema_names = {tool.name for tool in TOOL_SCHEMAS}
        assert "session_breakout_scan" in schema_names

    def test_session_breakout_scan_schema_has_session_required(self):
        """session_breakout_scan schema has session as required param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "session_breakout_scan":
                props = tool.inputSchema.get("properties", {})
                assert "session" in props
                assert "session" in tool.inputSchema.get("required", [])

    def test_session_breakout_scan_schema_has_optional_symbols(self):
        """session_breakout_scan schema has symbols as optional param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "session_breakout_scan":
                props = tool.inputSchema.get("properties", {})
                assert "symbols" in props

    def test_session_breakout_scan_schema_has_optional_exchange(self):
        """session_breakout_scan schema has exchange as optional param."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "session_breakout_scan":
                props = tool.inputSchema.get("properties", {})
                assert "exchange" in props
                assert props["exchange"].get("default") == "binance"

    def test_session_breakout_scan_schema_has_proximity_pct(self):
        """session_breakout_scan schema has proximity_pct with default 1.0."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "session_breakout_scan":
                props = tool.inputSchema.get("properties", {})
                assert "proximity_pct" in props
                assert props["proximity_pct"].get("default") == 1.0

    def test_session_breakout_scan_schema_has_volume_multiplier(self):
        """session_breakout_scan schema has volume_multiplier with default 2.0."""
        from tempest_mcp.server import TOOL_SCHEMAS

        for tool in TOOL_SCHEMAS:
            if tool.name == "session_breakout_scan":
                props = tool.inputSchema.get("properties", {})
                assert "volume_multiplier" in props
                assert props["volume_multiplier"].get("default") == 2.0


class TestSessionBreakoutScanValidateArguments:
    """Tests for validate_tool_arguments with session_breakout_scan — ENG-35."""

    def test_session_breakout_scan_requires_session(self):
        """session_breakout_scan returns error when session is missing."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments("session_breakout_scan", {})
        assert result is not None
        assert "session" in result.lower()

    def test_session_breakout_scan_accepts_valid_session(self):
        """session_breakout_scan accepts valid session values."""
        from tempest_mcp.server import validate_tool_arguments

        for session in ("asia", "london", "ny", "new_york"):
            result = validate_tool_arguments(
                "session_breakout_scan",
                {"session": session},
            )
            assert result is None, f"Failed for session={session}"

    def test_session_breakout_scan_rejects_invalid_session(self):
        """session_breakout_scan rejects invalid session values."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "invalid"},
        )
        assert result is not None
        assert "session" in result.lower()

    def test_session_breakout_scan_validates_proximity_pct_range(self):
        """session_breakout_scan validates proximity_pct range."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "ny", "proximity_pct": -1.0},
        )
        assert result is not None
        assert "proximity_pct" in result.lower()

    def test_session_breakout_scan_validates_volume_multiplier_non_negative(self):
        """session_breakout_scan validates volume_multiplier non-negative."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "ny", "volume_multiplier": -1.0},
        )
        assert result is not None
        assert "volume_multiplier" in result.lower()

    def test_session_breakout_scan_validates_exchange(self):
        """session_breakout_scan validates exchange."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "ny", "exchange": "invalid_exchange"},
        )
        assert result is not None
        assert "exchange" in result.lower()

    def test_session_breakout_scan_validates_symbols_list(self):
        """session_breakout_scan validates symbols list."""
        from tempest_mcp.server import validate_tool_arguments

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "ny", "symbols": "not_a_list"},
        )
        assert result is not None
        assert "symbols" in result.lower()

    def test_session_breakout_scan_validates_symbols_cap(self):
        """session_breakout_scan validates symbols list max size."""
        from tempest_mcp.server import validate_tool_arguments
        from tempest_mcp.tools.screener_tools import MAX_SCAN_SYMBOLS

        result = validate_tool_arguments(
            "session_breakout_scan",
            {"session": "ny", "symbols": ["BTC/USDT"] * (MAX_SCAN_SYMBOLS + 1)},
        )
        assert result is not None
        assert "symbols" in result.lower()
