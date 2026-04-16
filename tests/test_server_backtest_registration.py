"""Tests for server.py backtest tool registration — ENG-17."""


from tempest_mcp.server import (
    TOOL_SCHEMAS,
    TOOLS,
    validate_tool_arguments,
)
from tempest_mcp.tools.backtest_window import SUPPORTED_TIMEFRAMES


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
