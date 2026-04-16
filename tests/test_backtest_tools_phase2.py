"""Tests for Phase 2 backtest tool handlers — ENG-17."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.tools.backtest_tools import (
    _STRATEGY_SPECS,
    BACKTEST_TOOLS,
    _internal_error,
    _parse_iso_datetime,
    _run_adapter,
    _run_direct_runner,
    _serialize_result,
    _validate_initial_capital,
    _validation_error,
    backtest_strategy,
)
from tempest_mcp.tools.backtest_window import ResolvedBacktestWindow


class TestBacktestToolsRegistry:
    """Tests for BACKTEST_TOOLS registry and _STRATEGY_SPECS dispatch map."""

    def test_all_six_tools_registered(self):
        """All 6 Phase 2 backtest tools are in BACKTEST_TOOLS."""
        expected = {
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        }
        assert set(BACKTEST_TOOLS.keys()) == expected

    def test_strategy_specs_has_all_six(self):
        """All 6 tools have entries in _STRATEGY_SPECS."""
        assert set(_STRATEGY_SPECS.keys()) == {
            "backtest_pdh_session",
            "backtest_rsi",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_order_blocks",
            "backtest_elliot_wave",
        }

    def test_direct_runner_strategies(self):
        """PDH, VWAP, EMA, Elliot Wave are direct_runner mode."""
        direct_modes = {k: v.mode for k, v in _STRATEGY_SPECS.items() if v.mode == "direct_runner"}
        assert set(direct_modes.keys()) == {
            "backtest_pdh_session",
            "backtest_vwap",
            "backtest_ema_stack",
            "backtest_elliot_wave",
        }

    def test_adapter_strategies(self):
        """RSI and Order Blocks are adapter mode."""
        adapter_modes = {k: v.mode for k, v in _STRATEGY_SPECS.items() if v.mode == "adapter"}
        assert set(adapter_modes.keys()) == {
            "backtest_rsi",
            "backtest_order_blocks",
        }

    def test_pdh_allowed_params(self):
        """PDH tool exposes only scoped params."""
        spec = _STRATEGY_SPECS["backtest_pdh_session"]
        assert spec.allowed_params == frozenset({"atr_period", "atr_multiplier", "session_types"})

    def test_rsi_allowed_params_includes_divergence_window(self):
        """RSI tool exposes divergence_window as required by acceptance criteria."""
        spec = _STRATEGY_SPECS["backtest_rsi"]
        assert "divergence_window" in spec.allowed_params

    def test_order_blocks_allowed_params(self):
        """Order Blocks tool exposes scoped params."""
        spec = _STRATEGY_SPECS["backtest_order_blocks"]
        expected = frozenset(
            {
                "confirmation_enabled",
                "atr_period",
                "impulse_atr_mult",
                "retest_atr_tolerance",
                "min_bars_before_entry",
                "max_zone_age_bars",
                "risk_reward_ratio",
            }
        )
        assert spec.allowed_params == expected

    def test_elliot_wave_no_extra_params(self):
        """Elliott Wave exposes no extra params beyond window (as specified)."""
        spec = _STRATEGY_SPECS["backtest_elliot_wave"]
        assert spec.allowed_params == frozenset()


class TestBacktestStrategyDeprecation:
    """Tests for legacy backtest_strategy deprecation response."""

    @pytest.mark.asyncio
    async def test_backtest_strategy_returns_deprecation_error(self):
        """Calling legacy backtest_strategy returns deterministic deprecation error."""
        result = await backtest_strategy(
            symbol="BTC/USDT",
            strategy_id="rsi_mean_reversion",
            timeframe="1h",
            period="1y",
            initial_capital=10000.0,
            exchange="binance",
            source="yf",
        )

        assert result["success"] is False
        assert "deprecated" in result["error"]["message"].lower()
        assert "backtest_pdh_session" in result["error"]["message"]
        assert "backtest_rsi" in result["error"]["message"]


class TestValidationErrorEnvelope:
    """Tests for _validation_error helper."""

    def test_validation_error_structure(self):
        """Validation error has correct envelope structure."""
        result = _validation_error("Invalid symbol format")
        assert result["success"] is False
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER
        assert result["error"]["message"] == "Invalid symbol format"


class TestInternalErrorEnvelope:
    """Tests for _internal_error helper."""

    def test_internal_error_structure(self):
        """Internal error has correct envelope structure."""
        result = _internal_error("An internal error occurred")
        assert result["success"] is False
        assert result["error"]["code"] == 9000  # INTERNAL_ERROR
        assert result["error"]["message"] == "An internal error occurred"


class TestSharedArgValidation:
    """Tests for shared backtest argument validation helpers."""

    def test_parse_iso_datetime_accepts_valid_iso8601(self):
        parsed = _parse_iso_datetime("start_at", "2024-01-01T00:00:00Z")
        assert parsed is not None
        assert parsed.tzinfo is not None

    def test_parse_iso_datetime_rejects_malformed_iso8601(self):
        with pytest.raises(ValueError, match="start_at must be a valid ISO 8601 datetime"):
            _parse_iso_datetime("start_at", "not-a-date")

    def test_validate_initial_capital_requires_finite_positive_number(self):
        assert _validate_initial_capital(100000) == 100000.0

        for invalid in (0, -1, float("inf"), float("nan"), True, "abc"):
            with pytest.raises(
                ValueError,
                match="initial_capital must be a finite number greater than 0",
            ):
                _validate_initial_capital(invalid)


class TestSerializeResult:
    """Tests for _serialize_result helper."""

    def test_serialize_result_structure(self):
        """Serialized result has correct success envelope structure."""
        mock_engine = MagicMock()
        mock_engine.metrics = {"total_return": 0.15, "sharpe_ratio": 1.2}
        mock_engine.trades = []
        mock_engine.open_position = False
        mock_engine.initial_capital = 100000.0
        mock_engine.final_equity = 115000.0

        window = ResolvedBacktestWindow(
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="1h",
            start_at_utc=pd.Timestamp("2024-01-01", tz="UTC"),
            end_at_utc=pd.Timestamp("2024-01-02", tz="UTC"),
            estimated_bars=24,
            exchange="binance",
        )

        result = _serialize_result(
            tool_name="backtest_pdh_session",
            strategy_id="pdh_session",
            symbol="BTC/USDT",
            window=window,
            engine=mock_engine,
            trade_count=5,
            open_position=False,
        )

        assert result["success"] is True
        assert result["data"]["tool"] == "backtest_pdh_session"
        assert result["data"]["strategy_id"] == "pdh_session"
        assert result["data"]["symbol"] == "BTC/USDT"
        assert result["data"]["trade_count"] == 5
        assert result["data"]["open_position"] is False
        assert result["data"]["window"]["trade_style"] == "day_trade"
        assert result["data"]["window"]["timeframe"] == "1h"

    def test_serialize_result_sanitizes_infinite_metrics_for_strict_json(self):
        """All-win runs with infinite profit_factor stay strict-JSON serializable."""
        index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 110.0, 110.0],
                "high": [101.0, 111.0, 111.0, 111.0],
                "low": [99.0, 99.0, 109.0, 109.0],
                "close": [100.0, 109.0, 110.0, 110.0],
                "volume": [10.0, 10.0, 10.0, 10.0],
            },
            index=index,
        )
        signals = pd.Series(
            [
                SignalAction.LONG_ENTRY,
                SignalAction.LONG_EXIT,
                SignalAction.HOLD,
                SignalAction.HOLD,
            ],
            index=index,
            dtype=object,
        )
        engine = BacktestEngine(
            initial_capital=100.0,
            commission_pct=0.0,
            slippage_bps=0.0,
        )
        engine.run(ohlcv_df, signals)

        window = ResolvedBacktestWindow(
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="1h",
            start_at_utc=index[0],
            end_at_utc=index[-1],
            estimated_bars=len(ohlcv_df),
            exchange="binance",
        )

        result = _serialize_result(
            tool_name="backtest_pdh_session",
            strategy_id="pdh_session",
            symbol="BTC/USDT",
            window=window,
            engine=engine,
            trade_count=len(engine.trades),
            open_position=engine.open_position,
        )

        assert result["data"]["metrics"]["profit_factor"] is None
        assert result["data"]["metrics"]["win_rate"] == 1.0
        assert result["data"]["metrics"]["total_trades"] == 1
        assert json.dumps(result, allow_nan=False)


class TestDirectRunnerDispatch:
    """Tests for _run_direct_runner dispatch logic."""

    def test_unknown_tool_raises(self):
        """Unknown tool name raises ValueError."""
        import pandas as pd

        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="No direct runner registered"):
            _run_direct_runner("nonexistent_tool", df, 100000.0, {})


class TestAdapterDispatch:
    """Tests for _run_adapter dispatch logic."""

    def test_unknown_adapter_raises(self):
        """Unknown adapter tool name raises ValueError."""
        import pandas as pd

        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="No adapter registered"):
            _run_adapter("nonexistent_adapter", df, 100000.0, {})


class TestBacktestHandlerValidation:
    """Regression tests for shared arg validation and sanitized error envelopes."""

    @pytest.fixture
    def sample_window(self):
        return ResolvedBacktestWindow(
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="1h",
            start_at_utc=pd.Timestamp("2024-01-01", tz="UTC"),
            end_at_utc=pd.Timestamp("2024-01-02", tz="UTC"),
            estimated_bars=24,
            exchange="binance",
        )

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [10.0, 11.0],
            }
        )

    @pytest.fixture
    def sample_engine(self, sample_df):
        signals = pd.Series(SignalAction.HOLD, index=sample_df.index, dtype=object)
        engine = BacktestEngine(initial_capital=25000.0)
        engine.run(sample_df, signals)
        return engine

    @pytest.mark.asyncio
    async def test_handler_rejects_invalid_trade_style(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            trade_style="position_trade",
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert "trade_style must be one of" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_rejects_invalid_timeframe(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="2h",
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert "timeframe must be one of" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_rejects_non_positive_max_bars(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            max_bars=0,
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert result["error"]["message"] == "max_bars must be an integer greater than 0"

    @pytest.mark.asyncio
    async def test_handler_rejects_non_positive_initial_capital(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            initial_capital=0,
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert (
            result["error"]["message"] == "initial_capital must be a finite number greater than 0"
        )

    @pytest.mark.asyncio
    async def test_handler_returns_invalid_parameter_for_malformed_start_at(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            trade_style="custom",
            start_at="not-an-iso-date",
            end_at="2024-01-02T00:00:00Z",
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert result["error"]["message"] == "start_at must be a valid ISO 8601 datetime"

    @pytest.mark.asyncio
    async def test_handler_returns_invalid_parameter_for_malformed_end_at(self):
        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            trade_style="custom",
            start_at="2024-01-01T00:00:00Z",
            end_at="not-an-iso-date",
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004
        assert result["error"]["message"] == "end_at must be a valid ISO 8601 datetime"

    @pytest.mark.asyncio
    async def test_handler_sanitizes_fetch_errors(self, monkeypatch):
        def boom(_request):
            raise RuntimeError("secret datasource details")

        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools.resolve_and_fetch_backtest_ohlcv",
            boom,
        )

        result = await BACKTEST_TOOLS["backtest_pdh_session"](symbol="BTC/USDT")

        assert result["success"] is False
        assert result["error"]["code"] == 9000
        assert result["error"]["message"] == "Data fetch failed"
        assert "secret datasource details" not in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_sanitizes_strategy_errors(
        self,
        monkeypatch,
        sample_df,
        sample_window,
    ):
        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools.resolve_and_fetch_backtest_ohlcv",
            lambda _request: (sample_df, sample_window),
        )

        def boom(_tool_name, _ohlcv_df, _params):
            raise RuntimeError("secret strategy details")

        monkeypatch.setattr("tempest_mcp.tools.backtest_tools._run_direct_runner", boom)

        result = await BACKTEST_TOOLS["backtest_pdh_session"](symbol="BTC/USDT")

        assert result["success"] is False
        assert result["error"]["code"] == 9000
        assert result["error"]["message"] == "Strategy execution failed"
        assert "secret strategy details" not in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_success_envelope_uses_engine_public_properties(
        self,
        monkeypatch,
        sample_df,
        sample_window,
        sample_engine,
    ):
        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools.resolve_and_fetch_backtest_ohlcv",
            lambda _request: (sample_df, sample_window),
        )
        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools._run_direct_runner",
            lambda _tool_name, _ohlcv_df, _initial_capital, _params: (
                pd.Series(SignalAction.HOLD, index=sample_df.index, dtype=object),
                sample_engine,
            ),
        )

        result = await BACKTEST_TOOLS["backtest_pdh_session"](symbol="BTC/USDT")

        assert result["success"] is True
        assert result["data"]["trade_count"] == 0
        assert result["data"]["open_position"] is False
        assert result["data"]["initial_capital"] == 25000.0
        assert result["data"]["final_equity"] == 25000.0
        assert result["data"]["metrics"]["total_trades"] == 0

    @pytest.mark.asyncio
    async def test_handler_forwards_initial_capital_to_direct_runners(
        self,
        monkeypatch,
        sample_df,
        sample_window,
        sample_engine,
    ):
        captured: dict[str, float] = {}

        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools.resolve_and_fetch_backtest_ohlcv",
            lambda _request: (sample_df, sample_window),
        )

        def fake_run_direct_runner(_tool_name, _ohlcv_df, initial_capital, _params):
            captured["initial_capital"] = initial_capital
            return (
                pd.Series(SignalAction.HOLD, index=sample_df.index, dtype=object),
                sample_engine,
            )

        monkeypatch.setattr(
            "tempest_mcp.tools.backtest_tools._run_direct_runner",
            fake_run_direct_runner,
        )

        result = await BACKTEST_TOOLS["backtest_pdh_session"](
            symbol="BTC/USDT",
            initial_capital=54321.0,
        )

        assert result["success"] is True
        assert captured["initial_capital"] == 54321.0
