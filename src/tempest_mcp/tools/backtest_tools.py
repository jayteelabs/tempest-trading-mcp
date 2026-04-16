"""Backtest MCP tool handlers — ENG-17 Phase 2.

This module is the canonical backtest execution layer for the MCP tool surface.

Public MCP handlers (6):
    backtest_pdh_session
    backtest_rsi
    backtest_vwap
    backtest_ema_stack
    backtest_order_blocks
    backtest_elliot_wave

Internal components:
    _STRATEGY_SPECS: dispatch registry (tool_name -> strategy spec)
    _run_direct_runner: invoke existing backtest runner
    _run_adapter: invoke thin MCP adapter for signal-generators

Phase 2 contract:
    - Tool boundary resolves/fetches OHLCV exactly once via backtest_window.py
    - Strategy functions receive resolved DataFrame (pure signal/backtest logic)
    - No legacy `period` or `source` inputs on Phase 2 public tools
    - Legacy backtest_strategy stub deprecated — returns deterministic error envelope
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
from structlog import get_logger

from tempest_mcp.backtest.engine import BacktestEngine
from tempest_mcp.config import ErrorCodes
from tempest_mcp.tools.backtest_window import (
    BacktestWindowRequest,
    validate_max_bars,
    validate_timeframe,
    validate_trade_style,
    resolve_and_fetch_backtest_ohlcv,
)

logger = get_logger(__name__)

# ── Strategy modes ──────────────────────────────────────────────────────────────

StrategyMode = Literal["direct_runner", "adapter"]

# ── Strategy spec registry ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategySpec:
    """Spec for a single backtest strategy tool."""

    strategy_id: str  # unique identifier (e.g., "pdh_session", "rsi")
    mode: StrategyMode  # "direct_runner" or "adapter"
    allowed_params: frozenset[str]  # strategy-specific params exposed via MCP
    # callable signature:
    #   direct_runner: (ohlcv_df, **params) -> tuple[signals, BacktestEngine]
    #   adapter: (ohlcv_df, initial_capital, **params) -> tuple[signals, BacktestEngine]


# Internal dispatch registry — maps tool name -> StrategySpec
_STRATEGY_SPECS: dict[str, StrategySpec] = {
    "backtest_pdh_session": StrategySpec(
        strategy_id="pdh_session",
        mode="direct_runner",
        allowed_params=frozenset({"atr_period", "atr_multiplier", "session_types"}),
    ),
    "backtest_rsi": StrategySpec(
        strategy_id="rsi",
        mode="adapter",
        allowed_params=frozenset(
            {
                "rsi_period",
                "confirmation_enabled",
                "oversold_threshold",
                "overbought_threshold",
                "risk_reward_ratio",
                "atr_stop_multiplier",
                "divergence_window",
            }
        ),
    ),
    "backtest_vwap": StrategySpec(
        strategy_id="vwap",
        mode="direct_runner",
        allowed_params=frozenset(
            {
                "vwap_anchor",
                "trend_fast_period",
                "trend_slow_period",
                "volume_lookback",
                "volume_multiplier",
            }
        ),
    ),
    "backtest_ema_stack": StrategySpec(
        strategy_id="ema_stack",
        mode="direct_runner",
        allowed_params=frozenset(
            {
                "ema_periods",
                "rr_multiple",
                "trend_confirmation_bars",
                "stop_buffer_pct",
            }
        ),
    ),
    "backtest_order_blocks": StrategySpec(
        strategy_id="order_blocks",
        mode="adapter",
        allowed_params=frozenset(
            {
                "confirmation_enabled",
                "atr_period",
                "impulse_atr_mult",
                "retest_atr_tolerance",
                "min_bars_before_entry",
                "max_zone_age_bars",
                "risk_reward_ratio",
            }
        ),
    ),
    "backtest_elliot_wave": StrategySpec(
        strategy_id="elliot_wave",
        mode="direct_runner",
        allowed_params=frozenset(),  # no extra params beyond window
    ),
}

# ── Exports for server.py ─────────────────────────────────────────────────────

# BACKTEST_TOOLS: tool name -> async handler
BACKTEST_TOOLS: dict[str, Any] = {}


def _register_handlers() -> None:
    """Register all 6 backtest handlers into BACKTEST_TOOLS."""
    for tool_name in _STRATEGY_SPECS:
        BACKTEST_TOOLS[tool_name] = _make_backtest_handler(tool_name)


def _make_backtest_handler(tool_name: str):
    """Factory: create an async handler for a given tool name."""

    async def handler(**kwargs: Any) -> dict[str, Any]:
        spec = _STRATEGY_SPECS[tool_name]

        try:
            # 1. Parse + validate shared args
            symbol = kwargs.pop("symbol")
            trade_style = validate_trade_style(kwargs.pop("trade_style", "day_trade"))
            timeframe = validate_timeframe(kwargs.pop("timeframe", None))
            start_at = _parse_iso_datetime("start_at", kwargs.pop("start_at", None))
            end_at = _parse_iso_datetime("end_at", kwargs.pop("end_at", None))
            exchange = kwargs.pop("exchange", "binance")
            initial_capital = _validate_initial_capital(kwargs.pop("initial_capital", 100000.0))
            max_bars = validate_max_bars(kwargs.pop("max_bars", None))
        except ValueError as e:
            return _validation_error(str(e))

        # 2. Filter to allowed strategy params
        strategy_params = {k: v for k, v in kwargs.items() if k in spec.allowed_params}

        # 3. Resolve window + fetch OHLCV once
        request = BacktestWindowRequest(
            symbol=symbol,
            trade_style=trade_style,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            exchange=exchange,
            max_bars=max_bars,
        )

        try:
            ohlcv_df, resolved_window = resolve_and_fetch_backtest_ohlcv(request)
        except ValueError as e:
            return _validation_error(str(e))
        except Exception as e:
            logger.error("Window resolution/fetch failed", tool=tool_name, error=str(e))
            return _internal_error("Data fetch failed")

        # 4. Validate fetched data minimum
        if len(ohlcv_df) < 2:
            return _validation_error(
                f"Insufficient data: only {len(ohlcv_df)} bars returned (minimum 2 required)"
            )

        # 5. Invoke strategy
        try:
            if spec.mode == "direct_runner":
                signals, engine = _run_direct_runner(
                    tool_name,
                    ohlcv_df,
                    initial_capital,
                    strategy_params,
                )
            else:  # adapter
                signals, engine = _run_adapter(
                    tool_name, ohlcv_df, initial_capital, strategy_params
                )
        except ValueError as e:
            return _validation_error(str(e))
        except Exception as e:
            logger.error("Strategy execution failed", tool=tool_name, error=str(e))
            return _internal_error("Strategy execution failed")

        # 6. Serialize result
        return _serialize_result(
            tool_name=tool_name,
            strategy_id=spec.strategy_id,
            symbol=symbol,
            window=resolved_window,
            engine=engine,
            trade_count=len(engine.trades),
            open_position=engine.open_position,
        )

    return handler


def _parse_iso_datetime(field_name: str, value: Any) -> datetime | None:
    """Parse an optional ISO-8601 datetime string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime") from exc


def _validate_initial_capital(value: Any) -> float:
    """Validate initial capital as a finite positive number."""
    if isinstance(value, bool):
        raise ValueError("initial_capital must be a finite number greater than 0")

    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_capital must be a finite number greater than 0") from exc

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("initial_capital must be a finite number greater than 0")

    return amount


# ── Direct runner dispatch ──────────────────────────────────────────────────────


def _run_direct_runner(
    tool_name: str,
    ohlcv_df: pd.DataFrame,
    initial_capital: float,
    params: dict[str, Any],
) -> tuple[pd.Series, BacktestEngine]:
    """Invoke an existing backtest runner for direct-runner strategies.

    PDH / VWAP / EMA Stack / Elliott Wave use this path.
    No strategy logic changes in ENG-17 — only wiring + parameter passing.
    """
    if tool_name == "backtest_pdh_session":
        from tempest_mcp.strategies import run_pdh_session_backtest

        result = run_pdh_session_backtest(
            ohlcv_df,
            initial_capital=initial_capital,
            **params,
        )
        return result

    elif tool_name == "backtest_vwap":
        from tempest_mcp.strategies import run_vwap_anchored_backtest

        result = run_vwap_anchored_backtest(
            ohlcv_df,
            initial_capital=initial_capital,
            **params,
        )
        return result

    elif tool_name == "backtest_ema_stack":
        from tempest_mcp.strategies import run_ema_stack_backtest

        result = run_ema_stack_backtest(
            ohlcv_df,
            initial_capital=initial_capital,
            **params,
        )
        return result

    elif tool_name == "backtest_elliot_wave":
        from tempest_mcp.strategies import run_elliot_wave_backtest

        result = run_elliot_wave_backtest(
            ohlcv_df,
            initial_capital=initial_capital,
            **params,
        )
        return result

    else:
        raise ValueError(f"No direct runner registered for tool: {tool_name}")


# ── Adapter dispatch ────────────────────────────────────────────────────────────


def _run_adapter(
    tool_name: str,
    ohlcv_df: pd.DataFrame,
    initial_capital: float,
    params: dict[str, Any],
) -> tuple[pd.Series, BacktestEngine]:
    """Invoke thin MCP adapter for signal-generator strategies.

    RSI and Order Blocks are signal generators that need a BacktestEngine wrapper.
    Adapters are thin and reversible when/if unified runner functions land upstream.
    """
    if tool_name == "backtest_rsi":
        return _run_rsi_backtest_adapter(ohlcv_df, initial_capital, **params)

    elif tool_name == "backtest_order_blocks":
        return _run_order_blocks_backtest_adapter(ohlcv_df, initial_capital, **params)

    else:
        raise ValueError(f"No adapter registered for tool: {tool_name}")


def _run_rsi_backtest_adapter(
    ohlcv_df: pd.DataFrame,
    initial_capital: float,
    **params: Any,
) -> tuple[pd.Series, BacktestEngine]:
    """RSI backtest adapter — wraps generate_rsi_signals with BacktestEngine.

    Executes the existing RSI signal generator through the shared backtest engine.
    """
    from tempest_mcp.strategies.backtest_rsi import generate_rsi_signals

    rsi_period = params.get("rsi_period", 14)
    confirmation_enabled = params.get("confirmation_enabled", False)
    oversold_threshold = params.get("oversold_threshold", 30.0)
    overbought_threshold = params.get("overbought_threshold", 70.0)
    risk_reward_ratio = params.get("risk_reward_ratio", 2.0)
    atr_stop_multiplier = params.get("atr_stop_multiplier", 1.5)
    divergence_window = params.get("divergence_window", 20)

    # Generate signals using existing signal generator
    signals = generate_rsi_signals(
        ohlcv_df,
        rsi_period=rsi_period,
        confirmation_enabled=confirmation_enabled,
        oversold_threshold=oversold_threshold,
        overbought_threshold=overbought_threshold,
        risk_reward_ratio=risk_reward_ratio,
        atr_stop_multiplier=atr_stop_multiplier,
        divergence_window=divergence_window,
    )

    # Run through backtest engine
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine


def _run_order_blocks_backtest_adapter(
    ohlcv_df: pd.DataFrame,
    initial_capital: float,
    **params: Any,
) -> tuple[pd.Series, BacktestEngine]:
    """Order Blocks backtest adapter — wraps generate_order_block_signals with BacktestEngine.

    Executes the existing Order Blocks signal generator through the shared backtest engine.
    """
    from tempest_mcp.strategies.backtest_order_blocks import (
        generate_order_block_signals,
    )

    confirmation_enabled = params.get("confirmation_enabled", True)
    atr_period = params.get("atr_period", 14)
    impulse_atr_mult = params.get("impulse_atr_mult", 1.0)
    retest_atr_tolerance = params.get("retest_atr_tolerance", 0.5)
    min_bars_before_entry = params.get("min_bars_before_entry", 2)
    max_zone_age_bars = params.get("max_zone_age_bars", 20)
    risk_reward_ratio = params.get("risk_reward_ratio", 2.0)

    # Generate signals using existing signal generator
    signals = generate_order_block_signals(
        ohlcv_df,
        confirmation_enabled=confirmation_enabled,
        atr_period=atr_period,
        impulse_atr_mult=impulse_atr_mult,
        retest_atr_tolerance=retest_atr_tolerance,
        min_bars_before_entry=min_bars_before_entry,
        max_zone_age_bars=max_zone_age_bars,
        risk_reward_ratio=risk_reward_ratio,
    )

    # Run through backtest engine
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine


# ── Result serialization ────────────────────────────────────────────────────────


def _sanitize_metric_value(value: Any) -> Any:
    """Normalize metrics to strict-JSON-safe scalar values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Replace non-finite metric values with null-safe None."""
    return {key: _sanitize_metric_value(value) for key, value in metrics.items()}


def _serialize_result(
    tool_name: str,
    strategy_id: str,
    symbol: str,
    window: Any,  # ResolvedBacktestWindow
    engine: BacktestEngine,
    trade_count: int,
    open_position: bool,
) -> dict[str, Any]:
    """Serialize a backtest run result into the standard success envelope."""
    return {
        "success": True,
        "data": {
            "tool": tool_name,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "window": {
                "trade_style": window.trade_style,
                "timeframe": window.timeframe,
                "start_at_utc": window.start_at_utc.isoformat(),
                "end_at_utc": window.end_at_utc.isoformat(),
                "estimated_bars": window.estimated_bars,
                "exchange": window.exchange,
            },
            "metrics": _sanitize_metrics(engine.metrics),
            "trade_count": trade_count,
            "open_position": open_position,
            "initial_capital": engine.initial_capital,
            "final_equity": engine.final_equity,
        },
    }


def _validation_error(message: str) -> dict[str, Any]:
    """Return a deterministic validation error envelope."""
    return {
        "success": False,
        "error": {
            "code": ErrorCodes.INVALID_PARAMETER,
            "message": message,
        },
    }


def _internal_error(message: str) -> dict[str, Any]:
    """Return a deterministic internal error envelope."""
    return {
        "success": False,
        "error": {
            "code": ErrorCodes.INTERNAL_ERROR,
            "message": message,
        },
    }


# ── Legacy stub (deprecated) ───────────────────────────────────────────────────


async def backtest_strategy(
    symbol: str,
    strategy_id: str = "rsi_mean_reversion",
    timeframe: str = "1h",
    period: str = "1y",
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    source: str = "yf",
) -> dict[str, Any]:
    """Deprecated: Use the dedicated Phase 2 backtest tools instead.

    This stub is kept for one transition cycle to provide a deterministic
    deprecation response. It does not execute any backtest logic.

    Migration guidance:
        - backtest_pdh_session  → PDH/PDL + Session Levels strategy
        - backtest_rsi           → RSI Mean Reversion strategy
        - backtest_vwap          → VWAP Anchored strategy
        - backtest_ema_stack     → EMA Stack strategy
        - backtest_order_blocks  → Order Blocks strategy
        - backtest_elliot_wave  → Elliott Wave strategy
    """
    logger.warning("Deprecated backtest_strategy called", symbol=symbol)
    return {
        "success": False,
        "error": {
            "code": ErrorCodes.VALIDATION_ERROR,
            "message": (
                "backtest_strategy is deprecated. "
                "Use one of the dedicated Phase 2 backtest tools: "
                "backtest_pdh_session, backtest_rsi, backtest_vwap, "
                "backtest_ema_stack, backtest_order_blocks, backtest_elliot_wave. "
                "Legacy inputs (period, source) are no longer supported."
            ),
        },
    }


# ── Initialize handler registry ────────────────────────────────────────────────

_register_handlers()
