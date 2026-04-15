"""Backtesting strategies package.

Keep imports lazy so consumers can import strategy modules directly without
pulling optional dependencies required by unrelated strategies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "run_pdh_session_backtest",
    "run_vwap_anchored_backtest",
    "run_elliot_wave_backtest",
]

if TYPE_CHECKING:
    from tempest_mcp.strategies.backtest_elliot_wave import run_elliot_wave_backtest
    from tempest_mcp.strategies.backtest_pdh_session import run_pdh_session_backtest
    from tempest_mcp.strategies.backtest_vwap import run_vwap_anchored_backtest


def __getattr__(name: str) -> Any:
    if name == "run_pdh_session_backtest":
        return import_module("tempest_mcp.strategies.backtest_pdh_session").run_pdh_session_backtest
    if name == "run_vwap_anchored_backtest":
        return import_module("tempest_mcp.strategies.backtest_vwap").run_vwap_anchored_backtest
    if name == "run_elliot_wave_backtest":
        return import_module("tempest_mcp.strategies.backtest_elliot_wave").run_elliot_wave_backtest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
