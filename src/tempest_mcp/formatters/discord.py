"""Discord output formatter for Kurisu-facing structured messages."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from typing import Any

# ── Color Constants ─────────────────────────────────────────────────────────────
COLOR_POSITIVE = 0x2ECC71  # Green — bullish / profitable
COLOR_NEGATIVE = 0xE74C3C  # Red — bearish / loss / error
COLOR_NEUTRAL = 0xF1C40F   # Yellow — neutral / warning
COLOR_DEFAULT = 0x3498DB    # Blue — generic / unknown

# ── Emoji Constants ─────────────────────────────────────────────────────────────
EMOJI_BULLISH = "📈"
EMOJI_BEARISH = "📉"
EMOJI_VOLUME_SPIKE = "⚡"
EMOJI_CROSS_SIGNAL = "⚠️"

# Score emojis
EMOJI_SCORE_LOCK = "🔒"   # ≥70%
EMOJI_SCORE_WARN = "⚠️"  # 40-69%
EMOJI_SCORE_BAD = "🔴"   # <40%

# Mood emojis
EMOJI_MOOD_BULLISH = "😀"
EMOJI_MOOD_NEUTRAL = "😐"
EMOJI_MOOD_BEARISH = "😠"

# ── Tool Category Sets ─────────────────────────────────────────────────────────
BACKTEST_TOOLS = frozenset({
    "backtest_pdh_session",
    "backtest_rsi",
    "backtest_vwap",
    "backtest_ema_stack",
    "backtest_order_blocks",
    "backtest_elliot_wave",
})

SCREENER_TOOLS = frozenset({
    "screener_scan",
    "session_breakout_scan",
    "order_block_screener_scan",
})

MARKET_TOOLS = frozenset({
    "fetch_ticker",
    "fetch_klines",
    "fetch_orderbook",
})

ANALYTICAL_TOOLS = frozenset({
    "calculate_volume_profile",
    "detect_order_blocks",
    "calculate_fibonacci",
    "calculate_tpo",
    "detect_elliot_wave",
    "get_market_structure",
})

# ── Discord Limits ─────────────────────────────────────────────────────────────
DISCORD_TITLE_LIMIT = 256
DISCORD_DESCRIPTION_LIMIT = 4096
DISCORD_FIELD_NAME_LIMIT = 256
DISCORD_FIELD_VALUE_LIMIT = 1024  # chars per field value
DISCORD_TOTAL_FIELDS_LIMIT = 25

# ── Generic Truncation / File Write Constants ───────────────────────────────────
_GENERIC_TRUNCATE_LIMIT = 1800  # chars — generous for a single field value
_TMP_DIR = tempfile.gettempdir()


class DiscordFormatter:
    """Pure utility that converts MCP tool result envelopes into Discord embed dicts.

    No I/O, no external API calls, no discord.py dependency.
    Input dict is never mutated.
    """

    # ── Public Entry Point ─────────────────────────────────────────────────────

    def format(self, result: dict) -> dict:
        """Dispatch entrypoint — routes to the correct formatter method."""
        if not isinstance(result, dict):
            return self._error_embed(
                code="INVALID_FORMAT_PAYLOAD",
                message="Expected a dict envelope, got unknown type",
            )

        if not result.get("success", True):
            return self.format_error(result)

        data = result.get("data", {})
        if not isinstance(data, dict):
            return self._error_embed(
                code="INVALID_FORMAT_PAYLOAD",
                message="Expected success envelope data to be a dict",
            )

        tool = data.get("tool")

        if tool in BACKTEST_TOOLS:
            return self.format_backtest(result)
        if tool == "compare_strategies":
            return self.format_compare(result)
        if tool in SCREENER_TOOLS:
            return self.format_screener(result)
        if tool == "get_combined_sentiment_dashboard":
            return self.format_sentiment(result)
        if tool in MARKET_TOOLS:
            return self.format_market(result)
        if tool in ANALYTICAL_TOOLS:
            return self.format_analytical(result)
        if tool == "indicator_rsi":
            return self.format_indicator(result)
        return self.format_generic(result)

    # ── Formatter Methods ─────────────────────────────────────────────────────

    def format_backtest(self, result: dict) -> dict:
        """Render backtest tool result — strategy metrics with color by P&L."""
        data = result.get("data", {})

        strategy_id = self._safe_value(data.get("strategy_id"))
        symbol = self._safe_value(data.get("symbol"))
        trade_count = self._safe_value(data.get("trade_count"))
        initial_capital = self._safe_value(data.get("initial_capital"))
        final_equity = self._safe_value(data.get("final_equity"))

        # Metrics block
        metrics = data.get("metrics", {})
        total_return = self._safe_value(metrics.get("total_return"))
        sharpe_ratio = self._safe_value(metrics.get("sharpe_ratio"))
        max_drawdown = self._safe_value(metrics.get("max_drawdown"))
        win_rate = self._safe_value(metrics.get("win_rate"))
        profit_factor = self._safe_value(metrics.get("profit_factor"))
        avg_win = self._safe_value(metrics.get("avg_win"))
        avg_loss = self._safe_value(metrics.get("avg_loss"))

        # Color by P&L
        try:
            ic = float(initial_capital) if initial_capital != "N/A" else None
            fe = float(final_equity) if final_equity != "N/A" else None
            color = COLOR_POSITIVE if (fe is not None and ic is not None and fe >= ic) else COLOR_NEGATIVE
        except (TypeError, ValueError):
            color = COLOR_NEUTRAL

        fields = [
            {"name": "Strategy", "value": strategy_id, "inline": True},
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Trades", "value": str(trade_count), "inline": True},
            {"name": "Initial Capital", "value": self._fmt_price(initial_capital), "inline": True},
            {"name": "Final Equity", "value": self._fmt_price(final_equity), "inline": True},
            {"name": "Total Return", "value": self._fmt_percent(total_return), "inline": True},
            {"name": "Sharpe Ratio", "value": self._fmt_number(sharpe_ratio), "inline": True},
            {"name": "Max Drawdown", "value": self._fmt_percent(max_drawdown), "inline": True},
            {"name": "Win Rate", "value": self._fmt_percent(win_rate), "inline": True},
            {"name": "Profit Factor", "value": self._fmt_number(profit_factor), "inline": True},
            {"name": "Avg Win", "value": self._fmt_price(avg_win), "inline": True},
            {"name": "Avg Loss", "value": self._fmt_price(avg_loss), "inline": True},
        ]

        return self._embed(
            title=f"📊 Backtest: {strategy_id} ({symbol})",
            color=color,
            fields=self._cap_fields(fields),
        )

    def format_compare(self, result: dict) -> dict:
        """Render compare_strategies result — ranked list with top-N + count metadata.

        Locked decision (Josh): use top-N + count metadata.
        """
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}
        best_strategy_id = self._safe_value(data.get("best_strategy_id"))
        ranking_metric = self._safe_value(data.get("ranking_metric"))
        all_results = data.get("results", [])

        # top-N + count metadata (top 10 by default)
        TOP_N = 10
        displayed = all_results[:TOP_N]
        total_count = len(all_results)
        has_more = total_count > TOP_N

        fields = []
        for row in displayed:
            sid = self._safe_value(row.get("strategy_id"))
            rank = self._safe_value(row.get("rank"))
            total_ret = self._safe_value(row.get("total_return"))
            sharpe = self._safe_value(row.get("sharpe_ratio"))
            trades = self._safe_value(row.get("trade_count"))
            open_pos = self._safe_value(row.get("open_position"))

            fields.append({
                "name": f"#{rank} — {sid}",
                "value": (
                    f"Return: {self._fmt_percent(total_ret)} | "
                    f"Sharpe: {self._fmt_number(sharpe)} | "
                    f"Trades: {trades} | "
                    f"Open: {open_pos}"
                ),
                "inline": False,
            })

        # Metadata footer
        meta_parts = [f"Showing top {len(displayed)} of {total_count} strategies"]
        if has_more:
            meta_parts.append(" — truncated for display")
        if ranking_metric:
            meta_parts.append(f" | Ranked by: {ranking_metric}")

        fields.append({
            "name": "📋 Metadata",
            "value": "".join(meta_parts),
            "inline": False,
        })

        return self._embed(
            title=f"⚔️ Compare: Best = {best_strategy_id}",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    def format_screener(self, result: dict) -> dict:
        """Render screener result — handles both results (screener/session) and candidates (order-block) shapes."""
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}

        # Determine shape: candidates (order_block) or results (screener/session)
        candidates = data.get("candidates")
        results = data.get("results")
        if isinstance(candidates, list) and candidates:
            rows = candidates
            row_shape = "candidates"
        elif isinstance(results, list) and results:
            rows = results
            row_shape = "results"
        elif isinstance(candidates, list):
            rows = candidates
            row_shape = "candidates"
        elif isinstance(results, list):
            rows = results
            row_shape = "results"
        else:
            rows = []
            row_shape = "results"

        failures = data.get("failures", [])

        fields = []
        MAX_ROWS = 5

        for row in rows[:MAX_ROWS]:
            symbol = self._safe_value(row.get("symbol"))
            exchange = self._safe_value(row.get("exchange"))

            # Score & emoji
            raw_score = row.get("score")
            score_normalized = self._normalize_score(raw_score)
            score_emoji = self._score_emoji(score_normalized) if score_normalized is not None else ""

            # Shape-specific detail
            if row_shape == "candidates":
                # order_block_screener_scan shape
                zone_type = self._safe_value(row.get("zone_type"))
                detail = zone_type
            else:
                # screener_scan / session_breakout_scan shape
                filters_matched = self._safe_value(row.get("filters_matched"))
                detail = f"Filters: {filters_matched}"

            price = self._safe_value(row.get("price"))
            fields.append({
                "name": f"{score_emoji} {symbol} ({exchange})",
                "value": f"Price: {self._fmt_price(price)} | {detail}",
                "inline": False,
            })

        # Row count metadata
        total = len(rows)
        fields.append({
            "name": "📋 Metadata",
            "value": f"Showing top {MAX_ROWS} of {total} results",
            "inline": False,
        })

        # Failures summary (compact)
        if failures:
            fail_count = len(failures)
            fields.append({
                "name": "⚠️ Failures",
                "value": f"{fail_count} symbol(s) failed scan",
                "inline": False,
            })

        return self._embed(
            title="🔍 Screener Results",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    def format_sentiment(self, result: dict) -> dict:
        """Render sentiment dashboard result."""
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}

        sentiment_polarity = data.get("sentiment_polarity", "neutral")
        normalized_polarity = str(sentiment_polarity).lower() if sentiment_polarity is not None else "neutral"
        mood = self._mood_emoji(sentiment_polarity)

        sentiment_index = self._safe_value(data.get("sentiment_index"))
        combination_mode = self._safe_value(data.get("combination_mode"))
        cross_signal_flags = data.get("cross_signal_flags")
        diagnostics = data.get("diagnostics")

        fields = [
            {
                "name": f"Mood {mood}",
                "value": f"Index: {self._fmt_number(sentiment_index, decimals=2)} | Mode: {combination_mode}",
                "inline": False,
            },
        ]

        if cross_signal_flags:
            flags_str = ", ".join(cross_signal_flags) if isinstance(cross_signal_flags, list) else str(cross_signal_flags)
            fields.append({
                "name": f"{EMOJI_CROSS_SIGNAL} Cross-Signal Flags",
                "value": flags_str,
                "inline": False,
            })

        if diagnostics:
            diag_field = self._diagnostics_field(diagnostics)
            if diag_field:
                fields.append(diag_field)

        # Determine color from polarity
        if normalized_polarity == "bullish":
            color = COLOR_POSITIVE
        elif normalized_polarity == "bearish":
            color = COLOR_NEGATIVE
        else:
            color = COLOR_NEUTRAL

        return self._embed(
            title=f"{EMOJI_VOLUME_SPIKE} Sentiment Dashboard",
            color=color,
            fields=self._cap_fields(fields),
        )

    def format_market(self, result: dict) -> dict:
        """Render market tool stubs — concise embed with symbol and available fields."""
        data = result.get("data", {})

        symbol = self._safe_value(data.get("symbol"))
        exchange = self._safe_value(data.get("exchange"))
        note = self._safe_value(data.get("note"))

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Exchange", "value": exchange, "inline": True},
        ]

        # Tool-specific fields
        timeframe = data.get("timeframe")
        limit = data.get("limit")
        if timeframe is not None:
            fields.append({"name": "Timeframe", "value": str(timeframe), "inline": True})
        if limit is not None:
            fields.append({"name": "Limit", "value": str(limit), "inline": True})

        if note and note != "N/A":
            fields.append({"name": "Note", "value": note, "inline": False})

        return self._embed(
            title=f"📡 Market: {symbol}",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    def format_analytical(self, result: dict) -> dict:
        """Render analytical tool outputs — scalar summary or list-row summarization."""
        data = result.get("data", {})
        tool = data.get("tool")

        # get_market_structure: scalar summary — render all fields inline
        if tool == "get_market_structure":
            summary = data.get("summary", {})
            fields = []
            for key, val in summary.items():
                fields.append({
                    "name": str(key).replace("_", " ").title(),
                    "value": self._safe_value(val),
                    "inline": True,
                })
            return self._embed(
                title=f"📊 Market Structure: {self._safe_value(data.get('symbol'))}",
                color=COLOR_DEFAULT,
                fields=self._cap_fields(fields),
            )

        # List-row tools: summarize count + top 5 rows in one field
        count = data.get("count", 0)
        rows = data.get("rows", [])

        MAX_ROWS = 5
        top_rows = rows[:MAX_ROWS] if isinstance(rows, list) else []

        summary_lines = [f"Count: {count}"]
        if top_rows:
            summary_lines.append("")
            for row in top_rows:
                summary_lines.append(self._summarize_row(row, tool))

        summary_text = "\n".join(summary_lines)

        # Cap the summary text to avoid Discord limits
        summary_text = self._truncate_text(summary_text, _GENERIC_TRUNCATE_LIMIT)

        fields = [
            {
                "name": "Results",
                "value": f"```\n{summary_text}\n```",
                "inline": False,
            },
        ]

        # Add count metadata if truncated
        if len(rows) > MAX_ROWS:
            fields.append({
                "name": "📋 Metadata",
                "value": f"Showing top {MAX_ROWS} of {len(rows)} rows",
                "inline": False,
            })

        return self._embed(
            title=f"📈 Analytical: {self._safe_value(data.get('symbol'))}",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    def format_indicator(self, result: dict) -> dict:
        """Render indicator_rsi result."""
        data = result.get("data", {})

        symbol = self._safe_value(data.get("symbol"))
        period = self._safe_value(data.get("period"))
        timeframe = self._safe_value(data.get("timeframe"))
        values = data.get("values", [])

        values_count = len(values) if isinstance(values, list) else "N/A"

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Period", "value": str(period), "inline": True},
            {"name": "Timeframe", "value": str(timeframe), "inline": True},
            {"name": "Values Count", "value": str(values_count), "inline": True},
        ]

        return self._embed(
            title=f"📉 RSI Indicator: {symbol}",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    def format_alert(self, alert: dict) -> dict:
        """Render caller-provided alert dict into a colored embed.

        Input schema: {symbol, signal, message, timestamp, confidence}
        Color: green for bullish, red for bearish, yellow for neutral/warning.
        """
        symbol = self._safe_value(alert.get("symbol"))
        signal = alert.get("signal", "neutral")
        message = self._safe_value(alert.get("message"))
        timestamp = self._valid_embed_timestamp(alert.get("timestamp"))
        confidence = alert.get("confidence")

        # Color by signal
        if signal in ("bullish", "long", "buy"):
            color = COLOR_POSITIVE
        elif signal in ("bearish", "short", "sell"):
            color = COLOR_NEGATIVE
        else:
            color = COLOR_NEUTRAL

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Signal", "value": str(signal), "inline": True},
        ]

        if confidence is not None:
            fields.append({
                "name": "Confidence",
                "value": f"{self._fmt_percent(confidence * 100 if isinstance(confidence, (int, float)) and confidence <= 1.0 else confidence)}",
                "inline": True,
            })

        if message and message != "N/A":
            fields.append({
                "name": "Message",
                "value": message,
                "inline": False,
            })

        title = f"🚨 Alert: {symbol}"
        return self._embed(
            title=title,
            color=color,
            fields=self._cap_fields(fields),
            timestamp=timestamp,
        )

    def format_error(self, result: dict) -> dict:
        """Render error envelope — red embed with code and message; include diagnostics if present."""
        error = result.get("error", {})
        if not isinstance(error, dict):
            return self._error_embed(
                code="INVALID_FORMAT_PAYLOAD",
                message="Expected error envelope error to be a dict",
            )

        code = self._safe_value(error.get("code"))
        message = self._safe_value(error.get("message"))

        fields = [
            {"name": "Code", "value": str(code), "inline": True},
            {"name": "Error", "value": str(message), "inline": False},
        ]

        # Diagnostics block if present
        diag_data = result.get("data")
        if isinstance(diag_data, dict) and diag_data != result.get("error"):
            diag_field = self._diagnostics_field(diag_data)
            if diag_field:
                fields.append(diag_field)

        return self._embed(
            title="❌ Error",
            color=COLOR_NEGATIVE,
            fields=self._cap_fields(fields),
        )

    def format_generic(self, result: dict) -> dict:
        """Render unknown tool result as indented JSON code block.

        Locked decision (Josh): hard-truncate oversized JSON for inline display; if it is
        still too large for Discord, write the full payload to temp storage and avoid
        exposing the local path.
        """
        data = result.get("data", {})

        # Serialize to JSON
        try:
            json_str = json.dumps(data, indent=2, default=str)
        except (TypeError, ValueError):
            json_str = "# [Serialization error]"

        # Truncate if needed
        truncated, _ = self._truncate_with_indicator(json_str, _GENERIC_TRUNCATE_LIMIT)

        inline_value = self._codeblock(truncated)

        # If still too large after hard truncation, write the full payload to temp storage.
        if len(inline_value) > DISCORD_FIELD_VALUE_LIMIT:
            tmp_path = self._write_payload_to_tmp(json_str)
            if tmp_path:
                value = self._codeblock(
                    "# Payload too large for Discord — written to local temp storage\n"
                    "# Local temp path intentionally omitted from Discord embed\n"
                    "# TODO: Upload to cloud/public hosting for shared access\n"
                )
            else:
                value = inline_value
        else:
            value = inline_value

        fields = [
            {
                "name": "Raw Data",
                "value": value,
                "inline": False,
            },
        ]

        return self._embed(
            title="📦 Tool Output",
            color=COLOR_DEFAULT,
            fields=self._cap_fields(fields),
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _embed(
        self,
        title: str,
        color: int,
        fields: list[dict],
        description: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        """Build a Discord embed dict."""
        embed: dict[str, Any] = {
            "title": self._sanitize_discord_text(title, DISCORD_TITLE_LIMIT, fallback="Untitled"),
            "color": color,
            "fields": self._cap_fields(fields),
        }
        if description:
            embed["description"] = self._sanitize_discord_text(
                description,
                DISCORD_DESCRIPTION_LIMIT,
                fallback="N/A",
            )
        if timestamp:
            embed["timestamp"] = timestamp
        return embed

    def _error_embed(self, code: str, message: str) -> dict:
        """Build an error embed for malformed input."""
        return self._embed(
            title="❌ Invalid Payload",
            color=COLOR_NEGATIVE,
            fields=[
                {"name": "Code", "value": str(code), "inline": True},
                {"name": "Message", "value": str(message), "inline": False},
            ],
        )

    def _safe_value(self, value: Any) -> str:
        """Return 'N/A' for None or non-finite floats; otherwise str."""
        if value is None:
            return "N/A"
        if isinstance(value, float) and not math.isfinite(value):
            return "N/A"
        return str(value)

    def _valid_embed_timestamp(self, value: Any) -> str | None:
        """Return a Discord-safe ISO-8601 timestamp string, else None."""
        if not isinstance(value, str):
            return None

        timestamp = value.strip()
        if not timestamp or timestamp == "N/A" or "T" not in timestamp:
            return None

        normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        try:
            datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return timestamp

    def _fmt_price(self, value: Any) -> str:
        """Format price — up to 4 decimal places."""
        if value == "N/A":
            return "N/A"
        try:
            f = float(value)
            return f"{f:.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_percent(self, value: Any) -> str:
        """Format percentage — 2 decimal places + % sign."""
        if value == "N/A":
            return "N/A"
        try:
            f = float(value)
            return f"{f:.2f}%"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_number(self, value: Any, decimals: int = 2) -> str:
        """Format a generic number to specified decimal places."""
        if value == "N/A":
            return "N/A"
        try:
            f = float(value)
            fmt = f"{{:.{decimals}f}}"
            return fmt.format(f)
        except (TypeError, ValueError):
            return str(value)

    def _normalize_score(self, score: Any) -> float | None:
        """Normalize score to 0-100 range.

        Handles both 0-1 scale and 0-100 scale.
        """
        if score is None:
            return None
        try:
            f = float(score)
        except (TypeError, ValueError):
            return None

        # Scale detection: if > 1.0, assume 0-100; otherwise 0-1
        if f > 1.0:
            return max(0.0, min(100.0, f))
        return max(0.0, min(100.0, f * 100.0))

    def _score_emoji(self, score: float | None) -> str:
        """Return emoji for normalized score threshold."""
        if score is None:
            return ""
        if score >= 70:
            return EMOJI_SCORE_LOCK
        if score >= 40:
            return EMOJI_SCORE_WARN
        return EMOJI_SCORE_BAD

    def _mood_emoji(self, polarity: str | None) -> str:
        """Return mood emoji for sentiment polarity."""
        if polarity is None:
            return EMOJI_MOOD_NEUTRAL
        p = str(polarity).lower()
        if p in ("bullish", "positive", "buy"):
            return EMOJI_MOOD_BULLISH
        if p in ("bearish", "negative", "sell"):
            return EMOJI_MOOD_BEARISH
        return EMOJI_MOOD_NEUTRAL

    def _diagnostics_field(self, data: Any) -> dict | None:
        """Build a compact Diagnostics field from data dict."""
        if not data:
            return None
        try:
            diag_json = json.dumps(data, default=str)
            diag_truncated, _ = self._truncate_with_indicator(diag_json, 500)
            return {
                "name": "🔧 Diagnostics",
                "value": self._codeblock(diag_truncated, language="json"),
                "inline": False,
            }
        except (TypeError, ValueError):
            return None

    def _codeblock(self, text: str, language: str | None = None) -> str:
        """Wrap text in a Discord code block."""
        if language:
            return f"```{language}\n{text}\n```"
        return f"```\n{text}\n```"

    def _truncate_text(self, text: str, limit: int) -> str:
        """Truncate text to limit chars, adding ellipsis if cut."""
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3] + "..."

    def _truncate_with_indicator(
        self, text: str, limit: int
    ) -> tuple[str, int]:
        """Truncate text to limit chars, appending '...' if cut. Returns (text, original_len)."""
        original_len = len(text)
        if original_len <= limit:
            return text, original_len
        return text[: limit - 3] + "...", original_len

    def _sanitize_discord_text(self, text: Any, limit: int, fallback: str = "N/A") -> str:
        """Escape mentions and hard-cap untrusted text for Discord embeds.

        Defense in depth: downstream Discord senders should still use
        allowed_mentions=none when posting embed payloads.
        """
        sanitized = re.sub(r"@(?!\u200b)", "@\u200b", self._safe_value(text))
        sanitized = self._truncate_text(sanitized, limit)
        return sanitized or fallback

    def _write_payload_to_tmp(self, payload_json: str) -> str | None:
        """Write payload JSON to the temp directory and return the file path.

        TODO: Upload to cloud/public hosting for shared access.
        """
        try:
            fd, filepath = tempfile.mkstemp(
                dir=_TMP_DIR,
                prefix="discord_payload_",
                suffix=".json",
                text=True,
            )
        except OSError:
            return None

        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                # Best-effort hardening only; some platforms lack fchmod or deny it.
                pass

            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload_json)

            try:
                os.chmod(filepath, 0o600)
            except OSError:
                # Best-effort hardening only; keep the temp file if chmod is rejected.
                pass
            return filepath
        except OSError:
            try:
                os.close(fd)
            except OSError:
                # Cleanup only; ignore close failures so the original write error still wins.
                pass
            try:
                os.unlink(filepath)
            except OSError:
                # Cleanup only; unlink failure is non-fatal here and intentionally ignored.
                pass
            return None

    def _cap_fields(self, fields: list[dict]) -> list[dict]:
        """Hard-cap fields and enforce Discord-safe field text limits."""
        capped_fields = []
        for field in fields[:DISCORD_TOTAL_FIELDS_LIMIT]:
            capped_fields.append({
                "name": self._sanitize_discord_text(
                    field.get("name"),
                    DISCORD_FIELD_NAME_LIMIT,
                    fallback="Field",
                ),
                "value": self._sanitize_discord_text(
                    field.get("value"),
                    DISCORD_FIELD_VALUE_LIMIT,
                    fallback="N/A",
                ),
                "inline": bool(field.get("inline", False)),
            })
        return capped_fields

    def _summarize_row(self, row: dict, tool: str | None = None) -> str:
        """Summarize a single row for analytical/screener list display."""
        if not isinstance(row, dict):
            return str(row)

        # Key fields to show per tool type
        if tool == "calculate_volume_profile":
            price = row.get("price", row.get("poc_price", "?"))
            volume = row.get("volume", "?")
            return f"  Price {price}: vol {volume}"
        if tool == "detect_order_blocks":
            zone_type = row.get("zone_type", "?")
            zone_high = row.get("zone_high", "?")
            zone_low = row.get("zone_low", "?")
            return f"  {zone_type}: [{zone_low} – {zone_high}]"
        if tool == "calculate_fibonacci":
            level = row.get("level", "?")
            price = row.get("price", "?")
            return f"  {level}: {price}"
        if tool == "calculate_tpo":
            tpo_count = row.get("tpo_count", row.get("count", "?"))
            price = row.get("price", "?")
            return f"  Price {price}: TPO {tpo_count}"
        if tool == "detect_elliot_wave":
            wave = row.get("wave", "?")
            score = row.get("score", "?")
            return f"  Wave {wave} (score {score})"
        if tool == "screener_scan" or tool == "session_breakout_scan":
            symbol = row.get("symbol", "?")
            score = row.get("score", "?")
            return f"  {symbol}: score {score}"
        if tool == "order_block_screener_scan":
            symbol = row.get("symbol", "?")
            zone_type = row.get("zone_type", "?")
            return f"  {symbol}: {zone_type}"

        # Generic fallback: show first few string/number fields
        parts = []
        for k, v in list(row.items())[:4]:
            parts.append(f"{k}={v}")
        return " | ".join(parts)
