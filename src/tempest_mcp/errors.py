"""
Tempest MCP Error Hierarchy.

Error code taxonomy:
- 1xxx: Validation errors
- 2xxx: Authentication/authorization errors
- 3xxx: Data source errors (Yahoo Finance, TradingView, CCXT)
- 5xxx: Indicator calculation errors
- 9xxx: Internal/unexpected errors
"""



class TempestError(Exception):
    """Base exception for all Tempest MCP errors."""

    def __init__(self, message: str, code: int = 9000) -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ValidationError(TempestError):
    """Validation errors (1xxx range)."""

    def __init__(self, message: str, code: int = 1000) -> None:
        super().__init__(message, code)


class DataSourceError(TempestError):
    """Base class for data source errors (3xxx range)."""

    def __init__(self, message: str, code: int = 3000) -> None:
        super().__init__(message, code)


class TradingViewError(DataSourceError):
    """TradingView API errors (3001-3005 range).

    Code assignments:
    - 3001: Authentication error (invalid API key)
    - 3002: Rate limit exceeded
    - 3003: Invalid symbol
    - 3004: Data unavailable
    - 3005: Connection/timeout error
    """

    def __init__(self, message: str, code: int = 3001) -> None:
        if code < 3001 or code > 3005:
            code = 3001
        super().__init__(message, code)


class CCXTError(DataSourceError):
    """CCXT adapter errors (3101-3105 range).

    Code assignments:
    - 3101: Exchange connection error
    - 3102: Rate limit exceeded
    - 3103: Invalid symbol
    - 3104: Data unavailable
    - 3105: Network timeout
    """

    def __init__(self, message: str, code: int = 3101) -> None:
        if code < 3101 or code > 3105:
            code = 3101
        super().__init__(message, code)


class YFinanceError(DataSourceError):
    """Yahoo Finance adapter errors (3201-3205 range)."""

    def __init__(self, message: str, code: int = 3201) -> None:
        if code < 3201 or code > 3205:
            code = 3201
        super().__init__(message, code)


class IndicatorError(TempestError):
    """Indicator calculation errors (5xxx range)."""

    def __init__(self, message: str, code: int = 5000) -> None:
        super().__init__(message, code)


class InternalError(TempestError):
    """Internal/unexpected errors (9xxx range)."""

    def __init__(self, message: str, code: int = 9000) -> None:
        super().__init__(message, code)
