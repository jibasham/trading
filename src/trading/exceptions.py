"""Trading system exception hierarchy.

All trading-specific exceptions derive from :class:`TradingError` so callers can
catch all trading-related errors uniformly.
"""

from __future__ import annotations

try:  # pragma: no cover
    from trading._core import AccountError as _CoreAccountError
except ImportError:  # pragma: no cover

    class _CoreAccountError(Exception):  # type: ignore[no-redef]
        """Fallback base for account errors when Rust extension is unavailable."""

        ...


class TradingError(Exception):
    """Base class for trading-related exceptions.

    Derived exceptions should extend this class so that callers can catch all
    trading-specific errors uniformly.
    """


class ConfigError(TradingError):
    """Raised when configuration files or parameters are invalid."""


class DataSourceError(TradingError):
    """Raised when accessing or processing a data source fails."""


class DataValidationError(TradingError):
    """Raised when data fails validation checks.

    Named DataValidationError to avoid conflict with pydantic's ValidationError.
    """


class StorageError(TradingError):
    """Raised when reading from or writing to storage fails."""


class StrategyError(TradingError):
    """Raised when strategy execution fails."""


class AccountError(TradingError, _CoreAccountError):
    """Raised when account state manipulation fails due to invalid input or limits."""


__all__ = [
    "TradingError",
    "ConfigError",
    "DataSourceError",
    "DataValidationError",
    "StorageError",
    "StrategyError",
    "AccountError",
]
