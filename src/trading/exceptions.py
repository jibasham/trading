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


class AccountError(TradingError, _CoreAccountError):
    """Raised when account state manipulation fails due to invalid input or limits."""


__all__ = ["TradingError", "AccountError"]
