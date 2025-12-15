"""Trading package root."""

try:
    # Import Rust extension module
    from trading._core import hello_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    hello_rust = None

from trading.exceptions import AccountError, TradingError  # noqa: E402  (import after try/except)

__all__ = ["RUST_AVAILABLE", "hello_rust", "AccountError", "TradingError"]
