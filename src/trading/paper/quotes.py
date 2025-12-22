"""Live quote sources for paper trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from trading.types import NormalizedBar, Symbol

if TYPE_CHECKING:
    pass


class QuoteSource(ABC):
    """Abstract base class for live quote sources."""

    @abstractmethod
    def get_quotes(self, symbols: list[Symbol]) -> dict[str, NormalizedBar]:
        """Get current quotes for the given symbols.

        :param symbols: List of symbols to fetch.
        :returns: Dictionary mapping symbol to current bar data.
        """
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        :returns: True if market is open for trading.
        """
        pass


class LiveQuoteSource(QuoteSource):
    """Live quote source using Yahoo Finance.

    Polls Yahoo Finance for current quotes. Works during and outside
    market hours (returns last available quote when closed).

    :param cache_seconds: How long to cache quotes before re-fetching.
    """

    def __init__(self, cache_seconds: float = 5.0) -> None:
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[datetime, NormalizedBar]] = {}

    def get_quotes(self, symbols: list[Symbol]) -> dict[str, NormalizedBar]:
        """Get current quotes from Yahoo Finance.

        :param symbols: List of symbols to fetch.
        :returns: Dictionary mapping symbol to current bar data.
        """
        import yfinance as yf

        now = datetime.now(timezone.utc)
        result: dict[str, NormalizedBar] = {}
        symbols_to_fetch: list[str] = []

        # Check cache first
        for sym in symbols:
            sym_str = str(sym)
            if sym_str in self._cache:
                cached_time, cached_bar = self._cache[sym_str]
                age = (now - cached_time).total_seconds()
                if age < self.cache_seconds:
                    result[sym_str] = cached_bar
                    continue
            symbols_to_fetch.append(sym_str)

        if not symbols_to_fetch:
            return result

        # Fetch from Yahoo
        try:
            tickers = yf.Tickers(" ".join(symbols_to_fetch))

            for sym_str in symbols_to_fetch:
                try:
                    ticker = tickers.tickers.get(sym_str)
                    if ticker is None:
                        continue

                    # Get the most recent data
                    hist = ticker.history(period="1d", interval="1m")
                    if hist.empty:
                        # Try daily data as fallback
                        hist = ticker.history(period="5d")

                    if hist.empty:
                        continue

                    # Get the last row
                    last_row = hist.iloc[-1]
                    bar = NormalizedBar(
                        symbol=Symbol(sym_str),
                        timestamp=now,
                        open=float(last_row["Open"]),
                        high=float(last_row["High"]),
                        low=float(last_row["Low"]),
                        close=float(last_row["Close"]),
                        volume=float(last_row["Volume"]),
                    )
                    result[sym_str] = bar
                    self._cache[sym_str] = (now, bar)
                except Exception:
                    # Skip symbols that fail
                    continue

        except Exception:
            # Return whatever we have from cache
            pass

        return result

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open.

        Uses simple time-based check for NYSE hours:
        9:30 AM - 4:00 PM Eastern, Mon-Fri.

        :returns: True if market is open.
        """
        from datetime import time

        try:
            # Get current time in US Eastern
            import zoneinfo

            eastern = zoneinfo.ZoneInfo("America/New_York")
        except ImportError:
            # Fallback for older Python
            from datetime import timedelta

            # Approximate Eastern time (doesn't handle DST perfectly)
            eastern = timezone(timedelta(hours=-5))

        now = datetime.now(eastern)

        # Check weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check time (9:30 AM - 4:00 PM)
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()

        return market_open <= current_time <= market_close

    def clear_cache(self) -> None:
        """Clear the quote cache."""
        self._cache.clear()


class MockQuoteSource(QuoteSource):
    """Mock quote source for testing.

    Returns pre-configured quotes for testing purposes.

    :param quotes: Dictionary of symbol to bar data.
    :param market_open: Whether to report market as open.
    """

    def __init__(
        self,
        quotes: dict[str, NormalizedBar] | None = None,
        market_open: bool = True,
    ) -> None:
        self._quotes = quotes or {}
        self._market_open = market_open

    def set_quotes(self, quotes: dict[str, NormalizedBar]) -> None:
        """Set the quotes to return."""
        self._quotes = quotes

    def set_market_open(self, is_open: bool) -> None:
        """Set whether market is reported as open."""
        self._market_open = is_open

    def get_quotes(self, symbols: list[Symbol]) -> dict[str, NormalizedBar]:
        """Return pre-configured quotes."""
        return {str(s): self._quotes[str(s)] for s in symbols if str(s) in self._quotes}

    def is_market_open(self) -> bool:
        """Return configured market status."""
        return self._market_open


