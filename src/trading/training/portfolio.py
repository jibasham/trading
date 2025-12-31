"""Portfolio allocation strategies for multi-symbol trading.

Provides methods for allocating capital across multiple symbols based on
various strategies like equal weight, risk parity, momentum, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from trading.types import NormalizedBar

if TYPE_CHECKING:
    pass


class AllocationWeights(BaseModel):
    """Portfolio allocation weights.

    :param weights: Dictionary mapping symbol to allocation weight (0-1).
    :param timestamp: When these weights were computed.
    """

    weights: dict[str, float]
    timestamp: datetime | None = None

    def normalize(self) -> "AllocationWeights":
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total == 0:
            return self
        return AllocationWeights(
            weights={k: v / total for k, v in self.weights.items()},
            timestamp=self.timestamp,
        )


class AllocationStrategy(ABC):
    """Base class for portfolio allocation strategies."""

    @abstractmethod
    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute allocation weights for symbols.

        :param symbols: List of symbols to allocate across.
        :param bars_by_symbol: Historical bars for each symbol.
        :param current_time: Current timestamp for the allocation.
        :returns: Allocation weights.
        """
        pass


class EqualWeightAllocation(AllocationStrategy):
    """Equal weight allocation across all symbols.

    Allocates the same percentage to each symbol.
    """

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute equal weights for all symbols."""
        if not symbols:
            return AllocationWeights(weights={}, timestamp=current_time)

        weight = 1.0 / len(symbols)
        return AllocationWeights(
            weights={s: weight for s in symbols},
            timestamp=current_time,
        )


class MarketCapWeightAllocation(AllocationStrategy):
    """Market cap weighted allocation.

    Allocates based on relative market capitalization (proxied by price * volume).

    :param lookback_bars: Number of bars to average for market cap estimate.
    """

    def __init__(self, lookback_bars: int = 20) -> None:
        self.lookback_bars = lookback_bars

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute weights based on relative market cap (price * volume)."""
        market_caps = {}

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            if not bars:
                market_caps[symbol] = 0
                continue

            recent = bars[-self.lookback_bars:]
            avg_price = sum(b.close for b in recent) / len(recent)
            avg_volume = sum(b.volume for b in recent) / len(recent)
            market_caps[symbol] = avg_price * avg_volume

        weights = AllocationWeights(weights=market_caps, timestamp=current_time)
        return weights.normalize()


class MomentumAllocation(AllocationStrategy):
    """Momentum-based allocation.

    Allocates more to symbols with stronger recent returns.

    :param lookback_days: Days to measure momentum over.
    :param top_n: Only allocate to top N momentum symbols (None = all).
    """

    def __init__(self, lookback_days: int = 30, top_n: int | None = None) -> None:
        self.lookback_days = lookback_days
        self.top_n = top_n

    def _compute_momentum(self, bars: list[NormalizedBar]) -> float:
        """Compute momentum as return over lookback period."""
        if len(bars) < 2:
            return 0.0

        start_price = bars[0].close
        end_price = bars[-1].close

        if start_price <= 0:
            return 0.0

        return (end_price - start_price) / start_price

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute weights based on recent momentum."""
        momentums = {}

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            # Filter to lookback period
            # Approximate: use last N bars
            lookback_bars = bars[-self.lookback_days:] if bars else []
            momentums[symbol] = self._compute_momentum(lookback_bars)

        # Filter to top N if specified
        if self.top_n is not None:
            sorted_symbols = sorted(
                momentums.keys(), key=lambda s: momentums[s], reverse=True
            )
            top_symbols = set(sorted_symbols[: self.top_n])
            momentums = {s: m for s, m in momentums.items() if s in top_symbols}

        # Convert to weights (use positive momentum only)
        weights = {s: max(0, m) for s, m in momentums.items()}

        allocation = AllocationWeights(weights=weights, timestamp=current_time)
        return allocation.normalize()


class InverseVolatilityAllocation(AllocationStrategy):
    """Inverse volatility (risk parity lite) allocation.

    Allocates more to lower volatility symbols.

    :param lookback_days: Days to measure volatility over.
    """

    def __init__(self, lookback_days: int = 30) -> None:
        self.lookback_days = lookback_days

    def _compute_volatility(self, bars: list[NormalizedBar]) -> float:
        """Compute annualized volatility from daily returns."""
        if len(bars) < 2:
            return float("inf")

        returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].close > 0:
                ret = (bars[i].close - bars[i - 1].close) / bars[i - 1].close
                returns.append(ret)

        if not returns:
            return float("inf")

        # Standard deviation of returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        daily_vol = variance ** 0.5

        # Annualize (252 trading days)
        return daily_vol * (252 ** 0.5)

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute weights inversely proportional to volatility."""
        inv_vols = {}

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            lookback_bars = bars[-self.lookback_days:] if bars else []
            vol = self._compute_volatility(lookback_bars)

            if vol > 0 and vol != float("inf"):
                inv_vols[symbol] = 1.0 / vol
            else:
                inv_vols[symbol] = 0.0

        allocation = AllocationWeights(weights=inv_vols, timestamp=current_time)
        return allocation.normalize()


class MinVarianceAllocation(AllocationStrategy):
    """Minimum variance allocation.

    Simplified version that uses inverse variance as proxy.

    :param lookback_days: Days to measure variance over.
    """

    def __init__(self, lookback_days: int = 60) -> None:
        self.lookback_days = lookback_days

    def _compute_variance(self, bars: list[NormalizedBar]) -> float:
        """Compute variance of returns."""
        if len(bars) < 2:
            return float("inf")

        returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].close > 0:
                ret = (bars[i].close - bars[i - 1].close) / bars[i - 1].close
                returns.append(ret)

        if not returns:
            return float("inf")

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return variance

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Compute weights inversely proportional to variance."""
        inv_vars = {}

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            lookback_bars = bars[-self.lookback_days:] if bars else []
            var = self._compute_variance(lookback_bars)

            if var > 0 and var != float("inf"):
                inv_vars[symbol] = 1.0 / var
            else:
                inv_vars[symbol] = 0.0

        allocation = AllocationWeights(weights=inv_vars, timestamp=current_time)
        return allocation.normalize()


class CustomWeightAllocation(AllocationStrategy):
    """Custom fixed weight allocation.

    :param weights: Dictionary of symbol to weight.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[NormalizedBar]],
        current_time: datetime | None = None,
    ) -> AllocationWeights:
        """Return predefined weights."""
        filtered = {s: self._weights.get(s, 0) for s in symbols}
        return AllocationWeights(weights=filtered, timestamp=current_time).normalize()


def get_allocation_strategy(name: str, **kwargs: Any) -> AllocationStrategy:
    """Factory function to create allocation strategies by name.

    :param name: Strategy name ('equal', 'momentum', 'inverse_vol', 'min_variance').
    :param kwargs: Strategy-specific parameters.
    :returns: AllocationStrategy instance.
    """
    strategies = {
        "equal": EqualWeightAllocation,
        "market_cap": MarketCapWeightAllocation,
        "momentum": MomentumAllocation,
        "inverse_vol": InverseVolatilityAllocation,
        "min_variance": MinVarianceAllocation,
    }

    if name not in strategies:
        raise ValueError(f"Unknown allocation strategy: {name}")

    return strategies[name](**kwargs)

