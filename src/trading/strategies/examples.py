"""Example strategy implementations for testing and demonstration."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from trading.strategies.base import Strategy
from trading.types import OrderRequest, Symbol

if TYPE_CHECKING:
    from trading.types import Account, AnalysisSnapshot


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold strategy that buys once and holds.

    :param params: Configuration parameters:
        - symbol: Symbol to trade (default: "AAPL")
        - quantity: Number of shares to buy (default: 10)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize buy and hold strategy."""
        super().__init__(params)
        self.target_symbol = Symbol(self.params.get("symbol", "AAPL"))
        self.quantity = float(self.params.get("quantity", 10))
        self.has_bought = False

    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Buy once if we haven't already."""
        if not self.has_bought and snapshot.bars:
            self.has_bought = True
            return [
                OrderRequest(
                    symbol=self.target_symbol,
                    side="buy",
                    quantity=self.quantity,
                )
            ]
        return []


class MovingAverageCrossoverStrategy(Strategy):
    """Simple moving average crossover strategy.

    Buys when short MA crosses above long MA, sells when it crosses below.

    :param params: Configuration parameters:
        - symbol: Symbol to trade (default: "AAPL")
        - short_period: Short MA period (default: 5)
        - long_period: Long MA period (default: 20)
        - quantity: Shares per trade (default: 10)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize moving average crossover strategy."""
        super().__init__(params)
        self.target_symbol = Symbol(self.params.get("symbol", "AAPL"))
        self.short_period = int(self.params.get("short_period", 5))
        self.long_period = int(self.params.get("long_period", 20))
        self.quantity = float(self.params.get("quantity", 10))
        self.price_history: list[float] = []
        self.position_held = False

    def _calculate_ma(self, prices: list[float], period: int) -> float | None:
        """Calculate simple moving average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Generate signal based on MA crossover."""
        # Get current price for our target symbol
        symbol_key = str(self.target_symbol)
        if symbol_key not in snapshot.bars:
            return []

        bar = snapshot.bars[symbol_key]
        self.price_history.append(bar.close)

        # Need enough history for long MA
        if len(self.price_history) < self.long_period:
            return []

        short_ma = self._calculate_ma(self.price_history, self.short_period)
        long_ma = self._calculate_ma(self.price_history, self.long_period)

        if short_ma is None or long_ma is None:
            return []

        orders: list[OrderRequest] = []

        # Buy signal: short MA crosses above long MA
        if short_ma > long_ma and not self.position_held:
            self.position_held = True
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="buy",
                    quantity=self.quantity,
                )
            )
        # Sell signal: short MA crosses below long MA
        elif short_ma < long_ma and self.position_held:
            self.position_held = False
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="sell",
                    quantity=self.quantity,
                )
            )

        return orders


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy that buys dips and sells rallies.

    Buys when price falls below MA by threshold%, sells when above.

    :param params: Configuration parameters:
        - symbol: Symbol to trade (default: "AAPL")
        - ma_period: Moving average period (default: 20)
        - threshold: % deviation to trigger trade (default: 0.02 = 2%)
        - quantity: Shares per trade (default: 10)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize mean reversion strategy."""
        super().__init__(params)
        self.target_symbol = Symbol(self.params.get("symbol", "AAPL"))
        self.ma_period = int(self.params.get("ma_period", 20))
        self.threshold = float(self.params.get("threshold", 0.02))
        self.quantity = float(self.params.get("quantity", 10))
        self.price_history: list[float] = []
        self.position_held = False

    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Generate signal based on deviation from moving average."""
        symbol_key = str(self.target_symbol)
        if symbol_key not in snapshot.bars:
            return []

        bar = snapshot.bars[symbol_key]
        current_price = bar.close
        self.price_history.append(current_price)

        if len(self.price_history) < self.ma_period:
            return []

        ma = sum(self.price_history[-self.ma_period :]) / self.ma_period
        deviation = (current_price - ma) / ma

        orders: list[OrderRequest] = []

        # Buy when price is below MA by threshold
        if deviation < -self.threshold and not self.position_held:
            self.position_held = True
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="buy",
                    quantity=self.quantity,
                )
            )
        # Sell when price is above MA by threshold
        elif deviation > self.threshold and self.position_held:
            self.position_held = False
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="sell",
                    quantity=self.quantity,
                )
            )

        return orders


class RSIStrategy(Strategy):
    """RSI-based strategy that buys oversold and sells overbought.

    Uses Relative Strength Index to identify potential reversals.

    :param params: Configuration parameters:
        - symbol: Symbol to trade (default: "AAPL")
        - rsi_period: RSI calculation period (default: 14)
        - oversold: RSI level to buy (default: 30)
        - overbought: RSI level to sell (default: 70)
        - quantity: Shares per trade (default: 10)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize RSI strategy."""
        super().__init__(params)
        self.target_symbol = Symbol(self.params.get("symbol", "AAPL"))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.oversold = float(self.params.get("oversold", 30))
        self.overbought = float(self.params.get("overbought", 70))
        self.quantity = float(self.params.get("quantity", 10))
        self.price_history: list[float] = []
        self.position_held = False

    def _calculate_rsi(self) -> float | None:
        """Calculate RSI from price history."""
        if len(self.price_history) < self.rsi_period + 1:
            return None

        # Calculate price changes
        changes = [
            self.price_history[i] - self.price_history[i - 1]
            for i in range(1, len(self.price_history))
        ]

        # Get recent changes for RSI period
        recent_changes = changes[-self.rsi_period :]

        gains = [c for c in recent_changes if c > 0]
        losses = [-c for c in recent_changes if c < 0]

        avg_gain = sum(gains) / self.rsi_period if gains else 0
        avg_loss = sum(losses) / self.rsi_period if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Generate signal based on RSI levels."""
        symbol_key = str(self.target_symbol)
        if symbol_key not in snapshot.bars:
            return []

        bar = snapshot.bars[symbol_key]
        self.price_history.append(bar.close)

        rsi = self._calculate_rsi()
        if rsi is None:
            return []

        orders: list[OrderRequest] = []

        # Buy when oversold
        if rsi < self.oversold and not self.position_held:
            self.position_held = True
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="buy",
                    quantity=self.quantity,
                )
            )
        # Sell when overbought
        elif rsi > self.overbought and self.position_held:
            self.position_held = False
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="sell",
                    quantity=self.quantity,
                )
            )

        return orders


class RandomStrategy(Strategy):
    """Random strategy for benchmarking purposes.

    Makes random buy/sell decisions with configurable probability.

    :param params: Configuration parameters:
        - symbol: Symbol to trade (default: "AAPL")
        - trade_probability: Chance of trading each bar (default: 0.05)
        - quantity: Shares per trade (default: 10)
        - seed: Random seed for reproducibility (default: None)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize random strategy."""
        super().__init__(params)
        self.target_symbol = Symbol(self.params.get("symbol", "AAPL"))
        self.trade_probability = float(self.params.get("trade_probability", 0.05))
        self.quantity = float(self.params.get("quantity", 10))
        seed = self.params.get("seed")
        if seed is not None:
            random.seed(seed)
        self.position_held = False

    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Randomly decide to trade."""
        symbol_key = str(self.target_symbol)
        if symbol_key not in snapshot.bars:
            return []

        if random.random() > self.trade_probability:
            return []

        orders: list[OrderRequest] = []

        if not self.position_held:
            self.position_held = True
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="buy",
                    quantity=self.quantity,
                )
            )
        else:
            self.position_held = False
            orders.append(
                OrderRequest(
                    symbol=self.target_symbol,
                    side="sell",
                    quantity=self.quantity,
                )
            )

        return orders
