"""Wrappers to integrate RL models with the trading system.

Allows trained RL models to be used as regular Strategy objects
for backtesting and live trading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

from trading.rl.features import FeatureExtractor, OHLCVFeatures
from trading.strategies.base import Strategy
from trading.types import (
    Account,
    AnalysisSnapshot,
    NormalizedBar,
    OrderRequest,
    Symbol,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class PolicyProtocol(Protocol):
    """Protocol for RL policy/model interface."""

    def predict(
        self,
        observation: "NDArray[np.float32]",
        deterministic: bool = True,
    ) -> tuple[int | "NDArray", Any]:
        """Predict action from observation.

        :param observation: Current observation.
        :param deterministic: Use deterministic policy.
        :returns: (action, state) tuple.
        """
        ...


class RLStrategy(Strategy):
    """Wrapper to use trained RL model as a Strategy.

    This allows RL models to be used in backtesting, paper trading,
    or live trading through the standard Strategy interface.

    Example::

        from stable_baselines3 import PPO
        from trading.rl import RLStrategy, TradingEnv

        # Train a model
        env = TradingEnv(bars)
        model = PPO("MlpPolicy", env).learn(10000)

        # Use as strategy
        strategy = RLStrategy(
            model=model,
            symbol="AAPL",
            trade_quantity=10,
        )

        # Run backtest
        backtest = Backtest(bars, strategy)
        result = backtest.run()

    :param model: Trained RL model with predict() method.
    :param symbol: Symbol to trade.
    :param trade_quantity: Shares per trade.
    :param feature_extractor: Feature extractor matching training.
    :param deterministic: Use deterministic predictions.
    """

    def __init__(
        self,
        model: PolicyProtocol,
        symbol: str,
        trade_quantity: float = 10.0,
        feature_extractor: FeatureExtractor | None = None,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.symbol = symbol
        self.trade_quantity = trade_quantity
        self.feature_extractor = feature_extractor or OHLCVFeatures(lookback=10)
        self.deterministic = deterministic

        # Track bars seen for feature extraction
        self._bars_history: list[NormalizedBar] = []

    def on_start(self) -> None:
        """Reset state at start of run."""
        self._bars_history = []

    def on_end(self) -> None:
        """Cleanup at end of run."""
        self._bars_history = []

    def decide(
        self, snapshot: AnalysisSnapshot, account: Account
    ) -> list[OrderRequest]:
        """Get orders from RL model prediction.

        :param snapshot: Current market snapshot.
        :param account: Current account state.
        :returns: List of order requests (0 or 1).
        """
        # Get current bar
        bar = snapshot.bars.get(self.symbol)
        if bar is None:
            return []

        # Add to history
        self._bars_history.append(bar)

        # Extract features
        current_idx = len(self._bars_history) - 1
        observation = self.feature_extractor.extract(
            self._bars_history, current_idx, account
        )

        # Get model prediction
        action, _ = self.model.predict(observation, deterministic=self.deterministic)

        # Convert to int if needed
        if hasattr(action, "item"):
            action = action.item()

        # Map action to orders
        if action == 0:  # Hold
            return []

        elif action == 1:  # Buy
            # Check if we have cash
            price = bar.close
            cost = self.trade_quantity * price

            if cost > account.cleared_balance:
                return []

            return [
                OrderRequest(
                    symbol=Symbol(self.symbol),
                    side="buy",
                    quantity=self.trade_quantity,
                )
            ]

        elif action == 2:  # Sell
            # Check if we have position
            position = account.positions.get(self.symbol)
            if not position or position.quantity <= 0:
                return []

            sell_qty = min(self.trade_quantity, position.quantity)
            return [
                OrderRequest(
                    symbol=Symbol(self.symbol),
                    side="sell",
                    quantity=sell_qty,
                )
            ]

        return []


class SimplePolicy:
    """Simple rule-based policy for testing.

    Not an RL model, but follows the same interface.

    :param action_fn: Function that takes observation and returns action.
    """

    def __init__(self, action_fn: Callable[["NDArray[np.float32]"], int]) -> None:
        self.action_fn = action_fn

    def predict(
        self,
        observation: "NDArray[np.float32]",
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Predict action using rule-based function."""
        return self.action_fn(observation), None


class RandomPolicy:
    """Random policy for baseline comparison.

    :param seed: Random seed.
    :param buy_prob: Probability of buy action.
    :param sell_prob: Probability of sell action.
    """

    def __init__(
        self,
        seed: int | None = None,
        buy_prob: float = 0.1,
        sell_prob: float = 0.1,
    ) -> None:
        import numpy as np

        self.rng = np.random.default_rng(seed)
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob

    def predict(
        self,
        observation: "NDArray[np.float32]",
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Random action selection."""
        r = self.rng.random()
        if r < self.buy_prob:
            return 1, None  # Buy
        elif r < self.buy_prob + self.sell_prob:
            return 2, None  # Sell
        return 0, None  # Hold

