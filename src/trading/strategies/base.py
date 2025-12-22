"""Base strategy class that all trading strategies must implement.

Strategies receive market snapshots and decide what orders to place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from trading.types import OrderRequest

if TYPE_CHECKING:
    from trading.types import Account, AnalysisSnapshot


class Strategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement the `decide` method, which receives the
    current market snapshot and account state, and returns a list of order
    requests.

    Strategies can optionally implement `update_from_reward` for reinforcement
    learning use cases.

    :param params: Strategy-specific configuration parameters.

    Example usage::

        class BuyAndHoldStrategy(Strategy):
            def __init__(self, params: dict[str, Any]) -> None:
                super().__init__(params)
                self.target_symbol = params.get("symbol", "AAPL")
                self.has_position = False

            def decide(
                self,
                snapshot: AnalysisSnapshot,
                account: Account,
            ) -> list[OrderRequest]:
                if not self.has_position and snapshot.bars:
                    self.has_position = True
                    return [OrderRequest(
                        symbol=self.target_symbol,
                        side="buy",
                        quantity=100.0,
                    )]
                return []
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize strategy with parameters.

        :param params: Strategy-specific configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> list[OrderRequest]:
        """Decide what orders to place given current market state.

        This method is called once per time slice during backtesting or
        live trading. It should analyze the current market snapshot and
        account state, then return a list of order requests.

        :param snapshot: Current market data and analysis context.
        :param account: Current account state (balances, positions).
        :returns: List of order requests to execute (can be empty).
        """
        ...

    def update_from_reward(
        self,
        reward: float,
        snapshot: AnalysisSnapshot,
        account: Account,
    ) -> None:
        """Update strategy based on reward signal (for RL strategies).

        This method is called after order execution with a reward signal
        that can be used for reinforcement learning. The default implementation
        does nothing - override in RL-based strategies.

        :param reward: Reward signal (e.g., PnL change, risk-adjusted return).
        :param snapshot: Market snapshot at time of reward.
        :param account: Account state at time of reward.
        """
        pass

    def on_start(self) -> None:
        """Called when backtest/trading session starts.

        Override to perform any initialization that requires market data
        to be available.
        """
        pass

    def on_end(self) -> None:
        """Called when backtest/trading session ends.

        Override to perform any cleanup or final computations.
        """
        pass



