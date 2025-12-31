"""Reward functions for RL trading environments.

Different reward signals encourage different trading behaviors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading.types import Account, Execution


class RewardFunction(ABC):
    """Base class for reward functions.

    Reward functions compute the reward signal given the current
    state transition in the trading environment.
    """

    @abstractmethod
    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Compute reward for a step.

        :param prev_equity: Portfolio equity before action.
        :param curr_equity: Portfolio equity after action.
        :param action: Action taken (0=hold, 1=buy, 2=sell).
        :param execution: Execution if an order was filled, None otherwise.
        :param account: Current account state.
        :param step: Current step number in episode.
        :param done: Whether episode is ending.
        :returns: Reward value.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state for new episode."""
        pass


class SimpleReturnReward(RewardFunction):
    """Reward based on simple return.

    Reward = (current_equity - previous_equity) / previous_equity

    :param scale: Multiplier for the reward.
    """

    def __init__(self, scale: float = 100.0) -> None:
        self.scale = scale

    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Return percentage change in equity."""
        if prev_equity <= 0:
            return 0.0
        return ((curr_equity - prev_equity) / prev_equity) * self.scale


class RiskAdjustedReward(RewardFunction):
    """Return adjusted for volatility.

    Tracks recent returns and divides by rolling volatility.

    :param lookback: Number of steps for volatility estimation.
    :param min_volatility: Floor for volatility to avoid division issues.
    """

    def __init__(self, lookback: int = 20, min_volatility: float = 0.001) -> None:
        self.lookback = lookback
        self.min_volatility = min_volatility
        self.returns: list[float] = []

    def reset(self) -> None:
        """Reset return history."""
        self.returns = []

    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Compute risk-adjusted return."""
        if prev_equity <= 0:
            return 0.0

        ret = (curr_equity - prev_equity) / prev_equity
        self.returns.append(ret)

        # Keep only recent returns
        if len(self.returns) > self.lookback:
            self.returns = self.returns[-self.lookback:]

        # Compute volatility
        if len(self.returns) < 2:
            return ret * 10  # Scale up early returns

        import numpy as np

        volatility = max(np.std(self.returns), self.min_volatility)
        return ret / volatility


class SharpeReward(RewardFunction):
    """Reward approximating Sharpe ratio.

    Computes differential Sharpe ratio - the marginal contribution
    of each step to the overall Sharpe.

    :param eta: Decay factor for rolling statistics.
    :param scale: Multiplier for the reward.
    """

    def __init__(self, eta: float = 0.01, scale: float = 1.0) -> None:
        self.eta = eta
        self.scale = scale
        self.A: float = 0.0  # Running mean return
        self.B: float = 0.0  # Running mean squared return

    def reset(self) -> None:
        """Reset running statistics."""
        self.A = 0.0
        self.B = 0.0

    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Compute differential Sharpe ratio."""
        if prev_equity <= 0:
            return 0.0

        ret = (curr_equity - prev_equity) / prev_equity

        # Update running statistics
        delta_A = ret - self.A
        delta_B = ret**2 - self.B

        # Differential Sharpe
        denom = (self.B - self.A**2) ** 1.5
        if abs(denom) < 1e-10:
            reward = ret
        else:
            reward = (
                self.B * delta_A - 0.5 * self.A * delta_B
            ) / denom

        # Update running means
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return reward * self.scale


class DrawdownPenaltyReward(RewardFunction):
    """Return with penalty for drawdowns.

    Combines return reward with a penalty proportional to
    current drawdown from peak equity.

    :param return_weight: Weight for return component.
    :param drawdown_weight: Weight for drawdown penalty.
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        drawdown_weight: float = 0.5,
    ) -> None:
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.peak_equity: float = 0.0

    def reset(self) -> None:
        """Reset peak tracking."""
        self.peak_equity = 0.0

    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Compute return with drawdown penalty."""
        if prev_equity <= 0:
            return 0.0

        # Update peak
        self.peak_equity = max(self.peak_equity, curr_equity)

        # Return component
        ret = (curr_equity - prev_equity) / prev_equity
        return_reward = ret * self.return_weight * 100

        # Drawdown penalty
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - curr_equity) / self.peak_equity
            drawdown_penalty = drawdown * self.drawdown_weight
        else:
            drawdown_penalty = 0.0

        return return_reward - drawdown_penalty


class ProfitPerTradeReward(RewardFunction):
    """Reward based on realized profit per trade.

    Only gives reward when positions are closed, based on
    actual P&L from the trade.

    :param scale: Multiplier for P&L reward.
    :param holding_penalty: Small penalty per step while holding.
    """

    def __init__(self, scale: float = 1.0, holding_penalty: float = 0.0001) -> None:
        self.scale = scale
        self.holding_penalty = holding_penalty

    def compute(
        self,
        prev_equity: float,
        curr_equity: float,
        action: int,
        execution: "Execution | None",
        account: "Account",
        step: int,
        done: bool,
    ) -> float:
        """Reward realized P&L on sells."""
        # Check if this was a sell (position closed)
        if execution is not None and execution.side == "sell":
            # Reward is the equity change (includes P&L from sale)
            if prev_equity > 0:
                return ((curr_equity - prev_equity) / prev_equity) * self.scale * 100
            return 0.0

        # Small penalty for holding to encourage action
        return -self.holding_penalty


