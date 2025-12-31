"""Gymnasium-compatible trading environment.

Wraps the trading system to provide a standard RL interface.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from trading._core import execute_orders
from trading.rl.features import FeatureExtractor, OHLCVFeatures
from trading.rl.rewards import RewardFunction, SimpleReturnReward
from trading.types import Account, Execution, NormalizedBar, Symbol


class TradingEnvConfig(BaseModel):
    """Configuration for TradingEnv.

    :param symbol: Symbol to trade.
    :param initial_balance: Starting cash balance.
    :param trade_quantity: Shares to trade per action.
    :param commission_per_trade: Commission per trade.
    :param slippage_pct: Slippage percentage.
    :param max_steps: Maximum steps per episode (None = full data).
    :param random_start: Start from random position in data.
    """

    symbol: str = "STOCK"
    initial_balance: float = 10000.0
    trade_quantity: float = 10.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0
    max_steps: int | None = None
    random_start: bool = False


class TradingEnv(gym.Env):
    """Gymnasium environment for trading a single symbol.

    Actions:
        0: Hold (do nothing)
        1: Buy (if have cash)
        2: Sell (if have position)

    Observations:
        Feature vector from the configured FeatureExtractor.

    Example::

        from trading.rl import TradingEnv, TradingEnvConfig

        config = TradingEnvConfig(
            symbol="AAPL",
            initial_balance=10000.0,
            trade_quantity=10,
        )

        env = TradingEnv(bars, config)

        obs, info = env.reset()
        while True:
            action = agent.choose(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    :param bars: List of NormalizedBar data for the symbol.
    :param config: Environment configuration.
    :param feature_extractor: Feature extractor (default: OHLCVFeatures).
    :param reward_function: Reward function (default: SimpleReturnReward).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        bars: list[NormalizedBar],
        config: TradingEnvConfig | None = None,
        feature_extractor: FeatureExtractor | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        super().__init__()

        self.bars = sorted(bars, key=lambda b: b.timestamp)
        self.config = config or TradingEnvConfig()

        # Set up feature extractor
        self.feature_extractor = feature_extractor or OHLCVFeatures(lookback=10)

        # Set up reward function
        self.reward_function = reward_function or SimpleReturnReward()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        num_features = self.feature_extractor.num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32,
        )

        # Episode state
        self._current_idx: int = 0
        self._start_idx: int = 0
        self._account: Account | None = None
        self._prev_equity: float = 0.0
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._executions: list[Execution] = []

        # For rendering
        self._last_action: int = 0
        self._last_reward: float = 0.0

    def _create_account(self) -> Account:
        """Create a fresh account."""
        return Account(
            account_id="rl-env",
            base_currency="USD",
            cleared_balance=self.config.initial_balance,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
            clearing_delay_hours=0,
            use_business_days=False,
            pending_transactions=[],
        )

    def _get_equity(self) -> float:
        """Calculate current portfolio equity."""
        if self._account is None:
            return self.config.initial_balance

        cash = self._account.cleared_balance
        position = self._account.positions.get(self.config.symbol)

        if position and position.quantity > 0:
            current_price = self.bars[self._current_idx].close
            position_value = position.quantity * current_price
        else:
            position_value = 0.0

        return cash + position_value

    def _get_observation(self) -> NDArray[np.float32]:
        """Get current observation."""
        if self._account is None:
            self._account = self._create_account()

        return self.feature_extractor.extract(
            self.bars, self._current_idx, self._account
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to initial state.

        :param seed: Random seed for reproducibility.
        :param options: Additional options (unused).
        :returns: Initial observation and info dict.
        """
        super().reset(seed=seed)

        # Reset reward function
        self.reward_function.reset()

        # Determine starting index
        if self.config.random_start and len(self.bars) > 50:
            # Start somewhere in the first 80% of data
            max_start = int(len(self.bars) * 0.8)
            self._start_idx = self.np_random.integers(0, max_start)
        else:
            self._start_idx = 0

        self._current_idx = self._start_idx
        self._account = self._create_account()
        self._prev_equity = self._get_equity()
        self._step_count = 0
        self._total_reward = 0.0
        self._executions = []
        self._last_action = 0
        self._last_reward = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        :param action: 0=hold, 1=buy, 2=sell.
        :returns: (observation, reward, terminated, truncated, info).
        """
        assert self._account is not None

        self._step_count += 1
        self._last_action = action

        # Store previous equity for reward
        prev_equity = self._get_equity()

        # Execute action
        execution = self._execute_action(action)

        # Advance to next bar
        self._current_idx += 1

        # Check if episode is done
        terminated = False
        truncated = False

        # Check for end of data
        if self._current_idx >= len(self.bars):
            self._current_idx = len(self.bars) - 1
            terminated = True

        # Check for max steps
        if self.config.max_steps and self._step_count >= self.config.max_steps:
            truncated = True

        # Check for bankruptcy
        current_equity = self._get_equity()
        if current_equity <= 0:
            terminated = True

        # Compute reward
        reward = self.reward_function.compute(
            prev_equity=prev_equity,
            curr_equity=current_equity,
            action=action,
            execution=execution,
            account=self._account,
            step=self._step_count,
            done=terminated or truncated,
        )

        self._prev_equity = current_equity
        self._last_reward = reward
        self._total_reward += reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Execution | None:
        """Execute the given action.

        :param action: 0=hold, 1=buy, 2=sell.
        :returns: Execution if order was filled, None otherwise.
        """
        if action == 0:  # Hold
            return None

        if self._account is None:
            return None

        symbol = self.config.symbol
        current_bar = self.bars[self._current_idx]

        if action == 1:  # Buy
            # Check if we have enough cash
            price = current_bar.close * (1 + self.config.slippage_pct)
            cost = self.config.trade_quantity * price + self.config.commission_per_trade

            if cost > self._account.cleared_balance:
                return None  # Can't afford

            orders = [
                {
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": self.config.trade_quantity,
                }
            ]

        elif action == 2:  # Sell
            # Check if we have position
            position = self._account.positions.get(symbol)
            if not position or position.quantity <= 0:
                return None  # Nothing to sell

            sell_qty = min(self.config.trade_quantity, position.quantity)
            orders = [
                {
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": sell_qty,
                }
            ]
        else:
            return None

        # Execute via Rust
        bars_dict = {
            symbol: {
                "open": current_bar.open,
                "high": current_bar.high,
                "low": current_bar.low,
                "close": current_bar.close,
                "volume": current_bar.volume,
            }
        }

        account_dict = self._account.model_dump()

        executions = execute_orders(
            orders,
            bars_dict,
            account_dict,
            current_bar.timestamp,
            self.config.commission_per_trade if self.config.commission_per_trade > 0 else None,
            self.config.slippage_pct if self.config.slippage_pct > 0 else None,
        )

        # Update account from modified dict
        self._account = Account.model_validate(account_dict)

        if executions:
            exec_dict = executions[0]
            execution = Execution(
                symbol=Symbol(exec_dict["symbol"]),
                side=exec_dict["side"],
                quantity=exec_dict["quantity"],
                price=exec_dict["price"],
                timestamp=exec_dict["timestamp"],
                order_id=f"rl-{self._step_count}",
                commission=exec_dict.get("commission", 0.0),
                slippage_pct=exec_dict.get("slippage_pct", 0.0),
            )
            self._executions.append(execution)
            return execution

        return None

    def _get_info(self) -> dict[str, Any]:
        """Get info dict for current state."""
        position = (
            self._account.positions.get(self.config.symbol)
            if self._account
            else None
        )

        return {
            "step": self._step_count,
            "equity": self._get_equity(),
            "cash": self._account.cleared_balance if self._account else 0,
            "position": position.quantity if position else 0,
            "total_reward": self._total_reward,
            "num_trades": len(self._executions),
            "current_price": self.bars[self._current_idx].close,
        }

    def render(self) -> str | None:
        """Render current state."""
        info = self._get_info()
        action_names = ["HOLD", "BUY", "SELL"]

        output = (
            f"Step {info['step']:4d} | "
            f"Action: {action_names[self._last_action]:4s} | "
            f"Price: ${info['current_price']:8.2f} | "
            f"Position: {info['position']:6.1f} | "
            f"Equity: ${info['equity']:10.2f} | "
            f"Reward: {self._last_reward:+8.4f}"
        )

        print(output)
        return output

    def get_executions(self) -> list[Execution]:
        """Get all executions from current episode."""
        return self._executions.copy()

    def get_episode_return(self) -> float:
        """Get total return for current episode."""
        initial = self.config.initial_balance
        current = self._get_equity()
        return (current - initial) / initial

