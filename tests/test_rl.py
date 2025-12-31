"""Tests for RL components."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.rl.env import TradingEnv, TradingEnvConfig
from trading.rl.features import (
    AccountFeatures,
    CombinedFeatures,
    OHLCVFeatures,
    TechnicalFeatures,
)
from trading.rl.rewards import (
    DrawdownPenaltyReward,
    SimpleReturnReward,
)
from trading.rl.wrappers import RandomPolicy, RLStrategy, SimplePolicy
from trading.types import Account, NormalizedBar, Symbol


def generate_bars(
    symbol: str,
    start_date: datetime,
    num_days: int,
    start_price: float = 100.0,
    trend: float = 0.001,
) -> list[NormalizedBar]:
    """Generate synthetic bars."""
    bars = []
    price = start_price

    for i in range(num_days):
        price *= 1 + trend
        bars.append(
            NormalizedBar(
                symbol=Symbol(symbol),
                timestamp=start_date + timedelta(days=i),
                open=price * 0.99,
                high=price * 1.02,
                low=price * 0.98,
                close=price,
                volume=10000.0,
            )
        )

    return bars


class TestOHLCVFeatures:
    """Tests for OHLCVFeatures extractor."""

    def test_feature_count(self) -> None:
        """Feature count matches lookback * 5."""
        extractor = OHLCVFeatures(lookback=10)
        assert extractor.num_features == 50

    def test_extract_returns_correct_shape(self) -> None:
        """Extracted features have correct shape."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = generate_bars("TEST", start, 50)
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )

        extractor = OHLCVFeatures(lookback=10)
        features = extractor.extract(bars, 25, account)

        assert features.shape == (50,)
        assert features.dtype == np.float32

    def test_features_normalized(self) -> None:
        """Features are normalized relative to current price."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = generate_bars("TEST", start, 50)
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )

        extractor = OHLCVFeatures(lookback=5)
        features = extractor.extract(bars, 25, account)

        # Last close feature should be 0 (current price / current price - 1)
        last_close_idx = 4 * 5 + 3  # Last bar, close field
        assert features[last_close_idx] == pytest.approx(0.0, abs=0.001)


class TestTechnicalFeatures:
    """Tests for TechnicalFeatures extractor."""

    def test_feature_count(self) -> None:
        """Returns correct number of features."""
        extractor = TechnicalFeatures()
        assert extractor.num_features == 5

    def test_extract_returns_correct_shape(self) -> None:
        """Features have correct shape."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = generate_bars("TEST", start, 100)
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )

        extractor = TechnicalFeatures()
        features = extractor.extract(bars, 50, account)

        assert features.shape == (5,)


class TestAccountFeatures:
    """Tests for AccountFeatures extractor."""

    def test_no_position(self) -> None:
        """Returns correct features with no position."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = generate_bars("TEST", start, 50)
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )

        extractor = AccountFeatures(symbol="TEST")
        features = extractor.extract(bars, 25, account)

        # position_ratio=0, cash_ratio=1, unrealized_pnl=0
        assert features[0] == pytest.approx(0.0)  # No position
        assert features[1] == pytest.approx(1.0)  # All cash
        assert features[2] == pytest.approx(0.0)  # No P&L


class TestCombinedFeatures:
    """Tests for CombinedFeatures extractor."""

    def test_combines_extractors(self) -> None:
        """Combines multiple extractors."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = generate_bars("TEST", start, 100)
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )

        ohlcv = OHLCVFeatures(lookback=5)
        tech = TechnicalFeatures()
        combined = CombinedFeatures([ohlcv, tech])

        assert combined.num_features == 25 + 5

        features = combined.extract(bars, 50, account)
        assert features.shape == (30,)


class TestSimpleReturnReward:
    """Tests for SimpleReturnReward."""

    def test_positive_return(self) -> None:
        """Positive equity change gives positive reward."""
        reward_fn = SimpleReturnReward(scale=100)
        reward = reward_fn.compute(
            prev_equity=10000,
            curr_equity=10100,
            action=0,
            execution=None,
            account=None,  # type: ignore
            step=1,
            done=False,
        )
        assert reward == pytest.approx(1.0)  # 1% * 100

    def test_negative_return(self) -> None:
        """Negative equity change gives negative reward."""
        reward_fn = SimpleReturnReward(scale=100)
        reward = reward_fn.compute(
            prev_equity=10000,
            curr_equity=9900,
            action=0,
            execution=None,
            account=None,  # type: ignore
            step=1,
            done=False,
        )
        assert reward == pytest.approx(-1.0)


class TestDrawdownPenaltyReward:
    """Tests for DrawdownPenaltyReward."""

    def test_penalty_on_drawdown(self) -> None:
        """Penalizes drawdowns from peak."""
        reward_fn = DrawdownPenaltyReward()
        reward_fn.reset()

        # First step: peak = 10000
        _ = reward_fn.compute(10000, 10000, 0, None, None, 1, False)  # type: ignore

        # Second step: new peak = 10100
        _ = reward_fn.compute(10000, 10100, 0, None, None, 2, False)  # type: ignore

        # Third step: drawdown to 10000
        reward = reward_fn.compute(10100, 10000, 0, None, None, 3, False)  # type: ignore

        # Should have negative component from drawdown
        assert reward < 0


class TestTradingEnv:
    """Tests for TradingEnv."""

    @pytest.fixture
    def bars(self) -> list[NormalizedBar]:
        """Generate test bars."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 100, trend=0.001)

    def test_reset_returns_observation(self, bars: list[NormalizedBar]) -> None:
        """Reset returns initial observation."""
        config = TradingEnvConfig(symbol="TEST")
        env = TradingEnv(bars, config)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert "equity" in info
        assert info["equity"] == config.initial_balance

    def test_action_space(self, bars: list[NormalizedBar]) -> None:
        """Action space is Discrete(3)."""
        env = TradingEnv(bars)
        assert env.action_space.n == 3

    def test_observation_space(self, bars: list[NormalizedBar]) -> None:
        """Observation space matches feature extractor."""
        extractor = OHLCVFeatures(lookback=10)
        env = TradingEnv(bars, feature_extractor=extractor)

        assert env.observation_space.shape == (50,)

    def test_step_returns_tuple(self, bars: list[NormalizedBar]) -> None:
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = TradingEnv(bars)
        env.reset()

        result = env.step(0)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_buy_action_creates_position(self, bars: list[NormalizedBar]) -> None:
        """Buy action creates a position."""
        config = TradingEnvConfig(symbol="TEST", trade_quantity=10)
        env = TradingEnv(bars, config)
        env.reset()

        # Take buy action
        _, _, _, _, info = env.step(1)

        assert info["position"] == 10

    def test_sell_without_position(self, bars: list[NormalizedBar]) -> None:
        """Sell without position does nothing."""
        config = TradingEnvConfig(symbol="TEST")
        env = TradingEnv(bars, config)
        env.reset()

        # Try to sell without position
        _, _, _, _, info = env.step(2)

        assert info["position"] == 0
        assert info["num_trades"] == 0

    def test_episode_terminates_at_end(self, bars: list[NormalizedBar]) -> None:
        """Episode terminates when data ends."""
        config = TradingEnvConfig(symbol="TEST")
        env = TradingEnv(bars, config)
        env.reset()

        # Step through all data
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if steps > 200:  # Safety limit
                break

        assert terminated

    def test_max_steps_truncates(self, bars: list[NormalizedBar]) -> None:
        """Episode truncates after max_steps."""
        config = TradingEnvConfig(symbol="TEST", max_steps=10)
        env = TradingEnv(bars, config)
        env.reset()

        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(0)

        assert truncated

    def test_render_returns_string(self, bars: list[NormalizedBar]) -> None:
        """Render returns formatted string."""
        env = TradingEnv(bars)
        env.reset()
        env.step(0)

        output = env.render()
        assert isinstance(output, str)
        assert "HOLD" in output


class TestRLStrategy:
    """Tests for RLStrategy wrapper."""

    @pytest.fixture
    def bars(self) -> list[NormalizedBar]:
        """Generate test bars."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 100)

    def test_random_policy_generates_actions(self, bars: list[NormalizedBar]) -> None:
        """Random policy generates valid actions."""
        policy = RandomPolicy(seed=42, buy_prob=0.3, sell_prob=0.3)

        obs = np.zeros(50, dtype=np.float32)
        action, _ = policy.predict(obs)

        assert action in [0, 1, 2]

    def test_simple_policy(self, bars: list[NormalizedBar]) -> None:
        """Simple policy follows rule function."""

        def always_buy(obs: np.ndarray) -> int:
            return 1

        policy = SimplePolicy(always_buy)
        action, _ = policy.predict(np.zeros(10, dtype=np.float32))

        assert action == 1

    def test_rl_strategy_with_random_policy(self, bars: list[NormalizedBar]) -> None:
        """RLStrategy works with random policy."""
        policy = RandomPolicy(seed=42)
        strategy = RLStrategy(
            model=policy,
            symbol="TEST",
            trade_quantity=5,
        )

        from trading.training.backtest import Backtest

        backtest = Backtest(bars, strategy)
        result = backtest.run()

        # Should complete without error
        assert result.metrics is not None


class TestGymCompatibility:
    """Tests for Gymnasium compatibility."""

    @pytest.fixture
    def bars(self) -> list[NormalizedBar]:
        """Generate test bars."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 100)

    def test_env_check(self, bars: list[NormalizedBar]) -> None:
        """Environment passes Gymnasium check."""
        from gymnasium.utils.env_checker import check_env

        env = TradingEnv(bars)

        # This will raise if env doesn't follow Gym API
        check_env(env, skip_render_check=True)

    def test_reproducibility_with_seed(self, bars: list[NormalizedBar]) -> None:
        """Same seed produces same results."""
        config = TradingEnvConfig(symbol="TEST", random_start=True)

        env1 = TradingEnv(bars, config)
        obs1, _ = env1.reset(seed=42)

        env2 = TradingEnv(bars, config)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

