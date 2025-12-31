#!/usr/bin/env python3
"""
RL Agent Training for Stock Trading
====================================

This script trains a Reinforcement Learning agent to trade SPY using our
Gymnasium-compatible TradingEnv and Stable-Baselines3's PPO algorithm.

Goal: Beat Buy & Hold
---------------------
The challenge is to train an agent that can outperform simple buy-and-hold
by learning when to buy, sell, or hold based on price patterns.

Architecture
------------
We use Proximal Policy Optimization (PPO), a state-of-the-art policy gradient
algorithm that:
- Balances exploration vs exploitation
- Handles continuous action spaces (discretized here to Buy/Hold/Sell)
- Is stable and sample-efficient
- Works well with neural network function approximators

The agent observes:
- OHLCV features (normalized price/volume data)
- Technical indicators (optional)
- Account state (cash, position, equity)

And learns to output:
- Action 0: HOLD (do nothing)
- Action 1: BUY (add to position)
- Action 2: SELL (reduce position)

Training Strategy
-----------------
1. **Train/Test Split**: Use 80% of data for training, 20% for evaluation
2. **Reward Shaping**: Risk-adjusted returns penalizing drawdowns
3. **Episode Structure**: Each episode walks through training data
4. **Evaluation**: Test on unseen data to measure generalization

Expected Runtime
----------------
Training takes 5-15 minutes depending on:
- Number of timesteps (default: 50,000)
- Network size (default: [64, 64])
- Data length (~2,000 training bars)

Usage
-----
    python scripts/train_rl_agent.py

    # Or with custom parameters:
    python scripts/train_rl_agent.py --timesteps 100000 --symbol QQQ
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from trading.data.sources import YahooDataSource
from trading.rl import (
    TradingEnv,
    TradingEnvConfig,
    CombinedFeatures,
    OHLCVFeatures,
    AccountFeatures,
    SimpleReturnReward,
)
from trading.types import DateRange, NormalizedBar, Symbol


class TradingCallback(BaseCallback):
    """Custom callback for logging training progress."""

    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Log episode info when available
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

        # Periodic logging
        if self.n_calls % self.eval_freq == 0 and self.verbose:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                print(
                    f"Step {self.n_calls:,}: "
                    f"Mean reward (last 10): {mean_reward:.4f}, "
                    f"Mean length: {mean_length:.0f}"
                )

        return True


def convert_bars_to_normalized(bars: list) -> list[NormalizedBar]:
    """Convert fetched bars to NormalizedBar format."""
    normalized = []
    for bar in bars:
        if isinstance(bar, NormalizedBar):
            normalized.append(bar)
        else:
            # Convert Bar to NormalizedBar
            normalized.append(
                NormalizedBar(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            )
    return normalized


def fetch_data(
    symbol: str, years: int = 10, granularity: str = "1d"
) -> tuple[list[NormalizedBar], list[NormalizedBar]]:
    """Fetch and split data into train/test sets.

    :param symbol: Stock symbol to fetch.
    :param years: Years of data to fetch (ignored for intraday).
    :param granularity: Bar granularity ("1d", "1h", "5m").
    :returns: (train_bars, test_bars) tuple.
    """
    # Yahoo Finance limitations:
    # - 5m/15m/30m: Last 60 days only
    # - 1h: Last 730 days
    # - 1d: Full history
    
    if granularity in ["5m", "15m", "30m"]:
        days = 59  # Yahoo limit for intraday
        print(f"📊 Fetching {days} days of {granularity} {symbol} data...")
        print(f"   (Yahoo Finance limits intraday data to ~60 days)")
    elif granularity == "1h":
        days = min(years * 365, 729)  # Yahoo limit for hourly
        print(f"📊 Fetching {days} days of {granularity} {symbol} data...")
    else:
        days = years * 365
        print(f"📊 Fetching {years} years of {granularity} {symbol} data...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data_source = YahooDataSource()
    date_range = DateRange(start=start_date, end=end_date)

    bars = list(
        data_source.fetch_bars(
            symbols=[Symbol(symbol)],
            date_range=date_range,
            granularity=granularity,
        )
    )

    # Convert to NormalizedBar
    bars = convert_bars_to_normalized(bars)

    bars_per_day = {"5m": 78, "15m": 26, "30m": 13, "1h": 7, "1d": 1}.get(granularity, 1)
    print(f"   Fetched {len(bars):,} bars ({granularity})")
    print(f"   ~{bars_per_day} bars per trading day")
    if bars:
        print(f"   Date range: {bars[0].timestamp} to {bars[-1].timestamp}")

    # Split 80/20 train/test
    split_idx = int(len(bars) * 0.8)
    train_bars = bars[:split_idx]
    test_bars = bars[split_idx:]

    print(f"   Train set: {len(train_bars):,} bars")
    print(f"   Test set:  {len(test_bars):,} bars")

    return train_bars, test_bars


def create_env(
    bars: list[NormalizedBar],
    symbol: str,
    initial_balance: float = 100_000.0,
    trade_quantity: float = 100.0,
    random_start: bool = False,
) -> TradingEnv:
    """Create a trading environment.

    :param bars: Bar data for the environment.
    :param symbol: Symbol being traded.
    :param initial_balance: Starting cash.
    :param trade_quantity: Shares per trade.
    :param random_start: Whether to start from random position.
    :returns: TradingEnv instance.
    """
    config = TradingEnvConfig(
        symbol=symbol,
        initial_balance=initial_balance,
        trade_quantity=trade_quantity,
        commission_per_trade=1.0,  # $1 per trade
        slippage_pct=0.001,  # 0.1% slippage
        random_start=random_start,
    )

    # Use combined features for richer observations
    # For intraday, use shorter lookback (fewer historical bars needed)
    lookback = 10 if any(g in str(bars[0].timestamp) for g in [":", "T"]) else 20
    feature_extractor = CombinedFeatures(
        extractors=[
            OHLCVFeatures(lookback=lookback),
            AccountFeatures(symbol=symbol),
        ]
    )

    # Use simple return reward - directly rewards equity changes
    # This encourages the agent to take positions when profitable
    reward_function = SimpleReturnReward()

    return TradingEnv(
        bars=bars,
        config=config,
        feature_extractor=feature_extractor,
        reward_function=reward_function,
    )


def evaluate_agent(
    model: PPO,
    test_bars: list[NormalizedBar],
    symbol: str,
    initial_balance: float = 100_000.0,
    trade_quantity: float = 100.0,
    n_episodes: int = 1,
    verbose: bool = True,
) -> dict:
    """Evaluate trained agent on test data.

    :param model: Trained PPO model.
    :param test_bars: Test set bars.
    :param symbol: Symbol being traded.
    :param initial_balance: Starting cash.
    :param trade_quantity: Shares per trade.
    :param n_episodes: Number of evaluation episodes.
    :param verbose: Print progress.
    :returns: Evaluation metrics dict.
    """
    env = create_env(
        test_bars,
        symbol,
        initial_balance,
        trade_quantity,
        random_start=False,
    )

    total_returns = []
    total_trades = []
    final_equities = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_trades = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if verbose and env._step_count % 100 == 0:
                env.render()

        # Episode complete
        episode_return = env.get_episode_return()
        total_returns.append(episode_return)
        total_trades.append(info["num_trades"])
        final_equities.append(info["equity"])

        if verbose:
            print(f"\nEpisode {ep + 1} complete:")
            print(f"  Return: {episode_return:+.2%}")
            print(f"  Trades: {info['num_trades']}")
            print(f"  Final equity: ${info['equity']:,.2f}")

    return {
        "mean_return": np.mean(total_returns),
        "std_return": np.std(total_returns),
        "mean_trades": np.mean(total_trades),
        "final_equities": final_equities,
    }


def calculate_buy_hold_return(bars: list[NormalizedBar]) -> float:
    """Calculate buy and hold return for the given bars."""
    if len(bars) < 2:
        return 0.0
    return (bars[-1].close - bars[0].close) / bars[0].close


def main() -> None:
    """Train and evaluate RL trading agent."""
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument(
        "--symbol", type=str, default="SPY", help="Symbol to trade (default: SPY)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps (default: 50000)",
    )
    parser.add_argument(
        "--years", type=int, default=10, help="Years of data (default: 10, ignored for intraday)"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="1d",
        choices=["5m", "15m", "30m", "1h", "1d"],
        help="Bar granularity (default: 1d)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=100_000.0,
        help="Initial balance (default: 100000)",
    )
    parser.add_argument(
        "--quantity",
        type=float,
        default=250.0,  # ~$87k at $350/share, allows near-full investment
        help="Shares per trade (default: 250)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RL TRADING AGENT TRAINING")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Fetch and prepare data
    # -------------------------------------------------------------------------
    train_bars, test_bars = fetch_data(args.symbol, args.years, args.granularity)
    print()

    # Calculate buy & hold baseline for comparison
    train_bh_return = calculate_buy_hold_return(train_bars)
    test_bh_return = calculate_buy_hold_return(test_bars)

    print(f"📈 Buy & Hold Returns:")
    print(f"   Train period: {train_bh_return:+.2%}")
    print(f"   Test period:  {test_bh_return:+.2%}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Create training environment
    # -------------------------------------------------------------------------
    print("🏗️  Creating training environment...")

    def make_train_env():
        return create_env(
            train_bars,
            args.symbol,
            args.balance,
            args.quantity,
            random_start=True,  # Random starts for better exploration
        )

    # Wrap in DummyVecEnv for SB3 compatibility
    train_env = DummyVecEnv([make_train_env])

    print(f"   Observation space: {train_env.observation_space.shape}")
    print(f"   Action space: {train_env.action_space}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Create and train PPO agent
    # -------------------------------------------------------------------------
    print("🤖 Initializing PPO agent...")

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,  # Higher LR for faster learning
        n_steps=512,  # Shorter rollouts for more updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Standard discount
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher entropy = more exploration
        seed=args.seed,  # For reproducibility
        policy_kwargs={
            "net_arch": [64, 64],  # Smaller network for faster training
        },
        tensorboard_log=None,
    )

    print(f"   Policy: MLP with layers [64, 64]")
    print(f"   Learning rate: 3e-4")
    print(f"   Entropy coef: 0.05 (high exploration)")
    print(f"   Total timesteps: {args.timesteps:,}")
    print()

    print("🚀 Training started...")
    print("-" * 70)

    callback = TradingCallback(eval_freq=5000, verbose=1)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=False,  # Disabled to avoid tqdm/rich dependency
    )

    print("-" * 70)
    print("✅ Training complete!")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Evaluate on test data
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("📊 EVALUATION ON TEST DATA (Out-of-Sample)")
    print("=" * 70)
    print()

    eval_results = evaluate_agent(
        model,
        test_bars,
        args.symbol,
        args.balance,
        args.quantity,
        n_episodes=1,
        verbose=True,
    )

    print()

    # -------------------------------------------------------------------------
    # Step 5: Compare to benchmarks
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("📈 PERFORMANCE COMPARISON (Test Period)")
    print("=" * 70)

    agent_return = eval_results["mean_return"]
    agent_final = eval_results["final_equities"][0]

    # Simulate buy & hold final value
    bh_final = args.balance * (1 + test_bh_return)

    print(f"\n{'Strategy':<25} {'Final Value':>14} {'Return':>12}")
    print("-" * 55)
    print(f"{'RL Agent (PPO)':<25} ${agent_final:>12,.0f} {agent_return:>+11.2%}")
    print(f"{'Buy & Hold':<25} ${bh_final:>12,.0f} {test_bh_return:>+11.2%}")
    print("-" * 55)

    alpha = agent_return - test_bh_return
    if alpha > 0:
        print(f"\n🏆 RL Agent OUTPERFORMED Buy & Hold by {alpha:+.2%}!")
    else:
        print(f"\n📉 RL Agent underperformed Buy & Hold by {alpha:.2%}")

    # Trading activity
    print(f"\n📋 Trading Activity:")
    print(f"   Number of trades: {eval_results['mean_trades']:.0f}")

    # -------------------------------------------------------------------------
    # Step 6: Save model if requested
    # -------------------------------------------------------------------------
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"\n💾 Model saved to: {save_path}")

    # Default save location
    default_path = Path("models") / f"ppo_{args.symbol.lower()}_trading"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(default_path))
    print(f"💾 Model saved to: {default_path}")

    print()
    print("=" * 70)
    print("✅ RL Agent training and evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

