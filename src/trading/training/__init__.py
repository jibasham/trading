"""Training and backtesting module."""

from trading.training.backtest import Backtest, BacktestResult
from trading.training.multi_backtest import (
    ComparisonResult,
    MultiBacktest,
    StrategyConfig,
    compare_strategies,
)

__all__ = [
    "Backtest",
    "BacktestResult",
    "ComparisonResult",
    "MultiBacktest",
    "StrategyConfig",
    "compare_strategies",
]
