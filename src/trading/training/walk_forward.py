"""Walk-forward validation for strategy testing.

Walk-forward validation prevents overfitting by:
1. Training/optimizing on a window of historical data
2. Testing on the next (out-of-sample) period
3. Rolling the window forward and repeating
4. Aggregating metrics across all out-of-sample periods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

from trading.training.backtest import Backtest, BacktestResult
from trading.types import NormalizedBar, RunId, RunMetrics

if TYPE_CHECKING:
    from trading.strategies.base import Strategy


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation.

    :param train_period_days: Length of training window in days.
    :param test_period_days: Length of test window in days.
    :param step_days: Days to step forward between windows (defaults to test_period_days).
    :param min_train_bars: Minimum bars required in training window.
    :param initial_balance: Starting balance for each window.
    :param commission_per_trade: Commission per trade.
    :param slippage_pct: Slippage percentage.
    """

    train_period_days: int = 252  # ~1 year of trading days
    test_period_days: int = 63  # ~3 months
    step_days: int | None = None  # Default to test_period_days (non-overlapping)
    min_train_bars: int = 20
    initial_balance: float = 10000.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0


@dataclass
class WindowResult:
    """Result from a single train/test window.

    :param window_index: Index of this window (0-based).
    :param train_start: Start of training period.
    :param train_end: End of training period.
    :param test_start: Start of test period.
    :param test_end: End of test period.
    :param train_result: Backtest result from training period.
    :param test_result: Backtest result from test period.
    """

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation.

    :param windows: Results from each window.
    :param aggregate_metrics: Combined metrics across all test periods.
    :param total_test_return: Compounded return across all test periods.
    :param avg_test_sharpe: Average Sharpe ratio across test periods.
    :param consistency_ratio: Fraction of windows with positive test return.
    :param overfitting_ratio: Ratio of test performance to train performance.
    """

    windows: list[WindowResult] = field(default_factory=list)
    aggregate_metrics: RunMetrics | None = None
    total_test_return: float = 0.0
    avg_test_sharpe: float | None = None
    consistency_ratio: float = 0.0
    overfitting_ratio: float | None = None


StrategyFactory = Callable[[], "Strategy"]


class WalkForwardValidator:
    """Walk-forward validation engine.

    Walk-forward validation tests strategy robustness by:
    1. Splitting data into sequential train/test windows
    2. Running the strategy on each window independently
    3. Measuring out-of-sample (test) performance
    4. Aggregating results to assess overfitting

    Example::

        from trading.training import WalkForwardValidator, WalkForwardConfig
        from trading.strategies import MovingAverageCrossoverStrategy

        config = WalkForwardConfig(
            train_period_days=252,  # 1 year train
            test_period_days=63,    # 3 months test
        )

        def make_strategy():
            return MovingAverageCrossoverStrategy({"symbol": "AAPL"})

        validator = WalkForwardValidator(bars, config, make_strategy)
        result = validator.run()

        print(f"Total test return: {result.total_test_return:.2%}")
        print(f"Consistency: {result.consistency_ratio:.1%} of windows profitable")

    :param bars: Full dataset of bars for validation.
    :param config: Walk-forward configuration.
    :param strategy_factory: Callable that creates a fresh strategy instance.
    """

    def __init__(
        self,
        bars: list[NormalizedBar],
        config: WalkForwardConfig,
        strategy_factory: StrategyFactory,
    ) -> None:
        self.bars = sorted(bars, key=lambda b: b.timestamp)
        self.config = config
        self.strategy_factory = strategy_factory

        # Compute effective step
        self.step_days = config.step_days or config.test_period_days

    def _get_bars_in_range(
        self, start: datetime, end: datetime
    ) -> list[NormalizedBar]:
        """Get bars within a date range (inclusive of start, exclusive of end)."""
        return [b for b in self.bars if start <= b.timestamp < end]

    def _compute_windows(self) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Compute train/test window boundaries.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        if not self.bars:
            return []

        windows = []
        data_start = self.bars[0].timestamp
        data_end = self.bars[-1].timestamp

        train_days = timedelta(days=self.config.train_period_days)
        test_days = timedelta(days=self.config.test_period_days)
        step = timedelta(days=self.step_days)

        train_start = data_start

        while True:
            train_end = train_start + train_days
            test_start = train_end
            test_end = test_start + test_days

            # Stop if test period would extend beyond data
            if test_end > data_end + timedelta(days=1):
                break

            windows.append((train_start, train_end, test_start, test_end))
            train_start += step

        return windows

    def run(self) -> WalkForwardResult:
        """Run walk-forward validation.

        :returns: Aggregated results across all windows.
        """
        windows = self._compute_windows()

        if not windows:
            return WalkForwardResult()

        window_results: list[WindowResult] = []
        all_test_executions = []
        all_test_equity: list[tuple[datetime, float]] = []
        cumulative_return = 1.0

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Get bars for each period
            train_bars = self._get_bars_in_range(train_start, train_end)
            test_bars = self._get_bars_in_range(test_start, test_end)

            # Skip if insufficient data
            if len(train_bars) < self.config.min_train_bars:
                continue
            if not test_bars:
                continue

            # Create fresh strategy instances
            train_strategy = self.strategy_factory()
            test_strategy = self.strategy_factory()

            # Run training period backtest
            train_backtest = Backtest(
                bars=train_bars,
                strategy=train_strategy,
                initial_balance=self.config.initial_balance,
                commission_per_trade=self.config.commission_per_trade,
                slippage_pct=self.config.slippage_pct,
            )
            train_result = train_backtest.run()

            # Run test period backtest
            test_backtest = Backtest(
                bars=test_bars,
                strategy=test_strategy,
                initial_balance=self.config.initial_balance,
                commission_per_trade=self.config.commission_per_trade,
                slippage_pct=self.config.slippage_pct,
            )
            test_result = test_backtest.run()

            window_results.append(
                WindowResult(
                    window_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_result=train_result,
                    test_result=test_result,
                )
            )

            # Accumulate test results
            all_test_executions.extend(test_result.executions)
            all_test_equity.extend(test_result.equity_history)
            cumulative_return *= 1 + test_result.metrics.total_return

        if not window_results:
            return WalkForwardResult()

        # Compute aggregate metrics
        total_test_return = cumulative_return - 1

        # Average Sharpe across windows (excluding None values)
        sharpes = [
            w.test_result.metrics.sharpe_ratio
            for w in window_results
            if w.test_result.metrics.sharpe_ratio is not None
        ]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else None

        # Consistency: fraction of profitable windows
        profitable_windows = sum(
            1 for w in window_results if w.test_result.metrics.total_return > 0
        )
        consistency = profitable_windows / len(window_results)

        # Overfitting ratio: test performance / train performance
        # Lower values suggest overfitting
        avg_train_return = sum(
            w.train_result.metrics.total_return for w in window_results
        ) / len(window_results)
        avg_test_return = sum(
            w.test_result.metrics.total_return for w in window_results
        ) / len(window_results)

        if avg_train_return > 0:
            overfit_ratio = avg_test_return / avg_train_return
        else:
            overfit_ratio = None

        # Create aggregate metrics from test periods
        aggregate = RunMetrics(
            run_id=RunId("walk-forward-aggregate"),
            total_return=total_test_return,
            max_drawdown=max(
                w.test_result.metrics.max_drawdown for w in window_results
            ),
            volatility=sum(
                w.test_result.metrics.volatility for w in window_results
            ) / len(window_results) if window_results else 0.0,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=None,  # Would need to recompute from raw data
            num_trades=sum(
                w.test_result.metrics.num_trades for w in window_results
            ),
            win_rate=sum(
                (w.test_result.metrics.win_rate or 0) for w in window_results
            ) / len(window_results) if window_results else None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            expectancy=None,
        )

        return WalkForwardResult(
            windows=window_results,
            aggregate_metrics=aggregate,
            total_test_return=total_test_return,
            avg_test_sharpe=avg_sharpe,
            consistency_ratio=consistency,
            overfitting_ratio=overfit_ratio,
        )

    def summary(self) -> str:
        """Run validation and return formatted summary string."""
        result = self.run()

        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 60,
            f"Windows analyzed: {len(result.windows)}",
            f"Total test return: {result.total_test_return:+.2%}",
            f"Avg test Sharpe: {result.avg_test_sharpe:.2f}" if result.avg_test_sharpe else "Avg test Sharpe: N/A",
            f"Consistency: {result.consistency_ratio:.1%} of windows profitable",
            f"Overfitting ratio: {result.overfitting_ratio:.2f}" if result.overfitting_ratio else "Overfitting ratio: N/A",
            "",
            "Window Details:",
            "-" * 60,
        ]

        for w in result.windows:
            train_ret = w.train_result.metrics.total_return
            test_ret = w.test_result.metrics.total_return
            lines.append(
                f"  Window {w.window_index}: "
                f"Train {train_ret:+.2%} -> Test {test_ret:+.2%}"
            )

        return "\n".join(lines)

