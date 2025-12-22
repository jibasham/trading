"""Multi-strategy backtest runner for comparing strategies.

This module provides functionality to run multiple strategies on the same data
and compare their performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from trading.training.backtest import Backtest, BacktestResult
from trading.types import NormalizedBar

if TYPE_CHECKING:
    from trading.strategies.base import Strategy


@dataclass
class StrategyConfig:
    """Configuration for a strategy in a multi-backtest run.

    :param name: Display name for this strategy.
    :param strategy: Strategy instance to run.
    :param params: Optional parameters used to create the strategy.
    """

    name: str
    strategy: Strategy
    params: dict[str, Any] | None = None


class ComparisonResult(BaseModel):
    """Results from comparing multiple strategies.

    :param strategies: List of strategy names in order.
    :param results: Mapping of strategy name to BacktestResult.
    :param rankings: Strategies ranked by total return.
    :param best_strategy: Name of the best performing strategy.
    :param worst_strategy: Name of the worst performing strategy.
    """

    strategies: list[str]
    results: dict[str, BacktestResult]
    rankings: list[tuple[str, float]] = Field(default_factory=list)
    best_strategy: str | None = None
    worst_strategy: str | None = None


class MultiBacktest:
    """Run multiple strategies on the same data for comparison.

    Example usage::

        from trading.training import MultiBacktest
        from trading.training.multi_backtest import StrategyConfig
        from trading.strategies.examples import BuyAndHoldStrategy, MovingAverageCrossoverStrategy

        strategies = [
            StrategyConfig("Buy & Hold", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})),
            StrategyConfig("MA Cross", MovingAverageCrossoverStrategy({"symbol": "AAPL"})),
        ]

        multi = MultiBacktest(
            bars=bars,
            strategies=strategies,
            initial_balance=10000.0,
        )
        comparison = multi.run()

        print(f"Best strategy: {comparison.best_strategy}")

    :param bars: List of NormalizedBar or bar dicts to backtest against.
    :param strategies: List of StrategyConfig objects.
    :param initial_balance: Starting account balance for each strategy.
    :param max_position_size: Optional max position size in dollars.
    :param max_leverage: Max leverage allowed (default 1.0).
    :param parallel: If True, run strategies in parallel processes.
    """

    def __init__(
        self,
        bars: list[NormalizedBar] | list[dict[str, Any]],
        strategies: list[StrategyConfig],
        initial_balance: float = 10000.0,
        max_position_size: float | None = None,
        max_leverage: float = 1.0,
        parallel: bool = False,
    ) -> None:
        """Initialize multi-strategy backtest."""
        self.bars = bars
        self.strategies = strategies
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.parallel = parallel

    def run(self) -> ComparisonResult:
        """Run all strategies and compare results.

        :returns: ComparisonResult with all strategy results and rankings.
        """
        results: dict[str, BacktestResult] = {}
        strategy_names = [s.name for s in self.strategies]

        if self.parallel and len(self.strategies) > 1:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        # Rank strategies by total return
        rankings = sorted(
            [(name, results[name].metrics.total_return) for name in strategy_names],
            key=lambda x: x[1],
            reverse=True,
        )

        best = rankings[0][0] if rankings else None
        worst = rankings[-1][0] if rankings else None

        return ComparisonResult(
            strategies=strategy_names,
            results=results,
            rankings=rankings,
            best_strategy=best,
            worst_strategy=worst,
        )

    def _run_sequential(self) -> dict[str, BacktestResult]:
        """Run strategies sequentially."""
        results: dict[str, BacktestResult] = {}

        for config in self.strategies:
            backtest = Backtest(
                bars=self.bars,
                strategy=config.strategy,
                initial_balance=self.initial_balance,
                run_id=f"multi-{config.name.lower().replace(' ', '_')}",
                max_position_size=self.max_position_size,
                max_leverage=self.max_leverage,
            )
            results[config.name] = backtest.run()

        return results

    def _run_parallel(self) -> dict[str, BacktestResult]:
        """Run strategies in parallel using ProcessPoolExecutor.

        Note: This requires strategies to be picklable.
        For complex strategies, fall back to sequential execution.
        """
        # For now, use sequential since Strategy objects may not be picklable
        # TODO: Implement proper parallel execution with serializable strategy configs
        return self._run_sequential()

    def summary_table(self) -> str:
        """Generate a formatted summary table of results.

        :returns: Formatted string table comparing all strategies.
        """
        result = self.run()

        # Header
        lines = [
            "=" * 90,
            f"{'Strategy':<30} {'Return':>10} {'Drawdown':>10} {'Sharpe':>8} "
            f"{'Win Rate':>10} {'Trades':>8}",
            "-" * 90,
        ]

        # Data rows
        for name, _ in result.rankings:
            r = result.results[name]
            m = r.metrics
            sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio else "N/A"
            win_rate = f"{m.win_rate:.1%}" if m.win_rate else "N/A"
            lines.append(
                f"{name:<30} {m.total_return:>+9.2%} {m.max_drawdown:>9.2%} "
                f"{sharpe:>8} {win_rate:>10} {m.num_trades:>8}"
            )

        lines.append("=" * 90)

        if result.best_strategy:
            lines.append(f"🏆 Best: {result.best_strategy}")

        return "\n".join(lines)


def compare_strategies(
    bars: list[NormalizedBar] | list[dict[str, Any]],
    strategies: list[tuple[str, Strategy]],
    initial_balance: float = 10000.0,
    max_position_size: float | None = None,
    max_leverage: float = 1.0,
) -> ComparisonResult:
    """Convenience function to compare multiple strategies.

    :param bars: Historical bar data to backtest against.
    :param strategies: List of (name, strategy) tuples.
    :param initial_balance: Starting balance for each strategy.
    :param max_position_size: Optional max position size.
    :param max_leverage: Max leverage allowed.
    :returns: ComparisonResult with rankings and detailed results.
    """
    configs = [StrategyConfig(name=name, strategy=strat) for name, strat in strategies]
    multi = MultiBacktest(
        bars=bars,
        strategies=configs,
        initial_balance=initial_balance,
        max_position_size=max_position_size,
        max_leverage=max_leverage,
    )
    return multi.run()

