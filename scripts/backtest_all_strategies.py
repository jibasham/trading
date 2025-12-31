#!/usr/bin/env python3
"""
Multi-Strategy 10-Year Backtest Comparison
===========================================

This script runs a comprehensive backtest comparison of ALL available trading
strategies over a 10-year period. It demonstrates the full power of the trading
system by comparing multiple strategies against each other and market benchmarks.

What This Script Does
---------------------
1. **Data Fetching**: Downloads 10 years of daily data for SPY and benchmark indices
2. **Strategy Execution**: Runs all 5 built-in strategies:
   - Buy and Hold (baseline passive strategy)
   - Moving Average Crossover (trend following)
   - Mean Reversion (contrarian/reversion to mean)
   - RSI (momentum/overbought-oversold)
   - Random (control/baseline for comparison)
3. **Performance Analysis**: Compares all strategies on:
   - Total return and CAGR (Compound Annual Growth Rate)
   - Max drawdown and volatility
   - Sharpe ratio and Sortino ratio
   - Win rate and number of trades
4. **Benchmark Comparison**: Shows performance vs S&P 500, Dow Jones, NASDAQ

Why Compare Multiple Strategies
-------------------------------
Different strategies perform better in different market conditions:
- **Buy & Hold**: Best in strong bull markets, worst in bear markets
- **MA Crossover**: Good in trending markets, whipsaws in sideways markets
- **Mean Reversion**: Good in ranging markets, dangerous in trending markets
- **RSI**: Captures momentum shifts, can miss extended trends
- **Random**: Baseline to ensure strategies beat chance

A 10-year period captures multiple market regimes:
- Bull markets (2014-2019, 2020-2021, 2023-2024)
- Bear markets/corrections (2018 Q4, 2020 COVID crash, 2022)
- Sideways consolidation periods

Expected Runtime
----------------
This script takes 30-60 seconds to complete:
- ~5-10 seconds to fetch 10 years of daily data (~2,500 bars)
- ~5-10 seconds per strategy backtest (5 strategies = 25-50 seconds)
- Instant metrics computation and display

The longer runtime is justified by the comprehensive analysis produced.

Usage
-----
    python scripts/backtest_all_strategies.py

    # Or from project root:
    cd /Users/Shared/git/trading
    source venv/bin/activate
    python scripts/backtest_all_strategies.py
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from trading.data.sources import YahooDataSource
from trading.strategies.examples import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    RandomStrategy,
)
from trading.training.backtest import Backtest, BacktestResult
from trading.types import DateRange, Symbol


@dataclass
class StrategyConfig:
    """Configuration for a strategy to backtest."""

    name: str
    strategy_class: type
    params: dict


def calculate_cagr(total_return: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate.

    :param total_return: Total return as decimal (e.g., 0.50 for 50%).
    :param years: Number of years.
    :return: CAGR as decimal.
    """
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def main() -> None:
    """Run the multi-strategy 10-year backtest comparison."""
    print("=" * 70)
    print("MULTI-STRATEGY 10-YEAR BACKTEST COMPARISON")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Define backtest parameters
    # -------------------------------------------------------------------------
    # 10-year period ending recently
    end_date = datetime(2024, 12, 1)
    start_date = end_date - timedelta(days=365 * 10)  # ~10 years
    years = 10.0

    symbol = Symbol("SPY")
    starting_balance = 100_000.0

    print(f"📅 Date Range: {start_date.date()} to {end_date.date()} ({years:.0f} years)")
    print(f"📈 Symbol: {symbol}")
    print(f"💰 Starting Balance: ${starting_balance:,.2f}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Define all strategies to test
    # -------------------------------------------------------------------------
    strategies: list[StrategyConfig] = [
        StrategyConfig(
            name="Buy & Hold",
            strategy_class=BuyAndHoldStrategy,
            params={"symbol": str(symbol), "quantity": 100},
        ),
        StrategyConfig(
            name="MA Crossover (10/30)",
            strategy_class=MovingAverageCrossoverStrategy,
            params={
                "symbol": str(symbol),
                "short_period": 10,
                "long_period": 30,
                "quantity": 100,
            },
        ),
        StrategyConfig(
            name="MA Crossover (20/50)",
            strategy_class=MovingAverageCrossoverStrategy,
            params={
                "symbol": str(symbol),
                "short_period": 20,
                "long_period": 50,
                "quantity": 100,
            },
        ),
        StrategyConfig(
            name="Mean Reversion",
            strategy_class=MeanReversionStrategy,
            params={
                "symbol": str(symbol),
                "lookback_period": 20,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "quantity": 100,
            },
        ),
        StrategyConfig(
            name="RSI Strategy",
            strategy_class=RSIStrategy,
            params={
                "symbol": str(symbol),
                "rsi_period": 14,
                "oversold": 30,
                "overbought": 70,
                "quantity": 100,
            },
        ),
        StrategyConfig(
            name="Random (Control)",
            strategy_class=RandomStrategy,
            params={
                "symbol": str(symbol),
                "trade_probability": 0.02,  # 2% chance per day
                "quantity": 100,
                "seed": 42,  # Fixed seed for reproducibility
            },
        ),
    ]

    print(f"📊 Strategies to test: {len(strategies)}")
    for cfg in strategies:
        print(f"   • {cfg.name}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Fetch historical data
    # -------------------------------------------------------------------------
    print("🔄 Fetching 10 years of historical data from Yahoo Finance...")

    data_source = YahooDataSource()
    date_range = DateRange(start=start_date, end=end_date)

    bars = list(
        data_source.fetch_bars(
            symbols=[symbol],
            date_range=date_range,
            granularity="1d",
        )
    )

    print(f"✅ Fetched {len(bars):,} daily bars")
    if bars:
        print(f"   First bar: {bars[0].timestamp.date()} @ ${bars[0].close:.2f}")
        print(f"   Last bar:  {bars[-1].timestamp.date()} @ ${bars[-1].close:.2f}")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Fetch benchmark data
    # -------------------------------------------------------------------------
    print("🔄 Fetching benchmark index data...")

    benchmarks = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
    }

    benchmark_returns: dict[str, float | None] = {}

    # Traded symbol buy & hold return
    if bars:
        symbol_start = bars[0].close
        symbol_end = bars[-1].close
        benchmark_returns[str(symbol)] = (symbol_end - symbol_start) / symbol_start

    # Fetch index benchmarks
    for ticker, name in benchmarks.items():
        try:
            bench_bars = list(
                data_source.fetch_bars(
                    symbols=[Symbol(ticker)],
                    date_range=date_range,
                    granularity="1d",
                )
            )
            if bench_bars:
                bench_start = bench_bars[0].close
                bench_end = bench_bars[-1].close
                benchmark_returns[ticker] = (bench_end - bench_start) / bench_start
        except Exception:
            benchmark_returns[ticker] = None

    print("✅ Benchmark data fetched")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Run backtests for all strategies
    # -------------------------------------------------------------------------
    print("🚀 Running backtests...")
    print("-" * 70)

    results: list[tuple[StrategyConfig, BacktestResult]] = []

    for i, cfg in enumerate(strategies, 1):
        print(f"   [{i}/{len(strategies)}] Running {cfg.name}...", end=" ", flush=True)

        strategy = cfg.strategy_class(params=cfg.params)

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=starting_balance,
            commission_per_trade=1.0,
            slippage_pct=0.001,
        )

        result = backtest.run()
        results.append((cfg, result))

        # Calculate final equity
        if result.equity_history:
            final_eq = result.equity_history[-1][1]
        else:
            final_eq = result.final_account.cleared_balance

        ret = (final_eq - starting_balance) / starting_balance
        print(f"Done! Return: {ret:+.2%}")

    print("-" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 6: Display Results Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("📊 STRATEGY PERFORMANCE COMPARISON (10-Year Period)")
    print("=" * 70)

    # Prepare data for display
    strategy_data: list[dict] = []
    for cfg, result in results:
        if result.equity_history:
            final_equity = result.equity_history[-1][1]
        else:
            final_equity = result.final_account.cleared_balance

        total_return = (final_equity - starting_balance) / starting_balance
        cagr = calculate_cagr(total_return, years)

        strategy_data.append(
            {
                "name": cfg.name,
                "final_equity": final_equity,
                "total_return": total_return,
                "cagr": cagr,
                "max_drawdown": result.metrics.max_drawdown,
                "sharpe": result.metrics.sharpe_ratio,
                "sortino": result.metrics.sortino_ratio,
                "volatility": result.metrics.volatility,
                "num_trades": result.metrics.num_trades,
                "win_rate": result.metrics.win_rate,
            }
        )

    # Sort by total return (best first)
    strategy_data.sort(key=lambda x: x["total_return"], reverse=True)

    # Print comparison table
    print(f"\n{'Rank':<5} {'Strategy':<22} {'Final Value':>14} {'Return':>10} {'CAGR':>8} {'MaxDD':>8}")
    print("-" * 70)

    for rank, data in enumerate(strategy_data, 1):
        print(
            f"{rank:<5} "
            f"{data['name']:<22} "
            f"${data['final_equity']:>12,.0f} "
            f"{data['total_return']:>+9.2%} "
            f"{data['cagr']:>+7.2%} "
            f"{data['max_drawdown']:>7.2%}"
        )

    print("-" * 70)

    # Risk-adjusted metrics table
    print(f"\n{'Strategy':<22} {'Sharpe':>8} {'Sortino':>8} {'Vol':>8} {'Trades':>8} {'Win%':>8}")
    print("-" * 70)

    for data in strategy_data:
        sharpe_str = f"{data['sharpe']:.2f}" if data["sharpe"] else "N/A"
        sortino_str = f"{data['sortino']:.2f}" if data["sortino"] else "N/A"
        vol_str = f"{data['volatility']:.4f}" if data["volatility"] else "N/A"
        win_str = f"{data['win_rate']:.1%}" if data["win_rate"] is not None else "N/A"

        print(
            f"{data['name']:<22} "
            f"{sharpe_str:>8} "
            f"{sortino_str:>8} "
            f"{vol_str:>8} "
            f"{data['num_trades']:>8} "
            f"{win_str:>8}"
        )

    print("-" * 70)

    # -------------------------------------------------------------------------
    # Step 7: Benchmark Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📈 BENCHMARK COMPARISON")
    print("=" * 70)

    # Get the best strategy
    best_strategy = strategy_data[0]

    print(f"\n{'Investment':<25} {'10-Year Return':>15} {'CAGR':>10}")
    print("-" * 50)

    # Best strategy
    print(
        f"{'🏆 ' + best_strategy['name']:<25} "
        f"{best_strategy['total_return']:>+14.2%} "
        f"{best_strategy['cagr']:>+9.2%}"
    )

    print("-" * 50)

    # Benchmarks
    for ticker, name in [(str(symbol), f"{symbol} (Buy & Hold)")] + list(benchmarks.items()):
        ret = benchmark_returns.get(ticker if ticker != str(symbol) else str(symbol))
        if ret is not None:
            cagr = calculate_cagr(ret, years)
            display_name = name if ticker in benchmarks else f"{symbol} (Buy & Hold)"
            print(f"{display_name:<25} {ret:>+14.2%} {cagr:>+9.2%}")
        else:
            print(f"{name:<25} {'N/A':>15} {'N/A':>10}")

    print("-" * 50)

    # Alpha calculation
    sp500_return = benchmark_returns.get("^GSPC")
    if sp500_return is not None:
        alpha = best_strategy["total_return"] - sp500_return
        alpha_label = "OUTPERFORMED ✅" if alpha > 0 else "underperformed ❌"
        print(f"\n📊 Best Strategy Alpha vs S&P 500: {alpha:>+.2%} ({alpha_label})")

    # -------------------------------------------------------------------------
    # Step 8: Summary Insights
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)

    # Find strategy with best Sharpe ratio
    best_sharpe = max(
        (d for d in strategy_data if d["sharpe"] is not None),
        key=lambda x: x["sharpe"],
        default=None,
    )

    # Find strategy with lowest max drawdown
    lowest_dd = min(strategy_data, key=lambda x: abs(x["max_drawdown"]))

    # Find strategy with most trades
    most_active = max(strategy_data, key=lambda x: x["num_trades"])

    print(f"\n📈 Highest Return:        {strategy_data[0]['name']} ({strategy_data[0]['total_return']:+.2%})")
    if best_sharpe:
        print(f"📊 Best Risk-Adjusted:    {best_sharpe['name']} (Sharpe: {best_sharpe['sharpe']:.2f})")
    print(f"🛡️  Lowest Drawdown:       {lowest_dd['name']} ({lowest_dd['max_drawdown']:.2%})")
    print(f"🔄 Most Active:           {most_active['name']} ({most_active['num_trades']} trades)")

    # Did any strategy beat buy & hold?
    buy_hold_return = benchmark_returns.get(str(symbol), 0) or 0
    beaters = [d for d in strategy_data if d["total_return"] > buy_hold_return]

    if beaters:
        print(f"\n✅ Strategies that beat Buy & Hold: {len(beaters)}")
        for d in beaters:
            excess = d["total_return"] - buy_hold_return
            print(f"   • {d['name']}: +{excess:.2%} excess return")
    else:
        print("\n❌ No strategy beat simple Buy & Hold over this period")
        print("   This is common in strong bull markets - passive beats active!")

    print()
    print("=" * 70)
    print("✅ Multi-strategy backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

