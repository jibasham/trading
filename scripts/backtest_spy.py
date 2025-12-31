#!/usr/bin/env python3
"""
SPY Moving Average Crossover Backtest Demo
==========================================

This script demonstrates the complete backtesting workflow using our hybrid
Rust + Python trading system. It fetches real historical data from Yahoo Finance,
runs a Moving Average Crossover strategy, and displays comprehensive performance metrics.

What This Script Does
---------------------
1. **Data Fetching**: Downloads 6 months of daily SPY (S&P 500 ETF) data from Yahoo Finance
2. **Strategy Execution**: Runs a Moving Average Crossover strategy that:
   - Goes LONG when the 10-day SMA crosses above the 30-day SMA (bullish signal)
   - Goes SHORT (sells position) when the 10-day SMA crosses below the 30-day SMA
3. **Performance Analysis**: Computes and displays trading metrics including:
   - Total return, max drawdown, volatility
   - Sharpe ratio, Sortino ratio
   - Win rate, profit factor, expectancy

Why This Approach Works
-----------------------
The Moving Average Crossover is a classic trend-following strategy that:
- Filters out market noise by smoothing price data
- Captures momentum shifts when short-term trend crosses long-term trend
- Works well in trending markets (though may whipsaw in ranging markets)

We use a 10/30 day combination which is moderately responsive - not too fast
(which would generate excessive trades) and not too slow (which would miss moves).

How the System Architecture Works
---------------------------------
This demo showcases the hybrid Rust + Python architecture:

**Python Layer** (flexibility):
- `YahooDataSource`: Fetches data via yfinance library
- `MovingAverageCrossoverStrategy`: Implements trading logic with pandas
- `Backtest`: Orchestrates the simulation loop

**Rust Layer** (performance):
- `execute_orders()`: Executes trades with commission/slippage modeling
- `compute_run_metrics()`: Calculates performance statistics efficiently
- Data normalization and validation

The Backtest class iterates through each bar chronologically, calling:
1. `strategy.decide()` - Python strategy generates OrderRequests
2. `execute_orders()` - Rust executes orders at bar close price
3. Account state is updated after each trade

Running Time
------------
This script should complete in 5-15 seconds:
- ~1-2 seconds for Yahoo Finance data fetch
- ~1-2 seconds for strategy initialization
- ~1-2 seconds for backtest execution (6 months of daily bars = ~126 bars)
- Instant metrics computation

Usage
-----
    python scripts/backtest_spy.py

    # Or from project root:
    cd /Users/Shared/git/trading
    source venv/bin/activate
    python scripts/backtest_spy.py

Expected Output
---------------
The script prints:
- Data fetch confirmation with bar count
- Trade-by-trade execution log
- Final performance metrics table
- Equity curve summary (start -> end values)

Author: Trading Bot Development Team
"""

from datetime import datetime, timedelta

from trading.data.sources import YahooDataSource
from trading.strategies.examples import MovingAverageCrossoverStrategy
from trading.training.backtest import Backtest
from trading.types import DateRange, Symbol


def main() -> None:
    """Run the SPY backtest demonstration."""
    print("=" * 60)
    print("SPY Moving Average Crossover Backtest")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Define backtest parameters
    # -------------------------------------------------------------------------
    # Use 6 months of data ending recently - short enough to run fast,
    # long enough to get meaningful statistics
    end_date = datetime(2024, 12, 1)
    start_date = end_date - timedelta(days=180)  # ~6 months

    symbol = Symbol("SPY")
    starting_balance = 100_000.0  # $100k starting capital

    print(f"📅 Date Range: {start_date.date()} to {end_date.date()}")
    print(f"📈 Symbol: {symbol}")
    print(f"💰 Starting Balance: ${starting_balance:,.2f}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Fetch historical data from Yahoo Finance
    # -------------------------------------------------------------------------
    print("🔄 Fetching data from Yahoo Finance...")

    data_source = YahooDataSource()
    date_range = DateRange(start=start_date, end=end_date)
    bars = list(
        data_source.fetch_bars(
            symbols=[symbol],
            date_range=date_range,
            granularity="1d",  # Daily bars
        )
    )

    print(f"✅ Fetched {len(bars)} daily bars")
    if bars:
        print(f"   First bar: {bars[0].timestamp.date()} @ ${bars[0].close:.2f}")
        print(f"   Last bar:  {bars[-1].timestamp.date()} @ ${bars[-1].close:.2f}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Initialize strategy and account
    # -------------------------------------------------------------------------
    # Moving Average Crossover with 10-day and 30-day SMAs
    # - Fast period (10): Responsive to recent price action
    # - Slow period (30): Smooths out noise, shows longer-term trend
    strategy = MovingAverageCrossoverStrategy(
        params={
            "symbol": str(symbol),
            "short_period": 10,
            "long_period": 30,
            "quantity": 100,  # Trade 100 shares per signal
        }
    )

    print(f"📊 Strategy: {strategy.__class__.__name__}")
    print(f"   Short SMA Period: {strategy.short_period}")
    print(f"   Long SMA Period:  {strategy.long_period}")
    print(f"   Trade Quantity:   {strategy.quantity} shares")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Run the backtest
    # -------------------------------------------------------------------------
    print("🚀 Running backtest...")
    print("-" * 60)

    backtest = Backtest(
        bars=bars,
        strategy=strategy,
        initial_balance=starting_balance,
        # Add realistic trading costs
        commission_per_trade=1.0,  # $1 per trade
        slippage_pct=0.001,  # 0.1% slippage (10 bps)
    )

    result = backtest.run()

    print("-" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 5: Fetch benchmark data for comparison
    # -------------------------------------------------------------------------
    print("🔄 Fetching benchmark data for comparison...")

    # Define benchmarks: ticker symbol -> display name
    benchmarks = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
    }

    # Calculate percent changes for the traded symbol and benchmarks
    benchmark_returns: dict[str, float | None] = {}

    # First, get the traded symbol's return (buy & hold baseline)
    if bars:
        symbol_start_price = bars[0].close
        symbol_end_price = bars[-1].close
        symbol_return = (symbol_end_price - symbol_start_price) / symbol_start_price
        benchmark_returns[str(symbol)] = symbol_return
    else:
        benchmark_returns[str(symbol)] = None

    # Fetch benchmark index data
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
            else:
                benchmark_returns[ticker] = None
        except Exception:
            benchmark_returns[ticker] = None

    print(f"✅ Benchmark data fetched")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Display results
    # -------------------------------------------------------------------------
    print("📈 BACKTEST RESULTS")
    print("=" * 60)

    # Calculate final equity from the last entry in equity_history or account balance
    if result.equity_history:
        final_equity = result.equity_history[-1][1]
    else:
        final_equity = result.final_account.cleared_balance

    strategy_return = (final_equity - starting_balance) / starting_balance

    # =========================================================================
    # PERFORMANCE COMPARISON REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON REPORT")
    print("=" * 60)

    print(f"\n{'Investment':<25} {'Start':>12} {'End':>12} {'Change':>10}")
    print("-" * 60)

    # Strategy performance
    print(
        f"{'Strategy (Your Account)':<25} "
        f"${starting_balance:>10,.0f} "
        f"${final_equity:>10,.0f} "
        f"{strategy_return:>+9.2%}"
    )

    print("-" * 60)

    # Traded symbol (buy & hold baseline)
    if benchmark_returns.get(str(symbol)) is not None:
        sym_ret = benchmark_returns[str(symbol)]
        sym_start = bars[0].close
        sym_end = bars[-1].close
        print(
            f"{str(symbol) + ' (Buy & Hold)':<25} "
            f"${sym_start:>10,.2f} "
            f"${sym_end:>10,.2f} "
            f"{sym_ret:>+9.2%}"
        )

    # Benchmark indices
    for ticker, name in benchmarks.items():
        ret = benchmark_returns.get(ticker)
        if ret is not None:
            print(f"{name:<25} {'-':>12} {'-':>12} {ret:>+9.2%}")
        else:
            print(f"{name:<25} {'-':>12} {'-':>12} {'N/A':>10}")

    print("-" * 60)

    # Alpha calculation (strategy return vs S&P 500)
    sp500_return = benchmark_returns.get("^GSPC")
    if sp500_return is not None:
        alpha = strategy_return - sp500_return
        alpha_label = "outperformed" if alpha > 0 else "underperformed"
        print(f"\n📈 Alpha vs S&P 500: {alpha:>+.2%} ({alpha_label})")
    print()

    # =========================================================================
    # Account Summary
    # =========================================================================
    print("=" * 60)
    print("💰 ACCOUNT SUMMARY")
    print("=" * 60)
    print(f"\n   Starting Equity:  ${starting_balance:>12,.2f}")
    print(f"   Final Equity:     ${final_equity:>12,.2f}")
    print(f"   Net P&L:          ${final_equity - starting_balance:>+12,.2f}")
    print(f"   Return:           {strategy_return:>+12.2%}")

    # Performance metrics
    metrics = result.metrics
    print(f"\n📊 Performance Metrics:")
    print(f"   Total Return:     {metrics.total_return:>12.2%}")
    print(f"   Max Drawdown:     {metrics.max_drawdown:>12.2%}")
    print(f"   Volatility:       {metrics.volatility:>12.4f}")

    if metrics.sharpe_ratio is not None:
        print(f"   Sharpe Ratio:     {metrics.sharpe_ratio:>12.2f}")
    else:
        print(f"   Sharpe Ratio:     {'N/A':>12}")

    if metrics.sortino_ratio is not None:
        print(f"   Sortino Ratio:    {metrics.sortino_ratio:>12.2f}")
    else:
        print(f"   Sortino Ratio:    {'N/A':>12}")

    # Trade statistics
    print(f"\n📋 Trade Statistics:")
    print(f"   Number of Trades: {metrics.num_trades:>12}")

    # Show closed trade stats (win rate only meaningful for closed trades)
    closed_trades = len(result.trade_pnls)
    print(f"   Closed Trades:    {closed_trades:>12}")

    if closed_trades > 0 and metrics.win_rate is not None:
        print(f"   Win Rate:         {metrics.win_rate:>12.2%}")

        if metrics.avg_win is not None and metrics.avg_win > 0:
            print(f"   Average Win:      ${metrics.avg_win:>11,.2f}")
        if metrics.avg_loss is not None and metrics.avg_loss < 0:
            print(f"   Average Loss:     ${metrics.avg_loss:>11,.2f}")

        if metrics.profit_factor is not None and metrics.profit_factor > 0:
            print(f"   Profit Factor:    {metrics.profit_factor:>12.2f}")

        if metrics.expectancy is not None:
            print(f"   Expectancy:       ${metrics.expectancy:>11,.2f}")
    elif closed_trades == 0:
        print(f"   ℹ️  Position still open - no closed trade stats available")

    # Execution details
    print(f"\n📝 Execution Log ({len(result.executions)} trades):")
    for i, execution in enumerate(result.executions[:10], 1):  # Show first 10
        side_emoji = "🟢" if execution.side == "buy" else "🔴"
        print(
            f"   {i:2}. {side_emoji} {execution.side.upper():4} "
            f"{execution.quantity:>6.0f} {execution.symbol} "
            f"@ ${execution.price:>8.2f} "
            f"on {execution.timestamp.date()}"
        )

    if len(result.executions) > 10:
        print(f"   ... and {len(result.executions) - 10} more trades")

    print()
    print("=" * 60)
    print("✅ Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
