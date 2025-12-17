#!/usr/bin/env python3
"""Command-line interface for the trading backtester."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest with the specified strategy."""
    from trading.data import YahooDataSource
    from trading.strategies import (
        BuyAndHoldStrategy,
        MeanReversionStrategy,
        MovingAverageCrossoverStrategy,
        RandomStrategy,
        RSIStrategy,
    )
    from trading.training import Backtest
    from trading.types import DateRange, Symbol

    # Map strategy names to classes
    strategy_map = {
        "buy_hold": BuyAndHoldStrategy,
        "ma_crossover": MovingAverageCrossoverStrategy,
        "mean_reversion": MeanReversionStrategy,
        "rsi": RSIStrategy,
        "random": RandomStrategy,
    }

    if args.strategy not in strategy_map:
        print(f"Error: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {', '.join(strategy_map.keys())}")
        return 1

    # Parse dates
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    print("=" * 60)
    print("BACKTEST")
    print("=" * 60)
    print(f"Symbol:    {args.symbol}")
    print(f"Strategy:  {args.strategy}")
    print(f"Period:    {args.start} to {args.end}")
    print(f"Initial:   ${args.balance:,.2f}")
    print(f"Quantity:  {args.quantity}")

    # Fetch data
    print("\n📊 Fetching data from Yahoo Finance...")
    source = YahooDataSource()
    bars = list(
        source.fetch_bars(
            [Symbol(args.symbol)],
            DateRange(start=start_date, end=end_date),
            args.granularity,
        )
    )
    print(f"   Fetched {len(bars)} bars")

    if not bars:
        print("Error: No data fetched. Check symbol and date range.")
        return 1

    # Create strategy
    strategy_params = {
        "symbol": args.symbol,
        "quantity": args.quantity,
    }

    # Add strategy-specific params
    if args.strategy == "ma_crossover":
        strategy_params["short_period"] = args.short_ma
        strategy_params["long_period"] = args.long_ma
    elif args.strategy == "mean_reversion":
        strategy_params["ma_period"] = args.ma_period
        strategy_params["threshold"] = args.threshold
    elif args.strategy == "rsi":
        strategy_params["rsi_period"] = args.rsi_period
        strategy_params["oversold"] = args.oversold
        strategy_params["overbought"] = args.overbought
    elif args.strategy == "random":
        strategy_params["trade_probability"] = args.trade_prob
        if args.seed is not None:
            strategy_params["seed"] = args.seed

    strategy_class = strategy_map[args.strategy]
    strategy = strategy_class(strategy_params)

    # Run backtest
    print("\n🚀 Running backtest...")
    backtest = Backtest(
        bars=bars,
        strategy=strategy,
        initial_balance=args.balance,
    )
    result = backtest.run()

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Equity:    ${result.equity_history[-1][1]:,.2f}")
    print(f"Total Return:    {result.metrics.total_return:+.2%}")
    print(f"Max Drawdown:    {result.metrics.max_drawdown:.2%}")
    print(f"Volatility:      {result.metrics.volatility:.4f}")
    if result.metrics.sharpe_ratio is not None:
        print(f"Sharpe Ratio:    {result.metrics.sharpe_ratio:.2f}")
    print(f"Num Trades:      {result.metrics.num_trades}")

    if result.executions and args.show_trades:
        print("\n📋 Trade History:")
        for i, exec in enumerate(result.executions[:20]):
            print(
                f"   {i+1:3}. {exec.timestamp.date()} | "
                f"{exec.side.upper():4} {exec.quantity:.0f} {exec.symbol} @ ${exec.price:.2f}"
            )
        if len(result.executions) > 20:
            print(f"   ... and {len(result.executions) - 20} more trades")

    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    """Fetch and store historical market data."""
    from trading._core import store_dataset
    from trading.commands.fetch_data import load_fetch_data_config
    from trading.data.sources import resolve_data_source
    from trading.exceptions import ConfigError, DataSourceError
    from trading.types import DatasetMetadata, NormalizedBar

    try:
        config = load_fetch_data_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 1

    print("=" * 60)
    print("FETCH DATA")
    print("=" * 60)
    print(f"Dataset ID:  {config.dataset_id}")
    print(f"Symbols:     {', '.join(str(s) for s in config.symbols)}")
    print(
        f"Date Range:  {config.date_range.start.date()} to {config.date_range.end.date()}"
    )
    print(f"Granularity: {config.granularity}")
    print(f"Source:      {config.data_source}")

    try:
        source = resolve_data_source(config)
    except DataSourceError as e:
        print(f"Data source error: {e}")
        return 1

    print("\n📊 Fetching data...")

    try:
        bars = list(
            source.fetch_bars(
                config.symbols,
                config.date_range,
                config.granularity,
            )
        )
    except DataSourceError as e:
        print(f"Failed to fetch data: {e}")
        return 1

    if not bars:
        print("No data fetched. Check symbols and date range.")
        return 1

    print(f"   Fetched {len(bars)} bars")

    # Normalize bars
    normalized_bars = [
        NormalizedBar(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        for bar in bars
    ]

    # Create metadata
    metadata = DatasetMetadata(
        dataset_id=config.dataset_id,
        symbols=config.symbols,
        date_range=config.date_range.model_dump(),
        granularity=config.granularity,
        data_source=config.data_source,
        source_params=config.source_params,
        created_at=datetime.now(timezone.utc),
        bar_count=len(normalized_bars),
    )

    print("\n💾 Storing dataset...")

    try:
        store_dataset(
            [b.model_dump() for b in normalized_bars],
            str(config.dataset_id),
            metadata.model_dump_json(),
        )
    except Exception as e:
        print(f"Failed to store dataset: {e}")
        return 1

    print(f"   Stored to ~/.trading/datasets/{config.dataset_id}/")
    print("\n✅ Done!")

    return 0


def cmd_run_training(args: argparse.Namespace) -> int:
    """Run a training session from configuration."""
    import importlib

    from trading.commands.run_training import (
        load_training_config,
        validate_training_config,
    )
    from trading.exceptions import ConfigError
    from trading.training import Backtest

    try:
        config = load_training_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 1

    print("=" * 60)
    print("TRAINING RUN")
    print("=" * 60)
    print(f"Run ID:      {config.run_id}")
    print(f"Datasets:    {', '.join(str(d) for d in config.datasets)}")
    print(f"Strategy:    {config.strategy_class_path}")
    print(f"Balance:     ${config.account_starting_balance:,.2f}")

    # Validate datasets exist
    try:
        warnings = validate_training_config(config)
        for warning in warnings:
            print(f"⚠️  {warning}")
    except ConfigError as e:
        print(f"Validation error: {e}")
        return 1

    # Load datasets
    print("\n📊 Loading datasets...")
    all_bars = []
    for dataset_id in config.datasets:
        try:
            from trading._core import load_dataset

            # Load bars from storage
            bar_dicts = load_dataset(str(dataset_id))

            # Convert to Bar objects
            from trading.types import Bar

            for bar_dict in bar_dicts:
                all_bars.append(
                    Bar(
                        symbol=bar_dict["symbol"],
                        timestamp=bar_dict["timestamp"],
                        open=bar_dict["open"],
                        high=bar_dict["high"],
                        low=bar_dict["low"],
                        close=bar_dict["close"],
                        volume=bar_dict["volume"],
                    )
                )
            print(f"   Loaded {len(bar_dicts)} bars from {dataset_id}")
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}: {e}")
            return 1

    if not all_bars:
        print("No data loaded from datasets.")
        return 1

    # Instantiate strategy
    print("\n🧠 Loading strategy...")
    module_path, class_name = config.strategy_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    strategy = strategy_class(config.strategy_params)
    print(f"   Loaded {class_name}")

    # Run backtest
    print("\n🚀 Running training...")
    if config.risk_max_position_size or config.risk_max_leverage != 1.0:
        print(
            f"   Risk: max_position=${config.risk_max_position_size or 'unlimited'}, leverage={config.risk_max_leverage}x"
        )

    backtest = Backtest(
        bars=all_bars,
        strategy=strategy,
        initial_balance=config.account_starting_balance,
        run_id=str(config.run_id),
        max_position_size=config.risk_max_position_size,
        max_leverage=config.risk_max_leverage,
    )
    result = backtest.run()

    # Report any rejected orders
    if backtest.rejected_orders:
        print(
            f"   ⚠️  {len(backtest.rejected_orders)} orders rejected by risk constraints"
        )

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    final_equity = (
        result.equity_history[-1][1]
        if result.equity_history
        else config.account_starting_balance
    )
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Total Return:    {result.metrics.total_return:+.2%}")
    print(f"Max Drawdown:    {result.metrics.max_drawdown:.2%}")
    print(f"Volatility:      {result.metrics.volatility:.4f}")
    if result.metrics.sharpe_ratio is not None:
        print(f"Sharpe Ratio:    {result.metrics.sharpe_ratio:.2f}")
    print(f"Num Trades:      {result.metrics.num_trades}")

    # Store results
    if not args.no_save:
        import json

        from trading._core import store_run_results

        print("\n💾 Storing results...")

        # Prepare data for storage
        config_json = config.model_dump_json()
        metrics_json = result.metrics.model_dump_json()

        # Serialize executions
        executions_data = [
            {
                "symbol": str(e.symbol),
                "side": e.side,
                "quantity": e.quantity,
                "price": e.price,
                "timestamp": e.timestamp.isoformat(),
                "order_id": e.order_id,
            }
            for e in result.executions
        ]
        executions_json = json.dumps(executions_data)

        # Serialize equity history
        equity_data = [
            {"timestamp": ts.isoformat(), "equity": eq}
            for ts, eq in result.equity_history
        ]
        equity_json = json.dumps(equity_data)

        try:
            store_run_results(
                str(config.run_id),
                config_json,
                metrics_json,
                executions_json,
                equity_json,
                final_equity,
                result.metrics.num_trades,
            )
            print(f"   Saved to ~/.trading/runs/{config.run_id}/")
        except Exception as e:
            print(f"   Warning: Failed to save results: {e}")

    print("\n✅ Training complete!")

    return 0


def cmd_inspect_run(args: argparse.Namespace) -> int:
    """Inspect a completed training run."""
    from trading._core import list_runs, load_run_results, run_exists

    # If no run_id provided, list available runs
    if args.run_id is None:
        runs = list_runs()
        if not runs:
            print("No runs found in ~/.trading/runs/")
            return 0

        print("=" * 70)
        print("AVAILABLE RUNS")
        print("=" * 70)
        print(f"{'Run ID':<45} {'Trades':>8} {'Final Equity':>15}")
        print("-" * 70)

        for run_id in runs[:20]:  # Show latest 20
            try:
                data = load_run_results(run_id)
                summary = data.get("summary", {})
                final_eq = summary.get("final_equity", 0)
                num_trades = summary.get("num_trades", 0)
                print(f"{run_id:<45} {num_trades:>8} ${final_eq:>14,.2f}")
            except Exception:
                print(f"{run_id:<45} {'(error)':>8}")

        if len(runs) > 20:
            print(f"\n... and {len(runs) - 20} more runs")

        return 0

    # Check run exists
    if not run_exists(args.run_id):
        print(f"Run not found: {args.run_id}")
        return 1

    # Load run data
    try:
        data = load_run_results(args.run_id)
    except Exception as e:
        print(f"Failed to load run: {e}")
        return 1

    print("=" * 60)
    print(f"RUN: {args.run_id}")
    print("=" * 60)

    # Show metrics
    metrics = data.get("metrics", {})
    if metrics:
        print("\n📊 METRICS")
        print(f"   Total Return:  {metrics.get('total_return', 0):+.2%}")
        print(f"   Max Drawdown:  {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Volatility:    {metrics.get('volatility', 0):.4f}")
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is not None:
            print(f"   Sharpe Ratio:  {sharpe:.2f}")
        print(f"   Num Trades:    {metrics.get('num_trades', 0)}")

    # Show config
    config = data.get("config", {})
    if config and args.show_config:
        print("\n⚙️  CONFIGURATION")
        print(f"   Strategy:  {config.get('strategy_class_path', 'N/A')}")
        print(f"   Datasets:  {', '.join(config.get('datasets', []))}")
        print(f"   Balance:   ${config.get('account_starting_balance', 0):,.2f}")

    # Show executions
    executions = data.get("executions", [])
    if executions and args.show_trades:
        print(f"\n📋 TRADES ({len(executions)} total)")
        for i, e in enumerate(executions[:20]):
            ts = e.get("timestamp", "")[:10]
            side = e.get("side", "").upper()
            qty = e.get("quantity", 0)
            sym = e.get("symbol", "")
            price = e.get("price", 0)
            print(f"   {i+1:3}. {ts} | {side:4} {qty:.0f} {sym} @ ${price:.2f}")
        if len(executions) > 20:
            print(f"   ... and {len(executions) - 20} more trades")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare multiple strategies on the same data."""
    from trading.data import YahooDataSource
    from trading.strategies import (
        BuyAndHoldStrategy,
        MeanReversionStrategy,
        MovingAverageCrossoverStrategy,
        RandomStrategy,
        RSIStrategy,
    )
    from trading.training import Backtest
    from trading.types import DateRange, Symbol

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Symbol:  {args.symbol}")
    print(f"Period:  {args.start} to {args.end}")
    print(f"Initial: ${args.balance:,.2f}")

    # Fetch data
    print("\n📊 Fetching data...")
    source = YahooDataSource()
    bars = list(
        source.fetch_bars(
            [Symbol(args.symbol)],
            DateRange(start=start_date, end=end_date),
            args.granularity,
        )
    )
    print(f"   Fetched {len(bars)} bars\n")

    if not bars:
        print("Error: No data fetched.")
        return 1

    # Define strategies to compare
    strategies = [
        ("Buy & Hold", BuyAndHoldStrategy({"symbol": args.symbol, "quantity": 20})),
        (
            "MA Crossover (5/20)",
            MovingAverageCrossoverStrategy(
                {
                    "symbol": args.symbol,
                    "short_period": 5,
                    "long_period": 20,
                    "quantity": 20,
                }
            ),
        ),
        (
            "Mean Reversion (2%)",
            MeanReversionStrategy(
                {
                    "symbol": args.symbol,
                    "ma_period": 20,
                    "threshold": 0.02,
                    "quantity": 20,
                }
            ),
        ),
        (
            "RSI (14)",
            RSIStrategy({"symbol": args.symbol, "rsi_period": 14, "quantity": 20}),
        ),
        (
            "Random (5%)",
            RandomStrategy(
                {
                    "symbol": args.symbol,
                    "trade_probability": 0.05,
                    "quantity": 20,
                    "seed": 42,
                }
            ),
        ),
    ]

    print(
        f"{'Strategy':<25} {'Return':>10} {'Drawdown':>10} {'Sharpe':>8} {'Trades':>8}"
    )
    print("-" * 70)

    for name, strategy in strategies:
        backtest = Backtest(bars=bars, strategy=strategy, initial_balance=args.balance)
        result = backtest.run()

        sharpe = (
            f"{result.metrics.sharpe_ratio:.2f}"
            if result.metrics.sharpe_ratio
            else "N/A"
        )
        print(
            f"{name:<25} {result.metrics.total_return:>+9.2%} "
            f"{result.metrics.max_drawdown:>9.2%} {sharpe:>8} {result.metrics.num_trades:>8}"
        )

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading backtester CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a single backtest")
    backtest_parser.add_argument("symbol", help="Stock symbol (e.g., AAPL)")
    backtest_parser.add_argument(
        "-s",
        "--strategy",
        default="buy_hold",
        choices=["buy_hold", "ma_crossover", "mean_reversion", "rsi", "random"],
        help="Strategy to use (default: buy_hold)",
    )
    backtest_parser.add_argument(
        "--start", default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end", default="2024-01-01", help="End date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "-b", "--balance", type=float, default=10000.0, help="Initial balance"
    )
    backtest_parser.add_argument(
        "-q", "--quantity", type=float, default=10, help="Shares per trade"
    )
    backtest_parser.add_argument(
        "-g", "--granularity", default="1d", help="Data granularity (default: 1d)"
    )
    backtest_parser.add_argument(
        "--show-trades", action="store_true", help="Show trade history"
    )
    # MA Crossover params
    backtest_parser.add_argument(
        "--short-ma", type=int, default=5, help="Short MA period"
    )
    backtest_parser.add_argument(
        "--long-ma", type=int, default=20, help="Long MA period"
    )
    # Mean Reversion params
    backtest_parser.add_argument("--ma-period", type=int, default=20, help="MA period")
    backtest_parser.add_argument(
        "--threshold", type=float, default=0.02, help="Deviation threshold"
    )
    # RSI params
    backtest_parser.add_argument(
        "--rsi-period", type=int, default=14, help="RSI period"
    )
    backtest_parser.add_argument(
        "--oversold", type=float, default=30, help="Oversold level"
    )
    backtest_parser.add_argument(
        "--overbought", type=float, default=70, help="Overbought level"
    )
    # Random params
    backtest_parser.add_argument(
        "--trade-prob", type=float, default=0.05, help="Trade probability"
    )
    backtest_parser.add_argument("--seed", type=int, help="Random seed")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple strategies"
    )
    compare_parser.add_argument("symbol", help="Stock symbol (e.g., AAPL)")
    compare_parser.add_argument(
        "--start", default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    compare_parser.add_argument(
        "--end", default="2024-01-01", help="End date (YYYY-MM-DD)"
    )
    compare_parser.add_argument(
        "-b", "--balance", type=float, default=10000.0, help="Initial balance"
    )
    compare_parser.add_argument(
        "-g", "--granularity", default="1d", help="Data granularity (default: 1d)"
    )

    # Fetch data command
    fetch_parser = subparsers.add_parser(
        "fetch-data", help="Fetch and store historical market data"
    )
    fetch_parser.add_argument("config", help="Path to YAML configuration file")

    # Run training command
    train_parser = subparsers.add_parser(
        "run-training", help="Run a training session from configuration"
    )
    train_parser.add_argument("config", help="Path to YAML configuration file")
    train_parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to storage"
    )

    # Inspect run command
    inspect_parser = subparsers.add_parser(
        "inspect-run", help="Inspect a completed training run"
    )
    inspect_parser.add_argument(
        "run_id", nargs="?", default=None, help="Run ID to inspect (omit to list runs)"
    )
    inspect_parser.add_argument(
        "--show-config", action="store_true", help="Show run configuration"
    )
    inspect_parser.add_argument(
        "--show-trades", action="store_true", help="Show trade history"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "backtest":
        return cmd_backtest(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "fetch-data":
        return cmd_fetch_data(args)
    elif args.command == "run-training":
        return cmd_run_training(args)
    elif args.command == "inspect-run":
        return cmd_inspect_run(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
