# Getting Started with Trading Bot

This guide will help you get up and running with the trading bot system for backtesting, strategy development, and paper trading.

## Prerequisites

- Python 3.11+
- Rust toolchain (for building the native extension)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/trading.git
cd trading

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and build
pip install -e .
```

The installation will compile the Rust core library automatically via Maturin.

## Quick Start

### 1. Run Your First Backtest

The simplest way to start is with a single-symbol backtest using the CLI:

```bash
# Backtest Apple stock with Buy & Hold strategy
trading backtest AAPL --start 2024-01-01 --end 2024-12-01

# Try a moving average crossover strategy
trading backtest AAPL -s ma_crossover --start 2024-01-01 --end 2024-12-01 --show-trades

# Use RSI strategy with custom parameters
trading backtest SPY -s rsi --rsi-period 14 --oversold 30 --overbought 70 --start 2024-01-01
```

### 2. Compare Multiple Strategies

See how different strategies perform on the same data:

```bash
trading compare AAPL --start 2024-01-01 --end 2024-12-01
```

This compares Buy & Hold, MA Crossover, Mean Reversion, RSI, and Random strategies.

### 3. Paper Trading (Live Simulation)

Test your strategy with live market data (simulated trades):

```bash
# Start paper trading with RSI strategy
trading paper-trade SPY -s rsi --account-name my_test_account

# List your paper trading accounts
trading paper-trade --list-accounts
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `trading backtest <symbol>` | Run single strategy backtest |
| `trading compare <symbol>` | Compare multiple strategies |
| `trading fetch-data <config.yaml>` | Download and store historical data |
| `trading run-training <config.yaml>` | Run training from YAML config |
| `trading inspect-run <run_id>` | View results of past runs |
| `trading paper-trade <symbol>` | Paper trade with live quotes |

## Available Strategies

| Strategy | CLI Flag | Description |
|----------|----------|-------------|
| Buy & Hold | `buy_hold` | Buy at start, hold until end |
| MA Crossover | `ma_crossover` | Buy when short MA crosses above long MA |
| Mean Reversion | `mean_reversion` | Buy below moving average, sell above |
| RSI | `rsi` | Buy oversold, sell overbought |
| Random | `random` | Random buy/sell decisions (baseline) |

## Using the Python API

For more control, use the Python library directly:

### Basic Backtest

```python
from datetime import datetime
from trading.data.sources import YahooDataSource
from trading.strategies.examples import MovingAverageCrossoverStrategy
from trading.training.backtest import Backtest

# Fetch data
source = YahooDataSource()
bars = source.fetch("AAPL", "2024-01-01", "2024-12-01", "1d")

# Create strategy
strategy = MovingAverageCrossoverStrategy(short_period=10, long_period=30)

# Run backtest
backtest = Backtest(
    bars,
    strategy,
    initial_balance=10000.0,
)
result = backtest.run()

# View results
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
```

### Compare Multiple Strategies

```python
from trading.training import MultiBacktest, StrategyConfig
from trading.strategies.examples import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    RSIStrategy,
)

configs = [
    StrategyConfig("Buy & Hold", BuyAndHoldStrategy()),
    StrategyConfig("MA Crossover", MovingAverageCrossoverStrategy(5, 20)),
    StrategyConfig("RSI", RSIStrategy(period=14)),
]

multi = MultiBacktest(bars, configs)
comparison = multi.run()

# Get ranked results
for rank, result in enumerate(comparison.ranked_by_return(), 1):
    print(f"{rank}. {result.strategy_name}: {result.metrics.total_return:.2%}")
```

### Custom Strategy

Create your own trading strategy:

```python
from trading.strategies.base import Strategy
from trading.types import AnalysisSnapshot, OrderRequest

class MyStrategy(Strategy):
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.entry_price: float | None = None
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        orders = []
        
        for symbol, bar in snapshot.bars.items():
            position = snapshot.account.positions.get(symbol)
            
            # Entry logic: buy if we don't have a position
            if position is None or position.quantity == 0:
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="buy",
                    quantity=10,
                ))
                self.entry_price = bar.close
            
            # Exit logic: sell if we hit our target
            elif self.entry_price and bar.close > self.entry_price * (1 + self.threshold):
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                ))
                self.entry_price = None
        
        return orders
```

### Using Position Sizing

Dynamic position sizing based on account equity:

```python
from trading.strategies.sizing import PercentOfEquitySizer, KellyCriterionSizer

# Risk 2% of equity per trade
sizer = PercentOfEquitySizer(percent=0.02)

# Or use Kelly Criterion
sizer = KellyCriterionSizer(win_rate=0.55, avg_win=100, avg_loss=80)

backtest = Backtest(bars, strategy, position_sizer=sizer)
```

### Walk-Forward Validation

Test strategy robustness with out-of-sample validation:

```python
from trading.training import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_periods=63,    # ~3 months training
    test_periods=21,     # ~1 month testing
    step_size=21,        # Roll forward 1 month
)

validator = WalkForwardValidator(bars, config)
result = validator.validate(strategy)

print(f"In-Sample Return: {result.in_sample_metrics.total_return:.2%}")
print(f"Out-of-Sample Return: {result.out_of_sample_metrics.total_return:.2%}")
print(f"Consistency Ratio: {result.consistency_ratio:.2%}")
```

### Hyperparameter Optimization

Find optimal strategy parameters:

```python
from trading.training import GridSearchOptimizer, ParameterSpec

optimizer = GridSearchOptimizer(
    strategy_class=MovingAverageCrossoverStrategy,
    param_specs=[
        ParameterSpec("short_period", [5, 10, 15, 20]),
        ParameterSpec("long_period", [20, 30, 40, 50]),
    ],
    metric="sharpe_ratio",
)

best_params, best_score = optimizer.optimize(bars)
print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {best_score:.2f}")
```

### Reinforcement Learning

Train an RL agent on market data:

```python
from trading.rl import TradingEnv, TradingEnvConfig, RLStrategy

# Create Gym-compatible environment
config = TradingEnvConfig(
    symbol="SPY",
    initial_balance=10000,
    trade_quantity=10,
)
env = TradingEnv(bars, config)

# Train with stable-baselines3 (install separately)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Use trained model in backtest
strategy = RLStrategy(model, symbol="SPY", trade_quantity=10)
result = Backtest(bars, strategy).run()
```

## Configuration Files

For reproducible experiments, use YAML configuration files:

### Backtest Configuration

```yaml
# training_config.yaml
run_id: "my_experiment_001"
datasets:
  - "spy_2024"
strategy:
  class_path: "trading.strategies.examples.MovingAverageCrossoverStrategy"
  params:
    short_period: 10
    long_period: 30
account:
  starting_balance: 10000.0
  base_currency: "USD"
  clearing_delay_hours: 24
risk:
  max_position_size: 5000.0
  max_leverage: 1.0
```

Run with:
```bash
trading run-training training_config.yaml
```

### Fetch Data Configuration

```yaml
# fetch_data.yaml
symbols:
  - "SPY"
  - "QQQ"
date_range:
  start: "2020-01-01"
  end: "2024-12-01"
granularity: "1d"
data_source: "yahoo"
dataset_id: "spy_qqq_2020_2024"
```

Run with:
```bash
trading fetch-data fetch_data.yaml
```

## Data Storage

The trading bot stores data in `~/.trading/`:

```
~/.trading/
├── datasets/           # Historical market data (Parquet format)
│   └── spy_2024/
│       ├── bars.parquet
│       └── metadata.json
├── runs/               # Backtest results
│   └── run_001/
│       ├── config.json
│       ├── metrics.json
│       └── executions.json
└── paper/              # Paper trading accounts
    └── my_account/
        ├── account.json
        └── orders.json
```

## Next Steps

- See [Examples](examples.md) for more detailed usage patterns
- Read [Theory of Operation](theory-of-operation.md) to understand the architecture
- Check out the built-in strategies in `src/trading/strategies/examples.py`

## Troubleshooting

### "No data available"
Ensure you have internet access for Yahoo Finance data, or use locally stored datasets.

### "Strategy not found"
Check that your strategy class path is correct and the module is importable.

### Rust compilation errors
Ensure you have a working Rust toolchain: `rustup update`

