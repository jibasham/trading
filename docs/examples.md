# Examples

This document provides comprehensive examples for common use cases with the trading bot.

## Table of Contents

1. [Basic Backtesting](#basic-backtesting)
2. [Custom Strategies](#custom-strategies)
3. [Multi-Symbol Trading](#multi-symbol-trading)
4. [Position Sizing](#position-sizing)
5. [Walk-Forward Validation](#walk-forward-validation)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Reinforcement Learning](#reinforcement-learning)
8. [Paper Trading](#paper-trading)
9. [Commission and Slippage](#commission-and-slippage)
10. [Portfolio Allocation](#portfolio-allocation)

---

## Basic Backtesting

### Simple Buy & Hold

```python
from trading.data.sources import YahooDataSource
from trading.strategies.examples import BuyAndHoldStrategy
from trading.training.backtest import Backtest

# Fetch 5 years of daily data
source = YahooDataSource()
bars = source.fetch("SPY", "2019-01-01", "2024-01-01", "1d")

# Run backtest
strategy = BuyAndHoldStrategy()
result = Backtest(bars, strategy, initial_balance=100000).run()

# Print comprehensive metrics
print(f"""
Backtest Results
================
Total Return:    {result.metrics.total_return:>8.2%}
Max Drawdown:    {result.metrics.max_drawdown:>8.2%}
Sharpe Ratio:    {result.metrics.sharpe_ratio:>8.2f}
Sortino Ratio:   {result.metrics.sortino_ratio:>8.2f}
Win Rate:        {result.metrics.win_rate:>8.2%}
Profit Factor:   {result.metrics.profit_factor:>8.2f}
Num Trades:      {result.metrics.num_trades:>8d}
""")
```

### Moving Average Crossover with Custom Parameters

```python
from trading.strategies.examples import MovingAverageCrossoverStrategy

# Test different MA periods
short_periods = [5, 10, 15]
long_periods = [20, 30, 50]

for short in short_periods:
    for long in long_periods:
        strategy = MovingAverageCrossoverStrategy(short, long)
        result = Backtest(bars, strategy).run()
        print(f"MA({short}/{long}): Return={result.metrics.total_return:.2%}, "
              f"Sharpe={result.metrics.sharpe_ratio:.2f}")
```

### RSI Mean Reversion

```python
from trading.strategies.examples import RSIStrategy

# Conservative RSI settings
strategy = RSIStrategy(
    period=14,
    oversold=25,      # More extreme oversold
    overbought=75,    # More extreme overbought
    quantity=50,
)

result = Backtest(bars, strategy, initial_balance=50000).run()
```

---

## Custom Strategies

### Momentum Strategy

```python
from trading.strategies.base import Strategy
from trading.types import AnalysisSnapshot, OrderRequest, Symbol

class MomentumStrategy(Strategy):
    """Buy when price momentum is positive, sell when negative."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.05):
        self.lookback = lookback
        self.threshold = threshold
        self.price_history: dict[Symbol, list[float]] = {}
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        orders = []
        
        for symbol, bar in snapshot.bars.items():
            # Track price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(bar.close)
            
            # Need enough history
            if len(self.price_history[symbol]) < self.lookback:
                continue
            
            # Keep only lookback periods
            self.price_history[symbol] = self.price_history[symbol][-self.lookback:]
            
            # Calculate momentum
            old_price = self.price_history[symbol][0]
            momentum = (bar.close - old_price) / old_price
            
            position = snapshot.account.positions.get(symbol)
            has_position = position and position.quantity > 0
            
            # Strong positive momentum: buy
            if momentum > self.threshold and not has_position:
                orders.append(OrderRequest(symbol=symbol, side="buy", quantity=10))
            
            # Negative momentum: sell
            elif momentum < 0 and has_position:
                orders.append(OrderRequest(
                    symbol=symbol, side="sell", quantity=position.quantity
                ))
        
        return orders
```

### Bollinger Bands Strategy

```python
import statistics
from trading.strategies.base import Strategy
from trading.types import AnalysisSnapshot, OrderRequest, Symbol

class BollingerBandsStrategy(Strategy):
    """Trade based on Bollinger Band breakouts."""
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
        self.price_history: dict[Symbol, list[float]] = {}
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        orders = []
        
        for symbol, bar in snapshot.bars.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(bar.close)
            
            if len(self.price_history[symbol]) < self.period:
                continue
            
            prices = self.price_history[symbol][-self.period:]
            sma = statistics.mean(prices)
            std = statistics.stdev(prices)
            
            upper_band = sma + self.num_std * std
            lower_band = sma - self.num_std * std
            
            position = snapshot.account.positions.get(symbol)
            has_position = position and position.quantity > 0
            
            # Price below lower band: oversold, buy
            if bar.close < lower_band and not has_position:
                orders.append(OrderRequest(symbol=symbol, side="buy", quantity=10))
            
            # Price above upper band: overbought, sell
            elif bar.close > upper_band and has_position:
                orders.append(OrderRequest(
                    symbol=symbol, side="sell", quantity=position.quantity
                ))
        
        return orders
```

### Dual Moving Average with Volume Confirmation

```python
class VolumeConfirmedMAStrategy(Strategy):
    """MA crossover with volume confirmation."""
    
    def __init__(self, short_period: int = 10, long_period: int = 30, 
                 volume_threshold: float = 1.5):
        self.short_period = short_period
        self.long_period = long_period
        self.volume_threshold = volume_threshold
        self.price_history: dict[Symbol, list[float]] = {}
        self.volume_history: dict[Symbol, list[float]] = {}
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        orders = []
        
        for symbol, bar in snapshot.bars.items():
            # Track history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.volume_history[symbol] = []
            
            self.price_history[symbol].append(bar.close)
            self.volume_history[symbol].append(bar.volume)
            
            if len(self.price_history[symbol]) < self.long_period:
                continue
            
            prices = self.price_history[symbol]
            volumes = self.volume_history[symbol]
            
            short_ma = statistics.mean(prices[-self.short_period:])
            long_ma = statistics.mean(prices[-self.long_period:])
            avg_volume = statistics.mean(volumes[-self.long_period:])
            
            # Check for volume confirmation
            volume_confirmed = bar.volume > avg_volume * self.volume_threshold
            
            position = snapshot.account.positions.get(symbol)
            has_position = position and position.quantity > 0
            
            # Bullish crossover with volume
            if short_ma > long_ma and not has_position and volume_confirmed:
                orders.append(OrderRequest(symbol=symbol, side="buy", quantity=10))
            
            # Bearish crossover
            elif short_ma < long_ma and has_position:
                orders.append(OrderRequest(
                    symbol=symbol, side="sell", quantity=position.quantity
                ))
        
        return orders
```

---

## Multi-Symbol Trading

### Trading Multiple Symbols Simultaneously

```python
from trading.data.sources import YahooDataSource
from trading.strategies.examples import MovingAverageCrossoverStrategy
from trading.training.backtest import Backtest
from trading.types import NormalizedBar

# Fetch data for multiple symbols
source = YahooDataSource()
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

all_bars: list[NormalizedBar] = []
for symbol in symbols:
    bars = source.fetch(symbol, "2024-01-01", "2024-12-01", "1d")
    all_bars.extend(bars)

# Sort by timestamp for proper chronological processing
all_bars.sort(key=lambda b: b.timestamp)

# Run multi-symbol backtest
strategy = MovingAverageCrossoverStrategy(short_period=10, long_period=30)
result = Backtest(all_bars, strategy, initial_balance=100000).run()

print(f"Multi-symbol portfolio return: {result.metrics.total_return:.2%}")
```

### Symbol-Specific Strategies

```python
class MultiSymbolStrategy(Strategy):
    """Apply different strategies to different symbols."""
    
    def __init__(self):
        self.strategies = {
            "AAPL": MovingAverageCrossoverStrategy(5, 20),
            "GOOGL": RSIStrategy(period=14),
            "MSFT": MeanReversionStrategy(period=20, threshold=0.02),
        }
        self.default_strategy = BuyAndHoldStrategy()
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        all_orders = []
        
        for symbol, bar in snapshot.bars.items():
            # Create a snapshot with just this symbol
            single_snapshot = AnalysisSnapshot(
                timestamp=snapshot.timestamp,
                bars={symbol: bar},
                account=snapshot.account,
            )
            
            # Get appropriate strategy
            strategy = self.strategies.get(str(symbol), self.default_strategy)
            orders = strategy.decide(single_snapshot)
            all_orders.extend(orders)
        
        return all_orders
```

---

## Position Sizing

### Fixed Dollar Amount per Trade

```python
from trading.strategies.sizing import FixedDollarSizer

# Always risk $1000 per trade
sizer = FixedDollarSizer(dollar_amount=1000.0)

result = Backtest(
    bars, 
    strategy,
    position_sizer=sizer,
).run()
```

### Percent of Equity

```python
from trading.strategies.sizing import PercentOfEquitySizer

# Risk 5% of portfolio per trade
sizer = PercentOfEquitySizer(percent=0.05)

result = Backtest(bars, strategy, position_sizer=sizer).run()
```

### Volatility-Adjusted Sizing

```python
from trading.strategies.sizing import VolatilityAdjustedSizer

# Target 1% portfolio volatility per position
sizer = VolatilityAdjustedSizer(
    target_risk=0.01,
    lookback_period=20,
)

result = Backtest(bars, strategy, position_sizer=sizer).run()
```

### Kelly Criterion

```python
from trading.strategies.sizing import KellyCriterionSizer

# Based on historical win rate and average win/loss
sizer = KellyCriterionSizer(
    win_rate=0.55,
    avg_win=150.0,
    avg_loss=100.0,
    fraction=0.5,  # Half Kelly for safety
)

result = Backtest(bars, strategy, position_sizer=sizer).run()
```

### Risk-Based Sizing with Stop Loss

```python
from trading.strategies.sizing import RiskPercentSizer

# Risk 2% of portfolio per trade, with 5% stop loss
sizer = RiskPercentSizer(
    risk_percent=0.02,
    stop_loss_percent=0.05,
)

result = Backtest(bars, strategy, position_sizer=sizer).run()
```

---

## Walk-Forward Validation

### Basic Walk-Forward Test

```python
from trading.training import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_periods=126,   # 6 months training (daily bars)
    test_periods=21,     # 1 month testing
    step_size=21,        # Roll forward monthly
)

validator = WalkForwardValidator(bars, config)
result = validator.validate(strategy)

print(f"""
Walk-Forward Results
====================
Windows:              {result.num_windows}
In-Sample Return:     {result.in_sample_metrics.total_return:.2%}
Out-of-Sample Return: {result.out_of_sample_metrics.total_return:.2%}
Consistency:          {result.consistency_ratio:.2%}
Degradation:          {result.performance_degradation:.2%}
""")
```

### Walk-Forward with Optimization

```python
from trading.training import WalkForwardValidator, GridSearchOptimizer, ParameterSpec

def optimize_and_validate(bars, strategy_class, param_specs):
    """Optimize parameters in each training window."""
    
    config = WalkForwardConfig(
        train_periods=126,
        test_periods=21,
        step_size=21,
    )
    
    # Split data into windows
    windows = config.generate_windows(bars)
    
    results = []
    for train_bars, test_bars in windows:
        # Optimize on training data
        optimizer = GridSearchOptimizer(strategy_class, param_specs, "sharpe_ratio")
        best_params, _ = optimizer.optimize(train_bars)
        
        # Validate on test data
        strategy = strategy_class(**best_params)
        result = Backtest(test_bars, strategy).run()
        results.append(result)
    
    return results

# Example usage
param_specs = [
    ParameterSpec("short_period", [5, 10, 15]),
    ParameterSpec("long_period", [20, 30, 40]),
]

results = optimize_and_validate(bars, MovingAverageCrossoverStrategy, param_specs)
```

---

## Hyperparameter Optimization

### Grid Search

```python
from trading.training import GridSearchOptimizer, ParameterSpec

optimizer = GridSearchOptimizer(
    strategy_class=MovingAverageCrossoverStrategy,
    param_specs=[
        ParameterSpec("short_period", [5, 10, 15, 20]),
        ParameterSpec("long_period", [20, 30, 40, 50, 60]),
    ],
    metric="sharpe_ratio",  # Optimize for Sharpe
)

best_params, best_score = optimizer.optimize(bars)
print(f"Best: {best_params} with Sharpe={best_score:.2f}")

# Get all results sorted by metric
all_results = optimizer.get_all_results()
for params, score in all_results[:5]:
    print(f"  {params}: {score:.2f}")
```

### Random Search

```python
from trading.training import RandomSearchOptimizer, ParameterSpec

optimizer = RandomSearchOptimizer(
    strategy_class=RSIStrategy,
    param_specs=[
        ParameterSpec("period", range(5, 30)),
        ParameterSpec("oversold", range(20, 40)),
        ParameterSpec("overbought", range(60, 80)),
    ],
    n_iter=100,  # Try 100 random combinations
    metric="total_return",
    random_state=42,
)

best_params, best_return = optimizer.optimize(bars)
```

### Multi-Metric Optimization

```python
def optimize_multiple_metrics(bars, strategy_class, param_specs):
    """Find parameters that work well across multiple metrics."""
    
    metrics = ["sharpe_ratio", "total_return", "max_drawdown"]
    results = {}
    
    for metric in metrics:
        optimizer = GridSearchOptimizer(strategy_class, param_specs, metric)
        best_params, score = optimizer.optimize(bars)
        results[metric] = (best_params, score)
    
    return results

# Find best parameters for each metric
all_best = optimize_multiple_metrics(bars, MovingAverageCrossoverStrategy, param_specs)
for metric, (params, score) in all_best.items():
    print(f"{metric}: {params} = {score:.4f}")
```

---

## Reinforcement Learning

### Training an RL Agent

```python
from trading.rl import TradingEnv, TradingEnvConfig
from trading.rl.features import OHLCVFeatures, TechnicalFeatures, CombinedFeatures
from trading.rl.rewards import RiskAdjustedReward

# Create feature extractor
features = CombinedFeatures([
    OHLCVFeatures(lookback=20),
    TechnicalFeatures(),
])

# Create reward function
reward_fn = RiskAdjustedReward(risk_free_rate=0.0, window=20)

# Configure environment
config = TradingEnvConfig(
    symbol="SPY",
    initial_balance=10000,
    trade_quantity=10,
    max_steps=None,  # Use all data
)

# Create environment
env = TradingEnv(bars, config, feature_extractor=features, reward_function=reward_fn)

# Train with stable-baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback

# Create evaluation environment
eval_env = TradingEnv(test_bars, config, feature_extractor=features)

# Train
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/",
    eval_freq=5000,
)

model.learn(total_timesteps=100_000, callback=eval_callback)
```

### Using Trained Model for Trading

```python
from trading.rl import RLStrategy

# Load trained model
model = PPO.load("./logs/best_model/best_model.zip")

# Create strategy wrapper
strategy = RLStrategy(
    model=model,
    symbol="SPY",
    trade_quantity=10,
    feature_extractor=features,  # Must match training
)

# Backtest
result = Backtest(bars, strategy).run()
print(f"RL Strategy Return: {result.metrics.total_return:.2%}")
```

### Custom Observation Space

```python
from trading.rl.features import FeatureExtractor
import numpy as np

class CustomFeatures(FeatureExtractor):
    """Custom feature extractor with market regime detection."""
    
    @property
    def num_features(self) -> int:
        return 10
    
    def extract(self, bars, current_idx, account) -> np.ndarray:
        if current_idx < 50:
            return np.zeros(self.num_features, dtype=np.float32)
        
        recent_bars = bars[current_idx-50:current_idx]
        closes = [b.close for b in recent_bars]
        
        # Calculate features
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        volatility = np.std(closes[-20:])
        
        current_price = closes[-1]
        return np.array([
            (current_price - sma_20) / sma_20,  # Distance from SMA20
            (current_price - sma_50) / sma_50,  # Distance from SMA50
            (sma_20 - sma_50) / sma_50,         # MA spread
            volatility / current_price,          # Normalized volatility
            # ... more features
        ], dtype=np.float32)
```

---

## Paper Trading

### Starting Paper Trading Session

```python
from trading.paper import PaperTradingEngine, PaperTradingConfig, LiveQuoteSource

# Configure paper trading
config = PaperTradingConfig(
    account_name="test_account",
    initial_balance=10000,
    symbols=["SPY", "QQQ"],
    poll_interval=5,  # 5 seconds between quote updates
)

# Create quote source and strategy
quote_source = LiveQuoteSource()
strategy = MovingAverageCrossoverStrategy(10, 30)

# Create engine
engine = PaperTradingEngine(config, strategy, quote_source)

# Run (blocking, press Ctrl+C to stop)
engine.run()
```

### Paper Trading with Custom Logic

```python
from trading.paper import PaperTradingEngine, MockQuoteSource

class PaperTradingCallback:
    """Custom callbacks for paper trading events."""
    
    def on_quote(self, symbol: str, price: float):
        print(f"Quote: {symbol} @ ${price:.2f}")
    
    def on_order(self, order):
        print(f"Order: {order.side} {order.quantity} {order.symbol}")
    
    def on_fill(self, execution):
        print(f"Filled: {execution.side} {execution.quantity} @ ${execution.price:.2f}")

# Use mock quotes for testing
quotes = MockQuoteSource({"SPY": 450.0, "QQQ": 380.0})
callback = PaperTradingCallback()

engine = PaperTradingEngine(config, strategy, quotes, callbacks=[callback])
```

---

## Commission and Slippage

### Modeling Trading Costs

```python
# Add realistic costs to backtest
result = Backtest(
    bars,
    strategy,
    initial_balance=10000,
    commission_per_trade=1.0,    # $1 per trade
    slippage_pct=0.001,          # 0.1% slippage
).run()

print(f"Return (with costs): {result.metrics.total_return:.2%}")

# Compare with zero-cost backtest
result_no_costs = Backtest(bars, strategy, initial_balance=10000).run()
print(f"Return (no costs):   {result_no_costs.metrics.total_return:.2%}")

cost_impact = result_no_costs.metrics.total_return - result.metrics.total_return
print(f"Cost impact:         {cost_impact:.2%}")
```

### Analyzing Trade Costs

```python
# Get execution details with costs
for execution in result.executions[:10]:
    print(f"{execution.timestamp}: {execution.side} {execution.quantity} "
          f"@ ${execution.price:.2f}, "
          f"commission=${execution.commission:.2f}, "
          f"slippage={execution.slippage_pct:.4%}")

total_commission = sum(e.commission for e in result.executions)
print(f"\nTotal commission paid: ${total_commission:.2f}")
```

---

## Portfolio Allocation

### Equal Weight Allocation

```python
from trading.training.portfolio import EqualWeightAllocation

# Allocate equally across symbols
allocator = EqualWeightAllocation()
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

weights = allocator.allocate(symbols, account)
print(weights)  # {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
```

### Momentum-Based Allocation

```python
from trading.training.portfolio import MomentumAllocation

# Allocate based on recent momentum
allocator = MomentumAllocation(lookback=60, top_n=3)
weights = allocator.allocate(symbols, account, price_history=bars)

# Top 3 performers get the allocation
```

### Inverse Volatility Allocation

```python
from trading.training.portfolio import InverseVolatilityAllocation

# Less volatile stocks get more weight
allocator = InverseVolatilityAllocation(lookback=20)
weights = allocator.allocate(symbols, account, price_history=bars)
```

### Custom Allocation Strategy

```python
from trading.training.portfolio import AllocationStrategy

class SectorRotationAllocation(AllocationStrategy):
    """Rotate allocation based on sector momentum."""
    
    def __init__(self, sector_map: dict):
        self.sector_map = sector_map  # symbol -> sector
    
    def allocate(self, symbols, account, **kwargs):
        # Calculate sector momentum
        sector_momentum = {}
        for symbol in symbols:
            sector = self.sector_map.get(symbol, "other")
            # ... calculate momentum
        
        # Weight by sector momentum
        weights = {}
        total_momentum = sum(sector_momentum.values())
        for symbol in symbols:
            sector = self.sector_map.get(symbol, "other")
            weights[symbol] = sector_momentum[sector] / total_momentum
        
        return weights
```

---

## Complete Trading System Example

```python
"""
Complete example: Multi-symbol trading system with optimization,
position sizing, and comprehensive analysis.
"""

from datetime import datetime, timedelta
from trading.data.sources import YahooDataSource
from trading.training.backtest import Backtest
from trading.training import WalkForwardValidator, WalkForwardConfig
from trading.training import GridSearchOptimizer, ParameterSpec
from trading.training.portfolio import MomentumAllocation
from trading.strategies.sizing import VolatilityAdjustedSizer
from trading.strategies.examples import MovingAverageCrossoverStrategy

def run_complete_system():
    # 1. Fetch data
    source = YahooDataSource()
    symbols = ["SPY", "QQQ", "IWM", "DIA"]
    
    all_bars = []
    for symbol in symbols:
        bars = source.fetch(symbol, "2020-01-01", "2024-01-01", "1d")
        all_bars.extend(bars)
    all_bars.sort(key=lambda b: b.timestamp)
    
    # 2. Split into train/test
    split_date = datetime(2023, 1, 1)
    train_bars = [b for b in all_bars if b.timestamp < split_date]
    test_bars = [b for b in all_bars if b.timestamp >= split_date]
    
    # 3. Optimize strategy parameters
    optimizer = GridSearchOptimizer(
        strategy_class=MovingAverageCrossoverStrategy,
        param_specs=[
            ParameterSpec("short_period", [5, 10, 15, 20]),
            ParameterSpec("long_period", [30, 40, 50, 60]),
        ],
        metric="sharpe_ratio",
    )
    best_params, _ = optimizer.optimize(train_bars)
    print(f"Optimal parameters: {best_params}")
    
    # 4. Walk-forward validation
    config = WalkForwardConfig(train_periods=126, test_periods=21, step_size=21)
    strategy = MovingAverageCrossoverStrategy(**best_params)
    
    validator = WalkForwardValidator(train_bars, config)
    wf_result = validator.validate(strategy)
    print(f"Walk-forward consistency: {wf_result.consistency_ratio:.2%}")
    
    # 5. Final out-of-sample test
    sizer = VolatilityAdjustedSizer(target_risk=0.02)
    
    result = Backtest(
        test_bars,
        MovingAverageCrossoverStrategy(**best_params),
        initial_balance=100000,
        position_sizer=sizer,
        commission_per_trade=1.0,
        slippage_pct=0.001,
    ).run()
    
    print(f"""
    Final Out-of-Sample Results
    ===========================
    Total Return:    {result.metrics.total_return:>8.2%}
    Sharpe Ratio:    {result.metrics.sharpe_ratio:>8.2f}
    Max Drawdown:    {result.metrics.max_drawdown:>8.2%}
    Win Rate:        {result.metrics.win_rate:>8.2%}
    Number of Trades:{result.metrics.num_trades:>8d}
    """)
    
    return result

if __name__ == "__main__":
    run_complete_system()
```

