# Theory of Operation

This document explains the internal architecture and data flow of the trading bot system.

## Overview

The trading bot is a **hybrid Rust + Python** system designed for:

- **Backtesting**: Simulating trading strategies on historical data
- **Paper Trading**: Forward-testing with live market data (simulated execution)
- **Strategy Development**: Creating and optimizing trading algorithms
- **ML/RL Integration**: Training machine learning models on market data

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Trading Bot Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Python    │    │    Rust     │    │   Storage   │                │
│   │   Layer     │ ←→ │    Core     │ ←→ │   Layer     │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                                          │
│   • CLI & Config     • Data Processing   • Parquet I/O                  │
│   • Strategy Logic   • Execution Engine  • Run Artifacts                │
│   • ML/RL           • Metrics Compute   • Checkpoints                  │
│   • Data Sources    • Account Mgmt                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Pipeline

The data pipeline handles fetching, normalizing, validating, and storing market data.

#### Data Flow

```
External API (Yahoo, etc.)
         │
         ▼
┌─────────────────────┐
│   Data Source       │  Python: fetch_bars()
│   (YahooDataSource) │
└─────────────────────┘
         │
         ▼ Iterator[Bar]
┌─────────────────────┐
│   Normalization     │  Rust: normalize_bars()
│   - Timezone        │  - Ensure UTC timestamps
│   - Field types     │  - Validate OHLCV format
└─────────────────────┘
         │
         ▼ Iterator[NormalizedBar]
┌─────────────────────┐
│   Validation        │  Rust: validate_bars()
│   - Price sanity    │  - high >= low, etc.
│   - Chronological   │  - Timestamps increasing
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Storage           │  Rust: store_dataset()
│   (Parquet format)  │  ~/.trading/datasets/
└─────────────────────┘
```

#### Key Design Decisions

1. **Streaming with Iterators**: Data flows as iterators to handle large datasets without loading everything into memory.

2. **Rust for Performance**: Normalization, validation, and Parquet I/O are implemented in Rust for speed. A year of minute-level data for multiple symbols can be millions of rows.

3. **Python for Flexibility**: External API integration stays in Python because:
   - APIs change frequently
   - Authentication varies
   - Easy to add new sources

### 2. Backtest Engine

The backtest engine simulates trading on historical data.

#### Backtest Loop

```python
def run_backtest(bars, strategy, account):
    """Simplified backtest loop."""
    
    # Group bars by timestamp
    organized_bars = organize_bars_by_timestamp(bars)
    
    for timestamp, current_bars in organized_bars:
        # 1. Build market snapshot
        snapshot = AnalysisSnapshot(
            timestamp=timestamp,
            bars=current_bars,
            account=account,
        )
        
        # 2. Get strategy decisions
        order_requests = strategy.decide(snapshot)
        
        # 3. Apply risk constraints (Rust)
        accepted, rejected = apply_risk_constraints(
            order_requests, 
            account,
            max_position_size,
            max_leverage,
        )
        
        # 4. Execute orders (Rust)
        executions = execute_orders(
            accepted,
            current_bars,
            timestamp,
            commission_per_trade,
            slippage_pct,
        )
        
        # 5. Update account (Rust)
        account = update_account(account, executions, current_bars)
        
        # 6. Record for metrics
        equity_history.append(calculate_equity(account, current_bars))
    
    # Compute final metrics (Rust)
    metrics = compute_run_metrics(equity_history, executions)
    
    return BacktestResult(metrics, executions, equity_history)
```

#### Time Handling

The backtest processes data chronologically:

1. **Bar Organization**: Bars are grouped by timestamp. At each timestamp, the engine sees all bars available at that moment.

2. **No Look-Ahead Bias**: Strategies only receive data up to the current timestamp. Future bars are not visible.

3. **Execution at Bar Close**: Market orders execute at the closing price of the current bar. This is a simplified model that assumes:
   - Orders can be filled at the close price
   - No partial fills
   - No order book dynamics

### 3. Strategy Interface

Strategies are Python classes that implement a simple interface:

```python
class Strategy(ABC):
    @abstractmethod
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        """Generate trading decisions based on current market state."""
        pass
```

#### Strategy Anatomy

```python
class ExampleStrategy(Strategy):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.price_history: dict[Symbol, list[float]] = {}
    
    def decide(self, snapshot: AnalysisSnapshot) -> list[OrderRequest]:
        orders = []
        
        for symbol, bar in snapshot.bars.items():
            # 1. Update internal state
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(bar.close)
            
            # 2. Check if we have enough data
            if len(self.price_history[symbol]) < self.lookback:
                continue
            
            # 3. Calculate indicator
            avg = sum(self.price_history[symbol][-self.lookback:]) / self.lookback
            
            # 4. Check current position
            position = snapshot.account.positions.get(symbol)
            has_position = position and position.quantity > 0
            
            # 5. Generate orders based on signal
            if bar.close > avg * 1.02 and not has_position:
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="buy",
                    quantity=10,
                ))
            elif bar.close < avg * 0.98 and has_position:
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                ))
        
        return orders
```

#### Key Points

- **Stateful**: Strategies can maintain state (price history, indicators, etc.)
- **Symbol-Agnostic**: The same strategy can handle multiple symbols
- **Account-Aware**: Strategies see current positions and balances
- **Return Orders**: Strategies return order requests, not direct trades

### 4. Execution Engine

The execution engine (Rust) converts order requests into executions.

```
OrderRequest                 Execution
┌─────────────┐             ┌─────────────┐
│ symbol      │             │ symbol      │
│ side        │  execute_   │ side        │
│ quantity    │ ────────→   │ quantity    │
│ order_type  │  orders()   │ price       │
└─────────────┘             │ timestamp   │
                            │ commission  │
                            │ slippage    │
                            │ order_id    │
                            └─────────────┘
```

#### Execution Logic

```rust
fn execute_orders(
    orders: Vec<OrderRequest>,
    bars: HashMap<Symbol, Bar>,
    timestamp: DateTime,
    commission: f64,
    slippage_pct: f64,
) -> Vec<Execution> {
    let mut executions = Vec::new();
    
    for order in orders {
        // Get current price
        let bar = match bars.get(&order.symbol) {
            Some(b) => b,
            None => continue,  // Skip if no price data
        };
        
        // Apply slippage
        let slippage_mult = if order.side == "buy" {
            1.0 + slippage_pct
        } else {
            1.0 - slippage_pct
        };
        let execution_price = bar.close * slippage_mult;
        
        executions.push(Execution {
            symbol: order.symbol,
            side: order.side,
            quantity: order.quantity,
            price: execution_price,
            timestamp: timestamp,
            commission: commission,
            slippage_pct: slippage_pct,
            order_id: generate_uuid(),
        });
    }
    
    executions
}
```

### 5. Account Management

The account system tracks cash, positions, and pending transactions.

```
┌─────────────────────────────────────────────────────────┐
│                      Account                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   cleared_balance:  $8,500.00   ← Available for trading │
│   pending_balance:    $500.00   ← Awaiting settlement   │
│   reserved_balance:   $200.00   ← Reserved for orders   │
│                                                          │
│   positions:                                             │
│   ┌─────────┬──────────┬────────────┬─────────────┐    │
│   │ Symbol  │ Quantity │ Cost Basis │ Market Value│    │
│   ├─────────┼──────────┼────────────┼─────────────┤    │
│   │ AAPL    │   10     │  $150.00   │  $1,750.00  │    │
│   │ GOOGL   │   5      │  $140.00   │    $725.00  │    │
│   └─────────┴──────────┴────────────┴─────────────┘    │
│                                                          │
│   Total Equity: $11,675.00                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Balance Types

- **cleared_balance**: Cash that is immediately available for trading
- **pending_balance**: Cash from recent sales awaiting settlement (T+1 or T+2)
- **reserved_balance**: Cash reserved for pending buy orders

#### Position Tracking

Each position tracks:
- **quantity**: Number of shares held
- **cost_basis**: Average price paid per share (for P&L calculation)
- **pending_quantity**: Shares from pending transactions

#### Clearing Simulation

The system can simulate settlement delays:

```python
# Configure clearing delay
config = TrainingConfig(
    clearing_delay_hours=24,    # 24 hours
    use_business_days=False,    # Calendar days
)

# Or business day clearing (T+1)
config = TrainingConfig(
    clearing_delay_hours=24,
    use_business_days=True,     # Skip weekends/holidays
)
```

### 6. Risk Constraints

Risk constraints validate orders before execution.

```
                    apply_risk_constraints()
                    
OrderRequests  ──────────────────────────────→  Accepted Orders
                         │                       
                         │ Rejected Orders       
                         ▼                       
                    ┌─────────────┐             
                    │ Validation  │             
                    │ - Position  │             
                    │ - Leverage  │             
                    │ - Balance   │             
                    └─────────────┘             
```

#### Constraints Checked

1. **Sufficient Balance**: Buy orders require adequate cleared balance
2. **Position Size Limit**: Configurable maximum position value
3. **Leverage Limit**: Maximum leverage ratio (default: 1.0 = no margin)
4. **Existing Position**: Sell orders require sufficient shares

### 7. Metrics Computation

Metrics are computed in Rust for performance on large datasets.

#### Equity Curve

```
Equity ($)
    │
10K │    ╭──────╮
    │   ╱        ╲         ╭───────
    │  ╱          ╲       ╱
    │ ╱            ╲     ╱
 9K │╱              ╲   ╱
    │                ╲_╱
    │                  ↑
    │              Drawdown
    └────────────────────────────→ Time
```

#### Computed Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Total Return | (Final - Initial) / Initial | Overall P&L |
| Max Drawdown | max(peak - trough) / peak | Worst decline |
| Volatility | std(daily_returns) | Return variability |
| Sharpe Ratio | (return - rf) / volatility × √252 | Risk-adjusted return |
| Sortino Ratio | (return - rf) / downside_vol × √252 | Downside risk-adjusted |
| Win Rate | winning_trades / total_trades | Trade success rate |
| Profit Factor | gross_profit / gross_loss | Profitability ratio |
| Expectancy | (win_rate × avg_win) - (loss_rate × avg_loss) | Expected value per trade |

### 8. Position Sizing

Position sizers determine trade quantities dynamically.

```
                     Position Sizer
OrderRequest  ──────────────────────────→  Sized OrderRequest
(quantity=0)          │                    (quantity=15)
                      │
            ┌─────────┴─────────┐
            │ Calculation:      │
            │ - Account equity  │
            │ - Current price   │
            │ - Volatility      │
            │ - Risk parameters │
            └───────────────────┘
```

#### Sizing Strategies

| Strategy | Logic |
|----------|-------|
| FixedQuantity | Always trade N shares |
| FixedDollar | Always trade $X worth |
| PercentOfEquity | Trade X% of portfolio |
| RiskPercent | Risk X% per trade with stop loss |
| KellyCriterion | Optimal betting fraction |
| VolatilityAdjusted | Size inversely to volatility |

### 9. Walk-Forward Validation

Walk-forward testing prevents overfitting by validating on out-of-sample data.

```
Data Timeline
├────────────────────────────────────────────────────────────┤

Window 1:
├─── Train ───┤── Test ──┤

Window 2:
      ├─── Train ───┤── Test ──┤

Window 3:
            ├─── Train ───┤── Test ──┤

Window 4:
                  ├─── Train ───┤── Test ──┤
```

#### Process

1. **Divide data** into rolling train/test windows
2. **Optimize** strategy on each training period
3. **Validate** on corresponding test period
4. **Aggregate** out-of-sample results
5. **Compare** in-sample vs out-of-sample performance

#### Key Metrics

- **Consistency Ratio**: % of windows where OOS return is positive
- **Performance Degradation**: How much OOS underperforms IS
- **Stability**: Variance of OOS returns across windows

### 10. Reinforcement Learning Integration

The system provides a Gymnasium-compatible environment for RL.

```
┌─────────────────────────────────────────────────────────────┐
│                     TradingEnv                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │ Observation │     │   Action    │     │   Reward    │  │
│   │   Space     │     │   Space     │     │  Function   │  │
│   ├─────────────┤     ├─────────────┤     ├─────────────┤  │
│   │ Box(n,)     │     │ Discrete(3) │     │ Configurable│  │
│   │ - OHLCV     │     │ 0: Hold     │     │ - Return    │  │
│   │ - Technical │     │ 1: Buy      │     │ - Sharpe    │  │
│   │ - Account   │     │ 2: Sell     │     │ - Drawdown  │  │
│   └─────────────┘     └─────────────┘     └─────────────┘  │
│                                                              │
│   step(action) → (observation, reward, done, truncated, info)│
│   reset()      → (observation, info)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Episode Flow

```python
env = TradingEnv(bars, config)
obs, info = env.reset()

while True:
    action = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        break
```

#### Feature Extractors

Transform raw market data into RL observations:

- **OHLCVFeatures**: Normalized price/volume history
- **TechnicalFeatures**: RSI, moving averages, momentum
- **AccountFeatures**: Position ratio, cash ratio, unrealized P&L
- **CombinedFeatures**: Combine multiple extractors

#### Reward Functions

- **SimpleReturn**: Percentage change in equity
- **RiskAdjusted**: Return / rolling volatility
- **SharpeReward**: Differential Sharpe ratio
- **DrawdownPenalty**: Return with penalty for drawdowns

### 11. Paper Trading Engine

Paper trading runs strategies on live market data with simulated execution.

```
                    PaperTradingEngine
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ LiveQuoteSource│  │   Strategy    │  │   Account     │
│               │  │               │  │  (Persisted)  │
│ - Poll quotes │  │ - Same as    │  │               │
│ - Real-time   │  │   backtest   │  │ ~/.trading/   │
│   prices      │  │               │  │   paper/      │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌─────────────┐
                    │ Trade Loop  │
                    │ (every N s) │
                    └─────────────┘
```

#### Key Differences from Backtesting

| Aspect | Backtest | Paper Trading |
|--------|----------|---------------|
| Data | Historical | Live quotes |
| Speed | Fast (simulated time) | Real-time |
| Account | In-memory | Persisted to disk |
| Sessions | Single run | Multiple sessions |
| Execution | At bar close | At current quote |

### 12. Storage Architecture

All data is stored under `~/.trading/`:

```
~/.trading/
├── datasets/                    # Historical market data
│   └── {dataset_id}/
│       ├── bars.parquet        # OHLCV data (columnar)
│       └── metadata.json       # Dataset info
│
├── runs/                        # Backtest results
│   └── {run_id}/
│       ├── config.json         # Run configuration
│       ├── metrics.json        # Computed metrics
│       ├── executions.json     # Trade history
│       ├── equity_history.json # Equity curve
│       └── checkpoints/        # Resume points
│           └── checkpoint_{ts}.json
│
└── paper/                       # Paper trading state
    └── {account_name}/
        ├── account.json        # Account state
        └── orders.json         # Order history
```

#### Storage Formats

- **Parquet**: Efficient columnar format for bar data (Rust/Polars)
- **JSON**: Human-readable for configs and metrics (Rust/serde)

### 13. Error Handling

The system uses a hierarchy of exceptions:

```
TradingError
├── ConfigError          # Configuration issues
├── DataSourceError      # Data fetching problems
├── DataValidationError  # Invalid data
├── StorageError         # I/O failures
├── StrategyError        # Strategy execution issues
└── AccountError         # Account operation failures
```

#### Design Principles

1. **Fail Fast**: Invalid configurations raise immediately
2. **Graceful Degradation**: Missing data is skipped with warnings
3. **Recoverable State**: Checkpoints allow resuming failed runs
4. **Detailed Logging**: All operations are logged for debugging

### 14. Performance Considerations

#### Rust for Hot Paths

The following are implemented in Rust for performance:

- Data normalization and validation
- Parquet I/O
- Order execution
- Account updates
- Metrics computation
- Risk constraint checking

#### Python for Flexibility

These stay in Python:

- Strategy logic (enables ML/RL)
- External API integration
- CLI and configuration
- High-level orchestration

#### Memory Management

- **Streaming**: Data flows as iterators, not loaded all at once
- **Chunked Processing**: Large datasets processed in chunks
- **Efficient Types**: Rust uses stack-allocated types where possible

### 15. Extensibility

#### Adding a New Strategy

```python
# src/trading/strategies/my_strategy.py
from trading.strategies.base import Strategy

class MyStrategy(Strategy):
    def decide(self, snapshot):
        # Your logic here
        return orders
```

#### Adding a New Data Source

```python
# src/trading/data/sources.py
class MyDataSource:
    def fetch(self, symbol, start, end, granularity):
        # Fetch from your source
        return bars
```

#### Adding a New Metric

```rust
// rust/src/lib.rs
fn compute_custom_metric(equity_history: &[f64]) -> f64 {
    // Your calculation
}
```

### 16. Testing Philosophy

- **Unit Tests**: Each function tested in isolation
- **Integration Tests**: End-to-end backtests on known data
- **Property Tests**: Invariants checked (e.g., equity never negative)
- **Benchmark Tests**: Performance regression tracking

Current test coverage: **328+ tests passing**

