# Stock Market Trading Bot – High-Level Architecture

## 1. Purpose and Scope
This document outlines the high-level architecture of a stock market trading bot focused initially on a training environment. The system will simulate or connect to market data, execute strategies, and manage accounts whose balances and clearing behavior can be configured for experimentation and learning.

The primary intended user is a single developer/researcher, running the system privately. The system should support both short exploratory experiments and long-running processes that may remain active for days, weeks, or months.

**Architecture Decision**: The system uses a **hybrid Rust + Python architecture**:
- **Rust Core**: High-performance components (data processing, account management, execution engine, metrics computation)
- **Python Layer**: Strategy development, ML/RL integration, CLI, and configuration management
- **Integration**: Rust functions are exposed to Python via PyO3 bindings, allowing Python strategies to call Rust core functions

This hybrid approach provides performance where needed (backtesting large datasets) while maintaining flexibility for strategy development and ML/RL experimentation.

## 2. Core System Capabilities
At a high level, the trading bot should provide:

- **Market data access**: Consume real-time or historical price and volume data for stocks, with at least 5-minute granularity so that multiple trades per day are possible.
- **Strategy execution**: Evaluate configurable trading strategies against incoming data and generate orders.
- **Order lifecycle management**: Represent orders, trades, and their states within the system.
- **Training account management**: Maintain a primary training account (with the option to extend to multiple accounts later), with configurable starting balances and clearing rules.
- **Risk and exposure controls**: Enforce constraints aligned with a typical retail personal account (e.g., limited leverage, no institution-only products).
- **Monitoring and reporting**: Provide visibility into account balances, positions, and performance over time, including standard day-trading evaluation metrics (e.g., returns, drawdowns, risk-adjusted metrics).

## 3. High-Level Component Overview

### 3.1 Market Data and Dataset Layer
Responsible for providing a unified stream or interface of price and market information to the rest of the system, as well as managing locally stored datasets for repeated training cycles.

Key responsibilities (conceptual):
- Source market data from different origins:
  - **Historical data** (e.g., downloaded once at the beginning and reused across many runs) at a granularity sufficient for multiple trades per day (e.g., 5-minute bars or finer).
  - **Live or near-real-time data** (for forward-testing or paper trading scenarios).
  - **Synthetic data** generated to resemble real markets for benchmarking and stress testing.
- Normalize all data into a consistent internal format (e.g., timestamps, symbols, OHLCV, corporate actions if needed).
- Provide subscription or polling interfaces for other components (e.g., strategies, simulators, backtest engines).
- Coordinate access to locally stored datasets so that training runs can be replayed efficiently without repeatedly fetching external data.

### 3.2 Local Data Storage and Management
To support repeated training cycles and offline experimentation, the system should support local storage of market datasets, assuming total data sizes on the order of several gigabytes (up to roughly 10 GB) are acceptable.

High-level responsibilities (without specifying technology):
- Define a logical structure for datasets (e.g., by symbol, date range, and data source type such as historical vs. synthetic).
- Support an initial data aggregation step that can pull, clean, and persist data for later reuse.
- Expose primarily read-only views of datasets to training and simulation components, to support reproducible experiments when desired, even if strict determinism is not a hard requirement.
- Track simple metadata about datasets (e.g., date coverage, symbols included, data source, and version or generation parameters for synthetic data).
- For synthetic datasets, associate generation configuration and seeds so that synthetic series can be reproduced when needed.

### 3.3 Strategy Engine and Strategy Orchestration
The strategy engine consumes market data and account state, then decides when to place orders. It also needs to manage multiple strategies and allow switching or combining them over time.

High-level responsibilities:
- Host one or more pluggable trading strategies, each encapsulating its own decision logic. Classic, simple strategies (e.g., basic trend-following) may be provided as built-ins, but the primary aim is to support **configurable strategies** defined by rules and parameters rather than hard-coded decision logic.
- Expose a simple interface for strategies to express desired actions (e.g., "buy X shares of Y").
- Enforce basic constraints before submitting actions to the order management layer (e.g., avoid duplicate orders, respect simple throttling rules, respect capital and opportunity-cost constraints).
- Provide a **strategy orchestration** capability that can:
  - Evaluate multiple strategies at each time slice to determine which are currently "viable" according to their rules and constraints.
  - Select which strategy (or set of strategies) is active at a given time or under certain conditions.
  - Support rotating between strategies across training cycles or based on market regime, account performance, or other signals.
  - Allow blending or weighting of outputs from multiple strategies, where appropriate, while still appearing as a coherent source of orders to the rest of the system.
- Treat capital, clearing delays, and other resource constraints as shared across strategies so that taking one opportunity may preclude others, reflecting realistic opportunity cost.

### 3.4 Order and Execution Layer
This layer represents the lifecycle of an order from creation to completion or cancellation.

High-level responsibilities:
- Accept order requests from the strategy engine (regardless of which underlying strategy generated them).
- Route orders to either:
  - An external broker/exchange interface; or
  - An internal simulator/execution model for training and backtesting.
- Track order states (e.g., pending, partially filled, filled, canceled).
- Emit trade events when executions occur.

### 3.5 Training and Simulation Engine
To support many training cycles over historical and synthetic data, the system benefits from an explicit training/simulation engine. Conceptually, a **training cycle** consists of feeding a dataset to the system as if it were arriving in real time, allowing strategies to act, and then evaluating the outcomes.

High-level responsibilities:
- Coordinate the playback of historical datasets and the generation/streaming of synthetic datasets into the strategy engine, stepping forward in time as though the data were live.
- Manage the lifecycle of a training run (e.g., initialization of the training account, selection of strategy or strategy mix, iteration over the dataset, and finalization/reporting).
- Support reinforcement-learning-style workflows, where the strategy logic may learn or adapt based on rewards derived from trading performance over a run.
- Allow multiple runs over the same dataset with different configurations (e.g., different strategies, clearing delays, or starting balances) for comparison.
- Emit artifacts and summaries that can be used for benchmarking performance across strategies and data regimes, especially standard day-trading metrics (returns, drawdowns, risk-adjusted ratios, and trade-level statistics).

### 3.6 Paper Trading Engine

The paper trading engine enables forward-testing strategies against live market data using simulated accounts. Unlike backtesting (which replays historical data), paper trading operates in real-time, making actual buy/sell decisions as markets move.

**Key characteristics:**
- **Real-time execution**: Strategies receive live quotes and make decisions in real-time
- **Simulated fills**: Orders are executed at current market prices without real money
- **Persistent accounts**: Account state survives between sessions
- **Same strategies**: Identical strategy code works in both backtest and paper modes

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    PaperTradingEngine                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │ LiveQuote    │    │   Strategy   │    │ execute_    │  │
│   │ Source       │ →  │   decide()   │ →  │ orders      │  │
│   │ (polling)    │    │              │    │ (Rust)      │  │
│   └──────────────┘    └──────────────┘    └─────────────┘  │
│          │                   │                   │          │
│          ▼                   ▼                   ▼          │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                  Account (Rust)                      │  │
│   │  • cleared_balance, pending_balance                  │  │
│   │  • positions: {symbol: (quantity, cost_basis)}       │  │
│   │  • pending_transactions                              │  │
│   │  • Persisted to ~/.trading/paper/{account_id}/       │  │
│   └─────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                  Order Log                           │  │
│   │  • All orders with timestamps                        │  │
│   │  • Execution history                                 │  │
│   │  • Stored for audit/analysis                         │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**High-level responsibilities:**
- **Live data acquisition**: Poll or stream current market prices at configurable intervals
- **Strategy execution**: Run the same strategy interface used in backtesting against live data
- **Order management**: Track order lifecycle (submitted → filled/rejected)
- **Account persistence**: Save account state to disk, load on restart
- **Session management**: Start/stop trading sessions, handle market hours
- **Execution logging**: Record all orders and fills for later analysis

**Data flow:**
1. Engine starts, loads persisted account state (or creates new account)
2. At each tick interval (e.g., every 5 seconds during market hours):
   - Fetch current quotes for watched symbols
   - Build an `AnalysisSnapshot` with current prices and account state
   - Call `strategy.decide()` to get order requests
   - Apply risk constraints (`apply_risk_constraints`)
   - Execute accepted orders (`execute_orders`) at current prices
   - Update and persist account state
3. On shutdown, save final account state

**Relationship to backtesting:**
The paper trading engine reuses core components from the backtesting system:
- Same `Strategy` base class and strategy implementations
- Same `execute_orders` Rust function for order fills
- Same `apply_risk_constraints` for risk management
- Same `Account` type for state management
- Same metrics computation for performance tracking

This ensures that a strategy validated in backtesting behaves identically in paper trading.

## 4. Training Account System
The training account system is a central feature in this project. It represents the primary training account, its balances, and the effect of trades over time, with a focus on configurable clearing behavior and realistic retail constraints.

### 4.1 Account Model (Conceptual)
Each training account conceptually has:
- **Identifier**: A unique ID or name.
- **Base currency**: The currency in which balances and P&L are expressed.
- **Configurable starting balance**: Initial cash or equity value for training.
- **Positions**: Holdings in various instruments (e.g., per-symbol quantities and cost basis).
- **Available vs. pending balances**: Distinguish immediately usable funds from funds subject to clearing delays.

This level focuses on what information the account must represent, not how it is stored.

### 4.2 Clearing Time Configuration
Transactions (trades) do not necessarily settle instantly in real markets. For training, we want configurable clearing behavior to simulate different environments.

Key concepts:
- **Clearing delay setting**: The training account (or environment) can specify a time interval between trade execution and the moment funds/positions are fully cleared, with a default aligned to typical retail behavior (e.g., clearing on the next business day) but configurable for experimentation.
- **Pending balances**: After a trade, changes to cash and positions may initially be marked as "pending" for the duration of the clearing delay.
- **Cleared balances**: Once the clearing delay elapses, pending changes are incorporated into the cleared balance.

High-level behavior:
- When a transaction is completed (e.g., a trade is executed), the account records:
  - The trade details (instrument, quantity, price, side).
  - The immediate impact on pending cash and positions.
  - A scheduled clearing time based on the configured delay.
- A background process or periodic evaluation transitions pending amounts to cleared amounts when their clearing time has passed.

### 4.3 Balance Inquiry
The system should provide a clear, consistent way to inquire about an account’s balance for training and analysis.

At a high level, the account interface should support:
- **Current cleared balance**: What funds are fully available for new trades under current rules.
- **Current pending amounts**: Funds or P&L that are in the clearing window and not yet fully available.
- **Total equity view**: A roll-up that includes cleared balance, pending amounts, and the mark-to-market value of open positions.

These views should be consumable by:
- Strategies (for making decisions based on available capital and risk).
- Monitoring/reporting tools (for dashboards or logs).
- Evaluation tools (for analyzing training runs).

### 4.4 Interaction with Orders and Trades
High-level interactions between the training account system and order/execution layer:
- When an order is placed, the account system may reserve or check sufficient cleared balance to support the order.
- When a trade is executed:
  - The account system records the trade and updates pending balances/positions based on the configured clearing rules.
  - The system schedules when these changes become cleared.
- If an order is canceled, any temporary reservations (if modeled) may be released.

## 5. Supporting Services and Cross-Cutting Concerns

### 5.1 Configuration and Environment Profiles
The system should support configuration profiles for different training scenarios, for example:
- Different starting balances for the training account.
- Different clearing delays (e.g., near-instant for rapid iteration, or a default of next-business-day retail-style clearing).
- Different market data sources (sandbox vs. live vs. historical replay).
- Different strategy configurations or strategy sets used for a given training run.

### 5.2 Logging, Monitoring, and Metrics
High-level expectations:
- Capture events such as order submissions, executions, balance changes, and clearing transitions.
- Provide metrics around account performance, drawdowns, turnover, and utilization of available capital.
- Compute common evaluation metrics relevant to day trading (e.g., cumulative and per-period returns, max drawdown, volatility, Sharpe/Sortino-like ratios, and trade-level statistics such as win rate and average win/loss).

### 5.3 Security and Access Control (Conceptual)
Even in a training environment, it is useful to consider:
- Basic user or role concepts controlling who can access which training accounts.
- Separation of concerns between "operators" (who configure environments) and "strategies" (which operate within constraints).

## 6. Future Extensions and Roadmap

This section outlines planned extensions and future directions, organized by priority and implementation phase.

### 6.1 Near-Term Priorities

#### Live/Paper Trading Integration
Connect the backtesting engine to real broker APIs for paper trading and eventually live trading:
- **Alpaca API**: Commission-free trading with excellent API for paper trading
- **Interactive Brokers**: Comprehensive market access for serious traders
- **Architecture considerations**:
  - Abstract broker interface allowing strategy code to work identically in backtest and live modes
  - Order state synchronization between local state and broker
  - Connection resilience and retry logic
  - Rate limiting and API quota management

#### Strategy Optimization
Automated hyperparameter tuning and strategy discovery:
- **Grid search**: Exhaustive parameter space exploration
- **Random search**: Efficient sampling for high-dimensional spaces
- **Bayesian optimization**: Sample-efficient optimization for expensive evaluations
- **Walk-forward optimization**: Rolling window validation to prevent overfitting
- **Multi-objective optimization**: Balance return, drawdown, and other metrics

#### Event-Driven Architecture
For real-time streaming data and low-latency execution:
- **Event bus**: Central pub/sub system for market data, order events, and signals
- **Async processing**: Non-blocking handlers for concurrent data streams
- **WebSocket integration**: Real-time market data from exchanges
- **Latency monitoring**: Track and optimize event processing times

### 6.2 Medium-Term Extensions

#### Portfolio-Level Allocation
Managing multiple strategies as a unified portfolio:
- **Capital allocation**: Dynamic allocation across strategies based on performance
- **Correlation analysis**: Identify and manage strategy correlation
- **Risk budgeting**: Allocate risk (not just capital) across strategies
- **Rebalancing**: Automatic portfolio rebalancing on schedule or triggers

#### Advanced Execution Modeling
More realistic simulation of trade execution:
- **Slippage models**: Market impact based on order size and liquidity
- **Commission structures**: Per-share, per-trade, and tiered commission models
- **Partial fills**: Simulate large orders filling over multiple bars
- **Queue position**: Model order book dynamics for limit orders

#### ML/RL Integration
Deep integration with machine learning workflows:
- **Feature engineering pipeline**: Technical indicators, market regime detection
- **Model training hooks**: Integration with PyTorch, TensorFlow, scikit-learn
- **Reinforcement learning**: Gym-compatible environment for RL agents
- **Online learning**: Strategies that adapt during live trading

### 6.3 Long-Term Vision

#### Multi-Asset Support
Extend beyond equities:
- **Options**: Greeks, expiration handling, exercise/assignment
- **Futures**: Contract specifications, rollover, margin requirements
- **Crypto**: 24/7 markets, exchange-specific APIs
- **Forex**: Currency pair conventions, pip calculations

#### Distributed Training
Scale training across multiple machines:
- **Parameter sweep parallelization**: Run many backtests concurrently
- **Distributed data storage**: Handle datasets larger than single-machine memory
- **Cloud integration**: AWS/GCP/Azure deployment for elastic compute

#### Advanced Risk Management
Institutional-grade risk controls:
- **Value at Risk (VaR)**: Daily VaR limits and monitoring
- **Stress testing**: Scenario analysis and Monte Carlo simulation
- **Correlation stress**: Portfolio behavior under correlation breakdown
- **Regulatory compliance**: Pattern day trader rules, wash sale detection

### 6.4 Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Data pipeline (fetch/normalize/store) | ✅ Complete | Rust core + Python sources |
| Backtesting engine | ✅ Complete | Strategies, execution, metrics |
| Risk constraints | ✅ Complete | Position size, leverage limits |
| Run management | ✅ Complete | Storage, checkpoints, resume |
| Multi-strategy comparison | ✅ Complete | Parallel runs, rankings |
| Position sizing strategies | ✅ Complete | 6 sizing algorithms |
| Advanced metrics | ✅ Complete | Sortino, win rate, expectancy |
| Live/paper trading | 🔲 Planned | Next priority |
| Strategy optimization | 🔲 Planned | Grid search first |
| Event-driven architecture | 🔲 Planned | For real-time data |
| Portfolio allocation | 🔲 Planned | After live trading |
| ML/RL integration | 🔲 Planned | Gym environment |

These extensions can be layered on top of the current architecture as the project evolves, maintaining backward compatibility with existing strategies and configurations.

## 7. Interaction Model and Interfaces

### 7.1 Overall Approach
The system is a **hybrid Rust + Python library** exposing core concepts such as datasets, strategies, training runs, and accounts. On top of this library, a **command-line interface (CLI)** provides the main day-to-day interaction surface.

**Language Split**:
- **Rust**: Data processing, storage, account management, execution engine, metrics computation (performance-critical paths)
- **Python**: Strategy logic, ML/RL, CLI, configuration parsing, event logging (flexibility layer)
- **Integration**: Rust functions are exposed via PyO3 as `trading._core` module, callable from Python

The CLI is a thin layer that:
- Parses command-line arguments and experiment configuration files (Python).
- Constructs library objects (datasets, strategies, training runs) from configuration (Python).
- Invokes well-defined library entrypoints to perform actions (Python calls Rust core functions).

This separation allows:
- Automated and repeatable experiments via CLI and config files.
- Direct use of the same core functionality from Python for debugging, custom workflows, or notebook-based analysis.
- High performance for data-intensive operations (Rust) while maintaining flexibility for strategy development (Python).

### 7.2 Library-Level Concepts (High-Level)
At the library level, the following conceptual objects are expected:
- **Dataset / DataSource abstractions**: encapsulate access to historical, live, and synthetic data, including the analysis vs. tradable universes.
- **Strategy base class and concrete strategies**: Python classes that implement strategy behavior with configurable parameters (e.g., risk tolerance, max trades per day, skill-level-like controls), including hooks for reinforcement-learning-style updates.
- **TrainingRun / SimulationRun**: orchestrates stepping through a dataset as if in real time, coordinating strategy decisions, order submission, account updates, and metrics collection.
- **Account / Risk model**: enforces retail-style constraints, clearing delays, and opportunity-cost-aware capital usage.
- **Metrics and reporting utilities**: compute standard day-trading metrics and provide structured outputs for inspection.

The CLI commands will map directly onto these objects and their responsibilities.

### 7.3 Configuration-Driven Experiments
Experiments are described in configuration files (e.g., YAML or similar), which specify:
- Datasets to use (historical, synthetic, or mixed), including symbol universes.
- Strategy class to instantiate (by fully qualified Python name) and its parameters.
- Account settings (starting balance, clearing delay, risk constraints).
- Any additional run-level settings (e.g., random seeds for synthetic data, logging verbosity, output locations).

The CLI reads these configurations, resolves the appropriate Python classes and parameters, and invokes the underlying library to execute the specified actions. This design supports both **batch-style experimentation via CLI** and **fine-grained control via direct Python API usage** without duplicating core logic.
