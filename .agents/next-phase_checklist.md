# Next Phase Implementation Checklist

**Created**: 2024-12-16  
**Objective**: Extend PoC into production-ready training system with CLI commands, proper orchestration, checkpointing, and logging.

## Current Status Summary

### ✅ Completed
- Type definitions (Pydantic migration) - 28 tests
- Exception definitions - 4 custom exceptions
- Data normalization (Rust) - `normalize_bars`, timezone handling
- Data validation (Rust) - `validate_bars`, `detect_data_gaps`, `fill_data_gaps`
- Data storage (Rust) - `store_dataset`, `load_dataset`, `list_datasets`, `dataset_exists`, `read_dataset_metadata`
- Data sources (Python) - `YahooDataSource`, `LocalDataSource`, `CSVDataSource`
- Account helpers (Rust, partial) - `calculate_account_equity`, `process_clearing`, `is_business_day`, `next_business_day`, `reserve_funds`, `release_reservation`
- Execution (Rust, partial) - `execute_orders`
- Strategy base (Python) - `Strategy` ABC with `decide` method
- Example strategies - `BuyAndHoldStrategy`, `MovingAverageCrossoverStrategy`, `MeanReversionStrategy`, `RSIStrategy`, `RandomStrategy`
- Backtest engine (Python) - `Backtest`, `BacktestResult`
- Metrics (Rust, partial) - `compute_run_metrics` (basic version)
- CLI (Python, partial) - `backtest`, `compare` commands
- **130 tests passing**

---

## Phase 1: CLI Commands for Data Management

### 1.1 fetch-data Command
- [ ] [CRITICAL] Create `src/trading/commands/__init__.py` module
- [ ] [CRITICAL] Create `src/trading/commands/fetch_data.py`
- [ ] Implement `load_fetch_data_config()` - parse YAML config
- [ ] Add `trading fetch-data` CLI command to `cli.py`
- [ ] Add tests for `load_fetch_data_config()`
- [ ] Add integration test: fetch real data with config file

### 1.2 gen-synth Command (Synthetic Data)
- [ ] Add `rand` crate to `rust/Cargo.toml`
- [ ] Implement `generate_synth_bars()` in Rust - geometric Brownian motion generator
- [ ] Create `src/trading/data/synthetic.py` - Python config wrapper
- [ ] Implement `build_synth_generator()` - create generator from config
- [ ] Create `src/trading/commands/gen_synth.py`
- [ ] Implement `load_gen_synth_config()` - parse YAML config
- [ ] Add `trading gen-synth` CLI command
- [ ] Add tests for synthetic data generation

---

## Phase 2: Training Orchestration Enhancement

### 2.1 Dataset Loading
- [ ] [CRITICAL] Implement `load_datasets()` in Rust - load multiple datasets into DatasetBundle
- [ ] Implement `DatasetBundle.get_bars_at(timestamp)` method
- [ ] Implement `DatasetBundle.get_symbol_history(symbol, end_time, lookback)` method
- [ ] Add tests for dataset bundle operations

### 2.2 Strategy Registry
- [ ] Create `src/trading/strategies/registry.py`
- [ ] Implement `resolve_strategy_class()` - resolve class path to Strategy class
- [ ] Implement `instantiate_strategy()` - create strategy from config
- [ ] Add tests for strategy resolution

### 2.3 Iteration Scheduler
- [ ] Implement `iteration_schedule()` in Rust - generate TimeSlice sequence
- [ ] Handle variable sampling rates across symbols
- [ ] Handle forward-fill of missing bars
- [ ] Add tests for iteration schedule generation

### 2.4 Analysis Snapshot Builder
- [ ] Create `src/trading/training/snapshot.py`
- [ ] Implement `build_analysis_snapshot()` - filter bars by universe, handle staleness
- [ ] Add tests for snapshot building

### 2.5 Strategy Executor
- [ ] Create `src/trading/training/strategy_executor.py`
- [ ] Implement `strategy_decide()` - invoke strategy with error handling
- [ ] Add tests for strategy execution

---

## Phase 3: Risk Management

### 3.1 Risk Constraints
- [ ] [CRITICAL] Implement `apply_risk_constraints()` in Rust
- [ ] Check cleared balance for buy orders
- [ ] Check position for sell orders
- [ ] Check max_position_size constraint
- [ ] Check max_leverage constraint
- [ ] Create `src/trading/risk/__init__.py`
- [ ] Create `src/trading/risk/constraints.py` - Python wrapper
- [ ] Add tests for risk constraint application

---

## Phase 4: Full Order Lifecycle

### 4.1 Order Management
- [ ] Implement `create_order()` in Rust - create order with unique ID
- [ ] Implement `cancel_order()` in Rust - cancel pending order
- [ ] Add OrderStatus enum to types if not present
- [ ] Add Order type to track order state
- [ ] Update `execute_orders()` to use full order lifecycle
- [ ] Add tests for order creation and cancellation

### 4.2 Account Updates
- [ ] Implement `initialize_account()` in Rust - create account from config
- [ ] Implement `update_account_from_executions()` in Rust - apply executions to account
- [ ] Add tests for account initialization and updates

---

## Phase 5: Persistence and Checkpointing

### 5.1 Run Results Storage
- [ ] [CRITICAL] Implement `store_run_results()` in Rust - persist run to ~/.trading/runs/
- [ ] Write run_state.json
- [ ] Write metrics.json
- [ ] Write config.json
- [ ] Write steps.json
- [ ] Write executions.json
- [ ] Add tests for run storage

### 5.2 Checkpointing
- [ ] Implement `checkpoint_run_state()` in Rust - save checkpoint atomically
- [ ] Implement `resume_training_run()` in Rust - load latest checkpoint
- [ ] Implement checkpoint rotation (keep last N)
- [ ] Add tests for checkpointing and resume

---

## Phase 6: Logging and Monitoring

### 6.1 Event Logging
- [ ] Create `src/trading/training/logging.py`
- [ ] Implement `log_event()` - structured event logging
- [ ] Define event types (order, execution, clearing, etc.)
- [ ] Add tests for event logging

### 6.2 Run Monitoring
- [ ] Create `src/trading/training/monitoring.py`
- [ ] Implement `get_run_progress()` - get current run status
- [ ] Add tests for run progress

### 6.3 Recording
- [ ] Create `src/trading/training/recording.py`
- [ ] Implement `record_step()` - record step data for analysis
- [ ] Add tests for step recording

---

## Phase 7: Full Training Command

### 7.1 Configuration
- [ ] [CRITICAL] Create `src/trading/commands/run_training.py`
- [ ] Implement `load_training_config()` - parse YAML training config
- [ ] Implement `validate_training_config()` - validate config completeness

### 7.2 Training Loop
- [ ] Implement `execute_training_run()` - main training orchestration
- [ ] Integrate all components (dataset loading, iteration, strategy, execution, etc.)
- [ ] Add checkpoint support during training
- [ ] Add event logging integration
- [ ] Add `trading run-training` CLI command

### 7.3 Run Inspection
- [ ] Create `src/trading/commands/inspect_run.py`
- [ ] Implement `load_run_results()` - load completed run
- [ ] Implement `format_run_summary()` - format for display
- [ ] Add `trading inspect-run` CLI command

---

## Phase 8: Enhanced Metrics

### 8.1 Advanced Metrics
- [ ] Extend `compute_run_metrics()` to accept full RunState
- [ ] Implement win_rate calculation from executions
- [ ] Implement Sortino ratio
- [ ] Implement trade-level statistics (avg win, avg loss, profit factor)
- [ ] Add tests for advanced metrics

---

## Resources

### Documentation
- `/Users/Shared/git/trading/docs/cli-and-implementation.md` - Full spec (2805 lines)
- `/Users/Shared/git/trading/docs/architecture.md` - High-level architecture
- `/Users/Shared/git/trading/docs/design-analysis.md` - Design decisions

### Existing Code to Reference
- `src/trading/training/backtest.py` - PoC backtest engine (can be extended)
- `src/trading/strategies/base.py` - Strategy ABC
- `src/trading/data/sources.py` - Data source implementations
- `rust/src/lib.rs` - Rust core functions

### External Libraries
- PyO3 docs: https://pyo3.rs/v0.27.2/
- Polars docs: https://pola.rs/
- Pydantic docs: https://docs.pydantic.dev/latest/
- yfinance: https://pypi.org/project/yfinance/

---

## Implementation Priority

**Recommended order for next work session:**

1. **Phase 1.1**: `fetch-data` command - enables reproducible data collection
2. **Phase 3**: Risk constraints - critical for realistic simulation
3. **Phase 5.1**: Run storage - enables inspection and analysis
4. **Phase 7**: Full training command - production training capability
5. **Phase 2**: Training orchestration - fills gaps in current backtest
6. **Phase 6**: Logging/monitoring - observability
7. **Phase 1.2**: Synthetic data - for testing/benchmarking
8. **Phase 8**: Advanced metrics - polish

---

## Notes

- The PoC backtest engine (`training/backtest.py`) can serve as the foundation for the full training orchestration
- Consider extracting common logic from Backtest class into reusable components
- Risk constraints should be integrated into the execution flow
- Checkpointing is essential for long-running training sessions



