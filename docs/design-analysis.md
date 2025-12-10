# Design Analysis: Architecture vs Implementation Alignment

This document applies a hermeneutic circle analysis to evaluate the alignment between `architecture.md` (the whole/vision) and `cli-and-implementation.md` (the parts/details), identifying gaps and proposing refinements.

## Executive Summary

The implementation specification is **well-planned and comprehensive** for an initial MVP, but several architectural concepts from the high-level design are deferred or simplified. The gaps are intentional simplifications for initial implementation, but should be documented as known limitations and future work.

## 1. Whole-to-Parts Analysis: Architecture Vision

The architecture document envisions:
- A flexible, extensible trading bot system
- Support for multiple data sources (historical, synthetic, live)
- Strategy orchestration (multiple strategies, switching, blending)
- Realistic account simulation with clearing delays
- Comprehensive metrics and monitoring
- Library-first design with CLI wrapper

## 2. Parts-to-Whole Analysis: Implementation Reality

The implementation spec provides:
- Detailed function-level specifications for 4 CLI commands
- Complete type system with Pydantic models
- Storage layer design
- Account management with clearing delays
- Basic strategy interface
- Metrics computation

## 3. Identified Gaps and Refinements

### 3.1 Strategy Orchestration (Major Gap)

**Architecture Says**: "Strategy orchestration capability that can evaluate multiple strategies, select active strategies, rotate between strategies, and blend outputs."

**Implementation Has**: Single strategy per run, no orchestration layer.

**Impact**: Medium - This is a core architectural feature that's completely missing.

**Recommendation**: 
- **For MVP**: Document as "Future Enhancement" and add a note that the current design supports only single-strategy runs.
- **For Future**: Add `StrategyOrchestrator` class and `MultiStrategyConfig` type. Functions needed:
  - `evaluate_strategy_viability(strategy, snapshot, account) -> bool`
  - `select_active_strategies(strategies, snapshot, account) -> list[Strategy]`
  - `blend_order_requests(order_lists, weights) -> list[OrderRequest]`

### 3.2 Order Lifecycle Management (Partial Gap)

**Architecture Says**: "Track order states (pending, partially filled, filled, canceled). Route orders to external broker or internal simulator."

**Implementation Has**: Only market orders, immediate execution, no order states, no cancellation.

**Impact**: Low for MVP (simulation doesn't need complex order management), but limits extensibility.

**Recommendation**:
- Add `Order` type with `status: OrderStatus` enum (PENDING, PARTIALLY_FILLED, FILLED, CANCELED)
- Add `OrderManager` class to track orders between request and execution
- Add `cancel_order(order_id: str)` function (even if just marks as canceled in simulation)
- Document that partial fills are not supported in initial implementation

### 3.3 Analysis vs Tradable Universes (Missing)

**Architecture Says**: "Including analysis vs tradable universes" - strategies can analyze more symbols than they can trade.

**Implementation Has**: No distinction - all symbols in dataset are both analyzable and tradable.

**Impact**: Low for MVP, but important for realistic strategy development.

**Recommendation**:
- Add to `TrainingConfig`: `analysis_universe: list[Symbol]` and `tradable_universe: list[Symbol]`
- Modify `build_analysis_snapshot` to include bars from analysis_universe
- Modify `apply_risk_constraints` to reject orders for symbols not in tradable_universe
- Default: if not specified, use all symbols from datasets

### 3.4 Business Day Handling for Clearing (Missing)

**Architecture Says**: "Clearing on the next business day" - implies handling weekends/holidays.

**Implementation Has**: Simple time-based clearing delay (24 hours = next day, doesn't account for weekends/holidays).

**Impact**: Low for MVP (training data may not include weekends anyway), but inaccurate for realistic simulation.

**Recommendation**:
- Add `business_days` module with `is_business_day(date: datetime) -> bool` function
- Modify `process_clearing` to advance to next business day instead of adding hours
- Use `pandas.bdate_range` or similar for business day calculations
- Document current behavior as "calendar days" vs future "business days" enhancement

### 3.5 Order Reservations (Missing)

**Architecture Says**: "When an order is placed, the account system may reserve or check sufficient cleared balance."

**Implementation Has**: No reservation system - checks balance at execution time only.

**Impact**: Low for MVP (simulation executes immediately), but could cause issues if order queuing is added.

**Recommendation**:
- Add `reserved_balance: float` field to `Account` model
- Add `reserve_funds(amount: float)` and `release_reservation(amount: float)` functions
- Modify `apply_risk_constraints` to reserve funds when order passes validation
- Modify `execute_orders` to convert reservation to pending balance
- Document as "Future Enhancement" if not implementing initially

### 3.6 Live/Real-Time Data Support (Deferred)

**Architecture Says**: "Live or near-real-time data (for forward-testing or paper trading scenarios)."

**Implementation Has**: Only historical and synthetic data sources.

**Impact**: Low for MVP (focus is on backtesting), but limits forward-testing capability.

**Recommendation**:
- Document as "Phase 2" feature
- Add placeholder `LiveDataSource` protocol in data sources
- Note that `fetch-data` command currently only supports historical/synthetic

### 3.7 Reinforcement Learning Hooks (Underspecified)

**Architecture Says**: "Support reinforcement-learning-style workflows, where strategy logic may learn or adapt based on rewards."

**Implementation Has**: Strategy interface exists but no RL-specific hooks or reward signals.

**Impact**: Medium - RL is mentioned but not actionable.

**Recommendation**:
- Add `Strategy` abstract method: `update_from_reward(reward: float, context: dict) -> None`
- Add `RewardSignal` type to encapsulate reward computation
- Add `compute_reward(execution: Execution, account_before: Account, account_after: Account) -> float`
- Document that RL support is optional - strategies can implement `update_from_reward` as no-op

### 3.8 Logging and Monitoring (Underspecified)

**Architecture Says**: "Capture events such as order submissions, executions, balance changes, and clearing transitions. Provide metrics around account performance."

**Implementation Has**: Basic logging level config, but no structured event logging or monitoring infrastructure.

**Impact**: Medium - Hard to debug and monitor long-running training runs.

**Recommendation**:
- Add `EventLogger` class with methods: `log_order_submission`, `log_execution`, `log_balance_change`, `log_clearing_transition`
- Add structured logging format (JSON) for easy parsing
- Add `TrainingRunMonitor` class that can be queried for current state during long runs
- Specify log file locations: `~/.trading/runs/{run_id}/events.log`

### 3.9 Multiple Accounts Support (Simplified)

**Architecture Says**: "Primary training account (with option to extend to multiple accounts later)."

**Implementation Has**: Single account per run, no multi-account support.

**Impact**: Low - Architecture explicitly says this is future work.

**Recommendation**:
- Document current limitation clearly
- Design `Account` to be account-agnostic (already is)
- Note that multi-account would require `AccountManager` and changes to `TrainingConfig`

### 3.10 DatasetBundle Type (Underspecified)

**Architecture Says**: "Dataset / DataSource abstractions encapsulate access to historical, live, and synthetic data."

**Implementation Has**: `DatasetBundle` type mentioned but not fully defined.

**Impact**: High - This is a core type that needs specification.

**Recommendation**:
- Add complete `DatasetBundle` specification to types section:
```python
class DatasetBundle(BaseModel):
    """Container for multiple datasets providing unified access."""
    datasets: dict[DatasetId, DatasetMetadata]
    bars_by_timestamp: dict[datetime, dict[Symbol, NormalizedBar]]  # Or use efficient lookup structure
    
    def get_bars_at(self, timestamp: datetime) -> dict[Symbol, NormalizedBar]:
        """Get all bars available at a specific timestamp."""
        pass
    
    def get_symbol_history(self, symbol: Symbol, end_time: datetime, lookback_periods: int) -> list[NormalizedBar]:
        """Get historical bars for a symbol (for lookback strategies)."""
        pass
```

### 3.11 Error Recovery and Resilience (Missing)

**Architecture Says**: System should support "long-running processes that may remain active for days, weeks, or months."

**Implementation Has**: No checkpoint/resume capability, no error recovery.

**Impact**: High for long-running processes.

**Recommendation**:
- Add `checkpoint_run_state(run_state: RunState) -> None` function
- Add `resume_training_run(run_id: RunId) -> RunState` function
- Modify `run_training_command` to checkpoint periodically (e.g., every N time slices)
- Add `--resume` flag to `run-training` command
- Store checkpoints in `~/.trading/runs/{run_id}/checkpoints/`

### 3.12 Configuration Validation (Partial)

**Architecture Says**: Configuration-driven experiments with validation.

**Implementation Has**: Basic validation in config loaders, but no comprehensive validation.

**Impact**: Medium - Invalid configs could cause runtime errors.

**Recommendation**:
- Add Pydantic validators to all config models (e.g., `@field_validator` for date ranges, symbol lists)
- Add `validate_training_config(config: TrainingConfig) -> list[str]` returning validation errors
- Validate that datasets exist before starting training run
- Validate that strategy class path is importable
- Validate that date ranges are valid (start < end, not in future for historical data)

### 3.13 Metrics Extensibility (Limited)

**Architecture Says**: "Standard day-trading evaluation metrics (returns, drawdowns, risk-adjusted metrics, trade-level statistics)."

**Implementation Has**: Basic metrics (total_return, max_drawdown, volatility, sharpe_ratio, num_trades, win_rate).

**Impact**: Low - Core metrics covered, but missing some advanced metrics.

**Recommendation**:
- Add to `RunMetrics`: `sortino_ratio`, `calmar_ratio`, `average_win`, `average_loss`, `profit_factor`, `max_consecutive_wins`, `max_consecutive_losses`
- Add `compute_advanced_metrics(run_state: RunState) -> AdvancedMetrics` function
- Make metrics computation pluggable (list of metric calculators)

### 3.14 Data Quality and Gap Handling (Underspecified)

**Architecture Says**: Data normalization and validation.

**Implementation Has**: Basic validation (price sanity checks, timestamp ordering), but no gap handling.

**Impact**: Medium - Real market data has gaps (holidays, after-hours, etc.).

**Recommendation**:
- Add `detect_data_gaps(bars: Iterator[NormalizedBar], expected_granularity: str) -> list[Gap]`
- Add `fill_data_gaps(bars: Iterator[NormalizedBar], method: str) -> Iterator[NormalizedBar]` (forward-fill, backward-fill, interpolate)
- Document gap handling strategy in `validate_bars` function
- Add config option: `gap_handling: str` ("skip", "forward_fill", "error")

### 3.15 Time Zone Handling (Underspecified)

**Architecture Says**: Timestamps with timezone info.

**Implementation Has**: Mentions UTC default but no explicit timezone handling strategy.

**Impact**: Medium - Market data often in exchange timezone, needs conversion.

**Recommendation**:
- Document timezone strategy: All timestamps stored in UTC, convert on input/output
- Add `normalize_timezone(timestamp: datetime, source_tz: str) -> datetime` function
- Add timezone field to `DatasetMetadata`: `timezone: str = "UTC"`
- Validate that all bars have timezone-aware timestamps

## 4. Design Refinements Needed

### 4.1 Add Missing Types

Add to `trading/types.py`:

```python
class OrderStatus(str, Enum):
    """Status of an order in the system."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"

class Order(BaseModel):
    """Order with lifecycle tracking."""
    order_id: str
    symbol: Symbol
    side: str
    quantity: float
    order_type: str
    status: OrderStatus
    created_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: float | None = None

class DatasetBundle(BaseModel):
    """Container for multiple datasets."""
    # See recommendation 3.10 for full specification

class Gap(BaseModel):
    """Represents a gap in market data."""
    symbol: Symbol
    start_time: datetime
    end_time: datetime
    expected_bars: int
```

### 4.2 Add Missing Functions

Add to implementation spec:

**Data Quality**:
- `detect_data_gaps(bars: Iterator[NormalizedBar], granularity: str) -> list[Gap]`
- `fill_data_gaps(bars: Iterator[NormalizedBar], method: str) -> Iterator[NormalizedBar]`
- `normalize_timezone(timestamp: datetime, source_tz: str) -> datetime`

**Order Management**:
- `create_order(order_request: OrderRequest) -> Order`
- `cancel_order(order_id: str) -> None`
- `get_order_status(order_id: str) -> OrderStatus`

**Account Management**:
- `reserve_funds(account: Account, amount: float) -> Account`
- `release_reservation(account: Account, amount: float) -> Account`
- `is_business_day(date: datetime) -> bool` (for future business day clearing)

**Run Management**:
- `checkpoint_run_state(run_state: RunState) -> None`
- `resume_training_run(run_id: RunId) -> RunState`

**Configuration**:
- `validate_training_config(config: TrainingConfig) -> list[str]`
- `validate_dataset_exists(dataset_id: DatasetId) -> bool`

**Monitoring**:
- `log_event(event_type: str, data: dict) -> None`
- `get_run_progress(run_id: RunId) -> RunProgress`

### 4.3 Update Existing Functions

**`apply_risk_constraints`**: Add check for tradable_universe if specified in config.

**`build_analysis_snapshot`**: Add support for analysis_universe vs tradable_universe distinction.

**`process_clearing`**: Document current calendar-day behavior, note future business-day enhancement.

**`store_dataset`**: Add timezone field to metadata.

**`load_training_config`**: Add validation that datasets exist, strategy class is importable.

## 5. Documentation Additions Needed

### 5.1 Known Limitations Section

Add to `cli-and-implementation.md`:

```markdown
## 14. Known Limitations and Future Enhancements

### Current Limitations (MVP)
- Single strategy per training run (no orchestration)
- Market orders only (no limit orders, stop orders)
- Immediate execution (no order queuing or partial fills)
- Calendar-day clearing (not business-day aware)
- No order cancellation support
- No checkpoint/resume for long-running runs
- No live data source support
- All symbols are both analyzable and tradable (no universe distinction)

### Planned Enhancements (Post-MVP)
- Strategy orchestration and multi-strategy support
- Order lifecycle management with states
- Business day-aware clearing
- Analysis vs tradable universe distinction
- Checkpoint/resume capability
- Live data source integration
- Advanced metrics (Sortino, Calmar, etc.)
- Reinforcement learning hooks
- Structured event logging and monitoring
```

### 5.2 Architecture Alignment Notes

Add cross-references between architecture.md and implementation.md showing:
- What's implemented vs. what's deferred
- Design decisions that differ from architecture (with rationale)
- Future work items mapped to architectural components

## 6. Overall Assessment

### Strengths
1. **Comprehensive function-level detail** - Each function is well-specified
2. **Type system** - Complete Pydantic models with validation
3. **Storage design** - Clear data persistence strategy
4. **Account simulation** - Realistic clearing delay model (even if simplified)
5. **Metrics** - Core trading metrics covered
6. **Extensibility** - Design allows for future enhancements

### Weaknesses
1. **Strategy orchestration** - Core architectural feature missing
2. **Order lifecycle** - Simplified to immediate execution only
3. **Data quality** - Gap handling and timezone management underspecified
4. **Resilience** - No checkpoint/resume for long runs
5. **Monitoring** - Logging infrastructure not detailed
6. **Configuration validation** - Could be more comprehensive

### Recommendation

**The project is well-planned for an MVP**, but should:
1. **Document limitations clearly** - Add "Known Limitations" section
2. **Add missing type definitions** - DatasetBundle, Order, OrderStatus, Gap
3. **Enhance configuration validation** - Add comprehensive validators
4. **Add checkpoint/resume** - Critical for long-running processes
5. **Specify data quality handling** - Gap detection and timezone strategy
6. **Plan for Phase 2** - Strategy orchestration, order lifecycle, business days

The implementation spec is **ready for development** with the understanding that some architectural features are intentionally deferred. The gaps are manageable and don't prevent building a functional MVP.

## 7. Priority Refinements

### High Priority (Before MVP)
1. Add `DatasetBundle` complete specification
2. Add comprehensive configuration validation
3. Add checkpoint/resume capability
4. Document timezone handling strategy
5. Add "Known Limitations" section

### Medium Priority (Post-MVP)
1. Strategy orchestration layer
2. Order lifecycle management
3. Analysis vs tradable universe distinction
4. Business day-aware clearing
5. Structured event logging

### Low Priority (Future)
1. Live data source support
2. Multiple accounts
3. Advanced metrics
4. Reinforcement learning hooks
5. Order reservations

---

This analysis shows that while the implementation spec is comprehensive and well-structured, it intentionally simplifies some architectural concepts for the MVP. The gaps are well-defined and can be addressed incrementally without requiring major architectural changes.

