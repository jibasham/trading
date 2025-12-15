# CLI and Implementation Specificationpip 

This document provides a complete, function-by-function specification for implementing the trading bot system. Each function is designed to be implementable by a low-parameter-count AI agent working in isolation, given the type definitions and context provided here.

## 1. Design Principles for Implementation

**Hybrid Architecture**: This project uses Rust for performance-critical components and Python for flexibility and ML/RL integration.

- **Library first, CLI second**: Core behavior lives in Rust/Python library; the CLI is a thin Python wrapper
- **Small work items**: Implementation is decomposed into very small steps; ideally, each work item corresponds to implementing or modifying a **single function**
- **Clear responsibilities**: Each function should have a single clear responsibility with well-defined inputs and outputs
- **Config-driven behavior**: CLI commands largely translate configuration files and arguments into calls into the library
- **Type safety**: 
  - **Python**: Type hints with Pydantic models for validation and serialization
  - **Rust**: Strong type system with serde for serialization, PyO3 for Python bindings
- **Error handling**: 
  - **Python**: Domain-specific exceptions (defined in `trading.exceptions`)
  - **Rust**: Result types with custom error types, converted to Python exceptions via PyO3
- **Storage design**: Local datasets are stored in `~/.trading/datasets/` directory. Run artifacts are stored in `~/.trading/runs/` directory. Metadata is stored as JSON files alongside data.

**Language Assignment**:
- **[RUST]**: Data processing (normalization, validation, resampling), storage (Parquet I/O), account management, execution engine, metrics computation, business day calculations
- **[PYTHON]**: Configuration parsing, data source integration (external APIs), strategy logic, CLI, event logging, training orchestration

Each function specification below is marked with **[RUST]** or **[PYTHON]** to indicate implementation language.

## 2. Core Type Definitions

All types are defined in `trading/types.py`. The following types are already defined:
- `Symbol`: NewType wrapping str
- `DatasetId`: NewType wrapping str
- `RunId`: NewType wrapping str
- `DateRange`: Pydantic BaseModel with `start: datetime` and `end: datetime`
- `Bar`: Pydantic BaseModel with symbol, timestamp, open, high, low, close, volume
- `NormalizedBar`: Pydantic BaseModel inheriting from Bar
- `RunMetrics`: Pydantic BaseModel with run_id, total_return, max_drawdown, volatility, sharpe_ratio, num_trades, win_rate

**Design Decision**: All data models use Pydantic BaseModel instead of dataclasses. This provides:
- Automatic validation of field types and constraints
- Built-in JSON serialization/deserialization via `.model_dump()` and `.model_validate()`
- Better error messages for invalid data
- Support for field validators and computed fields

Additional types needed (to be added to `trading/types.py`):

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from enum import Enum

class FetchDataConfig(BaseModel):
    """Configuration for fetching historical market data."""
    symbols: list[Symbol]
    date_range: DateRange
    granularity: str  # e.g., "5m", "1h", "1d"
    data_source: str  # e.g., "yahoo", "local", "csv"
    source_params: dict[str, Any]  # Provider-specific parameters
    dataset_id: DatasetId | None = None  # If None, auto-generate from params

class GenSynthConfig(BaseModel):
    """Configuration for generating synthetic market data."""
    symbols: list[Symbol]
    date_range: DateRange
    granularity: str
    generator_type: str  # e.g., "geometric_brownian", "mean_reverting"
    generator_params: dict[str, Any]  # Generator-specific parameters
    random_seed: int | None
    dataset_id: DatasetId | None = None

class TrainingConfig(BaseModel):
    """Configuration for a training/simulation run."""
    run_id: RunId | None = None  # If None, auto-generate
    datasets: list[DatasetId]
    strategy_class_path: str  # Fully qualified Python class path
    strategy_params: dict[str, Any]
    account_starting_balance: float
    account_base_currency: str = "USD"
    clearing_delay_hours: int = 24  # Default: 24 (next business day)
    use_business_days: bool = False  # If True, clearing uses business days instead of calendar days
    risk_max_position_size: float | None = None  # Max $ per position, None = no limit
    risk_max_leverage: float = 1.0  # Default: 1.0 (no leverage)
    analysis_universe: list[Symbol] | None = None  # Symbols for analysis (None = all symbols)
    tradable_universe: list[Symbol] | None = None  # Symbols that can be traded (None = all symbols)
    log_level: str = "INFO"
    checkpoint_interval: int | None = None  # Checkpoint every N time slices (None = no checkpoints)
    enable_event_logging: bool = True  # Enable structured event logging

class OrderRequest(BaseModel):
    """Request from strategy to place an order."""
    symbol: Symbol
    side: str  # "buy" or "sell"
    quantity: float  # Number of shares
    order_type: str = "market"  # "market" (only type supported initially)

class Execution(BaseModel):
    """Record of a completed trade execution."""
    symbol: Symbol
    side: str
    quantity: float
    price: float  # Execution price
    timestamp: datetime
    order_id: str  # Unique identifier for the originating order

class Position(BaseModel):
    """Current position in a symbol."""
    symbol: Symbol
    quantity: float  # Positive for long, negative for short
    cost_basis: float  # Average cost per share
    pending_quantity: float = 0.0  # Quantity pending clearing

class Account(BaseModel):
    """Training account state."""
    account_id: str
    base_currency: str
    cleared_balance: float  # Available funds
    pending_balance: float  # Funds pending clearing
    reserved_balance: float = 0.0  # Funds reserved for pending orders
    positions: dict[Symbol, Position] = Field(default_factory=dict)
    clearing_delay_hours: int
    use_business_days: bool = False  # If True, clearing uses business days

class TimeSlice(BaseModel):
    """A single time point in a dataset.
    
    Supports variable sampling rates - bars dict may contain bars from different
    timestamps (the most recent available bar for each symbol at or before this timestamp).
    Missing bars are simply not included in the dict.
    """
    timestamp: datetime
    bars: dict[Symbol, NormalizedBar] = Field(default_factory=dict)  # Most recent bars available at or before this timestamp
    bar_timestamps: dict[Symbol, datetime] = Field(default_factory=dict)  # Actual timestamp of each bar (may differ from TimeSlice.timestamp)

class AnalysisSnapshot(BaseModel):
    """Market snapshot provided to strategies."""
    timestamp: datetime
    bars: dict[Symbol, NormalizedBar]
    account: Account

class RunState(BaseModel):
    """State maintained during a training run."""
    run_id: RunId
    config: TrainingConfig
    account: Account
    time_slices: list[TimeSlice]
    executions: list[Execution]
    order_requests: list[OrderRequest]
    step_records: list[dict[str, Any]] = Field(default_factory=list)  # Per-step records for analysis

class DatasetMetadata(BaseModel):
    """Metadata about a stored dataset."""
    dataset_id: DatasetId
    symbols: list[Symbol]
    date_range: DateRange
    granularity: str
    data_source: str
    source_params: dict[str, Any]
    created_at: datetime
    bar_count: int

class RunArtifacts(BaseModel):
    """All artifacts from a completed training run."""
    run_id: RunId
    config: TrainingConfig
    metrics: RunMetrics
    run_state: RunState
    created_at: datetime

class InspectRunRequest(BaseModel):
    """Request parameters for inspect-run command."""
    run_id: RunId
    output_path: str | None = None

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
    """Container for multiple datasets providing unified access to bars.
    
    Supports variable sampling rates per symbol and mixed granularities.
    Bars are stored per-symbol with their own timestamps, allowing different
    symbols to have different resolutions and handle missing data gracefully.
    """
    datasets: dict[DatasetId, DatasetMetadata] = Field(default_factory=dict)
    # Internal storage: bars indexed by symbol then timestamp
    # Structure: dict[Symbol, list[NormalizedBar]] where each list is sorted by timestamp
    # This allows each symbol to have its own time resolution
    
    def get_bars_at(self, timestamp: datetime, tolerance: timedelta | None = None) -> dict[Symbol, NormalizedBar]:
        """Get bars available at or near a specific timestamp.
        
        For each symbol, returns the most recent bar at or before the timestamp.
        If tolerance is provided, only returns bars within tolerance of timestamp.
        """
        pass
    
    def get_latest_bar(self, symbol: Symbol, end_time: datetime) -> NormalizedBar | None:
        """Get the most recent bar for a symbol at or before end_time."""
        pass
    
    def get_symbol_history(
        self, 
        symbol: Symbol, 
        end_time: datetime, 
        lookback_periods: int
    ) -> list[NormalizedBar]:
        """Get historical bars for a symbol (for lookback strategies).
        
        Returns the last N bars for the symbol at or before end_time.
        Handles variable granularity - returns actual bars, not resampled.
        """
        pass
    
    def get_all_timestamps(self) -> list[datetime]:
        """Get all unique timestamps across all symbols, sorted chronologically.
        
        Useful for iteration schedule - includes timestamps from all symbols
        regardless of their individual granularities.
        """
        pass
    
    def get_symbol_timestamps(self, symbol: Symbol) -> list[datetime]:
        """Get all timestamps for a specific symbol, sorted chronologically."""
        pass

class Gap(BaseModel):
    """Represents a gap in market data."""
    symbol: Symbol
    start_time: datetime
    end_time: datetime
    expected_bars: int

class RewardSignal(BaseModel):
    """Reward signal for reinforcement learning."""
    reward: float
    timestamp: datetime
    execution_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

class RunProgress(BaseModel):
    """Current progress of a running training run."""
    run_id: RunId
    current_timestamp: datetime
    total_timestamps: int
    completed_timestamps: int
    current_account_equity: float
    num_executions: int
    elapsed_time_seconds: float

## 3. Exception Definitions

Create `trading/exceptions.py` with the following exceptions:

```python
class TradingError(Exception):
    """Base exception for all trading system errors."""
    pass

class ConfigError(TradingError):
    """Error in configuration file or parameters."""
    pass

class DataSourceError(TradingError):
    """Error accessing or processing data source."""
    pass

class ValidationError(TradingError):
    """Error validating data or state."""
    pass

class StorageError(TradingError):
    """Error reading from or writing to storage."""
    pass

class StrategyError(TradingError):
    """Error in strategy execution."""
    pass

class AccountError(TradingError):
    """Error in account operations (e.g., insufficient funds)."""
    pass
```

## 4. Command: `trading fetch-data`

### 4.1 High-Level Behavior

The `fetch-data` command aggregates and stores historical market data locally. It reads a YAML configuration file, fetches data from the specified source, normalizes and validates it, then stores it with metadata.

### 4.2 Configuration File Format

The configuration file is YAML with the following structure:

```yaml
symbols:
  - "QQQ"
  - "SPY"
date_range:
  start: "2020-01-01T00:00:00Z"
  end: "2024-01-01T00:00:00Z"
granularity: "5m"
data_source: "yahoo"  # or "local", "csv"
source_params:
  # Provider-specific parameters
dataset_id: "qqq_spy_5m_2020_2024"  # Optional, auto-generated if omitted
```

### 4.3 Function Specifications

#### Function: `load_fetch_data_config`

**Location**: `trading/commands/fetch_data.py`

**Signature**:
```python
def load_fetch_data_config(config_path: str) -> FetchDataConfig:
```

**Responsibility**: Parse and validate the YAML configuration file for `fetch-data` command.

**Input**:
- `config_path`: Path to YAML configuration file (relative or absolute)

**Output**:
- `FetchDataConfig`: Validated configuration object

**Behavior**:
1. Read the YAML file from `config_path`
2. Parse YAML content into a dictionary
3. Validate required fields: `symbols`, `date_range`, `granularity`, `data_source`
4. Convert `symbols` list to list of `Symbol` types
5. Parse `date_range.start` and `date_range.end` strings into `datetime` objects (assume UTC if no timezone specified)
6. Validate `granularity` is a recognized format (e.g., "1m", "5m", "15m", "1h", "1d")
7. Validate `data_source` is a recognized source type
8. If `dataset_id` is not provided, generate one using format: `{data_source}_{symbols}_{granularity}_{start_date}_{end_date}` (sanitized)
9. Return `FetchDataConfig` object

**Error Handling**:
- Raise `ConfigError` if file cannot be read
- Raise `ConfigError` if YAML is invalid
- Raise `ConfigError` if required fields are missing
- Raise `ConfigError` if date parsing fails
- Raise `ConfigError` if granularity or data_source is unrecognized

**Dependencies**:
- `yaml` module (PyYAML)
- `trading.types.FetchDataConfig`
- `trading.exceptions.ConfigError`

---

#### Function: `resolve_data_source` **[PYTHON]**

**Location**: `src/trading/data/sources.py`

**Signature**:
```python
def resolve_data_source(config: FetchDataConfig) -> DataSource:
```

**Responsibility**: Construct a data source abstraction from configuration.

**Input**:
- `config`: `FetchDataConfig` object

**Output**:
- `DataSource`: Protocol/ABC object that implements data fetching interface

**Behavior**:
1. Based on `config.data_source`:
   - If `"yahoo"`: Return `YahooDataSource(config.source_params)`
   - If `"local"`: Return `LocalDataSource(config.source_params)` (reads from local storage)
   - If `"csv"`: Return `CSVDataSource(config.source_params)` (reads from CSV file)
   - Otherwise: Raise `DataSourceError`
2. The returned `DataSource` object must implement a `fetch_bars` method (see below)

**Design Decision**: Use a Protocol/ABC for `DataSource` to allow different implementations. For initial implementation, create concrete classes for each source type.

**Error Handling**:
- Raise `DataSourceError` if data_source type is unrecognized
- Raise `DataSourceError` if source_params are invalid for the chosen source

**Dependencies**:
- `trading.types.FetchDataConfig`
- `trading.exceptions.DataSourceError`
- `trading.data.sources.YahooDataSource`, `LocalDataSource`, `CSVDataSource`

---

#### Function: `fetch_bars` **[PYTHON]**

**Location**: `src/trading/data/sources.py` (as method of DataSource implementations)

**Note**: External API integration (yfinance, etc.) stays in Python for flexibility. High-volume data processing can call Rust functions.

**Signature**:
```python
def fetch_bars(
    self,
    symbols: list[Symbol],
    date_range: DateRange,
    granularity: str
) -> Iterator[Bar]:
```

**Responsibility**: Retrieve raw bar data for the requested universe and time span.

**Input**:
- `symbols`: List of symbols to fetch
- `date_range`: Time range to fetch
- `granularity`: Bar granularity (e.g., "5m")

**Output**:
- `Iterator[Bar]`: Iterator yielding `Bar` objects in chronological order

**Behavior**:
1. For each symbol in `symbols`:
   - Fetch bars for the symbol within `date_range` at `granularity`
   - Yield `Bar` objects as they are fetched (streaming, not all in memory)
   - Bars should be yielded in chronological order (earliest first)
2. Bars from different symbols may be interleaved, but each symbol's bars must be in order

**Design Decision**: Return an iterator rather than a list to support large datasets without loading everything into memory.

**Error Handling**:
- Raise `DataSourceError` if symbol is invalid
- Raise `DataSourceError` if date_range is invalid
- Raise `DataSourceError` if network/local file access fails
- Log warnings for missing data but continue processing other symbols

**Dependencies**:
- `trading.types.Bar`, `Symbol`, `DateRange`
- `trading.exceptions.DataSourceError`

**Implementation Notes**:
- For Yahoo: Use `yfinance` library (add to dependencies)
- For Local: Read from `~/.trading/datasets/{dataset_id}/bars.parquet`
- For CSV: Read from CSV file specified in source_params

---

#### Function: `normalize_bars` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.normalize_bars`)

**Rust Implementation Notes**:
- Use PyO3 to accept Python iterator of Bar objects
- Convert to Rust structs, normalize, yield back as Python objects
- Use `PyIterator` and `PyResult` for Python interop

**Signature**:
```python
def normalize_bars(raw_bars: Iterator[Bar]) -> Iterator[NormalizedBar]:
```

**Responsibility**: Convert provider-specific bar format into the internal normalized bar representation.

**Input**:
- `raw_bars`: Iterator of `Bar` objects from data source

**Output**:
- `Iterator[NormalizedBar]`: Iterator yielding normalized bars

**Behavior**:
1. For each `Bar` in `raw_bars`:
   - Ensure timestamp has timezone info (default to UTC if missing)
   - Ensure all price fields (open, high, low, close) are positive floats
   - Ensure volume is non-negative float
   - Create `NormalizedBar` with same fields as `Bar`
   - Yield the `NormalizedBar`

**Design Decision**: For now, `NormalizedBar` is identical to `Bar`, but this function provides a hook for future transformations (currency conversion, corporate actions, etc.).

**Error Handling**:
- Raise `ValidationError` if timestamp cannot be normalized
- Raise `ValidationError` if price/volume fields are invalid
- Log warnings for bars that fail normalization but continue processing

**Dependencies**:
- `trading.types.Bar`, `NormalizedBar`
- `trading.exceptions.ValidationError`

---

#### Function: `validate_bars` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.validate_bars`)

**Rust Implementation Notes**:
- Accept Python iterator, validate in Rust for performance
- Use Rust's iterator combinators for efficient validation
- Yield validated bars back to Python

**Signature**:
```python
def validate_bars(normalized_bars: Iterator[NormalizedBar]) -> Iterator[NormalizedBar]:
```

**Responsibility**: Perform basic sanity checks on normalized bars.

**Input**:
- `normalized_bars`: Iterator of `NormalizedBar` objects

**Output**:
- `Iterator[NormalizedBar]`: Iterator yielding validated bars

**Behavior**:
1. Track the last timestamp seen per symbol
2. For each `NormalizedBar`:
   - Validate `high >= low` and `high >= open` and `high >= close` and `low <= open` and `low <= close`
   - Validate timestamp is after the last timestamp for this symbol (strictly increasing)
   - Validate all numeric fields are finite (not NaN, not inf)
   - If validation passes, yield the bar
   - If validation fails, log a warning and skip the bar (do not raise exception)

**Design Decision**: Validation is non-fatal - invalid bars are skipped rather than stopping the entire process. This allows datasets with minor data quality issues to still be usable.

**Error Handling**:
- Log warnings for invalid bars
- Do not raise exceptions (validation failures are logged but processing continues)

**Dependencies**:
- `trading.types.NormalizedBar`
- `logging` module

---

#### Function: `store_dataset` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.store_dataset`)

**Rust Implementation Notes**:
- Use `polars` or `arrow` crate for Parquet I/O (high performance)
- Accept Python iterator, collect into Rust Vec, write to Parquet
- Use `serde_json` for metadata serialization
- Create directories using `std::fs` or `walkdir` crate

**Signature**:
```python
def store_dataset(
    normalized_bars: Iterator[NormalizedBar],
    dataset_id: DatasetId,
    metadata: DatasetMetadata
) -> None:
```

**Responsibility**: Persist the dataset to local storage and update metadata.

**Input**:
- `normalized_bars`: Iterator of validated `NormalizedBar` objects
- `dataset_id`: Unique identifier for the dataset
- `metadata`: `DatasetMetadata` object with dataset information

**Output**:
- None (side effect: writes to disk)

**Behavior**:
1. Create directory `~/.trading/datasets/{dataset_id}/` if it doesn't exist
2. Collect all bars from iterator into a list (for now - can optimize later with streaming writes)
3. Write bars to `~/.trading/datasets/{dataset_id}/bars.parquet` using Parquet format
4. Write metadata to `~/.trading/datasets/{dataset_id}/metadata.json` using `metadata.model_dump_json()` (Pydantic serialization)
5. Ensure metadata includes `bar_count` field with actual count of bars written

**Design Decision**: Use Parquet format for efficient storage and querying. Use JSON for metadata for human readability. Store in user's home directory under `.trading/` to avoid permission issues.

**Error Handling**:
- Raise `StorageError` if directory cannot be created
- Raise `StorageError` if file write fails
- Raise `StorageError` if metadata serialization fails

**Dependencies**:
- `trading.types.NormalizedBar`, `DatasetId`, `DatasetMetadata`
- `trading.exceptions.StorageError`
- `pathlib.Path` for path handling
- `pyarrow` or `pandas` for Parquet writing (add to dependencies)
- `json` module for metadata

**Implementation Notes**:
- Use `pathlib.Path.home() / ".trading" / "datasets" / dataset_id` for path construction
- For Parquet: Use `pyarrow.parquet.write_table()` or `pandas.DataFrame.to_parquet()`
- Convert `NormalizedBar` objects to dict using `.model_dump()` before writing to Parquet, or convert to DataFrame directly

---

#### Function: `run_fetch_data_command` **[PYTHON]**

**Location**: `src/trading/commands/fetch_data.py`

**Note**: Orchestrates Python config loading, Python data fetching, then calls Rust functions for normalization, validation, and storage.

**Signature**:
```python
def run_fetch_data_command(args: argparse.Namespace) -> int:
```

**Responsibility**: Glue function invoked by the CLI; orchestrates the above steps using command-line arguments.

**Input**:
- `args`: Parsed command-line arguments from `argparse`

**Output**:
- `int`: Exit code (0 for success, non-zero for failure)

**Behavior**:
1. Extract `config_path` from `args.config` (required argument)
2. Call `load_fetch_data_config(config_path)` to get config
3. Call `resolve_data_source(config)` to get data source
4. Call `fetch_bars(data_source, config.symbols, config.date_range, config.granularity)` to get raw bars
5. Call `normalize_bars(raw_bars)` to normalize
6. Call `validate_bars(normalized_bars)` to validate
7. Build `DatasetMetadata` from config and bar count (need to count bars)
8. Call `store_dataset(validated_bars, config.dataset_id, metadata)`
9. Print success message with dataset_id
10. Return 0

**Error Handling**:
- Catch `ConfigError`, `DataSourceError`, `ValidationError`, `StorageError`
- Print error message to stderr
- Return non-zero exit code (1)

**Dependencies**:
- All functions above
- `argparse` module
- `sys` module for stderr

---

## 5. Command: `trading gen-synth`

### 5.1 High-Level Behavior

The `gen-synth` command generates synthetic market data that statistically resembles real markets, normalizes it, and stores it with metadata including generation parameters and seed.

### 5.2 Configuration File Format

```yaml
symbols:
  - "SYNTH1"
  - "SYNTH2"
date_range:
  start: "2020-01-01T00:00:00Z"
  end: "2024-01-01T00:00:00Z"
granularity: "5m"
generator_type: "geometric_brownian"  # or "mean_reverting"
generator_params:
  initial_price: 100.0
  drift: 0.0001
  volatility: 0.02
random_seed: 42  # Optional
dataset_id: "synth_gb_2020_2024"  # Optional
```

### 5.3 Function Specifications

#### Function: `load_gen_synth_config` **[PYTHON]**

**Location**: `src/trading/commands/gen_synth.py`

**Signature**:
```python
def load_gen_synth_config(config_path: str) -> GenSynthConfig:
```

**Responsibility**: Parse and validate the YAML configuration file for `gen-synth` command.

**Input**:
- `config_path`: Path to YAML configuration file

**Output**:
- `GenSynthConfig`: Validated configuration object

**Behavior**:
1. Read and parse YAML file
2. Validate required fields: `symbols`, `date_range`, `granularity`, `generator_type`, `generator_params`
3. Convert symbols to `Symbol` types
4. Parse date_range into `DateRange` object
5. Validate `generator_type` is recognized ("geometric_brownian" or "mean_reverting")
6. Validate `generator_params` contains required fields for the generator type
7. If `random_seed` not provided, use `None` (generator will use system randomness)
8. If `dataset_id` not provided, generate one: `synth_{generator_type}_{symbols}_{granularity}_{start}_{end}`
9. Return `GenSynthConfig` object

**Error Handling**:
- Raise `ConfigError` for file/parsing issues
- Raise `ConfigError` if generator_type is unrecognized
- Raise `ConfigError` if generator_params are invalid

**Dependencies**:
- `yaml` module
- `trading.types.GenSynthConfig`
- `trading.exceptions.ConfigError`

---

#### Function: `build_synth_generator` **[PYTHON]**

**Location**: `src/trading/data/synthetic.py`

**Note**: Generator construction stays in Python for flexibility. Actual generation can call Rust for performance.

**Signature**:
```python
def build_synth_generator(config: GenSynthConfig) -> SynthGenerator:
```

**Responsibility**: Construct a synthetic data generator from configuration.

**Input**:
- `config`: `GenSynthConfig` object

**Output**:
- `SynthGenerator`: Protocol/ABC object that implements generation interface

**Behavior**:
1. Based on `config.generator_type`:
   - If `"geometric_brownian"`: Return `GeometricBrownianGenerator(config)`
   - If `"mean_reverting"`: Return `MeanRevertingGenerator(config)`
   - Otherwise: Raise `DataSourceError`
2. The generator must implement a `generate_bars` method

**Design Decision**: Use Protocol/ABC pattern similar to DataSource. Each generator type is a separate class.

**Error Handling**:
- Raise `DataSourceError` if generator_type is unrecognized
- Raise `DataSourceError` if generator_params are invalid

**Dependencies**:
- `trading.types.GenSynthConfig`
- `trading.exceptions.DataSourceError`
- `trading.data.synthetic.GeometricBrownianGenerator`, `MeanRevertingGenerator`

---

#### Function: `generate_synth_bars` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.generate_synth_bars`)

**Rust Implementation Notes**:
- Use `rand` crate for random number generation
- Implement GBM and mean-reverting processes in Rust
- Accept config from Python, return iterator of bars
- Use `PyIterator` to yield bars to Python efficiently

**Signature**:
```python
def generate_bars(self) -> Iterator[Bar]:
```

**Responsibility**: Generate synthetic bar data based on generator configuration.

**Input**:
- None (uses `self.config`)

**Output**:
- `Iterator[Bar]`: Iterator yielding synthetic `Bar` objects

**Behavior**:
1. Initialize random number generator with `config.random_seed` if provided
2. For each symbol in `config.symbols`:
   - Initialize price series starting from `generator_params.initial_price`
   - For each time point in `config.date_range` at `config.granularity`:
     - Generate next price using generator's stochastic process
     - Create OHLC bar from price movement (simplified: open=prev_close, close=new_price, high=max, low=min)
     - Generate volume from log-normal distribution (mean from params, or default)
     - Yield `Bar` object
3. Bars must be in chronological order

**Design Decision**: For geometric Brownian motion, use standard GBM formula: `S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)` where Z is standard normal. For mean reverting, use Ornstein-Uhlenbeck process.

**Error Handling**:
- Raise `DataSourceError` if generation fails
- Log warnings for invalid parameters but continue

**Dependencies**:
- `trading.types.Bar`, `GenSynthConfig`
- `trading.exceptions.DataSourceError`
- `numpy` for random number generation (add to dependencies)

---

#### Function: `normalize_synth_bars` **[RUST]**

**Location**: `rust/src/lib.rs` (reuses `normalize_bars` function)

**Note**: Delegates to the same Rust `normalize_bars` function used for historical data.

**Signature**:
```python
def normalize_synth_bars(raw_bars: Iterator[Bar]) -> Iterator[NormalizedBar]:
```

**Responsibility**: Convert synthetic bars into normalized format (same as historical data normalization).

**Input**:
- `raw_bars`: Iterator of synthetic `Bar` objects

**Output**:
- `Iterator[NormalizedBar]`: Iterator yielding normalized bars

**Behavior**:
1. Same as `normalize_bars` function (reuse that function or call it)

**Design Decision**: Synthetic bars use the same normalization as historical bars, so this can delegate to `normalize_bars`.

**Error Handling**:
- Same as `normalize_bars`

**Dependencies**:
- `trading.data.normalize.normalize_bars` (reuse)

---

#### Function: `store_synth_dataset` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.store_synth_dataset`)

**Note**: Extends `store_dataset` to include synthetic-specific metadata.

**Signature**:
```python
def store_synth_dataset(
    bars: Iterator[NormalizedBar],
    dataset_id: DatasetId,
    metadata: DatasetMetadata,
    synth_config: GenSynthConfig
) -> None:
```

**Responsibility**: Store synthetic dataset with generation parameters in metadata.

**Input**:
- `bars`: Iterator of normalized bars
- `dataset_id`: Dataset identifier
- `metadata`: Base dataset metadata
- `synth_config`: Synthetic generation configuration (to include in metadata)

**Output**:
- None (side effect: writes to disk)

**Behavior**:
1. Call `store_dataset(bars, dataset_id, metadata)` to store bars and base metadata
2. Read back metadata JSON file and deserialize using `DatasetMetadata.model_validate_json(file_content)`
3. Create updated metadata dict using `metadata.model_dump()` and add `synthetic_config` field with:
   - `generator_type`
   - `generator_params`
   - `random_seed`
4. Create new `DatasetMetadata` instance with updated data and write back using `.model_dump_json()`

**Design Decision**: Extend base metadata with synthetic-specific fields rather than creating a separate metadata type.

**Error Handling**:
- Raise `StorageError` if storage operations fail
- Raise `StorageError` if metadata update fails

**Dependencies**:
- `trading.data.storage.store_dataset`
- `trading.types.GenSynthConfig`, `DatasetId`, `DatasetMetadata`
- `trading.exceptions.StorageError`
- `json` module

---

#### Function: `run_gen_synth_command` **[PYTHON]**

**Location**: `src/trading/commands/gen_synth.py`

**Note**: Python orchestrator that calls Rust functions for generation and storage.

**Signature**:
```python
def run_gen_synth_command(args: argparse.Namespace) -> int:
```

**Responsibility**: Orchestrate synthetic data generation command.

**Input**:
- `args`: Parsed command-line arguments

**Output**:
- `int`: Exit code

**Behavior**:
1. Load config: `config = load_gen_synth_config(args.config)`
2. Build generator: `generator = build_synth_generator(config)`
3. Generate bars: `raw_bars = generator.generate_bars()`
4. Normalize: `normalized_bars = normalize_synth_bars(raw_bars)`
5. Count bars and build metadata (include synth config)
6. Store: `store_synth_dataset(normalized_bars, config.dataset_id, metadata, config)`
7. Print success message
8. Return 0

**Error Handling**:
- Catch exceptions and return non-zero exit code

**Dependencies**:
- All gen-synth functions above
- `argparse`, `sys`

---

## 6. Command: `trading run-training`

### 6.1 High-Level Behavior

The `run-training` command executes a training/simulation cycle over specified datasets. It loads datasets, steps through them in time order, evaluates strategies, executes orders, updates accounts, and computes metrics.

### 6.2 Configuration File Format

```yaml
run_id: "run-001"  # Optional, auto-generated if omitted
datasets:
  - "qqq_spy_5m_2020_2024"
  - "synth_gb_2020_2024"
strategy:
  class_path: "trading.strategies.trend_following.TrendFollowingStrategy"
  params:
    lookback_periods: 20
    entry_threshold: 0.02
    exit_threshold: 0.01
account:
  starting_balance: 10000.0
  base_currency: "USD"
  clearing_delay_hours: 24
risk:
  max_position_size: 5000.0  # Optional
  max_leverage: 1.0
logging:
  level: "INFO"
```

### 6.3 Function Specifications

#### Function: `load_training_config` **[PYTHON]**

**Location**: `src/trading/commands/run_training.py`

**Signature**:
```python
def load_training_config(config_path: str) -> TrainingConfig:
```

**Responsibility**: Parse and validate training configuration file.

**Input**:
- `config_path`: Path to YAML configuration file

**Output**:
- `TrainingConfig`: Validated configuration object

**Behavior**:
1. Read and parse YAML file
2. Validate required sections: `datasets`, `strategy`, `account`
3. Parse `datasets` list into list of `DatasetId` types
4. Validate `strategy.class_path` is a valid Python import path
5. Validate `strategy.params` is a dictionary
6. Parse `account.starting_balance` as float (must be > 0)
7. Parse `account.base_currency` (default: "USD")
8. Parse `account.clearing_delay_hours` as int (default: 24, must be >= 0)
9. Parse `risk.max_position_size` as float or None (optional)
10. Parse `risk.max_leverage` as float (default: 1.0, must be > 0)
11. Parse `logging.level` (default: "INFO")
12. If `run_id` not provided, generate: `run_{timestamp}_{hash}` where hash is first 8 chars of datasets+strategy hash
13. Return `TrainingConfig` object

**Error Handling**:
- Raise `ConfigError` for file/parsing issues
- Raise `ConfigError` if required fields missing
- Raise `ConfigError` if values are invalid (negative balance, etc.)

**Dependencies**:
- `yaml` module
- `trading.types.TrainingConfig`
- `trading.exceptions.ConfigError`

---

#### Function: `load_datasets` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.load_datasets`)

**Rust Implementation Notes**:
- Use `polars` or `arrow` crate to read Parquet files efficiently
- Build `DatasetBundle` structure in Rust
- Implement efficient lookup methods (binary search for timestamps)
- Return Python-wrapped DatasetBundle object via PyO3

**Signature**:
```python
def load_datasets(config: TrainingConfig) -> DatasetBundle:
```

**Responsibility**: Load all datasets specified in training config.

**Input**:
- `config`: `TrainingConfig` object

**Output**:
- `DatasetBundle`: Object containing loaded datasets (to be defined)

**Design Decision**: `DatasetBundle` is a Pydantic model that provides efficient access to bars by timestamp and symbol. Internally, it stores bars in a structure optimized for chronological iteration (e.g., sorted list of timestamps with bars grouped by timestamp). The `DatasetBundle` type includes methods `get_bars_at(timestamp)` and `get_symbol_history(symbol, end_time, lookback_periods)`.

**Behavior**:
1. For each `dataset_id` in `config.datasets`:
   - Load dataset metadata from `~/.trading/datasets/{dataset_id}/metadata.json` using `DatasetMetadata.model_validate_json()`
   - Load bars from `~/.trading/datasets/{dataset_id}/bars.parquet` using Parquet reader
   - Validate dataset exists and is readable
   - Convert Parquet data to list of `NormalizedBar` objects
2. Merge bars from all datasets:
   - Group bars by timestamp (bars from different datasets with same timestamp are merged)
   - Handle overlapping symbols/times (later dataset overwrites earlier for same symbol+timestamp)
   - Sort all timestamps chronologically
3. Create `DatasetBundle` instance:
   - Store metadata dict: `{dataset_id: metadata}`
   - Store bars in efficient structure: `dict[datetime, dict[Symbol, NormalizedBar]]`
4. Implement `get_bars_at(timestamp)` method: Return bars dict for that timestamp, or empty dict if no bars
5. Implement `get_symbol_history(symbol, end_time, lookback_periods)` method: Return list of bars for symbol up to end_time
6. Return `DatasetBundle`

**Error Handling**:
- Raise `StorageError` if dataset not found
- Raise `StorageError` if dataset cannot be read
- Raise `ValidationError` if dataset metadata is invalid

**Dependencies**:
- `trading.types.TrainingConfig`, `DatasetId`
- `trading.exceptions.StorageError`, `ValidationError`
- `pyarrow` or `pandas` for Parquet reading

---

#### Function: `resolve_strategy_class` **[PYTHON]**

**Location**: `src/trading/strategies/registry.py`

**Note**: Strategy loading stays in Python for flexibility and ML/RL integration.

**Signature**:
```python
def resolve_strategy_class(class_path: str) -> type[Strategy]:
```

**Responsibility**: Resolve a fully qualified Python class path to a Strategy class.

**Input**:
- `class_path`: Fully qualified class path (e.g., "trading.strategies.trend_following.TrendFollowingStrategy")

**Output**:
- `type[Strategy]`: Strategy class (not instance)

**Behavior**:
1. Split `class_path` into module path and class name (last component is class name)
2. Use `importlib.import_module()` to import the module
3. Get the class attribute from the module
4. Validate the class is a subclass of `Strategy` (ABC)
5. Return the class

**Design Decision**: Strategy is an ABC with abstract method `decide(snapshot: AnalysisSnapshot, account: Account) -> list[OrderRequest]`.

**Error Handling**:
- Raise `StrategyError` if module cannot be imported
- Raise `StrategyError` if class not found in module
- Raise `StrategyError` if class is not a Strategy subclass

**Dependencies**:
- `importlib` module
- `trading.strategies.base.Strategy` (ABC)
- `trading.exceptions.StrategyError`

---

#### Function: `instantiate_strategy` **[PYTHON]**

**Location**: `src/trading/strategies/registry.py`

**Note**: Strategy instantiation stays in Python.

**Signature**:
```python
def instantiate_strategy(
    strategy_class: type[Strategy],
    params: dict[str, Any]
) -> Strategy:
```

**Responsibility**: Create a Strategy instance with given parameters.

**Input**:
- `strategy_class`: Strategy class to instantiate
- `params`: Dictionary of parameters to pass to constructor

**Output**:
- `Strategy`: Instantiated strategy object

**Behavior**:
1. Call `strategy_class(**params)` to instantiate
2. Return the instance

**Error Handling**:
- Raise `StrategyError` if instantiation fails (wrong parameters, etc.)

**Dependencies**:
- `trading.strategies.base.Strategy`
- `trading.exceptions.StrategyError`

---

#### Function: `initialize_account` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.initialize_account`)

**Rust Implementation Notes**:
- Create Account struct in Rust
- Use PyO3 to accept Python TrainingConfig, return Account
- Account struct should be `#[derive(Serialize, Deserialize)]` for JSON storage

**Signature**:
```python
def initialize_account(config: TrainingConfig) -> Account:
```

**Responsibility**: Create and initialize a training account from configuration.

**Input**:
- `config`: `TrainingConfig` object

**Output**:
- `Account`: Initialized account object

**Behavior**:
1. Create `Account` object with:
   - `account_id`: Generated unique ID (e.g., `"account_{run_id}"`)
   - `base_currency`: `config.account_base_currency`
   - `cleared_balance`: `config.account_starting_balance`
   - `pending_balance`: `0.0`
   - `positions`: Empty dictionary
   - `clearing_delay_hours`: `config.clearing_delay_hours`
2. Return account

**Error Handling**:
- Raise `AccountError` if starting balance is invalid

**Dependencies**:
- `trading.types.Account`, `TrainingConfig`
- `trading.exceptions.AccountError`

---

#### Function: `iteration_schedule` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.iteration_schedule`)

**Rust Implementation Notes**:
- Accept DatasetBundle from Rust, iterate efficiently
- Use Rust iterators for performance
- Yield TimeSlice objects to Python via PyIterator
- Handle variable sampling rates efficiently in Rust

**Signature**:
```python
def iteration_schedule(dataset_bundle: DatasetBundle) -> Iterator[TimeSlice]:
```

**Responsibility**: Generate time slices from dataset bundle in chronological order.

**Input**:
- `dataset_bundle`: `DatasetBundle` containing all loaded datasets

**Output**:
- `Iterator[TimeSlice]`: Iterator yielding time slices in chronological order

**Behavior**:
1. Extract all unique timestamps from dataset_bundle (across all symbols)
2. Sort timestamps chronologically
3. For each timestamp:
   - Get all bars available at that timestamp (from all symbols)
   - Create `TimeSlice` with `timestamp` and `bars` dict
   - Yield the `TimeSlice`

**Design Decision**: Each time slice contains all bars available at that exact timestamp. If a symbol doesn't have a bar at a timestamp, it's simply not included in that time slice's bars dict.

**Error Handling**:
- Raise `ValidationError` if dataset_bundle is empty
- Log warnings for duplicate timestamps but continue

**Dependencies**:
- `trading.types.TimeSlice`, `DatasetBundle`
- `trading.exceptions.ValidationError`

---

#### Function: `build_analysis_snapshot` **[PYTHON]**

**Location**: `src/trading/training/snapshot.py`

**Note**: Python wrapper that calls Rust DatasetBundle methods and combines with Python Account object.

**Signature**:
```python
def build_analysis_snapshot(
    time_slice: TimeSlice,
    dataset_bundle: DatasetBundle,
    account: Account,
    config: TrainingConfig,
    max_bar_age: timedelta | None = None
) -> AnalysisSnapshot:
```

**Responsibility**: Build a market snapshot for strategy evaluation, handling variable sampling rates and missing data.

**Input**:
- `time_slice`: Current time slice
- `dataset_bundle`: Dataset bundle (for potential lookback/history access)
- `account`: Current account state
- `config`: Training configuration (for analysis universe)
- `max_bar_age`: Optional maximum age for bars (bars older than this are excluded)

**Output**:
- `AnalysisSnapshot`: Snapshot object for strategy

**Behavior**:
1. Filter bars based on analysis universe:
   - If `config.analysis_universe` is specified, only include bars for symbols in that list
   - Otherwise, include all bars from `time_slice.bars`
2. Filter bars by age (if `max_bar_age` provided):
   - For each bar, check if `time_slice.timestamp - bar_timestamp <= max_bar_age`
   - Exclude bars that are too old
3. Handle missing data:
   - Symbols in analysis universe but not in `time_slice.bars` are simply omitted
   - This is expected behavior for variable sampling rates and missing data
4. Create `AnalysisSnapshot` with:
   - `timestamp`: `time_slice.timestamp` (the evaluation timestamp)
   - `bars`: Filtered bars dict (only analysis universe symbols, may be missing some)
   - `account`: Current account state (passed in)
5. Return snapshot

**Design Decision**: 
- Analysis universe allows strategies to analyze more symbols than they can trade
- Missing bars are handled gracefully: strategies receive only available data
- `max_bar_age` allows filtering stale data (e.g., exclude bars older than 1 hour for real-time strategies)
- Variable sampling rates are transparent to strategies: they receive the most recent available bar per symbol
- Historical bars for lookback can be accessed via `dataset_bundle.get_symbol_history()` if needed

**Error Handling**:
- None (always succeeds, missing symbols simply not included)
- Log debug messages for missing symbols if verbose logging enabled

**Dependencies**:
- `trading.types.TimeSlice`, `AnalysisSnapshot`, `Account`, `DatasetBundle`, `TrainingConfig`
- `datetime.timedelta`

---

#### Function: `strategy_decide` **[PYTHON]**

**Location**: `src/trading/training/strategy_executor.py`

**Note**: Strategy execution stays in Python for ML/RL flexibility.

**Signature**:
```python
def strategy_decide(
    strategy: Strategy,
    snapshot: AnalysisSnapshot,
    account: Account
) -> list[OrderRequest]:
```

**Responsibility**: Invoke strategy to generate order requests.

**Input**:
- `strategy`: Strategy instance
- `snapshot`: Current market snapshot
- `account`: Current account state

**Output**:
- `list[OrderRequest]`: List of order requests from strategy

**Behavior**:
1. Call `strategy.decide(snapshot, account)` (abstract method)
2. Validate returned list contains only `OrderRequest` objects
3. Return the list

**Error Handling**:
- Raise `StrategyError` if strategy.decide() raises exception
- Raise `ValidationError` if returned objects are not OrderRequest instances
- Log strategy errors but allow training to continue (return empty list)

**Dependencies**:
- `trading.strategies.base.Strategy`
- `trading.types.OrderRequest`, `AnalysisSnapshot`, `Account`
- `trading.exceptions.StrategyError`, `ValidationError`

---

#### Function: `apply_risk_constraints` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.apply_risk_constraints`)

**Rust Implementation Notes**:
- Accept Python list of OrderRequest, Account, TrainingConfig
- Validate constraints efficiently in Rust
- Return filtered list of OrderRequest to Python
- Use PyO3 to convert between Python and Rust types

**Signature**:
```python
def apply_risk_constraints(
    order_requests: list[OrderRequest],
    account: Account,
    config: TrainingConfig
) -> list[OrderRequest]:
```

**Responsibility**: Apply risk and account constraints to order requests.

**Input**:
- `order_requests`: List of order requests from strategy
- `account`: Current account state
- `config`: Training configuration with risk parameters

**Output**:
- `list[OrderRequest]`: Filtered/modified order requests that pass constraints

**Behavior**:
1. For each `OrderRequest`:
   - Check if account has sufficient cleared balance for buy orders (quantity * current_price estimate)
   - Check if account has sufficient position for sell orders
   - Check if order would exceed `max_position_size` (if configured)
   - Check if order would exceed `max_leverage` (if configured)
   - If all checks pass, keep the order; otherwise, remove it
2. Return filtered list

**Design Decision**: For price estimation, use the most recent bar's close price for the symbol. If no bar available, reject the order.

**Error Handling**:
- Log warnings for rejected orders
- Do not raise exceptions (constraint violations result in order removal)

**Dependencies**:
- `trading.types.OrderRequest`, `Account`, `TrainingConfig`
- `logging` module

---

#### Function: `execute_orders` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.execute_orders`)

**Rust Implementation Notes**:
- Accept list of Orders and TimeSlice from Python
- Execute orders efficiently in Rust
- Return list of Execution objects to Python
- Use PyO3 for type conversions

**Signature**:
```python
def execute_orders(
    order_requests: list[OrderRequest],
    account: Account,
    time_slice: TimeSlice
) -> list[Execution]:
```

**Responsibility**: Execute order requests against current market data (simulation).

**Input**:
- `order_requests`: List of validated order requests
- `account`: Current account state
- `time_slice`: Current time slice with market data

**Output**:
- `list[Execution]`: List of execution records

**Behavior**:
1. For each `OrderRequest`:
   - Get current bar for the symbol from `time_slice.bars`
   - Use bar's close price as execution price (simplified market order execution)
   - Create `Execution` object with:
     - `symbol`: From order request
     - `side`: From order request
     - `quantity`: From order request
     - `price`: Bar's close price
     - `timestamp`: `time_slice.timestamp`
     - `order_id`: Generated unique ID
   - Add to executions list
2. Return list of executions

**Design Decision**: Simplified execution model - market orders execute at the close price of the current bar. No slippage, no partial fills, no order book simulation.

**Error Handling**:
- Skip orders for symbols not in `time_slice.bars` (log warning)
- Do not raise exceptions (missing data results in skipped execution)

**Dependencies**:
- `trading.types.OrderRequest`, `Execution`, `Account`, `TimeSlice`
- `uuid` module for order_id generation
- `logging` module

---

#### Function: `update_account_from_executions` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.update_account_from_executions`)

**Rust Implementation Notes**:
- Core account update logic in Rust for performance
- Handle clearing, reservations, position updates
- Use mutable reference to Account (or return new Account)
- Call `process_clearing` internally

**Signature**:
```python
def update_account_from_executions(
    account: Account,
    executions: list[Execution],
    time_slice: TimeSlice
) -> None:
```

**Responsibility**: Update account state based on trade executions.

**Input**:
- `account`: Account to update (modified in place)
- `executions`: List of executions to process
- `time_slice`: Current time slice (for reference)

**Output**:
- None (side effect: modifies account)

**Behavior**:
1. Process scheduled clearing first: Call `process_clearing(account, time_slice.timestamp)`
2. For each `Execution`:
   - Calculate trade value: `quantity * price`
   - Release reservation: If funds were reserved, call `release_reservation(account, trade_value)` for buy orders
   - If `side == "buy"`:
     - Deduct `trade_value` from `account.pending_balance` (not cleared yet)
     - Update or create position: add `quantity` to `pending_quantity`, update cost basis
   - If `side == "sell"`:
     - Add `trade_value` to `account.pending_balance`
     - Update position: subtract `quantity` from position (from cleared quantity first, then pending)
   - Schedule clearing: record that `trade_value` (and position change) should clear at calculated clearing time
     - Use `next_business_day` if `account.use_business_days`, else add hours

**Design Decision**: Account maintains a list of pending transactions with their clearing times. Reservations are released when orders execute. Business day clearing is optional.

**Error Handling**:
- Raise `AccountError` if sell order exceeds position quantity
- Raise `AccountError` if buy order exceeds available balance (shouldn't happen if constraints applied)

**Dependencies**:
- `trading.types.Account`, `Execution`, `TimeSlice`
- `trading.exceptions.AccountError`
- `process_clearing`, `release_reservation`, `next_business_day` functions

---

#### Function: `record_training_step` **[PYTHON]**

**Location**: `src/trading/training/recording.py`

**Note**: Python wrapper that records state. Can call Rust functions for efficient data serialization if needed.

**Signature**:
```python
def record_training_step(
    run_state: RunState,
    time_slice: TimeSlice,
    snapshot: AnalysisSnapshot,
    orders: list[OrderRequest],
    executions: list[Execution],
    account: Account
) -> None:
```

**Responsibility**: Record state of a single training step for later analysis.

**Input**:
- `run_state`: Run state to update
- `time_slice`: Current time slice
- `snapshot`: Analysis snapshot used
- `orders`: Order requests generated
- `executions`: Executions that occurred
- `account`: Account state after update

**Output**:
- None (side effect: updates run_state)

**Behavior**:
1. Create step record dictionary with:
   - `timestamp`: `time_slice.timestamp`
   - `account_cleared_balance`: `account.cleared_balance`
   - `account_pending_balance`: `account.pending_balance`
   - `account_total_equity`: Calculate from cleared + pending + mark-to-market positions
   - `num_orders`: `len(orders)`
   - `num_executions`: `len(executions)`
   - `executions`: List of execution details
2. Append to `run_state.step_records`
3. Append `time_slice` to `run_state.time_slices`
4. Extend `run_state.executions` with new executions
5. Extend `run_state.order_requests` with new orders
6. Update `run_state.account` to current account state

**Design Decision**: Store lightweight step records for metrics computation. Full account state is also stored in run_state for detailed analysis.

**Error Handling**:
- Log warnings for recording errors but continue

**Dependencies**:
- `trading.types.RunState`, `TimeSlice`, `AnalysisSnapshot`, `OrderRequest`, `Execution`, `Account`
- `logging` module

---

#### Function: `compute_run_metrics` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.compute_run_metrics`)

**Rust Implementation Notes**:
- Use `ndarray` or `polars` for efficient statistical calculations
- Compute all metrics in Rust for performance
- Return RunMetrics struct to Python via PyO3
- Handle edge cases (empty runs, no trades) gracefully

**Signature**:
```python
def compute_run_metrics(run_state: RunState) -> RunMetrics:
```

**Responsibility**: Compute summary metrics from completed training run.

**Input**:
- `run_state`: Completed run state with all step records

**Output**:
- `RunMetrics`: Computed metrics object

**Behavior**:
1. Extract equity series from `run_state.step_records` (account_total_equity over time)
2. Compute `total_return`: `(final_equity - initial_equity) / initial_equity`
3. Compute `max_drawdown`: Maximum peak-to-trough decline in equity series
4. Compute `volatility`: Standard deviation of period-over-period returns
5. Compute `sharpe_ratio`: `(total_return / volatility) * sqrt(252)` if volatility > 0, else None (assuming daily periods, adjust for actual granularity)
6. Count `num_trades`: Total number of executions
7. Compute `win_rate`: Fraction of profitable trades (if num_trades > 0), else None
8. Create and return `RunMetrics` object

**Design Decision**: Sharpe ratio calculation assumes daily periods. For 5-minute bars, adjust period count accordingly (e.g., 252 * 78 periods per day for 5-minute bars).

**Error Handling**:
- Handle edge cases (empty run, no trades, etc.)
- Return None for metrics that cannot be computed

**Dependencies**:
- `trading.types.RunState`, `RunMetrics`
- `numpy` or standard library `statistics` for calculations

---

#### Function: `store_run_results` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.store_run_results`)

**Rust Implementation Notes**:
- Use `serde_json` for JSON serialization
- Write files using `std::fs` or `tokio::fs` for async (if needed)
- Efficient serialization of large run_state objects

**Signature**:
```python
def store_run_results(
    run_id: RunId,
    run_state: RunState,
    metrics: RunMetrics
) -> None:
```

**Responsibility**: Persist training run results to storage.

**Input**:
- `run_id`: Run identifier
- `run_state`: Complete run state
- `metrics`: Computed metrics

**Output**:
- None (side effect: writes to disk)

**Behavior**:
1. Create directory `~/.trading/runs/{run_id}/` if it doesn't exist
2. Write `run_state` to `~/.trading/runs/{run_id}/run_state.json` using `run_state.model_dump_json()` (Pydantic serialization)
3. Write `metrics` to `~/.trading/runs/{run_id}/metrics.json` using `metrics.model_dump_json()`
4. Write `run_state.config` to `~/.trading/runs/{run_id}/config.json` using `run_state.config.model_dump_json()`
5. Write step records to `~/.trading/runs/{run_id}/steps.json` (array of step records, use `json.dump()` for plain dicts)
6. Write executions to `~/.trading/runs/{run_id}/executions.json` (serialize list using `[e.model_dump() for e in executions]` then `json.dump()`)

**Design Decision**: Store run data as JSON for human readability. Large datasets (like full bar history) are not stored in run artifacts - they reference the original datasets.

**Error Handling**:
- Raise `StorageError` if directory creation fails
- Raise `StorageError` if file writes fail
- Raise `StorageError` if serialization fails

**Dependencies**:
- `trading.types.RunId`, `RunState`, `RunMetrics`
- `trading.exceptions.StorageError`
- `json` module
- `pathlib.Path`

---

#### Function: `checkpoint_run_state` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.checkpoint_run_state`)

**Rust Implementation Notes**:
- Serialize RunState efficiently using serde
- Write checkpoint files atomically (write to temp, then rename)

**Signature**:
```python
def checkpoint_run_state(run_state: RunState) -> None:
```

**Responsibility**: Save current run state as a checkpoint for resume capability.

**Input**:
- `run_state`: Current run state to checkpoint

**Output**:
- None (side effect: writes checkpoint to disk)

**Behavior**:
1. Create checkpoint directory: `~/.trading/runs/{run_state.run_id}/checkpoints/` if it doesn't exist
2. Generate checkpoint filename: `checkpoint_{timestamp}.json`
3. Serialize run_state to JSON using `run_state.model_dump_json()`
4. Write to checkpoint file
5. Optionally: Keep only last N checkpoints (e.g., last 10) to manage disk space

**Error Handling**:
- Raise `StorageError` if checkpoint write fails
- Log warnings but continue if checkpoint directory creation fails

**Dependencies**:
- `trading.types.RunState`
- `trading.exceptions.StorageError`
- `pathlib.Path`, `json` module

---

#### Function: `resume_training_run` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.resume_training_run`)

**Rust Implementation Notes**:
- Read checkpoint JSON using serde_json
- Deserialize to RunState struct
- Return to Python via PyO3

**Signature**:
```python
def resume_training_run(run_id: RunId) -> RunState:
```

**Responsibility**: Load the most recent checkpoint and resume training run.

**Input**:
- `run_id`: Run identifier

**Output**:
- `RunState`: Restored run state

**Behavior**:
1. List checkpoint files in `~/.trading/runs/{run_id}/checkpoints/`
2. Find most recent checkpoint (by timestamp in filename)
3. Load checkpoint JSON file
4. Deserialize using `RunState.model_validate_json(file_content)`
5. Return restored run_state

**Error Handling**:
- Raise `StorageError` if no checkpoints found
- Raise `StorageError` if checkpoint cannot be loaded

**Dependencies**:
- `trading.types.RunId`, `RunState`
- `trading.exceptions.StorageError`
- `pathlib.Path`, `json` module

---

#### Function: `log_event` **[PYTHON]**

**Location**: `src/trading/training/logging.py`

**Note**: Event logging stays in Python for flexibility and integration with Python logging ecosystem.

**Signature**:
```python
def log_event(event_type: str, data: dict[str, Any], run_id: RunId) -> None:
```

**Responsibility**: Log structured event to event log file.

**Input**:
- `event_type`: Type of event (e.g., "order_submission", "execution", "balance_change")
- `data`: Event data dictionary
- `run_id`: Run identifier

**Output**:
- None (side effect: writes to event log)

**Behavior**:
1. Create event log file: `~/.trading/runs/{run_id}/events.log` if it doesn't exist
2. Create event record:
   - `timestamp`: Current timestamp (ISO format)
   - `event_type`: Event type
   - `data`: Event data
3. Append JSON line to event log file (one JSON object per line)
4. Optionally: Also log to standard Python logger

**Error Handling**:
- Log warnings if file write fails but don't raise exceptions
- Continue execution even if logging fails

**Dependencies**:
- `trading.types.RunId`
- `json` module, `pathlib.Path`
- `logging` module

---

#### Function: `get_run_progress` **[PYTHON]**

**Location**: `src/trading/training/monitoring.py`

**Note**: Python wrapper that queries RunState. Can call Rust for efficient calculations if needed.

**Signature**:
```python
def get_run_progress(run_state: RunState, start_time: datetime) -> RunProgress:
```

**Responsibility**: Calculate current progress of a running training run.

**Input**:
- `run_state`: Current run state
- `start_time`: When the run started

**Output**:
- `RunProgress`: Progress information

**Behavior**:
1. Calculate `completed_timestamps`: `len(run_state.time_slices)`
2. Estimate `total_timestamps`: From dataset metadata or current progress rate
3. Calculate `elapsed_time_seconds`: `(datetime.now() - start_time).total_seconds()`
4. Get `current_timestamp`: Last timestamp in `run_state.time_slices` or None
5. Calculate `current_account_equity`: From last step record or account state
6. Count `num_executions`: `len(run_state.executions)`
7. Create and return `RunProgress` object

**Error Handling**:
- Handle edge cases (empty run_state, etc.)
- Return progress with defaults if calculation fails

**Dependencies**:
- `trading.types.RunState`, `RunProgress`
- `datetime` module

---

#### Function: `run_training_command` **[PYTHON]**

**Location**: `src/trading/commands/run_training.py`

**Note**: Main Python orchestrator that coordinates Rust core functions and Python strategy logic.

**Signature**:
```python
def run_training_command(args: argparse.Namespace) -> int:
```

**Responsibility**: Orchestrate the complete training run workflow.

**Input**:
- `args`: Parsed command-line arguments

**Output**:
- `int`: Exit code

**Behavior**:
1. Check for resume flag: If `args.resume` and `args.run_id` provided:
   - Load checkpoint: `run_state = resume_training_run(args.run_id)`
   - Extract config and account from run_state
   - Skip to step 7 (continue from checkpoint)
2. Load config: `config = load_training_config(args.config)`
3. Validate config: `errors = validate_training_config(config)` - if errors, print and return 1
4. Load datasets: `dataset_bundle = load_datasets(config)`
5. Resolve strategy: `strategy_class = resolve_strategy_class(config.strategy_class_path)`
6. Instantiate strategy: `strategy = instantiate_strategy(strategy_class, config.strategy_params)`
7. Initialize account: `account = initialize_account(config)` (or use from checkpoint)
8. Create run_state: `run_state = RunState(run_id=config.run_id, config=config, account=account, ...)`
9. Parse iteration granularity: If `config.iteration_granularity` provided, convert to `timedelta` (e.g., "5m" -> 5 minutes, "1h" -> 1 hour)
10. Parse max bar age: If `config.max_bar_age` provided, convert to `timedelta` (e.g., "1h" -> 1 hour, "1d" -> 1 day)
11. Get iteration schedule: `time_slices = iteration_schedule(dataset_bundle, min_granularity=parsed_granularity)`
12. Start time tracking: `start_time = datetime.now()`
13. For each `time_slice` in `time_slices`:
    - Build snapshot: `snapshot = build_analysis_snapshot(time_slice, dataset_bundle, account, config, max_bar_age=parsed_max_age)`
    - Get strategy decisions: `order_requests = strategy_decide(strategy, snapshot, account)`
    - Create orders: `orders = [create_order(req) for req in order_requests]`
    - Log order submissions: `log_event("order_submission", {...}, config.run_id)` if enabled
    - Reserve funds: For buy orders, call `reserve_funds(account, ...)` 
    - Apply constraints: `filtered_orders = apply_risk_constraints(orders, account, config, time_slice.bars)`
    - Execute orders: `executions = execute_orders(filtered_orders, account, time_slice)`
    - Log executions: `log_event("execution", {...}, config.run_id)` if enabled
    - Update account: `update_account_from_executions(account, executions, time_slice)`
    - Log balance changes: `log_event("balance_change", {...}, config.run_id)` if enabled
    - Record step: `record_training_step(run_state, time_slice, snapshot, orders, executions, account)`
    - Checkpoint: If `config.checkpoint_interval` and `len(run_state.time_slices) % config.checkpoint_interval == 0`:
      - `checkpoint_run_state(run_state)`
12. Compute metrics: `metrics = compute_run_metrics(run_state)`
13. Store results: `store_run_results(config.run_id, run_state, metrics)`
14. Print summary with metrics
15. Return 0

**Error Handling**:
- Catch all exceptions, log errors, return non-zero exit code
- Ensure partial results are saved even if run fails partway through
- Save checkpoint before exiting on error

**Dependencies**:
- All training functions above
- `argparse`, `sys`, `logging`, `datetime`

---

## 7. Command: `trading inspect-run`

### 7.1 High-Level Behavior

The `inspect-run` command loads and displays results from a previous training run, including metrics, configuration, and optionally exports data.

### 7.2 Function Specifications

#### Function: `parse_inspect_run_args` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Signature**:
```python
def parse_inspect_run_args(args: argparse.Namespace) -> InspectRunRequest:
```

**Responsibility**: Parse and validate command-line arguments for inspect-run.

**Input**:
- `args`: Parsed arguments from argparse

**Output**:
- `InspectRunRequest`: Pydantic BaseModel with `run_id: RunId` and `output_path: str | None`

**Behavior**:
1. Extract `run_id` from `args.run_id` (required)
2. Extract `output_path` from `args.output` (optional)
3. Validate `run_id` is not empty
4. If `output_path` provided, validate parent directory exists or can be created
5. Return `InspectRunRequest` object

**Error Handling**:
- Raise `ConfigError` if run_id is missing or invalid
- Raise `ConfigError` if output_path is invalid

**Dependencies**:
- `trading.types.RunId`
- `trading.exceptions.ConfigError`
- `pathlib.Path`

---

#### Function: `load_run_artifacts` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.load_run_artifacts`)

**Rust Implementation Notes**:
- Read JSON files using serde_json
- Deserialize to Rust structs
- Return RunArtifacts to Python via PyO3

**Signature**:
```python
def load_run_artifacts(run_id: RunId) -> RunArtifacts:
```

**Responsibility**: Load all artifacts from a completed training run.

**Input**:
- `run_id`: Run identifier

**Output**:
- `RunArtifacts`: Complete run artifacts object

**Behavior**:
1. Check if `~/.trading/runs/{run_id}/` directory exists
2. Load `config.json` and deserialize using `TrainingConfig.model_validate_json(file_content)`
3. Load `metrics.json` and deserialize using `RunMetrics.model_validate_json(file_content)`
4. Load `run_state.json` and deserialize using `RunState.model_validate_json(file_content)`
5. Load `steps.json` using `json.load()` (plain dicts)
6. Load `executions.json` and deserialize list using `[Execution.model_validate(e) for e in executions_list]`
7. Get `created_at` from directory modification time or metadata file
8. Create and return `RunArtifacts` object

**Error Handling**:
- Raise `StorageError` if run directory not found
- Raise `StorageError` if required files missing
- Raise `StorageError` if deserialization fails

**Dependencies**:
- `trading.types.RunId`, `RunArtifacts`
- `trading.exceptions.StorageError`
- `json` module
- `pathlib.Path`

---

#### Function: `summarize_run_metrics` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Note**: Formatting and display logic stays in Python.

**Signature**:
```python
def summarize_run_metrics(metrics: RunMetrics) -> str:
```

**Responsibility**: Format run metrics as a human-readable string.

**Input**:
- `metrics`: `RunMetrics` object

**Output**:
- `str`: Formatted metrics summary

**Behavior**:
1. Format metrics as:
   ```
   Run Metrics:
   - Total Return: {total_return:.2%}
   - Max Drawdown: {max_drawdown:.2%}
   - Volatility: {volatility:.4f}
   - Sharpe Ratio: {sharpe_ratio:.2f} (or "N/A")
   - Number of Trades: {num_trades}
   - Win Rate: {win_rate:.2%} (or "N/A")
   ```
2. Return formatted string

**Error Handling**:
- Handle None values gracefully (display "N/A")

**Dependencies**:
- `trading.types.RunMetrics`

---

#### Function: `format_run_summary` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Signature**:
```python
def format_run_summary(artifacts: RunArtifacts) -> str:
```

**Responsibility**: Format complete run summary including config and metrics.

**Input**:
- `artifacts`: `RunArtifacts` object

**Output**:
- `str`: Complete formatted summary

**Behavior**:
1. Build summary string with:
   - Run ID
   - Created at timestamp
   - Configuration summary (datasets, strategy, account settings)
   - Metrics summary (call `summarize_run_metrics`)
   - Key statistics (total executions, final account balance, etc.)
2. Return formatted string

**Error Handling**:
- Handle missing fields gracefully

**Dependencies**:
- `trading.types.RunArtifacts`
- `summarize_run_metrics` function

---

#### Function: `print_run_summary` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Signature**:
```python
def print_run_summary(summary: str) -> None:
```

**Responsibility**: Print summary to stdout.

**Input**:
- `summary`: Formatted summary string

**Output**:
- None (side effect: prints to stdout)

**Behavior**:
1. Print `summary` to stdout
2. Add newline at end

**Error Handling**:
- None (always succeeds)

**Dependencies**:
- `print` builtin

---

#### Function: `export_run_data` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Note**: Can call Rust functions for efficient CSV/JSON serialization if needed.

**Signature**:
```python
def export_run_data(
    artifacts: RunArtifacts,
    output_path: str
) -> None:
```

**Responsibility**: Export run data to a file (JSON or CSV based on extension).

**Input**:
- `artifacts`: `RunArtifacts` object
- `output_path`: Path to output file

**Output**:
- None (side effect: writes file)

**Behavior**:
1. Determine file format from extension (`.json` or `.csv`)
2. If JSON:
   - Serialize `artifacts` to JSON using `.model_dump()` method (Pydantic provides built-in JSON serialization)
   - Write to file
3. If CSV:
   - Export step records as CSV (timestamp, equity, etc.)
   - Write to file
4. Print confirmation message

**Design Decision**: Support both JSON (full data) and CSV (tabular data for analysis) export formats.

**Error Handling**:
- Raise `StorageError` if file write fails
- Raise `ConfigError` if format is unsupported

**Dependencies**:
- `trading.types.RunArtifacts`
- `trading.exceptions.StorageError`, `ConfigError`
- `json` module or `csv` module
- `pathlib.Path`

---

#### Function: `run_inspect_run_command` **[PYTHON]**

**Location**: `src/trading/commands/inspect_run.py`

**Signature**:
```python
def run_inspect_run_command(args: argparse.Namespace) -> int:
```

**Responsibility**: Orchestrate inspect-run command workflow.

**Input**:
- `args`: Parsed command-line arguments

**Output**:
- `int`: Exit code

**Behavior**:
1. Parse args: `request = parse_inspect_run_args(args)`
2. Load artifacts: `artifacts = load_run_artifacts(request.run_id)`
3. Format summary: `summary = format_run_summary(artifacts)`
4. Print summary: `print_run_summary(summary)`
5. If `request.output_path` is provided:
   - Export data: `export_run_data(artifacts, request.output_path)`
6. Return 0

**Error Handling**:
- Catch exceptions, print error, return non-zero exit code

**Dependencies**:
- All inspect-run functions above
- `argparse`, `sys`

---

## 8. CLI Entry Point and Argument Parsing

### 8.1 Main CLI Function

**Location**: `src/trading/main.py` **[PYTHON]**

**Signature**:
```python
def main() -> None:
```

**Responsibility**: Main entry point for CLI, delegates to subcommands.

**Behavior**:
1. Create argument parser with subcommands: `fetch-data`, `gen-synth`, `run-training`, `inspect-run`
2. Parse arguments
3. Route to appropriate command function based on subcommand
4. Call command function with parsed args
5. Exit with returned code

**Dependencies**:
- `argparse` module
- All `run_*_command` functions

### 8.2 Argument Parser Setup

Each subcommand should have:
- `--config` or positional argument for config file path
- `--help` for usage information
- Subcommand-specific arguments (e.g., `--output` for inspect-run)

---

## 9. Additional Supporting Functions

### 9.1 Dataset Metadata Management

#### Function: `read_dataset_metadata` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.read_dataset_metadata`)

**Rust Implementation Notes**:
- Read JSON metadata file using serde_json
- Return DatasetMetadata struct to Python via PyO3

**Signature**:
```python
def read_dataset_metadata(dataset_id: DatasetId) -> DatasetMetadata:
```

**Responsibility**: Read dataset metadata from storage.

**Behavior**:
1. Read `~/.trading/datasets/{dataset_id}/metadata.json`
2. Deserialize using `DatasetMetadata.model_validate_json(file_content)`
3. Return metadata

**Error Handling**:
- Raise `StorageError` if file not found or invalid

---

#### Function: `list_datasets` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.list_datasets`)

**Rust Implementation Notes**:
- Use `walkdir` crate to scan `~/.trading/datasets/`
- Read metadata files efficiently
- Return list of DatasetMetadata to Python

**Signature**:
```python
def list_datasets() -> list[DatasetId]:
```

**Responsibility**: List all available dataset IDs.

**Behavior**:
1. Scan `~/.trading/datasets/` directory
2. Return list of dataset IDs (directory names)

**Error Handling**:
- Return empty list if directory doesn't exist
- Log warnings for invalid datasets but continue

---

#### Function: `validate_dataset_exists` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.validate_dataset_exists`)

**Rust Implementation Notes**:
- Check file existence using `std::path::Path::exists()`
- Raise StorageError if not found (converted to Python exception via PyO3)

**Signature**:
```python
def validate_dataset_exists(dataset_id: DatasetId) -> bool:
```

**Responsibility**: Check if a dataset exists in storage.

**Input**:
- `dataset_id`: Dataset identifier to check

**Output**:
- `bool`: True if dataset exists and is valid, False otherwise

**Behavior**:
1. Check if directory `~/.trading/datasets/{dataset_id}/` exists
2. Check if `metadata.json` file exists in that directory
3. Attempt to read and validate metadata using `read_dataset_metadata` (catch exceptions)
4. Return True if all checks pass, False otherwise

**Error Handling**:
- Return False (don't raise) if dataset doesn't exist or is invalid
- Log warnings for invalid datasets

**Dependencies**:
- `trading.types.DatasetId`
- `pathlib.Path`
- `read_dataset_metadata` function

---

### 9.2 Account Helper Functions

#### Function: `calculate_account_equity` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.calculate_account_equity`)

**Rust Implementation Notes**:
- Core equity calculation in Rust for performance
- Accept Account and current prices (dict from Python)
- Return float to Python

**Signature**:
```python
def calculate_account_equity(
    account: Account,
    current_prices: dict[Symbol, float]
) -> float:
```

**Responsibility**: Calculate total account equity (cash + positions at mark-to-market).

**Behavior**:
1. Start with `account.cleared_balance + account.pending_balance`
2. For each position in `account.positions`:
   - Get current price from `current_prices` dict
   - Add `position.quantity * current_price` to equity
3. Return total equity

**Error Handling**:
- Use cost basis if current price not available (log warning)
- Return 0 if account is invalid

---

#### Function: `process_clearing` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.process_clearing`)

**Rust Implementation Notes**:
- Core clearing logic in Rust
- Update positions, balances, pending quantities
- Handle business day logic if `use_business_days` is True

**Signature**:
```python
def process_clearing(
    account: Account,
    current_timestamp: datetime
) -> None:
```

**Responsibility**: Process any pending transactions that should be cleared.

**Input**:
- `account`: Account to update (modified in place)
- `current_timestamp`: Current simulation timestamp

**Output**:
- None (side effect: modifies account)

**Behavior**:
1. For each pending transaction in account:
   - Calculate clearing time:
     - If `account.use_business_days`:
       - Use `next_business_day(transaction.timestamp, account.clearing_delay_hours)`
     - Else:
       - Use `transaction.timestamp + timedelta(hours=account.clearing_delay_hours)`
   - If clearing time <= current_timestamp:
     - Move amount from `pending_balance` to `cleared_balance`
     - Move `pending_quantity` to regular `quantity` in position
     - Remove transaction from pending list

**Design Decision**: Account maintains internal list of pending transactions. This function is called periodically during training. Business day calculation is optional and uses calendar days by default.

**Error Handling**:
- Log warnings for clearing errors but continue processing

**Dependencies**:
- `trading.types.Account`
- `datetime.timedelta`
- `is_business_day` function (if business days enabled)

---

#### Function: `is_business_day` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.is_business_day`)

**Rust Implementation Notes**:
- Use `chrono` crate for date handling
- Check against US market holidays (hardcoded list or use `calendars` crate)
- Return bool to Python

**Signature**:
```python
def is_business_day(date: datetime) -> bool:
```

**Responsibility**: Check if a date is a business day (excludes weekends and US holidays).

**Input**:
- `date`: Datetime to check

**Output**:
- `bool`: True if business day, False otherwise

**Behavior**:
1. Check if date is weekend (Saturday or Sunday) - return False
2. Check if date is US market holiday (New Year's, Independence Day, Christmas, etc.) - return False
3. Return True otherwise

**Design Decision**: Uses US market holidays. Can be extended to support other markets.

**Error Handling**:
- Return True for unknown dates (fail open)

**Dependencies**:
- `datetime` module
- `pandas.tseries.holiday` or custom holiday list (add to dependencies if using pandas)

---

#### Function: `next_business_day` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.next_business_day`)

**Rust Implementation Notes**:
- Use `chrono` crate for date arithmetic
- Skip weekends and holidays
- Return datetime to Python (via PyO3 datetime conversion)

**Signature**:
```python
def next_business_day(start_date: datetime, days: int) -> datetime:
```

**Responsibility**: Calculate the Nth business day after a start date.

**Input**:
- `start_date`: Starting date
- `days`: Number of business days to advance

**Output**:
- `datetime`: Date that is N business days after start_date

**Behavior**:
1. Start from `start_date`
2. Advance one day at a time
3. Skip weekends and holidays (use `is_business_day`)
4. Count only business days
5. Return date after advancing N business days

**Error Handling**:
- Raise `AccountError` if days < 0

**Dependencies**:
- `is_business_day` function
- `trading.exceptions.AccountError`

---

#### Function: `reserve_funds` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.reserve_funds`)

**Rust Implementation Notes**:
- Update Account's reserved_balance in Rust
- Validate available balance
- Return updated Account to Python

**Signature**:
```python
def reserve_funds(account: Account, amount: float) -> Account:
```

**Responsibility**: Reserve funds in account for a pending order.

**Input**:
- `account`: Account to modify
- `amount`: Amount to reserve

**Output**:
- `Account`: Updated account (new instance or modified in place)

**Behavior**:
1. Check that `account.cleared_balance - account.reserved_balance >= amount`
2. If sufficient:
   - Increase `account.reserved_balance` by `amount`
   - Return updated account
3. If insufficient:
   - Raise `AccountError`

**Error Handling**:
- Raise `AccountError` if insufficient cleared balance

**Dependencies**:
- `trading.types.Account`
- `trading.exceptions.AccountError`

---

#### Function: `release_reservation` **[RUST]**

**Location**: `rust/src/lib.rs` (exposed as `trading._core.release_reservation`)

**Rust Implementation Notes**:
- Decrease Account's reserved_balance
- Validate reserved_balance >= amount
- Return updated Account to Python

**Signature**:
```python
def release_reservation(account: Account, amount: float) -> Account:
```

**Responsibility**: Release previously reserved funds.

**Input**:
- `account`: Account to modify
- `amount`: Amount to release

**Output**:
- `Account`: Updated account

**Behavior**:
1. Check that `account.reserved_balance >= amount`
2. Decrease `account.reserved_balance` by `amount`
3. Return updated account

**Error Handling**:
- Raise `AccountError` if reserved_balance < amount

**Dependencies**:
- `trading.types.Account`
- `trading.exceptions.AccountError`

---

### 9.3 Strategy Base Class

**Location**: `src/trading/strategies/base.py` **[PYTHON]**

**Note**: Strategy base class stays in Python for ML/RL integration.

**Signature**:
```python
class Strategy(ABC):
    @abstractmethod
    def decide(
        self,
        snapshot: AnalysisSnapshot,
        account: Account
    ) -> list[OrderRequest]:
        """Generate order requests based on market snapshot and account state."""
        pass
```

**Responsibility**: Abstract base class for all trading strategies.

**Design Decision**: Simple interface - strategies receive snapshot and account, return order requests. No state management in base class (strategies can maintain their own state if needed).

---

## 10. Implementation Order and Dependencies

Recommended implementation order:

1. ✅ **Type definitions** (`src/trading/types.py` **[PYTHON]**): Add all missing types (OrderStatus, Order, DatasetBundle, Gap, RewardSignal, RunProgress). Types are Python Pydantic models but may have Rust equivalents for performance. **COMPLETED**: All types migrated to Pydantic BaseModel with comprehensive tests.
2. ✅ **Exception definitions** (`src/trading/exceptions.py` **[PYTHON]**): Create exception hierarchy. Rust errors convert to Python exceptions via PyO3. **COMPLETED**: ConfigError, DataSourceError, DataValidationError, StorageError, StrategyError added with tests.
3. **Data normalization** (`rust/src/lib.rs` **[RUST]**): `normalize_bars`, `normalize_synth_bars`, `normalize_timezone` - all in Rust for performance
4. **Data validation** (`rust/src/lib.rs` **[RUST]**): `validate_bars`, `detect_data_gaps`, `fill_data_gaps` - all in Rust
5. **Data storage** (`rust/src/lib.rs` **[RUST]**): `store_dataset`, `load_datasets`, `read_dataset_metadata`, `list_datasets`, `validate_dataset_exists` - Parquet I/O in Rust
6. **Data sources** (`src/trading/data/sources.py` **[PYTHON]**): `resolve_data_source`, `fetch_bars` implementations - external API integration stays in Python
7. **Synthetic data** (`rust/src/lib.rs` **[RUST]** for generation, `src/trading/data/synthetic.py` **[PYTHON]** for config): `build_synth_generator` (Python), `generate_synth_bars` (Rust)
8. ✅ **Account management** (`rust/src/lib.rs` **[RUST]**): `initialize_account`, `update_account_from_executions`, `calculate_account_equity`, `process_clearing`, `is_business_day`, `next_business_day`, `reserve_funds`, `release_reservation` - all core account logic in Rust. **PARTIAL**: `calculate_account_equity`, `process_clearing`, `is_business_day`, `next_business_day`, `reserve_funds`, `release_reservation` implemented and tested.
9. **Risk constraints** (`rust/src/lib.rs` **[RUST]**): `apply_risk_constraints` (with universe support) - validation logic in Rust
10. **Execution** (`rust/src/lib.rs` **[RUST]**): `create_order`, `cancel_order`, `execute_orders` - execution engine in Rust
11. **Strategy base** (`src/trading/strategies/base.py` **[PYTHON]**): `Strategy` ABC with optional `update_from_reward` method - stays in Python for ML/RL
12. **Strategy registry** (`src/trading/strategies/registry.py` **[PYTHON]**): `resolve_strategy_class`, `instantiate_strategy` - Python for flexibility
13. **Training components** (`rust/src/lib.rs` **[RUST]** for iteration, `src/trading/training/` **[PYTHON]** for orchestration): `iteration_schedule` (Rust), `build_analysis_snapshot` (Python wrapper), `checkpoint_run_state` (Rust), `resume_training_run` (Rust)
14. **Training logging** (`src/trading/training/logging.py` **[PYTHON]**): `log_event` - Python for flexibility
15. **Training monitoring** (`src/trading/training/monitoring.py` **[PYTHON]**): `get_run_progress` - Python wrapper
16. **Metrics** (`rust/src/lib.rs` **[RUST]**): `compute_run_metrics` (with advanced metrics) - statistical calculations in Rust
17. **Configuration validation** (`src/trading/commands/run_training.py` **[PYTHON]**): `validate_training_config` - Python, calls Rust for dataset validation
18. **Commands** (`src/trading/commands/` **[PYTHON]**): All command functions - Python orchestrators that call Rust core
19. **CLI entry point** (`src/trading/main.py` **[PYTHON]**): `main()` function - Python CLI

---

## 11. Testing Strategy

Each function should have corresponding unit tests in `tests/` directory:
- Test happy path
- Test error conditions
- Test edge cases (empty inputs, None values, etc.)
- Use pytest fixtures for common test data
- Mock external dependencies (file I/O, network calls)

Test file structure mirrors source structure:
- `tests/test_types.py` - Type tests
- `tests/test_data_normalize.py` - Normalization tests
- `tests/test_data_storage.py` - Storage tests
- etc.

---

## 12. Dependencies to Add

Add to `pyproject.toml` dependencies:
- ✅ `pyyaml>=6.0.3` (already present)
- ✅ `pytest>=9.0.2` (already present)
- ✅ `pydantic>=2.0.0` (for data models and validation) **ADDED**
- ✅ `ruff>=0.8.0` (for linting) **ADDED**
- `pyarrow>=14.0.0` or `pandas>=2.0.0` (for Parquet I/O)
- `yfinance>=0.2.0` (for Yahoo Finance data source)
- `numpy>=1.24.0` (for synthetic data generation and metrics)

---

#### Function: `validate_training_config` **[PYTHON]**

**Location**: `src/trading/commands/run_training.py`

**Note**: Configuration validation stays in Python, but can call Rust functions to validate dataset existence.

**Signature**:
```python
def validate_training_config(config: TrainingConfig) -> list[str]:
```

**Responsibility**: Perform comprehensive validation on training configuration and return list of errors.

**Input**:
- `config`: `TrainingConfig` object to validate

**Output**:
- `list[str]`: List of validation error messages (empty if valid)

**Behavior**:
1. Initialize empty errors list
2. Validate datasets:
   - Check that datasets list is not empty
   - For each dataset_id, call `validate_dataset_exists` - add error if not found
3. Validate strategy:
   - Try to resolve strategy class using `resolve_strategy_class(config.strategy_class_path)` - add error if fails
   - Validate strategy_params is a dict
4. Validate account settings:
   - Check starting_balance > 0
   - Check clearing_delay_hours >= 0
   - Check base_currency is non-empty string
5. Validate risk settings:
   - Check max_position_size is None or > 0
   - Check max_leverage > 0
6. Return errors list

**Error Handling**:
- Catch exceptions from `resolve_strategy_class` and add to errors list
- Don't raise exceptions - return all errors for user to fix

**Dependencies**:
- `trading.types.TrainingConfig`
- `trading.data.storage.validate_dataset_exists`
- `trading.strategies.registry.resolve_strategy_class`

---

## 13. Known Limitations and Future Enhancements

### 13.1 Current Limitations (MVP)

The following architectural features are intentionally simplified or deferred for the initial MVP:

- **Single strategy per run**: No strategy orchestration, switching, or blending (architecture section 3.3) - **IMPLEMENTED**: Functions documented but marked as future enhancement
- **Market orders only**: No limit orders, stop orders, or other order types - **IMPLEMENTED**: Order type system in place, only market orders execute
- **Immediate execution**: No order queuing, partial fills - **IMPLEMENTED**: Orders execute immediately, full fills only
- **Order cancellation**: Basic cancellation support - **IMPLEMENTED**: `cancel_order` function exists
- **Calendar-day clearing**: Default is calendar days, but business day support - **IMPLEMENTED**: `use_business_days` flag and `is_business_day` function added
- **Order reservations**: Basic reservation system - **IMPLEMENTED**: `reserve_funds` and `release_reservation` functions added
- **Universe distinction**: Analysis vs tradable universes - **IMPLEMENTED**: `analysis_universe` and `tradable_universe` in TrainingConfig
- **No live data**: Only historical and synthetic data sources supported - **DEFERRED**: Placeholder added for future
- **Checkpoint/resume**: Checkpoint system - **IMPLEMENTED**: `checkpoint_run_state` and `resume_training_run` functions added
- **Structured event logging**: Event logging - **IMPLEMENTED**: `log_event` function and `enable_event_logging` config option
- **Advanced metrics**: Extended metrics - **IMPLEMENTED**: Sortino, Calmar, profit factor, consecutive wins/losses added
- **Data gap handling**: Gap detection and filling - **IMPLEMENTED**: `detect_data_gaps` and `fill_data_gaps` functions added
- **Timezone handling**: Timezone normalization - **IMPLEMENTED**: `normalize_timezone` function added

### 13.2 Planned Enhancements (Post-MVP)

These features align with the architecture document but are deferred:

- **Strategy orchestration**: Multi-strategy evaluation, selection, rotation, and blending - **DOCUMENTED**: Functions specified but not implemented
- **Order lifecycle management**: Partial fills, order queuing - **PARTIAL**: Order states and cancellation implemented, partial fills deferred
- **Live data sources**: Real-time or near-real-time data integration - **PLACEHOLDER**: Function signature documented
- **Reinforcement learning hooks**: Strategy update methods - **IMPLEMENTED**: `update_from_reward` method added to Strategy base class
- **Multiple accounts**: Support for multiple concurrent accounts - **DEFERRED**: Single account per run remains

### 13.3 Design Decisions Documented

- **Simplified execution model**: Market orders execute immediately at bar close price (no slippage, no order book simulation) - sufficient for backtesting MVP
- **Single account**: Architecture mentions "primary training account" - MVP implements single account per run, multi-account support deferred
- **No partial fills**: Simulation assumes full order execution - realistic for market orders in liquid instruments
- **Calendar-day clearing**: Sufficient for MVP where training data may not include weekends; business-day logic can be added later without breaking changes

---

## 14. Directory Structure

**Hybrid Rust + Python Structure**:

```
trading/
  rust/                    # Rust core library
    Cargo.toml
    src/
      lib.rs              # Main Rust module (PyO3 bindings)
      data.rs             # Data processing functions
      account.rs          # Account management
      execution.rs        # Order execution
      metrics.rs          # Metrics computation
      storage.rs          # Parquet I/O and storage
  src/                    # Python package
    trading/
      __init__.py         # Package init, imports Rust extension
      main.py             # CLI entry point
      types.py            # Pydantic models
      exceptions.py       # Exception hierarchy
      commands/           # CLI command implementations
        fetch_data.py
        gen_synth.py
        run_training.py
        inspect_run.py
      data/               # Data source integration (Python)
        sources.py
        synthetic.py
      strategies/         # Strategy framework (Python)
        base.py
        registry.py
      training/           # Training orchestration (Python)
        snapshot.py
        strategy_executor.py
        recording.py
        logging.py
        monitoring.py
  tests/                  # Test suite
    test_types.py
    test_rust_integration.py
    ...
  docs/                   # Documentation
    architecture.md
    cli-and-implementation.md
    design-analysis.md
  pyproject.toml          # Python project config (Maturin build)
  README.md
```

**Key Points**:
- Rust code in `rust/` directory, exposed via PyO3 as `trading._core` module
- Python code in `src/trading/` directory
- Python imports Rust functions: `from trading._core import normalize_bars, ...`
- Build system: Maturin (configured in `pyproject.toml`)
  types.py
  exceptions.py
  commands/
    __init__.py
    fetch_data.py
    gen_synth.py
    run_training.py
    inspect_run.py
  data/
    __init__.py
    normalize.py
    validate.py
    storage.py
    sources.py
    synthetic.py
  account/
    __init__.py
    account.py
  risk/
    __init__.py
    constraints.py
  execution/
    __init__.py
    simulator.py
  strategies/
    __init__.py
    base.py
    registry.py
    trend_following.py  # Example strategy implementation
  training/
    __init__.py
    scheduler.py
    snapshot.py
    strategy_executor.py
    recording.py
    storage.py
  metrics/
    __init__.py
    compute.py
```

---

## 15. Rust Implementation Guidelines

### 15.1 PyO3 Integration

All Rust functions exposed to Python must:
- Use `#[pyfunction]` macro for standalone functions
- Use `#[pymodule]` macro for module initialization
- Return `PyResult<T>` for error handling (converts to Python exceptions)
- Accept Python types via PyO3 bindings (`PyAny`, `PyDict`, `PyList`, etc.)
- Convert Rust types to Python using `IntoPy<PyObject>` trait

**Example**:
```rust
use pyo3::prelude::*;

#[pyfunction]
fn normalize_bars(py: Python, bars: &PyAny) -> PyResult<PyObject> {
    // Convert Python iterator to Rust Vec
    // Process in Rust
    // Return Python iterator
}
```

### 15.2 Type Conversions

- **Pydantic Models**: Rust structs should mirror Python Pydantic models. Use `serde` for serialization/deserialization.
- **Datetimes**: Use `chrono` crate, convert to/from Python `datetime` via PyO3
- **Collections**: Convert Python lists/dicts to Rust Vec/HashMap using PyO3 bindings
- **Errors**: Define Rust error types, convert to Python exceptions using `PyErr`

### 15.3 Performance Considerations

- **Parquet I/O**: Use `polars` or `arrow` crate for high-performance Parquet operations
- **Iterators**: Prefer Rust iterators over collecting into Vec when possible
- **Memory**: Minimize data copying between Python and Rust
- **Parallelization**: Use `rayon` crate for parallel processing where beneficial

### 15.4 Rust Dependencies

Add to `rust/Cargo.toml`:
- `pyo3 = { version = "0.21", features = ["extension-module"] }`
- `serde = { version = "1.0", features = ["derive"] }`
- `serde_json = "1.0"`
- `chrono = { version = "0.4", features = ["serde"] }`
- `polars = "0.40"` or `arrow = "50"` (for Parquet)
- `uuid = { version = "1.0", features = ["v4"] }` (for order IDs)
- `rand = "0.8"` (for synthetic data)
- `walkdir = "2"` (for directory scanning)

---

This specification provides complete function-level details for implementing the trading bot system. Each function can be implemented independently by a low-parameter-count AI agent given this specification and the type definitions.
