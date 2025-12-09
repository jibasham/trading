# CLI and Implementation Overview

This document describes the minimal command-line interface (CLI) for the trading bot and outlines an implementation approach that breaks work down into small, function-sized units suitable for assignment to small AI models.

## 1. Design Principles for Implementation

- [ ] **Library first, CLI second**: Core behavior lives in a Python library; the CLI is a thin wrapper.
- [ ] **Small work items**: Implementation is decomposed into very small steps; ideally, each work item corresponds to implementing or modifying a **single function**.
- [ ] **Clear responsibilities**: Each function should have a single clear responsibility with well-defined inputs and outputs.
- [ ] **Config-driven behavior**: CLI commands largely translate configuration files and arguments into calls into the library.

## 2. Minimal CLI Command Set (Conceptual)

The initial CLI is conceptually named `trading` and provides these core commands:

- [ ] `trading fetch-data` — Aggregate and store historical market data locally.
- [ ] `trading gen-synth` — Generate and store synthetic datasets based on configurable seeds/parameters.
- [ ] `trading run-training` — Run a training/simulation cycle over specified datasets using one or more strategies.
- [ ] `trading inspect-run` — Inspect results and metrics from a previous run.

These can be extended later (e.g., `run-daemon` for long-lived processes), but this set is sufficient to exercise the core architecture.

## 3. Command: `trading fetch-data`

### 3.1 High-Level Behavior

- [ ] Read a configuration file describing:
  - Symbols to fetch (analysis universe and tradable universe if relevant).
  - Date range and granularity (e.g., 5-minute bars).
  - Data source parameters (e.g., provider identifiers, credentials, or local CSV/Parquet paths).
- [ ] Fetch or load the data.
- [ ] Normalize and validate it into the internal format.
- [ ] Store it in the local dataset store, updating dataset metadata.

### 3.2 Example Small Work Items (Function-Sized)

Each bullet below is intended to correspond to one function or a very small cluster of related functions.

- [ ] `load_fetch_data_config(path) -> FetchDataConfig`
  - Responsibility: Parse and validate the config file for `fetch-data`.
- [ ] `resolve_data_source(config: FetchDataConfig) -> DataSource`
  - Responsibility: Construct a data source abstraction from config (e.g., historical provider or local files).
- [ ] `fetch_bars(data_source, symbols, date_range, granularity) -> Iterable[Bar]`
  - Responsibility: Retrieve raw bar data for the requested universe and time span.
- [ ] `normalize_bars(raw_bars) -> Iterable[NormalizedBar]`
  - Responsibility: Convert provider-specific bar format into the internal normalized bar representation.
- [ ] `validate_bars(normalized_bars) -> None`
  - Responsibility: Perform basic sanity checks (e.g., no negative prices, timestamp ordering).
- [ ] `store_dataset(normalized_bars, dataset_id, metadata) -> None`
  - Responsibility: Persist the dataset to local storage and update metadata.
- [ ] `run_fetch_data_command(args) -> int`
  - Responsibility: Glue function invoked by the CLI; orchestrates the above steps using command-line arguments.

## 4. Command: `trading gen-synth`

### 4.1 High-Level Behavior

- [ ] Read a configuration file describing synthetic data generation:
  - Target symbol(s) (e.g., QQQ-only or small tradable universe).
  - Time range, granularity, and sampling frequency.
  - Generation parameters and random seed.
- [ ] Use a synthetic data generator to produce a series that statistically resembles real markets for the chosen universe.
- [ ] Normalize it into the same internal bar format as historical data.
- [ ] Store it in the local dataset store with appropriate metadata (including generation parameters and seed).

### 4.2 Example Small Work Items (Function-Sized)

- [ ] `load_gen_synth_config(path) -> GenSynthConfig`
- [ ] `build_synth_generator(config: GenSynthConfig) -> SynthGenerator`
- [ ] `generate_synth_bars(generator: SynthGenerator) -> Iterable[Bar]`
- [ ] `normalize_synth_bars(raw_bars) -> Iterable[NormalizedBar]`
- [ ] `store_synth_dataset(bars, dataset_id, metadata) -> None`
- [ ] `run_gen_synth_command(args) -> int`

Each function handles one responsibility: interpreting config, constructing a generator, generating data, normalizing, storing, or gluing together for the CLI.

## 5. Command: `trading run-training`

### 5.1 High-Level Behavior

- [ ] Read an experiment configuration describing:
  - Datasets to use (historical and/or synthetic), including analysis vs. tradable universes.
  - Strategy class and parameters (Python class path + parameter dict).
  - Account settings (starting balance, clearing delay, risk constraints).
  - Training run settings (e.g., run ID, logging, evaluation metrics to compute).
- [ ] Load datasets and step through them as if in real time.
- [ ] At each time slice:
  - Build an analysis snapshot from the analysis universe.
  - Provide the strategy with the current snapshot and account state.
  - Receive desired orders from the strategy.
  - Apply risk and account constraints, including opportunity cost and clearing behavior.
  - Update account and record trades.
- [ ] At the end of the run, compute metrics and store run artifacts.

### 5.2 Example Small Work Items (Function-Sized)

- [ ] `load_training_config(path) -> TrainingConfig`
- [ ] `load_datasets(config: TrainingConfig) -> DatasetBundle`
- [ ] `resolve_strategy_class(path: str) -> Type[Strategy]`
- [ ] `instantiate_strategy(cls: Type[Strategy], params: dict) -> Strategy`
- [ ] `initialize_account(config: TrainingConfig) -> Account`
- [ ] `iteration_schedule(dataset_bundle) -> Iterable[TimeSlice]`
- [ ] `build_analysis_snapshot(time_slice, dataset_bundle) -> AnalysisSnapshot`
- [ ] `strategy_decide(strategy: Strategy, snapshot: AnalysisSnapshot, account: Account) -> List[OrderRequest]`
- [ ] `apply_risk_constraints(order_requests, account, config) -> List[OrderRequest]`
- [ ] `execute_orders(order_requests, account, time_slice) -> List[Execution]`
- [ ] `update_account_from_executions(account, executions, time_slice) -> None`
- [ ] `record_training_step(run_state, time_slice, snapshot, orders, executions, account) -> None`
- [ ] `compute_run_metrics(run_state) -> RunMetrics`
- [ ] `store_run_results(run_id, run_state, metrics) -> None`
- [ ] `run_training_command(args) -> int`

Each of these is designed to be a self-contained, single-purpose function that a small AI model could implement in isolation, given the relevant type definitions and context.

## 6. Command: `trading inspect-run`

### 6.1 High-Level Behavior

- [ ] Accept a run identifier (or path) as input.
- [ ] Load run artifacts (e.g., logs, trades, metrics) from storage.
- [ ] Present a textual summary in the CLI and optionally export structured data.

### 6.2 Example Small Work Items (Function-Sized)

- [ ] `parse_inspect_run_args(args) -> InspectRunRequest`
- [ ] `load_run_artifacts(run_id) -> RunArtifacts`
- [ ] `summarize_run_metrics(metrics: RunMetrics) -> str`
- [ ] `format_run_summary(artifacts: RunArtifacts) -> str`
- [ ] `print_run_summary(summary: str) -> None`
- [ ] `export_run_data(artifacts: RunArtifacts, output_path: str) -> None`
- [ ] `run_inspect_run_command(args) -> int`

## 7. Next Steps

- [ ] Define the core data types (e.g., `Bar`, `NormalizedBar`, `Account`, `OrderRequest`, `Execution`, `RunMetrics`) at a conceptual level.
- [ ] For each command, take the proposed function list and refine names, inputs, and outputs into a small, coherent module interface.
- [ ] Use this document as a backlog: each bullet point under "Example Small Work Items" can become a discrete implementation task for a small AI model or worker.
