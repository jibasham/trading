"""Configuration and execution for the run-training command.

Example config file (training.yaml):

    run_id: "run-001"  # Optional, auto-generated if omitted
    datasets:
      - "qqq_spy_5m_2020_2024"
      - "synth_gb_2020_2024"
    strategy:
      class_path: "trading.strategies.examples.BuyAndHoldStrategy"
      params:
        symbol: "SPY"
        quantity: 10
    account:
      starting_balance: 10000.0
      base_currency: "USD"
      clearing_delay_hours: 24
      use_business_days: false
    risk:
      max_position_size: 5000.0  # Optional
      max_leverage: 1.0
    universe:
      analysis: null  # All symbols, or list like ["SPY", "QQQ"]
      tradable: null  # All symbols, or list
    logging:
      level: "INFO"
      enable_events: true
    checkpoint:
      interval: null  # Checkpoint every N time slices, null = no checkpoints
"""

from __future__ import annotations

import hashlib
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading.exceptions import ConfigError
from trading.strategies.base import Strategy
from trading.types import DatasetId, RunId, Symbol, TrainingConfig

# Valid log levels
VALID_LOG_LEVELS = frozenset(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])


def _generate_run_id(datasets: list[DatasetId], strategy_class_path: str) -> RunId:
    """Generate a run ID from configuration parameters.

    Format: run_{timestamp}_{hash}

    :param datasets: List of dataset IDs.
    :param strategy_class_path: Strategy class path.
    :returns: Generated run ID.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    content = f"{datasets}_{strategy_class_path}"
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return RunId(f"run_{timestamp}_{content_hash}")


def _validate_strategy_class_path(class_path: str) -> None:
    """Validate that a strategy class path is valid and importable.

    :param class_path: Fully qualified class path.
    :raises ConfigError: If class path is invalid or not a Strategy subclass.
    """
    if not class_path or "." not in class_path:
        raise ConfigError(
            f"Invalid strategy class_path '{class_path}'. "
            "Must be a fully qualified path like 'module.ClassName'"
        )

    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ConfigError(f"Invalid strategy class_path format: {class_path}")

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigError(f"Cannot import module '{module_path}': {e}") from e

    if not hasattr(module, class_name):
        raise ConfigError(f"Module '{module_path}' has no attribute '{class_name}'")

    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, Strategy):
        raise ConfigError(
            f"'{class_path}' is not a Strategy subclass. "
            f"Got: {type(cls)}"
        )


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Parse and validate a training configuration file.

    :param config_path: Path to YAML configuration file.
    :returns: Validated TrainingConfig object.
    :raises ConfigError: If file cannot be read or config is invalid.
    """
    config_path = Path(config_path)

    # Read and parse YAML
    try:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}") from e

    if not isinstance(raw_config, dict):
        raise ConfigError("Configuration must be a YAML mapping")

    # Validate required fields
    required_fields = ["datasets", "strategy", "account"]
    for field in required_fields:
        if field not in raw_config:
            raise ConfigError(f"Missing required field: {field}")

    # Parse datasets
    raw_datasets = raw_config["datasets"]
    if not isinstance(raw_datasets, list) or len(raw_datasets) == 0:
        raise ConfigError("'datasets' must be a non-empty list")
    datasets = [DatasetId(d) for d in raw_datasets]

    # Parse strategy
    raw_strategy = raw_config["strategy"]
    if not isinstance(raw_strategy, dict):
        raise ConfigError("'strategy' must be a mapping")
    if "class_path" not in raw_strategy:
        raise ConfigError("'strategy.class_path' is required")

    strategy_class_path = raw_strategy["class_path"]
    _validate_strategy_class_path(strategy_class_path)

    strategy_params: dict[str, Any] = raw_strategy.get("params", {})
    if not isinstance(strategy_params, dict):
        raise ConfigError("'strategy.params' must be a mapping")

    # Parse account
    raw_account = raw_config["account"]
    if not isinstance(raw_account, dict):
        raise ConfigError("'account' must be a mapping")
    if "starting_balance" not in raw_account:
        raise ConfigError("'account.starting_balance' is required")

    starting_balance = float(raw_account["starting_balance"])
    if starting_balance <= 0:
        raise ConfigError("'account.starting_balance' must be positive")

    base_currency = raw_account.get("base_currency", "USD")
    if not isinstance(base_currency, str):
        raise ConfigError("'account.base_currency' must be a string")

    clearing_delay_hours = raw_account.get("clearing_delay_hours", 24)
    if not isinstance(clearing_delay_hours, int) or clearing_delay_hours < 0:
        raise ConfigError("'account.clearing_delay_hours' must be a non-negative integer")

    use_business_days = raw_account.get("use_business_days", False)
    if not isinstance(use_business_days, bool):
        raise ConfigError("'account.use_business_days' must be a boolean")

    # Parse risk (optional)
    raw_risk = raw_config.get("risk", {})
    if not isinstance(raw_risk, dict):
        raise ConfigError("'risk' must be a mapping")

    max_position_size: float | None = raw_risk.get("max_position_size")
    if max_position_size is not None:
        max_position_size = float(max_position_size)
        if max_position_size <= 0:
            raise ConfigError("'risk.max_position_size' must be positive")

    max_leverage = float(raw_risk.get("max_leverage", 1.0))
    if max_leverage <= 0:
        raise ConfigError("'risk.max_leverage' must be positive")

    # Parse universe (optional)
    raw_universe = raw_config.get("universe", {})
    if not isinstance(raw_universe, dict):
        raise ConfigError("'universe' must be a mapping")

    analysis_universe: list[Symbol] | None = None
    if "analysis" in raw_universe and raw_universe["analysis"] is not None:
        analysis_universe = [Symbol(s) for s in raw_universe["analysis"]]

    tradable_universe: list[Symbol] | None = None
    if "tradable" in raw_universe and raw_universe["tradable"] is not None:
        tradable_universe = [Symbol(s) for s in raw_universe["tradable"]]

    # Parse logging (optional)
    raw_logging = raw_config.get("logging", {})
    if not isinstance(raw_logging, dict):
        raise ConfigError("'logging' must be a mapping")

    log_level = raw_logging.get("level", "INFO").upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(
            f"Invalid log level '{log_level}'. "
            f"Valid options: {sorted(VALID_LOG_LEVELS)}"
        )

    enable_event_logging = raw_logging.get("enable_events", True)
    if not isinstance(enable_event_logging, bool):
        raise ConfigError("'logging.enable_events' must be a boolean")

    # Parse checkpoint (optional)
    raw_checkpoint = raw_config.get("checkpoint", {})
    if not isinstance(raw_checkpoint, dict):
        raise ConfigError("'checkpoint' must be a mapping")

    checkpoint_interval: int | None = raw_checkpoint.get("interval")
    if checkpoint_interval is not None:
        if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
            raise ConfigError("'checkpoint.interval' must be a positive integer")

    # Parse or generate run_id
    run_id: RunId | None = None
    if "run_id" in raw_config:
        run_id = RunId(raw_config["run_id"])
    else:
        run_id = _generate_run_id(datasets, strategy_class_path)

    return TrainingConfig(
        run_id=run_id,
        datasets=datasets,
        strategy_class_path=strategy_class_path,
        strategy_params=strategy_params,
        account_starting_balance=starting_balance,
        account_base_currency=base_currency,
        clearing_delay_hours=clearing_delay_hours,
        use_business_days=use_business_days,
        risk_max_position_size=max_position_size,
        risk_max_leverage=max_leverage,
        analysis_universe=analysis_universe,
        tradable_universe=tradable_universe,
        log_level=log_level,
        checkpoint_interval=checkpoint_interval,
        enable_event_logging=enable_event_logging,
    )


def validate_training_config(config: TrainingConfig) -> list[str]:
    """Perform additional validation on a training config.

    Checks that referenced datasets exist, strategy is loadable, etc.

    :param config: Training configuration to validate.
    :returns: List of warning messages (empty if no warnings).
    :raises ConfigError: If config is invalid.
    """
    from trading._core import dataset_exists

    warnings: list[str] = []

    # Check all datasets exist
    for dataset_id in config.datasets:
        if not dataset_exists(str(dataset_id)):
            raise ConfigError(f"Dataset not found: {dataset_id}")

    # Validate strategy is loadable (already done in load_training_config)
    # but we can do additional checks here

    # Check universe consistency
    if config.tradable_universe and config.analysis_universe:
        tradable_set = set(config.tradable_universe)
        analysis_set = set(config.analysis_universe)
        if not tradable_set.issubset(analysis_set):
            extra = tradable_set - analysis_set
            warnings.append(
                f"Tradable symbols {extra} not in analysis universe - "
                "strategy won't see these symbols"
            )

    return warnings




