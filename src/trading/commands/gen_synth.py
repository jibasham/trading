"""Configuration and execution for the gen-synth command.

Example config file (gen_synth.yaml):

    symbols:
      - "SYNTH_A"
      - "SYNTH_B"
    date_range:
      start: "2020-01-01"
      end: "2024-01-01"
    granularity: "1d"
    generator_type: "geometric_brownian"
    generator_params:
      initial_price: 100.0
      drift: 0.0001
      volatility: 0.02
    random_seed: 42
    dataset_id: "synth_gb_2020_2024"  # Optional
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading.exceptions import ConfigError
from trading.types import DatasetId, DateRange, GenSynthConfig, Symbol

# Valid granularities for synthetic data
VALID_GRANULARITIES = frozenset([
    "1m", "5m", "15m", "30m", "1h", "1d",
])

# Valid generator types
VALID_GENERATOR_TYPES = frozenset([
    "geometric_brownian",  # Standard GBM for stock-like behavior
    "mean_reverting",      # Ornstein-Uhlenbeck process
    "jump_diffusion",      # GBM with jumps
])


def _parse_datetime(value: str | datetime) -> datetime:
    """Parse a datetime string or pass through datetime objects.

    :param value: ISO format string or datetime object.
    :returns: Timezone-aware datetime (UTC if no timezone specified).
    :raises ConfigError: If parsing fails.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ConfigError(f"Invalid datetime format: {value}") from e


def _generate_dataset_id(
    symbols: list[Symbol],
    date_range: DateRange,
    granularity: str,
    generator_type: str,
    random_seed: int | None,
) -> DatasetId:
    """Generate a dataset ID from configuration parameters.

    :param symbols: List of symbols.
    :param date_range: Date range.
    :param granularity: Data granularity.
    :param generator_type: Generator type.
    :param random_seed: Random seed (included in ID for reproducibility).
    :returns: Generated dataset ID.
    """
    # Short generator type abbreviation
    gen_abbrev = {
        "geometric_brownian": "gb",
        "mean_reverting": "mr",
        "jump_diffusion": "jd",
    }.get(generator_type, generator_type[:2])

    symbols_str = "_".join(sorted(str(s) for s in symbols))
    if len(symbols_str) > 15:
        symbols_hash = hashlib.md5(symbols_str.encode()).hexdigest()[:6]
        symbols_part = f"{len(symbols)}s_{symbols_hash}"
    else:
        symbols_part = symbols_str.lower()

    start_str = date_range.start.strftime("%Y%m%d")
    end_str = date_range.end.strftime("%Y%m%d")

    seed_part = f"_s{random_seed}" if random_seed is not None else ""

    return DatasetId(f"synth_{gen_abbrev}_{symbols_part}_{granularity}_{start_str}_{end_str}{seed_part}")


def _validate_generator_params(generator_type: str, params: dict[str, Any]) -> None:
    """Validate generator-specific parameters.

    :param generator_type: Type of generator.
    :param params: Generator parameters.
    :raises ConfigError: If parameters are invalid.
    """
    if generator_type == "geometric_brownian":
        # GBM requires: initial_price, drift (mu), volatility (sigma)
        if "initial_price" in params:
            if not isinstance(params["initial_price"], (int, float)) or params["initial_price"] <= 0:
                raise ConfigError("'initial_price' must be a positive number")
        if "drift" in params:
            if not isinstance(params["drift"], (int, float)):
                raise ConfigError("'drift' must be a number")
        if "volatility" in params:
            if not isinstance(params["volatility"], (int, float)) or params["volatility"] < 0:
                raise ConfigError("'volatility' must be a non-negative number")

    elif generator_type == "mean_reverting":
        # OU process requires: mean_level, reversion_speed, volatility
        if "mean_level" in params:
            if not isinstance(params["mean_level"], (int, float)) or params["mean_level"] <= 0:
                raise ConfigError("'mean_level' must be a positive number")
        if "reversion_speed" in params:
            if not isinstance(params["reversion_speed"], (int, float)) or params["reversion_speed"] <= 0:
                raise ConfigError("'reversion_speed' must be a positive number")

    elif generator_type == "jump_diffusion":
        # Jump diffusion: GBM params + jump_intensity, jump_mean, jump_std
        if "jump_intensity" in params:
            if not isinstance(params["jump_intensity"], (int, float)) or params["jump_intensity"] < 0:
                raise ConfigError("'jump_intensity' must be a non-negative number")


def load_gen_synth_config(config_path: str | Path) -> GenSynthConfig:
    """Parse and validate a gen-synth configuration file.

    :param config_path: Path to YAML configuration file.
    :returns: Validated GenSynthConfig object.
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
    required_fields = ["symbols", "date_range", "granularity", "generator_type"]
    for field in required_fields:
        if field not in raw_config:
            raise ConfigError(f"Missing required field: {field}")

    # Parse symbols
    raw_symbols = raw_config["symbols"]
    if not isinstance(raw_symbols, list) or len(raw_symbols) == 0:
        raise ConfigError("'symbols' must be a non-empty list")
    symbols = [Symbol(s) for s in raw_symbols]

    # Parse date_range
    raw_date_range = raw_config["date_range"]
    if not isinstance(raw_date_range, dict):
        raise ConfigError("'date_range' must be a mapping with 'start' and 'end'")
    if "start" not in raw_date_range or "end" not in raw_date_range:
        raise ConfigError("'date_range' must contain 'start' and 'end'")

    start_dt = _parse_datetime(raw_date_range["start"])
    end_dt = _parse_datetime(raw_date_range["end"])

    if start_dt >= end_dt:
        raise ConfigError("'date_range.start' must be before 'date_range.end'")

    date_range = DateRange(start=start_dt, end=end_dt)

    # Parse granularity
    granularity = raw_config["granularity"]
    if granularity not in VALID_GRANULARITIES:
        raise ConfigError(
            f"Invalid granularity '{granularity}'. "
            f"Valid options: {sorted(VALID_GRANULARITIES)}"
        )

    # Parse generator_type
    generator_type = raw_config["generator_type"]
    if generator_type not in VALID_GENERATOR_TYPES:
        raise ConfigError(
            f"Invalid generator_type '{generator_type}'. "
            f"Valid options: {sorted(VALID_GENERATOR_TYPES)}"
        )

    # Parse generator_params (optional)
    generator_params: dict[str, Any] = raw_config.get("generator_params", {})
    if not isinstance(generator_params, dict):
        raise ConfigError("'generator_params' must be a mapping")
    _validate_generator_params(generator_type, generator_params)

    # Parse random_seed (optional)
    random_seed: int | None = raw_config.get("random_seed")
    if random_seed is not None and not isinstance(random_seed, int):
        raise ConfigError("'random_seed' must be an integer")

    # Parse or generate dataset_id
    dataset_id: DatasetId | None = None
    if "dataset_id" in raw_config:
        dataset_id = DatasetId(raw_config["dataset_id"])
    else:
        dataset_id = _generate_dataset_id(
            symbols, date_range, granularity, generator_type, random_seed
        )

    return GenSynthConfig(
        symbols=symbols,
        date_range=date_range,
        granularity=granularity,
        generator_type=generator_type,
        generator_params=generator_params,
        random_seed=random_seed,
        dataset_id=dataset_id,
    )



