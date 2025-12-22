"""Configuration and execution for the fetch-data command.

Example config file (fetch_data.yaml):

    symbols:
      - "QQQ"
      - "SPY"
    date_range:
      start: "2020-01-01"
      end: "2024-01-01"
    granularity: "1d"
    data_source: "yahoo"
    source_params: {}
    dataset_id: "qqq_spy_1d_2020_2024"  # Optional
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading.exceptions import ConfigError
from trading.types import DatasetId, DateRange, FetchDataConfig, Symbol

# Valid granularities supported by data sources
VALID_GRANULARITIES = frozenset([
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
])

# Valid data source types
VALID_DATA_SOURCES = frozenset(["yahoo", "local", "csv"])


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
        # Try ISO format first
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Try simple date format
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ConfigError(f"Invalid datetime format: {value}") from e


def _generate_dataset_id(
    symbols: list[Symbol],
    date_range: DateRange,
    granularity: str,
    data_source: str,
) -> DatasetId:
    """Generate a dataset ID from configuration parameters.

    Format: {source}_{symbols_hash}_{granularity}_{start}_{end}

    :param symbols: List of symbols.
    :param date_range: Date range.
    :param granularity: Data granularity.
    :param data_source: Data source type.
    :returns: Generated dataset ID.
    """
    # Create a short hash of symbols for uniqueness
    symbols_str = "_".join(sorted(str(s) for s in symbols))
    if len(symbols_str) > 20:
        symbols_hash = hashlib.md5(symbols_str.encode()).hexdigest()[:8]
        symbols_part = f"{len(symbols)}syms_{symbols_hash}"
    else:
        symbols_part = symbols_str.lower()

    start_str = date_range.start.strftime("%Y%m%d")
    end_str = date_range.end.strftime("%Y%m%d")

    return DatasetId(f"{data_source}_{symbols_part}_{granularity}_{start_str}_{end_str}")


def load_fetch_data_config(config_path: str | Path) -> FetchDataConfig:
    """Parse and validate a fetch-data configuration file.

    :param config_path: Path to YAML configuration file.
    :returns: Validated FetchDataConfig object.
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
    required_fields = ["symbols", "date_range", "granularity", "data_source"]
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

    # Parse data_source
    data_source = raw_config["data_source"]
    if data_source not in VALID_DATA_SOURCES:
        raise ConfigError(
            f"Invalid data_source '{data_source}'. "
            f"Valid options: {sorted(VALID_DATA_SOURCES)}"
        )

    # Parse source_params (optional)
    source_params: dict[str, Any] = raw_config.get("source_params", {})
    if not isinstance(source_params, dict):
        raise ConfigError("'source_params' must be a mapping")

    # Parse or generate dataset_id
    dataset_id: DatasetId | None = None
    if "dataset_id" in raw_config:
        dataset_id = DatasetId(raw_config["dataset_id"])
    else:
        dataset_id = _generate_dataset_id(symbols, date_range, granularity, data_source)

    return FetchDataConfig(
        symbols=symbols,
        date_range=date_range,
        granularity=granularity,
        data_source=data_source,
        source_params=source_params,
        dataset_id=dataset_id,
    )



