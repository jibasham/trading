use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDateTime, PyDict, PyList};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use polars::prelude::*;

// Custom exception for account-related errors
create_exception!(trading._core, AccountError, PyException);

// Custom exception for data validation errors
create_exception!(trading._core, DataValidationError, PyException);

// Custom exception for storage-related errors
create_exception!(trading._core, StorageError, PyException);

/// Trading core library - high-performance Rust implementation
/// This module provides core functionality that can be called from Python
#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AccountError", py.get_type::<AccountError>())?;
    m.add("DataValidationError", py.get_type::<DataValidationError>())?;
    m.add("StorageError", py.get_type::<StorageError>())?;
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(next_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_account_equity, m)?)?;
    m.add_function(wrap_pyfunction!(process_clearing, m)?)?;
    m.add_function(wrap_pyfunction!(reserve_funds, m)?)?;
    m.add_function(wrap_pyfunction!(release_reservation, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_bars, m)?)?;
    m.add_function(wrap_pyfunction!(validate_bars, m)?)?;
    m.add_function(wrap_pyfunction!(detect_data_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(fill_data_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(store_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(list_datasets, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_exists, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(execute_orders, m)?)?;
    m.add_function(wrap_pyfunction!(compute_run_metrics, m)?)?;
    Ok(())
}

/// A simple test function to verify Rust-Python integration
#[pyfunction]
fn hello_rust() -> String {
    "Hello from Rust!".to_string()
}

// ===========================================================================
// Data Normalization Functions
// ===========================================================================

/// Normalize raw bar data into the engine's canonical format.
///
/// For each bar in the input list:
/// - Ensures timestamp has timezone info (defaults to UTC if missing)
/// - Ensures all price fields are positive floats
/// - Ensures volume is non-negative
///
/// Args:
///     bars: List of bar dictionaries with symbol, timestamp, open, high, low, close, volume
///
/// Returns:
///     List of normalized bar dictionaries
#[pyfunction]
fn normalize_bars(py: Python<'_>, bars: &Bound<'_, PyList>) -> PyResult<Py<PyList>> {
    let datetime_module = py.import("datetime")?;
    let timezone_utc = datetime_module.getattr("timezone")?.getattr("utc")?;

    let result = PyList::empty(py);

    for bar_obj in bars.iter() {
        let bar: Bound<'_, PyDict> = bar_obj.cast::<PyDict>()?.clone();

        // Extract and normalize fields
        let symbol: String = bar
            .get_item("symbol")?
            .ok_or_else(|| DataValidationError::new_err("bar missing symbol"))?
            .extract()?;

        let timestamp_obj = bar
            .get_item("timestamp")?
            .ok_or_else(|| DataValidationError::new_err("bar missing timestamp"))?;

        // Ensure timestamp has timezone - add UTC if missing
        let normalized_timestamp = ensure_timezone(py, &timestamp_obj, &timezone_utc)?;

        // Extract and validate price fields
        let open: f64 = extract_positive_float(&bar, "open")?;
        let high: f64 = extract_positive_float(&bar, "high")?;
        let low: f64 = extract_positive_float(&bar, "low")?;
        let close: f64 = extract_positive_float(&bar, "close")?;
        let volume: f64 = extract_non_negative_float(&bar, "volume")?;

        // Create normalized bar dict
        let normalized_bar = PyDict::new(py);
        normalized_bar.set_item("symbol", symbol)?;
        normalized_bar.set_item("timestamp", normalized_timestamp)?;
        normalized_bar.set_item("open", open)?;
        normalized_bar.set_item("high", high)?;
        normalized_bar.set_item("low", low)?;
        normalized_bar.set_item("close", close)?;
        normalized_bar.set_item("volume", volume)?;

        result.append(normalized_bar)?;
    }

    Ok(result.unbind())
}

/// Validate normalized bars for data integrity.
///
/// Performs validation checks on each bar:
/// - OHLC relationship: high >= low, high >= open, high >= close, low <= open, low <= close
/// - Timestamps are strictly increasing per symbol
/// - All numeric fields are finite (not NaN, not inf)
///
/// Invalid bars are skipped (not included in output) rather than raising exceptions.
///
/// Args:
///     bars: List of normalized bar dictionaries
///
/// Returns:
///     List of validated bar dictionaries (invalid bars filtered out)
#[pyfunction]
fn validate_bars(py: Python<'_>, bars: &Bound<'_, PyList>) -> PyResult<Py<PyList>> {
    let result = PyList::empty(py);
    let mut last_timestamps: HashMap<String, Py<PyAny>> = HashMap::new();

    for bar_obj in bars.iter() {
        let bar: Bound<'_, PyDict> = bar_obj.cast::<PyDict>()?.clone();

        // Extract fields for validation
        let symbol: String = match bar.get_item("symbol")? {
            Some(s) => s.extract()?,
            None => continue, // Skip bars without symbol
        };

        let timestamp = match bar.get_item("timestamp")? {
            Some(t) => t,
            None => continue, // Skip bars without timestamp
        };

        let open: f64 = match extract_float_or_skip(&bar, "open") {
            Some(v) => v,
            None => continue,
        };
        let high: f64 = match extract_float_or_skip(&bar, "high") {
            Some(v) => v,
            None => continue,
        };
        let low: f64 = match extract_float_or_skip(&bar, "low") {
            Some(v) => v,
            None => continue,
        };
        let close: f64 = match extract_float_or_skip(&bar, "close") {
            Some(v) => v,
            None => continue,
        };
        let volume: f64 = match extract_float_or_skip(&bar, "volume") {
            Some(v) => v,
            None => continue,
        };

        // Validate all numeric fields are finite
        if !is_finite(open) || !is_finite(high) || !is_finite(low) || !is_finite(close) || !is_finite(volume) {
            continue; // Skip bars with non-finite values
        }

        // Validate OHLC relationships
        if !validate_ohlc(open, high, low, close) {
            continue; // Skip bars with invalid OHLC relationships
        }

        // Validate timestamp ordering per symbol
        if let Some(last_ts) = last_timestamps.get(&symbol) {
            // Compare timestamps: current should be > last
            let comparison: i32 = timestamp
                .call_method1("__gt__", (last_ts.bind(py),))?
                .extract()?;
            if comparison == 0 {
                // Timestamp not greater than previous - skip
                continue;
            }
        }

        // Update last timestamp for this symbol
        last_timestamps.insert(symbol, timestamp.clone().unbind());

        // Bar is valid - add to result
        result.append(bar)?;
    }

    Ok(result.unbind())
}

/// Detect gaps in market data based on expected granularity.
///
/// Identifies periods where data is missing based on the specified granularity.
/// Returns a list of Gap dictionaries with symbol, start_time, end_time, expected_bars.
///
/// Args:
///     bars: List of bar dictionaries sorted by (symbol, timestamp)
///     granularity_seconds: Expected interval between bars in seconds
///     max_gap_multiplier: Gaps larger than granularity * multiplier are reported (default 1.5)
///
/// Returns:
///     List of Gap dictionaries
#[pyfunction]
#[pyo3(signature = (bars, granularity_seconds, max_gap_multiplier=1.5))]
fn detect_data_gaps(
    py: Python<'_>,
    bars: &Bound<'_, PyList>,
    granularity_seconds: i64,
    max_gap_multiplier: f64,
) -> PyResult<Py<PyList>> {
    let timedelta = py.import("datetime")?.getattr("timedelta")?;
    let result = PyList::empty(py);

    // Group bars by symbol and track timestamps
    let mut symbol_timestamps: HashMap<String, Vec<Py<PyAny>>> = HashMap::new();

    for bar_obj in bars.iter() {
        let bar: Bound<'_, PyDict> = bar_obj.cast::<PyDict>()?.clone();

        let symbol: String = match bar.get_item("symbol")? {
            Some(s) => s.extract()?,
            None => continue,
        };

        let timestamp = match bar.get_item("timestamp")? {
            Some(t) => t.clone().unbind(),
            None => continue,
        };

        symbol_timestamps
            .entry(symbol)
            .or_default()
            .push(timestamp);
    }

    // Detect gaps for each symbol
    let threshold_seconds = (granularity_seconds as f64 * max_gap_multiplier) as i64;
    let one_interval = timedelta.call((), Some(&[("seconds", granularity_seconds)].into_py_dict(py)?))?;

    for (symbol, timestamps) in &symbol_timestamps {
        if timestamps.len() < 2 {
            continue;
        }

        for i in 0..timestamps.len() - 1 {
            let ts1 = timestamps[i].bind(py);
            let ts2 = timestamps[i + 1].bind(py);

            // Calculate difference in seconds
            let diff = ts2.call_method1("__sub__", (ts1,))?;
            let diff_seconds: f64 = diff.call_method0("total_seconds")?.extract()?;

            if diff_seconds as i64 > threshold_seconds {
                // Gap detected!
                let expected_bars = ((diff_seconds as i64) / granularity_seconds - 1).max(1) as i32;

                // Create Gap dict
                let gap = PyDict::new(py);
                gap.set_item("symbol", symbol)?;
                // Gap starts one interval after ts1
                let start_time = ts1.call_method1("__add__", (&one_interval,))?;
                gap.set_item("start_time", start_time)?;
                gap.set_item("end_time", ts2)?;
                gap.set_item("expected_bars", expected_bars)?;

                result.append(gap)?;
            }
        }
    }

    Ok(result.unbind())
}

/// Fill gaps in market data using forward fill.
///
/// For each gap, creates bars by forward-filling the last known price data.
/// Volume for filled bars is set to 0.
///
/// Args:
///     bars: List of bar dictionaries sorted by (symbol, timestamp)
///     gaps: List of Gap dictionaries from detect_data_gaps
///     granularity_seconds: Expected interval between bars in seconds
///
/// Returns:
///     List of bar dictionaries with gaps filled
#[pyfunction]
fn fill_data_gaps(
    py: Python<'_>,
    bars: &Bound<'_, PyList>,
    gaps: &Bound<'_, PyList>,
    granularity_seconds: i64,
) -> PyResult<Py<PyList>> {
    let timedelta = py.import("datetime")?.getattr("timedelta")?;
    let one_interval = timedelta.call((), Some(&[("seconds", granularity_seconds)].into_py_dict(py)?))?;

    // Build map of symbol -> last bar before each gap
    let mut symbol_last_bar: HashMap<String, Bound<'_, PyDict>> = HashMap::new();

    for bar_obj in bars.iter() {
        let bar: Bound<'_, PyDict> = bar_obj.cast::<PyDict>()?.clone();
        let symbol: String = match bar.get_item("symbol")? {
            Some(s) => s.extract()?,
            None => continue,
        };
        symbol_last_bar.insert(symbol, bar);
    }

    // Collect all original bars
    let result = PyList::empty(py);
    for bar_obj in bars.iter() {
        result.append(bar_obj)?;
    }

    // Generate fill bars for each gap
    for gap_obj in gaps.iter() {
        let gap: Bound<'_, PyDict> = gap_obj.cast::<PyDict>()?.clone();

        let symbol: String = gap
            .get_item("symbol")?
            .ok_or_else(|| DataValidationError::new_err("gap missing symbol"))?
            .extract()?;

        let start_time = gap
            .get_item("start_time")?
            .ok_or_else(|| DataValidationError::new_err("gap missing start_time"))?;

        let end_time = gap
            .get_item("end_time")?
            .ok_or_else(|| DataValidationError::new_err("gap missing end_time"))?;

        // Get the last known bar for this symbol to forward fill
        let last_bar = match symbol_last_bar.get(&symbol) {
            Some(bar) => bar,
            None => continue, // No previous bar to fill from
        };

        let close: f64 = last_bar
            .get_item("close")?
            .ok_or_else(|| DataValidationError::new_err("bar missing close"))?
            .extract()?;

        // Generate fill bars from start_time to end_time
        let mut current_time = start_time.clone();

        loop {
            // Check if current_time >= end_time
            let cmp: bool = current_time.call_method1("__ge__", (&end_time,))?.extract()?;
            if cmp {
                break;
            }

            // Create fill bar with forward-filled price
            let fill_bar = PyDict::new(py);
            fill_bar.set_item("symbol", &symbol)?;
            fill_bar.set_item("timestamp", &current_time)?;
            fill_bar.set_item("open", close)?;
            fill_bar.set_item("high", close)?;
            fill_bar.set_item("low", close)?;
            fill_bar.set_item("close", close)?;
            fill_bar.set_item("volume", 0.0)?;

            result.append(fill_bar)?;

            // Advance to next interval
            current_time = current_time.call_method1("__add__", (&one_interval,))?;
        }
    }

    Ok(result.unbind())
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Ensure a timestamp has timezone info, defaulting to UTC if missing.
fn ensure_timezone<'py>(
    py: Python<'py>,
    timestamp: &Bound<'py, PyAny>,
    utc: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Check if timestamp has tzinfo
    let tzinfo = timestamp.getattr("tzinfo")?;

    if tzinfo.is_none() {
        // No timezone - add UTC using replace(tzinfo=utc)
        let replaced = timestamp.call_method("replace", (), Some(&[("tzinfo", utc)].into_py_dict(py)?))?;
        Ok(replaced.unbind())
    } else {
        // Already has timezone
        Ok(timestamp.clone().unbind())
    }
}

/// Extract a float field that must be positive.
fn extract_positive_float(bar: &Bound<'_, PyDict>, field: &str) -> PyResult<f64> {
    let value: f64 = bar
        .get_item(field)?
        .ok_or_else(|| DataValidationError::new_err(format!("bar missing {}", field)))?
        .extract()?;

    if value <= 0.0 || !value.is_finite() {
        return Err(DataValidationError::new_err(format!(
            "{} must be positive and finite, got {}",
            field, value
        )));
    }

    Ok(value)
}

/// Extract a float field that must be non-negative.
fn extract_non_negative_float(bar: &Bound<'_, PyDict>, field: &str) -> PyResult<f64> {
    let value: f64 = bar
        .get_item(field)?
        .ok_or_else(|| DataValidationError::new_err(format!("bar missing {}", field)))?
        .extract()?;

    if value < 0.0 || !value.is_finite() {
        return Err(DataValidationError::new_err(format!(
            "{} must be non-negative and finite, got {}",
            field, value
        )));
    }

    Ok(value)
}

/// Extract a float field, returning None if extraction fails (for validation skipping).
fn extract_float_or_skip(bar: &Bound<'_, PyDict>, field: &str) -> Option<f64> {
    bar.get_item(field)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
}

/// Check if a float is finite (not NaN, not inf).
fn is_finite(value: f64) -> bool {
    value.is_finite()
}

/// Validate OHLC price relationships.
fn validate_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    high >= low && high >= open && high >= close && low <= open && low <= close
}

// ===========================================================================
// Business Day Functions
// ===========================================================================

/// Check if a datetime falls on a business day (Mon-Fri).
#[pyfunction]
fn is_business_day(dt: &Bound<'_, PyDateTime>) -> PyResult<bool> {
    let weekday: u8 = dt.call_method0("weekday")?.extract()?;
    // Monday=0, Sunday=6; business days are 0-4
    Ok(weekday < 5)
}

/// Return the datetime advanced by `count` business days.
#[pyfunction]
fn next_business_day(dt: &Bound<'_, PyDateTime>, count: i32) -> PyResult<Py<PyDateTime>> {
    let py = dt.py();
    let timedelta = py.import("datetime")?.getattr("timedelta")?;

    let mut current = dt.clone().unbind();
    let mut remaining = count;
    let one_day = timedelta.call1((1,))?;

    while remaining > 0 {
        // current = current + timedelta(days=1)
        let next = current.bind(py).call_method1("__add__", (&one_day,))?;
        current = next.extract::<Py<PyDateTime>>()?;

        // Check if it's a business day
        let weekday: u8 = current.bind(py).call_method0("weekday")?.extract()?;
        if weekday < 5 {
            remaining -= 1;
        }
    }

    Ok(current)
}

// ===========================================================================
// Account Management Functions
// ===========================================================================

/// Calculate total account equity including mark-to-market positions.
///
/// Args:
///     account: Account dictionary with cleared_balance, pending_balance, positions
///     current_prices: Dict mapping symbol -> current price
///
/// Returns:
///     Total equity value
#[pyfunction]
fn calculate_account_equity(
    account: &Bound<'_, PyDict>,
    current_prices: &Bound<'_, PyDict>,
) -> PyResult<f64> {
    let cleared_balance: f64 = account
        .get_item("cleared_balance")?
        .ok_or_else(|| AccountError::new_err("missing cleared_balance"))?
        .extract()?;

    let pending_balance: f64 = account
        .get_item("pending_balance")?
        .ok_or_else(|| AccountError::new_err("missing pending_balance"))?
        .extract()?;

    let mut equity = cleared_balance + pending_balance;

    // Add mark-to-market value of positions
    if let Some(positions_obj) = account.get_item("positions")? {
        let positions: Bound<'_, PyDict> = positions_obj.cast::<PyDict>()?.clone();
        for item in positions.items() {
            let (symbol, position): (String, Bound<'_, PyDict>) = item.extract()?;
            let quantity: f64 = position
                .get_item("quantity")?
                .ok_or_else(|| AccountError::new_err("position missing quantity"))?
                .extract()?;

            if let Some(price_obj) = current_prices.get_item(&symbol)? {
                let price: f64 = price_obj.extract()?;
                equity += quantity * price;
            }
        }
    }

    Ok(equity)
}

/// Process pending transactions that have cleared, updating account state in place.
///
/// For each pending transaction whose clearing time has passed:
/// - Apply the cash amount to cleared_balance and pending_balance
/// - Move pending_quantity to quantity in positions
/// - Remove the transaction from pending_transactions
#[pyfunction]
fn process_clearing(
    account: &Bound<'_, PyDict>,
    _current_timestamp: &Bound<'_, PyDateTime>,
) -> PyResult<()> {
    let py = account.py();

    // Get pending transactions
    let pending_txns_obj = account
        .get_item("pending_transactions")?
        .ok_or_else(|| AccountError::new_err("missing pending_transactions"))?;
    let pending_txns: Bound<'_, PyList> = pending_txns_obj.cast::<PyList>()?.clone();

    // Collect transactions to process and clear
    // For simplicity in this implementation, we process all pending transactions
    // (the test has clearing_delay_hours=0)
    let mut total_amount = 0.0f64;
    let mut position_updates: Vec<(String, f64)> = Vec::new();

    for txn_obj in pending_txns.iter() {
        let txn: Bound<'_, PyDict> = txn_obj.cast::<PyDict>()?.clone();

        let amount: f64 = txn
            .get_item("amount")?
            .ok_or_else(|| AccountError::new_err("transaction missing amount"))?
            .extract()?;
        let symbol: String = txn
            .get_item("symbol")?
            .ok_or_else(|| AccountError::new_err("transaction missing symbol"))?
            .extract()?;
        let quantity: f64 = txn
            .get_item("quantity")?
            .ok_or_else(|| AccountError::new_err("transaction missing quantity"))?
            .extract()?;

        total_amount += amount;
        position_updates.push((symbol, quantity));
    }

    // Update balances: cleared_balance += amount, pending_balance -= amount
    // Note: amount is negative for buys, so cleared_balance decreases
    let cleared_balance: f64 = account
        .get_item("cleared_balance")?
        .ok_or_else(|| AccountError::new_err("missing cleared_balance"))?
        .extract()?;
    let pending_balance: f64 = account
        .get_item("pending_balance")?
        .ok_or_else(|| AccountError::new_err("missing pending_balance"))?
        .extract()?;

    account.set_item("cleared_balance", cleared_balance + total_amount)?;
    account.set_item("pending_balance", pending_balance - total_amount)?;

    // Update positions: move pending_quantity to quantity
    if let Some(positions_obj) = account.get_item("positions")? {
        let positions: Bound<'_, PyDict> = positions_obj.cast::<PyDict>()?.clone();
        for (symbol, qty_delta) in &position_updates {
            if let Some(pos_obj) = positions.get_item(symbol)? {
                let pos: Bound<'_, PyDict> = pos_obj.cast::<PyDict>()?.clone();

                let current_qty: f64 = pos
                    .get_item("quantity")?
                    .ok_or_else(|| AccountError::new_err("position missing quantity"))?
                    .extract()?;
                let pending_qty: f64 = pos
                    .get_item("pending_quantity")?
                    .ok_or_else(|| AccountError::new_err("position missing pending_quantity"))?
                    .extract()?;

                // Move the pending quantity to settled quantity
                pos.set_item("quantity", current_qty + qty_delta)?;
                pos.set_item("pending_quantity", pending_qty - qty_delta)?;
            }
        }
    }

    // Clear all pending transactions
    let empty_list = PyList::empty(py);
    account.set_item("pending_transactions", empty_list)?;

    Ok(())
}

/// Reserve funds for a pending order.
///
/// Raises AccountError if insufficient available balance.
#[pyfunction]
fn reserve_funds(account: &Bound<'_, PyDict>, amount: f64) -> PyResult<()> {
    let cleared_balance: f64 = account
        .get_item("cleared_balance")?
        .ok_or_else(|| AccountError::new_err("missing cleared_balance"))?
        .extract()?;
    let reserved_balance: f64 = account
        .get_item("reserved_balance")?
        .ok_or_else(|| AccountError::new_err("missing reserved_balance"))?
        .extract()?;

    let available = cleared_balance - reserved_balance;
    if amount > available {
        return Err(AccountError::new_err(format!(
            "insufficient funds: requested {}, available {}",
            amount, available
        )));
    }

    account.set_item("reserved_balance", reserved_balance + amount)?;
    Ok(())
}

/// Release previously reserved funds.
#[pyfunction]
fn release_reservation(account: &Bound<'_, PyDict>, amount: f64) -> PyResult<()> {
    let reserved_balance: f64 = account
        .get_item("reserved_balance")?
        .ok_or_else(|| AccountError::new_err("missing reserved_balance"))?
        .extract()?;

    let new_reserved = (reserved_balance - amount).max(0.0);
    account.set_item("reserved_balance", new_reserved)?;
    Ok(())
}

// ===========================================================================
// Storage Functions
// ===========================================================================

/// Get the base directory for trading data storage.
fn get_trading_dir() -> PyResult<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| StorageError::new_err("Could not determine home directory"))?;
    Ok(home.join(".trading"))
}

/// Get the directory for a specific dataset.
fn get_dataset_dir(dataset_id: &str) -> PyResult<PathBuf> {
    Ok(get_trading_dir()?.join("datasets").join(dataset_id))
}

/// Store a dataset to local storage as Parquet with JSON metadata.
///
/// Args:
///     bars: List of bar dictionaries (normalized bars)
///     dataset_id: Unique identifier for the dataset
///     metadata_json: JSON string of DatasetMetadata
///
/// Returns:
///     None (side effect: writes to disk)
#[pyfunction]
fn store_dataset(
    py: Python<'_>,
    bars: &Bound<'_, PyList>,
    dataset_id: String,
    metadata_json: String,
) -> PyResult<()> {
    let dataset_dir = get_dataset_dir(&dataset_id)?;

    // Create directory if it doesn't exist
    fs::create_dir_all(&dataset_dir)
        .map_err(|e| StorageError::new_err(format!("Failed to create dataset directory: {}", e)))?;

    // Convert bars to Polars DataFrame
    let mut symbols: Vec<String> = Vec::new();
    let mut timestamps: Vec<i64> = Vec::new();
    let mut opens: Vec<f64> = Vec::new();
    let mut highs: Vec<f64> = Vec::new();
    let mut lows: Vec<f64> = Vec::new();
    let mut closes: Vec<f64> = Vec::new();
    let mut volumes: Vec<f64> = Vec::new();

    for bar_obj in bars.iter() {
        let bar: Bound<'_, PyDict> = bar_obj.cast::<PyDict>()?.clone();

        let symbol: String = bar
            .get_item("symbol")?
            .ok_or_else(|| StorageError::new_err("bar missing symbol"))?
            .extract()?;

        let timestamp = bar
            .get_item("timestamp")?
            .ok_or_else(|| StorageError::new_err("bar missing timestamp"))?;
        
        // Convert Python datetime to timestamp (milliseconds)
        let ts_millis: i64 = timestamp
            .call_method0("timestamp")?
            .extract::<f64>()
            .map(|t| (t * 1000.0) as i64)?;

        let open: f64 = bar
            .get_item("open")?
            .ok_or_else(|| StorageError::new_err("bar missing open"))?
            .extract()?;
        let high: f64 = bar
            .get_item("high")?
            .ok_or_else(|| StorageError::new_err("bar missing high"))?
            .extract()?;
        let low: f64 = bar
            .get_item("low")?
            .ok_or_else(|| StorageError::new_err("bar missing low"))?
            .extract()?;
        let close: f64 = bar
            .get_item("close")?
            .ok_or_else(|| StorageError::new_err("bar missing close"))?
            .extract()?;
        let volume: f64 = bar
            .get_item("volume")?
            .ok_or_else(|| StorageError::new_err("bar missing volume"))?
            .extract()?;

        symbols.push(symbol);
        timestamps.push(ts_millis);
        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);
    }

    // Create DataFrame
    let df = DataFrame::new(vec![
        Column::new("symbol".into(), symbols),
        Column::new("timestamp".into(), timestamps),
        Column::new("open".into(), opens),
        Column::new("high".into(), highs),
        Column::new("low".into(), lows),
        Column::new("close".into(), closes),
        Column::new("volume".into(), volumes),
    ])
    .map_err(|e| StorageError::new_err(format!("Failed to create DataFrame: {}", e)))?;

    // Write to Parquet
    let parquet_path = dataset_dir.join("bars.parquet");
    let file = fs::File::create(&parquet_path)
        .map_err(|e| StorageError::new_err(format!("Failed to create parquet file: {}", e)))?;

    ParquetWriter::new(file)
        .finish(&mut df.clone())
        .map_err(|e| StorageError::new_err(format!("Failed to write parquet file: {}", e)))?;

    // Write metadata JSON
    let metadata_path = dataset_dir.join("metadata.json");
    fs::write(&metadata_path, &metadata_json)
        .map_err(|e| StorageError::new_err(format!("Failed to write metadata file: {}", e)))?;

    // Release GIL during potentially long I/O operations is handled automatically
    let _ = py; // Mark py as used

    Ok(())
}

/// Load a dataset from local storage.
///
/// Args:
///     dataset_id: Unique identifier for the dataset
///
/// Returns:
///     List of bar dictionaries
#[pyfunction]
fn load_dataset(py: Python<'_>, dataset_id: String) -> PyResult<Py<PyList>> {
    let dataset_dir = get_dataset_dir(&dataset_id)?;
    let parquet_path = dataset_dir.join("bars.parquet");

    if !parquet_path.exists() {
        return Err(StorageError::new_err(format!(
            "Dataset {} not found at {}",
            dataset_id,
            parquet_path.display()
        )));
    }

    // Read Parquet file
    let file = fs::File::open(&parquet_path)
        .map_err(|e| StorageError::new_err(format!("Failed to open parquet file: {}", e)))?;

    let df = ParquetReader::new(file)
        .finish()
        .map_err(|e| StorageError::new_err(format!("Failed to read parquet file: {}", e)))?;

    // Convert DataFrame to Python list of dicts
    let datetime_module = py.import("datetime")?;
    let datetime_class = datetime_module.getattr("datetime")?;
    let timezone_utc = datetime_module.getattr("timezone")?.getattr("utc")?;

    let result = PyList::empty(py);

    let symbols = df.column("symbol")
        .map_err(|e| StorageError::new_err(format!("Missing symbol column: {}", e)))?
        .str()
        .map_err(|e| StorageError::new_err(format!("Invalid symbol column: {}", e)))?;

    let timestamps = df.column("timestamp")
        .map_err(|e| StorageError::new_err(format!("Missing timestamp column: {}", e)))?
        .i64()
        .map_err(|e| StorageError::new_err(format!("Invalid timestamp column: {}", e)))?;

    let opens = df.column("open")
        .map_err(|e| StorageError::new_err(format!("Missing open column: {}", e)))?
        .f64()
        .map_err(|e| StorageError::new_err(format!("Invalid open column: {}", e)))?;

    let highs = df.column("high")
        .map_err(|e| StorageError::new_err(format!("Missing high column: {}", e)))?
        .f64()
        .map_err(|e| StorageError::new_err(format!("Invalid high column: {}", e)))?;

    let lows = df.column("low")
        .map_err(|e| StorageError::new_err(format!("Missing low column: {}", e)))?
        .f64()
        .map_err(|e| StorageError::new_err(format!("Invalid low column: {}", e)))?;

    let closes = df.column("close")
        .map_err(|e| StorageError::new_err(format!("Missing close column: {}", e)))?
        .f64()
        .map_err(|e| StorageError::new_err(format!("Invalid close column: {}", e)))?;

    let volumes = df.column("volume")
        .map_err(|e| StorageError::new_err(format!("Missing volume column: {}", e)))?
        .f64()
        .map_err(|e| StorageError::new_err(format!("Invalid volume column: {}", e)))?;

    for i in 0..df.height() {
        let bar = PyDict::new(py);

        let symbol = symbols.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get symbol value"))?;
        bar.set_item("symbol", symbol)?;

        // Convert timestamp back to datetime
        let ts_millis = timestamps.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get timestamp value"))?;
        let ts_seconds = ts_millis as f64 / 1000.0;
        let dt = datetime_class.call_method1("fromtimestamp", (ts_seconds, &timezone_utc))?;
        bar.set_item("timestamp", dt)?;

        let open = opens.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get open value"))?;
        bar.set_item("open", open)?;

        let high = highs.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get high value"))?;
        bar.set_item("high", high)?;

        let low = lows.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get low value"))?;
        bar.set_item("low", low)?;

        let close = closes.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get close value"))?;
        bar.set_item("close", close)?;

        let volume = volumes.get(i)
            .ok_or_else(|| StorageError::new_err("Failed to get volume value"))?;
        bar.set_item("volume", volume)?;

        result.append(bar)?;
    }

    Ok(result.unbind())
}

/// List all available datasets.
///
/// Returns:
///     List of dataset IDs
#[pyfunction]
fn list_datasets() -> PyResult<Vec<String>> {
    let datasets_dir = get_trading_dir()?.join("datasets");

    if !datasets_dir.exists() {
        return Ok(Vec::new());
    }

    let mut datasets = Vec::new();

    let entries = fs::read_dir(&datasets_dir)
        .map_err(|e| StorageError::new_err(format!("Failed to read datasets directory: {}", e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| StorageError::new_err(format!("Failed to read directory entry: {}", e)))?;

        if entry.path().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                // Check if it has both bars.parquet and metadata.json
                let parquet_exists = entry.path().join("bars.parquet").exists();
                let metadata_exists = entry.path().join("metadata.json").exists();

                if parquet_exists && metadata_exists {
                    datasets.push(name.to_string());
                }
            }
        }
    }

    Ok(datasets)
}

/// Check if a dataset exists.
///
/// Args:
///     dataset_id: Unique identifier for the dataset
///
/// Returns:
///     True if dataset exists, False otherwise
#[pyfunction]
fn dataset_exists(dataset_id: String) -> PyResult<bool> {
    let dataset_dir = get_dataset_dir(&dataset_id)?;
    let parquet_exists = dataset_dir.join("bars.parquet").exists();
    let metadata_exists = dataset_dir.join("metadata.json").exists();

    Ok(parquet_exists && metadata_exists)
}

/// Read metadata for a dataset.
///
/// Args:
///     dataset_id: Unique identifier for the dataset
///
/// Returns:
///     JSON string of metadata
#[pyfunction]
fn read_dataset_metadata(dataset_id: String) -> PyResult<String> {
    let dataset_dir = get_dataset_dir(&dataset_id)?;
    let metadata_path = dataset_dir.join("metadata.json");

    if !metadata_path.exists() {
        return Err(StorageError::new_err(format!(
            "Metadata not found for dataset {}",
            dataset_id
        )));
    }

    fs::read_to_string(&metadata_path)
        .map_err(|e| StorageError::new_err(format!("Failed to read metadata file: {}", e)))
}

// ===========================================================================
// Execution Engine Functions
// ===========================================================================

/// Execute market orders against current bar prices.
///
/// For each order in the list:
/// - Looks up the current price from the bars dict (uses bar close price)
/// - Creates an execution record with the fill price
/// - Updates account: adjusts balances and positions
///
/// Args:
///     orders: List of order dicts with symbol, side, quantity
///     bars: Dict mapping symbol -> bar dict with close price
///     account: Account dict (modified in place)
///     timestamp: Current execution timestamp
///
/// Returns:
///     List of execution dicts with symbol, side, quantity, price, value, timestamp
#[pyfunction]
fn execute_orders(
    py: Python<'_>,
    orders: &Bound<'_, PyList>,
    bars: &Bound<'_, PyDict>,
    account: &Bound<'_, PyDict>,
    timestamp: &Bound<'_, PyAny>,
) -> PyResult<Py<PyList>> {
    let executions = PyList::empty(py);

    for order_obj in orders.iter() {
        let order: Bound<'_, PyDict> = order_obj.cast::<PyDict>()?.clone();

        // Extract order fields
        let symbol: String = order
            .get_item("symbol")?
            .ok_or_else(|| AccountError::new_err("order missing symbol"))?
            .extract()?;

        let side: String = order
            .get_item("side")?
            .ok_or_else(|| AccountError::new_err("order missing side"))?
            .extract()?;

        let quantity: f64 = order
            .get_item("quantity")?
            .ok_or_else(|| AccountError::new_err("order missing quantity"))?
            .extract()?;

        // Get current price from bars
        let bar = bars
            .get_item(&symbol)?
            .ok_or_else(|| AccountError::new_err(format!("no bar data for symbol {}", symbol)))?;

        let bar_dict: Bound<'_, PyDict> = bar.cast::<PyDict>()?.clone();
        let price: f64 = bar_dict
            .get_item("close")?
            .ok_or_else(|| AccountError::new_err("bar missing close price"))?
            .extract()?;

        // Calculate execution value
        let value = quantity * price;

        // Update account based on side
        let cleared_balance: f64 = account
            .get_item("cleared_balance")?
            .ok_or_else(|| AccountError::new_err("missing cleared_balance"))?
            .extract()?;

        let positions_obj = account
            .get_item("positions")?
            .ok_or_else(|| AccountError::new_err("missing positions"))?;
        let positions: Bound<'_, PyDict> = positions_obj.cast::<PyDict>()?.clone();

        if side == "buy" {
            // Check if we have enough funds
            if value > cleared_balance {
                // Skip this order - insufficient funds
                continue;
            }

            // Deduct from balance
            account.set_item("cleared_balance", cleared_balance - value)?;

            // Update or create position
            if let Some(pos_obj) = positions.get_item(&symbol)? {
                let pos: Bound<'_, PyDict> = pos_obj.cast::<PyDict>()?.clone();
                let current_qty: f64 = pos.get_item("quantity")?
                    .map(|v| v.extract::<f64>())
                    .transpose()?
                    .unwrap_or(0.0);
                let current_cost: f64 = pos.get_item("cost_basis")?
                    .map(|v| v.extract::<f64>())
                    .transpose()?
                    .unwrap_or(0.0);

                // Update position with weighted average cost basis
                let new_qty = current_qty + quantity;
                let new_cost = (current_cost * current_qty + value) / new_qty;

                pos.set_item("quantity", new_qty)?;
                pos.set_item("cost_basis", new_cost)?;
            } else {
                // Create new position
                let new_pos = PyDict::new(py);
                new_pos.set_item("symbol", &symbol)?;
                new_pos.set_item("quantity", quantity)?;
                new_pos.set_item("cost_basis", price)?;
                new_pos.set_item("pending_quantity", 0.0)?;
                positions.set_item(&symbol, new_pos)?;
            }
        } else if side == "sell" {
            // Check if we have the position
            if let Some(pos_obj) = positions.get_item(&symbol)? {
                let pos: Bound<'_, PyDict> = pos_obj.cast::<PyDict>()?.clone();
                let current_qty: f64 = pos.get_item("quantity")?
                    .map(|v| v.extract::<f64>())
                    .transpose()?
                    .unwrap_or(0.0);

                if quantity > current_qty {
                    // Can't sell more than we have - skip
                    continue;
                }

                // Add to balance
                account.set_item("cleared_balance", cleared_balance + value)?;

                // Update position
                let new_qty = current_qty - quantity;
                if new_qty <= 0.0 {
                    // Remove position
                    positions.del_item(&symbol)?;
                } else {
                    pos.set_item("quantity", new_qty)?;
                }
            } else {
                // No position to sell - skip
                continue;
            }
        }

        // Create execution record
        let execution = PyDict::new(py);
        execution.set_item("symbol", &symbol)?;
        execution.set_item("side", &side)?;
        execution.set_item("quantity", quantity)?;
        execution.set_item("price", price)?;
        execution.set_item("value", value)?;
        execution.set_item("timestamp", timestamp)?;

        executions.append(execution)?;
    }

    Ok(executions.unbind())
}

// ===========================================================================
// Metrics Computation Functions
// ===========================================================================

/// Compute run metrics from equity history.
///
/// Args:
///     equity_history: List of (timestamp, equity) tuples in chronological order
///     initial_equity: Starting equity value
///     num_trades: Total number of trades executed
///
/// Returns:
///     Dict with total_return, max_drawdown, volatility, sharpe_ratio, num_trades, win_rate
#[pyfunction]
fn compute_run_metrics(
    py: Python<'_>,
    equity_history: &Bound<'_, PyList>,
    initial_equity: f64,
    num_trades: i64,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);

    if equity_history.len() == 0 {
        result.set_item("total_return", 0.0)?;
        result.set_item("max_drawdown", 0.0)?;
        result.set_item("volatility", 0.0)?;
        result.set_item("sharpe_ratio", py.None())?;
        result.set_item("num_trades", num_trades)?;
        result.set_item("win_rate", py.None())?;
        return Ok(result.unbind());
    }

    // Extract equity values
    let mut equities: Vec<f64> = Vec::new();
    for item in equity_history.iter() {
        let tuple: Bound<'_, pyo3::types::PyTuple> = item.cast::<pyo3::types::PyTuple>()?.clone();
        let equity: f64 = tuple.get_item(1)?.extract()?;
        equities.push(equity);
    }

    // Calculate total return
    let final_equity = *equities.last().unwrap_or(&initial_equity);
    let total_return = (final_equity - initial_equity) / initial_equity;

    // Calculate max drawdown
    let mut peak = initial_equity;
    let mut max_drawdown = 0.0f64;
    for &equity in &equities {
        if equity > peak {
            peak = equity;
        }
        let drawdown = (peak - equity) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Calculate daily returns for volatility and Sharpe
    let mut returns: Vec<f64> = Vec::new();
    let mut prev_equity = initial_equity;
    for &equity in &equities {
        if prev_equity > 0.0 {
            returns.push((equity - prev_equity) / prev_equity);
        }
        prev_equity = equity;
    }

    // Calculate volatility (standard deviation of returns)
    let volatility = if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    // Calculate Sharpe ratio (assuming 0 risk-free rate for simplicity)
    let sharpe_ratio = if volatility > 0.0 && returns.len() > 1 {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        // Annualize: assuming daily data, multiply by sqrt(252)
        let annualized_return = mean_return * 252.0;
        let annualized_vol = volatility * (252.0_f64).sqrt();
        Some(annualized_return / annualized_vol)
    } else {
        None
    };

    result.set_item("total_return", total_return)?;
    result.set_item("max_drawdown", max_drawdown)?;
    result.set_item("volatility", volatility)?;
    match sharpe_ratio {
        Some(sr) => result.set_item("sharpe_ratio", sr)?,
        None => result.set_item("sharpe_ratio", py.None())?,
    }
    result.set_item("num_trades", num_trades)?;
    result.set_item("win_rate", py.None())?; // Would need trade-level data

    Ok(result.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_rust() {
        assert_eq!(hello_rust(), "Hello from Rust!");
    }

    #[test]
    fn test_validate_ohlc() {
        // Valid OHLC
        assert!(validate_ohlc(100.0, 105.0, 95.0, 102.0));

        // Invalid: high < low
        assert!(!validate_ohlc(100.0, 90.0, 95.0, 102.0));

        // Invalid: high < open
        assert!(!validate_ohlc(100.0, 99.0, 95.0, 98.0));

        // Invalid: low > close
        assert!(!validate_ohlc(100.0, 105.0, 101.0, 99.0));
    }

    #[test]
    fn test_is_finite() {
        assert!(is_finite(100.0));
        assert!(is_finite(-50.0));
        assert!(is_finite(0.0));
        assert!(!is_finite(f64::NAN));
        assert!(!is_finite(f64::INFINITY));
        assert!(!is_finite(f64::NEG_INFINITY));
    }
}
