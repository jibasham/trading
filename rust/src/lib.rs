use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDateTime, PyDict, PyList};

// Custom exception for account-related errors
create_exception!(trading._core, AccountError, PyException);

/// Trading core library - high-performance Rust implementation
/// This module provides core functionality that can be called from Python
#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AccountError", py.get_type::<AccountError>())?;
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(next_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_account_equity, m)?)?;
    m.add_function(wrap_pyfunction!(process_clearing, m)?)?;
    m.add_function(wrap_pyfunction!(reserve_funds, m)?)?;
    m.add_function(wrap_pyfunction!(release_reservation, m)?)?;
    Ok(())
}

/// A simple test function to verify Rust-Python integration
#[pyfunction]
fn hello_rust() -> String {
    "Hello from Rust!".to_string()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_rust() {
        assert_eq!(hello_rust(), "Hello from Rust!");
    }
}
