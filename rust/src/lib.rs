use pyo3::prelude::*;

/// Trading core library - high-performance Rust implementation
/// This module provides core functionality that can be called from Python

#[pymodule]
fn trading_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core modules will be added here
    // For now, just a placeholder
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    Ok(())
}

/// A simple test function to verify Rust-Python integration
#[pyfunction]
fn hello_rust() -> String {
    "Hello from Rust!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_rust() {
        assert_eq!(hello_rust(), "Hello from Rust!");
    }
}
