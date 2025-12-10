"""Test Rust-Python integration."""

import pytest


def test_rust_module_import():
    """Test that Rust module can be imported."""
    try:
        from trading._core import hello_rust
        assert hello_rust is not None
    except ImportError:
        pytest.skip("Rust extension not built. Run 'maturin develop' first.")


def test_rust_function_call():
    """Test calling a Rust function from Python."""
    try:
        from trading._core import hello_rust
        
        result = hello_rust()
        assert isinstance(result, str)
        assert "Rust" in result
    except ImportError:
        pytest.skip("Rust extension not built. Run 'maturin develop' first.")


def test_rust_availability_flag():
    """Test that RUST_AVAILABLE flag works."""
    from trading import RUST_AVAILABLE
    
    # This will be True if Rust extension is built, False otherwise
    assert isinstance(RUST_AVAILABLE, bool)

