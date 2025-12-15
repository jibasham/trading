# Trading Bot System

A hybrid Rust + Python trading bot system for backtesting and strategy development.

## Architecture

- **Rust Core**: High-performance data processing, account management, execution engine, and metrics computation
- **Python Layer**: Strategy development, ML/RL integration, and CLI interface

## Project Structure

```
trading/
├── rust/              # Rust core library
│   ├── src/
│   │   └── lib.rs    # Rust implementation
│   ├── Cargo.toml    # Rust dependencies
│   └── build.rs      # Build configuration
├── src/
│   └── trading/      # Python package
│       ├── __init__.py
│       ├── main.py
│       └── types.py
├── tests/            # Python tests
├── docs/             # Documentation
├── pyproject.toml    # Python project config (uses maturin for Rust integration)
└── Cargo.toml        # Root Cargo.toml (if needed)
```

## Setup

### Prerequisites

- Python 3.12+
- Rust (install from https://rustup.rs/)
- maturin (install with `pip install maturin` or `cargo install maturin`)

### Development Setup

1. Install dependencies:
```bash
# Install Python dependencies
pip install -e .

# Or using uv (if you have it)
uv sync
```

2. Build Rust extension:
```bash
# Development build (faster, includes debug info)
maturin develop

# Or release build (optimized)
maturin develop --release
```

3. Run tests:
```bash
pytest
```

## Development Workflow

### Adding Rust Code

1. Add Rust functions to `rust/src/lib.rs` or create new modules
2. Expose Python bindings using PyO3 macros (`#[pyfunction]`, `#[pymodule]`)
3. Rebuild: `maturin develop`
4. Import in Python: `from trading._core import your_function`

### Adding Python Code

1. Add Python modules to `src/trading/`
2. Import Rust functions from `trading._core` when needed
3. No rebuild needed for pure Python changes

## Building

```bash
# Development build
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheel
maturin build
```

## Notes

- This is a private project - no publishing to PyPI or crates.io needed
- The Rust extension is built as a Python module `trading._core`
- Python strategies can call Rust functions for performance-critical operations
- Rust core handles: data processing, account management, execution, metrics
- Python handles: strategy logic, ML/RL, CLI, configuration


