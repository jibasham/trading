# Setup Guide for Hybrid Rust + Python Project

## Quick Start

1. **Install maturin** (build tool for Rust-Python integration):
```bash
pip install maturin
# OR
cargo install maturin
```

2. **Build the Rust extension**:
```bash
# Development build (faster, includes debug info)
maturin develop --manifest-path rust/Cargo.toml

# OR use the Makefile
make dev
```

3. **Install Python package**:
```bash
pip install -e .
```

4. **Verify installation**:
```bash
python -c "from trading import hello_rust; print(hello_rust())"
# Should print: Hello from Rust!
```

5. **Run tests**:
```bash
pytest
# OR
make test
```

## Project Structure

- `rust/` - Rust core library (high-performance components)
- `src/trading/` - Python package (strategies, CLI, ML/RL)
- `tests/` - Python tests
- `docs/` - Documentation

## Development Workflow

### Adding Rust Functions

1. Add function to `rust/src/lib.rs`:
```rust
#[pyfunction]
fn my_function(input: String) -> String {
    // Your Rust code here
    format!("Processed: {}", input)
}
```

2. Register in `#[pymodule]`:
```rust
m.add_function(wrap_pyfunction!(my_function, m)?)?;
```

3. Rebuild:
```bash
maturin develop --manifest-path rust/Cargo.toml
```

4. Use in Python:
```python
from trading._core import my_function
result = my_function("test")
```

### Adding Python Code

Just add files to `src/trading/` - no rebuild needed for pure Python.

## Common Commands

```bash
# Build Rust extension (dev)
make dev

# Build Rust extension (release/optimized)
make build

# Run Python tests
make test

# Run Rust tests
make rust-test

# Clean everything
make clean
```

## Troubleshooting

**ImportError: No module named 'trading._core'**
- Run `maturin develop --manifest-path rust/Cargo.toml` first

**Rust compilation errors**
- Check Rust version: `rustc --version` (should be 1.70+)
- Update dependencies: `cd rust && cargo update`

**Python can't find module**
- Ensure you're in the project root
- Check `pythonpath` in `pyproject.toml`
- Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`


