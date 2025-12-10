.PHONY: help install dev build test clean rust-test

help:
	@echo "Available targets:"
	@echo "  install    - Install Python dependencies"
	@echo "  dev        - Build Rust extension in development mode"
	@echo "  build      - Build Rust extension in release mode"
	@echo "  test       - Run Python tests"
	@echo "  rust-test  - Run Rust tests"
	@echo "  clean      - Clean build artifacts"

install:
	pip install -e .

dev:
	maturin develop --manifest-path rust/Cargo.toml

build:
	maturin develop --release --manifest-path rust/Cargo.toml

test:
	pytest

rust-test:
	cd rust && cargo test

clean:
	cd rust && cargo clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

