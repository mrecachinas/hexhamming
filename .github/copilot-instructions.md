# hexhamming - Fast Hamming Distance Calculation

hexhamming is a Python extension module written in Rust (via PyO3/maturin) that provides blazingly fast bitwise Hamming distance calculation for hexadecimal strings and byte arrays. It uses vectorized algorithms (SSE4.1, AVX2, AVX-512, ARM NEON) for optimal performance.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Essential Setup Commands
Run these commands in sequence to set up the development environment:

```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install the package (RECOMMENDED METHOD)
pip3 install .
# Takes 1-2 minutes. NEVER CANCEL - Rust compilation requires time.
# Set timeout to 5+ minutes for build commands.

# Or use maturin directly for development
pip install maturin
maturin develop --release
```

### Running Tests
```bash
# Run the complete test suite with benchmarks  
python3 -m pytest -vls .
# Takes ~27 seconds. NEVER CANCEL - includes performance benchmarks.
# Set timeout to 2+ minutes for test commands.

# Quick test run (no benchmarks)
python3 -m pytest test/ -k "not bench"
# Takes ~5 seconds for functional tests only.

# Run Rust unit tests
cargo test
```

### Code Quality and CI Requirements
```bash
# CRITICAL: Check code formatting before committing
ruff check .
ruff format --check .

# Fix code formatting (if needed)
ruff format .

# Rust formatting
cargo fmt --check
cargo fmt  # to fix
```

## Validation Scenarios

### ALWAYS Test These After Making Changes
After any code modifications, run these validation scenarios:

```bash
# 1. Basic string functionality test
python3 -c "
from hexhamming import hamming_distance_string
result = hamming_distance_string('deadbeef', '00000000')
assert result == 24, f'Expected 24, got {result}'
print('✓ String hamming distance: PASS')
"

# 2. Bytes functionality test  
python3 -c "
from hexhamming import hamming_distance_bytes
result = hamming_distance_bytes(b'\xde\xad\xbe\xef', b'\x00\x00\x00\x00')
assert result == 24, f'Expected 24, got {result}'
print('✓ Bytes hamming distance: PASS')
"

# 3. Within distance check test
python3 -c "
from hexhamming import check_hexstrings_within_dist
result1 = check_hexstrings_within_dist('ffff', 'fffe', 2)
result2 = check_hexstrings_within_dist('ffff', '0000', 2)
assert result1 == True, f'Expected True, got {result1}'
assert result2 == False, f'Expected False, got {result2}'
print('✓ Within distance check: PASS')
"

# 4. Algorithm switching test
python3 -c "
from hexhamming import set_algo, hamming_distance_string
set_algo('classic')
result = hamming_distance_string('abc', 'def')
assert isinstance(result, int), f'Algorithm switch failed'
print('✓ Algorithm switching: PASS')
"
```

### Build Validation Scenarios
Test that your changes don't break the build:

```bash
# Clean build test (removes any cached artifacts)
rm -rf target/wheels/ test/__pycache__/ .pytest_cache/
pip3 install .
# Takes 1-2 minutes. NEVER CANCEL.

# Verify installation worked
python3 -c "import hexhamming; print('✓ Import successful')"
```

## Common Build Issues and Solutions

### Rust Toolchain Missing
If you get errors about `cargo` or `rustc` not found:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Test Failures
If tests fail after your changes:
- Focus only on failures related to your changes
- Ignore unrelated benchmark timing variations
- Always run the validation scenarios above to verify core functionality

## Performance Expectations

### CRITICAL Timing Information - NEVER CANCEL
- **Package build**: 1-2 minutes (Rust compilation takes time)
- **Full test suite**: ~27 seconds (includes performance benchmarks)
- **Code formatting**: 1-2 seconds

**NEVER CANCEL builds or long-running commands.** Rust compilation and performance benchmarks require time to complete. Always set timeouts of 5+ minutes for builds and 2+ minutes for tests.

## Repository Structure

### Key Files and Directories
```
hexhamming/
├── README.rst              # Main documentation and usage examples
├── Cargo.toml              # Rust package manifest and dependencies
├── pyproject.toml           # Python package metadata (maturin build backend)
├── src/
│   └── lib.rs              # Rust implementation (PyO3 bindings + SIMD algorithms)
├── benches/
│   └── bench.rs            # Criterion benchmarks for Rust API
├── test/
│   └── test_hexhamming.py  # Comprehensive Python test suite with benchmarks
└── .github/workflows/
    └── pythonpackage.yml   # CI/CD pipeline (maturin + PyPI publish)
```

### Important Code Locations
- **Algorithm implementations**: `src/lib.rs` (SIMD dispatch, SSE/AVX2/AVX-512/NEON)
- **Python bindings**: `src/lib.rs` (PyO3 `#[pyfunction]` exports)
- **Performance tests**: `test/test_hexhamming.py` (lines 100+)
- **Build configuration**: `Cargo.toml` + `pyproject.toml`

## Development Workflow

### Making Changes
1. **ALWAYS** run the setup commands first
2. Make your code changes
3. **IMMEDIATELY** test with validation scenarios
4. Run `ruff format --check .` and fix formatting if needed
5. Run full test suite: `python3 -m pytest -vls .`

### Before Committing
```bash
# Required checks that CI will run
ruff check .              # Linting
ruff format --check .     # Code formatting
cargo fmt --check         # Rust formatting
python3 -m pytest -vls .  # Full test suite
```

### CI/CD Information
The project uses GitHub Actions with maturin for cross-platform wheel building:
- Builds wheels for Linux (manylinux), macOS, Windows
- Tests on Python 3.10-3.14
- **Build time in CI**: 15-45 minutes depending on platform

If CI fails on build steps, it's often due to:
1. Code formatting issues (run `ruff format .`)
2. Test failures (run validation scenarios locally)

## Troubleshooting

### "Module not found" errors
Ensure you've installed the package: `pip3 install .`

### Compilation errors
Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Test failures
Run validation scenarios individually to isolate issues.