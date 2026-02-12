# hexhamming - Fast Hamming Distance Calculation

hexhamming is a Python C extension module that provides blazingly fast bitwise Hamming distance calculation for hexadecimal strings and byte arrays. It uses vectorized algorithms (SSE4.1, AVX2, ARM NEON) for optimal performance.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Essential Setup Commands
Run these commands in sequence to set up the development environment:

```bash
# Install development dependencies
pip3 install -r requirements-dev.txt
# Takes ~30 seconds - includes pytest, black, pytest-benchmark

# Build and install the package (RECOMMENDED METHOD)
pip3 install .
# Takes 2-3 minutes. NEVER CANCEL - C++ compilation requires time.
# Set timeout to 5+ minutes for build commands.
```

### Alternative Build Methods
If the recommended method fails due to network issues:

```bash
# Legacy build method (always works offline)
python3 setup.py install --user
# Takes 2-3 minutes. NEVER CANCEL. Shows deprecation warnings but works reliably.
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
```

### Code Quality and CI Requirements
```bash
# CRITICAL: Check code formatting before committing
black --check .
# Returns exit code 1 if formatting needed

# Fix code formatting (if needed)
black .
# Reformats Python files to match project standards

# Validate package manifest
python3 -m pip install check-manifest
python3 -m check_manifest
# Verifies all files are properly included in package
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
rm -rf build/ dist/ *.egg-info/ test/__pycache__/ .pytest_cache/
pip3 install .
# Takes 2-3 minutes. NEVER CANCEL.

# Verify installation worked
python3 -c "import hexhamming; print('✓ Import successful')"
```

## Common Build Issues and Solutions

### Network Connectivity Problems
If `pip install` or `python -m build` fail with network errors:
- Use the legacy method: `python3 setup.py install --user`
- This method works completely offline after initial dependency installation

### Build Dependencies Missing
If you get compiler errors:
```bash
# Install build essentials (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install build-essential python3-dev

# On other systems, ensure you have a C++ compiler and Python headers
```

### Test Failures
If tests fail after your changes:
- Focus only on failures related to your changes
- Ignore unrelated benchmark timing variations
- Always run the validation scenarios above to verify core functionality

## Performance Expectations

### CRITICAL Timing Information - NEVER CANCEL
- **Development setup**: 30 seconds to 2 minutes
- **Package build**: 2-3 minutes (C++ compilation takes time)
- **Full test suite**: ~27 seconds (includes performance benchmarks)
- **Code formatting**: 1-2 seconds
- **Manifest check**: 5-10 seconds

**NEVER CANCEL builds or long-running commands.** C++ compilation and performance benchmarks require time to complete. Always set timeouts of 5+ minutes for builds and 2+ minutes for tests.

## Repository Structure

### Key Files and Directories
```
hexhamming/
├── README.rst              # Main documentation and usage examples
├── setup.py                # Build configuration with C++ extension
├── requirements-dev.txt    # Development dependencies (pytest, black, etc.)
├── hexhamming/            # C++ source code directory
│   ├── python_hexhamming.cc    # Main C++ implementation
│   ├── python_hexhamming.h     # Header with vectorized algorithms
│   └── _version.h              # Version information
├── test/
│   └── test_hexhamming.py      # Comprehensive test suite with benchmarks
├── .github/workflows/
│   └── pythonpackage.yml      # CI/CD pipeline using cibuildwheel
└── MANIFEST.in             # Package file inclusion rules
```

### Important Code Locations
- **Algorithm implementations**: `hexhamming/python_hexhamming.h` (lines 150-630)
- **Python bindings**: `hexhamming/python_hexhamming.cc`
- **Performance tests**: `test/test_hexhamming.py` (lines 100+)
- **Build configuration**: `setup.py` (platform-specific optimizations)

## Development Workflow

### Making Changes
1. **ALWAYS** run the setup commands first
2. Make your code changes
3. **IMMEDIATELY** test with validation scenarios
4. Run `black --check .` and fix formatting if needed
5. Run full test suite: `python3 -m pytest -vls .`
6. Run `check-manifest` to verify package integrity

### Before Committing
```bash
# Required checks that CI will run
black --check .           # Code formatting
python3 -m check_manifest # Package manifest
python3 -m pytest -vls .  # Full test suite
```

### CI/CD Information
The project uses GitHub Actions with cibuildwheel for cross-platform wheel building:
- Builds wheels for Linux (manylinux, musllinux), macOS, Windows
- Tests on Python 3.6-3.10
- **Build time in CI**: 15-45 minutes depending on platform
- **Network dependencies**: Requires PyPI access for dependencies

If CI fails on build steps, it's often due to:
1. Code formatting issues (run `black .`)
2. Package manifest issues (run `check-manifest`)
3. Test failures (run validation scenarios locally)

## Troubleshooting

### "Module not found" errors
Ensure you've installed the package: `pip3 install .`

### Compilation errors
Check that you have build tools: `sudo apt-get install build-essential python3-dev`

### Test failures
Run validation scenarios individually to isolate issues.

### Network timeouts
Use legacy build method: `python3 setup.py install --user`

### Formatting issues
Run `black .` to fix all formatting issues automatically.