#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
# Keep ruff in sync with the lint job in .github/workflows/pythonpackage.yml.
python -m pip install ruff==0.16.8 maturin pytest pytest-benchmark

python -m pip install .
cargo fetch
