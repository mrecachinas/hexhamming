#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "error: x86 SIMD benchmarks require an x86_64 machine" >&2
    exit 1
fi

label="${1:-working-tree}"
output_dir="${2:-benchmark-results/${label}}"
mkdir -p "${output_dir}"

{
    echo "label=${label}"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "kernel=$(uname -a)"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "python=$(python3 --version 2>&1)"
    if command -v lscpu >/dev/null 2>&1; then
        lscpu
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -a 2>/dev/null | grep -E 'machdep.cpu.(brand_string|features|leaf7_features)'
    fi
} >"${output_dir}/system.txt"

python3 -m pip install --quiet pytest pytest-benchmark
python3 -m pip install --quiet .

for run in 1 2 3; do
    echo "Rust benchmark run ${run}/3"
    cargo bench --bench hamming_bench -- \
        --noplot \
        --save-baseline "${label}-rust-${run}" \
        2>&1 | tee "${output_dir}/criterion-${run}.txt"

    echo "Python benchmark run ${run}/3"
    python3 -m pytest test/ \
        -k bench \
        --benchmark-only \
        --benchmark-disable-gc \
        --benchmark-json="${output_dir}/python-${run}.json"
done

echo "Results written to ${output_dir}"
