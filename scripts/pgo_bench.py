"""Time representative hexhamming calls and print them as JSON (ns per call).

Used by the PGO wheel workflow to compare a PGO build with a plain build on
the same runner. Each case reports the best of several rounds of repeated
calls.

Separate processes vary by several percent (frequency, code placement), so
run each build a few times, alternating, and compare the per-case minimum.

Usage: python scripts/pgo_bench.py [module] > results.json
       python scripts/pgo_bench.py --compare plain-*.json vs pgo-*.json
"""

import argparse
import importlib
import json
import random
import time


def cases(h):
    rng = random.Random(99)
    s16, t16 = rng.randbytes(8).hex(), rng.randbytes(8).hex()
    s64, t64 = rng.randbytes(32).hex(), rng.randbytes(32).hex()
    s1k, t1k = rng.randbytes(512).hex(), rng.randbytes(512).hex()
    b16, c16 = rng.randbytes(16), rng.randbytes(16)
    b64, c64 = rng.randbytes(64), rng.randbytes(64)
    cat16 = rng.randbytes(16 * 1024)
    cat64 = rng.randbytes(64 * 16384)
    pa, pb = rng.randbytes(16 * 64), rng.randbytes(16 * 64)
    return [
        ("hamming_distance_string/16", h.hamming_distance_string, (s16, t16)),
        ("hamming_distance_string/64", h.hamming_distance_string, (s64, t64)),
        ("hamming_distance_string/1024", h.hamming_distance_string, (s1k, t1k)),
        ("hamming_distance_bytes/16", h.hamming_distance_bytes, (b16, c16)),
        ("hamming_distance_bytes/64", h.hamming_distance_bytes, (b64, c64)),
        (
            "check_hexstrings_within_dist/16",
            h.check_hexstrings_within_dist,
            (s16, t16, 10),
        ),
        ("check_bytes_within_dist/16", h.check_bytes_within_dist, (b16, c16, 10)),
        ("best_within/1024x16", h.check_bytes_arrays_best_within_dist, (cat16, b16, 0)),
        ("all_within/1024x16", h.check_bytes_arrays_all_within_dist, (cat16, b16, 0)),
        (
            "best_within/16384x64",
            h.check_bytes_arrays_best_within_dist,
            (cat64, b64, 0),
        ),
        (
            "hamming_distances_bytes_packed/64x16",
            h.hamming_distances_bytes_packed,
            (pa, pb, 16),
        ),
    ]


def measure(fn, args, rounds=7, budget=0.05):
    calls = 16
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            fn(*args)
        if time.perf_counter() - start > budget / 10:
            break
        calls *= 4
    calls = max(1, int(calls * budget / max(time.perf_counter() - start, 1e-9) / 10))
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn(*args)
        best = min(best, (time.perf_counter_ns() - start) / calls)
    return best


def load_min(paths):
    best = {}
    for path in paths:
        with open(path) as f:
            for name, value in json.load(f).items():
                best[name] = min(value, best.get(name, float("inf")))
    return best


def compare(paths):
    split = paths.index("vs")
    plain, pgo = load_min(paths[:split]), load_min(paths[split + 1 :])
    print(f"Best of {split} plain and {len(paths) - split - 1} PGO runs.")
    print()
    print("| Case | plain (ns) | PGO (ns) | speedup |")
    print("|---|---:|---:|---:|")
    for name, base in plain.items():
        new = pgo[name]
        print(f"| `{name}` | {base:,.1f} | {new:,.1f} | {base / new:.2f}x |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", nargs="?", default="hexhamming")
    parser.add_argument("--compare", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return
    h = importlib.import_module(args.module)
    print(
        json.dumps({name: measure(fn, fargs) for name, fn, fargs in cases(h)}, indent=1)
    )


if __name__ == "__main__":
    main()
