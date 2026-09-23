"""Training workload for profile-guided optimization (PGO) builds.

Run against an instrumented build (``-Cprofile-generate``) so the compiler
learns which branches and indirect calls are hot. The mix mirrors typical use:
many small single calls, fixed-width catalog scans (serial and parallel),
batch and many-query APIs, catalogs with and without an index, and the
error paths. It only needs the standard library.

Usage: python scripts/pgo_train.py [module] [--rounds N]
"""

import argparse
import importlib
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", nargs="?", default="hexhamming")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    h = importlib.import_module(args.module)
    rng = random.Random(1234)

    def hexstr(n):
        return rng.randbytes((n + 1) // 2).hex()[:n]

    hex_pairs = {
        n: [(hexstr(n), hexstr(n)) for _ in range(32)]
        for n in (8, 16, 24, 32, 64, 128, 254, 1024, 4096)
    }
    byte_pairs = {
        n: [(rng.randbytes(n), rng.randbytes(n)) for _ in range(32)]
        for n in (8, 16, 20, 32, 64, 127, 1024, 16384)
    }
    catalogs = {}
    for width, count in (
        (1, 4096),
        (8, 4096),
        (16, 1024),
        (16, 16384),
        (20, 4096),
        (32, 1024),
        (32, 16384),
        (64, 4096),
        (128, 1024),
    ):
        records = bytearray(rng.randbytes(width * count))
        query = rng.randbytes(width)
        at = count * 3 // 4
        records[at * width : (at + 1) * width] = query
        catalogs[(width, count)] = (bytes(records), query)
    # Large enough to run on the scan pool.
    large = {
        width: (rng.randbytes(width * (4 << 20) // width), rng.randbytes(width))
        for width in (16, 64)
    }
    pair_a, pair_b = rng.randbytes(16 * 8192), rng.randbytes(16 * 8192)
    queries = rng.randbytes(16 * 64)
    index_records = rng.randbytes(8 * 200_000)
    indexed = h.Catalog(index_records, 8, index=True)
    linear = h.Catalog(catalogs[(16, 16384)][0], 16)
    out = bytearray(8 * 8192)
    distances_out = bytearray(2 * 16384)
    indices_out = bytearray(4 * 16384)

    start = time.perf_counter()
    for _ in range(args.rounds):
        for n, pairs in hex_pairs.items():
            reps = 400 if n <= 254 else 40
            for i in range(reps):
                a, b = pairs[i % 32]
                h.hamming_distance_string(a, b)
                h.check_hexstrings_within_dist(a, b, n // 2 + i % 7)
        for n, pairs in byte_pairs.items():
            reps = 400 if n <= 1024 else 20
            for i in range(reps):
                a, b = pairs[i % 32]
                h.hamming_distance_bytes(a, b)
                h.check_bytes_within_dist(a, b, n * 4 - 5 + i % 11)
            a, b = pairs[0]
            ba, mv = bytearray(a), memoryview(b)
            for _ in range(100):
                h.hamming_distance_bytes(ba, mv)
                h.check_bytes_within_dist(mv, ba, n)
        for (width, _count), (records, query) in catalogs.items():
            for max_dist in (0, width * 2, width * 4):
                h.check_bytes_arrays_first_within_dist(records, query, max_dist)
                h.check_bytes_arrays_best_within_dist(records, query, max_dist)
                h.check_bytes_arrays_all_within_dist(records, query, max_dist)
            h.check_bytes_arrays_within_dist(records, query, width)
            h.check_bytes_arrays_all_within_dist_packed(records, query, width * 3)
        for records, query in large.values():
            for _ in range(10):
                h.check_bytes_arrays_first_within_dist(records, query, 0)
                h.check_bytes_arrays_best_within_dist(records, query, 3)
                h.check_bytes_arrays_all_within_dist(records, query, 3)
        for _ in range(20):
            for width in (8, 16, 20, 32, 64):
                a, b = pair_a[: width * 2048], pair_b[: width * 2048]
                h.hamming_distances_bytes(a, b, width)
                h.hamming_distances_bytes_packed(a, b, width)
            try:
                for width in (8, 16, 20, 32, 64):
                    a, b = pair_a[: width * 2048], pair_b[: width * 2048]
                    h.hamming_distances_bytes_into(a, b, width, out)
                h.check_bytes_arrays_all_within_dist_into(
                    catalogs[(16, 16384)][0],
                    queries[:16],
                    40,
                    distances_out,
                    indices_out,
                )
            except ValueError:
                pass  # writable `_into` APIs are unavailable on free-threaded builds
        catalog = catalogs[(16, 16384)][0]
        for max_dist in (0, 8, 40):
            h.check_bytes_arrays_first_many_within_dist(catalog, queries, 16, max_dist)
            h.check_bytes_arrays_best_many_within_dist(catalog, queries, 16, max_dist)
            h.check_bytes_arrays_all_many_within_dist(catalog, queries, 16, max_dist)
        for i in range(2000):
            query = index_records[(i * 97 % 200_000) * 8 :][:8]
            radius = i % 10
            indexed.best_within(query, radius)
            indexed.first_within(query, radius)
            if i % 4 == 0:
                indexed.all_within(query, radius)
        for i in range(50):
            linear.best_within(queries[(i % 64) * 16 :][:16], 20)
        indexed.best_many_within(index_records[: 8 * 256], 2)
        for algo in ("classic", "native"):
            h.set_algo(algo)
            h.hamming_distance_string(hex_pairs[64][0][0], hex_pairs[64][0][1])
            h.hamming_distance_bytes(*byte_pairs[64][0])
        for bad in (("zz", "00"), ("0" * 70, "0" * 69 + "g"), ("ab", "abc")):
            try:
                h.hamming_distance_string(*bad)
            except ValueError:
                pass
            try:
                h.check_hexstrings_within_dist(*bad, 3)
            except ValueError:
                pass
        try:
            h.check_bytes_arrays_first_within_dist(b"abc", b"ab", 1)
        except ValueError:
            pass
    print(f"PGO training finished in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
