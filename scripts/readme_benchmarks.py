"""Regenerate the Python-side benchmark tables in README.rst.

Each case is timed with ``timeit.repeat`` (best of 7 repeats) in each of
``--runs`` independent passes, and the table shows the median of those
per-pass results in nanoseconds per call. Tables are printed as
reStructuredText, ready to paste into the README.

Usage: python scripts/readme_benchmarks.py [--runs 3] [--only SECTION]
"""

import argparse
import random
import statistics
import timeit

import hexhamming as h

RNG = random.Random(2024)


def rb(n):
    return RNG.randbytes(n)


def best_ns(fn, number):
    return min(timeit.repeat(fn, number=number, repeat=7)) / number * 1e9


def calibrated(fn):
    number = 1
    while timeit.timeit(fn, number=number) < 0.02:
        number *= 4
    return number


def table(title, header, rows):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    rule = "  ".join("=" * w for w in widths)
    lines = [title, "", rule]
    lines.append("  ".join(str(c).ljust(w) for c, w in zip(header, widths)).rstrip())
    lines.append(rule)
    for row in rows:
        cells = [str(row[0]).ljust(widths[0])]
        cells += [str(c).rjust(w) for c, w in zip(row[1:], widths[1:])]
        lines.append("  ".join(cells).rstrip())
    lines.append(rule)
    return "\n".join(lines)


def fmt(ns):
    return f"{ns:,.1f}" if ns < 100_000 else f"{ns:,.0f}"


def planted(width, count, at):
    records = bytearray(rb(width * count))
    query = rb(width)
    if at is not None:
        records[at * width : (at + 1) * width] = query
    return bytes(records), query


def section_stdlib():
    def stdlib_hex(a, b):
        return (int(a, 16) ^ int(b, 16)).bit_count()

    def stdlib_bytes(a, b):
        return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).bit_count()

    cases = []
    for n in (16, 64, 1024):
        a, b = rb(n), rb(n)
        cases.append(
            (
                f"bytes [{n}]",
                lambda a=a, b=b: h.hamming_distance_bytes(a, b),
                lambda a=a, b=b: stdlib_bytes(a, b),
            )
        )
    for n in (16, 64, 1024):
        a, b = rb(n // 2).hex(), rb(n // 2).hex()
        cases.append(
            (
                f"hex [{n} chars]",
                lambda a=a, b=b: h.hamming_distance_string(a, b),
                lambda a=a, b=b: stdlib_hex(a, b),
            )
        )
    return "stdlib", ("Input", "hexhamming (ns)", "stdlib (ns)", "Speedup"), cases


def section_api():
    s3, s3b = "abc", "abd"
    s64, t64 = rb(32).hex(), rb(32).hex()
    s1k, t1k = rb(512).hex(), rb(512).hex()
    f1k, z1k = "F" * 1000, "0" * 1000
    b3, c3 = b"abc", b"abd"
    b64, c64 = rb(64), rb(64)
    b1k, c1k = rb(1024), rb(1024)
    ba64, mv64 = bytearray(b64), memoryview(b64)
    b16, c16 = rb(16), rb(16)
    b127, c127 = rb(127), rb(127)
    small = {
        pos: planted(16, 512, at)
        for pos, at in (("at start", 0), ("mid", 256), ("at end", 511))
    }
    big = {
        pos: planted(64, 16384, at)
        for pos, at in (("at start", 0), ("mid", 8192), ("at end", 16383))
    }
    rows = [
        (
            "hamming_distance_string [3 chars, same]",
            lambda: h.hamming_distance_string(s3, s3),
        ),
        (
            "hamming_distance_string [3 chars, diff]",
            lambda: h.hamming_distance_string(s3, s3b),
        ),
        (
            "hamming_distance_string [64 chars, diff]",
            lambda: h.hamming_distance_string(s64, t64),
        ),
        (
            "hamming_distance_string [1024 chars, diff]",
            lambda: h.hamming_distance_string(s1k, t1k),
        ),
        (
            "hamming_distance_bytes [3 bytes, same]",
            lambda: h.hamming_distance_bytes(b3, b3),
        ),
        (
            "hamming_distance_bytes [3 bytes, diff]",
            lambda: h.hamming_distance_bytes(b3, c3),
        ),
        (
            "hamming_distance_bytes [64 bytes, diff]",
            lambda: h.hamming_distance_bytes(b64, c64),
        ),
        (
            "hamming_distance_bytes [1024 bytes, diff]",
            lambda: h.hamming_distance_bytes(b1k, c1k),
        ),
        (
            "hamming_distance_bytes [64-byte bytearray]",
            lambda: h.hamming_distance_bytes(ba64, ba64),
        ),
        (
            "hamming_distance_bytes [64-byte memoryview]",
            lambda: h.hamming_distance_bytes(mv64, mv64),
        ),
        (
            "check_hexstrings_within_dist [1000 chars, early exit]",
            lambda: h.check_hexstrings_within_dist(f1k, z1k, 20),
        ),
        (
            "check_bytes_within_dist [16 bytes]",
            lambda: h.check_bytes_within_dist(b16, c16, 40),
        ),
        (
            "check_bytes_within_dist [64 bytes]",
            lambda: h.check_bytes_within_dist(b64, c64, 200),
        ),
        (
            "check_bytes_within_dist [127 bytes]",
            lambda: h.check_bytes_within_dist(b127, c127, 400),
        ),
    ]
    for label, cats in (("512×16", small), ("16384×64", big)):
        for pos, (records, query) in cats.items():
            rows.append(
                (
                    f"first_within_dist [{label}, {pos}]",
                    lambda r=records, q=query: h.check_bytes_arrays_first_within_dist(
                        r, q, 0
                    ),
                )
            )
    for label, cats in (("512×16", small), ("16384×64", big)):
        for pos in ("at start", "at end") if label == "512×16" else ("mid",):
            records, query = cats[pos]
            rows.append(
                (
                    f"best_within_dist [{label}, {pos}]",
                    lambda r=records, q=query: h.check_bytes_arrays_best_within_dist(
                        r, q, 0
                    ),
                )
            )
    for label, cats in (("512×16", small), ("16384×64", big)):
        for pos in ("at start", "at end") if label == "512×16" else ("mid",):
            records, query = cats[pos]
            rows.append(
                (
                    f"all_within_dist [{label}, {pos}]",
                    lambda r=records, q=query: h.check_bytes_arrays_all_within_dist(
                        r, q, 0
                    ),
                )
            )
    return "api", ("Name", "Mean (ns)"), rows


def section_matrix():
    rows = []
    for case in ("random no-match", "exact midpoint"):
        for op in ("first", "best", "all"):
            fns = []
            for width in (16, 32):
                records, query = planted(
                    width, 1024, 512 if case == "exact midpoint" else None
                )
                fn = getattr(h, f"check_bytes_arrays_{op}_within_dist")
                fns.append(lambda f=fn, r=records, q=query: f(r, q, 0))
            rows.append((f"{case} / {op}", *fns))
    return "matrix", ("Case", "16-byte (ns)", "32-byte (ns)"), rows


def section_pairwise():
    rows = []
    for width in (16, 32):
        for count in (100, 1000, 10000):
            a, b = rb(width * count), rb(width * count)
            out = bytearray(8 * count)

            def loop(a=a, b=b, w=width, n=count):
                return [
                    h.hamming_distance_bytes(
                        a[i * w : (i + 1) * w], b[i * w : (i + 1) * w]
                    )
                    for i in range(n)
                ]

            rows.append(
                (
                    f"pairwise {count:,}×{width}",
                    loop,
                    lambda a=a, b=b, w=width: h.hamming_distances_bytes(a, b, w),
                    lambda a=a, b=b, w=width: h.hamming_distances_bytes_packed(a, b, w),
                    lambda a=a, b=b, w=width, o=out: h.hamming_distances_bytes_into(
                        a, b, w, o
                    ),
                )
            )
    return (
        "pairwise",
        ("Case", "loop (ns)", "list (ns)", "packed (ns)", "into (ns)"),
        rows,
    )


def section_many():
    catalog = rb(16 * 1024)
    queries = rb(16 * 100)
    qs = [queries[i * 16 : (i + 1) * 16] for i in range(100)]
    rows = [
        (
            "first_many 100×1024×16 (permissive threshold)",
            lambda: [
                h.check_bytes_arrays_first_within_dist(catalog, q, 128) for q in qs
            ],
            lambda: h.check_bytes_arrays_first_many_within_dist(
                catalog, queries, 16, 128
            ),
        ),
        (
            "best_many 100×1024×16 (max_dist=128)",
            lambda: [
                h.check_bytes_arrays_best_within_dist(catalog, q, 128) for q in qs
            ],
            lambda: h.check_bytes_arrays_best_many_within_dist(
                catalog, queries, 16, 128
            ),
        ),
    ]
    return "many", ("Case", "loop (ns)", "batch (ns)"), rows


def section_dense():
    catalog = rb(16 * 1024)
    query = rb(16)
    d_out, i_out = bytearray(2 * 1024), bytearray(4 * 1024)
    rows = [
        (
            "all 1024×16 (max_dist=128, all match)",
            lambda: h.check_bytes_arrays_all_within_dist(catalog, query, 128),
            lambda: h.check_bytes_arrays_all_within_dist_packed(catalog, query, 128),
            lambda: h.check_bytes_arrays_all_within_dist_into(
                catalog, query, 128, d_out, i_out
            ),
        )
    ]
    return "dense", ("Case", "list (ns)", "packed (ns)", "into (ns)"), rows


def section_large():
    rows = []
    for mb in (1, 16, 64):
        records, query = planted(16, mb * 1024 * 1024 // 16, None)
        rows.append(
            (
                f"{mb} MiB, 16-byte records, no match",
                lambda r=records, q=query: h.check_bytes_arrays_first_within_dist(
                    r, q, 0
                ),
                lambda r=records, q=query: h.check_bytes_arrays_best_within_dist(
                    r, q, 0
                ),
                lambda r=records, q=query: h.check_bytes_arrays_all_within_dist(
                    r, q, 0
                ),
            )
        )
    return "large", ("Catalog", "first (ns)", "best (ns)", "all (ns)"), rows


def section_catalog():
    rows = []
    for width, radii in ((8, (2, 8, 12)), (32, (8, 32))):
        records = rb(width * 1_000_000)
        indexed = h.Catalog(records, width, index=True)
        target = bytearray(records[123_456 * width : 123_457 * width])
        for r in radii:
            q = bytearray(target)
            for pos in RNG.sample(range(width * 8), r):
                q[pos // 8] ^= 1 << (pos % 8)
            q = bytes(q)
            rows.append(
                (
                    f"{width} B, radius {r}",
                    lambda rec=records, q=q, r=r: h.check_bytes_arrays_best_within_dist(
                        rec, q, r
                    ),
                    lambda cat=indexed, q=q, r=r: cat.best_within(q, r),
                )
            )
    return "catalog", ("1M records", "free function (ns)", "Catalog, index (ns)"), rows


SECTIONS = [
    section_stdlib,
    section_api,
    section_matrix,
    section_pairwise,
    section_many,
    section_dense,
    section_large,
    section_catalog,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    for section in SECTIONS:
        # Skip before building cases, so older builds can run other sections.
        if args.only and section.__name__ != f"section_{args.only}":
            continue
        name, header, cases = section()
        results = []
        for label, *fns in cases:
            numbers = [calibrated(fn) for fn in fns]
            per_run = [
                [best_ns(fn, n) for fn, n in zip(fns, numbers)]
                for _ in range(args.runs)
            ]
            results.append((label, [statistics.median(col) for col in zip(*per_run)]))
        rows = []
        for label, values in results:
            cells = [fmt(v) for v in values]
            if name == "stdlib":
                cells.append(f"{values[1] / values[0]:.2f}×")
            rows.append((label, *cells))
        print(table(f"[{name}]", header, rows))
        print()


if __name__ == "__main__":
    main()
