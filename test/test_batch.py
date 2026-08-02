"""Tests for the batch APIs added in perf/python-batch-apis.

Covers:
* Pairwise list/packed/into against a hand-oracle and each other.
* Multi-query first/best/all_many parity with repeated single-call use.
* Packed and _into transport for `all_within_dist` (dense + sparse).
* Error shapes for element_size / length mismatches, readonly outputs,
  non-contiguous outputs, wrong-size outputs, and unaligned writable memoryview
  targets.
* Algorithm invariance and ordering / tie behavior preserved.
"""

import array
import random

import pytest
from hexhamming import (
    check_bytes_arrays_all_many_within_dist,
    check_bytes_arrays_all_within_dist,
    check_bytes_arrays_all_within_dist_into,
    check_bytes_arrays_all_within_dist_packed,
    check_bytes_arrays_best_many_within_dist,
    check_bytes_arrays_best_within_dist,
    check_bytes_arrays_first_many_within_dist,
    check_bytes_arrays_first_within_dist,
    hamming_distances_bytes,
    hamming_distances_bytes_into,
    hamming_distances_bytes_packed,
    set_algo,
)


def _random_bytes(length: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(length))


def _oracle_distance(a: bytes, b: bytes) -> int:
    assert len(a) == len(b)
    total = 0
    for x, y in zip(a, b):
        total += (x ^ y).bit_count()
    return total


# ---------------------------------------------------------------------------
# Pairwise batch API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", (1, 16, 24, 32, 33))
def test_pairwise_matches_oracle_random(width):
    count = 41
    a = _random_bytes(width * count, 0x51_0000 + width)
    b = _random_bytes(width * count, 0x52_0000 + width)
    expected = [
        _oracle_distance(a[i * width : (i + 1) * width], b[i * width : (i + 1) * width])
        for i in range(count)
    ]
    assert hamming_distances_bytes(a, b, width) == expected


def test_pairwise_empty_batch():
    assert hamming_distances_bytes(b"", b"", 16) == []
    assert hamming_distances_bytes_packed(b"", b"", 16) == b""
    out = bytearray(0)
    assert hamming_distances_bytes_into(b"", b"", 16, out) == 0
    assert bytes(out) == b""


def test_pairwise_error_shapes():
    with pytest.raises(ValueError):
        hamming_distances_bytes(b"aa", b"bb", 0)
    with pytest.raises(ValueError):
        hamming_distances_bytes(b"aa", b"bbb", 1)
    with pytest.raises(ValueError):
        hamming_distances_bytes(b"aaa", b"bbb", 2)


def test_pairwise_packed_matches_list():
    width = 16
    count = 5
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    dists = hamming_distances_bytes(a, b, width)
    packed = hamming_distances_bytes_packed(a, b, width)
    assert len(packed) == count * 8
    parsed = [
        int.from_bytes(packed[i * 8 : (i + 1) * 8], "little") for i in range(count)
    ]
    assert parsed == dists


def test_pairwise_into_matches_packed():
    width = 32
    count = 7
    a = _random_bytes(width * count, 10)
    b = _random_bytes(width * count, 11)
    dists = hamming_distances_bytes(a, b, width)
    out = bytearray(count * 8)
    n = hamming_distances_bytes_into(a, b, width, out)
    assert n == count
    parsed = [int.from_bytes(out[i * 8 : (i + 1) * 8], "little") for i in range(count)]
    assert parsed == dists


def test_pairwise_into_rejects_readonly():
    width = 16
    count = 3
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    with pytest.raises(ValueError):
        hamming_distances_bytes_into(a, b, width, bytes(count * 8))


def test_pairwise_into_rejects_wrong_size():
    width = 16
    count = 3
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    with pytest.raises(ValueError):
        hamming_distances_bytes_into(a, b, width, bytearray(count * 8 - 1))
    with pytest.raises(ValueError):
        hamming_distances_bytes_into(a, b, width, bytearray(count * 8 + 1))


def test_pairwise_into_rejects_noncontiguous_output():
    width = 16
    count = 4
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    backing = bytearray(count * 8 * 2)
    # Every-other-byte stride is not C-contiguous.
    mv = memoryview(backing)[::2]
    with pytest.raises(ValueError):
        hamming_distances_bytes_into(a, b, width, mv)


def test_pairwise_into_accepts_unaligned_memoryview():
    # Provide an intentionally unaligned writable byte view: slice a byte off
    # the front of a larger backing buffer. Writes must be safe regardless of
    # alignment because we use write_unaligned.
    width = 16
    count = 5
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    backing = bytearray(count * 8 + 3)
    view = memoryview(backing)[3 : 3 + count * 8]
    n = hamming_distances_bytes_into(a, b, width, view)
    assert n == count
    dists = hamming_distances_bytes(a, b, width)
    parsed = [
        int.from_bytes(bytes(view[i * 8 : (i + 1) * 8]), "little") for i in range(count)
    ]
    assert parsed == dists


def test_pairwise_into_rejects_overlapping_input():
    backing = bytearray(range(20))
    a = memoryview(backing)[:16]
    b = bytes(16)
    output = memoryview(backing)[4:20]
    with pytest.raises(ValueError, match="must not overlap input"):
        hamming_distances_bytes_into(a, b, 8, output)


def test_pairwise_algorithm_invariance():
    width = 16
    count = 20
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    baseline = hamming_distances_bytes(a, b, width)
    for algo in ("classic", "native"):
        assert set_algo(algo) == ""
        assert hamming_distances_bytes(a, b, width) == baseline
    set_algo("native")


# ---------------------------------------------------------------------------
# Multi-query catalog scans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", (16, 24, 32))
def test_multi_query_first_matches_repeated(width):
    catalog = _random_bytes(width * 50, 0x100 + width)
    queries = _random_bytes(width * 4, 0x200 + width)
    got = check_bytes_arrays_first_many_within_dist(catalog, queries, width, 8)
    expected = [
        check_bytes_arrays_first_within_dist(
            catalog, queries[i * width : (i + 1) * width], 8
        )
        for i in range(len(queries) // width)
    ]
    assert got == expected


@pytest.mark.parametrize("width", (16, 24, 32))
def test_multi_query_best_matches_repeated(width):
    catalog = _random_bytes(width * 50, 0x300 + width)
    queries = _random_bytes(width * 4, 0x400 + width)
    got = check_bytes_arrays_best_many_within_dist(catalog, queries, width, 32)
    expected = [
        check_bytes_arrays_best_within_dist(
            catalog, queries[i * width : (i + 1) * width], 32
        )
        for i in range(len(queries) // width)
    ]
    assert got == expected


@pytest.mark.parametrize("width", (16, 24, 32))
def test_multi_query_all_matches_repeated(width):
    catalog = _random_bytes(width * 30, 0x500 + width)
    queries = _random_bytes(width * 3, 0x600 + width)
    max_dist = 30
    got = check_bytes_arrays_all_many_within_dist(catalog, queries, width, max_dist)
    expected = [
        check_bytes_arrays_all_within_dist(
            catalog, queries[i * width : (i + 1) * width], max_dist
        )
        for i in range(len(queries) // width)
    ]
    assert got == expected


def test_multi_query_empty_queries_produces_empty_list():
    catalog = _random_bytes(16 * 5, 1)
    assert check_bytes_arrays_first_many_within_dist(catalog, b"", 16, 8) == []
    assert check_bytes_arrays_best_many_within_dist(catalog, b"", 16, 8) == []
    assert check_bytes_arrays_all_many_within_dist(catalog, b"", 16, 8) == []


def test_multi_query_missing_uses_minus_one_sentinels():
    catalog = b"\xff" * 32
    query = b"\x00" * 16  # distance 128, always > max_dist 4
    got_first = check_bytes_arrays_first_many_within_dist(catalog, query, 16, 4)
    assert got_first == [-1]
    got_best = check_bytes_arrays_best_many_within_dist(catalog, query, 16, 4)
    assert got_best == [(-1, -1)]
    got_all = check_bytes_arrays_all_many_within_dist(catalog, query, 16, 4)
    assert got_all == [[]]


def test_multi_query_error_shapes():
    catalog = _random_bytes(16 * 3, 1)
    with pytest.raises(ValueError):
        check_bytes_arrays_first_many_within_dist(catalog, b"", 0, 0)
    with pytest.raises(ValueError):
        check_bytes_arrays_first_many_within_dist(b"\x00\x00\x00", b"\x00\x00", 2, 0)
    with pytest.raises(ValueError):
        check_bytes_arrays_first_many_within_dist(catalog, b"\x00\x00\x00", 2, 0)
    with pytest.raises(ValueError):
        check_bytes_arrays_first_many_within_dist(catalog, b"\x00" * 16, 16, -1)


def test_multi_query_best_tiebreak_lowest_index():
    width = 16
    records = [b"\xff" * width] * 20
    records[4] = b"\x00" * width
    records[9] = b"\x00" * width  # duplicate exact match at higher index
    catalog = b"".join(records)
    got = check_bytes_arrays_best_many_within_dist(catalog, b"\x00" * width, width, 4)
    assert got == [(0, 4)]


def test_multi_query_all_ordering_preserved():
    width = 16
    records = [b"\xff" * width] * 12
    for idx in (1, 5, 8, 11):
        records[idx] = b"\x00" * width
    catalog = b"".join(records)
    got = check_bytes_arrays_all_many_within_dist(catalog, b"\x00" * width, width, 4)
    assert got == [[(0, 1), (0, 5), (0, 8), (0, 11)]]


# ---------------------------------------------------------------------------
# Packed / into transport for all-results
# ---------------------------------------------------------------------------


def test_packed_all_dense_parity_with_list():
    width = 16
    catalog = _random_bytes(width * 64, 71)
    query = catalog[3 * width : 4 * width]
    list_result = check_bytes_arrays_all_within_dist(catalog, query, 128)
    d_bytes, i_bytes = check_bytes_arrays_all_within_dist_packed(catalog, query, 128)
    assert len(d_bytes) == len(list_result) * 2
    assert len(i_bytes) == len(list_result) * 4
    for k, (d, i) in enumerate(list_result):
        assert int.from_bytes(d_bytes[k * 2 : (k + 1) * 2], "little") == d
        assert int.from_bytes(i_bytes[k * 4 : (k + 1) * 4], "little") == i


def test_packed_all_sparse_parity_with_list():
    width = 16
    catalog = _random_bytes(width * 200, 81)
    query = catalog[7 * width : 8 * width]
    list_result = check_bytes_arrays_all_within_dist(catalog, query, 0)
    d_bytes, i_bytes = check_bytes_arrays_all_within_dist_packed(catalog, query, 0)
    assert len(d_bytes) // 2 == len(list_result)
    parsed = [
        (
            int.from_bytes(d_bytes[k * 2 : (k + 1) * 2], "little"),
            int.from_bytes(i_bytes[k * 4 : (k + 1) * 4], "little"),
        )
        for k in range(len(list_result))
    ]
    assert parsed == list_result


def test_into_all_matches_packed():
    width = 16
    count = 64
    catalog = _random_bytes(width * count, 91)
    query = catalog[2 * width : 3 * width]
    d_bytes, i_bytes = check_bytes_arrays_all_within_dist_packed(catalog, query, 128)
    matches = len(d_bytes) // 2
    d_out = bytearray(count * 2)
    i_out = bytearray(count * 4)
    n = check_bytes_arrays_all_within_dist_into(catalog, query, 128, d_out, i_out)
    assert n == matches
    assert bytes(d_out[: n * 2]) == bytes(d_bytes)
    assert bytes(i_out[: n * 4]) == bytes(i_bytes)


def test_into_all_rejects_readonly_buffers():
    width = 16
    catalog = _random_bytes(width * 8, 101)
    query = catalog[:width]
    with pytest.raises(ValueError):
        check_bytes_arrays_all_within_dist_into(
            catalog, query, 128, bytes(16), bytearray(32)
        )
    with pytest.raises(ValueError):
        check_bytes_arrays_all_within_dist_into(
            catalog, query, 128, bytearray(16), bytes(32)
        )


def test_into_all_rejects_short_buffers():
    width = 16
    catalog = _random_bytes(width * 8, 111)
    query = catalog[:width]
    with pytest.raises(ValueError):
        check_bytes_arrays_all_within_dist_into(
            catalog, query, 128, bytearray(4), bytearray(32)
        )
    with pytest.raises(ValueError):
        check_bytes_arrays_all_within_dist_into(
            catalog, query, 128, bytearray(16), bytearray(4)
        )


def test_into_all_accepts_unaligned_memoryview():
    width = 16
    count = 16
    catalog = _random_bytes(width * count, 121)
    query = catalog[3 * width : 4 * width]
    d_backing = bytearray(count * 2 + 5)
    i_backing = bytearray(count * 4 + 7)
    d_view = memoryview(d_backing)[5 : 5 + count * 2]
    i_view = memoryview(i_backing)[7 : 7 + count * 4]
    n = check_bytes_arrays_all_within_dist_into(catalog, query, 128, d_view, i_view)
    d_bytes, i_bytes = check_bytes_arrays_all_within_dist_packed(catalog, query, 128)
    assert n == len(d_bytes) // 2
    assert bytes(d_view[: n * 2]) == bytes(d_bytes)
    assert bytes(i_view[: n * 4]) == bytes(i_bytes)


def test_into_all_rejects_overlapping_outputs():
    catalog = bytes(range(64))
    query = bytes(16)
    backing = bytearray(20)
    with pytest.raises(ValueError, match="must not overlap each other"):
        check_bytes_arrays_all_within_dist_into(
            catalog,
            query,
            128,
            memoryview(backing)[:8],
            memoryview(backing)[4:20],
        )


def test_into_all_rejects_overlapping_input():
    catalog = bytearray(range(64))
    query = bytes(16)
    with pytest.raises(ValueError, match="must not overlap input"):
        check_bytes_arrays_all_within_dist_into(
            catalog,
            query,
            128,
            memoryview(catalog)[:8],
            bytearray(16),
        )


def test_pairwise_accepts_bytearray_and_memoryview_and_array():
    width = 16
    count = 4
    a = _random_bytes(width * count, 1)
    b = _random_bytes(width * count, 2)
    expected = hamming_distances_bytes(a, b, width)
    assert hamming_distances_bytes(bytearray(a), bytearray(b), width) == expected
    assert hamming_distances_bytes(memoryview(a), memoryview(b), width) == expected
    aa = array.array("B", a)
    bb = array.array("B", b)
    assert hamming_distances_bytes(aa, bb, width) == expected
