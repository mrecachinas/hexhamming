import inspect
import random
from platform import machine, python_implementation

import pytest
from hexhamming import (
    Catalog,
    check_bytes_arrays_all_many_within_dist,
    check_bytes_arrays_all_within_dist,
    check_bytes_arrays_best_many_within_dist,
    check_bytes_arrays_best_within_dist,
    check_bytes_arrays_first_many_within_dist,
    check_bytes_arrays_first_within_dist,
    check_bytes_arrays_within_dist,
    check_bytes_within_dist,
    check_hexstrings_within_dist,
    hamming_distance_bytes,
    hamming_distance_string,
    hamming_distances_bytes,
    set_algo,
)

############################
# Core Hamming distance APIs
#
# These cases cover strings and bytes from empty inputs through large repeated
# buffers, so both scalar and vectorized chunks must return the same distance.
############################


@pytest.mark.parametrize(
    "hex1,hex2,expected",
    (
        ("abc", "abc", 0),
        ("000", "001", 1),
        ("ABCDEF", "000001", 16),
        ("", "", 0),
        ("f" * 64, "0" * 64, 256),
        ("f" * 64, "f" * 64, 0),
        ("0" * 64, "0" * 64, 0),
        ("f" * 10000, "0" * 10000, 40000),
        ("f" * 10000, "f" * 10000, 0),
    ),
    ids=(
        "3-same",
        "3-diff",
        "6-different",
        "empty-empty",
        "64-f-0",
        "64-f-f",
        "64-0-0",
        "10000-f-0",
        "10000-f-f",
    ),
)
def test_hamming_distance_string(hex1, hex2, expected):
    assert expected == hamming_distance_string(hex1, hex2)
    # Empty string means the requested backend was accepted; rechecking locks
    # the explicit classic string path to the same result as the default path.
    assert len(set_algo("classic")) == 0
    assert expected == hamming_distance_string(hex1, hex2)


############################
# Catalog API
############################


def _lcg_bytes(size, seed):
    state = seed
    out = bytearray(size)
    for i in range(size):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out[i] = (state >> 32) & 0xFF
    return bytes(out)


def _flip_bits(buf, bits):
    data = bytearray(buf)
    for bit in bits:
        data[bit // 8] ^= 1 << (bit & 7)
    return bytes(data)


def test_catalog_api_matches_free_functions():
    width = 8
    records = bytearray(_lcg_bytes(width * 2048, 11))
    query = _flip_bits(records[123 * width : 124 * width], [1, 9])
    records[700 * width : 701 * width] = query
    records = bytes(records)

    linear = Catalog(records, width)
    indexed = Catalog(records, width, index=True)
    assert len(linear) == 2048
    assert linear.width == width
    assert not linear.has_index
    assert indexed.has_index

    for radius in (0, 2, 8, 64):
        expected_first = check_bytes_arrays_first_within_dist(records, query, radius)
        expected_best = check_bytes_arrays_best_within_dist(records, query, radius)
        expected_all = check_bytes_arrays_all_within_dist(records, query, radius)
        assert linear.first_within(query, radius) == expected_first
        assert indexed.first_within(query, radius) == expected_first
        assert linear.best_within(query, radius) == expected_best
        assert indexed.best_within(query, radius) == expected_best
        assert linear.all_within(query, radius) == expected_all
        assert indexed.all_within(query, radius) == expected_all


def test_catalog_many_api_matches_free_functions():
    width = 8
    records = _lcg_bytes(width * 2048, 21)
    queries = (
        records[0:width]
        + _flip_bits(records[99 * width : 100 * width], [0, 17])
        + _lcg_bytes(width, 22)
    )
    cat = Catalog(records, width, index=True)
    for radius in (0, 2, 8, 64):
        assert cat.first_many_within(
            queries, radius
        ) == check_bytes_arrays_first_many_within_dist(records, queries, width, radius)
        assert cat.best_many_within(
            queries, radius
        ) == check_bytes_arrays_best_many_within_dist(records, queries, width, radius)
        assert cat.all_many_within(
            queries, radius
        ) == check_bytes_arrays_all_many_within_dist(records, queries, width, radius)


def test_catalog_buffer_inputs_and_errors():
    np = pytest.importorskip("numpy")
    records = bytearray(b"\x00\x00\xff\xff\x0f\xf0")
    cat = Catalog(memoryview(records), 2, index=True)
    assert cat.first_within(bytearray(b"\xff\xff"), 0) == 1
    assert cat.best_within(np.frombuffer(b"\x0f\xf0", dtype=np.uint8), 0) == (0, 2)
    assert cat.all_within(memoryview(b"\x00\x00"), 0) == [(0, 0)]

    with pytest.raises(ValueError, match="width must be >0"):
        Catalog(b"abc", 0)
    with pytest.raises(ValueError, match="records size must be multiplier"):
        Catalog(b"abc", 2)
    with pytest.raises(ValueError, match="query size must equal catalog width"):
        cat.first_within(b"\x00", 0)
    with pytest.raises(ValueError, match="queries size must be multiplier"):
        cat.first_many_within(b"\x00", 0)
    with pytest.raises(ValueError, match="`max_dist` must be >=0"):
        cat.first_within(b"\x00\x00", -1)


def test_catalog_queries_release_gil_from_threads():
    from concurrent.futures import ThreadPoolExecutor

    width = 8
    records = _lcg_bytes(width * 4096, 31)
    cat = Catalog(records, width, index=True)
    queries = [records[i * width : (i + 1) * width] for i in range(0, 128, 7)]

    def run(query):
        return (
            cat.first_within(query, 0),
            cat.best_within(query, 0),
            cat.all_within(query, 0),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        got = list(executor.map(run, queries * 20))
    assert all(
        first >= 0 and best[0] == 0 and all_matches for first, best, all_matches in got
    )


@pytest.mark.parametrize(
    "hex1,hex2,expected",
    (
        (b"\xab\x0c", b"\xab\x0c", 0),
        (b"\x00", b"\x01", 1),
        (b"\xab\xcd\xef", b"\x00\x00\x01", 16),
        (b"", b"", 0),
        (b"\xff" * 32, b"\x00" * 32, 256),
        (b"\xff" * 32, b"\xff" * 32, 0),
        (b"\x00" * 32, b"\x00" * 32, 0),
        (b"\xff" * 5000, b"\x00" * 5000, 40000),
        (b"\xff" * 5000, b"\xff" * 5000, 0),
    ),
    ids=(
        "4-same",
        "2-diff",
        "6-different",
        "empty-empty",
        "64-f-0",
        "64-f-f",
        "64-0-0",
        "10000-f-0",
        "10000-f-f",
    ),
)
def test_hamming_distance_byte(hex1, hex2, expected):
    algorithm_list = ["extra", "native", "classic"]
    if machine().lower().startswith("x86"):
        algorithm_list.append("sse41")
    # Every available backend should agree; unsupported optional SIMD backends
    # return a message from set_algo and are skipped on that machine.
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f"Warning: Skipping {algorithm}, reason: {result}")
            continue
        assert expected == hamming_distance_bytes(hex1, hex2)


@pytest.mark.parametrize(
    "hex1,hex2,exception,msg",
    (
        ("abc", 3, TypeError, "not an instance of 'str'"),
        ("abc", "a", ValueError, "strings are NOT the same length"),
        ("lol", "foo", ValueError, "hex string contains invalid char"),
        ("000abcdef", "011abcdgf", ValueError, "hex string contains invalid char"),
        ("f" * 32, "f" * 31 + "g", ValueError, "hex string contains invalid char"),
        ("f" * 30, "f" * 29 + "g", ValueError, "hex string contains invalid char"),
        ("ggg", "ggg", ValueError, "hex string contains invalid char"),
        (
            "g" * 15 + "fff",
            "g" * 15 + "000",
            ValueError,
            "hex string contains invalid char",
        ),
    ),
)
def test_hamming_distance_string_errors(hex1, hex2, exception, msg):
    # Python callers rely on both exception class and message fragment for
    # invalid types, length mismatches, and invalid hexadecimal characters.
    with pytest.raises(exception) as excinfo:
        _ = hamming_distance_string(hex1, hex2)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "hex1,hex2,max_dist,expected",
    (
        ("000abcdef", "011abcdef", 3, True),
        ("1f0abcdef", "011abcdef", 3, False),
        ("011abcdef", "011abcdef", 1000, True),
    ),
)
def test_check_hexstrings_within_dist(hex1, hex2, max_dist, expected):
    # Threshold checks should match full-distance semantics while allowing the
    # Rust implementation to stop once it exceeds max_dist.
    algorithm_list = ["extra", "native", "classic"]
    if machine().lower().startswith("x86"):
        algorithm_list.append("sse41")
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f"Warning: Skipping {algorithm}, reason: {result}")
            continue
        assert expected == check_hexstrings_within_dist(hex1, hex2, max_dist)


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,expected",
    (
        (b"\x00\x0a\xbc\xde\xf0", b"\x01\x1a\xbc\xde\xf0", 3, True),
        (b"\x1f\x0a\xbc\xde\xf0", b"\x01\x1a\xbc\xde\xf0", 3, False),
        (b"\x01\x1a\xbc\xde\xf0", b"\x01\x1a\xbc\xde\xf0", 1000, True),
    ),
)
def test_check_bytes_within_dist(bytes1, bytes2, max_dist, expected):
    # Byte threshold checks exercise the same backend dispatch as full byte
    # distances, but return only whether the distance is within max_dist.
    algorithm_list = ["extra", "native", "classic"]
    if machine().lower().startswith("x86"):
        algorithm_list.append("sse41")
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f"Warning: Skipping {algorithm}, reason: {result}")
            continue
        assert expected == check_bytes_within_dist(bytes1, bytes2, max_dist)


@pytest.mark.parametrize(
    "hex1,hex2,max_dist,exception,msg",
    (
        (
            "000abcdef",
            "011abcdef",
            None,
            TypeError,
            "object cannot be",
        ),
        (
            "000abcdef",
            "011abcdef",
            "HELLO",
            TypeError,
            "object cannot be",
        ),
        ("000abcdef", "011abcdef", -1, ValueError, "`max_dist` must be >0"),
        ("000abcdef", "011abcdzz", 3, ValueError, "hex string contains invalid char"),
        ("000abcdef", "011abcdgf", 3, ValueError, "hex string contains invalid char"),
        ("1f0abcdef", 3, 3, TypeError, "not an instance of 'str'"),
        ("011abcdef", "00", 3, ValueError, "strings are NOT the same length"),
    ),
)
def test_check_hexstrings_within_dist_errors(hex1, hex2, max_dist, exception, msg):
    # Error paths are part of the Python API contract, including max_dist
    # validation before doing expensive comparisons.
    with pytest.raises(exception) as excinfo:
        _ = check_hexstrings_within_dist(hex1, hex2, max_dist)
    assert msg in str(excinfo.value)


############################
# Array-of-elements byte APIs
#
# These APIs compare one element against consecutive fixed-size elements in a
# larger bytes-like array; size validation prevents ambiguous chunking.
############################


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,exception,msg",
    (
        (
            b"\x00" * 16,
            b"\x00" * 16,
            None,
            TypeError,
            "object cannot be",
        ),
        (
            b"\x00" * 16,
            b"\x00" * 16,
            "HELLO",
            TypeError,
            "object cannot be",
        ),
        (b"\x00" * 32, b"\x00" * 16, -1, ValueError, "`max_dist` must be >=0"),
        (
            b"\x00" * 31,
            b"\x00" * 16,
            3,
            ValueError,
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ),
        (b"\x00" * 32, b"", 3, ValueError, "`elem_to_compare` size must be >0"),
    ),
)
def test_check_bytes_arrays_first_within_dist_invalid_values(
    bytes1, bytes2, max_dist, exception, msg
):
    with pytest.raises(exception) as excinfo:
        _ = check_bytes_arrays_first_within_dist(bytes1, bytes2, max_dist)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,exception,msg",
    (
        (
            b"\x00" * 31,
            b"\x00" * 16,
            100,
            ValueError,
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ),
        (b"\x00" * 32, b"", 100, ValueError, "`elem_to_compare` size must be >0"),
    ),
)
def test_check_bytes_arrays_all_within_dist_invalid_values(
    bytes1, bytes2, max_dist, exception, msg
):
    with pytest.raises(exception) as excinfo:
        _ = check_bytes_arrays_all_within_dist(bytes1, bytes2, max_dist)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,exception,msg",
    (
        (
            b"\x00" * 16,
            b"\x00" * 16,
            None,
            TypeError,
            "object cannot be",
        ),
        (
            b"\x00" * 16,
            b"\x00" * 16,
            "HELLO",
            TypeError,
            "object cannot be",
        ),
        (b"\x00" * 32, b"\x00" * 16, -1, ValueError, "`max_dist` must be >=0"),
        (
            b"\x00" * 31,
            b"\x00" * 16,
            3,
            ValueError,
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ),
        (b"\x00" * 32, b"", 3, ValueError, "`elem_to_compare` size must be >0"),
    ),
)
def test_check_bytes_arrays_best_within_dist_invalid_values(
    bytes1, bytes2, max_dist, exception, msg
):
    with pytest.raises(exception) as excinfo:
        _ = check_bytes_arrays_best_within_dist(bytes1, bytes2, max_dist)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,expected",
    (
        (
            b"\x00" * 16,
            b"\xff" * 16,
            50,
            -1,
        ),
        (
            b"\x00" * 16,
            b"\x00" * 15 + b"\x0f" * 1,
            4,
            0,
        ),
        (
            b"\xff" * 16 * 8 + b"\x0f" * 16,
            b"\x00" * 2 + b"\x0f" * 14,
            8,
            8,
        ),
        (
            b"\xf0" * 64 + b"\x0a" * 64,
            b"\x0f" * 64,
            3 * 64,
            1,
        ),
    ),
)
def test_check_bytes_arrays_first_within_dist_calculation(
    bytes1, bytes2, max_dist, expected
):
    # The "first" API returns the index of the first within-threshold element,
    # or -1 when no element is close enough.
    algorithm_list = ["extra", "native", "classic"]
    if machine().lower().startswith("x86"):
        algorithm_list.append("sse41")
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f"Warning: Skipping {algorithm}, reason: {result}")
            continue
        assert expected == check_bytes_arrays_first_within_dist(
            bytes1, bytes2, max_dist
        )


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,expected",
    (
        (
            b"\x00" * 16 * 4,
            b"\xff" * 16,
            50,
            [],
        ),
        (
            b"\x00" * 16,
            b"\x00" * 15 + b"\x0f" * 1,
            4,
            [(4, 0)],
        ),
        (
            b"\xff" * 16 * 8 + b"\x0f" * 16,
            b"\x00" * 2 + b"\x0f" * 14,
            8,
            [(8, 8)],
        ),
        (
            b"\xf0" * 64 + b"\x0a" * 64,
            b"\x0f" * 64,
            3 * 64,
            [(128, 1)],
        ),
        (
            b"\xff" * 16 * 4 + b"\x0f" * 16 + b"\xff" * 16 * 4 + b"\x0e" * 16,
            b"\x00" * 2 + b"\x0f" * 14,
            32,
            [(8, 4), (20, 9)],
        ),
    ),
)
def test_check_bytes_arrays_all_within_dist_calculation(
    bytes1, bytes2, max_dist, expected
):
    # The "all" API returns every match as (distance, element_index), preserving
    # scan order for callers that need more than the first or best match.
    algorithm_list = ["extra", "native", "classic"]
    if machine().lower().startswith("x86"):
        algorithm_list.append("sse41")
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f"Warning: Skipping {algorithm}, reason: {result}")
            continue
        assert expected == check_bytes_arrays_all_within_dist(bytes1, bytes2, max_dist)


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_fixed_width_array_apis_randomized_oracle(width):
    rng = random.Random(0x51_0000 + width)
    count = 37
    needle = bytes(rng.randrange(256) for _ in range(width))
    records = [bytes(rng.randrange(256) for _ in range(width)) for _ in range(count)]
    for index in (2, 19, 36):
        records[index] = needle
    near = bytearray(needle)
    near[0] ^= 0x0F
    records[12] = bytes(near)
    array = b"".join(records)

    for max_dist in (0, 3, 4, 5, 8, width * 8 - 1, width * 8, width * 8 + 1):
        distances = [
            (int.from_bytes(record, "big") ^ int.from_bytes(needle, "big")).bit_count()
            for record in records
        ]
        expected = [
            (distance, index)
            for index, distance in enumerate(distances)
            if distance <= max_dist
        ]
        assert check_bytes_arrays_first_within_dist(array, needle, max_dist) == (
            expected[0][1] if expected else -1
        )
        assert check_bytes_arrays_best_within_dist(array, needle, max_dist) == (
            min(expected, key=lambda item: (item[0], item[1])) if expected else (-1, -1)
        )
        assert check_bytes_arrays_all_within_dist(array, needle, max_dist) == expected


############################
# Benchmarks
#
# pytest-benchmark cases are kept next to correctness tests because speed is a
# core feature; inputs include same/different pairs and realistic vector sizes.
############################


@pytest.mark.benchmark(group="hamming_distance_string")
@pytest.mark.parametrize(
    ("hex1", "hex2"),
    (
        ("ABC", "DEF"),
        ("BBB", "BBB"),
        ("B" * 1000, "B" * 1000),
        ("F" * 1000, "0" * 1000),
        ("B" * 1024, "B" * 1024),
        ("F" * 1024, "0" * 1024),
        ("F" * 64, "0" * 64),
    ),
    ids=(
        "3-diff",
        "3-same",
        "1000-same",
        "1000-diff",
        "1024-same",
        "1024-diff",
        "64-diff",
    ),
)
def test_hamming_distance_string_bench(benchmark, hex1, hex2):
    benchmark(hamming_distance_string, hex1, hex2)


@pytest.mark.benchmark(group="hamming_distance_bytes")
@pytest.mark.parametrize(
    ("hex1", "hex2"),
    (
        (b"\xab\x0c", b"\xde\x0f"),
        (b"\xbb\x0b", b"\xbb\x0b"),
        (b"\xbb" * 500, b"\xbb" * 500),
        (b"\xff" * 500, b"\x00" * 500),
        (b"\xbb" * 512, b"\xbb" * 512),
        (b"\xff" * 512, b"\x00" * 512),
        (b"\xff" * 32, b"\x00" * 32),
    ),
    ids=(
        "3-diff",
        "3-same",
        "1000-same",
        "1000-diff",
        "1024-same",
        "1024-diff",
        "64-diff",
    ),
)
def test_hamming_distance_bytes_bench(benchmark, hex1, hex2):
    benchmark(hamming_distance_bytes, hex1, hex2)


@pytest.mark.benchmark(group="hamming_distance_bytes_buffer")
@pytest.mark.parametrize(
    "factory",
    (bytearray, memoryview),
    ids=("bytearray", "memoryview"),
)
def test_hamming_distance_bytes_buffer_bench(benchmark, factory):
    a = factory(b"\xff" * 64)
    b = factory(b"\x00" * 64)
    benchmark(hamming_distance_bytes, a, b)


def test_check_hexstrings_within_dist_bench(benchmark):
    benchmark(check_hexstrings_within_dist, "F" * 1000, "0" * 1000, 20)


@pytest.mark.benchmark(group="hamming_distance_check_bytes_within_dist_bench")
@pytest.mark.parametrize(
    ("bytes1", "bytes2", "max_dist"),
    (
        (b"\x00" * 16, b"\x00" * 16, 0),
        (b"\xff" * 64, b"\x00" * 64, 100),
        (b"\xff" * 127, b"\x00" * 127, 500),
    ),
    ids=("16 bytes,d=0", "64 bytes,d=100", "127 bytes,d=500"),
)
def test_check_bytes_within_dist_bench(benchmark, bytes1, bytes2, max_dist):
    benchmark(check_bytes_within_dist, bytes1, bytes2, max_dist)


@pytest.mark.benchmark(
    group="hamming_distance_check_bytes_arrays_first_within_dist_bench"
)
@pytest.mark.parametrize(
    ("bytes1", "bytes2", "max_dist"),
    (
        (b"\x00" * 16 + b"\x00\x03" * 8 * 511, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 256 + b"\x00" * 16 + b"\x00\x03" * 8 * 255, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 511 + b"\x00" * 16, b"\x00" * 16, 1),
        (b"\xff" * 32 + b"\x11" * 32 * 1023, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 511 + b"\xff" * 32 + b"\x11" * 32 * 512, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 1023 + b"\xff" * 32, b"\xfb" * 32, 4 * 32),
        (b"\xcc" * 64 + b"\x01" * 64 * 16383, b"\xfb" * 64, 5 * 64),
        (
            b"\x01" * 64 * 8191 + b"\xcc" * 64 + b"\x01" * 64 * 8192,
            b"\xfb" * 64,
            5 * 64,
        ),
        (b"\x01" * 64 * 16383 + b"\xcc" * 64, b"\xfb" * 64, 5 * 64),
    ),
    ids=(
        "  512 elems,s=16,at 0",
        "  512 elems,s=16,mid",
        "  512 elems,s=16,end",
        " 1024 elems,s=32,at 0",
        " 1024 elems,s=32,mid",
        " 1024 elems,s=32,end",
        "16384 elems,s=64,at 0",
        "16384 elems,s=64,mid",
        "16384 elems,s=64,end",
    ),
)
def test_check_bytes_arrays_first_within_dist_bench(
    benchmark, bytes1, bytes2, max_dist
):
    # Match position varies across start/middle/end to measure early-exit scans
    # as well as the raw per-element distance calculation.
    benchmark(check_bytes_arrays_first_within_dist, bytes1, bytes2, max_dist)


@pytest.mark.benchmark(
    group="hamming_distance_check_bytes_arrays_best_within_dist_bench"
)
@pytest.mark.parametrize(
    ("bytes1", "bytes2", "max_dist"),
    (
        (b"\x00" * 16 + b"\x00\x03" * 8 * 511, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 256 + b"\x00" * 16 + b"\x00\x03" * 8 * 255, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 511 + b"\x00" * 16, b"\x00" * 16, 1),
        (b"\xff" * 32 + b"\x11" * 32 * 1023, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 511 + b"\xff" * 32 + b"\x11" * 32 * 512, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 1023 + b"\xff" * 32, b"\xfb" * 32, 4 * 32),
        (b"\xcc" * 64 + b"\x01" * 64 * 16383, b"\xfb" * 64, 5 * 64),
        (
            b"\x01" * 64 * 8191 + b"\xcc" * 64 + b"\x01" * 64 * 8192,
            b"\xfb" * 64,
            5 * 64,
        ),
        (b"\x01" * 64 * 16383 + b"\xcc" * 64, b"\xfb" * 64, 5 * 64),
    ),
    ids=(
        "  512 elems,s=16,at 0",
        "  512 elems,s=16,mid",
        "  512 elems,s=16,end",
        " 1024 elems,s=32,at 0",
        " 1024 elems,s=32,mid",
        " 1024 elems,s=32,end",
        "16384 elems,s=64,at 0",
        "16384 elems,s=64,mid",
        "16384 elems,s=64,end",
    ),
)
def test_check_bytes_arrays_best_within_dist_bench(benchmark, bytes1, bytes2, max_dist):
    benchmark(check_bytes_arrays_best_within_dist, bytes1, bytes2, max_dist)


@pytest.mark.benchmark(
    group="hamming_distance_check_bytes_arrays_all_within_dist_bench"
)
@pytest.mark.parametrize(
    ("bytes1", "bytes2", "max_dist"),
    (
        (b"\x00" * 16 + b"\x00\x03" * 8 * 511, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 256 + b"\x00" * 16 + b"\x00\x03" * 8 * 255, b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 511 + b"\x00" * 16, b"\x00" * 16, 1),
        (b"\xff" * 32 + b"\x11" * 32 * 1023, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 511 + b"\xff" * 32 + b"\x11" * 32 * 512, b"\xfb" * 32, 4 * 32),
        (b"\x11" * 32 * 1023 + b"\xff" * 32, b"\xfb" * 32, 4 * 32),
        (b"\xcc" * 64 + b"\x01" * 64 * 16383, b"\xfb" * 64, 5 * 64),
        (
            b"\x01" * 64 * 8191 + b"\xcc" * 64 + b"\x01" * 64 * 8192,
            b"\xfb" * 64,
            5 * 64,
        ),
        (b"\x01" * 64 * 16383 + b"\xcc" * 64, b"\xfb" * 64, 5 * 64),
    ),
    ids=(
        "  512 elems,s=16,at 0",
        "  512 elems,s=16,mid",
        "  512 elems,s=16,end",
        " 1024 elems,s=32,at 0",
        " 1024 elems,s=32,mid",
        " 1024 elems,s=32,end",
        "16384 elems,s=64,at 0",
        "16384 elems,s=64,mid",
        "16384 elems,s=64,end",
    ),
)
def test_check_bytes_arrays_all_within_dist_bench(benchmark, bytes1, bytes2, max_dist):
    benchmark(check_bytes_arrays_all_within_dist, bytes1, bytes2, max_dist)


############################
# Buffer-protocol inputs
#
# Rust bindings should accept common bytes-like Python objects, not only bytes.
############################


def test_hamming_distance_bytes_bytearray():
    """bytearray inputs accepted via buffer protocol."""
    a = bytearray(b"\xff\x00")
    b = bytearray(b"\x00\xff")
    assert hamming_distance_bytes(a, b) == 16


def test_hamming_distance_bytes_memoryview():
    """memoryview inputs accepted via buffer protocol."""
    a = memoryview(b"\xff\x00")
    b = memoryview(b"\x00\xff")
    assert hamming_distance_bytes(a, b) == 16


def test_hamming_distance_bytes_noncontiguous_memoryview():
    a = memoryview(b"\xff\x00\xff\x00")[::2]
    with pytest.raises(ValueError, match="input must be contiguous"):
        hamming_distance_bytes(a, a)


def test_check_bytes_within_dist_bytearray():
    a = bytearray(b"\xff\x00")
    b = bytearray(b"\xfe\x00")
    assert check_bytes_within_dist(a, b, 2) is True
    assert check_bytes_within_dist(a, b, 0) is False


def test_check_bytes_within_dist_memoryview():
    a = memoryview(b"\xff\x00")
    b = memoryview(b"\xfe\x00")
    assert check_bytes_within_dist(a, b, 2) is True


def test_check_bytes_arrays_first_within_dist_bytearray():
    big = bytearray(b"\xaa\xbb\xcc\xff")
    small = bytearray(b"\xff")
    assert check_bytes_arrays_first_within_dist(big, small, 4) == 0
    assert check_bytes_arrays_first_within_dist(big, small, 0) == 3


def test_check_bytes_arrays_best_within_dist_memoryview():
    big = memoryview(b"\xaa\xfe\xff")
    small = memoryview(b"\xff")
    dist, idx = check_bytes_arrays_best_within_dist(big, small, 8)
    assert (dist, idx) == (0, 2)


def test_check_bytes_arrays_all_within_dist_bytearray():
    big = bytearray(b"\xaa\xfe\xff")
    small = bytearray(b"\xff")
    result = check_bytes_arrays_all_within_dist(big, small, 8)
    assert len(result) == 3
    assert result[2] == (0, 2)


try:
    # NumPy is optional for this package, so skip these coverage tests when the
    # local environment does not provide numpy arrays.
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_hamming_distance_bytes_numpy():
    """numpy uint8 arrays accepted via buffer protocol."""
    a = np.array([0xFF, 0x00], dtype=np.uint8)
    b = np.array([0x00, 0xFF], dtype=np.uint8)
    assert hamming_distance_bytes(a, b) == 16


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_hamming_distance_bytes_numpy_wide_dtype_rejected():
    a = np.array([0xFF, 0x00], dtype=np.uint32)
    with pytest.raises(ValueError, match="error occurred while parsing arguments"):
        hamming_distance_bytes(a, a)


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_check_bytes_within_dist_numpy():
    a = np.array([0xFF, 0x00], dtype=np.uint8)
    b = np.array([0xFE, 0x00], dtype=np.uint8)
    assert check_bytes_within_dist(a, b, 2) is True


############################
# SIMD path for check_hexstrings_within_dist
#
# Lengths at or above 64 hex characters are intended to exercise vectorized
# threshold checks, including exact-boundary and over-boundary behavior.
############################


def test_check_hexstrings_within_dist_simd_equal():
    """Equal long strings → True (SIMD path, len == 64)."""
    s = "a" * 64
    assert check_hexstrings_within_dist(s, s, 0) is True


def test_check_hexstrings_within_dist_simd_at_boundary():
    """Strings differing in exactly max_dist bits → True."""
    # 'f' vs '0' has hamming distance 4 per hex char
    # 5 differing chars → distance 20
    a = "f" * 5 + "0" * 59
    b = "0" * 64
    assert check_hexstrings_within_dist(a, b, 20) is True


def test_check_hexstrings_within_dist_simd_over_boundary():
    """Strings differing in max_dist + 1 bits → False."""
    a = "f" * 5 + "0" * 59
    b = "0" * 64
    # distance is 20, max_dist 19 → False
    assert check_hexstrings_within_dist(a, b, 19) is False


def test_check_hexstrings_within_dist_simd_long():
    """Very long strings (10000 chars, well above 64 threshold)."""
    a = "f" * 10000
    b = "0" * 10000
    # distance = 40000
    assert check_hexstrings_within_dist(a, b, 40000) is True
    assert check_hexstrings_within_dist(a, b, 39999) is False


def test_check_hexstrings_within_dist_simd_invalid_char():
    """Invalid hex char in SIMD-length string raises ValueError."""
    a = "f" * 63 + "g"
    b = "0" * 64
    # max_dist must be high enough that SIMD processes all chunks
    # (including the one with invalid 'g') but < 4*len to avoid shortcut
    with pytest.raises(ValueError, match="hex string contains invalid char"):
        check_hexstrings_within_dist(a, b, 255)


############################
# Algorithm selection behavior
#
# set_algo is intentionally non-throwing: an empty string means success, while
# invalid or unavailable backends return a diagnostic string.
############################


def test_set_algo_valid_returns_empty():
    """set_algo returns empty string for valid algorithms."""
    for algo in ("classic", "native"):
        assert set_algo(algo) == ""


def test_set_algo_invalid_returns_nonempty():
    """set_algo returns non-empty error message for unknown algorithm."""
    result = set_algo("bogus_algo")
    assert len(result) > 0


def test_set_algo_roundtrip():
    """Verify set_algo + hamming_distance_string produces correct results."""
    set_algo("classic")
    assert hamming_distance_string("deadbeef", "00000000") == 24
    set_algo("native")
    assert hamming_distance_string("deadbeef", "00000000") == 24


def test_functions_are_bound_to_extension_module():
    """Module functions must not be METH_STATIC (see `pass_module` in python.rs).

    PyO3 marks functions without ``pass_module`` METH_STATIC (``__self__`` is
    None), which makes CPython 3.14 deopt every specialized call to the generic
    vectorcall path.
    """
    import hexhamming.hexhamming as ext

    names = [
        name
        for name in dir(ext)
        if callable(getattr(ext, name))
        and hasattr(getattr(ext, name), "__self__")
        and not name.startswith("_")
    ]
    assert "hamming_distance_string" in names
    for name in names:
        assert getattr(ext, name).__self__ is ext, name


RAW_FASTCALL_FUNCTIONS = {
    "hamming_distance_string": ("a", "b"),
    "hamming_distance_bytes": ("a", "b"),
    "check_hexstrings_within_dist": ("a", "b", "max_dist"),
    "check_bytes_within_dist": ("a", "b", "max_dist"),
    "check_bytes_arrays_within_dist": (
        "array_of_elems",
        "elem_to_compare",
        "max_dist",
    ),
    "check_bytes_arrays_first_within_dist": (
        "array_of_elems",
        "elem_to_compare",
        "max_dist",
    ),
    "check_bytes_arrays_best_within_dist": (
        "array_of_elems",
        "elem_to_compare",
        "max_dist",
    ),
    "check_bytes_arrays_all_within_dist": (
        "array_of_elems",
        "elem_to_compare",
        "max_dist",
    ),
    "hamming_distances_bytes": ("a", "b", "element_size"),
}


def test_raw_fastcall_metadata_matches_pyo3_shape():
    """Raw C-API replacements preserve PyO3-visible signatures and binding."""
    import hexhamming.hexhamming as ext

    for name, params in RAW_FASTCALL_FUNCTIONS.items():
        func = getattr(ext, name)
        assert func.__self__ is ext
        assert inspect.signature(func) == inspect.Signature(
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in params
        )
        assert func.__text_signature__ == f"($module, {', '.join(params)})"
        assert func.__doc__


def test_module_functions_pickle_by_reference():
    """Every function keeps a str ``__module__`` naming the extension module, so
    pickle (and therefore multiprocessing) can resolve it by reference."""
    import pickle

    import hexhamming.hexhamming as ext

    names = [
        name
        for name in dir(ext)
        if callable(getattr(ext, name)) and not name.startswith("_")
    ]
    assert set(RAW_FASTCALL_FUNCTIONS) <= set(names)
    for name in names:
        func = getattr(ext, name)
        if inspect.isbuiltin(func):
            assert func.__module__ == ext.__name__, name
        assert pickle.loads(pickle.dumps(func)) is func, name


@pytest.mark.parametrize(
    "func,args,kwargs,expected",
    (
        (hamming_distance_string, (), {"a": "ab", "b": "cd"}, 4),
        (hamming_distance_string, ("ab",), {"b": "cd"}, 4),
        (hamming_distance_bytes, (), {"a": b"\x00", "b": b"\xff"}, 8),
        (
            check_hexstrings_within_dist,
            (),
            {"a": "ffff", "b": "fffe", "max_dist": 1},
            True,
        ),
        (
            check_bytes_within_dist,
            (),
            {"a": b"\x00", "b": b"\x01", "max_dist": 1},
            True,
        ),
        (
            check_bytes_arrays_within_dist,
            (),
            {"array_of_elems": b"\x00\xff", "elem_to_compare": b"\xff", "max_dist": 0},
            1,
        ),
        (
            check_bytes_arrays_first_within_dist,
            (),
            {"array_of_elems": b"\x00\xff", "elem_to_compare": b"\xff", "max_dist": 0},
            1,
        ),
        (
            check_bytes_arrays_best_within_dist,
            (),
            {"array_of_elems": b"\x00\xff", "elem_to_compare": b"\xff", "max_dist": 8},
            (0, 1),
        ),
        (
            check_bytes_arrays_all_within_dist,
            (),
            {"array_of_elems": b"\x00\xff", "elem_to_compare": b"\xff", "max_dist": 8},
            [(8, 0), (0, 1)],
        ),
        (
            hamming_distances_bytes,
            (),
            {"a": b"\x00\xff", "b": b"\xff\xff", "element_size": 1},
            [8, 0],
        ),
    ),
)
def test_raw_fastcall_keyword_argument_parity(func, args, kwargs, expected):
    assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    "func,args,kwargs,exc_type,msg",
    (
        (
            hamming_distance_string,
            (),
            {},
            TypeError,
            "missing 2 required positional arguments: 'a' and 'b'",
        ),
        (
            hamming_distance_string,
            ("ab", "cd", "ef"),
            {},
            TypeError,
            "takes 2 positional arguments but 3 were given",
        ),
        (
            hamming_distance_string,
            ("ab", "cd"),
            {"a": "ab"},
            TypeError,
            "got multiple values for argument 'a'",
        ),
        (
            hamming_distance_string,
            ("ab",),
            {"x": "cd"},
            TypeError,
            "got an unexpected keyword argument 'x'",
        ),
        (
            check_hexstrings_within_dist,
            ("ab",),
            {},
            TypeError,
            "missing 2 required positional arguments: 'b' and 'max_dist'",
        ),
        (
            check_bytes_arrays_first_within_dist,
            ("ab",),
            {"b": "cd"},
            TypeError,
            "got an unexpected keyword argument 'b'",
        ),
    ),
)
def test_raw_fastcall_argument_error_parity(func, args, kwargs, exc_type, msg):
    with pytest.raises(exc_type) as excinfo:
        func(*args, **kwargs)
    assert msg in str(excinfo.value)


def test_raw_fastcall_type_buffer_and_unicode_parity():
    class StrSub(str):
        pass

    class BytesSub(bytes):
        pass

    assert hamming_distance_string(StrSub("ab"), StrSub("cd")) == 4
    assert hamming_distance_bytes(BytesSub(b"\x00"), BytesSub(b"\xff")) == 8
    with pytest.raises(UnicodeEncodeError):
        hamming_distance_string("\ud800", "0")
    with pytest.raises(ValueError, match="input must be contiguous"):
        hamming_distance_bytes(memoryview(bytearray(b"abcd"))[::2], b"ab")


# ---------------------------------------------------------------------------
# check_hexstrings_within_dist: SIMD early-exit correctness + perf
#
# Randomized cases cross-check the threshold API against the exact distance API,
# while the timing test guards the tight-threshold early-exit path.
# ---------------------------------------------------------------------------


def test_check_hexstrings_within_dist_long_random_correctness():
    """1024-char random strings: correctness of the SIMD early-exit path."""
    import secrets

    for _ in range(20):
        a = secrets.token_hex(512)  # 1024 hex chars
        b = secrets.token_hex(512)
        full_dist = hamming_distance_string(a, b)
        # Within: max_dist >= actual distance
        assert check_hexstrings_within_dist(a, b, full_dist) is True
        assert check_hexstrings_within_dist(a, b, full_dist + 100) is True
        # Not within: max_dist < actual distance (when distance > 0)
        if full_dist > 0:
            assert check_hexstrings_within_dist(a, b, full_dist - 1) is False
        # Tight threshold
        assert check_hexstrings_within_dist(a, b, 0) is (full_dist == 0)


@pytest.mark.skipif(
    python_implementation() == "PyPy",
    reason="PyPy's C-extension calls alone take about as long as this threshold",
)
def test_check_hexstrings_within_dist_long_random_fast():
    """1024-char random strings with tight max_dist: must be fast (<0.15 us)."""
    import secrets
    import time

    a = secrets.token_hex(512)
    c = secrets.token_hex(512)
    n = 10000
    # Warm up
    for _ in range(5):
        check_hexstrings_within_dist(a, c, 100)
    # Best of several rounds: shared CI runners are noisy, and a regression
    # slows every round while interference only slows some.
    rounds = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(n):
            check_hexstrings_within_dist(a, c, 100)
        rounds.append((time.perf_counter() - t0) / n * 1e6)
    elapsed_us = min(rounds)
    # Target: < 0.15 us (baseline was 0.095 us, regressed to ~0.19 us)
    assert elapsed_us < 0.15, (
        f"random+tight took {elapsed_us:.3f} us, expected < 0.15 us"
    )
