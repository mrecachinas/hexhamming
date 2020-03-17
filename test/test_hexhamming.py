#!/usr/bin/env python
import pytest
from hexhamming import check_hexstrings_within_dist, hamming_distance

############################
# hamming_distance tests
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
)
def test_hamming_distance(hex1, hex2, expected):
    assert hamming_distance(hex1, hex2) == expected


@pytest.mark.parametrize(
    "hex1,hex2,exception,msg",
    (
        ("abc", 3, ValueError, "error occurred while parsing arguments"),
        ("abc", "a", ValueError, "strings are NOT the same length"),
        ("lol", "foo", ValueError, "hex string contains invalid char"),
        ("000abcdef", "011abcdgf", ValueError, "hex string contains invalid char"),
    ),
)
def test_hamming_distance_errors(hex1, hex2, exception, msg):
    with pytest.raises(exception) as excinfo:
        _ = hamming_distance(hex1, hex2)
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
    assert check_hexstrings_within_dist(hex1, hex2, max_dist) == expected


@pytest.mark.parametrize(
    "hex1,hex2,max_dist,exception,msg",
    (
        ("000abcdef", "011abcdef", None, ValueError, "error occurred while parsing arguments"),
        ("000abcdef", "011abcdef", "HELLO", ValueError, "error occurred while parsing arguments"),
        ("000abcdef", "011abcdef", -1, ValueError, "`max_dist` must be >0"),
        ("000abcdef", "011abcdzz", 3, ValueError, "hex string contains invalid char"),
        ("000abcdef", "011abcdgf", 3, ValueError, "hex string contains invalid char"),
        ("1f0abcdef", 3, 3, ValueError, "error occurred while parsing arguments"),
        ("011abcdef", "00", 3, ValueError, "strings are NOT the same length"),
    ),
)
def test_check_hexstrings_within_dist(hex1, hex2, max_dist, exception, msg):
    with pytest.raises(exception) as excinfo:
        _ = check_hexstrings_within_dist(hex1, hex2, max_dist)
    assert msg in str(excinfo.value)


def test_hamming_distance_bench_short(benchmark):
    benchmark(hamming_distance, "ABC", "DEF")


def test_hamming_distance_bench_short_same(benchmark):
    benchmark(hamming_distance, "BBB", "BBB")


def test_hamming_distance_bench_long_same(benchmark):
    benchmark(hamming_distance, "B" * 1000, "B" * 1000)


def test_hamming_distance_bench_long(benchmark):
    benchmark(hamming_distance, "F" * 1000, "0" * 1000)


def test_hamming_distance_bench_256(benchmark):
    benchmark(hamming_distance, "F" * 64, "0" * 64)


def test_check_hexstrings_within_dist_bench(benchmark):
    benchmark(check_hexstrings_within_dist, "F" * 1000, "0" * 1000, 20)