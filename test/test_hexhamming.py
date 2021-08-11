#!/usr/bin/env python
import pytest
from hexhamming import check_hexstrings_within_dist, hamming_distance_string, hamming_distance_bytes

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
    assert hamming_distance_string(hex1, hex2) == expected


@pytest.mark.parametrize(
    "hex1,hex2,expected",
    (
        (b"\xab\x0c", b"\xab\x0c", 0),
        (b"\x00", b"\x01", 1),
        (b"\xAB\xCD\xEF", b"\x00\x00\x01", 16),
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
    assert hamming_distance_bytes(hex1, hex2) == expected


@pytest.mark.parametrize(
    "hex1,hex2,exception,msg",
    (
        ("abc", 3, ValueError, "error occurred while parsing arguments"),
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
    assert check_hexstrings_within_dist(hex1, hex2, max_dist) == expected


@pytest.mark.parametrize(
    "hex1,hex2,max_dist,exception,msg",
    (
        (
            "000abcdef",
            "011abcdef",
            None,
            ValueError,
            "error occurred while parsing arguments",
        ),
        (
            "000abcdef",
            "011abcdef",
            "HELLO",
            ValueError,
            "error occurred while parsing arguments",
        ),
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
        (b"\xAB\x0C", b"\xDE\x0F"),
        (b"\xBB\x0B", b"\xBB\x0B"),
        (b"\xBB" * 500, b"\xBB" * 500),
        (b"\xFF" * 500, b"\x00" * 500),
        (b"\xBB" * 512, b"\xBB" * 512),
        (b"\xFF" * 512, b"\x00" * 512),
        (b"\xFF" * 32, b"\x00" * 32),
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


def test_check_hexstrings_within_dist_bench(benchmark):
    benchmark(check_hexstrings_within_dist, "F" * 1000, "0" * 1000, 20)
