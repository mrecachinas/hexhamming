#!/usr/bin/env python
from platform import machine
import pytest
from hexhamming import check_hexstrings_within_dist, hamming_distance_string, \
                        hamming_distance_bytes, check_bytes_arrays_within_dist, set_algo

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
    assert expected == hamming_distance_string(hex1, hex2)
    assert len(set_algo('classic')) == 0        # we have only 2 algorithms for strings currently.
    assert expected == hamming_distance_string(hex1, hex2)


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
    algorithm_list = ['extra', 'native', 'classic']
    if machine().lower().startswith('x86'):
        algorithm_list.append('sse41')
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f'Warning: Skipping {algorithm}, reason: {result}')
            continue
        assert expected == hamming_distance_bytes(hex1, hex2)


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
    algorithm_list = ['extra', 'native', 'classic']
    if machine().lower().startswith('x86'):
        algorithm_list.append('sse41')
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f'Warning: Skipping {algorithm}, reason: {result}')
            continue
        assert expected == check_hexstrings_within_dist(hex1, hex2, max_dist)


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


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,exception,msg",
    (
        (
            b"\x00" * 16,
            b"\x00" * 16,
            None,
            ValueError,
            "error occurred while parsing arguments",
        ),
        (
            b"\x00" * 16,
            b"\x00" * 16,
            "HELLO",
            ValueError,
            "error occurred while parsing arguments",
        ),
        (b"\x00" * 32, b"\x00" * 16, -1, ValueError, "`max_dist` must be >=0"),
        (b"\x00" * 31, b"\x00" * 16, 3, ValueError, "`array_of_elems` size must be multiplier of `elem_to_compare`"),
        (b"\x00" * 32, b"", 3, ValueError, "`elem_to_compare` size must be >0"),
    ),
)
def test_check_bytes_arrays_within_dist_invalid_values(bytes1, bytes2, max_dist, exception, msg):
    with pytest.raises(exception) as excinfo:
        _ = check_bytes_arrays_within_dist(bytes1, bytes2, max_dist)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "bytes1,bytes2,max_dist,expected",
    (
        (
            b"\x00" * 16,
            b"\xFF" * 16,
            50, -1,
        ),
        (
            b"\x00" * 16,
            b"\x00" * 15 + b"\x0F" * 1,
            4, 0,
        ),
        (
            b"\xFF" * 16 * 8 + b"\x0F" * 16,
            b"\x00" * 2 + b"\x0F" * 14,
            8, 8,
        ),
        (
            b"\xF0" * 64 + b"\x0A" * 64,
            b"\x0F" * 64,
            3 * 64, 1,
        )
    ),
)
def test_check_bytes_arrays_within_dist_calculation(bytes1, bytes2, max_dist, expected):
    algorithm_list = ['extra', 'native', 'classic']
    if machine().lower().startswith('x86'):
        algorithm_list.append('sse41')
    for algorithm in algorithm_list:
        result = set_algo(algorithm)
        if len(result) > 0:
            print(f'Warning: Skipping {algorithm}, reason: {result}')
            continue
        assert expected == check_bytes_arrays_within_dist(bytes1, bytes2, max_dist)


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


@pytest.mark.benchmark(group="hamming_distance_bytes_arrays_within_dist")
@pytest.mark.parametrize(
    ("bytes1", "bytes2", "max_dist"),
    (
        (b"\x00" * 16 + b"\x00\x03" * 8 * 511,
         b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 256 + b"\x00" * 16 + b"\x00\x03" * 8 * 255,
         b"\x00" * 16, 1),
        (b"\x00\x03" * 8 * 511 + b"\x00" * 16,
         b"\x00" * 16, 1),
        (b"\xFF" * 32 + b"\x11" * 32 * 1023,
         b"\xFB" * 32, 4*32),
        (b"\x11" * 32 * 511 + b"\xFF" * 32 + b"\x11" * 32 * 512,
         b"\xFB" * 32, 4*32),
        (b"\x11" * 32 * 1023 + b"\xFF" * 32,
         b"\xFB" * 32, 4*32),
        (b"\xCC" * 64 + b"\x01" * 64 * 16383,
         b"\xFB" * 64, 5*64),
        (b"\x01" * 64 * 8191 + b"\xCC" * 64 + b"\x01" * 64 * 8192,
         b"\xFB" * 64, 5*64),
        (b"\x01" * 64 * 16383 + b"\xCC" * 64,
         b"\xFB" * 64, 5*64),
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
def test_check_bytes_arrays_within_dist_bench(benchmark, bytes1, bytes2, max_dist):
    benchmark(check_bytes_arrays_within_dist, bytes1, bytes2, max_dist)


def test_check_bytes_arrays_within_dist_gil():
    import threading
    import time

    def run_check():
        big_array = b"\x00" * 16 * 1000000
        small_array = b"\x00" * 16
        max_dist = 0
        result = check_bytes_arrays_within_dist(big_array, small_array, max_dist)
        assert result == 0

    def run_other_task():
        for _ in range(5):
            print("Running other task")
            time.sleep(0.1)

    check_thread = threading.Thread(target=run_check)
    other_task_thread = threading.Thread(target=run_other_task)

    check_thread.start()
    other_task_thread.start()

    check_thread.join()
    other_task_thread.join()
