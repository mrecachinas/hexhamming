import random

import pytest
from hexhamming import (
    check_bytes_arrays_all_within_dist,
    check_bytes_arrays_best_within_dist,
    check_bytes_arrays_first_within_dist,
    hamming_distance_bytes,
    hamming_distance_string,
)


def random_bytes(size):
    rng = random.Random(0xC0FFEE + size)
    return bytes(rng.randrange(256) for _ in range(size))


def stdlib_bytes_distance(a, b):
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).bit_count()


def stdlib_hex_distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


@pytest.mark.benchmark(group="random_bytes")
@pytest.mark.parametrize("size", (16, 64, 1024))
def test_hamming_distance_bytes_random_bench(benchmark, size):
    a = random_bytes(size)
    b = random_bytes(size + 1)[:size]
    benchmark(hamming_distance_bytes, a, b)


@pytest.mark.benchmark(group="random_bytes_stdlib")
@pytest.mark.parametrize("size", (16, 64, 1024))
def test_stdlib_bytes_distance_random_bench(benchmark, size):
    a = random_bytes(size)
    b = random_bytes(size + 1)[:size]
    benchmark(stdlib_bytes_distance, a, b)


@pytest.mark.benchmark(group="random_hex")
@pytest.mark.parametrize("size", (16, 64, 1024))
def test_hamming_distance_string_random_bench(benchmark, size):
    a = random_bytes(size // 2).hex()
    b = random_bytes(size // 2 + 1).hex()[:size]
    benchmark(hamming_distance_string, a, b)


@pytest.mark.benchmark(group="random_hex_stdlib")
@pytest.mark.parametrize("size", (16, 64, 1024))
def test_stdlib_hex_distance_random_bench(benchmark, size):
    a = random_bytes(size // 2).hex()
    b = random_bytes(size // 2 + 1).hex()[:size]
    benchmark(stdlib_hex_distance, a, b)


@pytest.mark.benchmark(group="gil_boundary")
@pytest.mark.parametrize("size", (16376, 16384, 16392))
def test_hamming_distance_bytes_gil_boundary_bench(benchmark, size):
    a = random_bytes(size)
    b = random_bytes(size + 1)[:size]
    benchmark(hamming_distance_bytes, a, b)


def fixed_width_array_case(width, scenario):
    count = 1024
    rng = random.Random(0x51_0000 + width)
    needle = bytes(rng.randrange(256) for _ in range(width))
    big = bytearray(rng.randrange(256) for _ in range(count * width))
    if scenario == "random_no_match":
        max_dist = 0
        index = None
    elif scenario == "exact_early":
        max_dist, index = 0, 0
    elif scenario == "exact_mid":
        max_dist, index = 0, count // 2
    elif scenario == "exact_late":
        max_dist, index = 0, count - 1
    elif scenario == "threshold_d_minus_1":
        max_dist, index = 3, count // 2
    elif scenario == "threshold_d":
        max_dist, index = 4, count // 2
    elif scenario == "threshold_d_plus_1":
        max_dist, index = 5, count // 2
    else:
        raise AssertionError(f"unknown scenario: {scenario}")

    if index is not None:
        start = index * width
        big[start : start + width] = needle
        if scenario.startswith("threshold_"):
            big[start] ^= 0x0F
    return bytes(big), needle, max_dist


@pytest.mark.benchmark(group="array_scan_matrix")
@pytest.mark.parametrize("width", (16, 32))
@pytest.mark.parametrize(
    "scenario",
    (
        "random_no_match",
        "exact_early",
        "exact_mid",
        "exact_late",
        "threshold_d_minus_1",
        "threshold_d",
        "threshold_d_plus_1",
    ),
)
@pytest.mark.parametrize(
    "operation",
    (
        check_bytes_arrays_first_within_dist,
        check_bytes_arrays_best_within_dist,
        check_bytes_arrays_all_within_dist,
    ),
    ids=("first", "best", "all"),
)
def test_fixed_width_array_scan_matrix_bench(benchmark, width, scenario, operation):
    big, needle, max_dist = fixed_width_array_case(width, scenario)
    benchmark(operation, big, needle, max_dist)
