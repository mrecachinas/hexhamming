import random

import pytest

from hexhamming import hamming_distance_bytes, hamming_distance_string


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
