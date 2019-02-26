#!/usr/bin/env python
import pytest
from hexhamming import hamming_distance

############################
# hamming_distance tests
############################

@pytest.mark.benchmark(group="12-bit same")
def test_hamming_distance_same(benchmark):
    result = benchmark(hamming_distance, *("abc", "abc"))
    assert result == 0


@pytest.mark.benchmark(group="12-bit different")
def test_hamming_distance_different(benchmark):
    result = benchmark(hamming_distance, *("000", "001"))
    assert result == 1


@pytest.mark.benchmark(group="empty string")
def test_hamming_distance_empty(benchmark):
    result = benchmark(hamming_distance, *("", ""))
    assert result == 0


def test_hamming_distance_invalid():
    with pytest.raises(ValueError):
        hamming_distance("abc", 3)


def test_hamming_distance_different_len():
    with pytest.raises(ValueError):
        hamming_distance("abc", "a")


def test_hamming_distance_invalid():
    with pytest.raises(ValueError):
        hamming_distance("lol", "foo")


@pytest.mark.benchmark(group="256-bit different")
def test_hamming_distance_max_len_different(benchmark):
    a = "f" * 64
    b = "0" * 64
    result = benchmark(hamming_distance, *(a, b))
    assert result == len(a) * 4


@pytest.mark.benchmark(group="256-bit same f")
def test_hamming_distance_max_len_same_f(benchmark):
    a = "f" * 64
    b = "f" * 64
    result = benchmark(hamming_distance, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="256-bit same 0")
def test_hamming_distance_max_len_same_0(benchmark):
    a = "f" * 64
    b = "f" * 64
    result = benchmark(hamming_distance, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="4000-bit different")
def test_hamming_distance_stress_test_len_different(benchmark):
    a = "f" * 1000
    b = "0" * 1000
    result = benchmark(hamming_distance, *(a, b))
    assert result == len(a) * 4


@pytest.mark.benchmark(group="4000-bit same")
def test_hamming_distance_stress_test_len_same(benchmark):
    a = "f" * 1000
    b = "f" * 1000
    result = benchmark(hamming_distance, *(a, b))
    assert result == 0
