#!/usr/bin/env python
import pytest
from hamming import (
    hamming_distance,
    hamming_distance_lookup,
    fast_hamming_distance,
    str_to_hex,
    str_to_hex256,
    xor_lists,
)


############################
# str_to_hex tests
############################


def test_str_to_hex():
    s = "0000"
    expected = map(int, bin(int(s, 16))[2:])
    assert str_to_hex(s) == expected


def test_str_to_hex():
    s = "deadbeef"
    expected = map(int, bin(int(s, 16))[2:])
    assert str_to_hex(s) == expected


def test_str_to_hex_out_of_range():
    s = "abcdefg"
    with pytest.raises(ValueError):
        str_to_hex(s)


def test_str_to_hex_invalid():
    s = 3
    with pytest.raises(ValueError):
        str_to_hex(s)


############################
# str_to_hex256 tests
############################

def test_str_to_hex256():
    s = "deadbeef"
    expected = [50, 54, 0, 0, 0, 0, 0, 0]
    assert str_to_hex256(s) == expected


def test_str_to_hex256_invalid():
    s = 3
    with pytest.raises(ValueError):
        str_to_hex256(s)


def test_str_to_hex256_out_of_range():
    s = "abcdefg"
    with pytest.raises(ValueError):
        str_to_hex256(s)


############################
# xor_lists tests
############################

def test_xor_lists_all_different():
    a = [1] * 64
    b = [0] * 64
    assert xor_lists(a, b) == len(a)


def test_xor_lists_all_same():
    a = [1] * 64
    b = [1] * 64
    assert xor_lists(a, b) == 0


def test_xor_lists_invalid():
    a = [1, 2, 3]
    b = [0, 0, 0]
    with pytest.raises(ValueError):
        xor_lists(a, b)


def test_xor_lists_bad_type():
    a = 3
    b = 4
    with pytest.raises(ValueError):
        xor_lists(a, b)


def test_xor_lists_different_lengths():
    a = [1]
    b = [0, 1]
    with pytest.raises(ValueError):
        xor_lists(a, b)

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


###############################
# hamming_distance_lookup tests
###############################

@pytest.mark.benchmark(group="empty string")
def test_hamming_distance_lookup_empty(benchmark):
    a = ''
    b = ''
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="12-bit different")
def test_hamming_distance_lookup_different(benchmark):
    a = 'abc'
    b = 'abd'
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 1


@pytest.mark.benchmark(group="12-bit same")
def test_hamming_distance_lookup_same(benchmark):
    a = 'abc'
    b = 'abc'
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="256-bit different")
def test_hamming_distance_lookup_max_len_different(benchmark):
    a = "f" * 64
    b = "0" * 64
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 256


@pytest.mark.benchmark(group="256-bit same 0")
def test_hamming_distance_lookup_max_len_same_0(benchmark):
    a = "0" * 64
    b = "0" * 64
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="256-bit same f")
def test_hamming_distance_lookup_max_len_same_f(benchmark):
    a = "f" * 64
    b = "f" * 64
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 0


@pytest.mark.benchmark(group="4000-bit different")
def test_hamming_distance_lookup_stress_test_len_different(benchmark):
    a = "f" * 1000
    b = "0" * 1000
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == len(a) * 4


@pytest.mark.benchmark(group="4000-bit same")
def test_hamming_distance_lookup_stress_test_len_same(benchmark):
    a = "f" * 1000
    b = "f" * 1000
    result = benchmark(hamming_distance_lookup, *(a, b))
    assert result == 0

############################
# fast_hamming_distance tests
############################

@pytest.mark.benchmark(group="12-bit same")
def test_fast_hamming_distance_same(benchmark):
    result = benchmark(fast_hamming_distance, *("abc", "abc"))
    assert result == 0


@pytest.mark.benchmark(group="empty string")
def test_fast_hamming_distance_empty(benchmark):
    result = benchmark(fast_hamming_distance, *("", ""))
    assert result == 0


@pytest.mark.benchmark(group="12-bit different")
def test_fast_hamming_distance_different(benchmark):
    result = benchmark(fast_hamming_distance, *("000", "001"))
    assert result == 1


def test_fast_hamming_distance_different_len():
    with pytest.raises(ValueError):
        fast_hamming_distance("abc", "a") == 0


def test_fast_hamming_distance_invalid():
    with pytest.raises(ValueError):
        fast_hamming_distance("abc", 3) == 0


def test_fast_hamming_distance_invalid():
    with pytest.raises(ValueError):
        fast_hamming_distance("lol", "foo") == 0


# @pytest.mark.benchmark(group="256-bit different")
# def test_fast_hamming_distance_max_len_different(benchmark):
#     a = "f" * 64
#     b = "0" * 64
#     result = benchmark(fast_hamming_distance, *(a, b))
#     assert result == 256


# def test_fast_hamming_distance_max_len_same(benchmark):
#     a = "f" * 64
#     b = "f" * 64
#     result = benchmark(fast_hamming_distance, *(a, b))
#     assert result == 0
