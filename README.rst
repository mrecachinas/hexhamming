``hexhamming``
====================

|Pip|_ |Prs|_ |Github|_

.. |Pip| image:: https://badge.fury.io/py/hexhamming.svg
.. _Pip: https://badge.fury.io/py/hexhamming

.. |Prs| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
.. _Prs: .github/CONTRIBUTING.md#pull-requests

.. |Github| image:: https://github.com/mrecachinas/hexhamming/workflows/build/badge.svg
.. _Github: https://github.com/mrecachinas/hexhamming/actions

What does it do?
----------------

This module performs a fast bitwise hamming distance of two hexadecimal strings.

This looks like::

    DEADBEEF = 11011110101011011011111011101111
    00000000 = 00000000000000000000000000000000
    XOR      = 11011110101011011011111011101111
    Hamming  = number of ones in DEADBEEF ^ 00000000 = 24

This essentially amounts to

::

    >>> import gmpy
    >>> gmpy.popcount(0xdeadbeef ^ 0x00000000)
    24

except with Python strings, so

::

    >>> import gmpy
    >>> gmpy.popcount(int("deadbeef", 16) ^ int("00000000", 16))
    24

A few assumptions are made and enforced:

* this is a valid hexadecimal string (i.e., ``[a-fA-F0-9]+``)
* the strings are the same length
* the strings do not begin with ``"0x"``

Why yet another Hamming distance library?
-----------------------------------------

There are a lot of fantastic (python) libraries that offer methods to calculate
various edit distances, including Hamming distances: Distance, textdistance,
scipy, jellyfish, etc.

In this case, I needed a hamming distance library that worked on hexadecimal
strings (i.e., a Python ``str``) and performed blazingly fast.
Furthermore, I often did not care about hex strings greater than 256 bits.
That length constraint is different vs all the other libraries and enabled me
to explore vectorization techniques via ``SSE/AVX`` and ``NEON`` intrinsics.

Lastly, I wanted to minimize dependencies, meaning you do not need to install
``numpy``, ``gmpy``, ``cython``, ``pypy``, ``pythran``, etc.

As of v3.0.0, ``hexhamming`` is written in Rust using `PyO3 <https://pyo3.rs>`_
and `maturin <https://www.maturin.rs>`_, providing memory safety, GIL release
during computation, and free-threaded Python support while maintaining the same
SIMD-accelerated performance (SSE4.1, AVX2, NEON).

Installation
-------------

To install, ensure you have Python 3.10+. Run

::

    pip install hexhamming

or to install from source (requires Rust toolchain)

::

    git clone https://github.com/mrecachinas/hexhamming
    cd hexhamming
    pip install .

If you want to contribute to hexhamming, you should install the dev
dependencies

::

    pip install -r requirements-dev.txt

and make sure the tests pass with

::

    python -m pytest -vls .

Example
-------

Using ``hexhamming`` is as simple as

::

    >>> from hexhamming import hamming_distance_string
    >>> hamming_distance_string("deadbeef", "00000000")
    24

**New in v2.0.0** : ``hexhamming`` now supports ``byte``s via ``hamming_distance_bytes``.
You use it in the exact same way as before, except you pass in a byte string.

::

    >>> from hexhamming import hamming_distance_bytes
    >>> hamming_distance_bytes(b"\xde\xad\xbe\xef", b"\x00\x00\x00\x00")
    24


We also provide a method for a quick boolean check of whether two hexadecimal strings
are within a given Hamming distance.

::

    >>> from hexhamming import check_hexstrings_within_dist
    >>> check_hexstrings_within_dist("ffff", "fffe", 2)
    True
    >>> check_hexstrings_within_dist("ffff", "0000", 2)
    False

Similarly, ``hexhamming`` supports a quick byte array check via ``check_bytes_within_dist``, which has
a similar API as ``check_hexstrings_within_dist``, except it expects a bytes array.

The API described above is targeted at comparing two individual records and calculating their hamming distance quickly.
For many applications the goal is to compare a given record to an array of other records and to find out if there
are elements in the array that are within a given hamming distance of the search record. To support these application
cases ``hexhamming`` has a set of array APIs. Given that these operations are often speed critical and require preparing data
anyway, they are only available for bytes strings, not for hex strings.

They all have the same signature, they take two bytes arrays and the ``max_dist`` to consider. The difference is, that the first
bytes string should be a concatenation of a number of records to compare to, i.e. the length needs to be a multiple of the length
of the second bytes string.

There are three functions that return different results, depending on what is needed by the application.

``check_bytes_arrays_first_within_dist`` returns the index of the first element that has a hamming distance less than ``max_dist``.

::

    >>> from hexhamming import check_bytes_arrays_first_within_dist
    >>> check_bytes_arrays_first_within_dist(b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff", b"\xff\xff", 4)
    1


``check_bytes_arrays_best_within_dist`` returns a tuple with the distance and the index of the element that has the lowest hamming
distance less than ``max_dist``, or ``(-1,-1)`` if none do.

::

    >>> from hexhamming import check_bytes_arrays_best_within_dist
    >>> check_bytes_arrays_best_within_dist(b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff", b"\xff\xff", 4)
    (0, 5)

    >>> check_bytes_arrays_best_within_dist(b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff", b"\xef\xfe", 4)
    (2, 4)


``check_bytes_arrays_all_within_dist`` returns a list of tuples with the distance and the index of the element that have a hamming
distance less than ``max_dist``, or ``[]`` if none do.

::

    >>> from hexhamming import check_bytes_arrays_all_within_dist
    >>> check_bytes_arrays_all_within_dist(b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff", b"\xff\xff", 4)
    [(4, 1), (4, 3), (4, 4), (0, 5)]


Tip: When you're assembling the long array of records to compare against, don't concatenate the different ``bytes`` together. As they're
immutable that is a very slow operation. Use a ``bytearray`` instead, and cast it to ``bytes`` at the end. See https://www.guyrutenberg.com/2020/04/04/fast-bytes-concatenation-in-python/ for more info and tests.

Benchmark
---------

All benchmarks were run on an Apple M4 Max (ARM64, 16 logical cores, 64 GiB)
with hexhamming v3.0.0, ``rustc`` 1.96.1, and Python 3.14.6. Values are the
median of the means from three independent runs.

Raw Rust (no Python overhead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These numbers show the pure computation time using Rust's ``criterion`` benchmarks
(``cargo bench --no-default-features``), with no Python/PyO3 overhead.

================================================  ===========
Name                                              Mean (ns)
================================================  ===========
hex_string (NEON) [16 chars]                           2.4
hex_string (NEON) [64 chars]                           8.3
hex_string (NEON) [128 chars]                         16.2
hex_string (NEON) [254 chars]                         30.2
bytes (native) [8 bytes]                               1.7
bytes (native) [32 bytes]                              2.4
bytes (native) [64 bytes]                              3.2
bytes (native) [127 bytes]                             8.4
bytes_within_dist [127 bytes]                          2.4
array first [512×16, at start]                         6.6
array first [512×16, at end]                       1,397.0
array best [512×16, exact at start]                     8.3
array best [512×16, exact at end]                   1,599.2
array all [512×16]                                  1,610.1
array best [16384×64, match at mid]                71,121.0
array all [16384×64, match at mid]                 79,365.0
array best [100000×128, parallel]                  50,996.0
array all [100000×128, parallel]                  144,800.0
================================================  ===========

On AArch64, LLVM's auto-vectorized native byte loop is faster than the
hand-written NEON byte kernel for these sizes, while hexadecimal strings still
use the packed NEON implementation. Large array workloads use four balanced
Rayon jobs to avoid oversubscribing the memory-bound scan.

Python API (via PyO3)
~~~~~~~~~~~~~~~~~~~~~

These numbers include Python wrapper and function-call overhead using
``pytest-benchmark``.

======================================================  ===========
Name                                                      Mean (ns)
======================================================  ===========
hamming_distance_string [3 chars, same]                       56.3
hamming_distance_string [3 chars, diff]                      105.8
hamming_distance_string [64 chars, diff]                      60.0
hamming_distance_string [1024 chars, diff]                   177.1
hamming_distance_bytes [3 bytes, same]                        51.7
hamming_distance_bytes [3 bytes, diff]                        51.8
hamming_distance_bytes [64 bytes, diff]                       51.9
hamming_distance_bytes [1024 bytes, diff]                     68.9
check_hexstrings_within_dist [1000 chars]                     56.4
check_bytes_within_dist [16 bytes]                            52.5
check_bytes_within_dist [64 bytes]                            51.8
check_bytes_within_dist [127 bytes]                           52.8
first_within_dist [512×16, at start]                          58.5
first_within_dist [512×16, mid]                              771.4
first_within_dist [512×16, at end]                         1,475.1
first_within_dist [16384×64, at start]                       160.3
first_within_dist [16384×64, mid]                         23,031.5
first_within_dist [16384×64, at end]                      45,801.2
best_within_dist [512×16, at start]                           75.3
best_within_dist [512×16, at end]                          1,703.5
best_within_dist [16384×64, mid]                          93,212.8
all_within_dist [512×16, at start]                         1,735.5
all_within_dist [512×16, at end]                           1,747.3
all_within_dist [16384×64, mid]                           93,056.3
======================================================  ===========

For small inputs, Python call and wrapper overhead dominates (roughly 40–55 ns
on this machine). For large inputs
(1024+ chars, 16384-element arrays), computation dominates and Python overhead
is negligible. Array APIs transparently parallelize with Rayon once the input
exceeds ~64 KiB; the ``first`` variant additionally short-circuits on the first
hit, so a match near the start is much faster than one near the end.
