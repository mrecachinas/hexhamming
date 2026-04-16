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

All benchmarks on Apple M-series (ARM64) with hexhamming v3.0.0, ``rustc`` 1.85, Python 3.14.

Raw Rust (no Python overhead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These numbers show the pure computation time using Rust's ``criterion`` benchmarks
(``cargo bench --no-default-features``), with no Python/PyO3 overhead.

===========================================  ===========
Name                                           Mean (ns)
===========================================  ===========
hex_string (NEON) [16 chars]                        2.0
hex_string (NEON) [64 chars]                        5.8
hex_string (NEON) [128 chars]                      11.2
hex_string (NEON) [254 chars]                      21.2
bytes (NEON) [8 bytes]                              1.8
bytes (NEON) [32 bytes]                             2.8
bytes (NEON) [64 bytes]                             2.6
bytes (NEON) [127 bytes]                            6.6
bytes_within_dist [127 bytes]                       1.6
array first [512×16, at start]                      2.3
array first [512×16, at end]                      671.4
array best [512×16]                               925.8
array all [512×16]                                799.4
===========================================  ===========

Larger array workloads cross the parallel (Rayon) threshold; see the Python
table below for representative end-to-end numbers.

Python API (via PyO3)
~~~~~~~~~~~~~~~~~~~~~

These numbers include Python function call overhead (~45 ns) using ``pytest-benchmark``.

======================================================  ===========
Name                                                      Mean (ns)
======================================================  ===========
hamming_distance_string [3 chars, same]                       84.6
hamming_distance_string [3 chars, diff]                       83.6
hamming_distance_string [64 chars, diff]                      91.3
hamming_distance_string [1024 chars, diff]                   207.4
hamming_distance_bytes [3 bytes, same]                       127.3
hamming_distance_bytes [3 bytes, diff]                        96.2
hamming_distance_bytes [64 bytes, diff]                      169.9
hamming_distance_bytes [1024 bytes, diff]                    175.2
check_hexstrings_within_dist [1000 chars]                    221.3
check_bytes_within_dist [16 bytes]                           146.2
check_bytes_within_dist [64 bytes]                           107.7
check_bytes_within_dist [127 bytes]                          100.6
first_within_dist [512×16, at start]                          98.7
first_within_dist [512×16, mid]                              570.8
first_within_dist [512×16, at end]                         1,030.1
first_within_dist [16384×64, at start]                        99.6
first_within_dist [16384×64, mid]                         20,838.8
first_within_dist [16384×64, at end]                      41,489.5
best_within_dist [512×16, at start]                        1,609.8
best_within_dist [512×16, at end]                          1,116.4
best_within_dist [16384×64, mid]                          46,826.1
all_within_dist [512×16, at start]                         1,342.9
all_within_dist [512×16, at end]                           1,365.9
all_within_dist [16384×64, mid]                           48,067.2
======================================================  ===========

For small inputs, Python call overhead dominates (~45 ns). For large inputs
(1024+ chars, 16384-element arrays), computation dominates and Python overhead
is negligible. Array APIs transparently parallelize with Rayon once the input
exceeds ~64 KiB; the ``first`` variant additionally short-circuits on the first
hit, so a match near the start is much faster than one near the end.
