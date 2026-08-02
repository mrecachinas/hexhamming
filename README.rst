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
SIMD-accelerated performance (SSE4.1, AVX2, AVX-512 BITALG, NEON).

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

For repeatable AVX2 and AVX-512 investigations, run the same checkout three
times on each representative x86 machine:

.. code-block:: bash

    scripts/benchmark_x86.sh before
    # Apply the candidate optimization, then:
    scripts/benchmark_x86.sh after

The script records CPU features and tool versions alongside Criterion output
and end-to-end Python benchmark JSON. Compare results only between runs from
the same machine.

AVX-512 results
~~~~~~~~~~~~~~~~

Three-run medians on a Google Cloud ``c4-standard-4`` with an Intel Xeon
Platinum 8581C (Emerald Rapids):

.. list-table::
   :header-rows: 1

   * - Workload
     - Before
     - After
     - Speedup
   * - Python 1024x16 first, random/no-match
     - 3.222 us
     - 0.679 us
     - 4.75x
   * - Python 1024x16 best, random/no-match
     - 3.720 us
     - 0.651 us
     - 5.72x
   * - Python 1024x16 all, random/no-match
     - 3.466 us
     - 0.710 us
     - 4.88x
   * - Python 1024x32 first, random/no-match
     - 3.199 us
     - 1.278 us
     - 2.50x
   * - Python 1024x32 best, random/no-match
     - 3.729 us
     - 1.383 us
     - 2.70x
   * - Python 1024x32 all, random/no-match
     - 3.445 us
     - 1.377 us
     - 2.50x

The AVX-512 byte kernel also uses masked loads below 64 bytes, improving the
measured 16-, 32-, 48-, and 63-byte Rust paths by 33%, 50%, 70%, and 194%
respectively. AVX2-only tuning remains hardware-dependent and should be
measured separately on a machine without AVX-512.

All benchmarks were run on an Apple M4 Max (ARM64, 16 logical cores, 64 GiB)
with hexhamming v3.0.0, ``rustc`` 1.97.1, and Python 3.14.6. Values are the
median of the means from three independent runs.

Raw Rust (no Python overhead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These numbers show the pure computation time using Rust's ``criterion`` benchmarks
(``cargo bench --no-default-features``), with no Python/PyO3 overhead.

Issue #51 fixed-width array matrix (1024 records; median of three run medians)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix uses deterministic random no-match and exact-midpoint cases for
16-byte and 32-byte records. Each run uses
``--warm-up-time 1 --measurement-time 1 --sample-size 20``.

====================================  ===========  ===========
Case                                 16-byte (ns) 32-byte (ns)
====================================  ===========  ===========
random no-match / first                    397.5        797.8
random no-match / best                     407.0        992.2
random no-match / all                      523.7       1047.7
exact midpoint / first                    216.3        422.7
exact midpoint / best                     216.6        414.0
exact midpoint / all                      540.9        822.0
====================================  ===========  ===========

================================================  ===========
Name                                              Mean (ns)
================================================  ===========
hex_string (NEON) [16 chars]                           1.6
hex_string (NEON) [64 chars]                           5.4
hex_string (NEON) [128 chars]                         10.5
hex_string (NEON) [254 chars]                         19.7
bytes (native) [8 bytes]                               1.1
bytes (native) [32 bytes]                              1.5
bytes (native) [64 bytes]                              2.1
bytes (native) [127 bytes]                             5.5
bytes_within_dist [127 bytes]                          1.6
array first [512×16, at start]                         1.9
array first [512×16, at end]                         402.4
array best [512×16, exact at start]                     3.3
array best [512×16, exact at end]                     526.4
array all [512×16]                                    449.0
array best [16384×64, match at mid]                10,986.0
array all [16384×64, match at mid]                 20,350.0
array best [100000×128, parallel]                  46,547.0
array all [100000×128, parallel]                   99,266.0
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
hamming_distance_string [3 chars, same]                       37.1
hamming_distance_string [3 chars, diff]                       70.7
hamming_distance_string [64 chars, diff]                      39.8
hamming_distance_string [1024 chars, diff]                   116.4
hamming_distance_bytes [3 bytes, same]                        33.4
hamming_distance_bytes [3 bytes, diff]                        39.9
hamming_distance_bytes [64 bytes, diff]                       33.9
hamming_distance_bytes [1024 bytes, diff]                     44.6
hamming_distance_bytes [64-byte bytearray]                    50.6
hamming_distance_bytes [64-byte memoryview]                   51.5
check_hexstrings_within_dist [1000 chars]                     37.3
check_bytes_within_dist [16 bytes]                            34.3
check_bytes_within_dist [64 bytes]                            33.7
check_bytes_within_dist [127 bytes]                           34.7
first_within_dist [512×16, at start]                          35.6
first_within_dist [512×16, mid]                              240.7
first_within_dist [512×16, at end]                           440.5
first_within_dist [16384×64, at start]                        74.2
first_within_dist [16384×64, mid]                         14,319.4
first_within_dist [16384×64, at end]                      28,619.9
best_within_dist [512×16, at start]                           47.0
best_within_dist [512×16, at end]                            584.3
best_within_dist [16384×64, mid]                          32,031.4
all_within_dist [512×16, at start]                           530.4
all_within_dist [512×16, at end]                             537.5
all_within_dist [16384×64, mid]                           30,725.4
======================================================  ===========

Issue #51 Python buffer matrix (1024 records; median of three run medians)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These end-to-end timings use ``timeit.repeat`` with 10,000 calls per sample,
including the PyO3 wrapper and buffer-protocol path.

====================================  ===========  ===========
Case                                 16-byte (ns) 32-byte (ns)
====================================  ===========  ===========
random no-match / first                    443.4        836.8
random no-match / best                     487.0        850.2
random no-match / all                      564.8        851.7
exact midpoint / first                    272.2        468.1
exact midpoint / best                     295.0        483.8
exact midpoint / all                      638.8        921.0
====================================  ===========  ===========

For random inputs, the direct APIs also avoid the temporary big integers used
by an equivalent standard-library implementation:

================  ===============  ==============  ========
Input             hexhamming (ns)  stdlib (ns)     Speedup
================  ===============  ==============  ========
bytes [16]                   33.5           158.3     4.72×
bytes [64]                   37.4           233.1     6.24×
bytes [1024]                 53.2         2,029.0    38.17×
hex [16 chars]               37.2           126.3     3.39×
hex [64 chars]               39.7           200.2     5.05×
hex [1024 chars]            116.5         1,708.1    14.66×
================  ===============  ==============  ========

For small exact ``str`` and ``bytes`` inputs, Python call and wrapper overhead
dominates (roughly 30–40 ns on this machine). For large inputs
(1024+ chars, 16384-element arrays), computation dominates and Python overhead
is negligible. Byte operations release the GIL at 16 KiB, while immutable
strings use a zero-copy detached path from 4 KiB. Array wrappers release the GIL
at 64 KiB; generic scans parallelize with Rayon at 5 MiB, while the optimized
16/32-byte NEON scanners use a measured 16 MiB crossover. The ``first`` variant
additionally short-circuits on the first hit, so a match near the start is much
faster than one near the end.
