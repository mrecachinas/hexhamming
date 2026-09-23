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

Batch APIs
~~~~~~~~~~

The per-call APIs above are still the right choice for one-off distances, but
computing many distances in Python for-loops pays repeated FFI overhead.
The batch APIs below fold that overhead into a single call by taking
contiguous buffers.

Pairwise distances between two equal-length contiguous buffers of fixed-width
records:

::

    >>> from hexhamming import (
    ...     hamming_distances_bytes,
    ...     hamming_distances_bytes_packed,
    ...     hamming_distances_bytes_into,
    ... )
    >>> a = b"\xde\xad\xbe\xef" * 4
    >>> b = b"\x00" * 16
    >>> hamming_distances_bytes(a, b, 4)      # list[int]
    [24, 24, 24, 24]
    >>> hamming_distances_bytes_packed(a, b, 4).hex()   # little-endian u64 bytes
    '1800000000000000180000000000000018000000000000001800000000000000'
    >>> out = bytearray(4 * 8)
    >>> hamming_distances_bytes_into(a, b, 4, out)      # writes u64 LE into `out`
    4

``hamming_distances_bytes_into`` requires ``out`` to be a writable,
C-contiguous byte buffer of exactly ``count * 8`` bytes; read-only,
non-contiguous, or wrong-size outputs raise ``ValueError``. Writable ``_into``
APIs are unavailable on free-threaded Python because the buffer protocol does
not provide exclusive access; use the corresponding ``_packed`` API there.
On standard Python builds, ``_into`` keeps the GIL while writing; use
``_packed`` when detached computation is more important than buffer reuse.

Multi-query catalog scans run one catalog against many contiguous queries in
one call, mirroring the shape of repeated single-query calls:

::

    >>> from hexhamming import (
    ...     check_bytes_arrays_first_many_within_dist,
    ...     check_bytes_arrays_best_many_within_dist,
    ...     check_bytes_arrays_all_many_within_dist,
    ... )
    >>> catalog = b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff"
    >>> queries = b"\xff\xff\xef\xfe"
    >>> check_bytes_arrays_first_many_within_dist(catalog, queries, 2, 4)
    [1, 4]
    >>> check_bytes_arrays_best_many_within_dist(catalog, queries, 2, 4)
    [(0, 5), (2, 4)]
    >>> check_bytes_arrays_all_many_within_dist(catalog, queries, 2, 4)
    [[(4, 1), (4, 3), (4, 4), (0, 5)], [(2, 4), (2, 5)]]

Semantics match the single-query calls exactly: ``-1`` and ``(-1, -1)``
sentinels for no-match, lowest-index tie-breaking for ``best_many``, exact-match
short-circuiting, and ascending index order for ``all_many``.

Catalog
~~~~~~~

If you query the same fixed-width byte catalog repeatedly, ``Catalog`` copies
and validates the records once and can optionally build a Multi-Index Hashing
(MIH) index for small-radius lookups:

::

    >>> import hexhamming
    >>> records = b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff"
    >>> cat = hexhamming.Catalog(records, 2, index=True)
    >>> len(cat), cat.width, cat.has_index
    (6, 2, True)
    >>> cat.first_within(b"\xff\xff", 4)
    1
    >>> cat.best_within(b"\xef\xfe", 4)
    (2, 4)
    >>> cat.all_within(b"\xff\xff", 4)
    [(4, 1), (4, 3), (4, 4), (0, 5)]

``Catalog(..., index=False)`` uses the same linear scanners as the free
functions while avoiding repeated catalog buffer acquisition. ``index=True`` is
best for large, mostly static catalogs queried repeatedly at small Hamming
radii, such as near-duplicate lookups on 64- or 256-bit perceptual hashes. For
each query the catalog estimates whether probing the index or scanning
linearly is faster and uses the cheaper one. Widths that cannot be indexed
within the memory cap always scan. Either way the results, including
tie-breaking, are identical to the free functions.

Measured on an Apple M4 Max with 1M records, ``best_within`` for one query
(the free function scans on every core):

======  ======  =============  ===========  ========
width   radius  free function  indexed      speedup
======  ======  =============  ===========  ========
8 B     2       17.3 µs        92 ns        189x
8 B     8       16.9 µs        1.4 µs       12x
8 B     12      19.2 µs        18.6 µs      1.0x
32 B    8       116 µs         143 ns       811x
32 B    32      94.6 µs        6.1 µs       16x
======  ======  =============  ===========  ========

At radius 12 on 8-byte records the planner expects probing to cost more than
a scan, so ``best_within`` scans linearly.

Building the index took 24 ms for the 8-byte catalog and 91 ms for the
32-byte one.

The MIH index uses flat CSR substring tables, not hash maps. For ``N`` records
and ``m`` substring tables, memory overhead is approximately::

    sum_j 4 * ((2 ** s_j + 1) + N) bytes

where ``s_j`` is the bit width of substring ``j``. The planner chooses
``s_j`` near ``ceil(log2(N)) + 2`` and caps it at 22 bits; this keeps small
catalogs from allocating oversized tables and bounds million-record 256-bit
catalog overhead much lower than an uncapped 24-bit plan.

Dense/compact match transport for ``all_within_dist`` uses ``u16``
distances and ``u32`` indices instead of Python tuples:

::

    >>> from hexhamming import (
    ...     check_bytes_arrays_all_within_dist_packed,
    ...     check_bytes_arrays_all_within_dist_into,
    ... )
    >>> dbytes, ibytes = check_bytes_arrays_all_within_dist_packed(catalog, b"\xff\xff", 4)
    >>> [int.from_bytes(dbytes[i:i+2], "little") for i in range(0, len(dbytes), 2)]
    [4, 4, 4, 0]
    >>> d_out = bytearray(len(catalog) // 2 * 2)   # worst case: num_records * 2
    >>> i_out = bytearray(len(catalog) // 2 * 4)   # worst case: num_records * 4
    >>> check_bytes_arrays_all_within_dist_into(catalog, b"\xff\xff", 4, d_out, i_out)
    4

The ``_packed`` variant returns two ``bytes`` objects; ``_into`` writes into
caller-provided writable buffers and returns the match count. Element widths
whose maximum possible distance exceeds ``u16::MAX`` bits, and catalogs with
more than ``u32::MAX`` records, are rejected. On free-threaded Python, use
``_packed`` because writable ``_into`` buffers cannot be made exclusive through
the Python buffer protocol.

Benchmark
---------

Unless noted otherwise, the numbers below were measured on an Apple M4 Max
(ARM64, 12 performance and 4 efficiency cores, 64 GiB) with ``rustc`` 1.98.1
and Python 3.14.7, using a local build without profile-guided optimization
(release wheels built natively with PGO are 4–11% faster on small calls).
Python tables come from ``scripts/readme_benchmarks.py``: each value is the
median of seven runs, each the best of seven ``timeit`` repeats. Rust numbers
are Criterion medians of three runs (``cargo bench --no-default-features``
with ``--warm-up-time 1 --measurement-time 1 --sample-size 20``). Scans that
use the thread pool vary by up to ~30% between runs.

Raw Rust (no Python overhead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

===========================================  =========
Name                                         Mean (ns)
===========================================  =========
hex_string (NEON) [16 chars]                       1.7
hex_string (NEON) [64 chars]                       3.7
hex_string (NEON) [128 chars]                      7.3
hex_string (NEON) [254 chars]                     14.0
bytes (native) [8 bytes]                           1.1
bytes (native) [32 bytes]                          1.6
bytes (native) [64 bytes]                          2.1
bytes (native) [127 bytes]                         5.3
bytes_within_dist [127 bytes]                      1.6
array first [512×16, at start]                     5.4
array first [512×16, at end]                     101.6
array best [512×16, exact at start]                5.7
array best [512×16, exact at end]                104.6
array all [512×16]                               106.5
array best [16384×64, match at mid]              5,750
array all [16384×64, match at mid]               4,780
array best [100000×128, exact match at 100]        158
array all [100000×128, parallel]                22,660
===========================================  =========

Block scanners compare 16 records at a time, so a match on the very first
record costs a whole block (~5 ns) instead of one comparison; every later
position is several times faster. On AArch64, LLVM's auto-vectorized native
byte loop is faster than a hand-written NEON byte kernel for these sizes.
Large array scans are split into chunks that a small built-in thread pool
claims dynamically.

Fixed-width array matrix, 1024 records, deterministic random no-match and
exact-midpoint cases:

=======================  ============  ============
Case                     16-byte (ns)  32-byte (ns)
=======================  ============  ============
random no-match / first         207.6         419.9
random no-match / best          208.4         434.6
random no-match / all           211.5         442.2
exact midpoint / first          109.9         209.8
exact midpoint / best           108.6         212.5
exact midpoint / all            215.2         422.7
=======================  ============  ============

Python API
~~~~~~~~~~

These include the Python call, argument parsing and buffer access:

=====================================================  =========
Name                                                   Mean (ns)
=====================================================  =========
hamming_distance_string [3 chars, same]                     26.8
hamming_distance_string [3 chars, diff]                     30.7
hamming_distance_string [64 chars, diff]                    36.2
hamming_distance_string [1024 chars, diff]                  78.8
hamming_distance_bytes [3 bytes, same]                      31.0
hamming_distance_bytes [3 bytes, diff]                      31.5
hamming_distance_bytes [64 bytes, diff]                     32.3
hamming_distance_bytes [1024 bytes, diff]                   51.9
hamming_distance_bytes [64-byte bytearray]                  39.2
hamming_distance_bytes [64-byte memoryview]                 42.9
check_hexstrings_within_dist [1000 chars, early exit]       30.4
check_bytes_within_dist [16 bytes]                          30.7
check_bytes_within_dist [64 bytes]                          31.4
check_bytes_within_dist [127 bytes]                         34.3
first_within_dist [512×16, at start]                        39.7
first_within_dist [512×16, mid]                             87.1
first_within_dist [512×16, at end]                         128.3
first_within_dist [16384×64, at start]                      54.3
first_within_dist [16384×64, mid]                        4,535.3
first_within_dist [16384×64, at end]                     4,645.5
best_within_dist [512×16, at start]                         44.6
best_within_dist [512×16, at end]                          143.8
best_within_dist [16384×64, mid]                         4,802.0
all_within_dist [512×16, at start]                         161.3
all_within_dist [512×16, at end]                           162.5
all_within_dist [16384×64, mid]                          4,640.4
=====================================================  =========

The same fixed-width matrix end to end (1024 records):

=======================  ============  ============
Case                     16-byte (ns)  32-byte (ns)
=======================  ============  ============
random no-match / first         226.8         467.8
random no-match / best          246.4         460.7
random no-match / all           238.7         457.4
exact midpoint / first          133.4         238.8
exact midpoint / best           150.4         261.0
exact midpoint / all            273.8         480.0
=======================  ============  ============

For random inputs, the direct APIs also avoid the temporary big integers used
by an equivalent standard-library implementation:

================  ===============  ===========  =======
Input             hexhamming (ns)  stdlib (ns)  Speedup
================  ===============  ===========  =======
bytes [16]                   31.4        156.9    4.99×
bytes [64]                   32.7        240.5    7.37×
bytes [1024]                 52.2      1,944.0   37.22×
hex [16 chars]               30.4        120.9    3.98×
hex [64 chars]               36.5        196.4    5.38×
hex [1024 chars]             79.9      1,732.2   21.67×
================  ===============  ===========  =======

For small ``str`` and ``bytes`` inputs, the call itself dominates (roughly
30 ns on this machine). For large inputs (1024+ chars, 16384-element arrays),
computation dominates and Python overhead is negligible. Byte operations
release the GIL at 16 KiB, while immutable strings use a zero-copy detached
path from 4 KiB. Array wrappers release the GIL at 64 KiB. Array scans run on
a built-in thread pool from 512 KiB for widths with a block scanner
(8/16/32/64 bytes) and from 64 KiB for other widths, the measured crossovers
on this machine. Set ``HEXHAMMING_NUM_THREADS`` to limit the pool (``1``
disables it). The ``first`` and ``best`` variants scan a short prefix serially
and stop early, so a match near the start is much faster than one near the
end; batches of them likewise start serially and only spread across threads
once they prove expensive.

Large catalogs
~~~~~~~~~~~~~~

One query against a large catalog uses every core; at 64 MiB the scan is
bound by memory bandwidth (~270 GB/s):

=================================  ==========  =========  ========
Catalog                            first (ns)  best (ns)  all (ns)
=================================  ==========  =========  ========
1 MiB, 16-byte records, no match      5,594.8    6,019.1   4,442.2
16 MiB, 16-byte records, no match    33,805.1   34,924.4  30,559.1
64 MiB, 16-byte records, no match     240,981    246,913   236,442
=================================  ==========  =========  ========

Batch APIs vs. Python for-loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "loop" columns run the equivalent single-call API inside a Python
for-loop.

Pairwise distances between two contiguous buffers of ``count`` records:

==================  =========  =========  ===========  =========
Case                loop (ns)  list (ns)  packed (ns)  into (ns)
==================  =========  =========  ===========  =========
pairwise 100×16      10,559.1      326.9        176.7       82.8
pairwise 1,000×16     109,158    2,474.0      1,004.1      377.7
pairwise 10,000×16  1,114,309   24,482.0     10,213.5    3,497.0
pairwise 100×32      10,618.4      364.5        221.5       94.5
pairwise 1,000×32     109,849    2,875.0      1,520.0      489.9
pairwise 10,000×32  1,100,303   28,953.9     15,064.2    5,883.6
==================  =========  =========  ===========  =========

Multi-query catalog scans against a 1,024×16-byte catalog with 100 queries:

=============================================  =========  ==========
Case                                           loop (ns)  batch (ns)
=============================================  =========  ==========
first_many 100×1024×16 (permissive threshold)    2,646.4       467.7
best_many 100×1024×16 (max_dist=128)            26,027.8     7,871.2
=============================================  =========  ==========

Dense-match transport for a single query against a 1,024×16-byte catalog:

=====================================  =========  ===========  =========
Case                                   list (ns)  packed (ns)  into (ns)
=====================================  =========  ===========  =========
all 1024×16 (max_dist=128, all match)   18,611.5      1,697.6      967.6
=====================================  =========  ===========  =========

Interpretation:

* Pairwise: the ``list`` API is 29–46× faster than the Python for-loop and is
  the recommended default. ``packed`` and ``into`` skip the per-distance
  Python ``int`` allocation for another 1.7–2.5× and 4–7× respectively; use
  them when the caller can consume little-endian ``u64`` bytes directly.
* ``first_many`` with a permissive threshold is a large win (≈5.7×): every
  inner scan stops at its first record, so per-call overhead dominates the
  loop, and a batch of such cheap scans runs serially rather than waking the
  thread pool. ``best_many`` and ``all_many`` scan the whole catalog for each
  query; batches as large as this one are spread across threads (≈3.3×).
* Dense ``all_within_dist``: ``packed`` avoids allocating ``num_records``
  Python 2-tuples (≈11×); ``into`` additionally reuses caller-owned
  buffers (≈19×).

x86 (AVX2 and AVX-512)
~~~~~~~~~~~~~~~~~~~~~~

CI runs the test suite under the Intel Software Development Emulator on
Nehalem, Haswell, Skylake-X and Ice Lake CPU models, so the SSE4.1, AVX2 and
AVX-512 paths are all checked for correctness, and the Benchmark workflow
runs on every pull request. For repeatable performance measurements, run the
same checkout three times on each representative x86 machine:

.. code-block:: bash

    scripts/benchmark_x86.sh before
    # Apply the candidate optimization, then:
    scripts/benchmark_x86.sh after

The script records CPU features and tool versions alongside Criterion output
and end-to-end Python benchmark JSON. Compare results only between runs from
the same machine.

On an Emerald Rapids CI runner (Intel Xeon Platinum 8573C), the AVX-512 block
scanners are 2.5–4.8× faster than the AVX2 scanners for 16-, 32- and 64-byte
records (the ``Benchmark PR`` comparison on #63). The table below predates
them: it compares the earlier AVX-512 kernels with the code before them, as
three-run medians on a Google Cloud ``c4-standard-4`` with an Intel Xeon
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
