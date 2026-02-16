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

Below is a benchmark using ``pytest-benchmark`` with hexhamming v3.0.0
on Apple M-series (ARM64) with Python 3.14 and ``rustc`` 1.85.

String and bytes hamming distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================================================  ===========  ==========
Name                                                      Mean (ns)    Std (ns)
======================================================  ===========  ==========
hamming_distance_string [3 chars, same]                       48.8        10.1
hamming_distance_string [3 chars, diff]                       48.4         4.4
hamming_distance_string [64 chars, diff]                      88.2        16.0
hamming_distance_string [1000 chars, same]                   754.7       251.2
hamming_distance_string [1000 chars, diff]                   762.3        75.7
hamming_distance_string [1024 chars, same]                   775.1        62.8
hamming_distance_string [1024 chars, diff]                   785.0       137.1
hamming_distance_bytes [3 bytes, same]                        48.5         5.5
hamming_distance_bytes [3 bytes, diff]                        49.0         5.2
hamming_distance_bytes [64 bytes, diff]                       50.3         8.4
hamming_distance_bytes [1000 bytes, same]                     64.9         5.4
hamming_distance_bytes [1000 bytes, diff]                     64.9        11.7
hamming_distance_bytes [1024 bytes, same]                     63.2        35.3
hamming_distance_bytes [1024 bytes, diff]                     69.1        16.0
check_bytes_within_dist [16 bytes]                            52.1         8.9
check_bytes_within_dist [64 bytes]                            51.2        23.2
check_bytes_within_dist [127 bytes]                           53.4         5.3
======================================================  ===========  ==========

Array API (batch search)
~~~~~~~~~~~~~~~~~~~~~~~~

===============================================================  =============  ============
Name                                                               Mean (ns)      Std (ns)
===============================================================  =============  ============
first_within_dist [512 elems × 16B, match at start]                     70.0           6.3
first_within_dist [512 elems × 16B, match at end]                      721.2          65.7
first_within_dist [16384 elems × 64B, match at end]                 48,927.1       8,321.5
best_within_dist [512 elems × 16B]                                     759.2         108.4
best_within_dist [16384 elems × 64B]                                46,295.0       3,793.5
all_within_dist [512 elems × 16B]                                      776.3          70.9
all_within_dist [16384 elems × 64B]                                 46,602.1       2,944.3
===============================================================  =============  ============
