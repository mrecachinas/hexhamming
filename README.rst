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
to explore vectorization techniques via ``numba``, ``numpy``, and
``SSE/AVX`` intrinsics.

Lastly, I wanted to minimize dependencies, meaning you do not need to install
``numpy``, ``gmpy``, ``cython``, ``pypy``, ``pythran``, etc.

Eventually, after playing around with ``gmpy.popcount``, ``numba.jit``,
``pythran.run``, ``numpy``, I decided to write what I wanted
in essentially raw C. At this point, I'm using raw ``char*`` and
``int*``, so exploring re-writing this in Fortran makes little sense.

Installation
-------------

To install, ensure you have Python 3.6+. Run

::

    pip install hexhamming

or to install from source

::

    git clone https://github.com/mrecachinas/hexhamming
    cd hexhamming
    python setup.py install # or pip install .

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

**New in v2.0.0** : ``hexhamming`` now supports `byte`s via ``hamming_distance_bytes``.
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
a similar API as ``check_hexstrings_within_dist``, except it expects a byte array. 

The API described above is targeted at comparing two individual records and calculate their hamming distance quickly.
For many applications the goal is compare a given record to an array of other records and to find out if there 
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


``check_bytes_arrays_all_within_dist`` returns a  list of tuples with the distance and the index of the element that have a hamming 
distance less than ``max_dist``, or ``[])`` if none do.

::

    >>> from hexhamming import check_bytes_arrays_all_within_dist
    >>> check_bytes_arrays_all_within_dist(b"\xaa\xaa\xbb\xbb\xcc\xcc\xdd\xdd\xee\xee\xff\xff", b"\xff\xff", 4)
    [(4, 1), (4, 3), (4, 4), (0, 5)]


Tip: When you're assembling the long array of records to compare against, don't concatenate the different ``bytes`` together. As they're
immutable that is a very slow operation. Use a ``bytearray`` instead, and cast it to ``bytes`` at the end. See https://www.guyrutenberg.com/2020/04/04/fast-bytes-concatenation-in-python/ for more info and tests.


Benchmark
---------

Below is a benchmark using ``pytest-benchmark`` with hexhamming==v1.3.2
my 2020 2.0 GHz quad-core Intel Core i5 16 GB 3733 MHz LPDDR4 macOS Catalina (10.15.5)
with Python 3.7.3 and Apple clang version 11.0.3 (clang-1103.0.32.62).

=======================================  ===========  ==========  =============  ========  ============
Name                                       Mean (ns)    Std (ns)    Median (ns)    Rounds    Iterations
=======================================  ===========  ==========  =============  ========  ============
test_hamming_distance_bench_3                93.8        10.5          94.3         53268           200
test_hamming_distance_bench_3_same           94.2        15.2          94.9        102146           100
test_check_hexstrings_within_dist_bench      231.9      104.2         216.5        195122            22
test_hamming_distance_bench_256              97.5        34.1          94.0        195122            22
test_hamming_distance_bench_1000             489.8      159.4         477.5         94411            20
test_hamming_distance_bench_1000_same        497.8       87.8         496.6         18971            20
test_hamming_distance_bench_1024             509.9      299.5         506.7         18652            10
test_hamming_distance_bench_1024_same        467.4      205.9         450.4        181819            10
=======================================  ===========  ==========  =============  ========  ============
