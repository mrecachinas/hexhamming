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

To install, ensure you have Python 2.7 or 3.4+. Run

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

    pytest -vls

Example
-------

Using ``hexhamming`` is as simple as

::

    >>> from hexhamming import hamming_distance
    >>> hamming_distance("deadbeef", "00000000")
    24

Benchmark
---------

Below is a benchmark using ``pytest-benchmark`` with hexhamming==v1.3.0
my early 2016 1.2 GHz Intel m5 8 GB 1867 MHz LPDDR3 macOS Mojave (10.14.3)
with Python 3.7.3 and Apple clang version 11.0.0 (clang-1100.0.33.17).

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
