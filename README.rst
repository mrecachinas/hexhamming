Hexadecimal Hamming
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
to explore vectorization techniques via ``numba``, ``numpy``, and even
``SSE/AVX``.

Lastly, I wanted to minimize dependencies, meaning you do not need to install
``numpy``, ``gmpy``, ``cython``, ``pypy``, ``pythran``, etc.

Eventually, after playing around with ``gmpy.popcount``, ``numba.jit``,
``pythran.run``, ``numpy``, and ``AVX2``, I decided to write what I wanted
in a raw C header. At this point, I'm using raw ``char*`` and
``int*``, so exploring re-writing this in Fortran makes little sense. Vectorization
techniques also ended up adding more overhead due to converting the hex strings to
vector-register-ingestible floats from ``char*``.

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

To use the base C++ extension, you can simply run

::

    >>> from hexhamming import hamming_distance
    >>> hamming_distance("deadbeef", "00000000")
    24

Benchmark
---------

Note: For the below image, to show how optimized this is, I included
the benchmark of a function that looks like

Below is a benchmark using ``pytest-benchmark`` with hexhamming==v1.3.0
my early 2016 1.2 GHz Intel m5 8 GB 1867 MHz LPDDR3 macOS Mojave (10.14.3)
with Python 3.7.3 and Apple clang version 11.0.0 (clang-1100.0.33.17).

=======================================  ===========  ==========  =============  ========  ============
Name                                       Mean (ns)    Std (ns)    Median (ns)    Rounds    Iterations
=======================================  ===========  ==========  =============  ========  ============
test_hamming_distance_bench_short_same       182.21      282.187        140.1      137742            30
test_hamming_distance_bench_short            204.275     353.317        154.156    183723            32
test_hamming_distance_bench_long_same        431.369     553.671        329.5      132838            20
test_check_hexstrings_within_dist_bench      419.923     489.503        330.1       83718            20
test_hamming_distance_bench_256              649.275    2854.9          505        172118             1
test_hamming_distance_bench_long            3569.42     6408.05        2758        160591             1
=======================================  ===========  ==========  =============  ========  ============
