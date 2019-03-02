Hexadecimal Hamming
====================

|Pip|_ |Prs|_ |Travis|_

.. |Pip| image:: https://badge.fury.io/py/hexhamming.svg
.. _Pip: https://badge.fury.io/py/hexhamming

.. |Prs| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
.. _Prs: .github/CONTRIBUTING.md#pull-requests

.. |Travis| image:: https://travis-ci.org/mrecachinas/hexhamming.svg?branch=master
.. _Travis: https://travis-ci.org/mrecachinas/hexhamming

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
in a raw C++ header. Note: the only C++-feature I'm exploiting is C++ exceptions;
without that, this could easily be C. At this point, I'm using raw ``char*`` and
``int*``, so exploring re-writing this in Fortran makes little sense. Vectorization
techniques also ended up adding more overhead from data transfer between
vector registers and normal registers; also, converting the hex strings to
vector-register-ingestible floats from ``char*`` proved to have a non-trivial
overhead.

Installation
-------------

To install, ensure you have Python 2.7 or 3.4+. Run::

    pip install hexhamming

or to install from source::

    git clone https://github.com/mrecachinas/hexhamming
    cd hexhamming
    python setup.py install # or pip install .

If you want to contribute to hexhamming, you should install the dev
dependencies::

    pip install -r requirements-dev.txt

and make sure the tests pass with::

    pytest # or tox -e py27,...

Example
-------

To use the base C++ extension, you can simply run::

    >>> from hexhamming import hamming_distance
    >>> hamming_distance('deadbeef', '00000000')
    24

To use the lookup-based C++ extension, replace the above
``hamming_distance`` with ``hamming_distance_lookup``.

If your machine supports the Intel SSE4/AVX2 instruction set,
replace the above ``hamming_distance`` with ``fast_hamming_distance``.
Note: to  use ``fast_hamming_distance``, your hex string must be 64
characters or less (i.e., 256 bits or less).

Benchmark
---------

Below is a benchmark using ``pytest-benchmark`` on my early 2016 1.2 GHz Intel
m5 8 GB 1867 MHz LPDDR3 macOS Mojave (10.14.3) with Python 2.7.15 and
clang-1000.11.45.5.

.. image:: https://github.com/mrecachinas/hexhamming/blob/master/docs/benchmark.png?raw=true
