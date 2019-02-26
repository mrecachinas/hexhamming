Hexadecimal Hamming
====================

Why yet another hamming distance library?
-----------------------------------------

There are a lot of fantastic (python) libraries that offer methods to calculate
various edit distances, including Hamming distances: Distance, textdistance,
scipy.spatial.distance.hamming, jellyfish, etc.

In this case, I needed a hamming distance library that worked on hexadecimal
strings (i.e., ``str`` in Python-speak) and performed blazingly fast.
Furthermore, I often did not care about hex strings greater than 256 bits.
That length constraint is different vs all the other libraries and enabled me
to explore vectorization techniques via ``numba``, ``numpy``, and even
``SSE/AVX``.

Lastly, I wanted to minimize dependencies, meaning you do not need to install
``numpy``, ``gmpy``, ``cython``, ``pypy``, etc.

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

    pytest # or tox

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

Benchmarks
----------

Some benchmarks between ``hamming_distance``, ``hamming_distance_lookup``,
and ``fast_hamming_distance``::

    [1] %timeit hamming_distance('deadbeef', '00000000')

    [2] %timeit hamming_distance_lookup('deadbeef', '00000000')

    [3] %timeit fast_hamming_distance('deadbeef', '00000000')

Below are some other implementations that were considered::

    [1] import gmpy
    [2] def hamming_distance_gmpy(a, b):
    ...     return gmpy.popcount(int(a, 16) ^ int(b, 16))

    [3] %timeit hamming_distance_gmpy('deadbeef', '00000000')

    [4] import numpy
    [5] from numba import jit
    [6] def hamming_distance_numpy(a, b):
