Hexadecimal Hamming
====================

Why yet another hamming distance library?
-----------------------------------------

There are a lot of fantastic (python) libraries to calculate various edit
distances, including pyhamming, etc.

In this case, I needed a hamming distance library that worked on hexadecimal
strings (i.e., ``str`` in Python-speak) and performed blazingly fast.
Furthermore, I often did not care about hex strings greater than 256 bits.
That length constraint is different vs all the other libraries and enabled me
to explore vectorization techniques via ``numba``, ``numpy``, and even
``SSE/AVX``.

Lastly, I wanted to minimize dependencies, meaning you do not need to install
``numpy`` or ``gmpy``.

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

    pytest

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

Benchmarks
----------
