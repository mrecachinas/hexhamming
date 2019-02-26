#!/usr/bin/env python
from distutils.core import setup, Extension


with open("./README.rst", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name='Hex-Hamming',
    version='1.0',
    description='Fast Hamming distance calculation for hexidecimal strings',
    url='https://github.com/mrecachinas/hex-hamming.git',
    long_description=LONG_DESCRIPTION,
    ext_modules=[
        Extension(
            name="hexhamming",
            sources=["hexhamming/python_hexhamming.cc", "hexhamming/hexhamming.cc"],
            extra_compile_args=["-O0", "-mavx512f"],
            include_dirs=['./hexhamming/'],
            language="c++",
        )
    ],
)
