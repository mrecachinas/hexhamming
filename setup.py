#!/usr/bin/env python
from distutils.core import setup, Extension


with open("./README.rst", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name='Hamming',
    version='1.0',
    description='',
    url='https://github.com/mrecachinas/hex-hamming.git'
    long_description=LONG_DESCRIPTION,
    ext_modules=[
        Extension(
            name="hamming",
            sources=["hamming/python_hamming.cc", "hamming/hamming.cc"],

            # TODO: Figure out why -O2 causes an illegal hardware instruction
            extra_compile_args=["-O0", "-mavx512f"],

            include_dirs=['./hamming/'],
            language="c++",
        )
    ],
)
