#!/usr/bin/env python
from setuptools import setup, Extension


with open("./README.rst", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name="hexhamming",
    version="2.0.1",
    description="Fast Hamming distance calculation for hexadecimal strings",
    url="https://github.com/mrecachinas/hexhamming.git",
    long_description=LONG_DESCRIPTION,
    ext_modules=[
        Extension(
            name="hexhamming",
            sources=["hexhamming/python_hexhamming.cc"],
            extra_compile_args=["-march=native"],
            language="c++11",
        )
    ],
)
