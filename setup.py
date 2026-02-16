#!/usr/bin/env python
"""Setup script for hexhamming C extension.

This file is kept for building the C extension with platform-specific
compiler flags. All other metadata is in pyproject.toml.
"""

from setuptools import setup, Extension
from platform import machine, system, uname
from os import environ

extra_compile_args = []
if system().lower() == "darwin" and (
    machine().lower() == "arm64" or environ.get("CIBW_ARCHS_MACOS", "") == "arm64"
):
    extra_compile_args.append("-mcpu=apple-m1")
elif uname().system == "Windows":
    extra_compile_args.append("-O2")
    extra_compile_args.append("/d2FH4-")
else:
    extra_compile_args.append("-march=native")

setup(
    ext_modules=[
        Extension(
            name="hexhamming",
            sources=["src/python_hexhamming.cc"],
            extra_compile_args=extra_compile_args,
            language="c++11",
        )
    ],
)
