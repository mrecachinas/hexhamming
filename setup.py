#!/usr/bin/env python
"""Setup script for hexhamming C extension.

This file is kept for building the C extension with platform-specific
compiler flags. All other metadata is in pyproject.toml.
"""

from setuptools import setup, Extension
from platform import machine, system, uname
from os import environ

extra_compile_args = []
cibw_arch = environ.get("CIBW_ARCHS_MACOS", "")
cibw_linux_arch = environ.get("CIBW_ARCHS_LINUX", "")
host_machine = machine().lower()

if system().lower() == "darwin":
    if cibw_arch == "x86_64" or (cibw_arch == "" and host_machine != "arm64"):
        extra_compile_args.extend(["-msse4.2", "-mpopcnt"])
    else:
        extra_compile_args.append("-mcpu=apple-m1")
elif uname().system == "Windows":
    extra_compile_args.append("-O2")
    extra_compile_args.append("/d2FH4-")
elif cibw_linux_arch == "aarch64" or host_machine == "aarch64":
    extra_compile_args.append("-march=armv8-a+simd")
else:
    extra_compile_args.extend(["-msse4.2", "-mpopcnt"])

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
