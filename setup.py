#!/usr/bin/env python
from setuptools import setup, Extension
from platform import machine, system, uname
from re import search, IGNORECASE
from os import environ


def get_version():
    version_file = "hexhamming/_version.h"
    with open(version_file) as f:
        return search(r'_version.*"(.*)";', f.read(), IGNORECASE).groups()[0]


# Platform-specific compiler optimization flags
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
    version=get_version(),
    ext_modules=[
        Extension(
            name="hexhamming",
            sources=["hexhamming/python_hexhamming.cc"],
            extra_compile_args=extra_compile_args,
            language="c++11",
        )
    ],
)
