#!/usr/bin/env python
from setuptools import setup, Extension
from platform import machine, system, uname
from re import search, IGNORECASE
from os import environ


def get_version():
    version_file = "hexhamming/_version.h"
    with open(version_file) as f:
        return search(r'_version.*"(.*)";', f.read(), IGNORECASE).groups()[0]


with open("./README.rst", "r") as readme_file:
    long_description = readme_file.read()

with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    test_requirements = [line.rstrip() for line in fh.readlines()]

extra_compile_args = []
if system().lower() == "darwin" and (machine().lower() == "arm64" or
                                     environ.get("CIBW_ARCHS_MACOS", "") == "arm64"):
    extra_compile_args.append("-mcpu=apple-m1")
elif uname().system == 'Windows':
    extra_compile_args.append("-O2")
    extra_compile_args.append("/d2FH4-")
else:
    extra_compile_args.append("-march=native")

setup(
    name="hexhamming",
    version=get_version(),
    description="Fast Hamming distance calculation for hexadecimal strings",
    url="https://github.com/mrecachinas/hexhamming.git",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    test_suite="test",
    tests_require=test_requirements,
    ext_modules=[
        Extension(
            name="hexhamming",
            sources=["hexhamming/python_hexhamming.cc"],
            extra_compile_args=extra_compile_args,
            language="c++11",
        )
    ],
    author="Michael Recachinas",
    author_email="m.recachinas@gmail.com",
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C"
    ],
    keywords="hamming distance simd",
    zip_safe=False,
)
