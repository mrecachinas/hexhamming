#!/usr/bin/env python
"""
Alternative Python setup script for the Rust implementation
"""
from setuptools import setup
from setuptools_rust import RustExtension

def get_version():
    return "2.2.3"

with open("../README.rst", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="hexhamming-rs",
    version=get_version(),
    description="Fast Hamming distance calculation for hexadecimal strings (Rust implementation)",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Michael Recachinas",
    author_email="m.recachinas@gmail.com",
    url="https://github.com/mrecachinas/hexhamming",
    rust_extensions=[
        RustExtension(
            "hexhamming_rs",
            path="Cargo.toml",
            binding="pyo3",
            debug=False,
        )
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    zip_safe=False,
    include_package_data=True,
)