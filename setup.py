#!/usr/bin/env python
from distutils.core import setup, Extension


#with open("./README.rst", "r") as readme_file:
#	LONG_DESCRIPTION = readme_file.read()
LONG_DESCRIPTION = ""

setup(
	name='Hamming',
	version='1.0',
	description='',
	long_description=LONG_DESCRIPTION,
    ext_modules=[
    	Extension(
    		name="hamming",
    		sources=["hamming/python_hamming.cc", "hamming/hamming.cc"],
    		extra_compile_args=["-O0", "-mavx512f"],
    		include_dirs=['./hamming/'],
    		language="c++",
    	)
    ],
)
