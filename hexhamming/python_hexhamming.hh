#ifndef PYTHON_HEXHAMMING_WRAPPER_H
#define PYTHON_HEXHAMMING_WRAPPER_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <bitset>

inline PyObject * array_to_pylist_int(const std::array<int, 8> &v);
inline PyObject * vector_to_pylist_bool(const std::vector<bool> &v);
inline std::vector<bool> pylist_int_to_vector(PyObject * list);

///////////////////////////////////////////////////////////////
// Python API Wrappers
///////////////////////////////////////////////////////////////
static PyObject * str_to_hex256_wrapper(PyObject *self, PyObject *args);
static PyObject * str_to_hex_wrapper(PyObject *self, PyObject *args);
static PyObject * xor_vec_wrapper(PyObject *self, PyObject *args);
static PyObject * hamming_distance_wrapper(PyObject *self, PyObject *args);
static PyObject * hamming_distance_lookup_wrapper(PyObject *self, PyObject *args);

#ifdef __AVX__

static PyObject * fast_hamming_distance_wrapper(PyObject *self, PyObject *args);

#endif

///////////////////////////////////////////////////////////////
// Docstrings
///////////////////////////////////////////////////////////////
#ifdef __AVX__

static char fast_hamming_docstring[] =
    "Calculate the hamming distance of two 256-bit hex strings (SSE-optimized)";

#endif

static char hamming_docstring[] =
    "Calculate the hamming distance of two strings";

static char hamming_distance_lookup_docstring[] =
    "Calculate the hamming distance of two 256-max-length hex strings";

static char str_to_hex256_docstring[] =
    "Convert hex string to list of 8 32-bit integers";

static char str_to_hex_docstring[] =
    "Convert hex string to list of binary";

static char xor_list_docstring[] =
    "XOR two lists of bool/binary-int";

static char CompareDocstring[] =
    "Module for calculating hamming distance";

///////////////////////////////////////////////////////////////
// Python C-extension Initialization
///////////////////////////////////////////////////////////////
static PyMethodDef CompareMethods[] = {

#ifdef __AVX__
    {"fast_hamming_distance", fast_hamming_distance_wrapper, METH_VARARGS, fast_hamming_docstring},
#endif

    {"hamming_distance", hamming_distance_wrapper, METH_VARARGS, hamming_docstring},
    {"hamming_distance_lookup", hamming_distance_lookup_wrapper, METH_VARARGS, hamming_distance_lookup_docstring},
    {"str_to_hex", str_to_hex_wrapper, METH_VARARGS, str_to_hex_docstring},
    {"str_to_hex256", str_to_hex256_wrapper, METH_VARARGS, str_to_hex256_docstring},
    {"xor_lists", xor_vec_wrapper, METH_VARARGS, xor_list_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inithexhamming(void) {
    PyObject *m = Py_InitModule3("hexhamming", CompareMethods, CompareDocstring);
    if (m == NULL) {
        return;
    }
}


#endif
