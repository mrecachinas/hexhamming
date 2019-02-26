#ifndef PYTHON_HEXHAMMING_WRAPPER_H
#define PYTHON_HEXHAMMING_WRAPPER_H

#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>

#include <Python.h>
#include "./hexhamming.hh"

///////////////////////////////////////////////////////////////
// Python API Wrappers
///////////////////////////////////////////////////////////////
static PyObject * hamming_distance_wrapper(PyObject *self, PyObject *args);

///////////////////////////////////////////////////////////////
// Docstrings
///////////////////////////////////////////////////////////////

static char hamming_docstring[] =
    "Calculate the hamming distance of two strings";

static char CompareDocstring[] =
    "Module for calculating hamming distance of two hex strings";

///////////////////////////////////////////////////////////////
// Python C-extension Initialization
///////////////////////////////////////////////////////////////
static PyMethodDef CompareMethods[] = {
    {"hamming_distance", hamming_distance_wrapper, METH_VARARGS, hamming_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inithexhamming(void) {
    PyObject *m = Py_InitModule3("hexhamming", CompareMethods, CompareDocstring);
    if (m == NULL) {
        return;
    }
}


#endif
