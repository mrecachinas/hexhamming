#include "./python_hexhamming.hh"

/**
 * Python interface for `hamming_distance`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `hamming_distance` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns         the integer hamming distance between the binary
 */
static PyObject * hamming_distance_wrapper(PyObject *self, PyObject *args) {
    char *input_s1;
    char *input_s2;

    if (!PyArg_ParseTuple(args, "ss", &input_s1, &input_s2)) {
        PyErr_SetString(PyExc_ValueError, "error occurred while parsing arguments");
        return NULL;
    }

    if (input_s1 == NULL || input_s2 == NULL) {
        PyErr_SetString(PyExc_ValueError, "one or no strings provided!");
        return NULL;
    }

    std::string s1(input_s1);
    std::string s2(input_s2);
    if (s1.length() != s2.length()) {
        PyErr_SetString(PyExc_ValueError, "strings are NOT the same length");
        return NULL;
    }

    unsigned dist = 0;
    try {
        dist = hamming_distance(s1, s2);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    return Py_BuildValue("I", dist);
}
