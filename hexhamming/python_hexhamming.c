#include "./python_hexhamming.h"

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

    // get the two strings from `args`
    // if they are incorrect types (i.e., not 's'), this will raise
    // a ValueError
    if (!PyArg_ParseTuple(args, "ss", &input_s1, &input_s2)) {
        PyErr_SetString(
            PyExc_ValueError,
            "error occurred while parsing arguments"
        );
        return NULL;
    }

    // if either c-string is NULL, can't move on, so raise
    if (input_s1 == NULL || input_s2 == NULL) {
        PyErr_SetString(PyExc_ValueError, "one or no strings provided!");
        return NULL;
    }

    size_t input_s1_len = strlen(input_s1);
    size_t input_s2_len = strlen(input_s2);

    if (input_s1_len != input_s2_len) {
        PyErr_SetString(PyExc_ValueError, "strings are NOT the same length");
        return NULL;
    }

    // at this point, we can safely proceed with
    // our `hamming_distance` computation
    unsigned dist = 0;
    dist = hamming_distance(input_s1, input_s2, input_s1_len, input_s2_len);
    if (dist == -1) {
      // this should only happen if the strings contain
      // invalid hexadecimal characters
      PyErr_SetString(PyExc_ValueError, "hex string contains invalid char");
      return NULL;
    } else {
      // put the unsigned int into a Python Int object
      // and return back to the caller!
      return Py_BuildValue("I", dist);
    }
}
