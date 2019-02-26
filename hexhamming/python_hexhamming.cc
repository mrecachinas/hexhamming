#include <Python.h>
#include "./hexhamming.hh"
#include "./python_hexhamming.hh"

/**
 * Wrapper for converting std::array<int> to PyList containing PyInts
 *
 * @param v     std::array<int> to be converted
 * @return      PyObject* that has the properties of a PyList
 * @throws      std::logic_error if memory can't be allocated or if non-binary values in list
 */
inline PyObject * array_to_pylist_int(const std::array<int, 8> &v) {
    PyObject *list = PyList_New(v.size());
    if (!list) {
        throw std::logic_error("unable to allocate memory for Python list");
    }
    PyObject *list_item;
    for (int i = 0; i < v.size(); ++i) {
        list_item = Py_BuildValue("I", v[i]);
        PyList_SetItem(list, i, list_item);
    }
    return list;
}

/**
 * Wrapper for converting std::vector<bool> to PyList containing PyInts
 *
 * @param v     std::vector<bool> to be converted
 * @return      PyObject* that has the properties of a PyList
 * @throws      std::logic_error if memory can't be allocated or if non-binary values in list
 */
inline PyObject * vector_to_pylist_bool(const std::vector<bool> &v) {
    PyObject *list = PyList_New(v.size());
    if (!list) {
        throw std::logic_error("unable to allocate memory for Python list");
    }
    PyObject *list_item;
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] != 0 && v[i] != 1) {
            throw std::logic_error("non-binary values in list");
        }
        list_item = Py_BuildValue("I", v[i] ? 1 : 0);
        PyList_SetItem(list, i, list_item);
    }
    return list;
}

/**
 * Convert PyObject* to std::vector<bool>
 *
 * @param list      either Python list or tuple as PyObject*
 * @return          std::vector<bool> representation of PyObject*
 * @throws          std::logic_error if non-binary values exist, std::invalid_argument if
 *                  list is not a PyList or PyTuple
 */
inline std::vector<bool> pylist_int_to_vector(PyObject * list) {
    size_t list_size = 0;
    if (PyTuple_Check(list)) {
        list_size = PyTuple_Size(list);
    } else if (PyList_Check(list)) {
        list_size = PyList_Size(list);
    }
    std::vector<bool> data(list_size);
    if (PyTuple_Check(list)) {
        for(Py_ssize_t i = 0; i < list_size; ++i) {
            PyObject *value = PyTuple_GetItem(list, i);
            long val = PyInt_AsLong(value);
            if (val != 0 && val != 1) {
                throw std::logic_error("non-binary values in list");
            }
            data[i] = val ? true : false;
        }
    } else {
        if (PyList_Check(list)) {
            for(Py_ssize_t i = 0; i < list_size; i++) {
                PyObject *value = PyList_GetItem(list, i);
                long val = PyInt_AsLong(value);
                if (val != 0 && val != 1) {
                    throw std::logic_error("non-binary values in list");
                }
                data[i] = val ? true : false;
            }
        } else {
            throw std::invalid_argument("not a list or tuple!");
        }
    }
    return data;
}



///////////////////////////////////////////////////////////////
// Python API Wrappers
///////////////////////////////////////////////////////////////

/**
 * Python interface for `str_to_hex256`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `str_to_hex256` interface
 *                  - `str` -- hex string that you want to convert to int list of size 8
 * @returns         Python list of int32 values representing hex string
 */
static PyObject * str_to_hex256_wrapper(PyObject *self, PyObject *args) {
    char *input;

    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_ValueError, "error occurred while parsing argument");
        return NULL;
    }

    if (input == NULL) {
        PyErr_SetString(PyExc_ValueError, "only one string provided!");
        return NULL;
    }

    std::string s(input);
    if (s.length() > 256) {
        PyErr_SetString(PyExc_ValueError, "string length must be <=256");
        return NULL;
    }
    try {
        std::array<int, 8> arr = str_to_hex256(s);
        return array_to_pylist_int(arr);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}

/**
 * Python interface for `str_to_hex`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `str_to_hex` interface
 *                  - `str` -- hex string that you want to convert to binary list
 * @returns         Python list of boolean/binary values representing hex string
 */
static PyObject * str_to_hex_wrapper(PyObject *self, PyObject *args) {
    char *input;

    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_ValueError, "error occurred while parsing argument");
        return NULL;
    }

    if (input == NULL) {
        PyErr_SetString(PyExc_ValueError, "only one string provided!");
        return NULL;
    }

    std::string s(input);
    try {
        std::vector<bool> bin_vec = str_to_hex(s);
        return vector_to_pylist_bool(bin_vec);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}

/**
 * Python interface for `xor_vec`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `xor_vec` interface
 *                  - `list1` -- binary list
 *                  - `list2` -- binary list
 * @returns         the (integer) number of signed bits in the result of
 *                  `list1 ^ list2`
 */
static PyObject * xor_vec_wrapper(PyObject *self, PyObject *args) {
    PyObject *list1;
    PyObject *list2;

    if (!PyArg_ParseTuple(args, "OO", &list1, &list2)) {
        PyErr_SetString(PyExc_ValueError, "error occurred while parsing arguments");
        return NULL;
    }

    if (list1 == NULL || list2 == NULL) {
        PyErr_SetString(PyExc_ValueError, "one or no lists provided!");
        return NULL;
    }

    std::vector<bool> v1, v2;
    try {
        v1 = pylist_int_to_vector(list1);
        v2 = pylist_int_to_vector(list2);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    try {
        unsigned xor_result = xor_vec(v1, v2);
        return Py_BuildValue("I", xor_result);
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, "lists/tuples are not the same length");
        return NULL;
    }
}

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

#ifdef __AVX__

/**
 * Python interface for `hamming_distance_sse`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `hamming_distance_sse` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns         the integer hamming distance between the binary
 */
static PyObject * fast_hamming_distance_wrapper(PyObject *self, PyObject *args) {
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

    if (s1.length() > 256 || s2.length() > 256) {
        PyErr_SetString(PyExc_ValueError, "string lengths must be <=256");
    }

    unsigned dist = 0;
    try {
        dist = hamming_distance_sse(s1, s2);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    return Py_BuildValue("I", dist);
}

/**
 * Python interface for `hamming_distance_sse`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `hamming_distance_sse` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns         the integer hamming distance between the binary
 */
static PyObject * hamming_distance_lookup_wrapper(PyObject *self, PyObject *args) {
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
        dist = hamming_distance_lookup(s1, s2);
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }

    return Py_BuildValue("I", dist);
}

#endif
