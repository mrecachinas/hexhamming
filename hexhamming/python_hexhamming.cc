#include <cstring>
#include <string.h>
#if _MSC_VER
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "python_hexhamming.h"

///////////////////////////////////////////////////////////////
// C API
///////////////////////////////////////////////////////////////

//Pointers to functions. Will be inited in PyMODINIT_FUNC by USE__* macros(can be found at end of header file).
static int (*ptr__hamming_distance_bytes)(const uint8_t*, const uint8_t*, const size_t, const ssize_t);
static int (*ptr__hamming_distance_string)(const char*, const char*, const size_t);
//Bit mask off CPU capabilities.
static int cpu_capabilities;


/**
 * Returns true if hexstrings are within a Hamming distance;
 * otherwise, returns false (stops early)
 *
 * @param a    hexadecimal char array
 * @param b    hexadecimal char array
 * @param string_length length of `a` and `b`
 * @param max_dist maximum allowable hamming distance
 * @return      whether or not the strings are within some
 *              predefined hamming distance; -1 means an error has occurred
 */
inline int check_hexstrings_within_dist(
    const char* a,
    const char* b,
    size_t string_length,
    size_t max_dist
  ) {
    // if both strings are the same, short circuit
    // and return 0
    if (strcmp(a, b) == 0) {
      return 1;
    }

    size_t result = 0;
    int val1, val2;
    size_t i;
    for (i = 0; i < string_length; ++i) {
        // Convert the hex ascii char to its actual hexadecimal value
        // e.g., '0' = 0, 'A' = 10, etc.
        // Note: this is case INSENSITIVE
        // Also note:
        //   * 65 = 'A'
        //   * -55 = -65 + 10
        val1 = (a[i] > '9') ? (a[i] &~ 0x20) - 55: (a[i] - '0');
        val2 = (b[i] > '9') ? (b[i] &~ 0x20) - 55: (b[i] - '0');

        // check to make sure all characters are valid
        // hexadecimal in both strings
        if (val1 > 15 || val1 < 0 || val2 > 15 || val2 < 0) {
            return -1;
        }

        result += LOOKUP[val1 ^ val2];
        if (result > max_dist) {
            return 0;
        }
    }

    return 1;
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
static PyObject * hamming_distance_string_wrapper(PyObject *self, PyObject *args) {
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

    // if the two strings are not the same length, can't move on, so raise
    if (input_s1_len != input_s2_len) {
        PyErr_SetString(PyExc_ValueError, "strings are NOT the same length");
        return NULL;
    }

    // at this point, we can safely proceed with
    // our `hamming_distance` computation
    int dist = ptr__hamming_distance_string(input_s1, input_s2, input_s1_len);
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

/**
 * Python interface for `hamming_distance`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `hamming_distance` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns         the integer hamming distance between the binary
 */
static PyObject * hamming_distance_byte_wrapper(PyObject *self, PyObject *args) {
    uint8_t *input_s1;
    uint8_t *input_s2;
    size_t input_s1_len;
    size_t input_s2_len;

    // get the two strings from `args`
    // if they are incorrect types (i.e., not 's'), this will raise
    // a ValueError
    if (!PyArg_ParseTuple(args, "s#s#", &input_s1, &input_s1_len, &input_s2, &input_s2_len)) {
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

    // if the two strings are not the same length, can't move on, so raise
    if (input_s1_len != input_s2_len) {
        PyErr_SetString(PyExc_ValueError, "bytes are NOT the same length");
        return NULL;
    }

    // at this point, we can safely proceed with
    // our `hamming_distance` computation
    int dist = ptr__hamming_distance_bytes(input_s1, input_s2, input_s1_len, -1);
    return Py_BuildValue("I", dist);
}

/**
 * Python interface for `check_hexstrings_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_hexstrings_within_dist` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns
 */
static PyObject * check_hexstrings_within_dist_wrapper(PyObject *self, PyObject *args) {
    char *input_s1;
    char *input_s2;
    unsigned int max_dist;

    // get the two strings from `args`
    // if they are incorrect types (i.e., not 's'), this will raise
    // a ValueError
    if (!PyArg_ParseTuple(args, "ssI", &input_s1, &input_s2, &max_dist)) {
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

    if ((int)max_dist < 0) {
        PyErr_SetString(PyExc_ValueError, "`max_dist` must be >0");
        return NULL;
    }

    if (max_dist > input_s1_len) {
        return Py_BuildValue("O", Py_True);
    }

    // at this point, we can safely proceed with
    // our `hamming_distance` computation
    int result = check_hexstrings_within_dist(
        input_s1,
        input_s2,
        input_s1_len,
        max_dist
    );
    if (result == -1) {
      // this should only happen if the strings contain
      // invalid hexadecimal characters
      PyErr_SetString(PyExc_ValueError, "hex string contains invalid char");
      return NULL;
    } else {
      // put the unsigned int into a Python Int object
      // and return back to the caller!
      return Py_BuildValue("O", result == 1 ? Py_True : Py_False);
    }
}

/**
 * Python interface for `check_bytes_arrays_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_bytes_arrays_within_dist` interface
 *                  - `array_of_elems` - bytes
 *                  - `elem_to_compare` - bytes
 *                  - `max_dist` - int
 * @returns         index of element in array_of_elems or -1.
 */
static PyObject * check_bytes_arrays_within_dist_wrapper(PyObject *self, PyObject *args) {
    uint8_t *big_array, *small_array;
    size_t big_array_size, small_array_size;
    ssize_t max_dist;

    if (!PyArg_ParseTuple(args, "s#s#n", &big_array, &big_array_size, &small_array, &small_array_size, &max_dist)) {
        PyErr_SetString(
            PyExc_ValueError,
            "error occurred while parsing arguments"
        );
        return NULL;
    }

    if (small_array_size == 0) {
        PyErr_SetString(PyExc_ValueError, "`elem_to_compare` size must be >0");
        return NULL;
    }

    if (max_dist < 0) {
        PyErr_SetString(PyExc_ValueError, "`max_dist` must be >=0");
        return NULL;
    }

    if (big_array_size % small_array_size != 0) {
        PyErr_SetString(PyExc_ValueError, "`array_of_elems` size must be multiplier of `elem_to_compare`");
        return NULL;
    }

    int res;
    int number_of_elements = big_array_size / small_array_size;
    uint8_t* pBig = big_array;
    for (int i = 0; i < number_of_elements; i++, pBig += small_array_size) {
        res = ptr__hamming_distance_bytes(pBig, small_array, small_array_size, max_dist);
        if (res == 1)
            return Py_BuildValue("i", i);
        }
    return Py_BuildValue("i", -1);
}

/**
 * Python interface for `set_algo`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `set_algo` interface
 *                  - `string` -- string with one of those: "extra","native","sse41","classic"
 * @returns         empty string if success or string with error.
 */
static PyObject * set_algo_wrapper(PyObject *self, PyObject *args) {
    char *algo_name;

    // get the one string with algo name. if they are incorrect types (i.e., not 's'), this will raise a ValueError
    if (!PyArg_ParseTuple(args, "s", &algo_name)) {
        PyErr_SetString(
            PyExc_ValueError,
            "error occurred while parsing arguments"
        );
        return NULL;
    }

    // if either c-string is NULL, can't move on, so raise
    if (algo_name == NULL) {
        PyErr_SetString(PyExc_ValueError, "no string provided!");
        return NULL;
    }

    const char *result = "";
    if (strcmp(algo_name, "extra") == 0) {
        if ((cpu_capabilities & bit_AVX2) == bit_AVX2) {
            USE__EXTRA
        }
        else
            result = "CPU doesnt support this feature.";
    }
#if defined(HAVE_NATIVE_POPCNT)
    else if (strcmp(algo_name, "native") == 0) {
        if ((cpu_capabilities & bit_POPCNT) == bit_POPCNT) {
            USE__NATIVE
        }
        else
            result = "CPU doesnt support this feature.";
    }
#endif
#if defined(CPU_X86_64)
    else if (strcmp(algo_name, "sse41") == 0) {
        if ((cpu_capabilities & bit_SSE41) == bit_SSE41) {
            USE__SSE41
        }
        else
            result = "CPU doesnt support this feature.";
    }
#endif
    else if (strcmp(algo_name, "classic") == 0) {
        USE__CLASSIC
    }
    else
        result = "Library was built without this algorithm.";
    return Py_BuildValue("s", result);
}

///////////////////////////////////////////////////////////////
// Docstrings
///////////////////////////////////////////////////////////////
static char hamming_string_docstring[] =
    "Calculate the hamming distance of two strings\n\n"
    "This is equivalent to\n\n"
    "    bin(int(a, 16) ^ int(b, 16)).count('1')\n\n"
    "with the only difference being it was written in C and optimized\n"
    "using a lookup table of pre-calculated hexadecimal hamming distances.\n"
    ":param a: hexadecimal string\n"
    ":type a: str\n"
    ":param b: hexadecimal string\n"
    ":type b: str\n"
    ":returns: the hamming distance between the bits of two hexadecimal strings\n"
    ":rtype: int\n"
    ":raises ValueError: if either string doesn't exist, "
    "if the strings are different lengths, or if the strings aren't valid hex";


static char hamming_byte_docstring[] =
    "Calculate the hamming distance of two byte strings\n\n"
    "with the only difference being it was written in C and optimized\n"
    "using a lookup table of pre-calculated hexadecimal hamming distances.\n"
    ":param a: hexadecimal string\n"
    ":type a: byte\n"
    ":param b: hexadecimal string\n"
    ":type b: byte\n"
    ":returns: the hamming distance between the bits of two hexadecimal strings\n"
    ":rtype: int\n"
    ":raises ValueError: if either string doesn't exist, "
    "if the strings are different lengths, or if the strings aren't valid hex";

static char check_hexstrings_within_dist_docstring[] =
    "Check if the hex strings are within a specified Hamming Distance\n\n"
    "This is equivalent to\n\n"
    "    bin(int(a, 16) ^ int(b, 16)).count('1') < max_dist \n\n"
    "with the only difference being it was written in C and optimized\n"
    "using a lookup table of pre-calculated hexadecimal hamming distances.\n"
    ":param a: hexadecimal string\n"
    ":type a: str\n"
    ":param b: hexadecimal string\n"
    ":type b: str\n"
    ":param max_dist: maximum allowable Hamming Distance\n"
    ":type max_dist: int\n"
    ":returns: whether or not the two hex strings are within `max_dist`\n"
    ":rtype: int\n"
    ":raises ValueError: if either string doesn't exist, "
    "if the strings are different lengths, or if the strings aren't valid hex";

static char check_bytes_arrays_within_dist_docstring[] =
    "Check if any element of byte array are within a specified Hamming Distance\n"
    "and return it's index or -1 otherwise.\n\n"
    "Size of `elem_to_compare ` must be multiplier of `array_of_elems` size. \n\n"
    ":param array_of_elems: array of bytes to search within\n"
    ":type array_of_elems: bytes\n"
    ":param elem_to_compare: will compare to each element in array_of_elems\n"
    ":type elem_to_compare: bytes\n"
    ":param max_dist: maximum allowable Hamming Distance\n"
    ":type max_dist: int\n"
    ":returns: index of first element in array_of_elems for which hamming distance <= `max_dist` or -1. \n"
    ":rtype: int\n"
    ":raises ValueError: if input parameters are invalid.";

static char set_algo_docstring[] =
    "Change algo used for calculations, return empty string if ok or string with error.\n\n"
    "For Internal and test/benchmark use.\n\n"
    ":param string: extra|native|sse41|classic\n"
    ":raises ValueError: if input parameters are invalid.";

static char CompareDocstring[] =
    "Module for calculating hamming distance of two hexadecimal strings";

///////////////////////////////////////////////////////////////
// Python C-extension Initialization
///////////////////////////////////////////////////////////////
static PyMethodDef CompareMethods[] = {
    {"hamming_distance_string", hamming_distance_string_wrapper, METH_VARARGS, hamming_string_docstring},
    {"hamming_distance_bytes", hamming_distance_byte_wrapper, METH_VARARGS, hamming_byte_docstring},
    {"check_hexstrings_within_dist", check_hexstrings_within_dist_wrapper, METH_VARARGS, check_hexstrings_within_dist_docstring},
    {"check_bytes_arrays_within_dist", check_bytes_arrays_within_dist_wrapper, METH_VARARGS, check_bytes_arrays_within_dist_docstring},
    {"set_algo", set_algo_wrapper, METH_VARARGS, set_algo_docstring},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef hexhammingdef = {
        PyModuleDef_HEAD_INIT,
        "hexhamming",
        CompareDocstring,
        -1,
        CompareMethods,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_hexhamming(void)

#else
#define INITERROR return

PyMODINIT_FUNC
inithexhamming(void)
#endif

{
     #if defined(CPU_X86_64)
        cpu_capabilities = get_cpuid();
        if ((cpu_capabilities & bit_AVX2) == bit_AVX2) {
            USE__EXTRA
        }
#if defined(HAVE_NATIVE_POPCNT)
        else if ((cpu_capabilities & bit_POPCNT) == bit_POPCNT) {
            USE__NATIVE
        }
#endif
        else if ((cpu_capabilities & bit_SSE41) == bit_SSE41) {
            USE__SSE41
        }
        else {
            USE__CLASSIC
        }
     #else
        #if defined(HAVE_NATIVE_POPCNT)
            cpu_capabilities = bit_POPCNT;
            USE__NATIVE
        #else
            USE__CLASSIC
        #endif
     #endif
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&hexhammingdef);
#else
    PyObject *module = Py_InitModule3("hexhamming", CompareMethods, CompareDocstring);
#endif
    if (PyModule_AddStringConstant(module, "__version__", "2.1.1")) {
        Py_XDECREF(module);
        INITERROR;
    }
    if (module == NULL) {
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
