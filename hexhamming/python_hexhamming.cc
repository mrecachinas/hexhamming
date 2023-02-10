#include <cstring>
#include <string.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "python_hexhamming.h"
#include "_version.h"


///////////////////////////////////////////////////////////////
// C API
///////////////////////////////////////////////////////////////

//Pointers to functions. Will be inited in PyMODINIT_FUNC by USE__* macros(can be found at end of header file).
static uint64_t (*ptr__hamming_distance_bytes)(const uint8_t*, const uint8_t*, const uint64_t, const int64_t);
static uint64_t (*ptr__hamming_distance_string)(const char*, const char*, const uint64_t);
static int cpu_capabilities;            //Bit mask off CPU capabilities.
char cpu_not_support_msg[64];           //"CPU doesnt support this feature. %X" , cpu_capabilities


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
    uint64_t string_length,
    int64_t max_dist
  ) {
    // if both strings are the same, short circuit
    // and return 0
    if (strcmp(a, b) == 0) {
      return 1;
    }

    int64_t result = 0;
    int val1, val2;
    for (size_t i = 0; i < string_length; ++i) {
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

    uint64_t input_s1_len = strlen(input_s1);
    uint64_t input_s2_len = strlen(input_s2);

    // if the two strings are not the same length, can't move on, so raise
    if (input_s1_len != input_s2_len) {
        PyErr_SetString(PyExc_ValueError, "strings are NOT the same length");
        return NULL;
    }

    // at this point, we can safely proceed with
    // our `hamming_distance` computation
    uint64_t dist = ptr__hamming_distance_string(input_s1, input_s2, input_s1_len);
    if (dist == UINT64_MAX) {
        // this should only happen if the strings contain
        // invalid hexadecimal characters
        PyErr_SetString(PyExc_ValueError, "hex string contains invalid char");
        return NULL;
    } else {
        // put the unsigned int64 into a Python Int object
        // and return back to the caller!
        return Py_BuildValue("K", dist);
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
    uint64_t input_s1_len = 0;
    uint64_t input_s2_len = 0;

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
    uint64_t dist = ptr__hamming_distance_bytes(input_s1, input_s2, input_s1_len, -1);
    return Py_BuildValue("K", dist);
}

/**
 * Python interface for `check_hexstrings_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_hexstrings_within_dist` interface
 *                  - `string1` -- hex string
 *                  - `string2` -- hex string
 * @returns			True if the hamming distance is less or equal to max_dist, False otherwise.
 */
static PyObject * check_hexstrings_within_dist_wrapper(PyObject *self, PyObject *args) {
    char *input_s1;
    char *input_s2;
    uint64_t max_dist;

    // get the two strings from `args`
    // if they are incorrect types (i.e., not 's'), this will raise
    // a ValueError
    if (!PyArg_ParseTuple(args, "ssK", &input_s1, &input_s2, &max_dist)) {
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

    uint64_t input_s1_len = strlen(input_s1);
    uint64_t input_s2_len = strlen(input_s2);

    if (input_s1_len != input_s2_len) {
        PyErr_SetString(PyExc_ValueError, "strings are NOT the same length");
        return NULL;
    }

    if ((int64_t)max_dist < 0) {
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
 * Python interface for `check_bytes_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_bytes_arrays_first_within_dist` interface
 *                  - `array_of_elems` - bytes
 *                  - `elem_to_compare` - bytes
 *                  - `max_dist` - int64
 * @returns         True if the hamming distance is less or equal to max_dist, False otherwise.
 */
static PyObject * check_bytes_within_dist_wrapper(PyObject *self, PyObject *args) {
    uint8_t *array_1, *array_2;
    uint64_t array_1_size = 0;
    uint64_t array_2_size = 0;
    int64_t max_dist;

    if (!PyArg_ParseTuple(args, "s#s#L", &array_1, &array_1_size, &array_2, &array_2_size, &max_dist)) {
        PyErr_SetString(
            PyExc_ValueError,
            "error occurred while parsing arguments"
        );
        return NULL;
    }

    if (array_2_size == 0 || array_1_size == 0) {
        PyErr_SetString(PyExc_ValueError, "array size must be >0");
        return NULL;
    }

    if (max_dist < 0) {
        PyErr_SetString(PyExc_ValueError, "`max_dist` must be >=0");
        return NULL;
    }

    if (array_1_size != array_2_size) {
        PyErr_SetString(PyExc_ValueError, "array sizes need to be the same");
        return NULL;
    }
    
    int64_t result;
    result = ptr__hamming_distance_bytes(array_1, array_2, array_2_size, max_dist);
		
    return Py_BuildValue("O", result == 1 ? Py_True : Py_False);
}

/**
 * Python interface for `check_bytes_arrays_first_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_bytes_arrays_first_within_dist` interface
 *                  - `array_of_elems` - bytes
 *                  - `elem_to_compare` - bytes
 *                  - `max_dist` - int64
 * @returns         index of element in array_of_elems or -1.
 */
static PyObject * check_bytes_arrays_first_within_dist_wrapper(PyObject *self, PyObject *args) {
    uint8_t *big_array, *small_array;
    uint64_t big_array_size = 0;
    uint64_t small_array_size = 0;
    int64_t max_dist;

    if (!PyArg_ParseTuple(args, "s#s#L", &big_array, &big_array_size, &small_array, &small_array_size, &max_dist)) {
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

    int64_t ret = -1;
	
    Py_BEGIN_ALLOW_THREADS
	
    uint64_t number_of_elements = big_array_size / small_array_size;
    uint8_t* pBig = big_array;
	uint64_t res;
    for (uint64_t i = 0; i < number_of_elements; i++, pBig += small_array_size) {
        res = ptr__hamming_distance_bytes(pBig, small_array, small_array_size, max_dist);
        if (res == 1){
            ret = i;
            break;
        }
    }
	
    Py_END_ALLOW_THREADS
	
    return Py_BuildValue("i", ret);
}

/**
 * Python interface for `check_bytes_arrays_best_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_bytes_arrays_first_within_dist` interface
 *                  - `array_of_elems` - bytes
 *                  - `elem_to_compare` - bytes
 *                  - `max_dist` - int64
 * @returns         tuple of best distance found and index of element in array_of_elems, or (-1, -1).
 */
static PyObject * check_bytes_arrays_best_within_dist_wrapper(PyObject *self, PyObject *args) {
    uint8_t *big_array, *small_array;
    uint64_t big_array_size = 0;
    uint64_t small_array_size = 0;
    int64_t max_dist;

    if (!PyArg_ParseTuple(args, "s#s#L", &big_array, &big_array_size, &small_array, &small_array_size, &max_dist)) {
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

    int64_t best_dist = max_dist + 1;
    int64_t best_index = -1;

    Py_BEGIN_ALLOW_THREADS

    uint64_t dist;
    uint64_t number_of_elements = big_array_size / small_array_size;
    uint8_t* pBig = big_array;
    for (uint64_t i = 0; i < number_of_elements; i++, pBig += small_array_size) {
        dist = ptr__hamming_distance_bytes(pBig, small_array, small_array_size, -1);
        if (dist < best_dist) {
            best_dist = dist;
            best_index = i;
        }
    }
	
    // Anything found? If not, set dist to -1
    if (best_index == -1)
    {
        best_dist = -1;
    }
	
    Py_END_ALLOW_THREADS
	
    return Py_BuildValue("ii", best_dist, best_index);
}

/**
 * Python interface for `check_bytes_arrays_all_within_dist`
 *
 * @param self      Python `self` object
 * @param args      Python arguments for `check_bytes_arrays_all_within_dist` interface
 *                  - `array_of_elems` - bytes
 *                  - `elem_to_compare` - bytes
 *                  - `max_dist` - int64
 * @returns         list of tuples of (distance, index) for all indices that have a distance < max_dist.
 */
static PyObject * check_bytes_arrays_all_within_dist_wrapper(PyObject *self, PyObject *args) {
    uint8_t *big_array, *small_array;
    uint64_t big_array_size = 0;
    uint64_t small_array_size = 0;
    int64_t max_dist;

    if (!PyArg_ParseTuple(args, "s#s#L", &big_array, &big_array_size, &small_array, &small_array_size, &max_dist)) {
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

    uint64_t number_of_elements = big_array_size / small_array_size;
    uint64_t *out = new uint64_t[number_of_elements * 2];
    uint64_t *o = out;

    Py_BEGIN_ALLOW_THREADS

    uint64_t dist;
    uint8_t* pBig = big_array;
    for (uint64_t i = 0; i < number_of_elements; i++, pBig += small_array_size) {
        dist = ptr__hamming_distance_bytes(pBig, small_array, small_array_size, -1);
        if (dist <= max_dist) {
            *o++ = dist;
            *o++ = i;
        }
    }
	
    Py_END_ALLOW_THREADS
	
    /* Assemble result... */
    PyObject *my_list = PyList_New(0);
    if ( my_list == NULL )  
    {
        PyErr_NoMemory();
    }
	
    for (uint64_t *op = out; op < o; op += 2)
    {
        PyObject *tup = Py_BuildValue("ii", op[0], op[1]);
        if (tup == NULL)
        {
            PyErr_NoMemory();
        }
		
        if(PyList_Append(my_list, tup) == -1) 
        {
            PyErr_NoMemory();
        }
    }
	
    delete [] out;
	
    return my_list;
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
            result = cpu_not_support_msg;
    }
#if defined(HAVE_NATIVE_POPCNT)
    else if (strcmp(algo_name, "native") == 0) {
        if ((cpu_capabilities & bit_POPCNT) == bit_POPCNT) {
            USE__NATIVE
        }
        else
            result = cpu_not_support_msg;
    }
#endif
#if defined(CPU_X86_64)
    else if (strcmp(algo_name, "sse41") == 0) {
        if ((cpu_capabilities & bit_SSE41) == bit_SSE41) {
            USE__SSE41
        }
        else
            result = cpu_not_support_msg;
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
    "with the only difference being it was written in C and optimized.\n"
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
    ":rtype: bool\n"
    ":raises ValueError: if either string doesn't exist, "
    "if the strings are different lengths, or if the strings aren't valid hex";

static char check_bytes_within_dist_docstring[] =
    "Check if the byte strings are within a specified Hamming Distance\n\n"
    ":param a: bytes array\n"
    ":type a: bytes\n"
    ":param b: bytes array\n"
    ":type b: bytes\n"
    ":param max_dist: maximum allowable Hamming Distance\n"
    ":type max_dist: int\n"
    ":returns: whether or not the two hex strings are within `max_dist`\n"
    ":rtype: bool\n"
    ":raises ValueError: if either string doesn't exist, "
    "if the strings are different lengths.";

static char check_bytes_arrays_first_within_dist_docstring[] =
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

static char check_bytes_arrays_best_within_dist_docstring[] =
    "Find element in bytes array that has the smallest Hamming distance to elem\n"
    "and return the distance and it's index.\n\n"
    "Size of `elem_to_compare ` must be multiplier of `array_of_elems` size. \n\n"
    ":param array_of_elems: array of bytes to search within\n"
    ":type array_of_elems: bytes\n"
    ":param elem_to_compare: will compare to each element in array_of_elems\n"
    ":type elem_to_compare: bytes\n"
    ":param max_dist: maximum allowable Hamming Distance\n"
    ":type max_dist: int\n"
    ":returns: tuple of best distance and index of first element in array_of_elems with that hamming distance, or -1,-1 if not found.\n"
    ":rtype: (int, int)\n"
    ":raises ValueError: if input parameters are invalid.";

static char check_bytes_arrays_all_within_dist_docstring[] =
    "Find all elements in bytes array that have a Hamming distance to elem less than max_dist\n"
    "and return a list of tuples of distance and index.\n\n"
    "Size of `elem_to_compare ` must be multiplier of `array_of_elems` size. \n\n"
    ":param array_of_elems: array of bytes to search within\n"
    ":type array_of_elems: bytes\n"
    ":param elem_to_compare: will compare to each element in array_of_elems\n"
    ":type elem_to_compare: bytes\n"
    ":param max_dist: maximum allowable Hamming Distance\n"
    ":type max_dist: int\n"
    ":returns: list of tuples of distance and index.\n"
    ":rtype: [(int, int)]\n"
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
    {"check_bytes_within_dist", check_bytes_within_dist_wrapper, METH_VARARGS, check_bytes_within_dist_docstring},
    {"check_bytes_arrays_first_within_dist", check_bytes_arrays_first_within_dist_wrapper, METH_VARARGS, check_bytes_arrays_first_within_dist_docstring},
    {"check_bytes_arrays_best_within_dist", check_bytes_arrays_best_within_dist_wrapper, METH_VARARGS, check_bytes_arrays_best_within_dist_docstring},
    {"check_bytes_arrays_all_within_dist", check_bytes_arrays_all_within_dist_wrapper, METH_VARARGS, check_bytes_arrays_all_within_dist_docstring},
    {"set_algo", set_algo_wrapper, METH_VARARGS, set_algo_docstring},
	/* Alias for backwards compatibility */
    {"check_bytes_arrays_within_dist", check_bytes_arrays_first_within_dist_wrapper, METH_VARARGS, check_bytes_arrays_first_within_dist_docstring},
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
        USE__EXTRA
     #endif
     snprintf(cpu_not_support_msg, sizeof(cpu_not_support_msg), "CPU doesnt support this feature. {%X}", cpu_capabilities);
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&hexhammingdef);
#else
    PyObject *module = Py_InitModule3("hexhamming", CompareMethods, CompareDocstring);
#endif
    if (PyModule_AddStringConstant(module, "__version__", _version)) {
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
