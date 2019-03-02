#ifndef HEXHAMMING_H
#define HEXHAMMING_H

#include <stdlib.h>
#include <string.h>

/**
 * A 16-by-16 matrix containing the hamming distances of
 * every hex character against every other hex character.
 */
const int LOOKUP_MATRIX[16][16] = {
  { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 },
  { 1, 0, 2, 1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 4, 3 },
  { 1, 2, 0, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3 },
  { 2, 1, 1, 0, 3, 2, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2 },
  { 1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 4, 1, 2, 2, 3 },
  { 2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2 },
  { 2, 3, 1, 2, 1, 2, 0, 1, 3, 4, 2, 3, 2, 3, 1, 2 },
  { 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1 },
  { 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3 },
  { 2, 1, 3, 2, 3, 2, 4, 3, 1, 0, 2, 1, 2, 1, 3, 2 },
  { 2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2 },
  { 3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1 },
  { 2, 3, 3, 4, 1, 2, 2, 3, 1, 2, 2, 3, 0, 1, 1, 2 },
  { 3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1 },
  { 3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1 },
  { 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0 }
};

/**
 * Returns the hamming distance of the binary between two hexadecimal strings
 *
 * @param a    hexadecimal char array
 * @param b    hexadecimal char array
 * @param a_string_length length of `a`
 * @param b_string_length length of `b`
 * @return      the number of bits different between the hexadecimal strings
 */
inline int hamming_distance(
		const char* a,
		const char* b,
		size_t a_string_length,
		size_t b_string_length
	) {
		// if both strings are the same, short circuit
		// and return 0
    if (strcmp(a, b) == 0) {
      return 0;
    }

    int result = 0;
    int val1, val2;
    size_t i;
    for (i = 0; i < a_string_length; ++i) {
				// check to make sure all characters are valid
				// hexadecimal in both strings
        if ((a[i] > 'F' && a[i] < 'a') || (a[i] > 'f') ||
				    (b[i] > 'F' && b[i] < 'a') || (b[i] > 'f')) {
            return -1;
        }

				// Convert the hex ascii char to its actual hexadecimal value
				// e.g., '0' = 0, 'A' = 10, etc.
				// Note: this is case INSENSITIVE
        val1 = (a[i] > '9') ? (a[i] &~ 0x20) - 'A' + 10: (a[i] - '0');
        val2 = (b[i] > '9') ? (b[i] &~ 0x20) - 'A' + 10: (b[i] - '0');

        result += LOOKUP_MATRIX[val1][val2];
    }

    return result;
}

#endif
