#ifndef HEXHAMMING_H
#define HEXHAMMING_H

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <array>
#ifdef __AVX__
	#include <x86intrin.h>
#endif

///////////////////////////////////////////////////////////////
// Structs and constants
///////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////
// C++ Functions - String Matching
///////////////////////////////////////////////////////////////
inline unsigned int hamming_distance(const std::string &a, const std::string &b) {
    if (a == b) {
      return 0;
    }

    size_t a_string_length = a.length();

    unsigned int result = 0;
    unsigned int val1, val2;
    for (size_t i = 0; i < a_string_length; ++i) {
        if ((a[i] > 'F' && a[i] < 'a') || (a[i] > 'f')) {
            throw std::invalid_argument(
              "hex string '" + a + "' contains invalid char"
            );
        }
        if ((b[i] > 'F' && b[i] < 'a') || (b[i] > 'f')) {
          throw std::invalid_argument(
            "hex string '" + b + "' contains invalid char"
          );
        }
        val1 = (a[i] > '9') ? (a[i] &~ 0x20) - 'A' + 10: (a[i] - '0');
        val2 = (b[i] > '9') ? (b[i] &~ 0x20) - 'A' + 10: (b[i] - '0');
        result += LOOKUP_MATRIX[val1][val2];
    }

    return result;
}

#endif
