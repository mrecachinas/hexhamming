#ifndef HAMMING_H
#define HAMMING_H

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

// 4-bit struct for hex representation
struct hex_byte {
    unsigned val:4;
};

///////////////////////////////////////////////////////////////
// C++ Functions - String Matching
///////////////////////////////////////////////////////////////
std::array<int, 8> str_to_hex256(const std::string &s);
std::vector<bool> str_to_hex(const std::string &s);
int xor_vec(const std::vector<bool> &v1, const std::vector<bool> &v2);
unsigned int hamming_distance(const std::string &s1, const std::string &s2);
unsigned int hamming_distance_lookup(const std::string &s1, const std::string &s2);

#ifdef __AVX__
	unsigned int hamming_distance_sse(const std::string &s1, const std::string &s2);
#endif

#endif
