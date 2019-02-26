#include "hexhamming.hh"


unsigned int hamming_distance_lookup(const std::string &a, const std::string &b) {
    if (a == b) {
      return 0;
    }

    size_t a_string_length = a.length();

    unsigned int result = 0;
    unsigned int val1, val2;
    for (size_t i = 0; i < a_string_length; ++i) {
        if ((a[i] > 'F' && a[i] < 'a') || (a[i] > 'f')) {
            throw std::invalid_argument("hex string contains invalid char");
        }
        if ((b[i] > 'F' && b[i] < 'a') || (b[i] > 'f')) {
            throw std::invalid_argument("hex string contains invalid char");
        }
        val1 = (a[i] > '9') ? (a[i] &~ 0x20) - 'A' + 10: (a[i] - '0');
        val2 = (b[i] > '9') ? (b[i] &~ 0x20) - 'A' + 10: (b[i] - '0');
        result += LOOKUP_MATRIX[val1][val2];
    }

    return result;
}

/**
 * Convert std::string to a vector of bools
 *
 * @param s     hex string - ['A'-'F''a'-'f''0'-'9']+
 * @return      int array containing 8 int32s
 * @throws      std::invalid_argument if string `s` is greater than 256
 * @throws      std::invalid_argument if string `s` contains non-hex characters
 */
std::array<int, 8> str_to_hex256(const std::string &s) {
    size_t string_length = s.length();
    printf("%s %d\n", __FILE__, __LINE__);
    // create an int array of size 8
    std::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    printf("%s %d\n", __FILE__, __LINE__);
    // iterate through the string and convert to binary
    for (int i = 0; i < string_length; ++i) {
        hex_byte hex_val;
        printf("%s %d %d\n", __FILE__, __LINE__, i);
        if ((s[i] > 'F' && s[i] < 'a') || (s[i] > 'f')) {
            throw std::invalid_argument("hex string contains invalid char");
        }
        printf("%s %d %d\n", __FILE__, __LINE__, i);
        hex_val.val = (s[i] > '9') ? (s[i] &~ 0x20) - 'A' + 10: (s[i] - '0');
        printf("%s %d %d\n", __FILE__, __LINE__, i);
        result[i >> 3] ^= ((hex_val.val & 8) ^ (hex_val.val & 4) ^ (hex_val.val & 2) ^ (hex_val.val & 1)) << (32 - 4 - (4 * (i % 8)));
        printf("%s %d %d\n", __FILE__, __LINE__, i);
    }
    printf("result = ");
    for (size_t tmp = 0; tmp < 8; ++tmp) {
        printf("%d, ", result[tmp]);
    }
    printf("\n");
    printf("%s %d\n", __FILE__, __LINE__);
    return result;
}


/**
 * Convert std::string to a vector of bools
 *
 * @param s     hex string - ['A'-'F''a'-'f''0'-'9']+
 * @return      vector of bools -- the binary representation of the hex string
 * @throws      std::invalid_argument if string `s` contains non-hex characters
 */
std::vector<bool> str_to_hex(const std::string &s) {
    // create a vector of size `bit_length`
    size_t string_length = s.length();
    size_t bit_length  = string_length * 4;
    std::vector<bool> hex(bit_length);

    // iterate through the string and convert to binary
    for (int i = 0; i < string_length; i++) {
        hex_byte hex_val;
        if ((s[i] > 'F' && s[i] < 'a') || (s[i] > 'f')) {
            throw std::invalid_argument("hex string contains invalid char");
        }
        hex_val.val = (s[i] > '9') ? (s[i] &~ 0x20) - 'A' + 10: (s[i] - '0');
        hex[i * 4] = hex_val.val & 8;
        hex[i * 4 + 1] = hex_val.val & 4;
        hex[i * 4 + 2] = hex_val.val & 2;
        hex[i * 4 + 3] = hex_val.val & 1;
    }
    return hex;
}

/**
 * XOR two vector of bools and sum the result (i.e., the differences)
 *
 * @param v1    vector of bools
 * @param s2    vector of bools
 * @return      number of signed bits in v1 ^ v2
 * @throws      std::logic_error if vectors are not same size
 */
int xor_vec(const std::vector<bool> &v1, const std::vector<bool> &v2) {
    if (v1.size() != v2.size()) {
        throw std::logic_error("vectors are not the same size");
    }

    unsigned int xor_val = 0;
    for (int i = 0; i < v1.size(); i++) {
        xor_val += v1[i] ^ v2[i];
    }
    return xor_val;
}

/**
 * Returns the hamming distance of the bits between two hex strings
 *
 * @param s1    hex-string
 * @param s2    hex-string
 * @return      the number of bits different between the hex
 */
unsigned int hamming_distance(const std::string &s1, const std::string &s2) {
    std::vector<bool> hex1 = str_to_hex(s1);
    std::vector<bool> hex2 = str_to_hex(s2);
    if (s1 == s2) {
        return 0;
    } else {
        return xor_vec(hex1, hex2);
    }
}


#ifdef __AVX__

/**
 * Returns the hamming distance of the bits between two hex strings
 *
 * Note: this version is SSE-optimized.
 *
 * @param s1    hex-string
 * @param s2    hex-string
 * @return      the number of bits different between the hex
 */
unsigned int hamming_distance_sse(const std::string &s1, const std::string &s2) {
    std::array<int, 8> a = str_to_hex256(s1);
    std::array<int, 8> b = str_to_hex256(s2);

    if (s1 == s2) {
        return 0;
    } else {
        __m256 a8 = _mm256_loadu_ps((float*)a.data());
        __m256 b8 = _mm256_loadu_ps((float*)b.data());
        __m256 c8 = _mm256_xor_ps(a8, b8);
        return __builtin_popcount(_mm256_extract_epi64(c8, 0)) +
              __builtin_popcount(_mm256_extract_epi64(c8, 1)) +
              __builtin_popcount(_mm256_extract_epi64(c8, 2)) +
              __builtin_popcount(_mm256_extract_epi64(c8, 3));
        // int c[8]; _mm256_storeu_ps((float*)c, c8);
    }
}

#endif
