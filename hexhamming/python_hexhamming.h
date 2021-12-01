#ifndef HEXHAMMING_H
#define HEXHAMMING_H

#if defined(_MSC_VER)
#include <cstdint>
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

#ifndef __has_attribute
    #define __has_attribute(x) 0
#endif

#ifdef __GNUC__
    #define GNUC_PREREQ(x, y) \
        (__GNUC__ > x || (__GNUC__ == x && __GNUC_MINOR__ >= y))
#else
    #define GNUC_PREREQ(x, y) 0
#endif

#ifdef __clang__
    #define CLANG_PREREQ(x, y) \
        (__clang_major__ > x || (__clang_major__ == x && __clang_minor__ >= y))
#else
    #define CLANG_PREREQ(x, y) 0
#endif

#if (defined(__x86_64__) || defined(_M_X64))
    #define CPU_X86_64

    #if GNUC_PREREQ(4, 9)
        #define HAVE_AVX2
    #endif
    #if defined(_MSC_VER) /* MSVC compatible compilers (Windows) */
        #if defined(__clang__) /* clang-cl (LLVM 10 from 2020) requires /arch:AVX2(512) to enable vector instructions */
            #if defined(__AVX2__)
                #define HAVE_AVX2
            #endif
            #if defined(__AVX512__)
                #define HAVE_AVX2
                #define HAVE_AVX512
            #endif
        #elif _MSC_VER >= 1910 /* MSVC 2017 or later does not require /arch:AVX2 or /arch:AVX512 */
            #define HAVE_AVX2
            #define HAVE_AVX512
        #endif
    #elif CLANG_PREREQ(3, 8) && __has_attribute(target) && \
          (!defined(__apple_build_version__) || __apple_build_version__ >= 8000000) /* Clang (Unix-like OSes) */
        #define HAVE_AVX2
        #define HAVE_AVX512
    #endif

    #if defined(_MSC_VER)
        #include <intrin.h>
        #include <immintrin.h>
    #endif

    // ecx flags
    #define bit_SSE41  (1 << 19)
    #define bit_POPCNT (1 << 23)
    // ebx flags
    #define bit_AVX2   (1 << 5)
    #define bit_AVX512 (1 << 30)
    // xgetbv bit flags
    #define XSTATE_SSE (1 << 1)
    #define XSTATE_YMM (1 << 2)
    #define XSTATE_ZMM (7 << 5)

    static inline void call_cpuid(int eax, int ecx, int* output)
    {
        #if defined(_MSC_VER)
            __cpuidex(output, eax, ecx);
        #else
            int ebx = 0;
            int edx = 0;
            __asm__ ("cpuid;" : "+b" (ebx), "+a" (eax), "+c" (ecx), "=d" (edx));
            output[0] = eax;
            output[1] = ebx;
            output[2] = ecx;
            output[3] = edx;
        #endif
    }

    static inline int get_cpuid()
    {
        int flags = 0;
        int cpu_flags[4];
        call_cpuid(1, 0, cpu_flags);
        if ((cpu_flags[2] & bit_POPCNT) == bit_POPCNT)
            flags |= bit_POPCNT;
        if ((cpu_flags[2] & bit_SSE41) == bit_SSE41)
            flags |= bit_SSE41;
        #if defined(HAVE_AVX2) || defined(HAVE_AVX512)
            if ((cpu_flags[2] & (1 << 27)) != (1 << 27)) //check if processor state management supported
                return flags;
            int xcr0;
            #if defined(_MSC_VER)
                xcr0 = (int) _xgetbv(0);
            #else
                __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
            #endif
            if ((xcr0 & (XSTATE_SSE | XSTATE_YMM)) == (XSTATE_SSE | XSTATE_YMM))
            {
                call_cpuid(7, 0, cpu_flags);
                if ((cpu_flags[1] & bit_AVX2) == bit_AVX2)
                    flags |= bit_AVX2;

                if ((xcr0 & (XSTATE_SSE | XSTATE_YMM | XSTATE_ZMM)) == (XSTATE_SSE | XSTATE_YMM | XSTATE_ZMM))
                {
                    if ((cpu_flags[1] & bit_AVX512) == bit_AVX512)
                        flags |= bit_AVX512;
                }
            }
        #endif
        return flags;
    }
    #if (defined(HAVE_AVX2) || defined(HAVE_AVX512))
        #define X64_EXTRA
    #endif
#else //non x86_64 CPU
    #if (defined(__ARM_NEON) || defined(__aarch64__))
        #define ARM_EXTRA
        #include <arm_neon.h>
    #endif
#endif


/* hamming_distance_bytes__classic, hamming_distance_bytes__native, hamming_distance_bytes__extra:
   If max_dist < 0, then return hamming distance between arrays.
   If max_dist >= 0, then return 0 if difference bigger then max_dist, or 1 if difference less then max_dist. */

/*------- arm7 or kvm64 -------*/
            /* BYTES */
static inline size_t popcnt64__classic(size_t x) {
    //http://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
    size_t m1 = 0x5555555555555555ll;
    size_t m2 = 0x3333333333333333ll;
    size_t m4 = 0x0F0F0F0F0F0F0F0Fll;
    size_t h01 = 0x0101010101010101ll;
    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return (x * h01) >> 56;
}

static int hamming_distance_bytes__classic(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
    size_t difference = 0;
    size_t i = 0;
    if (max_dist < 0)
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
                difference += popcnt64__classic(*(size_t*)(a + i) ^ *(size_t*)(b + i));
        for (; i < length; i++)
            difference += popcnt64__classic(a[i] ^ b[i]);
        return difference;
    }
    else
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
            {
                difference += popcnt64__classic(*(size_t*)(a + i) ^ *(size_t*)(b + i));
                if (difference > (size_t)max_dist)
                    return 0;
            }
        for (; i < length; i++)
        {
            difference += popcnt64__classic(a[i] ^ b[i]);
            if (difference > (size_t)max_dist)
                return 0;
        }
        return 1;
    }
}

            /* STRINGS */
/**
 * An array of size 16 containing the XOR result of
 * two numbers between 0 and 15 (i.e., '0' - 'F').
 */
static const unsigned char LOOKUP[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

inline int hamming_distance_loop_string(const char* a, const char* b, const size_t string_length) {
    int result = 0;
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
    }
    return result;
}

/**
 * Returns the hamming distance of the binary between two hexadecimal strings
 * of the same length.
 *
 * @param a    hexadecimal char array
 * @param b    hexadecimal char array
 * @param string_length length of `a` and `b`
 * @return      the number of bits different between the hexadecimal strings
 */
static int hamming_distance_string__classic(const char* a, const char* b, const size_t string_length) {
    return hamming_distance_loop_string(a, b, string_length);
}



/*------- SSE4.1 -------*/
#ifdef CPU_X86_64
            /* BYTES */
    static inline int popcnt128__sse(__m128i n) {
        const __m128i n_hi = _mm_unpackhi_epi64(n, n);
        return _mm_popcnt_u64(_mm_cvtsi128_si64(n)) + _mm_popcnt_u64(_mm_cvtsi128_si64(n_hi));
    }

    #define SSE_ITERATION { \
            const __m128i a16 = _mm_loadu_si128((__m128i *)&a[i]); \
            const __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]); \
            const __m128i xor_result = _mm_xor_si128(a16, b16); \
            const __m128i lo  = _mm_and_si128(xor_result, sse_popcount_mask); \
            const __m128i hi  = _mm_and_si128(_mm_srli_epi16(xor_result, 4), sse_popcount_mask); \
            const __m128i cnt_low = _mm_shuffle_epi8(sse_popcount_table, lo); \
            const __m128i cnt_high = _mm_shuffle_epi8(sse_popcount_table, hi); \
            local = _mm_add_epi8(local, cnt_low); \
            local = _mm_add_epi8(local, cnt_high); \
            i += 16; \
        }
    static int hamming_distance_bytes__sse(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
        size_t i = 0;
        size_t difference = 0;
        if (max_dist < 0)
        {
            if (length > 16)
            {
                const __m128i sse_popcount_mask = _mm_set1_epi8(0x0F);
                const __m128i sse_popcount_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
                __m128i sse_difference = _mm_setzero_si128();
                while (i + 16 * 4 <= length)
                {
                    __m128i local = _mm_setzero_si128();
                    SSE_ITERATION SSE_ITERATION SSE_ITERATION SSE_ITERATION
                    sse_difference = _mm_add_epi64(sse_difference, _mm_sad_epu8(local, _mm_setzero_si128()));
                }
                __m128i local = _mm_setzero_si128();
                while (i + 16 <= length)
                    SSE_ITERATION
                sse_difference = _mm_add_epi64(sse_difference, _mm_sad_epu8(local, _mm_setzero_si128()));
                difference = (uint64_t)(_mm_extract_epi64(sse_difference, 0));
                difference += (uint64_t)(_mm_extract_epi64(sse_difference, 1));
            }
            for (; i < length; i++)
                difference += popcnt64__classic(a[i] ^ b[i]);
            return difference;
        }
        else
        {
            if (length > 16)
                for (; i < length - length % 16; i += 16)
                {
                    const __m128i a16 = _mm_loadu_si128((__m128i *)&a[i]);
                    const __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]);
                    const __m128i xor_result = _mm_xor_si128(a16, b16);
                    difference += popcnt128__sse(xor_result);
                    if (difference > (size_t)max_dist)
                        return 0;
                }
            for (; i < length; i++)
            {
                difference += popcnt64__classic(a[i] ^ b[i]);
                if (difference > (size_t)max_dist)
                    return 0;
            }
            return 1;
        }
    }
    #undef SSE_ITERATION

            /* STRINGS */
    /**
     * SSE4.1 implementation of bitwise hamming distance of hex strings
     *
     * @param a    hexadecimal char array
     * @param b    hexadecimal char array
     * @param string_length length of `a` and `b` (MUST be a multiple of 16)
     * @return      the number of bits different between the hexadecimal strings
     */
    static inline int hamming_distance_sse41_string(const char* a, const char* b, const size_t string_length) {
        bool a_not_lt_0, a_not_gt_15, b_not_lt_0, b_not_gt_15;
        int result = 0;

        int fifteen_less = string_length - 15;

        __m128i zero = _mm_setzero_si128();
        __m128i fifteen = _mm_set1_epi8(15);
        __m128i xor_result, a_hex, b_hex;
        for (int i = 0; i < fifteen_less; i += 16) {
            // Check if greater than 15 or less than 0

            // load 16 chars from `a` and `b`
            __m128i a16 = _mm_loadu_si128((__m128i *)&a[i]);
            __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]);

            // Set up our masks for both branches of (x > '9') ? (x & ~0x20) - 55: x - '0'
            __m128i subtract0vec = _mm_set1_epi8('0');  // ['0', '0', ...]
            __m128i subtract55vec = _mm_set1_epi8(55);  // [55, 55, ...]
            __m128i andvec = _mm_set1_epi8(~0x20);  // [-33, -33, ...]
            __m128i isdigit_mask = _mm_set1_epi8('9');  // ['9', '9', ...]

            // Perform the x > '9' comparison
            __m128i a_cmp_mask = _mm_cmpgt_epi8(a16, isdigit_mask);
            __m128i b_cmp_mask = _mm_cmpgt_epi8(b16, isdigit_mask);

            // Compute for both branches

            // x > '9'

            // tmp = x & ~0x20
            __m128i a_letter = _mm_and_si128(a16, andvec);
            __m128i b_letter = _mm_and_si128(b16, andvec);

            // tmp - 55
            __m128i a_letter_normalized = _mm_sub_epi8(a_letter, subtract55vec);
            __m128i b_letter_normalized = _mm_sub_epi8(b_letter, subtract55vec);

            // x <= '9'

            // x - '0'
            __m128i a_digit_normalized = _mm_sub_epi8(a16, subtract0vec);
            __m128i b_digit_normalized = _mm_sub_epi8(b16, subtract0vec);

            // Put the ternary operator together
            // Note: if result in _mm_cmpgt_ps is a 1, it means it's > '9'
            a_hex = _mm_blendv_epi8(a_digit_normalized, a_letter_normalized, a_cmp_mask);
            b_hex = _mm_blendv_epi8(b_digit_normalized, b_letter_normalized, b_cmp_mask);

            // Greater than 15?
            __m128i a15 = _mm_cmpgt_epi8(a_hex, fifteen);
            __m128i b15 = _mm_cmpgt_epi8(b_hex, fifteen);

            // Less than 0?
            __m128i a0 = _mm_cmplt_epi16(a_hex, zero);
            __m128i b0 = _mm_cmplt_epi16(b_hex, zero);

            a_not_gt_15 = _mm_testz_si128(a15, a15);
            b_not_gt_15 = _mm_testz_si128(b15, b15);
            a_not_lt_0 = _mm_testz_si128(a0, a0);
            b_not_lt_0 = _mm_testz_si128(b0, b0);

            // If out of bounds, quit
            if (!(a_not_lt_0 && a_not_gt_15 && b_not_lt_0 && b_not_gt_15)) {
                return -1;
            }

            // Do the XOR
            xor_result = _mm_xor_si128(a_hex, b_hex);

            // Store the results
            result += popcnt128__sse(xor_result);
        }
        return result;
    }

    /**
     * Returns the hamming distance of the binary between two hexadecimal strings
     * of the same length.
     *
     * @param a    hexadecimal char array
     * @param b    hexadecimal char array
     * @param string_length length of `a` and `b`
     * @return      the number of bits different between the hexadecimal strings
     */
    static int hamming_distance_string__sse(const char* a, const char* b, const size_t string_length) {
        int result = hamming_distance_sse41_string(a, b, string_length);
        if (result == -1) {
            return result;
        }

        char mod16 = string_length & 15;
        if (mod16 != 0) {
            int fifteen_less = string_length - mod16;
            int start_index = fifteen_less >= 0 ? fifteen_less : 0;
            int dist = hamming_distance_loop_string(&a[start_index], &b[start_index], mod16);
            if (dist == -1) {
                return dist;
            } else {
                result += dist;
            }
        }
        return result;
    }
#endif



/*------- Native popcnt64 -------*/
#if defined(CPU_X86_64) && defined(_MSC_VER)
    #define HAVE_NATIVE_POPCNT
    #include <nmmintrin.h>
    static inline size_t popcnt64__native(size_t x) {
        return _mm_popcnt_u64(x);
    }
#elif GNUC_PREREQ(4, 2) || __has_builtin(__builtin_popcount)
    #define HAVE_NATIVE_POPCNT
    static inline size_t popcnt64__native(size_t x) {
        return __builtin_popcountll(x);
    }
#endif
#ifdef HAVE_NATIVE_POPCNT
    static int hamming_distance_bytes__native(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
        size_t difference = 0;
        size_t i = 0;
        if (max_dist < 0)
        {
            if (length > 8)
                for (; i < length - length % 8; i += 8)
                    difference += popcnt64__native(*(size_t*)(a + i) ^ *(size_t*)(b + i));
            for (; i < length; i++)
                difference += popcnt64__native(a[i] ^ b[i]);
            return difference;
        }
        else
        {
            if (length > 8)
                for (; i < length - length % 8; i += 8)
                {
                    difference += popcnt64__native(*(size_t*)(a + i) ^ *(size_t*)(b + i));
                    if (difference > (size_t)max_dist)
                        return 0;
                }
            for (; i < length; i++)
            {
                difference += popcnt64__native(a[i] ^ b[i]);
                if (difference > (size_t)max_dist)
                    return 0;
            }
            return 1;
        }
    }
#endif



/*------- AVX2 / Neon -------*/
#if defined(X64_EXTRA)
    #include <immintrin.h>
    #if !defined(_MSC_VER)
        __attribute__ ((target ("avx2")))
    #endif
    static inline int popcnt256__avx2(__m256i v) {
         const __m256i lookup1 = _mm256_setr_epi8(
            4, 5, 5, 6, 5, 6, 6, 7,
            5, 6, 6, 7, 6, 7, 7, 8,
            4, 5, 5, 6, 5, 6, 6, 7,
            5, 6, 6, 7, 6, 7, 7, 8 );

         const __m256i lookup2 = _mm256_setr_epi8(
            4, 3, 3, 2, 3, 2, 2, 1,
            3, 2, 2, 1, 2, 1, 1, 0,
            4, 3, 3, 2, 3, 2, 2, 1,
            3, 2, 2, 1, 2, 1, 1, 0 );

        __m256i low_mask = _mm256_set1_epi8(0x0f);
        __m256i lo = _mm256_and_si256(v, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
        __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);
        __m256i r = _mm256_sad_epu8(popcnt1, popcnt2);
        return _mm256_extract_epi64(r, 0) + _mm256_extract_epi64(r, 1) +\
                        _mm256_extract_epi64(r, 2) + _mm256_extract_epi64(r, 3);
    }
    #if !defined(_MSC_VER)
        __attribute__ ((target ("avx2")))
    #endif
    static int hamming_distance_bytes__extra(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
        size_t difference = 0;
        size_t i = 0;
        __m256i xor_result;
        if (max_dist < 0)
        {
            if (length > 32)
                for (; i < length - length % 32; i += 32)
                {
                    __m256i a32 = _mm256_loadu_si256((__m256i *)&a[i]);
                    __m256i b32 = _mm256_loadu_si256((__m256i *)&b[i]);
                    xor_result = _mm256_xor_si256(a32, b32);
                    difference += popcnt256__avx2(xor_result);
                }
            for (; i < length; i++)
                difference += popcnt64__native(a[i] ^ b[i]);
            return difference;
        }
        else
        {
            if (length > 32)
                for (; i < length - length % 32; i += 32)
                {
                    __m256i a32 = _mm256_loadu_si256((__m256i *)&a[i]);
                    __m256i b32 = _mm256_loadu_si256((__m256i *)&b[i]);
                    xor_result = _mm256_xor_si256(a32, b32);
                    difference += popcnt256__avx2(xor_result);
                    if (difference > (size_t)max_dist)
                        return 0;
                }
            for (; i < length; i++)
            {
                difference += popcnt64__native(a[i] ^ b[i]);
                if (difference > (size_t)max_dist)
                    return 0;
            }
            return 1;
        }
    }
#elif defined(ARM_EXTRA)
    static inline uint64x2_t vpadalq(uint64x2_t sum, uint8x16_t t) {
        return vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t)));
    }
    static int hamming_distance_bytes__extra(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
        return -1; // TODO.
    }
#else
    static int hamming_distance_bytes__extra(const uint8_t* a, const uint8_t* b, const size_t length, const ssize_t max_dist) {
        return -1;  // We will never call this func.
    }
#endif


//  MACROs for setting algorithm.
#define USE__EXTRA   ptr__hamming_distance_bytes = &hamming_distance_bytes__extra; \
                     ptr__hamming_distance_string = &hamming_distance_string__sse;


#if defined(CPU_X86_64)
#define USE__NATIVE  ptr__hamming_distance_bytes = &hamming_distance_bytes__native; \
                     ptr__hamming_distance_string = &hamming_distance_string__sse;
#else
#define USE__NATIVE  ptr__hamming_distance_bytes = &hamming_distance_bytes__native; \
                     ptr__hamming_distance_string = &hamming_distance_loop_string;
#endif


#define USE__SSE41   ptr__hamming_distance_bytes = &hamming_distance_bytes__sse; \
                     ptr__hamming_distance_string = &hamming_distance_string__sse;


#define USE__CLASSIC ptr__hamming_distance_bytes = &hamming_distance_bytes__classic; \
                     ptr__hamming_distance_string = &hamming_distance_loop_string;

#endif  //HEXHAMMING_H
