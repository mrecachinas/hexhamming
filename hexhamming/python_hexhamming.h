#ifndef HEXHAMMING_H
#define HEXHAMMING_H

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
    #endif
#endif

static inline size_t classic_popcnt64(size_t x) {
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

#if defined(CPU_X86_64) && (GNUC_PREREQ(4, 2) || CLANG_PREREQ(3, 0))
    #define HAVE_NATIVE_POPCNT
    static inline size_t native_popcnt64(size_t x) {
        __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
        return x;
    }
#elif defined(CPU_X86_64) && defined(_MSC_VER)
    #define HAVE_NATIVE_POPCNT
    #include <nmmintrin.h>
    static inline size_t native_popcnt64(size_t x) {
        return _mm_popcnt_u64(x);
    }
#elif GNUC_PREREQ(4, 2) || __has_builtin(__builtin_popcount)
    #define HAVE_NATIVE_POPCNT
    static inline size_t native_popcnt64(size_t x) {
        return __builtin_popcountll(x);
    }
#endif


/* hamming_distance_bytes__basic, hamming_distance_bytes__native, hamming_distance_bytes__extra:
   If max_dist < 0, then return hamming distance between arrays.
   If max_dist >= 0, then return 0 if difference bigger then max_dist, or 1 if difference less then max_dist. */

    /* Tier3(basic) functions. When nothing better is available. */
#ifdef CPU_X86_64
static const __m128i sse_popcount_mask = _mm_set1_epi8(0x0F);
static const __m128i sse_popcount_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
static inline __m128i sse_popcnt8(__m128i n) {
    const __m128i pcnt0 = _mm_shuffle_epi8(sse_popcount_table, _mm_and_si128(n, sse_popcount_mask));
    const __m128i pcnt1 = _mm_shuffle_epi8(sse_popcount_table, _mm_and_si128(_mm_srli_epi16(n, 4), sse_popcount_mask));
    return _mm_add_epi8(pcnt0, pcnt1);
}

static inline __m128i sse_popcnt64(__m128i n) {
    const __m128i cnt8 = sse_popcnt8(n);
    return _mm_sad_epu8(cnt8, _mm_setzero_si128());
}

static inline int sse_popcnt128(__m128i n) {
    // "Wojciech Mula" SSE3 version
    const __m128i cnt64 = sse_popcnt64(n);
    const __m128i cnt64_hi = _mm_unpackhi_epi64(cnt64, cnt64);
    const __m128i cnt128 = _mm_add_epi32(cnt64, cnt64_hi);
    return _mm_cvtsi128_si32(cnt128);
}

static inline int hamming_distance_bytes__basic(const char* a, const char* b, size_t length, ssize_t max_dist) {
    size_t difference = 0;
    size_t i = 0;
    __m128i xor_result;
    if (max_dist < 0)
    {
        if (length > 16)
            for (; i < length - length % 16; i += 16)
            {
                __m128i a16 = _mm_loadu_si128((__m128i *)&a[i]);
                __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]);
                xor_result = _mm_xor_si128(a16, b16);
                difference += sse_popcnt128(xor_result);
            }
        for (; i < length; i++)
            difference += classic_popcnt64(a[i] ^ b[i]);
        return difference;
    }
    else
    {
        if (length > 16)
            for (; i < length - length % 16; i += 16)
            {
                __m128i a16 = _mm_loadu_si128((__m128i *)&a[i]);
                __m128i b16 = _mm_loadu_si128((__m128i *)&b[i]);
                xor_result = _mm_xor_si128(a16, b16);
                difference += sse_popcnt128(xor_result);
                if (difference > (size_t)max_dist)
                    return 0;
            }
        for (; i < length; i++)
        {
            difference += classic_popcnt64(a[i] ^ b[i]);
            if (difference > (size_t)max_dist)
                return 0;
        }
        return 1;
    }
}
#else
static inline int hamming_distance_bytes__basic(const char* a, const char* b, size_t length, ssize_t max_dist) {
    size_t difference = 0;
    size_t i = 0;
    if (max_dist < 0)
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
                difference += classic_popcnt64(*(size_t*)(a + i) ^ *(size_t*)(b + i));
        for (; i < length; i++)
            difference += classic_popcnt64(a[i] ^ b[i]);
        return difference;
    }
    else
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
            {
                difference += classic_popcnt64(*(size_t*)(a + i) ^ *(size_t*)(b + i));
                if (difference > (size_t)max_dist)
                    return 0;
            }
        for (; i < length; i++)
        {
            difference += classic_popcnt64(a[i] ^ b[i]);
            if (difference > (size_t)max_dist)
                return 0;
        }
        return 1;
    }
}
#endif

    /* Tier2(native) Functions. When AVX/Neon better is not available. */
#ifdef HAVE_NATIVE_POPCNT
static inline int hamming_distance_bytes__native(const char* a, const char* b, size_t length, ssize_t max_dist) {
    size_t difference = 0;
    size_t i = 0;
    if (max_dist < 0)
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
                difference += native_popcnt64(*(size_t*)(a + i) ^ *(size_t*)(b + i));
        for (; i < length; i++)
            difference += native_popcnt64(a[i] ^ b[i]);
        return difference;
    }
    else
    {
        if (length > 8)
            for (; i < length - length % 8; i += 8)
            {
                difference += native_popcnt64(*(size_t*)(a + i) ^ *(size_t*)(b + i));
                if (difference > (size_t)max_dist)
                    return 0;
            }
        for (; i < length; i++)
        {
            difference += native_popcnt64(a[i] ^ b[i]);
            if (difference > (size_t)max_dist)
                return 0;
        }
        return 1;
    }
}
#else
static inline int hamming_distance_bytes__native(const char* a, const char* b, size_t length, ssize_t max_dist) {
    // We will never call this func. We check if CPU doesn't support popcount, then using Tier3 functions.
    return -1;
}
#endif

    /* Tier1(extra) Functions. When AVX/Neon  is available. */
#if defined(X64_EXTRA)
#include <immintrin.h>
#if !defined(_MSC_VER)
    __attribute__ ((target ("avx2")))
#endif
static inline int avx2_popcnt256(__m256i v) {
    __m256i lookup1 = _mm256_setr_epi8(
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8,
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8 );

    __m256i lookup2 = _mm256_setr_epi8(
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
static inline int hamming_distance_bytes__extra(const char* a, const char* b, size_t length, ssize_t max_dist) {
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
                difference += avx2_popcnt256(xor_result);
            }
        for (; i < length; i++)
            difference += native_popcnt64(a[i] ^ b[i]);
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
                difference += avx2_popcnt256(xor_result);
                if (difference > (size_t)max_dist)
                    return 0;
            }
        for (; i < length; i++)
        {
            difference += native_popcnt64(a[i] ^ b[i]);
            if (difference > (size_t)max_dist)
                return 0;
        }
        return 1;
    }
}
#elif defined(ARM_EXTRA)
static inline int hamming_distance_bytes__extra(const char* a, const char* b, size_t length, ssize_t max_dist) {
    // TODO.
    return -1;
}
#else
static inline int hamming_distance_bytes__extra(const char* a, const char* b, size_t length, ssize_t max_dist) {
    // We will never call this func. We check if CPU doesn't support AVX/Neon, then using Tier2(3) functions.
    return -1;
}
#endif

#endif  //HEXHAMMING_H