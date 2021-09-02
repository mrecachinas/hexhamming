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
        return flags;
    }
#endif

#if GNUC_PREREQ(4, 2) || __has_builtin(__builtin_popcount)
    #define HAVE_BUILTIN_POPCOUNT
#endif

#if GNUC_PREREQ(4, 2) || CLANG_PREREQ(3, 0)
    #define HAVE_ASM_POPCNT
#endif

#if defined(HAVE_ASM_POPCNT)
    #define NATIVE_POPCNT
    static inline size_t native_popcnt64(size_t x)
    {
        __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
        return x;
    }
#elif defined(_MSC_VER)
    #define NATIVE_POPCNT
    #include <nmmintrin.h>
    static inline size_t native_popcnt64(size_t x)
    {
        return _mm_popcnt_u64(x);
    }
#elif defined(HAVE_BUILTIN_POPCOUNT)
    #define NATIVE_POPCNT
    static inline size_t native_popcnt64(size_t x)
    {
        return __builtin_popcountll(x);
    }
#endif

static inline size_t classic_popcnt64(size_t x)
{
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
