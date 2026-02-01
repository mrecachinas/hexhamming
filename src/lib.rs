//! Fast Hamming distance calculation for hexadecimal strings and byte arrays.
//!
//! This module provides blazingly fast bitwise Hamming distance calculation
//! using SIMD intrinsics where available.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Lookup table for popcount of 4-bit values (0-15)
const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Algorithm selection constants
const ALGO_CLASSIC: u8 = 0;
const ALGO_NATIVE: u8 = 1;
#[cfg(target_arch = "x86_64")]
const ALGO_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
const ALGO_AVX2: u8 = 3;
#[cfg(target_arch = "aarch64")]
const ALGO_NEON: u8 = 4;

/// Current algorithm selection (global state)
static CURRENT_ALGO: AtomicU8 = AtomicU8::new(ALGO_NATIVE);

/// Classic popcount implementation using bit manipulation
#[inline]
fn popcnt64_classic(mut x: u64) -> u64 {
    const M1: u64 = 0x5555555555555555;
    const M2: u64 = 0x3333333333333333;
    const M4: u64 = 0x0F0F0F0F0F0F0F0F;
    const H01: u64 = 0x0101010101010101;
    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    (x.wrapping_mul(H01)) >> 56
}

/// Native popcount implementation using CPU instruction
#[inline]
fn popcnt64_native(x: u64) -> u64 {
    x.count_ones() as u64
}

/// Convert a hex character to its numeric value (0-15)
/// Returns None if the character is not a valid hex digit
#[inline]
fn hex_char_to_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'F' => Some(c - b'A' + 10),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// Calculate hamming distance between two hex strings using classic algorithm
fn hamming_distance_string_classic(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let mut result: u64 = 0;
    for i in 0..a.len() {
        let val1 = hex_char_to_val(a[i]).ok_or("hex string contains invalid char")?;
        let val2 = hex_char_to_val(b[i]).ok_or("hex string contains invalid char")?;
        result += LOOKUP[(val1 ^ val2) as usize] as u64;
    }
    Ok(result)
}

/// Calculate hamming distance between two byte arrays using classic algorithm
fn hamming_distance_bytes_classic(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let mut difference: u64 = 0;
    let length = a.len();

    if max_dist < 0 {
        // Full distance calculation
        let mut i = 0;
        // Process 8 bytes at a time
        while i + 8 <= length {
            let a_chunk = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
            let b_chunk = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
            difference += popcnt64_classic(a_chunk ^ b_chunk);
            i += 8;
        }
        // Process remaining bytes
        while i < length {
            difference += popcnt64_classic((a[i] ^ b[i]) as u64);
            i += 1;
        }
        difference
    } else {
        // Early termination if exceeds max_dist
        let mut i = 0;
        while i + 8 <= length {
            let a_chunk = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
            let b_chunk = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
            difference += popcnt64_classic(a_chunk ^ b_chunk);
            if difference > max_dist as u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            difference += popcnt64_classic((a[i] ^ b[i]) as u64);
            if difference > max_dist as u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}

/// Calculate hamming distance between two byte arrays using native popcount
fn hamming_distance_bytes_native(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let mut difference: u64 = 0;
    let length = a.len();

    if max_dist < 0 {
        let mut i = 0;
        while i + 8 <= length {
            let a_chunk = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
            let b_chunk = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
            difference += popcnt64_native(a_chunk ^ b_chunk);
            i += 8;
        }
        while i < length {
            difference += popcnt64_native((a[i] ^ b[i]) as u64);
            i += 1;
        }
        difference
    } else {
        let mut i = 0;
        while i + 8 <= length {
            let a_chunk = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
            let b_chunk = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
            difference += popcnt64_native(a_chunk ^ b_chunk);
            if difference > max_dist as u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            difference += popcnt64_native((a[i] ^ b[i]) as u64);
            if difference > max_dist as u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}

// x86_64 SIMD implementations
#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use super::*;

    #[cfg(target_feature = "sse4.1")]
    use std::arch::x86_64::*;

    /// SSE4.1 popcount for 128-bit value
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    #[cfg(target_feature = "sse4.1")]
    pub unsafe fn popcnt128_sse(n: __m128i) -> i64 {
        let n_hi = _mm_unpackhi_epi64(n, n);
        _mm_popcnt_u64(_mm_cvtsi128_si64(n) as u64) as i64
            + _mm_popcnt_u64(_mm_cvtsi128_si64(n_hi) as u64) as i64
    }

    /// SSE4.1 implementation for byte arrays
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    #[cfg(target_feature = "sse4.1")]
    pub unsafe fn hamming_distance_bytes_sse(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();
        let mut difference: u64 = 0;
        let mut i = 0;

        let popcount_mask = _mm_set1_epi8(0x0F);
        let popcount_table =
            _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

        if max_dist < 0 {
            if length > 16 {
                let mut sse_difference = _mm_setzero_si128();

                // Process 64 bytes at a time (4 iterations of 16 bytes)
                while i + 64 <= length {
                    let mut local = _mm_setzero_si128();
                    for _ in 0..4 {
                        let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                        let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                        let xor_result = _mm_xor_si128(a16, b16);
                        let lo = _mm_and_si128(xor_result, popcount_mask);
                        let hi = _mm_and_si128(_mm_srli_epi16(xor_result, 4), popcount_mask);
                        let cnt_low = _mm_shuffle_epi8(popcount_table, lo);
                        let cnt_high = _mm_shuffle_epi8(popcount_table, hi);
                        local = _mm_add_epi8(local, cnt_low);
                        local = _mm_add_epi8(local, cnt_high);
                        i += 16;
                    }
                    sse_difference =
                        _mm_add_epi64(sse_difference, _mm_sad_epu8(local, _mm_setzero_si128()));
                }

                // Process remaining 16-byte chunks
                let mut local = _mm_setzero_si128();
                while i + 16 <= length {
                    let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                    let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                    let xor_result = _mm_xor_si128(a16, b16);
                    let lo = _mm_and_si128(xor_result, popcount_mask);
                    let hi = _mm_and_si128(_mm_srli_epi16(xor_result, 4), popcount_mask);
                    let cnt_low = _mm_shuffle_epi8(popcount_table, lo);
                    let cnt_high = _mm_shuffle_epi8(popcount_table, hi);
                    local = _mm_add_epi8(local, cnt_low);
                    local = _mm_add_epi8(local, cnt_high);
                    i += 16;
                }
                sse_difference =
                    _mm_add_epi64(sse_difference, _mm_sad_epu8(local, _mm_setzero_si128()));

                difference = (_mm_extract_epi64(sse_difference, 0) as u64)
                    + (_mm_extract_epi64(sse_difference, 1) as u64);
            }

            // Process remaining bytes
            while i < length {
                difference += popcnt64_classic((a[i] ^ b[i]) as u64);
                i += 1;
            }
            difference
        } else {
            // With max_dist check
            if length > 16 {
                while i + 16 <= length {
                    let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                    let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                    let xor_result = _mm_xor_si128(a16, b16);
                    difference += popcnt128_sse(xor_result) as u64;
                    if difference > max_dist as u64 {
                        return 0;
                    }
                    i += 16;
                }
            }

            while i < length {
                difference += popcnt64_classic((a[i] ^ b[i]) as u64);
                if difference > max_dist as u64 {
                    return 0;
                }
                i += 1;
            }
            1
        }
    }

    /// AVX2 popcount for 256-bit value
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn popcnt256_avx2(v: __m256i) -> u64 {
        let lookup1 = _mm256_setr_epi8(
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
            7, 7, 8,
        );
        let lookup2 = _mm256_setr_epi8(
            4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2,
            1, 1, 0,
        );

        let low_mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
        let popcnt2 = _mm256_shuffle_epi8(lookup2, hi);
        let r = _mm256_sad_epu8(popcnt1, popcnt2);
        (_mm256_extract_epi64(r, 0) as u64)
            + (_mm256_extract_epi64(r, 1) as u64)
            + (_mm256_extract_epi64(r, 2) as u64)
            + (_mm256_extract_epi64(r, 3) as u64)
    }

    /// AVX2 implementation for byte arrays
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();
        let mut difference: u64 = 0;
        let mut i = 0;

        if max_dist < 0 {
            while i + 32 <= length {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                difference += popcnt256_avx2(_mm256_xor_si256(a32, b32));
                i += 32;
            }
            while i < length {
                difference += popcnt64_native((a[i] ^ b[i]) as u64);
                i += 1;
            }
            difference
        } else {
            while i + 32 <= length {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                difference += popcnt256_avx2(_mm256_xor_si256(a32, b32));
                if difference > max_dist as u64 {
                    return 0;
                }
                i += 32;
            }
            while i < length {
                difference += popcnt64_native((a[i] ^ b[i]) as u64);
                if difference > max_dist as u64 {
                    return 0;
                }
                i += 1;
            }
            1
        }
    }

    /// SSE4.1 implementation for hex strings
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    #[cfg(target_feature = "sse4.1")]
    pub unsafe fn hamming_distance_string_sse(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();
        let mut result: u64 = 0;

        let zero = _mm_setzero_si128();
        let fifteen = _mm_set1_epi8(15);
        let subtract0vec = _mm_set1_epi8(b'0' as i8);
        let subtract55vec = _mm_set1_epi8(55);
        let andvec = _mm_set1_epi8(!0x20i8);
        let isdigit_mask = _mm_set1_epi8(b'9' as i8);

        let fifteen_less = length.saturating_sub(15);
        let mut i = 0;

        while i < fifteen_less {
            let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);

            // x > '9' comparison
            let a_cmp_mask = _mm_cmpgt_epi8(a16, isdigit_mask);
            let b_cmp_mask = _mm_cmpgt_epi8(b16, isdigit_mask);

            // For x > '9': (x & ~0x20) - 55
            let a_letter = _mm_and_si128(a16, andvec);
            let b_letter = _mm_and_si128(b16, andvec);
            let a_letter_normalized = _mm_sub_epi8(a_letter, subtract55vec);
            let b_letter_normalized = _mm_sub_epi8(b_letter, subtract55vec);

            // For x <= '9': x - '0'
            let a_digit_normalized = _mm_sub_epi8(a16, subtract0vec);
            let b_digit_normalized = _mm_sub_epi8(b16, subtract0vec);

            // Blend based on comparison
            let a_hex = _mm_blendv_epi8(a_digit_normalized, a_letter_normalized, a_cmp_mask);
            let b_hex = _mm_blendv_epi8(b_digit_normalized, b_letter_normalized, b_cmp_mask);

            // Check bounds
            let a15 = _mm_cmpgt_epi8(a_hex, fifteen);
            let b15 = _mm_cmpgt_epi8(b_hex, fifteen);
            let a0 = _mm_cmplt_epi16(a_hex, zero);
            let b0 = _mm_cmplt_epi16(b_hex, zero);

            if !(_mm_testz_si128(a15, a15) != 0
                && _mm_testz_si128(b15, b15) != 0
                && _mm_testz_si128(a0, a0) != 0
                && _mm_testz_si128(b0, b0) != 0)
            {
                return Err("hex string contains invalid char");
            }

            // XOR and popcount
            let xor_result = _mm_xor_si128(a_hex, b_hex);
            result += popcnt128_sse(xor_result) as u64;

            i += 16;
        }

        // Handle remaining bytes with scalar code
        let remaining = length & 15;
        if remaining != 0 {
            let start_index = if fifteen_less > 0 { fifteen_less } else { 0 };
            for j in start_index..length {
                let val1 = hex_char_to_val(a[j]).ok_or("hex string contains invalid char")?;
                let val2 = hex_char_to_val(b[j]).ok_or("hex string contains invalid char")?;
                result += LOOKUP[(val1 ^ val2) as usize] as u64;
            }
        }

        Ok(result)
    }
}

// ARM NEON implementations
#[cfg(target_arch = "aarch64")]
mod arm_simd {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON implementation for byte arrays
    #[target_feature(enable = "neon")]
    pub unsafe fn hamming_distance_bytes_neon(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();

        // For early termination, use native implementation (faster on ARM)
        if max_dist >= 0 {
            return hamming_distance_bytes_native(a, b, max_dist);
        }

        let mut difference: u64 = 0;
        let mut i = 0;
        let total_iters = length / 64;

        if total_iters >= 1 {
            let mut sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
            let zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
            let mut current_iter = 0;

            while current_iter < total_iters {
                let mut t0 = zero;
                let mut t1 = zero;
                let mut t2 = zero;
                let mut t3 = zero;

                let iter_limit = std::cmp::min(current_iter + 31, total_iters);

                while current_iter < iter_limit {
                    let input_a = vld4q_u8(a.as_ptr().add(i));
                    let input_b = vld4q_u8(b.as_ptr().add(i));
                    i += 64;

                    t0 = vaddq_u8(t0, vcntq_u8(veorq_u8(input_a.0, input_b.0)));
                    t1 = vaddq_u8(t1, vcntq_u8(veorq_u8(input_a.1, input_b.1)));
                    t2 = vaddq_u8(t2, vcntq_u8(veorq_u8(input_a.2, input_b.2)));
                    t3 = vaddq_u8(t3, vcntq_u8(veorq_u8(input_a.3, input_b.3)));

                    current_iter += 1;
                }

                // Accumulate results
                sum = vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t0)));
                sum = vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t1)));
                sum = vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t2)));
                sum = vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t3)));
            }

            let mut tmp = [0u64; 2];
            vst1q_u64(tmp.as_mut_ptr(), sum);
            difference += tmp[0] + tmp[1];
        }

        // Process remaining bytes
        while i < length {
            difference += popcnt64_native((a[i] ^ b[i]) as u64);
            i += 1;
        }

        difference
    }
}

/// Dispatch to appropriate byte distance implementation based on current algorithm
fn hamming_distance_bytes_dispatch(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => hamming_distance_bytes_classic(a, b, max_dist),

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX2 => {
            if is_x86_feature_detected!("avx2") {
                unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
            } else {
                hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_SSE41 => {
            if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
                unsafe { x86_simd::hamming_distance_bytes_sse(a, b, max_dist) }
            } else {
                hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => unsafe { arm_simd::hamming_distance_bytes_neon(a, b, max_dist) },

        _ => hamming_distance_bytes_native(a, b, max_dist),
    }
}

/// Dispatch to appropriate string distance implementation based on current algorithm
fn hamming_distance_string_dispatch(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        if (algo == ALGO_AVX2 || algo == ALGO_SSE41 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_sse(a, b) };
        }
    }

    hamming_distance_string_classic(a, b)
}

/// Calculate the hamming distance of two hexadecimal strings
///
/// This is equivalent to `bin(int(a, 16) ^ int(b, 16)).count('1')`
/// but optimized using SIMD instructions where available.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_string(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Extract strings with proper error handling
    let a_str: &str = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_str: &str = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_str.len() != b_str.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }

    if a_str.is_empty() {
        return Ok(0);
    }

    hamming_distance_string_dispatch(a_str.as_bytes(), b_str.as_bytes())
        .map_err(PyValueError::new_err)
}

/// Calculate the hamming distance of two byte arrays
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_bytes(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Extract bytes with proper error handling
    let a_bytes: &[u8] = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_bytes: &[u8] = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_bytes.len() != b_bytes.len() {
        return Err(PyValueError::new_err("bytes are NOT the same length"));
    }

    if a_bytes.is_empty() {
        return Ok(0);
    }

    Ok(hamming_distance_bytes_dispatch(a_bytes, b_bytes, -1))
}

/// Check if two hex strings are within a specified Hamming distance
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_hexstrings_within_dist(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // Extract strings with proper error handling
    let a_str: &str = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_str: &str = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    // Extract max_dist - need to handle negative numbers
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >0"));
    }

    let max_dist_u64 = max_dist_val as u64;

    if a_str.len() != b_str.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }

    if a_str == b_str {
        return Ok(true);
    }

    if max_dist_u64 > a_str.len() as u64 {
        return Ok(true);
    }

    let a_bytes = a_str.as_bytes();
    let b_bytes = b_str.as_bytes();

    let mut result: i64 = 0;
    for i in 0..a_bytes.len() {
        let val1 = hex_char_to_val(a_bytes[i]).ok_or_else(|| {
            PyValueError::new_err("hex string contains invalid char")
        })?;
        let val2 = hex_char_to_val(b_bytes[i]).ok_or_else(|| {
            PyValueError::new_err("hex string contains invalid char")
        })?;

        result += LOOKUP[(val1 ^ val2) as usize] as i64;
        if result > max_dist_val {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Check if any element of byte array is within a specified Hamming Distance
/// and return its index or -1 otherwise.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_within_dist(
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    // Extract bytes with proper error handling
    let big_array: &[u8] = array_of_elems.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let small_array: &[u8] = elem_to_compare.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if small_array.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }

    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }

    if big_array.len() % small_array.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;

    for i in 0..num_elements {
        let start = i * elem_size;
        let end = start + elem_size;
        let chunk = &big_array[start..end];

        let res = hamming_distance_bytes_dispatch(chunk, small_array, max_dist_val);
        if res == 1 {
            return Ok(i as i64);
        }
    }

    Ok(-1)
}

/// Change algorithm used for calculations
/// Returns empty string if successful, or error message otherwise
#[pyfunction]
fn set_algo(algo_name: &str) -> PyResult<String> {
    match algo_name.to_lowercase().as_str() {
        "extra" | "avx" | "avx2" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    CURRENT_ALGO.store(ALGO_AVX2, Ordering::Relaxed);
                    return Ok(String::new());
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                return Ok(String::new());
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Ok("CPU doesn't support this feature".to_string());
            }
            #[cfg(target_arch = "x86_64")]
            Ok("CPU doesn't support this feature".to_string())
        }

        "native" | "popcount" => {
            CURRENT_ALGO.store(ALGO_NATIVE, Ordering::Relaxed);
            Ok(String::new())
        }

        "sse41" | "sse" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.1") {
                    CURRENT_ALGO.store(ALGO_SSE41, Ordering::Relaxed);
                    return Ok(String::new());
                }
                Ok("CPU doesn't support this feature".to_string())
            }
            #[cfg(not(target_arch = "x86_64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

        "neon" => {
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                Ok(String::new())
            }
            #[cfg(not(target_arch = "aarch64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

        "classic" => {
            CURRENT_ALGO.store(ALGO_CLASSIC, Ordering::Relaxed);
            Ok(String::new())
        }

        _ => Ok("Library was built without this algorithm.".to_string()),
    }
}

/// Module for calculating hamming distance of two hexadecimal strings
#[pymodule]
fn hexhamming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "2.2.3")?;
    m.add_function(wrap_pyfunction!(hamming_distance_string, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(check_hexstrings_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(set_algo, m)?)?;

    // Auto-detect best algorithm on module load
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            CURRENT_ALGO.store(ALGO_AVX2, Ordering::Relaxed);
        } else if is_x86_feature_detected!("sse4.1") {
            CURRENT_ALGO.store(ALGO_SSE41, Ordering::Relaxed);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
    }

    Ok(())
}
