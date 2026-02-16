//! Fast Hamming distance calculation for hexadecimal strings and byte arrays.
//!
//! This module provides blazingly fast bitwise Hamming distance calculation
//! using SIMD intrinsics where available.
//!
//! # Optimizations
//! - Branchless hex parsing with 256-byte lookup table
//! - SIMD vectorized processing (AVX2/SSE4.1/NEON)
//! - Batched horizontal summation to minimize lane reductions
//! - Unsafe bounds elimination in hot loops
//! - Algorithm selection based on input size thresholds

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Lookup table for popcount of 4-bit values (0-15)
const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Branchless hex character to nibble lookup table (256 entries)
/// Invalid characters map to 0xFF for easy detection
const HEX_LOOKUP: [u8; 256] = {
    let mut table = [0xFFu8; 256];
    let mut i = 0u8;
    loop {
        table[i as usize] = match i {
            b'0'..=b'9' => i - b'0',
            b'A'..=b'F' => i - b'A' + 10,
            b'a'..=b'f' => i - b'a' + 10,
            _ => 0xFF,
        };
        if i == 255 { break; }
        i += 1;
    }
    table
};

/// Algorithm selection constants
const ALGO_CLASSIC: u8 = 0;
const ALGO_NATIVE: u8 = 1;
#[cfg(target_arch = "x86_64")]
const ALGO_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
const ALGO_AVX2: u8 = 3;
#[cfg(target_arch = "aarch64")]
const ALGO_NEON: u8 = 4;

/// Thresholds for algorithm selection (tuned for typical CPU cache behavior)
#[allow(dead_code)]
const SCALAR_THRESHOLD: usize = 16; // Below this, scalar may beat SIMD
#[allow(dead_code)]
const SSE_THRESHOLD: usize = 64;    // Use SSE for medium strings

/// Current algorithm selection (global state)
static CURRENT_ALGO: AtomicU8 = AtomicU8::new(ALGO_NATIVE);

/// Classic popcount implementation using bit manipulation (Wilkes-Wheeler-Gill)
#[inline(always)]
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
#[inline(always)]
fn popcnt64_native(x: u64) -> u64 {
    x.count_ones() as u64
}

/// Branchless hex character to nibble conversion using lookup table
/// Returns 0xFF for invalid characters
#[inline(always)]
fn hex_char_to_nibble(c: u8) -> u8 {
    // SAFETY: c is u8, so always in bounds of 256-element table
    unsafe { *HEX_LOOKUP.get_unchecked(c as usize) }
}

/// Convert a hex character to its numeric value (0-15)
/// Returns None if the character is not a valid hex digit
#[inline(always)]
#[allow(dead_code)]
fn hex_char_to_val(c: u8) -> Option<u8> {
    let val = hex_char_to_nibble(c);
    if val == 0xFF { None } else { Some(val) }
}

/// Calculate hamming distance between two hex strings using classic algorithm
/// Optimized with branchless lookup and bounds check elimination
#[inline(always)]
fn hamming_distance_string_classic(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let len = a.len();
    let mut result: u64 = 0;
    let mut i = 0;
    
    // Process 4 hex chars at a time to reduce loop overhead
    while i + 4 <= len {
        // SAFETY: i + 3 < len verified by loop condition
        unsafe {
            let val1_0 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2_0 = hex_char_to_nibble(*b.get_unchecked(i));
            let val1_1 = hex_char_to_nibble(*a.get_unchecked(i + 1));
            let val2_1 = hex_char_to_nibble(*b.get_unchecked(i + 1));
            let val1_2 = hex_char_to_nibble(*a.get_unchecked(i + 2));
            let val2_2 = hex_char_to_nibble(*b.get_unchecked(i + 2));
            let val1_3 = hex_char_to_nibble(*a.get_unchecked(i + 3));
            let val2_3 = hex_char_to_nibble(*b.get_unchecked(i + 3));
            
            // Check all 8 values for validity (0xFF indicates invalid)
            // Use bitwise OR to combine checks - any 0xFF will result in high bit set
            let invalid = (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
            if invalid != 0 {
                return Err("hex string contains invalid char");
            }
            
            result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
        }
        i += 4;
    }
    
    // Handle remaining characters
    while i < len {
        // SAFETY: i < len verified by loop condition
        unsafe {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        }
        i += 1;
    }
    
    Ok(result)
}

/// Calculate hamming distance between two byte arrays using classic algorithm
/// Optimized with loop unrolling and bounds check elimination
#[inline(always)]
fn hamming_distance_bytes_classic(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();

    if max_dist < 0 {
        // Full distance calculation - heavily optimized
        let mut difference: u64 = 0;
        let mut i = 0;
        
        // Process 32 bytes at a time (4 x 8-byte chunks)
        while i + 32 <= length {
            // SAFETY: i + 31 < length verified by loop condition
            unsafe {
                let a0 = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b0 = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                let a1 = u64::from_ne_bytes(*(a.as_ptr().add(i + 8) as *const [u8; 8]));
                let b1 = u64::from_ne_bytes(*(b.as_ptr().add(i + 8) as *const [u8; 8]));
                let a2 = u64::from_ne_bytes(*(a.as_ptr().add(i + 16) as *const [u8; 8]));
                let b2 = u64::from_ne_bytes(*(b.as_ptr().add(i + 16) as *const [u8; 8]));
                let a3 = u64::from_ne_bytes(*(a.as_ptr().add(i + 24) as *const [u8; 8]));
                let b3 = u64::from_ne_bytes(*(b.as_ptr().add(i + 24) as *const [u8; 8]));
                
                difference += popcnt64_classic(a0 ^ b0)
                           + popcnt64_classic(a1 ^ b1)
                           + popcnt64_classic(a2 ^ b2)
                           + popcnt64_classic(a3 ^ b3);
            }
            i += 32;
        }
        
        // Process remaining 8-byte chunks
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_classic(a_chunk ^ b_chunk);
            }
            i += 8;
        }
        
        // Process remaining bytes
        while i < length {
            unsafe {
                difference += popcnt64_classic((*a.get_unchecked(i) ^ *b.get_unchecked(i)) as u64);
            }
            i += 1;
        }
        difference
    } else {
        // Early termination if exceeds max_dist
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;
        let mut i = 0;
        
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_classic(a_chunk ^ b_chunk);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            unsafe {
                difference += popcnt64_classic((*a.get_unchecked(i) ^ *b.get_unchecked(i)) as u64);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}

/// Calculate hamming distance between two byte arrays using native popcount
/// Optimized with aggressive loop unrolling and bounds check elimination
#[inline(always)]
fn hamming_distance_bytes_native(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();

    if max_dist < 0 {
        let mut difference: u64 = 0;
        let mut i = 0;
        
        // Process 32 bytes at a time (4 x 8-byte chunks) to saturate execution units
        while i + 32 <= length {
            unsafe {
                let a0 = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b0 = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                let a1 = u64::from_ne_bytes(*(a.as_ptr().add(i + 8) as *const [u8; 8]));
                let b1 = u64::from_ne_bytes(*(b.as_ptr().add(i + 8) as *const [u8; 8]));
                let a2 = u64::from_ne_bytes(*(a.as_ptr().add(i + 16) as *const [u8; 8]));
                let b2 = u64::from_ne_bytes(*(b.as_ptr().add(i + 16) as *const [u8; 8]));
                let a3 = u64::from_ne_bytes(*(a.as_ptr().add(i + 24) as *const [u8; 8]));
                let b3 = u64::from_ne_bytes(*(b.as_ptr().add(i + 24) as *const [u8; 8]));
                
                difference += popcnt64_native(a0 ^ b0)
                           + popcnt64_native(a1 ^ b1)
                           + popcnt64_native(a2 ^ b2)
                           + popcnt64_native(a3 ^ b3);
            }
            i += 32;
        }
        
        // Process remaining 8-byte chunks
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_native(a_chunk ^ b_chunk);
            }
            i += 8;
        }
        
        // Process remaining bytes
        while i < length {
            unsafe {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            }
            i += 1;
        }
        difference
    } else {
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;
        let mut i = 0;
        
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_native(a_chunk ^ b_chunk);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            unsafe {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}

// x86_64 SIMD implementations
// Note: Unlike ARM64, x86 SSE/AVX2 implementations use VPSHUFB-based popcount which
// processes 16/32 bytes in parallel with a lookup table. This is typically faster than
// scalar count_ones() because it avoids horizontal reduction overhead. The ARM64 native
// approach works better there because the CNT instruction handles accumulation efficiently.
#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use super::*;
    #[allow(unused_imports)]
    use std::arch::x86_64::*;

    /// SSE4.1 popcount for 128-bit value using hardware popcnt
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    pub unsafe fn popcnt128_sse(n: __m128i) -> u64 {
        let lo = _mm_cvtsi128_si64(n) as u64;
        let hi = _mm_extract_epi64(n, 1) as u64;
        lo.count_ones() as u64 + hi.count_ones() as u64
    }

    /// SSE4.1 VPSHUFB-based popcount (accumulates into byte lanes)
    /// Returns vector with per-byte popcounts - use _mm_sad_epu8 to sum
    #[target_feature(enable = "ssse3")]
    unsafe fn popcnt128_shuffle(v: __m128i, mask: __m128i, table: __m128i) -> __m128i {
        let lo = _mm_and_si128(v, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        _mm_add_epi8(_mm_shuffle_epi8(table, lo), _mm_shuffle_epi8(table, hi))
    }

    /// SSE4.1 implementation for byte arrays - heavily optimized
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    pub unsafe fn hamming_distance_bytes_sse(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();
        let mut i = 0;

        // For small inputs, use scalar (SIMD setup overhead not worth it)
        if length < SCALAR_THRESHOLD {
            return hamming_distance_bytes_native(a, b, max_dist);
        }

        // VPSHUFB lookup table for 4-bit popcount
        let mask = _mm_set1_epi8(0x0F);
        let table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

        if max_dist < 0 {
            let mut total = _mm_setzero_si128();
            
            // Process 256 bytes at a time (16 x 16 bytes) before horizontal sum
            // This maximizes throughput by keeping counts in u8 lanes (max 255 per lane)
            // 16 iterations x 16 bytes x max 8 bits = max 2048 bits, but per-lane max is 16*8=128 < 255
            while i + 256 <= length {
                let mut acc = _mm_setzero_si128();
                for _ in 0..16 {
                    let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                    let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                    let xor = _mm_xor_si128(a16, b16);
                    acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                    i += 16;
                }
                total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));
            }

            // Process 64 bytes at a time (4 x 16 bytes)
            while i + 64 <= length {
                let mut acc = _mm_setzero_si128();
                for _ in 0..4 {
                    let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                    let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                    let xor = _mm_xor_si128(a16, b16);
                    acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                    i += 16;
                }
                total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));
            }

            // Process remaining 16-byte chunks
            let mut acc = _mm_setzero_si128();
            while i + 16 <= length {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let xor = _mm_xor_si128(a16, b16);
                acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                i += 16;
            }
            total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));

            // Extract final sum
            let mut difference = (_mm_extract_epi64(total, 0) + _mm_extract_epi64(total, 1)) as u64;

            // Process remaining bytes with native popcnt
            while i < length {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
                i += 1;
            }
            difference
        } else {
            // Early termination path - check every 16 bytes
            let max_dist_u64 = max_dist as u64;
            let mut difference: u64 = 0;

            while i + 16 <= length {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let xor = _mm_xor_si128(a16, b16);
                difference += popcnt128_sse(xor);
                if difference > max_dist_u64 {
                    return 0;
                }
                i += 16;
            }

            while i < length {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
                i += 1;
            }
            1
        }
    }

    /// AVX2 VPSHUFB-based popcount - accumulates into byte lanes
    #[target_feature(enable = "avx2")]
    unsafe fn popcnt256_shuffle(v: __m256i, mask: __m256i, table: __m256i) -> __m256i {
        let lo = _mm256_and_si256(v, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
        _mm256_add_epi8(_mm256_shuffle_epi8(table, lo), _mm256_shuffle_epi8(table, hi))
    }

    /// AVX2 implementation for byte arrays - heavily optimized with batched horizontal sums
    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();
        let mut i = 0;

        // For small inputs, fall back to SSE or scalar
        if length < 64 {
            if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
                return hamming_distance_bytes_sse(a, b, max_dist);
            }
            return hamming_distance_bytes_native(a, b, max_dist);
        }

        // VPSHUFB lookup table for 4-bit popcount
        let mask = _mm256_set1_epi8(0x0F);
        let table = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        );

        if max_dist < 0 {
            let mut total = _mm256_setzero_si256();

            // Process 512 bytes at a time (16 x 32 bytes) before horizontal sum
            // Per-lane accumulation: 16 iters x 8 bits max = 128 < 255, safe for u8
            while i + 512 <= length {
                let mut acc = _mm256_setzero_si256();
                for _ in 0..16 {
                    let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                    let xor = _mm256_xor_si256(a32, b32);
                    acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                    i += 32;
                }
                total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
            }

            // Process 128 bytes at a time (4 x 32 bytes)
            while i + 128 <= length {
                let mut acc = _mm256_setzero_si256();
                for _ in 0..4 {
                    let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                    let xor = _mm256_xor_si256(a32, b32);
                    acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                    i += 32;
                }
                total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
            }

            // Process remaining 32-byte chunks
            let mut acc = _mm256_setzero_si256();
            while i + 32 <= length {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                i += 32;
            }
            total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));

            // Extract final sum from 4 u64 lanes
            let mut difference = (_mm256_extract_epi64(total, 0)
                + _mm256_extract_epi64(total, 1)
                + _mm256_extract_epi64(total, 2)
                + _mm256_extract_epi64(total, 3)) as u64;

            // Process remaining bytes
            while i < length {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
                i += 1;
            }
            difference
        } else {
            // Early termination path
            let max_dist_u64 = max_dist as u64;
            let mut difference: u64 = 0;

            // Use batched counting but check periodically (every 128 bytes)
            while i + 128 <= length {
                let mut acc = _mm256_setzero_si256();
                for _ in 0..4 {
                    let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                    let xor = _mm256_xor_si256(a32, b32);
                    acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                    i += 32;
                }
                let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
                difference += (_mm256_extract_epi64(sad, 0)
                    + _mm256_extract_epi64(sad, 1)
                    + _mm256_extract_epi64(sad, 2)
                    + _mm256_extract_epi64(sad, 3)) as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
            }

            // Process remaining 32-byte chunks
            while i + 32 <= length {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                let cnt = popcnt256_shuffle(xor, mask, table);
                let sad = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
                difference += (_mm256_extract_epi64(sad, 0)
                    + _mm256_extract_epi64(sad, 1)
                    + _mm256_extract_epi64(sad, 2)
                    + _mm256_extract_epi64(sad, 3)) as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
                i += 32;
            }

            while i < length {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
                i += 1;
            }
            1
        }
    }

    /// SSE4.1 implementation for hex strings - optimized with batched popcount
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    pub unsafe fn hamming_distance_string_sse(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();
        
        // For short strings, scalar with lookup table is competitive
        if length < 32 {
            return hamming_distance_string_classic(a, b);
        }

        let zero = _mm_setzero_si128();
        let fifteen = _mm_set1_epi8(15);
        let subtract0vec = _mm_set1_epi8(b'0' as i8);
        let subtract55vec = _mm_set1_epi8(55);
        let andvec = _mm_set1_epi8(!0x20i8);
        let isdigit_mask = _mm_set1_epi8(b'9' as i8);
        
        // Popcount lookup table
        let popcnt_mask = _mm_set1_epi8(0x0F);
        let popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

        let mut i = 0;
        let mut total = _mm_setzero_si128();
        
        // Process 64 bytes at a time (4 x 16) with batched horizontal sum
        while i + 64 <= length {
            let mut acc = _mm_setzero_si128();
            
            for _ in 0..4 {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);

                // Branchless hex parsing: x > '9' ? ((x & ~0x20) - 55) : (x - '0')
                let a_cmp = _mm_cmpgt_epi8(a16, isdigit_mask);
                let b_cmp = _mm_cmpgt_epi8(b16, isdigit_mask);

                let a_letter = _mm_sub_epi8(_mm_and_si128(a16, andvec), subtract55vec);
                let b_letter = _mm_sub_epi8(_mm_and_si128(b16, andvec), subtract55vec);

                let a_digit = _mm_sub_epi8(a16, subtract0vec);
                let b_digit = _mm_sub_epi8(b16, subtract0vec);

                let a_hex = _mm_blendv_epi8(a_digit, a_letter, a_cmp);
                let b_hex = _mm_blendv_epi8(b_digit, b_letter, b_cmp);

                // Validate: all values must be 0-15
                let invalid = _mm_or_si128(
                    _mm_cmpgt_epi8(a_hex, fifteen),
                    _mm_cmpgt_epi8(b_hex, fifteen)
                );
                let negative = _mm_or_si128(
                    _mm_cmplt_epi8(a_hex, zero),
                    _mm_cmplt_epi8(b_hex, zero)
                );
                if _mm_testz_si128(_mm_or_si128(invalid, negative), 
                                  _mm_or_si128(invalid, negative)) == 0 {
                    return Err("hex string contains invalid char");
                }

                // XOR and popcount using VPSHUFB
                let xor = _mm_xor_si128(a_hex, b_hex);
                acc = _mm_add_epi8(acc, _mm_shuffle_epi8(popcnt_table, 
                    _mm_and_si128(xor, popcnt_mask)));
                
                i += 16;
            }
            total = _mm_add_epi64(total, _mm_sad_epu8(acc, zero));
        }

        // Process remaining 16-byte chunks
        let mut acc = _mm_setzero_si128();
        while i + 16 <= length {
            let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);

            let a_cmp = _mm_cmpgt_epi8(a16, isdigit_mask);
            let b_cmp = _mm_cmpgt_epi8(b16, isdigit_mask);

            let a_letter = _mm_sub_epi8(_mm_and_si128(a16, andvec), subtract55vec);
            let b_letter = _mm_sub_epi8(_mm_and_si128(b16, andvec), subtract55vec);

            let a_digit = _mm_sub_epi8(a16, subtract0vec);
            let b_digit = _mm_sub_epi8(b16, subtract0vec);

            let a_hex = _mm_blendv_epi8(a_digit, a_letter, a_cmp);
            let b_hex = _mm_blendv_epi8(b_digit, b_letter, b_cmp);

            let invalid = _mm_or_si128(
                _mm_cmpgt_epi8(a_hex, fifteen),
                _mm_cmpgt_epi8(b_hex, fifteen)
            );
            let negative = _mm_or_si128(
                _mm_cmplt_epi8(a_hex, zero),
                _mm_cmplt_epi8(b_hex, zero)
            );
            if _mm_testz_si128(_mm_or_si128(invalid, negative),
                              _mm_or_si128(invalid, negative)) == 0 {
                return Err("hex string contains invalid char");
            }

            let xor = _mm_xor_si128(a_hex, b_hex);
            acc = _mm_add_epi8(acc, _mm_shuffle_epi8(popcnt_table,
                _mm_and_si128(xor, popcnt_mask)));
            
            i += 16;
        }
        total = _mm_add_epi64(total, _mm_sad_epu8(acc, zero));

        let mut result = (_mm_extract_epi64(total, 0) + _mm_extract_epi64(total, 1)) as u64;

        // Handle remaining bytes with optimized scalar code
        while i < length {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
            i += 1;
        }

        Ok(result)
    }
}

// ARM64 NEON implementations
#[cfg(target_arch = "aarch64")]
mod neon_simd {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON vectorized hamming distance for hex strings.
    /// Processes 16 ASCII hex chars per iteration using:
    ///   - vqtbl1q_u8 for branchless hex→nibble conversion
    ///   - vcntq_u8 for parallel popcount
    ///   - vpaddlq cascade for horizontal summation
    #[inline]
    pub unsafe fn hamming_distance_string_neon(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();

        if length < 16 {
            return hamming_distance_string_classic(a, b);
        }

        // Hex→nibble lookup table for vqtbl1q_u8 (indices 0-15 map ASCII
        // low nibble to hex value; out-of-range produces 0xFF via saturation).
        // We split into two ranges: digits ('0'-'9') and letters ('A'-'F'/'a'-'f').
        //
        // Strategy: mask to low nibble, use vqtbl1q as a 16-entry LUT.
        // '0'(0x30)..'9'(0x39) have low nibbles 0x0..0x9 → identity
        // 'A'(0x41)..'F'(0x46) have low nibbles 0x1..0x6 → +9
        // 'a'(0x61)..'f'(0x66) have low nibbles 0x1..0x6 → +9
        // We detect digit vs letter via range comparison.

        let zero = vdupq_n_u8(0);
        let fifteen_u = vdupq_n_u8(15);
        let v_ascii_0 = vdupq_n_u8(b'0');
        let v_ascii_9 = vdupq_n_u8(b'9');
        let case_mask = vdupq_n_u8(!0x20u8); // 0xDF — clears bit 5 for case folding
        let v_ascii_a = vdupq_n_u8(b'A');
        let v_ascii_f = vdupq_n_u8(b'F');
        let offset_letter = vdupq_n_u8(b'A' - 10); // 55

        // Popcount lookup table: popcnt[i] = number of 1-bits in i, for i in 0..15
        let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());

        let mut i = 0usize;
        let mut total = vdupq_n_u64(0);

        // Process 64 chars at a time (4×16) to batch horizontal sums.
        // Per-byte accumulator holds at most 4*4 = 16 < 255, safe for u8 lanes.
        while i + 64 <= length {
            let mut acc = zero;

            for _ in 0..4 {
                let a16 = vld1q_u8(a.as_ptr().add(i));
                let b16 = vld1q_u8(b.as_ptr().add(i));

                let a_nib = hex_parse_neon(a16, v_ascii_0, v_ascii_9, case_mask, v_ascii_a, v_ascii_f, offset_letter);
                let b_nib = hex_parse_neon(b16, v_ascii_0, v_ascii_9, case_mask, v_ascii_a, v_ascii_f, offset_letter);

                // Validate: any lane > 15 means invalid char (0xFF from failed parse)
                let a_bad = vcgtq_u8(a_nib, fifteen_u);
                let b_bad = vcgtq_u8(b_nib, fifteen_u);
                let bad = vorrq_u8(a_bad, b_bad);
                if vmaxvq_u8(bad) != 0 {
                    return Err("hex string contains invalid char");
                }

                // XOR nibbles → popcount via table lookup (values are 0-15, only low nibble used)
                let xor = veorq_u8(a_nib, b_nib);
                let cnt = vqtbl1q_u8(popcnt_tbl, xor);
                acc = vaddq_u8(acc, cnt);

                i += 16;
            }

            // Horizontal sum: u8→u16→u32→u64, add into total
            total = vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(acc)));
        }

        // Process remaining 16-byte chunks
        let mut acc = zero;
        while i + 16 <= length {
            let a16 = vld1q_u8(a.as_ptr().add(i));
            let b16 = vld1q_u8(b.as_ptr().add(i));

            let a_nib = hex_parse_neon(a16, v_ascii_0, v_ascii_9, case_mask, v_ascii_a, v_ascii_f, offset_letter);
            let b_nib = hex_parse_neon(b16, v_ascii_0, v_ascii_9, case_mask, v_ascii_a, v_ascii_f, offset_letter);

            let a_bad = vcgtq_u8(a_nib, fifteen_u);
            let b_bad = vcgtq_u8(b_nib, fifteen_u);
            let bad = vorrq_u8(a_bad, b_bad);
            if vmaxvq_u8(bad) != 0 {
                return Err("hex string contains invalid char");
            }

            let xor = veorq_u8(a_nib, b_nib);
            let cnt = vqtbl1q_u8(popcnt_tbl, xor);
            acc = vaddq_u8(acc, cnt);

            i += 16;
        }
        total = vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(acc)));

        let mut result = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);

        // Scalar tail for remaining chars
        while i < length {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
            i += 1;
        }

        Ok(result)
    }

    /// Branchless vectorized hex ASCII → nibble (0-15) conversion.
    /// Invalid chars produce values > 15 (for easy detection).
    ///
    /// Logic per lane:
    ///   if '0' <= c <= '9':  c - '0'
    ///   elif 'A' <= (c & 0xDF) <= 'F':  (c & 0xDF) - 'A' + 10
    ///   else: 0xFF
    #[inline(always)]
    unsafe fn hex_parse_neon(
        chars: uint8x16_t,
        v_ascii_0: uint8x16_t,
        v_ascii_9: uint8x16_t,
        case_mask: uint8x16_t,
        v_ascii_a: uint8x16_t,
        v_ascii_f: uint8x16_t,
        offset_letter: uint8x16_t,
    ) -> uint8x16_t {
        // Digit path: result = c - '0'
        let digit_val = vsubq_u8(chars, v_ascii_0);
        let is_digit = vandq_u8(vcgeq_u8(chars, v_ascii_0), vcleq_u8(chars, v_ascii_9));

        // Letter path: fold case, then result = (c & 0xDF) - 55
        let upper = vandq_u8(chars, case_mask);
        let letter_val = vsubq_u8(upper, offset_letter);
        let is_letter = vandq_u8(vcgeq_u8(upper, v_ascii_a), vcleq_u8(upper, v_ascii_f));

        // Merge: pick digit_val where is_digit, letter_val where is_letter, 0xFF otherwise
        let invalid = vdupq_n_u8(0xFF);
        let result = vbslq_u8(is_digit, digit_val, vbslq_u8(is_letter, letter_val, invalid));
        result
    }
}

/// Dispatch to appropriate byte distance implementation based on current algorithm
#[inline(always)]
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

        // On ARM64, NEON and native use the same optimized implementation
        // Benchmarks show that Rust's auto-vectorized count_ones() on Apple Silicon
        // is faster than handwritten NEON intrinsics (vcntq_u8 + horizontal sums)
        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => hamming_distance_bytes_native(a, b, max_dist),

        _ => hamming_distance_bytes_native(a, b, max_dist),
    }
}

/// Dispatch to appropriate string distance implementation based on current algorithm
#[inline(always)]
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

    #[cfg(target_arch = "aarch64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        if algo == ALGO_NEON || algo == ALGO_NATIVE {
            return unsafe { neon_simd::hamming_distance_string_neon(a, b) };
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
/// Optimized with early termination and branchless parsing
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

    // Max possible hamming distance per hex char is 4, so if max_dist >= 4*len, always true
    if max_dist_u64 >= (a_str.len() as u64) * 4 {
        return Ok(true);
    }

    let a_bytes = a_str.as_bytes();
    let b_bytes = b_str.as_bytes();
    let len = a_bytes.len();

    let mut result: u64 = 0;
    let mut i = 0;

    // Process 4 chars at a time for better throughput
    // SAFETY: bounds checked by loop condition
    while i + 4 <= len {
        unsafe {
            let val1_0 = hex_char_to_nibble(*a_bytes.get_unchecked(i));
            let val2_0 = hex_char_to_nibble(*b_bytes.get_unchecked(i));
            let val1_1 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 1));
            let val2_1 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 1));
            let val1_2 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 2));
            let val2_2 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 2));
            let val1_3 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 3));
            let val2_3 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 3));
            
            // Validate all 8 values
            let invalid = (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
            if invalid != 0 {
                return Err(PyValueError::new_err("hex string contains invalid char"));
            }
            
            result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
        }
        
        // Early termination check
        if result > max_dist_u64 {
            return Ok(false);
        }
        i += 4;
    }

    // Handle remaining chars
    while i < len {
        unsafe {
            let val1 = hex_char_to_nibble(*a_bytes.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b_bytes.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err(PyValueError::new_err("hex string contains invalid char"));
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        }
        if result > max_dist_u64 {
            return Ok(false);
        }
        i += 1;
    }

    Ok(true)
}

/// Macro for zero-overhead array scanning loop
/// Duplicates loop body for each algorithm to eliminate ALL call overhead
macro_rules! array_scan_loop {
    ($big_array:expr, $elem_size:expr, $num_elements:expr, $small_array:expr, $max_dist:expr, $hamming_call:expr) => {{
        // Raw pointer arithmetic like C++ for zero-overhead iteration
        let big_ptr = $big_array.as_ptr();
        let elem_size = $elem_size;
        let num_elements = $num_elements;
        let max_dist = $max_dist;
        let mut i: usize = 0;
        
        while i < num_elements {
            // SAFETY: We've verified big_array.len() is a multiple of elem_size
            let chunk = unsafe { std::slice::from_raw_parts(big_ptr.add(i * elem_size), elem_size) };
            
            if $hamming_call(chunk, $small_array, max_dist) == 1 {
                return Ok(i as i64);
            }
            
            i += 1;
        }
        
        Ok(-1i64)
    }};
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

    // Resolve algorithm ONCE outside the loop (like C++ function pointer)
    // Each arm has its own inlined loop - no function pointer/closure overhead
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => {
            array_scan_loop!(big_array, elem_size, num_elements, small_array, max_dist_val,
                hamming_distance_bytes_classic)
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX2 if is_x86_feature_detected!("avx2") => {
            array_scan_loop!(big_array, elem_size, num_elements, small_array, max_dist_val,
                |a, b, m| unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, m) })
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_SSE41 if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") => {
            array_scan_loop!(big_array, elem_size, num_elements, small_array, max_dist_val,
                |a, b, m| unsafe { x86_simd::hamming_distance_bytes_sse(a, b, m) })
        }

        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => {
            // NEON now uses native count_ones() which auto-vectorizes well on ARM64
            array_scan_loop!(big_array, elem_size, num_elements, small_array, max_dist_val,
                hamming_distance_bytes_native)
        }

        _ => {
            array_scan_loop!(big_array, elem_size, num_elements, small_array, max_dist_val,
                hamming_distance_bytes_native)
        }
    }
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
