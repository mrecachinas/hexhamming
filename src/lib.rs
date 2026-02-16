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
#[cfg(target_arch = "x86_64")]
const ALGO_AVX512: u8 = 5;
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

    /// Branchless vectorized hex ASCII → nibble conversion for AVX2.
    /// Same subtract-and-correct strategy as SSE/NEON, but on 32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hex_parse_avx2(
        chars: __m256i,
        case_mask: __m256i,
        ascii_0: __m256i,
        seven: __m256i,
        nine: __m256i,
        ten: __m256i,
    ) -> __m256i {
        let digit_val = _mm256_sub_epi8(chars, ascii_0);
        let letter_val = _mm256_sub_epi8(_mm256_and_si256(chars, case_mask), ascii_0);
        let is_letter = _mm256_cmpgt_epi8(digit_val, nine);
        let adjusted = _mm256_sub_epi8(letter_val, seven);
        let result = _mm256_blendv_epi8(digit_val, adjusted, is_letter);
        let bad_letter = _mm256_and_si256(is_letter, _mm256_cmpgt_epi8(ten, adjusted));
        _mm256_or_si256(result, bad_letter)
    }

    /// AVX2 pack-to-bytes implementation for hex strings.
    /// Parses 64 hex chars (2×32) → nibbles, XORs, packs pairs into bytes,
    /// then uses hardware popcnt on u64 extracts.
    #[target_feature(enable = "avx2", enable = "popcnt")]
    pub unsafe fn hamming_distance_string_avx2(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();

        // Fall back to SSE for inputs < 64 chars
        if length < 64 {
            return hamming_distance_string_sse(a, b);
        }

        let zero = _mm256_setzero_si256();
        let fifteen = _mm256_set1_epi8(15);
        let case_mask = _mm256_set1_epi8(!0x20i8);     // 0xDF
        let ascii_0 = _mm256_set1_epi8(b'0' as i8);
        let seven = _mm256_set1_epi8(7);
        let nine = _mm256_set1_epi8(9);
        let ten = _mm256_set1_epi8(10);

        let mut i = 0;
        let mut difference: u64 = 0;

        // Process 64 hex chars at a time: 2×32 chars → parse → XOR → pack → popcnt
        while i + 64 <= length {
            let a_lo = hex_parse_avx2(
                _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_lo = hex_parse_avx2(
                _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let a_hi = hex_parse_avx2(
                _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_hi = hex_parse_avx2(
                _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i),
                case_mask, ascii_0, seven, nine, ten,
            );

            // Validate: any lane > 15 or < 0 (signed) means invalid
            let invalid = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpgt_epi8(a_lo, fifteen),
                    _mm256_cmpgt_epi8(b_lo, fifteen),
                ),
                _mm256_or_si256(
                    _mm256_cmpgt_epi8(a_hi, fifteen),
                    _mm256_cmpgt_epi8(b_hi, fifteen),
                ),
            );
            let negative = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpgt_epi8(zero, a_lo),
                    _mm256_cmpgt_epi8(zero, b_lo),
                ),
                _mm256_or_si256(
                    _mm256_cmpgt_epi8(zero, a_hi),
                    _mm256_cmpgt_epi8(zero, b_hi),
                ),
            );
            let bad = _mm256_or_si256(invalid, negative);
            if _mm256_testz_si256(bad, bad) == 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles
            let xor_lo = _mm256_xor_si256(a_lo, b_lo);
            let xor_hi = _mm256_xor_si256(a_hi, b_hi);

            // Pack nibble pairs into bytes using VPSHUFB to deinterleave
            let shuf_even = _mm256_setr_epi8(
                0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
                0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let shuf_odd = _mm256_setr_epi8(
                1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
            );

            // AVX2 VPSHUFB operates within 128-bit lanes, so each 256-bit register
            // gives us 8 even/odd bytes in low half of each 128-bit lane.
            // We need to collect them: use _mm256_unpacklo_epi64 to merge within lanes,
            // then _mm256_permute4x64_epi64 to consolidate across lanes.
            let even_lo = _mm256_shuffle_epi8(xor_lo, shuf_even);
            let odd_lo  = _mm256_shuffle_epi8(xor_lo, shuf_odd);
            let even_hi = _mm256_shuffle_epi8(xor_hi, shuf_even);
            let odd_hi  = _mm256_shuffle_epi8(xor_hi, shuf_odd);

            // Merge: low 64 bits of each lane contain our data
            // unpacklo_epi64 merges lo+hi within each 128-bit lane
            let even_merged = _mm256_unpacklo_epi64(even_lo, even_hi);
            let odd_merged  = _mm256_unpacklo_epi64(odd_lo, odd_hi);

            // Consolidate: permute to put lane0_lo64, lane0_hi64, lane1_lo64, lane1_hi64
            // into contiguous order. After unpacklo, layout is:
            //   lane0: [even_lo_lane0_8B | even_hi_lane0_8B]
            //   lane1: [even_lo_lane1_8B | even_hi_lane1_8B]
            // We want: [even_lo_lane0 | even_hi_lane0 | even_lo_lane1 | even_hi_lane1]
            // which is already the natural order: q0, q1, q2, q3 → permute 0,2,1,3
            let even = _mm256_permute4x64_epi64(even_merged, 0b11_01_10_00); // 0,2,1,3
            let odd  = _mm256_permute4x64_epi64(odd_merged, 0b11_01_10_00);

            // Pack: (even << 4) | odd, with mask to prevent cross-byte leakage
            let hi_nib_mask = _mm256_set1_epi8(0xF0u8 as i8);
            let packed = _mm256_or_si256(
                _mm256_and_si256(_mm256_slli_epi16(even, 4), hi_nib_mask),
                odd,
            );

            // Hardware popcnt on 32 packed bytes (extract as four u64s)
            let v128_lo = _mm256_castsi256_si128(packed);
            let v128_hi = _mm256_extracti128_si256(packed, 1);
            difference += (_mm_cvtsi128_si64(v128_lo) as u64).count_ones() as u64
                + (_mm_extract_epi64(v128_lo, 1) as u64).count_ones() as u64
                + (_mm_cvtsi128_si64(v128_hi) as u64).count_ones() as u64
                + (_mm_extract_epi64(v128_hi, 1) as u64).count_ones() as u64;

            i += 64;
        }

        // Fall through to SSE for remaining < 64 chars
        if i < length {
            let remaining = hamming_distance_string_sse(&a[i..], &b[i..])?;
            difference += remaining;
        }

        Ok(difference)
    }

    /// Branchless vectorized hex ASCII → nibble conversion for AVX-512BW.
    /// 64 lanes, same subtract-and-correct strategy.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn hex_parse_avx512(
        chars: __m512i,
        case_mask: __m512i,
        ascii_0: __m512i,
        seven: __m512i,
        nine: __m512i,
        ten: __m512i,
    ) -> __m512i {
        let digit_val = _mm512_sub_epi8(chars, ascii_0);
        let letter_val = _mm512_sub_epi8(_mm512_and_si512(chars, case_mask), ascii_0);
        let is_letter = _mm512_cmpgt_epi8_mask(digit_val, nine);
        let adjusted = _mm512_sub_epi8(letter_val, seven);
        let result = _mm512_mask_blend_epi8(is_letter, digit_val, adjusted);
        // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
        let bad_letter = is_letter & _mm512_cmpgt_epi8_mask(ten, adjusted);
        let ones = _mm512_set1_epi8(-1);  // 0xFF
        _mm512_mask_blend_epi8(bad_letter, result, ones)
    }

    /// AVX-512 BITALG implementation for hex strings.
    /// Parses 64 hex chars per load, XORs nibbles directly, uses VPOPCNTB
    /// for native per-byte popcount — no pack step needed.
    #[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
    pub unsafe fn hamming_distance_string_avx512(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();

        // With masked loads, we can handle any length efficiently
        if length < 16 {
            return hamming_distance_string_classic(a, b);
        }

        let fifteen = _mm512_set1_epi8(15);
        let case_mask = _mm512_set1_epi8(!0x20i8);     // 0xDF
        let ascii_0 = _mm512_set1_epi8(b'0' as i8);
        let seven = _mm512_set1_epi8(7);
        let nine = _mm512_set1_epi8(9);
        let ten = _mm512_set1_epi8(10);
        let zero = _mm512_setzero_si512();

        let mut i = 0;
        let mut total = _mm512_setzero_si512();

        // Process 64 hex chars at a time
        // Each nibble XOR produces at most 4 set bits, so per-byte popcount
        // accumulator maxes at 4 per lane. Safe to accumulate 63 iterations
        // (63 * 4 = 252 < 255) before horizontal sum. For <256 chars we
        // never exceed 4 iterations, so no overflow concern.
        while i + 64 <= length {
            let a_nib = hex_parse_avx512(
                _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_nib = hex_parse_avx512(
                _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i),
                case_mask, ascii_0, seven, nine, ten,
            );

            // Validate: any nibble > 15 or < 0 (signed) means invalid
            let invalid = _mm512_cmpgt_epi8_mask(a_nib, fifteen)
                | _mm512_cmpgt_epi8_mask(b_nib, fifteen)
                | _mm512_cmpgt_epi8_mask(zero, a_nib)
                | _mm512_cmpgt_epi8_mask(zero, b_nib);
            if invalid != 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles and VPOPCNTB — counts set bits per byte
            let xor = _mm512_xor_si512(a_nib, b_nib);
            let cnt = _mm512_popcnt_epi8(xor);
            total = _mm512_add_epi8(total, cnt);

            i += 64;
        }

        // Horizontal sum via SAD against zero → u64 lanes
        let sad = _mm512_sad_epu8(total, zero);
        let mut difference = _mm512_reduce_add_epi64(sad) as u64;

        // Handle remaining chars with masked AVX-512 load (no fallthrough)
        let remaining = length - i;
        if remaining > 0 {
            let mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
            let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);

            let a_nib = hex_parse_avx512(a_tail, case_mask, ascii_0, seven, nine, ten);
            let b_nib = hex_parse_avx512(b_tail, case_mask, ascii_0, seven, nine, ten);

            // Validate only the active lanes
            let invalid = (_mm512_cmpgt_epi8_mask(a_nib, fifteen)
                | _mm512_cmpgt_epi8_mask(b_nib, fifteen)
                | _mm512_cmpgt_epi8_mask(zero, a_nib)
                | _mm512_cmpgt_epi8_mask(zero, b_nib))
                & mask;
            if invalid != 0 {
                return Err("hex string contains invalid char");
            }

            let xor = _mm512_xor_si512(a_nib, b_nib);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
        }

        Ok(difference)
    }

    /// AVX-512 BITALG implementation for byte arrays.
    /// XOR + VPOPCNTB for native per-byte popcount.
    #[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
    pub unsafe fn hamming_distance_bytes_avx512(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let length = a.len();
        let mut i = 0;

        if length < 64 {
            return hamming_distance_bytes_avx2(a, b, max_dist);
        }

        let zero = _mm512_setzero_si512();

        if max_dist < 0 {
            let mut total = _mm512_setzero_si512();

            // Process 1024 bytes at a time (16 × 64) before horizontal sum
            // Per-lane max: 16 × 8 = 128 < 255, safe for u8
            while i + 1024 <= length {
                let mut acc = _mm512_setzero_si512();
                for _ in 0..16 {
                    let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
                    let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
                    let xor = _mm512_xor_si512(a64, b64);
                    acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
                    i += 64;
                }
                total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));
            }

            // Process remaining 64-byte chunks
            let mut acc = _mm512_setzero_si512();
            while i + 64 <= length {
                let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
                let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
                let xor = _mm512_xor_si512(a64, b64);
                acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
                i += 64;
            }
            total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));

            let mut difference = _mm512_reduce_add_epi64(total) as u64;

            // Masked tail — no scalar fallback
            let remaining = length - i;
            if remaining > 0 {
                let mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
                let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
                let xor = _mm512_xor_si512(a_tail, b_tail);
                let cnt = _mm512_popcnt_epi8(xor);
                let sad = _mm512_sad_epu8(cnt, zero);
                difference += _mm512_reduce_add_epi64(sad) as u64;
            }
            difference
        } else {
            // Early termination path
            let max_dist_u64 = max_dist as u64;
            let mut difference: u64 = 0;

            while i + 64 <= length {
                let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
                let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
                let xor = _mm512_xor_si512(a64, b64);
                let cnt = _mm512_popcnt_epi8(xor);
                let sad = _mm512_sad_epu8(cnt, zero);
                difference += _mm512_reduce_add_epi64(sad) as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
                i += 64;
            }

            // Masked tail for early termination path
            let remaining = length - i;
            if remaining > 0 {
                let mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
                let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
                let xor = _mm512_xor_si512(a_tail, b_tail);
                let cnt = _mm512_popcnt_epi8(xor);
                let sad = _mm512_sad_epu8(cnt, zero);
                difference += _mm512_reduce_add_epi64(sad) as u64;
                if difference > max_dist_u64 {
                    return 0;
                }
            }
            1
        }
    }

    /// Branchless vectorized hex ASCII → nibble conversion for SSE4.1.
    /// Same subtract-and-correct strategy as the NEON version:
    ///   1. digit_val = c - '0': digits → 0-9
    ///   2. letter_val = (c & 0xDF) - '0' - 7: letters → 10-15
    ///   3. Select letter path where digit_val > 9
    ///   4. Force invalid where letter result < 10 (catches '@', '`')
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn hex_parse_sse(
        chars: __m128i,
        case_mask: __m128i,
        ascii_0: __m128i,
        seven: __m128i,
        nine: __m128i,
        ten: __m128i,
    ) -> __m128i {
        let digit_val = _mm_sub_epi8(chars, ascii_0);
        let letter_val = _mm_sub_epi8(_mm_and_si128(chars, case_mask), ascii_0);
        let is_letter = _mm_cmpgt_epi8(digit_val, nine);
        let adjusted = _mm_sub_epi8(letter_val, seven);
        let result = _mm_blendv_epi8(digit_val, adjusted, is_letter);
        // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
        let bad_letter = _mm_and_si128(is_letter, _mm_cmplt_epi8(adjusted, ten));
        _mm_or_si128(result, bad_letter)
    }

    /// SSE4.1 pack-to-bytes implementation for hex strings.
    /// Parses 32 hex chars (2×16) → nibbles, XORs, packs pairs into bytes,
    /// then uses hardware popcnt on u64 extracts.
    #[target_feature(enable = "sse4.1", enable = "popcnt")]
    pub unsafe fn hamming_distance_string_sse(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();
        
        if length < 32 {
            return hamming_distance_string_classic(a, b);
        }

        let zero = _mm_setzero_si128();
        let fifteen = _mm_set1_epi8(15);
        let case_mask = _mm_set1_epi8(!0x20i8);     // 0xDF
        let ascii_0 = _mm_set1_epi8(b'0' as i8);
        let seven = _mm_set1_epi8(7);
        let nine = _mm_set1_epi8(9);
        let ten = _mm_set1_epi8(10);

        let mut i = 0;
        let mut difference: u64 = 0;
        
        // Process 32 hex chars at a time: parse→XOR→pack→popcnt
        while i + 32 <= length {
            let a_lo = hex_parse_sse(
                _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_lo = hex_parse_sse(
                _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let a_hi = hex_parse_sse(
                _mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_hi = hex_parse_sse(
                _mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );

            // Validate all 4 vectors: any lane > 15 or < 0 (signed) means invalid
            let invalid = _mm_or_si128(
                _mm_or_si128(
                    _mm_cmpgt_epi8(a_lo, fifteen),
                    _mm_cmpgt_epi8(b_lo, fifteen),
                ),
                _mm_or_si128(
                    _mm_cmpgt_epi8(a_hi, fifteen),
                    _mm_cmpgt_epi8(b_hi, fifteen),
                ),
            );
            let negative = _mm_or_si128(
                _mm_or_si128(
                    _mm_cmplt_epi8(a_lo, zero),
                    _mm_cmplt_epi8(b_lo, zero),
                ),
                _mm_or_si128(
                    _mm_cmplt_epi8(a_hi, zero),
                    _mm_cmplt_epi8(b_hi, zero),
                ),
            );
            let bad = _mm_or_si128(invalid, negative);
            if _mm_testz_si128(bad, bad) == 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles
            let xor_lo = _mm_xor_si128(a_lo, b_lo);
            let xor_hi = _mm_xor_si128(a_hi, b_hi);

            // Pack nibble pairs into bytes: even nibbles << 4 | odd nibbles
            // Deinterleave even/odd using shuffle masks
            let shuf_even = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
            let shuf_odd  = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

            // From xor_lo (16 nibbles) → 8 bytes in low half
            let even_lo = _mm_shuffle_epi8(xor_lo, shuf_even);
            let odd_lo  = _mm_shuffle_epi8(xor_lo, shuf_odd);
            // From xor_hi (16 nibbles) → 8 bytes in low half
            let even_hi = _mm_shuffle_epi8(xor_hi, shuf_even);
            let odd_hi  = _mm_shuffle_epi8(xor_hi, shuf_odd);

            // Combine: [even_lo_8 | even_hi_8] and [odd_lo_8 | odd_hi_8]
            // Use _mm_unpacklo_epi64 to merge the two 8-byte halves
            let even = _mm_unpacklo_epi64(even_lo, even_hi);
            let odd  = _mm_unpacklo_epi64(odd_lo, odd_hi);

            // Pack: (even << 4) | odd
            // _mm_slli_epi16 shifts 16-bit lanes, so bits leak across byte
            // boundaries. Mask to keep only the high nibble per byte.
            let hi_nib_mask = _mm_set1_epi8(0xF0u8 as i8);
            let packed = _mm_or_si128(
                _mm_and_si128(_mm_slli_epi16(even, 4), hi_nib_mask),
                odd,
            );

            // Hardware popcnt on the 16 packed bytes (extract as two u64s)
            let lo64 = _mm_cvtsi128_si64(packed) as u64;
            let hi64 = _mm_extract_epi64(packed, 1) as u64;
            difference += lo64.count_ones() as u64 + hi64.count_ones() as u64;

            i += 32;
        }

        // Process remaining 16-byte chunks with shuffle-based popcount
        let popcnt_mask = _mm_set1_epi8(0x0F);
        let popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        let mut acc = _mm_setzero_si128();
        while i + 16 <= length {
            let a_hex = hex_parse_sse(
                _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );
            let b_hex = hex_parse_sse(
                _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
                case_mask, ascii_0, seven, nine, ten,
            );

            let bad = _mm_or_si128(
                _mm_cmpgt_epi8(a_hex, fifteen),
                _mm_cmpgt_epi8(b_hex, fifteen),
            );
            if _mm_testz_si128(bad, bad) == 0 {
                return Err("hex string contains invalid char");
            }

            let xor = _mm_xor_si128(a_hex, b_hex);
            acc = _mm_add_epi8(acc, _mm_shuffle_epi8(popcnt_table,
                _mm_and_si128(xor, popcnt_mask)));
            
            i += 16;
        }
        let sad = _mm_sad_epu8(acc, zero);
        difference += (_mm_extract_epi64(sad, 0) + _mm_extract_epi64(sad, 1)) as u64;

        // Scalar tail
        while i < length {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
            i += 1;
        }

        Ok(difference)
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
        let case_mask = vdupq_n_u8(0xDF); // clears bit 5 for case folding
        let ascii_0 = vdupq_n_u8(b'0');
        let seven = vdupq_n_u8(7);
        let nine = vdupq_n_u8(9);
        let ten = vdupq_n_u8(10);

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

                let a_nib = hex_parse_neon(a16, case_mask, ascii_0, seven, nine, ten);
                let b_nib = hex_parse_neon(b16, case_mask, ascii_0, seven, nine, ten);

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

            let a_nib = hex_parse_neon(a16, case_mask, ascii_0, seven, nine, ten);
            let b_nib = hex_parse_neon(b16, case_mask, ascii_0, seven, nine, ten);

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
    /// Invalid chars produce values > 15 (for easy detection by caller).
    ///
    /// Strategy (7 NEON instructions):
    ///   1. digit_val = c - '0': digits become 0-9
    ///   2. letter_val = (c & 0xDF) - '0' - 7: letters become 10-15
    ///   3. Select letter path where digit_val > 9
    ///   4. Force invalid where letter result < 10 (catches '@', '`')
    #[inline(always)]
    unsafe fn hex_parse_neon(
        chars: uint8x16_t,
        case_mask: uint8x16_t,
        ascii_0: uint8x16_t,
        seven: uint8x16_t,
        nine: uint8x16_t,
        ten: uint8x16_t,
    ) -> uint8x16_t {
        let digit_val = vsubq_u8(chars, ascii_0);
        let letter_val = vsubq_u8(vandq_u8(chars, case_mask), ascii_0);
        let is_letter = vcgtq_u8(digit_val, nine);
        let adjusted = vsubq_u8(letter_val, seven);
        let result = vbslq_u8(is_letter, adjusted, digit_val);
        // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
        let bad_letter = vandq_u8(is_letter, vcltq_u8(adjusted, ten));
        vorrq_u8(result, bad_letter)
    }
    /// Alternative: parse 32 hex chars → pack into 16 bytes → use vcntq_u8.
    /// Processes 32 hex chars per iteration (vs 16 in the nibble approach),
    /// and replaces the vqtbl1q popcount lookup with the native vcntq_u8
    /// instruction which counts all 8 bits per byte in a single cycle.
    #[inline]
    pub unsafe fn hamming_distance_string_neon_pack(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
        let length = a.len();

        if length < 32 {
            return hamming_distance_string_neon(a, b);
        }

        let fifteen_u = vdupq_n_u8(15);
        let case_mask = vdupq_n_u8(0xDF);
        let ascii_0 = vdupq_n_u8(b'0');
        let seven = vdupq_n_u8(7);
        let nine = vdupq_n_u8(9);
        let ten = vdupq_n_u8(10);
        let four = vdupq_n_s8(4);

        let mut i = 0usize;
        let mut difference: u64 = 0;

        // Process 32 hex chars at a time:
        //   Load 32 chars (2×16), parse to nibbles, pack pairs into bytes,
        //   XOR the packed bytes, vcntq_u8 popcount.
        while i + 32 <= length {
            // Parse first 16 hex chars → nibbles
            let a_lo = hex_parse_neon(vld1q_u8(a.as_ptr().add(i)), case_mask, ascii_0, seven, nine, ten);
            let b_lo = hex_parse_neon(vld1q_u8(b.as_ptr().add(i)), case_mask, ascii_0, seven, nine, ten);
            // Parse next 16 hex chars → nibbles
            let a_hi = hex_parse_neon(vld1q_u8(a.as_ptr().add(i + 16)), case_mask, ascii_0, seven, nine, ten);
            let b_hi = hex_parse_neon(vld1q_u8(b.as_ptr().add(i + 16)), case_mask, ascii_0, seven, nine, ten);

            // Validate all 4 vectors
            let bad = vorrq_u8(
                vorrq_u8(vcgtq_u8(a_lo, fifteen_u), vcgtq_u8(b_lo, fifteen_u)),
                vorrq_u8(vcgtq_u8(a_hi, fifteen_u), vcgtq_u8(b_hi, fifteen_u)),
            );
            if vmaxvq_u8(bad) != 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles first (before packing — same result, fewer packs)
            let xor_lo = veorq_u8(a_lo, b_lo);  // 16 nibble XOR results
            let xor_hi = veorq_u8(a_hi, b_hi);  // 16 nibble XOR results

            // Pack: interleave even/odd nibbles into bytes
            // xor_lo has nibbles [0,1,2,3,...,15], xor_hi has [16,17,...,31]
            // We want bytes where each byte = (nibble[2k] << 4) | nibble[2k+1]
            // Use UZP to deinterleave even/odd lanes, then shift+OR
            let even_lo = vuzp1q_u8(xor_lo, xor_hi);  // even indices: 0,2,4,...
            let odd_lo = vuzp2q_u8(xor_lo, xor_hi);   // odd indices: 1,3,5,...
            // Shift even nibbles left 4 bits via reinterpret + signed shift
            let packed = vorrq_u8(
                vreinterpretq_u8_s8(vshlq_s8(vreinterpretq_s8_u8(even_lo), four)),
                odd_lo,
            );

            // Now packed has 16 bytes, each containing two XOR'd nibbles
            // vcntq_u8 counts all set bits per byte — exactly what we want
            let cnt = vcntq_u8(packed);
            // Horizontal sum
            difference += vaddlvq_u8(cnt) as u64;

            i += 32;
        }

        // Handle remaining chars with the nibble-based approach
        while i + 16 <= length {
            let a_nib = hex_parse_neon(vld1q_u8(a.as_ptr().add(i)), case_mask, ascii_0, seven, nine, ten);
            let b_nib = hex_parse_neon(vld1q_u8(b.as_ptr().add(i)), case_mask, ascii_0, seven, nine, ten);
            let bad = vorrq_u8(vcgtq_u8(a_nib, fifteen_u), vcgtq_u8(b_nib, fifteen_u));
            if vmaxvq_u8(bad) != 0 {
                return Err("hex string contains invalid char");
            }
            let xor = veorq_u8(a_nib, b_nib);
            let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());
            let cnt = vqtbl1q_u8(popcnt_tbl, xor);
            difference += vaddlvq_u8(cnt) as u64;
            i += 16;
        }

        // Scalar tail
        while i < length {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
            i += 1;
        }

        Ok(difference)
    }
}

/// Dispatch to appropriate byte distance implementation based on current algorithm
#[inline(always)]
fn hamming_distance_bytes_dispatch(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => hamming_distance_bytes_classic(a, b, max_dist),

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX512 => {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                unsafe { x86_simd::hamming_distance_bytes_avx512(a, b, max_dist) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
            } else {
                hamming_distance_bytes_native(a, b, max_dist)
            }
        }

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
        if (algo == ALGO_AVX512 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512bitalg")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_avx512(a, b) };
        }
        if (algo == ALGO_AVX512 || algo == ALGO_AVX2 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_avx2(a, b) };
        }
        if (algo == ALGO_AVX512 || algo == ALGO_AVX2 || algo == ALGO_SSE41 || algo == ALGO_NATIVE)
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
            return unsafe { neon_simd::hamming_distance_string_neon_pack(a, b) };
        }
    }

    hamming_distance_string_classic(a, b)
}

// ─── Python bindings (only compiled with the "python" feature) ───────────

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Calculate the hamming distance of two hexadecimal strings
    ///
    /// This is equivalent to `bin(int(a, 16) ^ int(b, 16)).count('1')`
    /// but optimized using SIMD instructions where available.
    #[pyfunction]
    #[pyo3(signature = (a, b))]
    fn hamming_distance_string(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
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

    let a_owned = a_str.as_bytes().to_vec();
    let b_owned = b_str.as_bytes().to_vec();
    py.allow_threads(move || {
        hamming_distance_string_dispatch(&a_owned, &b_owned)
    }).map_err(PyValueError::new_err)
}

/// Calculate the hamming distance of two byte arrays
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_bytes(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
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

    let a_owned = a_bytes.to_vec();
    let b_owned = b_bytes.to_vec();
    Ok(py.allow_threads(move || {
        hamming_distance_bytes_dispatch(&a_owned, &b_owned, -1)
    }))
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

/// Check if two byte arrays are within a specified Hamming distance
/// Returns True if distance <= max_dist, False otherwise
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_bytes_within_dist(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let a_bytes: &[u8] = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_bytes: &[u8] = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_bytes.is_empty() || b_bytes.is_empty() {
        return Err(PyValueError::new_err("array size must be >0"));
    }
    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if a_bytes.len() != b_bytes.len() {
        return Err(PyValueError::new_err("array sizes need to be the same"));
    }

    let a_owned = a_bytes.to_vec();
    let b_owned = b_bytes.to_vec();
    let result = py.allow_threads(move || {
        hamming_distance_bytes_dispatch(&a_owned, &b_owned, max_dist_val)
    });
    Ok(result == 1)
}

/// Check if any element of byte array is within a specified Hamming Distance
/// and return its index or -1 otherwise.
/// (Legacy name, equivalent to check_bytes_arrays_first_within_dist)
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    check_bytes_arrays_first_within_dist(py, array_of_elems, elem_to_compare, max_dist)
}

/// Check if any element of byte array is within a specified Hamming Distance
/// and return the index of the first match, or -1 otherwise.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_first_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<i64> {
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

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let result = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            if hamming_distance_bytes_dispatch(chunk, &small_owned, max_dist_val) == 1 {
                return i as i64;
            }
        }
        -1i64
    });
    Ok(result)
}

/// Find the element in byte array with the smallest Hamming distance
/// Returns (best_distance, best_index), or (-1, -1) if none found within max_dist
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_best_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<(i64, i64)> {
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

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let result = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        let mut best_dist: i64 = -1;
        let mut best_index: i64 = -1;

        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            // Use current best as threshold for early termination, or max_dist if no match yet
            let threshold = if best_dist >= 0 { best_dist - 1 } else { max_dist_val };
            if hamming_distance_bytes_dispatch(chunk, &small_owned, threshold) == 0 {
                continue;
            }
            let dist = hamming_distance_bytes_dispatch(chunk, &small_owned, -1) as i64;
            if best_dist < 0 || dist < best_dist {
                best_dist = dist;
                best_index = i as i64;
            }
        }
        (best_dist, best_index)
    });
    Ok(result)
}

/// Find all elements in byte array within a specified Hamming distance
/// Returns list of (distance, index) tuples
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_all_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u64, u64)>> {
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

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let results = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        let mut out: Vec<(u64, u64)> = Vec::new();

        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            if hamming_distance_bytes_dispatch(chunk, &small_owned, max_dist_val) == 0 {
                continue;
            }
            let dist = hamming_distance_bytes_dispatch(chunk, &small_owned, -1);
            out.push((dist, i as u64));
        }
        out
    });
    Ok(results)
}

/// Change algorithm used for calculations
/// Returns empty string if successful, or error message otherwise
#[pyfunction]
fn set_algo(algo_name: &str) -> PyResult<String> {
    match algo_name.to_lowercase().as_str() {
        "avx512" | "avx-512" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                    CURRENT_ALGO.store(ALGO_AVX512, Ordering::Relaxed);
                    return Ok(String::new());
                }
                return Ok("CPU doesn't support AVX-512 BITALG".to_string());
            }
            #[cfg(not(target_arch = "x86_64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

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
    m.add("__version__", "2.4.0")?;
    m.add_function(wrap_pyfunction!(hamming_distance_string, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(check_hexstrings_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_first_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_best_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_all_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(set_algo, m)?)?;

    // Auto-detect best algorithm on module load
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
            CURRENT_ALGO.store(ALGO_AVX512, Ordering::Relaxed);
        } else if is_x86_feature_detected!("avx2") {
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
} // mod python

/// Calculate the bitwise hamming distance between two equal-length hex strings.
///
/// Automatically uses the best SIMD implementation available (NEON/AVX2/SSE4.1).
///
/// # Errors
/// Returns `Err` if the strings differ in length or contain non-hex characters.
///
/// # Example
/// ```
/// let dist = hexhamming::hex_hamming_distance("deadbeef", "00000000").unwrap();
/// assert_eq!(dist, 24);
/// ```
pub fn hex_hamming_distance(a: &str, b: &str) -> Result<u64, &'static str> {
    if a.len() != b.len() {
        return Err("strings are NOT the same length");
    }
    if a.is_empty() {
        return Ok(0);
    }
    hamming_distance_string_dispatch(a.as_bytes(), b.as_bytes())
}

/// Calculate the bitwise hamming distance between two equal-length byte slices.
///
/// Automatically uses the best SIMD implementation available (NEON/AVX2/SSE4.1).
///
/// # Errors
/// Returns `Err` if the slices differ in length.
///
/// # Example
/// ```
/// let dist = hexhamming::bytes_hamming_distance(b"\xff", b"\x00").unwrap();
/// assert_eq!(dist, 8);
/// ```
pub fn bytes_hamming_distance(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    if a.len() != b.len() {
        return Err("bytes are NOT the same length");
    }
    if a.is_empty() {
        return Ok(0);
    }
    Ok(hamming_distance_bytes_dispatch(a, b, -1))
}

/// Experimental: hex hamming distance using pack-to-bytes approach.
/// Parses 32 hex chars → 16 packed bytes, then uses vcntq_u8.
#[cfg(target_arch = "aarch64")]
pub fn hex_hamming_distance_pack(a: &str, b: &str) -> Result<u64, &'static str> {
    if a.len() != b.len() {
        return Err("strings are NOT the same length");
    }
    if a.is_empty() {
        return Ok(0);
    }
    unsafe { neon_simd::hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()) }
}

/// Set the SIMD algorithm used for hamming distance calculations.
///
/// Valid algorithm names:
/// - `"avx512"` / `"avx-512"` — AVX-512 BITALG (requires avx512bw + avx512bitalg)
/// - `"avx2"` / `"avx"` / `"extra"` — AVX2
/// - `"sse41"` / `"sse"` — SSE4.1
/// - `"neon"` — ARM NEON (aarch64 only)
/// - `"native"` / `"popcount"` — platform native
/// - `"classic"` — scalar fallback
///
/// Returns `Ok(())` on success, `Err` if the CPU doesn't support the requested algorithm.
pub fn set_algorithm(algo_name: &str) -> Result<(), &'static str> {
    match algo_name.to_lowercase().as_str() {
        "avx512" | "avx-512" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                    CURRENT_ALGO.store(ALGO_AVX512, Ordering::Relaxed);
                    return Ok(());
                }
                return Err("CPU doesn't support AVX-512 BITALG");
            }
            #[cfg(not(target_arch = "x86_64"))]
            Err("AVX-512 not available on this architecture")
        }
        "extra" | "avx" | "avx2" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    CURRENT_ALGO.store(ALGO_AVX2, Ordering::Relaxed);
                    return Ok(());
                }
                return Err("CPU doesn't support AVX2");
            }
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                Ok(())
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            Err("not available on this architecture")
        }
        "sse41" | "sse" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.1") {
                    CURRENT_ALGO.store(ALGO_SSE41, Ordering::Relaxed);
                    return Ok(());
                }
                Err("CPU doesn't support SSE4.1")
            }
            #[cfg(not(target_arch = "x86_64"))]
            Err("SSE not available on this architecture")
        }
        "neon" => {
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                Ok(())
            }
            #[cfg(not(target_arch = "aarch64"))]
            Err("NEON not available on this architecture")
        }
        "native" | "popcount" => {
            CURRENT_ALGO.store(ALGO_NATIVE, Ordering::Relaxed);
            Ok(())
        }
        "classic" => {
            CURRENT_ALGO.store(ALGO_CLASSIC, Ordering::Relaxed);
            Ok(())
        }
        _ => Err("unknown algorithm"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_hamming() {
        assert_eq!(hex_hamming_distance("deadbeef", "00000000").unwrap(), 24);
        assert_eq!(hex_hamming_distance("ffff", "0000").unwrap(), 16);
        assert_eq!(hex_hamming_distance("0000", "0000").unwrap(), 0);
        assert_eq!(hex_hamming_distance("f", "0").unwrap(), 4);
    }

    #[test]
    fn test_mixed_case() {
        assert_eq!(hex_hamming_distance("DEADBEEF", "deadbeef").unwrap(), 0);
        assert_eq!(hex_hamming_distance("AbCdEf", "abcdef").unwrap(), 0);
        assert_eq!(hex_hamming_distance("aAbBcC", "AABBCC").unwrap(), 0);
    }

    #[test]
    fn test_long_strings_32plus() {
        // 32 chars — exercises the SSE pack/32-char loop
        let a32 = "f".repeat(32);
        let b32 = "0".repeat(32);
        assert_eq!(hex_hamming_distance(&a32, &b32).unwrap(), 128);

        // 64 chars — exercises the AVX2 64-char loop
        let a64 = "f".repeat(64);
        let b64 = "0".repeat(64);
        assert_eq!(hex_hamming_distance(&a64, &b64).unwrap(), 256);

        // 128 chars — multiple AVX2 iterations
        let a128 = "f".repeat(128);
        let b128 = "0".repeat(128);
        assert_eq!(hex_hamming_distance(&a128, &b128).unwrap(), 512);

        // 254 chars — AVX2 loop + SSE tail + scalar tail
        let a254 = "f".repeat(254);
        let b254 = "0".repeat(254);
        assert_eq!(hex_hamming_distance(&a254, &b254).unwrap(), 1016);
    }

    #[test]
    fn test_long_mixed_content() {
        // Mixed hex chars to exercise all parse paths across SIMD lanes
        let a = "0123456789abcdef".repeat(8); // 128 chars
        let b = "fedcba9876543210".repeat(8);
        let result = hex_hamming_distance(&a, &b).unwrap();
        // Each pair: 0^f=f(4), 1^e=f(4), 2^d=f(4), 3^c=f(4),
        //            4^b=f(4), 5^a=f(4), 6^9=f(4), 7^8=f(4),
        //            8^7=f(4), 9^6=f(4), a^5=f(4), b^4=f(4),
        //            c^3=f(4), d^2=f(4), e^1=f(4), f^0=f(4) = 64 per 16 chars
        assert_eq!(result, 64 * 8);

        // Mixed case in long string
        let a_mixed = "AaBbCcDdEeFf0011".repeat(4); // 64 chars
        let b_mixed = "aAbBcCdDeEfF0011".repeat(4);
        assert_eq!(hex_hamming_distance(&a_mixed, &b_mixed).unwrap(), 0);
    }

    #[test]
    fn test_invalid_chars() {
        assert!(hex_hamming_distance("zz", "00").is_err());
        assert!(hex_hamming_distance("gg", "00").is_err());
        assert!(hex_hamming_distance("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "00000000000000000000000000000000ff").is_err());
        assert!(hex_hamming_distance("``````````````````````````````````", "00000000000000000000000000000000ff").is_err());
    }

    #[test]
    fn test_length_mismatch() {
        assert!(hex_hamming_distance("ff", "f").is_err());
    }

    #[test]
    fn test_empty() {
        assert_eq!(hex_hamming_distance("", "").unwrap(), 0);
    }

    #[test]
    fn test_bytes_basic() {
        assert_eq!(bytes_hamming_distance(b"\xff", b"\x00").unwrap(), 8);
        assert_eq!(bytes_hamming_distance(b"\xde\xad\xbe\xef", b"\x00\x00\x00\x00").unwrap(), 24);
    }
}
