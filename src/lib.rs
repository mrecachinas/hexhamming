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

mod classic;
mod hex;
mod native;
#[cfg(target_arch = "x86_64")]
mod x86_simd;
#[cfg(target_arch = "aarch64")]
mod neon_simd;
#[cfg(feature = "python")]
mod python;
mod api;
#[cfg(test)]
mod tests;

pub use api::*;

/// Lookup table for popcount of 4-bit values (0-15)
pub(crate) const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Branchless hex character to nibble lookup table (256 entries)
/// Invalid characters map to 0xFF for easy detection
pub(crate) const HEX_LOOKUP: [u8; 256] = {
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
pub(crate) const ALGO_CLASSIC: u8 = 0;
pub(crate) const ALGO_NATIVE: u8 = 1;
#[cfg(target_arch = "x86_64")]
pub(crate) const ALGO_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
pub(crate) const ALGO_AVX2: u8 = 3;
#[cfg(target_arch = "x86_64")]
pub(crate) const ALGO_AVX512: u8 = 5;
#[cfg(target_arch = "aarch64")]
pub(crate) const ALGO_NEON: u8 = 4;

/// Thresholds for algorithm selection (tuned for typical CPU cache behavior)
#[allow(dead_code)]
pub(crate) const SCALAR_THRESHOLD: usize = 16; // Below this, scalar may beat SIMD
#[allow(dead_code)]
pub(crate) const SSE_THRESHOLD: usize = 64;    // Use SSE for medium strings

/// Current algorithm selection (global state)
pub(crate) static CURRENT_ALGO: AtomicU8 = AtomicU8::new(ALGO_NATIVE);

/// Dispatch to appropriate byte distance implementation based on current algorithm
#[inline(always)]
pub(crate) fn hamming_distance_bytes_dispatch(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => classic::hamming_distance_bytes_classic(a, b, max_dist),

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX512 => {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                unsafe { x86_simd::hamming_distance_bytes_avx512(a, b, max_dist) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
            } else {
                native::hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX2 => {
            if is_x86_feature_detected!("avx2") {
                unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
            } else {
                native::hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_SSE41 => {
            if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
                unsafe { x86_simd::hamming_distance_bytes_sse(a, b, max_dist) }
            } else {
                native::hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        // On ARM64, NEON and native use the same optimized implementation
        // Benchmarks show that Rust's auto-vectorized count_ones() on Apple Silicon
        // is faster than handwritten NEON intrinsics (vcntq_u8 + horizontal sums)
        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => native::hamming_distance_bytes_native(a, b, max_dist),

        _ => native::hamming_distance_bytes_native(a, b, max_dist),
    }
}

/// Dispatch to appropriate string distance implementation based on current algorithm
#[inline(always)]
pub(crate) fn hamming_distance_string_dispatch(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
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

    classic::hamming_distance_string_classic(a, b)
}
