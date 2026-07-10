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
//!
//! This file is the central routing layer: public Rust APIs live in `api`,
//! optional PyO3 bindings live in `python`, and the hot loops are delegated to
//! scalar or architecture-specific modules after lightweight runtime checks.

use std::sync::atomic::{AtomicU8, Ordering};

// `api` exposes the Rust API re-exported below. When the `python` feature is
// enabled, `python` registers the PyO3 module and maps Python errors/GIL policy
// onto the same dispatch functions used by Rust callers.
mod api;
mod classic;
mod hex;
mod native;
#[cfg(target_arch = "aarch64")]
mod neon_simd;
#[cfg(feature = "python")]
mod python;
#[cfg(test)]
mod tests;
#[cfg(target_arch = "x86_64")]
mod x86_simd;

pub use api::*;

/// Lookup table for popcount of 4-bit values (0-15).
/// Hex string distance is computed one nibble at a time, so this avoids a
/// wider popcount for the scalar tail and fallback paths.
pub(crate) const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Branchless ASCII hex character to nibble lookup table.
/// Invalid characters map to 0xFF, letting hot loops OR several parsed nibbles
/// together and test the high bits once instead of branching per character.
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
        if i == 255 {
            break;
        }
        i += 1;
    }
    table
};

/// Algorithm selection constants stored in `CURRENT_ALGO`.
/// Not every value exists on every target; `cfg` keeps unsupported SIMD code
/// out of the build and each dispatcher still verifies runtime CPU features.
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

/// Thresholds for algorithm selection (tuned for typical CPU cache behavior).
/// Very short inputs stay scalar because SIMD setup and horizontal reduction
/// can cost more than the distance calculation itself.
#[allow(dead_code)]
pub(crate) const SCALAR_THRESHOLD: usize = 16; // Below this, scalar may beat SIMD
#[allow(dead_code)]
pub(crate) const SSE_THRESHOLD: usize = 64; // Use SSE for medium strings

/// Current algorithm selection shared by Rust and Python entry points.
/// Relaxed ordering is enough: this is a best-effort performance knob, and all
/// implementations produce identical results.
pub(crate) static CURRENT_ALGO: AtomicU8 = AtomicU8::new(ALGO_NATIVE);

pub(crate) type BytesKernel = fn(&[u8], &[u8], i64) -> u64;

#[cfg(target_arch = "x86_64")]
#[inline]
fn hamming_distance_bytes_avx512(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    unsafe { x86_simd::hamming_distance_bytes_avx512(a, b, max_dist) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn hamming_distance_bytes_sse(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    unsafe { x86_simd::hamming_distance_bytes_sse(a, b, max_dist) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn hamming_distance_bytes_neon(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    unsafe { neon_simd::hamming_distance_bytes_neon(a, b, max_dist) }
}

/// Resolve the byte-distance backend once for callers that perform many
/// comparisons with the same algorithm selection.
#[inline]
pub(crate) fn select_bytes_kernel() -> BytesKernel {
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => classic::hamming_distance_bytes_classic,

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX512 => {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                hamming_distance_bytes_avx512
            } else if is_x86_feature_detected!("avx2") {
                hamming_distance_bytes_avx2
            } else {
                native::hamming_distance_bytes_native
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX2 => {
            if is_x86_feature_detected!("avx2") {
                hamming_distance_bytes_avx2
            } else {
                native::hamming_distance_bytes_native
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_SSE41 => {
            if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
                hamming_distance_bytes_sse
            } else {
                native::hamming_distance_bytes_native
            }
        }

        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => hamming_distance_bytes_neon,

        _ => native::hamming_distance_bytes_native,
    }
}

/// Dispatch to the byte distance implementation selected by `CURRENT_ALGO`.
///
/// Callers validate equal lengths before reaching this layer. `max_dist < 0`
/// means "compute the full distance"; otherwise implementations may return the
/// `u64::MAX` sentinel as soon as they know the distance exceeds the cutoff.
#[inline(always)]
pub(crate) fn hamming_distance_bytes_dispatch(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let algo = CURRENT_ALGO.load(Ordering::Relaxed);

    match algo {
        ALGO_CLASSIC => classic::hamming_distance_bytes_classic(a, b, max_dist),

        #[cfg(target_arch = "x86_64")]
        ALGO_AVX512 => {
            // A requested SIMD path is only used when the running CPU supports
            // it. Fallbacks preserve behavior on wheels built for many CPUs.
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
            // AVX2 byte distance uses a nibble popcount shuffle table; if AVX2
            // is unavailable, native scalar popcount is the safe fallback.
            if is_x86_feature_detected!("avx2") {
                unsafe { x86_simd::hamming_distance_bytes_avx2(a, b, max_dist) }
            } else {
                native::hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        #[cfg(target_arch = "x86_64")]
        ALGO_SSE41 => {
            // SSE path also requires POPCNT for its reduction helpers.
            if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
                unsafe { x86_simd::hamming_distance_bytes_sse(a, b, max_dist) }
            } else {
                native::hamming_distance_bytes_native(a, b, max_dist)
            }
        }

        // On ARM64, NEON counts set bits directly with vcntq_u8 after XORing.
        #[cfg(target_arch = "aarch64")]
        ALGO_NEON => unsafe { neon_simd::hamming_distance_bytes_neon(a, b, max_dist) },

        _ => native::hamming_distance_bytes_native(a, b, max_dist),
    }
}

/// Dispatch to the hex-string distance implementation selected by
/// `CURRENT_ALGO`.
///
/// The public APIs check length and empty inputs first. SIMD string paths parse
/// ASCII hex and count nibble differences in vector batches; scalar fallback
/// uses `HEX_LOOKUP` + `LOOKUP` and reports invalid hex characters.
#[inline(always)]
pub(crate) fn hamming_distance_string_dispatch(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        // Try widest vectors first. `ALGO_NATIVE` acts as auto-dispatch for
        // hex strings because vectorized parsing is usually the main win.
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
        // NEON packs pairs of hex nibbles into bytes before using vector
        // popcount, avoiding per-character scalar work on longer strings.
        if algo == ALGO_NEON || algo == ALGO_NATIVE {
            return unsafe { neon_simd::hamming_distance_string_neon_pack(a, b) };
        }
    }

    classic::hamming_distance_string_classic(a, b)
}

/// Dispatch to a hex-string implementation with early-exit at `max_dist`.
///
/// Returns `Ok(u64::MAX)` when the distance exceeds the cutoff. That sentinel
/// is internal: public "within distance" APIs translate it into `false`.
#[inline(always)]
pub(crate) fn hamming_distance_string_dispatch_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        // SIMD variants still validate hex input. Their cutoff checks are
        // batched, so they may process past the exact crossing point but never
        // return a false "within distance" result.
        if (algo == ALGO_AVX512 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512bitalg")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_avx512_with_max(a, b, max_dist) };
        }
        if (algo == ALGO_AVX512 || algo == ALGO_AVX2 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_avx2_with_max(a, b, max_dist) };
        }
        if (algo == ALGO_AVX512 || algo == ALGO_AVX2 || algo == ALGO_SSE41 || algo == ALGO_NATIVE)
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("popcnt")
        {
            return unsafe { x86_simd::hamming_distance_string_sse_with_max(a, b, max_dist) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        if algo == ALGO_NEON || algo == ALGO_NATIVE {
            return unsafe {
                neon_simd::hamming_distance_string_neon_pack_with_max(a, b, max_dist)
            };
        }
    }

    // Classic scalar fallback with early-exit. Odd-length hex strings are valid
    // here: each ASCII hex char is one 4-bit nibble, not a required byte pair.
    let length = a.len();
    let mut difference: u64 = 0;
    for i in 0..length {
        unsafe {
            let val1 = hex::hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex::hex_char_to_nibble(*b.get_unchecked(i));
            // `hex_char_to_nibble` returns 0xFF for invalid input; checking the
            // high nibble catches that sentinel for either side.
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        }
        if difference > max_dist {
            return Ok(u64::MAX);
        }
    }
    Ok(difference)
}
