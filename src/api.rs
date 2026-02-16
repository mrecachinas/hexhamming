use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    ALGO_CLASSIC, ALGO_NATIVE,
    CURRENT_ALGO,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};

use std::sync::atomic::Ordering;

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

/// Check if two byte arrays are within a specified Hamming distance.
///
/// Returns `Ok(true)` if distance <= max_dist, `Ok(false)` otherwise.
pub fn bytes_within_dist(a: &[u8], b: &[u8], max_dist: i64) -> Result<bool, &'static str> {
    if a.is_empty() || b.is_empty() {
        return Err("array size must be >0");
    }
    if a.len() != b.len() {
        return Err("array sizes need to be the same");
    }
    Ok(hamming_distance_bytes_dispatch(a, b, max_dist) == 1)
}

/// Find the first element in a byte array within a specified Hamming distance.
///
/// Returns the index of the first matching element, or `None`.
pub fn bytes_array_first_within_dist(big_array: &[u8], small_array: &[u8], max_dist: i64) -> Result<Option<usize>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        if hamming_distance_bytes_dispatch(chunk, small_array, max_dist) == 1 {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// Find the element in a byte array with the smallest Hamming distance.
///
/// Returns `Some((distance, index))` of the best match, or `None` if none within max_dist.
pub fn bytes_array_best_within_dist(big_array: &[u8], small_array: &[u8], max_dist: i64) -> Result<Option<(u64, usize)>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let mut best_dist: i64 = -1;
    let mut best_index: Option<usize> = None;

    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        let threshold = if best_dist >= 0 { best_dist - 1 } else { max_dist };
        if hamming_distance_bytes_dispatch(chunk, small_array, threshold) == 0 {
            continue;
        }
        let dist = hamming_distance_bytes_dispatch(chunk, small_array, -1) as i64;
        if best_dist < 0 || dist < best_dist {
            best_dist = dist;
            best_index = Some(i);
        }
    }
    Ok(best_index.map(|idx| (best_dist as u64, idx)))
}

/// Find all elements in a byte array within a specified Hamming distance.
///
/// Returns a Vec of `(distance, index)` tuples.
pub fn bytes_array_all_within_dist(big_array: &[u8], small_array: &[u8], max_dist: i64) -> Result<Vec<(u64, usize)>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let mut results = Vec::new();

    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        if hamming_distance_bytes_dispatch(chunk, small_array, max_dist) == 0 {
            continue;
        }
        let dist = hamming_distance_bytes_dispatch(chunk, small_array, -1);
        results.push((dist, i));
    }
    Ok(results)
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
    unsafe { crate::neon_simd::hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()) }
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
                CURRENT_ALGO.store(crate::ALGO_NEON, Ordering::Relaxed);
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
                CURRENT_ALGO.store(crate::ALGO_NEON, Ordering::Relaxed);
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
