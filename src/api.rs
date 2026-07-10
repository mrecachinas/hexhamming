use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    select_bytes_kernel_for_width, BytesKernel, ALGO_CLASSIC, ALGO_NATIVE, CURRENT_ALGO,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};

use rayon::prelude::*;
use std::sync::atomic::Ordering;

/// Minimum total byte size of big_array before we use rayon parallel paths.
const PAR_THRESHOLD_BYTES: usize = 256 * 1024;
/// Keep byte-array scans to a small number of coarse jobs. More workers spend
/// more time scheduling these very small per-record calculations than running
/// them on current many-core CPUs.
const PAR_JOBS: usize = 4;

#[inline]
fn partition_element_ranges(num_elements: usize) -> [(usize, usize); PAR_JOBS] {
    let base = num_elements / PAR_JOBS;
    let remainder = num_elements % PAR_JOBS;
    let mut ranges = [(0, 0); PAR_JOBS];
    let mut start = 0;

    for (job, range) in ranges.iter_mut().enumerate() {
        let end = start + base + usize::from(job < remainder);
        *range = (start, end);
        start = end;
    }

    ranges
}

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
#[inline]
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
    Ok(hamming_distance_bytes_dispatch(a, b, max_dist) != u64::MAX)
}

/// Find the first element in a byte array within a specified Hamming distance.
///
/// Returns the index of the first matching element, or `None`.
pub fn bytes_array_first_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Result<Option<usize>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    // `first` has early-exit semantics: the serial scan returns as soon as the
    // first match is found, which is essentially free for early/common matches.
    // Parallelizing this requires a full non-short-circuiting scan to compute
    // the minimum matching index, which is dramatically slower for early/mid
    // matches and cannot beat serial for a match at index 0. Always go serial.
    Ok(serial_first_within_dist(
        big_array,
        small_array,
        max_dist,
        select_bytes_kernel_for_width(small_array.len()),
    ))
}

#[inline]
fn serial_first_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    kernel: BytesKernel,
) -> Option<usize> {
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        if kernel(chunk, small_array, max_dist) != u64::MAX {
            return Some(i);
        }
    }
    None
}

/// Find the element in a byte array with the smallest Hamming distance.
///
/// Returns `Some((distance, index))` of the best match, or `None` if none within max_dist.
pub fn bytes_array_best_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Result<Option<(u64, usize)>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    let kernel = select_bytes_kernel_for_width(small_array.len());
    if big_array.len() < PAR_THRESHOLD_BYTES {
        return Ok(serial_best_within_dist(
            big_array,
            small_array,
            max_dist,
            kernel,
        ));
    }
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let ranges = partition_element_ranges(num_elements);

    Ok(ranges
        .par_iter()
        .with_max_len(1)
        .map(|&(start, end)| {
            let chunk = &big_array[start * elem_size..end * elem_size];
            serial_best_within_dist(chunk, small_array, max_dist, kernel)
                .map(|(distance, index)| (distance, index + start))
        })
        .reduce(|| None, merge_best))
}

#[inline]
fn merge_best(a: Option<(u64, usize)>, b: Option<(u64, usize)>) -> Option<(u64, usize)> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) if x.0 < y.0 || (x.0 == y.0 && x.1 <= y.1) => Some(x),
        (Some(_), Some(y)) => Some(y),
    }
}

#[inline]
fn serial_best_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    kernel: BytesKernel,
) -> Option<(u64, usize)> {
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let mut best: Option<(u64, usize)> = None;
    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        let threshold = best
            .map(|(d, _)| (d as i64).saturating_sub(1))
            .unwrap_or(max_dist);
        let d = kernel(chunk, small_array, threshold);
        if d == u64::MAX {
            continue;
        }
        if best.is_none() || d < best.unwrap().0 {
            best = Some((d, i));
            if d == 0 {
                return best;
            }
        }
    }
    best
}

/// Find all elements in a byte array within a specified Hamming distance.
///
/// Returns a Vec of `(distance, index)` tuples in ascending index order.
pub fn bytes_array_all_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Result<Vec<(u64, usize)>, &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    let kernel = select_bytes_kernel_for_width(small_array.len());
    if big_array.len() < PAR_THRESHOLD_BYTES {
        return Ok(serial_all_within_dist(
            big_array,
            small_array,
            max_dist,
            kernel,
        ));
    }
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let ranges = partition_element_ranges(num_elements);
    let per_job: Vec<Vec<(u64, usize)>> = ranges
        .par_iter()
        .with_max_len(1)
        .map(|&(start, end)| {
            let chunk = &big_array[start * elem_size..end * elem_size];
            serial_all_within_dist(chunk, small_array, max_dist, kernel)
                .into_iter()
                .map(|(distance, index)| (distance, index + start))
                .collect()
        })
        .collect();

    // The indexed parallel iterator preserves range order, and each serial
    // result is already ordered, so flattening preserves ascending indices.
    let result_count = per_job.iter().map(Vec::len).sum();
    let mut results = Vec::with_capacity(result_count);
    for job_results in per_job {
        results.extend(job_results);
    }
    Ok(results)
}

#[inline]
fn serial_all_within_dist(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    kernel: BytesKernel,
) -> Vec<(u64, usize)> {
    let elem_size = small_array.len();
    let num_elements = big_array.len() / elem_size;
    let mut results = Vec::new();
    for i in 0..num_elements {
        let chunk = &big_array[i * elem_size..(i + 1) * elem_size];
        let d = kernel(chunk, small_array, max_dist);
        if d != u64::MAX {
            results.push((d, i));
        }
    }
    results
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
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg")
                {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_algorithm_classic_and_native() {
        set_algorithm("classic").unwrap();
        assert_eq!(hex_hamming_distance("deadbeef", "00000000").unwrap(), 24);
        set_algorithm("native").unwrap();
        assert_eq!(hex_hamming_distance("deadbeef", "00000000").unwrap(), 24);
    }

    #[test]
    fn set_algorithm_unknown() {
        assert!(set_algorithm("bogus").is_err());
    }

    #[test]
    fn bytes_within_dist_basic() {
        assert_eq!(bytes_within_dist(b"\xff", b"\xfe", 2).unwrap(), true);
        assert_eq!(bytes_within_dist(b"\xff", b"\x00", 2).unwrap(), false);
    }

    #[test]
    fn bytes_within_dist_errors() {
        assert!(bytes_within_dist(b"", b"\xff", 1).is_err());
        assert!(bytes_within_dist(b"\xff", b"\xff\x00", 1).is_err());
    }

    #[test]
    fn array_first_within_dist_test() {
        let big = b"\xaa\xbb\xcc\xff";
        let small = b"\xff";
        // \xaa vs \xff = dist 4, within max_dist 4
        assert_eq!(
            bytes_array_first_within_dist(big, small, 4).unwrap(),
            Some(0)
        );
        // Only exact match at index 3
        assert_eq!(
            bytes_array_first_within_dist(big, small, 0).unwrap(),
            Some(3)
        );
        // dist(\x00, \xff) = 8, exceeds max_dist 1
        assert_eq!(
            bytes_array_first_within_dist(b"\x00", b"\xff", 1).unwrap(),
            None
        );
    }

    #[test]
    fn array_best_within_dist() {
        // \xfe is distance 1 from \xff, \xaa is distance 4
        let big = b"\xaa\xfe\xff";
        let small = b"\xff";
        let result = bytes_array_best_within_dist(big, small, 8).unwrap();
        assert_eq!(result, Some((0, 2))); // exact match at index 2
    }

    #[test]
    fn array_all_within_dist() {
        let big = b"\xaa\xfe\xff";
        let small = b"\xff";
        let result = bytes_array_all_within_dist(big, small, 8).unwrap();
        assert_eq!(result.len(), 3);
        // Last entry should be exact match
        assert_eq!(result[2], (0, 2));
    }

    #[test]
    fn array_errors() {
        assert!(bytes_array_first_within_dist(b"\xff", b"", 1).is_err()); // empty small
        assert!(bytes_array_first_within_dist(b"\xaa\xbb\xcc", b"\xff\xff", 1).is_err());
        // not a multiple
    }

    // -----------------------------------------------------------------------
    // Wave 2b: rayon parallelization regression tests
    // -----------------------------------------------------------------------

    /// Build a big array of `num_elements` chunks of size `elem_size`, all filled
    /// with `fill_byte`, then overwrite specific indices with `match_bytes`.
    fn make_batch(
        elem_size: usize,
        num_elements: usize,
        fill_byte: u8,
        match_indices: &[usize],
        match_bytes: &[u8],
    ) -> Vec<u8> {
        let mut big = vec![fill_byte; elem_size * num_elements];
        for &idx in match_indices {
            big[idx * elem_size..(idx + 1) * elem_size].copy_from_slice(match_bytes);
        }
        big
    }

    #[test]
    fn parallel_ranges_cover_elements_once() {
        for num_elements in [1, 3, 4, 5, 7, 8, 9, 100, 100_001] {
            let ranges = partition_element_ranges(num_elements);
            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges[PAR_JOBS - 1].1, num_elements);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].1, pair[1].0);
            }
            let lengths: Vec<usize> = ranges.iter().map(|&(start, end)| end - start).collect();
            assert!(lengths.iter().max().unwrap() - lengths.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn first_within_dist_small_batch() {
        // Below PAR_THRESHOLD_BYTES → serial path
        let elem_size = 4;
        let n = 100; // 400 bytes < parallel threshold
        let needle = vec![0x00u8; elem_size];
        let big = make_batch(elem_size, n, 0xFF, &[50], &needle);
        assert_eq!(
            bytes_array_first_within_dist(&big, &needle, 0).unwrap(),
            Some(50)
        );
    }

    #[test]
    fn first_within_dist_large_batch() {
        // Large batch → serial early-exit path (first is never parallelized)
        let elem_size = 16;
        let n = 100_000; // 1.6 MB > 64KB
        let needle = vec![0x00u8; elem_size];
        let big = make_batch(elem_size, n, 0xFF, &[50], &needle);
        assert_eq!(
            bytes_array_first_within_dist(&big, &needle, 0).unwrap(),
            Some(50)
        );
    }

    #[test]
    fn first_within_dist_returns_lowest_index() {
        let elem_size = 16;
        let n = 100_000;
        let needle = vec![0x00u8; elem_size];
        let big = make_batch(elem_size, n, 0xFF, &[50, 500, 5000, 50000], &needle);
        // Must return 50, the lowest matching index
        assert_eq!(
            bytes_array_first_within_dist(&big, &needle, 0).unwrap(),
            Some(50)
        );
    }

    #[test]
    fn best_within_dist_small_batch() {
        let elem_size = 4;
        let n = 100;
        let needle = vec![0x00u8; elem_size];
        // elem at index 30: 1 bit diff, elem at index 60: exact match
        let mut big = vec![0xFFu8; elem_size * n];
        big[60 * elem_size..(60 + 1) * elem_size].copy_from_slice(&needle);
        let mut one_bit = vec![0x00u8; elem_size];
        one_bit[0] = 0x01;
        big[30 * elem_size..(30 + 1) * elem_size].copy_from_slice(&one_bit);

        let result = bytes_array_best_within_dist(&big, &needle, 100).unwrap();
        assert_eq!(result, Some((0, 60)));
    }

    #[test]
    fn best_within_dist_large_batch() {
        let elem_size = 16;
        let n = 100_000;
        let needle = vec![0x00u8; elem_size];
        let mut big = vec![0xFFu8; elem_size * n];
        // Place exact match at index 75000
        big[75000 * elem_size..(75000 + 1) * elem_size].copy_from_slice(&needle);
        // Place 1-bit diff at index 25000
        let mut one_bit = vec![0x00u8; elem_size];
        one_bit[0] = 0x01;
        big[25000 * elem_size..(25000 + 1) * elem_size].copy_from_slice(&one_bit);

        let result = bytes_array_best_within_dist(&big, &needle, 200).unwrap();
        assert_eq!(result, Some((0, 75000)));
    }

    #[test]
    fn best_within_dist_tiebreak_lowest_index() {
        // Two elements with the same minimum distance — lower index must win.
        let elem_size = 16;
        let n = 100_000;
        let needle = vec![0x00u8; elem_size];
        let mut big = vec![0xFFu8; elem_size * n];
        // Exact matches at indices 300 and 700
        big[300 * elem_size..(300 + 1) * elem_size].copy_from_slice(&needle);
        big[700 * elem_size..(700 + 1) * elem_size].copy_from_slice(&needle);

        let result = bytes_array_best_within_dist(&big, &needle, 200).unwrap();
        assert_eq!(result, Some((0, 300)));
    }

    #[test]
    fn best_within_dist_tiebreak_lowest_index_small() {
        // Same test but below threshold (serial path)
        let elem_size = 4;
        let n = 10;
        let needle = vec![0x00u8; elem_size];
        let mut big = vec![0xFFu8; elem_size * n];
        big[3 * elem_size..(3 + 1) * elem_size].copy_from_slice(&needle);
        big[7 * elem_size..(7 + 1) * elem_size].copy_from_slice(&needle);

        let result = bytes_array_best_within_dist(&big, &needle, 200).unwrap();
        assert_eq!(result, Some((0, 3)));
    }

    #[test]
    fn all_within_dist_small_batch() {
        let elem_size = 4;
        let n = 100;
        let needle = vec![0x00u8; elem_size];
        let big = make_batch(elem_size, n, 0xFF, &[5, 20, 50, 99], &needle);
        let result = bytes_array_all_within_dist(&big, &needle, 0).unwrap();
        let indices: Vec<usize> = result.iter().map(|&(_, i)| i).collect();
        assert_eq!(indices, vec![5, 20, 50, 99]);
    }

    #[test]
    fn all_within_dist_large_batch_ordering() {
        // Matches at specific indices in a large batch — must come back sorted by index.
        let elem_size = 16;
        let n = 100_000;
        let needle = vec![0x00u8; elem_size];
        let match_at = vec![5, 100, 200, 500, 50000, 99999];
        let big = make_batch(elem_size, n, 0xFF, &match_at, &needle);
        let result = bytes_array_all_within_dist(&big, &needle, 0).unwrap();
        let indices: Vec<usize> = result.iter().map(|&(_, i)| i).collect();
        assert_eq!(indices, match_at);
    }

    #[test]
    fn serial_and_parallel_produce_identical_results_first() {
        let elem_size = 8;
        let needle = vec![0x00u8; elem_size];
        let match_at = &[10, 50, 100];
        // Small batch (serial)
        let small_n = 200; // 1600 bytes
        let big_small = make_batch(elem_size, small_n, 0xFF, match_at, &needle);
        let serial = bytes_array_first_within_dist(&big_small, &needle, 0).unwrap();
        // Large batch (parallel) — same logical positions
        let large_n = 100_000;
        let big_large = make_batch(elem_size, large_n, 0xFF, match_at, &needle);
        let parallel = bytes_array_first_within_dist(&big_large, &needle, 0).unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn serial_and_parallel_produce_identical_results_best() {
        let elem_size = 8;
        let needle = vec![0x00u8; elem_size];
        // Place elements with different distances
        let mut small_big = vec![0xFFu8; elem_size * 200];
        let mut large_big = vec![0xFFu8; elem_size * 100_000];

        // exact match at 50, 1-bit at 30
        let mut one_bit = vec![0x00u8; elem_size];
        one_bit[0] = 0x01;
        for big in [&mut small_big, &mut large_big] {
            big[30 * elem_size..(30 + 1) * elem_size].copy_from_slice(&one_bit);
            big[50 * elem_size..(50 + 1) * elem_size].copy_from_slice(&needle);
        }

        let serial = bytes_array_best_within_dist(&small_big, &needle, 200).unwrap();
        let parallel = bytes_array_best_within_dist(&large_big, &needle, 200).unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn serial_and_parallel_produce_identical_results_all() {
        let elem_size = 8;
        let needle = vec![0x00u8; elem_size];
        let match_at = &[5, 20, 50, 100];

        let small_big = make_batch(elem_size, 200, 0xFF, match_at, &needle);
        let large_big = make_batch(elem_size, 100_000, 0xFF, match_at, &needle);

        let serial = bytes_array_all_within_dist(&small_big, &needle, 0).unwrap();
        let parallel = bytes_array_all_within_dist(&large_big, &needle, 0).unwrap();
        // Both should find the same 4 matches at same indices with same distances
        assert_eq!(serial.len(), parallel.len());
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s, p);
        }
    }
}
