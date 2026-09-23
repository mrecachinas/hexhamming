#[cfg(target_arch = "aarch64")]
use crate::ALGO_NEON;
use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    select_bytes_kernel_for_width, BytesKernel, ALGO_CLASSIC, ALGO_NATIVE, CURRENT_ALGO,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};

use crate::par;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scans of at least this many bytes are split across threads when each record
/// costs a kernel call (widths without a block scanner). Measured on an Apple
/// M4 Max with the scan pool: 20-byte records are 2.0-3.5x faster at 128 KiB
/// and ~3x at 256 KiB.
pub(crate) const PAR_THRESHOLD_BYTES: usize = 64 * 1024;
/// Block scanners are cheaper per byte, so they need a larger scan before
/// splitting pays off: 16- and 64-byte records are 0.8x at 256 KiB, 1.4-1.7x
/// at 512 KiB and 2.5x at 1 MiB.
pub(crate) const FIXED_WIDTH_PAR_THRESHOLD_BYTES: usize = 512 * 1024;
/// `first` and `best` scan a serial prefix before going parallel, so early
/// matches (and early exact matches for `best`) never pay for waking workers.
/// It is 1/64 of the scan, clamped to 16-256 KiB, which keeps its cost on scans
/// without early matches to ~1.5% of the serial time.
const SERIAL_PREFIX_MIN_BYTES: usize = 16 * 1024;
const SERIAL_PREFIX_MAX_BYTES: usize = 256 * 1024;
const SERIAL_PREFIX_SHARE: usize = 64;

pub(crate) type ArrayFirstScanner = fn(&[u8], &[u8], i64) -> Option<usize>;
pub(crate) type ArrayBestScanner = fn(&[u8], &[u8], i64) -> Option<(u64, usize)>;
pub(crate) type ArrayAllScanner = fn(&[u8], &[u8], i64) -> Vec<(u64, usize)>;

#[derive(Clone, Copy)]
pub(crate) struct ArrayScanner {
    pub(crate) first: ArrayFirstScanner,
    pub(crate) best: ArrayBestScanner,
    pub(crate) all: ArrayAllScanner,
}

#[inline]
pub(crate) fn select_array_scanner_for_width(width: usize) -> Option<ArrayScanner> {
    #[cfg(target_arch = "aarch64")]
    {
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        if algo != ALGO_NATIVE && algo != ALGO_NEON {
            return None;
        }

        return match width {
            8 => Some(ArrayScanner {
                first: crate::neon_block::array_first_8,
                best: crate::neon_block::array_best_8,
                all: crate::neon_block::array_all_8,
            }),
            16 => Some(ArrayScanner {
                first: crate::neon_block::array_first_16,
                best: crate::neon_block::array_best_16,
                all: crate::neon_block::array_all_16,
            }),
            32 => Some(ArrayScanner {
                first: crate::neon_block::array_first_32,
                best: crate::neon_block::array_best_32,
                all: crate::neon_block::array_all_32,
            }),
            64 => Some(ArrayScanner {
                first: crate::neon_block::array_first_64,
                best: crate::neon_block::array_best_64,
                all: crate::neon_block::array_all_64,
            }),
            _ => None,
        };
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Only opt in to the AVX-512 block scanners when the user hasn't
        // explicitly requested a narrower or scalar backend, and only when the
        // host supports VPOPCNTDQ (every AVX-512 BITALG CPU does). Explicit
        // "classic"/"sse"/"avx2" keep their own paths.
        let algo = CURRENT_ALGO.load(Ordering::Relaxed);
        if (algo == ALGO_NATIVE || algo == ALGO_AVX512)
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("popcnt")
        {
            let scanner = match width {
                8 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx512_8_dispatch,
                    best: crate::x86_simd::array_best_avx512_8_dispatch,
                    all: crate::x86_simd::array_all_avx512_8_dispatch,
                }),
                16 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx512_16_dispatch,
                    best: crate::x86_simd::array_best_avx512_16_dispatch,
                    all: crate::x86_simd::array_all_avx512_16_dispatch,
                }),
                32 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx512_32_dispatch,
                    best: crate::x86_simd::array_best_avx512_32_dispatch,
                    all: crate::x86_simd::array_all_avx512_32_dispatch,
                }),
                64 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx512_64_dispatch,
                    best: crate::x86_simd::array_best_avx512_64_dispatch,
                    all: crate::x86_simd::array_all_avx512_64_dispatch,
                }),
                _ => None,
            };
            if scanner.is_some() || algo == ALGO_AVX512 {
                return scanner;
            }
        }
        if (algo == ALGO_NATIVE || algo == ALGO_AVX2)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("popcnt")
        {
            return match width {
                8 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx2_8_dispatch,
                    best: crate::x86_simd::array_best_avx2_8_dispatch,
                    all: crate::x86_simd::array_all_avx2_8_dispatch,
                }),
                16 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx2_16_dispatch,
                    best: crate::x86_simd::array_best_avx2_16_dispatch,
                    all: crate::x86_simd::array_all_avx2_16_dispatch,
                }),
                32 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx2_32_dispatch,
                    best: crate::x86_simd::array_best_avx2_32_dispatch,
                    all: crate::x86_simd::array_all_avx2_32_dispatch,
                }),
                64 => Some(ArrayScanner {
                    first: crate::x86_simd::array_first_avx2_64_dispatch,
                    best: crate::x86_simd::array_best_avx2_64_dispatch,
                    all: crate::x86_simd::array_all_avx2_64_dispatch,
                }),
                _ => None,
            };
        }
        None
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = width;
        None
    }
}

/// One fixed-width scan (query, threshold and the kernels for its width),
/// runnable serially or split across threads.
#[derive(Clone, Copy)]
pub(crate) struct Scan<'a> {
    query: &'a [u8],
    max_dist: i64,
    kernel: BytesKernel,
    scanner: Option<ArrayScanner>,
}

impl<'a> Scan<'a> {
    #[inline]
    pub(crate) fn new(query: &'a [u8], max_dist: i64) -> Self {
        Self::with_kernels(
            query,
            max_dist,
            select_bytes_kernel_for_width(query.len()),
            select_array_scanner_for_width(query.len()),
        )
    }

    /// Like `new`, with kernels already selected for `query.len()`.
    #[inline]
    pub(crate) fn with_kernels(
        query: &'a [u8],
        max_dist: i64,
        kernel: BytesKernel,
        scanner: Option<ArrayScanner>,
    ) -> Self {
        Self {
            query,
            max_dist,
            kernel,
            scanner,
        }
    }

    #[inline]
    fn width(&self) -> usize {
        self.query.len()
    }

    /// Scan size, in bytes, from which splitting across threads pays off.
    #[inline]
    pub(crate) fn parallel_threshold(&self) -> usize {
        parallel_threshold(self.scanner.is_some())
    }

    #[inline]
    pub(crate) fn first(&self, records: &[u8]) -> Option<usize> {
        match self.scanner {
            Some(scanner) => (scanner.first)(records, self.query, self.max_dist),
            None => serial_first_within_dist(records, self.query, self.max_dist, self.kernel),
        }
    }

    #[inline]
    pub(crate) fn best(&self, records: &[u8]) -> Option<(u64, usize)> {
        match self.scanner {
            Some(scanner) => (scanner.best)(records, self.query, self.max_dist),
            None => serial_best_within_dist(records, self.query, self.max_dist, self.kernel),
        }
    }

    #[inline]
    pub(crate) fn all(&self, records: &[u8]) -> Vec<(u64, usize)> {
        match self.scanner {
            Some(scanner) => (scanner.all)(records, self.query, self.max_dist),
            None => serial_all_within_dist(records, self.query, self.max_dist, self.kernel),
        }
    }

    /// `first`, split across threads for large catalogs. A serial prefix
    /// answers early matches without waking workers; after it, the lowest
    /// matching chunk wins and later chunks are skipped once it is found.
    pub(crate) fn first_parallel(&self, records: &[u8]) -> Option<usize> {
        self.first_split(records, self.parallel_threshold())
    }

    /// Records in the serial prefix of a split scan, and the plan for the
    /// rest; `None` when the scan should stay serial.
    #[inline]
    fn prefix_plan(&self, records: &[u8], min_bytes: usize) -> Option<(usize, par::Plan)> {
        let width = self.width();
        let count = records.len() / width;
        if records.len() < min_bytes {
            return None;
        }
        let prefix_bytes = (records.len() / SERIAL_PREFIX_SHARE)
            .clamp(SERIAL_PREFIX_MIN_BYTES, SERIAL_PREFIX_MAX_BYTES);
        let prefix = (prefix_bytes / width).next_multiple_of(16).min(count);
        par::split(count - prefix, width, min_bytes).map(|plan| (prefix, plan))
    }

    /// `first_parallel` with an explicit minimum scan size for splitting.
    pub(crate) fn first_split(&self, records: &[u8], min_bytes: usize) -> Option<usize> {
        let width = self.width();
        let count = records.len() / width;
        let Some((prefix, plan)) = self.prefix_plan(records, min_bytes) else {
            return self.first(records);
        };
        if let Some(index) = self.first(&records[..prefix * width]) {
            return Some(index);
        }
        let rest = &records[prefix * width..];
        par::find_first_chunk(plan.chunks, |chunk| {
            let (start, end) = plan.range(chunk, count - prefix);
            self.first(&rest[start * width..end * width])
                .map(|index| prefix + start + index)
        })
    }

    /// `best`, split across threads for large catalogs. A serial prefix settles
    /// early exact matches without waking workers and otherwise tightens the
    /// limit for the rest; there, an exact match ends the search for every
    /// later chunk, since ties go to the lower index.
    pub(crate) fn best_parallel(&self, records: &[u8]) -> Option<(u64, usize)> {
        self.best_split(records, self.parallel_threshold())
    }

    /// `best_parallel` with an explicit minimum scan size for splitting.
    pub(crate) fn best_split(&self, records: &[u8], min_bytes: usize) -> Option<(u64, usize)> {
        let width = self.width();
        let count = records.len() / width;
        let Some((prefix, plan)) = self.prefix_plan(records, min_bytes) else {
            return self.best(records);
        };
        let head = self.best(&records[..prefix * width]);
        let rest_scan = match head {
            Some((0, _)) => return head,
            // Later records must be strictly closer to win.
            Some((distance, _)) => Scan {
                max_dist: distance as i64 - 1,
                ..*self
            },
            None => *self,
        };
        let rest = &records[prefix * width..];
        let exact_chunk = AtomicUsize::new(usize::MAX);
        par::map_chunks(plan.chunks, |chunk| {
            if exact_chunk.load(Ordering::Relaxed) < chunk {
                return None;
            }
            let (start, end) = plan.range(chunk, count - prefix);
            let best = rest_scan
                .best(&rest[start * width..end * width])
                .map(|(distance, index)| (distance, prefix + start + index));
            if matches!(best, Some((0, _))) {
                exact_chunk.fetch_min(chunk, Ordering::Relaxed);
            }
            best
        })
        .into_iter()
        .fold(head, merge_best)
    }

    /// `all`, split across threads for large catalogs; chunk results are
    /// concatenated in chunk order, so indices stay ascending.
    pub(crate) fn all_parallel(&self, records: &[u8]) -> Vec<(u64, usize)> {
        self.all_split(records, self.parallel_threshold())
    }

    /// `all_parallel` with an explicit minimum scan size for splitting.
    pub(crate) fn all_split(&self, records: &[u8], min_bytes: usize) -> Vec<(u64, usize)> {
        let width = self.width();
        let count = records.len() / width;
        let Some(plan) = par::plan(count, width, min_bytes) else {
            return self.all(records);
        };
        let per_chunk = par::map_chunks(plan.chunks, |chunk| {
            let (start, end) = plan.range(chunk, count);
            let mut matches = self.all(&records[start * width..end * width]);
            for (_, index) in &mut matches {
                *index += start;
            }
            matches
        });
        let mut results = Vec::with_capacity(per_chunk.iter().map(Vec::len).sum());
        for matches in per_chunk {
            results.extend(matches);
        }
        results
    }
}

#[inline]
pub(crate) fn parallel_threshold(has_block_scanner: bool) -> usize {
    if has_block_scanner {
        FIXED_WIDTH_PAR_THRESHOLD_BYTES
    } else {
        PAR_THRESHOLD_BYTES
    }
}

#[inline]
fn validate_array_query(big_array: &[u8], small_array: &[u8]) -> Result<(), &'static str> {
    if small_array.is_empty() {
        return Err("elem_to_compare size must be >0");
    }
    if big_array.len() % small_array.len() != 0 {
        return Err("array_of_elems size must be multiplier of elem_to_compare");
    }
    Ok(())
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
    validate_array_query(big_array, small_array)?;
    Ok(Scan::new(small_array, max_dist).first_parallel(big_array))
}

#[inline]
pub(crate) fn serial_first_within_dist(
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
    validate_array_query(big_array, small_array)?;
    Ok(Scan::new(small_array, max_dist).best_parallel(big_array))
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
pub(crate) fn serial_best_within_dist(
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
    validate_array_query(big_array, small_array)?;
    Ok(Scan::new(small_array, max_dist).all_parallel(big_array))
}

#[inline]
pub(crate) fn serial_all_within_dist(
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

/// Experimental: hex hamming distance using the older pack-to-bytes NEON
/// kernel (32 hex chars → 16 packed bytes, then vcntq_u8). Kept as a benchmark
/// reference; `hex_hamming_distance` uses the faster table-lookup kernel.
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
    // Parallel scan regression tests
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

    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut v = state;
                v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((v ^ (v >> 31)) >> 56) as u8
            })
            .collect()
    }

    // Force splitting on small catalogs and compare with the serial scans.
    // Exact matches are planted in several chunks (and near matches with
    // tied distances), so the exact-match early exit and every tie-break are
    // exercised across chunk boundaries.
    #[test]
    fn split_scans_match_serial_scans() {
        for &width in &[1usize, 8, 16, 20, 32, 64] {
            // Past the serial prefix, so the parallel remainder always runs.
            let past_prefix = SERIAL_PREFIX_MAX_BYTES / width + 5000;
            for &count in &[33usize, 257, 4096, 20_000, past_prefix] {
                let query = pseudo_random(width, 0xA0 + width as u64);
                let mut records = pseudo_random(width * count, 0xB0 + (width * count) as u64);
                let mut near = query.clone();
                near[0] ^= 1;
                // Near matches first (a tie across the prefix boundary), exact
                // matches later, so a prefix best must be beaten or tied.
                let prefix_end = (SERIAL_PREFIX_MIN_BYTES / width).min(count - 1);
                for (k, &at) in [count / 5, prefix_end, count / 2, count - 1]
                    .iter()
                    .enumerate()
                {
                    let record = if k < 2 { &near } else { &query };
                    records[at * width..(at + 1) * width].copy_from_slice(record);
                }
                for &max_dist in &[-1i64, 0, 1, 3, width as i64 * 4, width as i64 * 8] {
                    let scan = Scan::new(&query, max_dist);
                    let label = format!("width={width} count={count} max_dist={max_dist}");
                    assert_eq!(
                        scan.first_split(&records, 0),
                        scan.first(&records),
                        "first {label}"
                    );
                    assert_eq!(
                        scan.best_split(&records, 0),
                        scan.best(&records),
                        "best {label}"
                    );
                    assert_eq!(
                        scan.all_split(&records, 0),
                        scan.all(&records),
                        "all {label}"
                    );
                }
            }
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
