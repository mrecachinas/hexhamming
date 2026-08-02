//! Batch APIs that amortize the Python↔Rust boundary across many distance
//! calculations.
//!
//! Callers pay a single validation + dispatch resolution once per batch, then
//! run a tight loop that reuses the already-resolved kernel or fixed-width
//! scanner. These APIs preserve the semantics of repeated single-call use:
//!
//! * Ordering of results matches the caller's element order.
//! * Ties in `best_*` break on lowest index, exact matches short-circuit.
//! * Threshold semantics for `max_dist` are unchanged (see `api.rs`).
//! * `set_algo` still controls the backend at call time; resolution happens
//!   once per batch call.

use crate::api::{
    select_array_scanner_for_width, serial_all_within_dist, serial_best_within_dist,
    serial_first_within_dist, should_parallel_array_scan, ArrayScanner,
};
use crate::{select_bytes_kernel_for_width, BytesKernel};

/// Compute Hamming distances between corresponding fixed-width records in two
/// contiguous buffers. Returns one `u64` per record in original order.
///
/// # Errors
/// * `element_size` is zero.
/// * `a.len() != b.len()`.
/// * Buffer length is not a multiple of `element_size`.
pub fn bytes_pairwise_distances(
    a: &[u8],
    b: &[u8],
    element_size: usize,
) -> Result<Vec<u64>, &'static str> {
    if element_size == 0 {
        return Err("`element_size` must be >0");
    }
    if a.len() != b.len() {
        return Err("bytes are NOT the same length");
    }
    if a.len() % element_size != 0 {
        return Err("length must be a multiple of `element_size`");
    }
    let count = a.len() / element_size;
    if count == 0 {
        return Ok(Vec::new());
    }
    let kernel = select_bytes_kernel_for_width(element_size);
    Ok(a.chunks_exact(element_size)
        .zip(b.chunks_exact(element_size))
        .map(|(a_chunk, b_chunk)| kernel(a_chunk, b_chunk, -1))
        .collect())
}

/// Compute Hamming distances and write them as little-endian `u64` values into
/// `out`. `out` must be exactly `count * 8` bytes, where `count = a.len() /
/// element_size`.
///
/// Returns the number of distances written.
///
/// # Errors
/// * Same input validation as [`bytes_pairwise_distances`].
/// * `out.len() != count * 8`.
pub fn bytes_pairwise_distances_into(
    a: &[u8],
    b: &[u8],
    element_size: usize,
    out: &mut [u8],
) -> Result<usize, &'static str> {
    if element_size == 0 {
        return Err("`element_size` must be >0");
    }
    if a.len() != b.len() {
        return Err("bytes are NOT the same length");
    }
    if a.len() % element_size != 0 {
        return Err("length must be a multiple of `element_size`");
    }
    let count = a.len() / element_size;
    let expected = count.checked_mul(8).ok_or("output capacity overflows")?;
    if out.len() != expected {
        return Err("`out` must be exactly count*8 bytes");
    }
    if count == 0 {
        return Ok(0);
    }
    let kernel = select_bytes_kernel_for_width(element_size);
    // Write via unaligned raw stores because caller-supplied memoryviews may
    // not be aligned to 8 bytes.
    let dst_ptr = out.as_mut_ptr();
    pairwise_distances_loop(a, b, element_size, kernel, |i, d| unsafe {
        std::ptr::write_unaligned(dst_ptr.add(i * 8) as *mut u64, d.to_le());
    });
    Ok(count)
}

#[inline(always)]
fn pairwise_distances_loop<F: FnMut(usize, u64)>(
    a: &[u8],
    b: &[u8],
    element_size: usize,
    kernel: BytesKernel,
    mut on_distance: F,
) {
    for (i, (a_chunk, b_chunk)) in a
        .chunks_exact(element_size)
        .zip(b.chunks_exact(element_size))
        .enumerate()
    {
        on_distance(i, kernel(a_chunk, b_chunk, -1));
    }
}

/// Validate common catalog + queries inputs and return `(query_count,
/// element_size, kernel, scanner)` for the multi-query APIs.
#[inline]
fn resolve_multi_scan<'a>(
    catalog: &'a [u8],
    queries: &'a [u8],
    query_width: usize,
) -> Result<(usize, BytesKernel, Option<ArrayScanner>), &'static str> {
    if query_width == 0 {
        return Err("`query_width` must be >0");
    }
    if catalog.len() % query_width != 0 {
        return Err("catalog length must be a multiple of `query_width`");
    }
    if queries.len() % query_width != 0 {
        return Err("queries length must be a multiple of `query_width`");
    }
    let kernel = select_bytes_kernel_for_width(query_width);
    let scanner = select_array_scanner_for_width(query_width);
    let query_count = queries.len() / query_width;
    Ok((query_count, kernel, scanner))
}

/// Multi-query variant of [`bytes_array_first_within_dist`]:
/// runs the same scan for every fixed-width slice of `queries` against the
/// same `catalog`, returning one `Option<usize>` per query in query order.
pub fn bytes_array_first_many_within_dist(
    catalog: &[u8],
    queries: &[u8],
    query_width: usize,
    max_dist: i64,
) -> Result<Vec<Option<usize>>, &'static str> {
    let (query_count, kernel, scanner) = resolve_multi_scan(catalog, queries, query_width)?;
    let mut out = Vec::with_capacity(query_count);
    // `first` is intentionally serial: the early-exit dominates parallel setup.
    for q in 0..query_count {
        let query = &queries[q * query_width..(q + 1) * query_width];
        let result = match scanner {
            Some(sc) => (sc.first)(catalog, query, max_dist),
            None => serial_first_within_dist(catalog, query, max_dist, kernel),
        };
        out.push(result);
    }
    Ok(out)
}

/// Multi-query variant of [`bytes_array_best_within_dist`].
pub fn bytes_array_best_many_within_dist(
    catalog: &[u8],
    queries: &[u8],
    query_width: usize,
    max_dist: i64,
) -> Result<Vec<Option<(u64, usize)>>, &'static str> {
    let (query_count, kernel, scanner) = resolve_multi_scan(catalog, queries, query_width)?;
    let mut out = Vec::with_capacity(query_count);
    for q in 0..query_count {
        let query = &queries[q * query_width..(q + 1) * query_width];
        let result = if should_parallel_array_scan(catalog.len(), query_width) {
            crate::bytes_array_best_within_dist(catalog, query, max_dist)?
        } else {
            match scanner {
                Some(sc) => (sc.best)(catalog, query, max_dist),
                None => serial_best_within_dist(catalog, query, max_dist, kernel),
            }
        };
        out.push(result);
    }
    Ok(out)
}

/// Multi-query variant of [`bytes_array_all_within_dist`].
pub fn bytes_array_all_many_within_dist(
    catalog: &[u8],
    queries: &[u8],
    query_width: usize,
    max_dist: i64,
) -> Result<Vec<Vec<(u64, usize)>>, &'static str> {
    let (query_count, kernel, scanner) = resolve_multi_scan(catalog, queries, query_width)?;
    let mut out = Vec::with_capacity(query_count);
    for q in 0..query_count {
        let query = &queries[q * query_width..(q + 1) * query_width];
        let result = if should_parallel_array_scan(catalog.len(), query_width) {
            crate::bytes_array_all_within_dist(catalog, query, max_dist)?
        } else {
            match scanner {
                Some(sc) => (sc.all)(catalog, query, max_dist),
                None => serial_all_within_dist(catalog, query, max_dist, kernel),
            }
        };
        out.push(result);
    }
    Ok(out)
}

/// Dense-transport variant of [`bytes_array_all_within_dist`]: returns matched
/// distances as `Vec<u16>` and matched indices as `Vec<u32>` in ascending
/// index order.
///
/// # Errors
/// * `small_array` is empty.
/// * `big_array.len() % small_array.len() != 0`.
/// * Element width would allow distances exceeding `u16::MAX` bits.
/// * Catalog would produce indices exceeding `u32::MAX`.
pub fn bytes_array_all_within_dist_packed(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Result<(Vec<u16>, Vec<u32>), &'static str> {
    let width = small_array.len();
    if width == 0 {
        return Err("`elem_to_compare` size must be >0");
    }
    if big_array.len() % width != 0 {
        return Err("`array_of_elems` size must be multiplier of `elem_to_compare`");
    }
    let max_bits = (width as u64).saturating_mul(8);
    if max_bits > u16::MAX as u64 {
        return Err("element width too large for u16 packed distances");
    }
    let num_records = big_array.len() / width;
    if num_records > u32::MAX as usize {
        return Err("catalog record count exceeds u32::MAX");
    }
    let matches = crate::bytes_array_all_within_dist(big_array, small_array, max_dist)?;
    let mut distances = Vec::with_capacity(matches.len());
    let mut indices = Vec::with_capacity(matches.len());
    for (d, i) in matches {
        distances.push(d as u16);
        indices.push(i as u32);
    }
    Ok((distances, indices))
}

/// Write `all_within_dist` results into caller-provided u16 distance and u32
/// index buffers. Returns the number of matches written.
///
/// The buffers must be able to hold the worst case (all records match); the
/// caller is responsible for sizing them appropriately.
///
/// # Errors
/// * Same as [`bytes_array_all_within_dist_packed`].
/// * `out_distances_u16.len() < num_records`.
/// * `out_indices_u32.len() < num_records`.
pub fn bytes_array_all_within_dist_into(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    out_distances_u16: &mut [u16],
    out_indices_u32: &mut [u32],
) -> Result<usize, &'static str> {
    let width = small_array.len();
    if width == 0 {
        return Err("`elem_to_compare` size must be >0");
    }
    if big_array.len() % width != 0 {
        return Err("`array_of_elems` size must be multiplier of `elem_to_compare`");
    }
    let max_bits = (width as u64).saturating_mul(8);
    if max_bits > u16::MAX as u64 {
        return Err("element width too large for u16 packed distances");
    }
    let num_records = big_array.len() / width;
    if num_records > u32::MAX as usize {
        return Err("catalog record count exceeds u32::MAX");
    }
    if out_distances_u16.len() < num_records || out_indices_u32.len() < num_records {
        return Err("output buffers must have capacity for at least num_records entries");
    }
    let matches = crate::bytes_array_all_within_dist(big_array, small_array, max_dist)?;
    for (i, (d, idx)) in matches.iter().enumerate() {
        out_distances_u16[i] = *d as u16;
        out_indices_u32[i] = *idx as u32;
    }
    Ok(matches.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_records(width: usize, count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(width * count);
        for _ in 0..(width * count) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((state >> 33) as u8);
        }
        out
    }

    fn oracle_distance(a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as u64)
            .sum()
    }

    #[test]
    fn pairwise_matches_oracle_various_widths() {
        for &width in &[1usize, 16, 24, 32, 33] {
            let count = 41;
            let a = make_records(width, count, 0xDEAD_BEEF);
            let b = make_records(width, count, 0xCAFE_F00D);
            let expected: Vec<u64> = (0..count)
                .map(|i| {
                    oracle_distance(
                        &a[i * width..(i + 1) * width],
                        &b[i * width..(i + 1) * width],
                    )
                })
                .collect();
            let got = bytes_pairwise_distances(&a, &b, width).unwrap();
            assert_eq!(got, expected, "mismatch at width {width}");
        }
    }

    #[test]
    fn pairwise_empty_batch() {
        let got = bytes_pairwise_distances(&[], &[], 16).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn pairwise_error_shapes() {
        assert!(bytes_pairwise_distances(b"aa", b"bb", 0).is_err());
        assert!(bytes_pairwise_distances(b"aaa", b"bbb", 2).is_err());
        assert!(bytes_pairwise_distances(b"aa", b"bbb", 1).is_err());
    }

    #[test]
    fn pairwise_into_writes_le_and_returns_count() {
        let width = 16;
        let count = 5;
        let a = make_records(width, count, 1);
        let b = make_records(width, count, 2);
        let mut out = vec![0u8; count * 8];
        let n = bytes_pairwise_distances_into(&a, &b, width, &mut out).unwrap();
        assert_eq!(n, count);
        let list = bytes_pairwise_distances(&a, &b, width).unwrap();
        for (i, d) in list.iter().enumerate() {
            let bytes: [u8; 8] = out[i * 8..(i + 1) * 8].try_into().unwrap();
            assert_eq!(u64::from_le_bytes(bytes), *d);
        }
    }

    #[test]
    fn pairwise_into_rejects_wrong_size() {
        let width = 16;
        let count = 5;
        let a = make_records(width, count, 1);
        let b = make_records(width, count, 2);
        let mut ok = vec![0u8; count * 8];
        assert!(bytes_pairwise_distances_into(&a, &b, width, &mut ok).is_ok());
        let mut short = vec![0u8; count * 8 - 1];
        assert!(bytes_pairwise_distances_into(&a, &b, width, &mut short).is_err());
        let mut long = vec![0u8; count * 8 + 1];
        assert!(bytes_pairwise_distances_into(&a, &b, width, &mut long).is_err());
    }

    #[test]
    fn multi_query_first_matches_repeated_calls() {
        let width = 16;
        let catalog = make_records(width, 100, 11);
        let queries = make_records(width, 7, 12);
        let batch = bytes_array_first_many_within_dist(&catalog, &queries, width, 8).unwrap();
        assert_eq!(batch.len(), 7);
        for (q_index, want) in batch.iter().enumerate() {
            let query = &queries[q_index * width..(q_index + 1) * width];
            let got = crate::bytes_array_first_within_dist(&catalog, query, 8).unwrap();
            assert_eq!(*want, got, "query {q_index}");
        }
    }

    #[test]
    fn multi_query_best_matches_repeated_calls() {
        let width = 16;
        let catalog = make_records(width, 200, 21);
        let queries = make_records(width, 5, 22);
        let batch = bytes_array_best_many_within_dist(&catalog, &queries, width, 64).unwrap();
        for (q_index, want) in batch.iter().enumerate() {
            let query = &queries[q_index * width..(q_index + 1) * width];
            let got = crate::bytes_array_best_within_dist(&catalog, query, 64).unwrap();
            assert_eq!(*want, got, "query {q_index}");
        }
    }

    #[test]
    fn multi_query_all_matches_repeated_calls() {
        let width = 16;
        let catalog = make_records(width, 40, 31);
        let queries = make_records(width, 3, 32);
        let batch = bytes_array_all_many_within_dist(&catalog, &queries, width, 62).unwrap();
        for (q_index, want) in batch.iter().enumerate() {
            let query = &queries[q_index * width..(q_index + 1) * width];
            let got = crate::bytes_array_all_within_dist(&catalog, query, 62).unwrap();
            assert_eq!(*want, got, "query {q_index}");
        }
    }

    #[test]
    fn multi_query_empty_queries() {
        let catalog = make_records(16, 5, 1);
        let queries: Vec<u8> = Vec::new();
        assert!(
            bytes_array_first_many_within_dist(&catalog, &queries, 16, 4)
                .unwrap()
                .is_empty()
        );
        assert!(bytes_array_best_many_within_dist(&catalog, &queries, 16, 4)
            .unwrap()
            .is_empty());
        assert!(bytes_array_all_many_within_dist(&catalog, &queries, 16, 4)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn packed_all_matches_list_and_dense_case() {
        let width = 16;
        let catalog = make_records(width, 128, 41);
        let query = &catalog[3 * width..4 * width]; // dense: at least one exact match
        let list = crate::bytes_array_all_within_dist(&catalog, query, 128).unwrap();
        let (dists, idxs) = bytes_array_all_within_dist_packed(&catalog, query, 128).unwrap();
        assert_eq!(dists.len(), list.len());
        assert_eq!(idxs.len(), list.len());
        for (i, (d, idx)) in list.iter().enumerate() {
            assert_eq!(dists[i] as u64, *d);
            assert_eq!(idxs[i] as usize, *idx);
        }
        // include an exact match at least once
        assert!(list.iter().any(|(d, _)| *d == 0));
    }

    #[test]
    fn packed_all_sparse_case() {
        let width = 16;
        let catalog = make_records(width, 200, 51);
        let query = &catalog[10 * width..11 * width];
        // Very tight max_dist: only exact matches.
        let (dists, idxs) = bytes_array_all_within_dist_packed(&catalog, query, 0).unwrap();
        assert!(!dists.is_empty());
        for (i, &d) in dists.iter().enumerate() {
            assert_eq!(d, 0);
            let idx = idxs[i] as usize;
            assert_eq!(&catalog[idx * width..(idx + 1) * width], query);
        }
    }

    #[test]
    fn packed_into_matches_packed() {
        let width = 16;
        let catalog = make_records(width, 128, 61);
        let query = &catalog[5 * width..6 * width];
        let num_records = catalog.len() / width;
        let (dists, idxs) = bytes_array_all_within_dist_packed(&catalog, query, 128).unwrap();
        let mut out_d = vec![0u16; num_records];
        let mut out_i = vec![0u32; num_records];
        let n =
            bytes_array_all_within_dist_into(&catalog, query, 128, &mut out_d, &mut out_i).unwrap();
        assert_eq!(n, dists.len());
        assert_eq!(&out_d[..n], &dists[..]);
        assert_eq!(&out_i[..n], &idxs[..]);
    }

    #[test]
    fn packed_into_rejects_short_buffers() {
        let width = 16;
        let catalog = make_records(width, 8, 71);
        let query = &catalog[0..width];
        let mut short_d = vec![0u16; 4];
        let mut ok_i = vec![0u32; 8];
        assert!(
            bytes_array_all_within_dist_into(&catalog, query, 128, &mut short_d, &mut ok_i,)
                .is_err()
        );
        let mut ok_d = vec![0u16; 8];
        let mut short_i = vec![0u32; 4];
        assert!(
            bytes_array_all_within_dist_into(&catalog, query, 128, &mut ok_d, &mut short_i,)
                .is_err()
        );
    }

    #[test]
    fn algorithm_invariance_pairwise() {
        let width = 16;
        let a = make_records(width, 50, 81);
        let b = make_records(width, 50, 82);
        let baseline = bytes_pairwise_distances(&a, &b, width).unwrap();
        for algo in ["classic", "native"] {
            crate::api::set_algorithm(algo).unwrap();
            let got = bytes_pairwise_distances(&a, &b, width).unwrap();
            assert_eq!(got, baseline, "mismatch under algo {algo}");
        }
        crate::api::set_algorithm("native").unwrap();
    }

    #[test]
    fn best_many_tiebreak_lowest_index() {
        let width = 16;
        let mut catalog = vec![0xFFu8; width * 20];
        // Two exact-match entries at indices 4 and 9.
        for &idx in &[4usize, 9] {
            for byte in &mut catalog[idx * width..(idx + 1) * width] {
                *byte = 0;
            }
        }
        let query = vec![0u8; width];
        let batch = bytes_array_best_many_within_dist(&catalog, &query, width, 128).unwrap();
        assert_eq!(batch, vec![Some((0u64, 4usize))]);
    }

    #[test]
    fn all_many_ordering_preserved() {
        let width = 16;
        let mut catalog = vec![0xFFu8; width * 12];
        for &idx in &[1usize, 5, 8, 11] {
            for byte in &mut catalog[idx * width..(idx + 1) * width] {
                *byte = 0;
            }
        }
        let query = vec![0u8; width];
        // 0xFF vs 0 distance = 128 bits over 16 bytes; use max_dist 4 so only
        // the exact matches survive.
        let batch = bytes_array_all_many_within_dist(&catalog, &query, width, 4).unwrap();
        assert_eq!(batch.len(), 1);
        let matches: Vec<usize> = batch[0].iter().map(|&(_, i)| i).collect();
        assert_eq!(matches, vec![1, 5, 8, 11]);
    }
}
