//! Parallel execution for large scans.
//!
//! A scan is split into chunks of whole records, and workers claim chunks
//! dynamically, so faster cores take more of them. Callers only see
//! [`map_chunks`] and [`find_first_chunk`], which return results in chunk
//! order; the executor behind them is an implementation detail.

use rayon::prelude::*;

/// Target bytes scanned per chunk: large enough that claiming a chunk is cheap
/// next to scanning it, small enough to balance load across cores.
const CHUNK_BYTES: usize = 256 * 1024;
/// Chunks per worker thread, so a slow core cannot hold up the whole scan.
const CHUNKS_PER_THREAD: usize = 4;
/// Record counts per chunk are rounded to this so block scanners (16 records
/// per block) see no partial blocks except at the very end.
const RECORD_ALIGN: usize = 16;

#[inline]
pub(crate) fn threads() -> usize {
    rayon::current_num_threads()
}

/// How to split `records` records across chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) records_per_chunk: usize,
    pub(crate) chunks: usize,
}

impl Plan {
    /// Record range of `chunk`, clamped to `records`.
    #[inline]
    pub(crate) fn range(self, chunk: usize, records: usize) -> (usize, usize) {
        let start = chunk * self.records_per_chunk;
        (start, (start + self.records_per_chunk).min(records))
    }
}

/// Chunk plan for scanning `records` records of `width` bytes, or `None` when
/// the scan (under `min_bytes`) is too small to be worth splitting.
pub(crate) fn plan(records: usize, width: usize, min_bytes: usize) -> Option<Plan> {
    let bytes = records.saturating_mul(width);
    let threads = threads();
    if bytes < min_bytes || threads < 2 || records < 2 * RECORD_ALIGN {
        return None;
    }
    let chunks = (bytes / CHUNK_BYTES).clamp(2, threads * CHUNKS_PER_THREAD);
    let records_per_chunk = records.div_ceil(chunks).next_multiple_of(RECORD_ALIGN);
    Some(Plan {
        records_per_chunk,
        chunks: records.div_ceil(records_per_chunk),
    })
}

/// Chunk plan for running `items` independent jobs of `bytes_per_item` bytes
/// each, or `None` when the total is under `min_bytes`.
pub(crate) fn plan_items(items: usize, bytes_per_item: usize, min_bytes: usize) -> Option<Plan> {
    let threads = threads();
    if items < 2 || threads < 2 || items.saturating_mul(bytes_per_item) < min_bytes {
        return None;
    }
    let chunks = items.min(threads * CHUNKS_PER_THREAD);
    let per_chunk = items.div_ceil(chunks);
    Some(Plan {
        records_per_chunk: per_chunk,
        chunks: items.div_ceil(per_chunk),
    })
}

/// `f(chunk)` for every chunk, in chunk order.
pub(crate) fn map_chunks<R, F>(chunks: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    (0..chunks)
        .into_par_iter()
        .with_max_len(1)
        .map(&f)
        .collect()
}

/// The result of the lowest-numbered chunk for which `f` returns `Some`.
/// Chunks after one that has returned `Some` may be skipped.
pub(crate) fn find_first_chunk<R, F>(chunks: usize, f: F) -> Option<R>
where
    R: Send,
    F: Fn(usize) -> Option<R> + Sync,
{
    (0..chunks)
        .into_par_iter()
        .with_max_len(1)
        .find_map_first(&f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_cover_every_record_exactly_once() {
        for records in [32usize, 33, 1000, 4096, 100_003, 1 << 20] {
            for width in [1usize, 8, 16, 20, 64] {
                let Some(plan) = plan(records, width, 0) else {
                    continue;
                };
                let mut next = 0;
                for chunk in 0..plan.chunks {
                    let (start, end) = plan.range(chunk, records);
                    assert_eq!(start, next);
                    assert!(end > start, "empty chunk {chunk} of {plan:?}");
                    next = end;
                }
                assert_eq!(next, records);
                assert_eq!(plan.records_per_chunk % RECORD_ALIGN, 0);
            }
        }
        assert_eq!(plan(1 << 20, 16, usize::MAX), None);
    }

    #[test]
    fn item_plans_cover_every_item_exactly_once() {
        for items in [2usize, 3, 17, 64, 1000] {
            let plan = plan_items(items, 1 << 20, 0).unwrap();
            let covered: usize = (0..plan.chunks)
                .map(|c| {
                    let (s, e) = plan.range(c, items);
                    e - s
                })
                .sum();
            assert_eq!(covered, items);
        }
        assert_eq!(plan_items(1, 1 << 30, 0), None);
    }

    #[test]
    fn chunk_helpers_keep_order() {
        assert_eq!(
            map_chunks(100, |c| c * 2),
            (0..100).map(|c| c * 2).collect::<Vec<_>>()
        );
        assert_eq!(
            find_first_chunk(100, |c| (c % 7 == 3).then_some(c)),
            Some(3)
        );
        assert_eq!(find_first_chunk(100, |_| None::<usize>), None);
    }
}
